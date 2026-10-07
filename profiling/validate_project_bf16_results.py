"""Audit completed benchmark artifacts and add a compact interpretation."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_current_int8_metrics import source_hashes
from profiling.run_project_bf16_5loads import render, save, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    out = args.outdir.resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    results = json.loads((out / "results.json").read_text())
    trials = results["trials"]
    expected = {(count, mode, trial) for count in manifest["request_counts"]
                for mode in ("auto", "int8_half") for trial in range(1, manifest["runs"] + 1)}
    assert {(r["requests"], r["mode"], r["trial"]) for r in trials} == expected
    assert len(trials) == len(expected)
    for mode in ("auto", "int8_half"):
        assert len({r["blocks"] for r in trials if r["mode"] == mode}) == 1, "native KV capacity drifted"
    assert results.get("completed_utc"), "benchmark is still running"
    assert source_hashes(Path(manifest["checkout"])) == manifest["measured_source_hashes"]
    assert source_hashes(ROOT) == manifest["root_source_hashes"], "root inference changed"
    workload_checks = {}
    for count, workload in manifest["workloads"].items():
        raw = (out / workload["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == workload["sha256"]
        data = json.loads(raw)
        assert len(data["prompts"]) == len(data["max_tokens"]) == int(count)
        assert sum(map(len, data["prompts"])) == workload["input_tokens"]
        assert sum(data["max_tokens"]) == workload["output_tokens"]
        workload_checks[count] = True
    compiler_events = []
    for row in trials:
        workload = manifest["workloads"][str(row["requests"])]
        assert row["workload_sha256"] == workload["sha256"]
        assert (row["input_tokens"], row["output_tokens"]) == (workload["input_tokens"], workload["output_tokens"])
        assert row["graph"] and row["seed"] == 0
        assert row["decode_batch_stats"]["maximum"] <= manifest["config"]["max_num_seqs"]
        assert Path(row["package"]).is_relative_to(Path(manifest["checkout"]))
        expected_dtype = "torch.bfloat16" if row["mode"] == "auto" else "torch.int8"
        assert row["actual_kv_dtype"] == expected_dtype
        histogram = row["request_preemption_stats"]["histogram"]
        assert sum(histogram.values()) == row["requests"]
        assert sum(int(k) * v for k, v in histogram.items()) == row["preemptions"]
        for distribution in row["latency_ms"].values():
            assert 0 < distribution["p50"] <= distribution["p95"] <= distribution["p99"]
            assert distribution["mean"] > 0
        # Text output rounds time to 0.001 s and rates to 0.01.
        assert abs(row["output_tokens"] / row["elapsed_s"] - row["output_tokens_per_s"]) <= .15
        progress = row["restore_progress"]
        assert 0 <= progress["same_schedule_preemptions"] <= progress["preemptions_without_token_progress"] <= progress["restore_events"]
        if row["mode"] == "auto":
            assert row["attention_dispatch"]["int8"] == 0
            assert row["scale_tensor_bytes"] == 0 and progress["restore_events"] == 0
        else:
            stats = row["offload_stats"]
            assert stats["active_handles"] == stats["restoring"] == 0
            assert stats["cpu_pool_bytes"] <= stats["cpu_budget_bytes"]
            assert stats["offloaded_sequences"] + stats["fallback_preemptions"] == row["preemptions"]
            assert stats["restored_sequences"] == stats["offloaded_sequences"] == progress["restore_events"]
            transport = stats["transport"]
            for direction in ("d2h", "h2d"):
                assert transport[direction + "_blocks"] == transport[direction + "_completed_events"]
                block_bytes = (row["kv_tensor_bytes"] + row["scale_tensor_bytes"]) // row["blocks"]
                assert transport[direction + "_bytes"] == transport[direction + "_blocks"] * block_bytes
        raw_log = (out / row["raw_log"]).read_text()
        assert "length check: PASS" in raw_log
        if "torch._dynamo hit config.cache_size_limit" in raw_log:
            compiler_events.append(dict(requests=row["requests"], variant=row["variant"],
                                        trial=row["trial"], log=row["raw_log"],
                                        functions=sorted(set(re.findall(r"function: '([^']+)'", raw_log))),
                                        cache_size_limit=8))
    a, b = (results["quality"][mode] for mode in ("auto", "int8_half"))
    assert a["tokens"] == b["tokens"] == 298938
    for key in ("text_sha256", "token_ids_sha256", "window_size", "chunk_size", "block_size"):
        assert a[key] == b[key]
    assert a["model_dtype"] == b["model_dtype"] == "torch.bfloat16"
    exact_checks = {}
    for mode in ("auto", "int8_half"):
        validation = json.loads((out / "validation" / f"restored_kv_{mode}.json").read_text())
        assert validation["offload"] and validation["blocks"] == 12
        assert validation["preemptions"] > 0 and validation["restored_block_exact_checks"] > 0
        assert validation["stats"]["active_handles"] == validation["stats"]["restoring"] == 0
        exact_checks[mode] = validation["restored_block_exact_checks"]
    results["medians"] = summarize(trials)
    render(out, manifest, results)
    lines = ["", "## 结果解读与传输统计", "",
             "这组结果比较的是当前完整推理系统：INT8 KV 容量、attention 路径与 CPU 卸载调度共同影响性能，不能把吞吐差直接归因于单个 kernel。",
             "显存预算相同，压缩后用于增加 KV 块数，所以整体 GPU 显存峰值不会按 KV 每 token 字节数同比下降。", "",
             "| 请求数 | 吞吐变化 | TTFT P95 变化 | TPOT P50 变化 | E2E P50 变化 | 抢占变化 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    transport_medians = {}
    for count in manifest["request_counts"]:
        a, b = (results["medians"][str(count)][mode] for mode in ("auto", "int8_half"))
        pairs = [(a["output_tokens_per_s"], b["output_tokens_per_s"]),
                 (a["latency_ms"]["TTFT"]["p95"], b["latency_ms"]["TTFT"]["p95"]),
                 (a["latency_ms"]["TPOT"]["p50"], b["latency_ms"]["TPOT"]["p50"]),
                 (a["latency_ms"]["End-to-end latency"]["p50"], b["latency_ms"]["End-to-end latency"]["p50"]),
                 (a["preemptions"], b["preemptions"])]
        changes = " | ".join(f"{(new / old - 1) * 100:+.2f}%" for old, new in pairs)
        lines.append(f"| {count} | {changes} |")
        values = [r["offload_stats"]["transport"] for r in trials if r["requests"] == count and r["mode"] == "int8_half"]
        transport_medians[str(count)] = {k: statistics.median(v[k] for v in values) for k in values[0]}
    lines += ["", "CPU 卸载传输统计（三轮中位数）：", "",
              "| 请求数 | D2H blocks | H2D blocks | D2H GiB | H2D GiB | D2H stream ms | H2D stream ms |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for count, transport in transport_medians.items():
        lines.append(f"| {count} | {transport['d2h_blocks']:.0f} | {transport['h2d_blocks']:.0f} | "
                     f"{transport['d2h_bytes']/1024**3:.3f} | {transport['h2d_bytes']/1024**3:.3f} | "
                     f"{transport['d2h_stream_ms']:.3f} | {transport['h2d_stream_ms']:.3f} |")
    lines += ["", "stream 时间为各复制事件的 CUDA event 时长之和，复制可能与模型重叠，不能直接从整批耗时相减。", ""]
    if compiler_events:
        lines += ["## Torch 编译缓存事件", "",
                  "本轮保留 PyTorch 2.5.1 的默认 Dynamo 编译缓存上限 8；未修改推理源码或提高该上限。",
                  "以下轮次在正式计时阶段出现 RMSNorm.rms_forward 达到缓存上限的提示；新形状的编译/回退耗时计入原生端到端结果。",
                  "decode CUDA Graph 在计时前已捕获，正式过程重放已保存的图；该提示的发生不能直接换算为整个模型均转为 eager。",
                  "这组结果用于描述当前系统行为；不能将 4096 档性能差全部归因于 INT8 attention 或卸载。", "",
                  "| 请求数 | 版本 | 轮次 | 原始日志 |", "| ---: | --- | ---: | --- |"]
        for event in compiler_events:
            lines.append(f"| {event['requests']} | {event['variant']} | {event['trial']} | {event['log']} |")
        lines += [""]
    report = out / "report.md"
    report.write_text(report.read_text() + "\n".join(lines))
    weight_hashes = {}
    for entry in manifest["weight_files"]:
        path = Path(manifest["model"]) / entry["name"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (entry["size"], entry["mtime_ns"])
        weight_hashes[entry["name"]] = digest.hexdigest()
    audit = dict(status="PASS", trials=len(trials), all_workloads=workload_checks,
                 snapshot_and_root_inference_unchanged=True, paired_quality_targets=298938,
                 restored_kv_exact_checks=exact_checks, weight_sha256=weight_hashes,
                 offload_transport_medians=transport_medians, compiler_cache_events=compiler_events)
    save(out / "artifact_validation.json", audit)
    print("PASS", len(trials), "fresh-process trials; frozen sources; paired workloads/PPL; restored KV")


if __name__ == "__main__":
    main()
