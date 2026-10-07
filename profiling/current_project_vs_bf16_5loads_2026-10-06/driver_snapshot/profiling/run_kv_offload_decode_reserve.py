"""Compare offload admission with and without next-decode block reservation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_current_int8_metrics import PYTHON, aggregate, source_hashes
from profiling.run_kv_metrics_comparison import parse_metrics

CHANGED = {"nanovllm/engine/scheduler.py", "nanovllm/engine/block_manager.py",
           "nanovllm/engine/kv_offload.py"}


def instrument_progress(code):
    marker = "    original_preempt = llm.scheduler.preempt\n"
    assert code.count(marker) == 1
    code = code.replace(marker, """    restore_progress = dict(restore_events=0,
                            preemptions_without_token_progress=0,
                            same_schedule_preemptions=0)
    last_restored = {}
    schedule_round = 0
    original_schedule = llm.scheduler.schedule
    def watched_schedule():
        nonlocal schedule_round
        schedule_round += 1
        return original_schedule()
    llm.scheduler.schedule = watched_schedule
    if llm.scheduler.offload is not None:
        original_poll = llm.scheduler.offload.poll
        def watched_poll():
            ready = original_poll()
            for seq in ready:
                restore_progress['restore_events'] += 1
                last_restored[seq.seq_id] = (seq.num_completion_tokens, schedule_round)
            return ready
        llm.scheduler.offload.poll = watched_poll
""" + marker)
    marker = "        preemptions += 1\n"
    assert code.count(marker) == 1
    code = code.replace(marker, marker + """        restored = last_restored.pop(seq.seq_id, None)
        if restored is not None and seq.num_completion_tokens == restored[0]:
            restore_progress['preemptions_without_token_progress'] += 1
            if restored[1] == schedule_round:
                restore_progress['same_schedule_preemptions'] += 1
""")
    marker = "    print('Preemptions: ' + str(preemptions))\n"
    assert code.count(marker) == 1
    code = code.replace(marker, marker +
                        "    print('Restore progress statistics: ' + json.dumps(restore_progress))\n")
    compile(code, "benchmark_inference_metrics.py", "exec")
    return code


def prepare(outdir, args):
    original = outdir / "variants/original"
    if not original.exists():
        shutil.copytree(ROOT / "profiling/async_kv_offload_2026-10-06/metrics/variant", original,
                        ignore=shutil.ignore_patterns("__pycache__"))
    reserved = outdir / "variants/reserved"
    assert not reserved.exists(), "use a fresh output directory"
    shutil.copytree(original, reserved, ignore=shutil.ignore_patterns("__pycache__"))
    for name in CHANGED:
        shutil.copyfile(ROOT / name, reserved / name)
    code = instrument_progress((original / "benchmark_inference_metrics.py").read_text())
    for variant in (original, reserved):
        (variant / "benchmark_inference_metrics.py").write_text(code)
    hashes = {name: source_hashes(outdir / "variants" / name) for name in ("original", "reserved")}
    assert {k for k in hashes["original"] if hashes["original"][k] != hashes["reserved"][k]} == CHANGED
    workload = outdir / "workload_1024.json"
    if not workload.exists():
        shutil.copyfile(ROOT / "profiling/async_kv_offload_2026-10-06/metrics/workload_1024.json", workload)
    return dict(started_utc=datetime.now(timezone.utc).isoformat(), runs=args.runs,
                policies=["original", "reserved"] + (["disabled"] if args.include_disabled else []),
                kv_blocks=args.kv_blocks, gpu=args.gpu, source_hashes=hashes,
                root_hashes=source_hashes(ROOT),
                workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
                changed_files=sorted(CHANGED), commands=[])


def json_line(text, prefix):
    match = re.search(r"^" + re.escape(prefix) + r": (.*)$", text, re.M)
    assert match is not None, prefix
    return json.loads(match[1])


def medians(trials):
    result = {}
    for policy in dict.fromkeys(row["policy"] for row in trials):
        rows = [r for r in trials if r["policy"] == policy]
        value = aggregate(rows)["int8_half"]
        for key in ("request_preemption_stats", "decode_batch_stats", "restore_progress"):
            value[key] = {k: statistics.median(r[key][k] for r in rows)
                          for k, v in rows[0][key].items() if isinstance(v, (int, float))}
        result[policy] = value
    return result


def render(outdir, results):
    labels = {"disabled": "关闭卸载", "original": "原卸载", "reserved": "预留 decode 块"}
    policies = [p for p in ("disabled", "original", "reserved") if p in results["medians"]]
    lines = ["# KV 卸载：预留下一批 decode 扩块空间", "",
             "相同 Qwen3-0.6B、GPU0 RTX 3090 Ti、快速双 scale INT8 kernel、1318 个 KV 块。",
             "相同 1024 请求（输入580663、输出583802 token），独立进程交替测量，各项取三轮中位数。",
             "预留量为 running 队首 max_num_seqs 个请求当前缺失的逻辑块数之和；",
             "恢复和 waiting prefill 都需在分配后保留这部分空闲块。关闭卸载的调度行为不变。", "",
             "| 指标 | " + " | ".join(labels[p] for p in policies) + " |",
             "| --- | " + " | ".join("---:" for _ in policies) + " |"]
    def row(label, getter, digits=2):
        lines.append("| " + label + " | " + " | ".join(
            f"{getter(results['medians'][p]):.{digits}f}" for p in policies) + " |")
    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for stat in ("mean", "p50", "p95", "p99"):
            row(f"{metric} {stat} ms", lambda v, m=metric, s=stat: v["latency_ms"][m][s])
    for key in ("output_tokens_per_s", "requests_per_s", "elapsed_s", "preemptions",
                "peak_allocated_gib", "peak_reserved_gib"):
        row(key, lambda v, k=key: v[k], 0 if key == "preemptions" else 3)
    for key in ("preempted_requests", "repeatedly_preempted_requests", "maximum_per_request"):
        row(key, lambda v, k=key: v["request_preemption_stats"][k], 0)
    for key in ("restore_events", "preemptions_without_token_progress", "same_schedule_preemptions"):
        row(key, lambda v, k=key: v["restore_progress"][k], 0)
    for phase in ("prefill", "decode"):
        for key in ("tokens", "seconds", "tokens_per_s"):
            row(f"{phase} {key}", lambda v, p=phase, k=key: v["stages"][p][k], 0 if key == "tokens" else 3)
    lines += ["", "每轮结果：", "",
              "| 策略 | 轮次 | 输出 token/s | 抢占 | 恢复后无进度再抢占 | 同一次调度内 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in results["trials"]:
        lines.append(f"| {labels[r['policy']]} | {r['trial']} | {r['output_tokens_per_s']:.2f} | "
                     f"{r['preemptions']} | {r['restore_progress']['preemptions_without_token_progress']} | "
                     f"{r['restore_progress']['same_schedule_preemptions']} |")
    lines += ["", "测量边界：", "",
              "- max_num_seqs512、max_num_batched_tokens16384、block256、max_model_len4096。",
              "- CPU pinned 池4GiB、最多8个同时恢复请求；两种卸载策略的 staging 和 GPU KV 容量相同。",
              "- attention 使用既有 fast8 测量副本，根目录 attention 未修改；两组差异仅在三个调度/准入文件。",
              "- 初始化、编译、代表性普通/缓存 prefill 预热在计时外；CPU池首次分配在计时内。",
              "- TTFT包含排队；TPOT是每请求首末 token 间隔除以后续 token 数。",
              "- 恢复后无进度再抢占表示恢复完成后尚未生成任何新token即被抢占；同一次调度内是其子集。",
              "- 所有策略使用相同 CPU 计数探针；未锁GPU频率，异步事件完成时机可改变调度。",
              "- 检查每请求输出长度、时间戳数量、完整负载hash、抢占直方图总和、传输结束和CPU池预算。",
              "- 随机采样随 batch/调度变化；本次性能对照不构成PPL或生成质量评估。", ""]
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--kv-blocks", type=int, default=1318)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else prepare(outdir, args)
    assert not manifest["commands"], "completed or partial trials already exist; use a fresh output directory"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print("PREPARED", outdir, flush=True)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["gpu"], PYTHONUNBUFFERED="1")
    results = dict(trials=[], medians={})
    for trial in range(1, manifest["runs"] + 1):
        policies = manifest["policies"] if trial % 2 else list(reversed(manifest["policies"]))
        for policy in policies:
            variant_name = "original" if policy == "disabled" else policy
            variant = outdir / "variants" / variant_name
            assert source_hashes(variant) == manifest["source_hashes"][variant_name]
            assert source_hashes(ROOT) == manifest["root_hashes"]
            env["PYTHONPATH"] = str(variant)
            command = [PYTHON, "benchmark_inference_metrics.py", "--requests", "1024",
                       "--kv-cache-dtype", "int8_half", "--kv-blocks", str(manifest["kv_blocks"])]
            if policy != "disabled":
                command.append("--kv-cpu-offload")
            log = outdir / f"{policy}_{trial}.txt"
            print("START", trial, policy, flush=True)
            started = perf_counter()
            with log.open("w") as fp:
                run = subprocess.run(command, cwd=variant, env=env, stdout=fp, stderr=subprocess.STDOUT, timeout=600)
            assert run.returncode == 0, log
            text = log.read_text()
            row = parse_metrics(text)
            assert row["blocks"] == manifest["kv_blocks"] and row["requests"] == 1024
            assert row["input_tokens"] == 580663 and row["output_tokens"] == 583802
            assert f"Package: {variant}/nanovllm/__init__.py" in text
            assert f"Generated workload SHA256: {manifest['workload_sha256']}" in text
            row.update(policy=policy, trial=trial, log=log.name, process_wall_s=perf_counter()-started,
                       preemptions=int(re.search(r"Preemptions: (\d+)", text)[1]),
                       request_preemption_stats=json_line(text, "Per-request preemptions"),
                       decode_batch_stats=json_line(text, "Actual decode batch sizes"),
                       prefill_batches=json_line(text, "Prefill batches"),
                       restore_progress=json_line(text, "Restore progress statistics"),
                       kv_tensor_bytes=int(re.search(r"KV tensor bytes: (\d+)", text)[1]),
                       scale_tensor_bytes=int(re.search(r"Scale tensor bytes: (\d+)", text)[1]))
            assert sum(int(k)*v for k,v in row["request_preemption_stats"]["histogram"].items()) == row["preemptions"]
            assert sum(row["request_preemption_stats"]["histogram"].values()) == 1024
            dispatch = json_line(text, "Prefill attention dispatch")
            assert dispatch["flash"] == 0 and dispatch["int8"] > 0
            if policy != "disabled":
                stats = json_line(text, "KV offload statistics")
                assert stats["active_handles"] == stats["restoring"] == 0
                assert stats["cpu_pool_bytes"] <= stats["cpu_budget_bytes"]
                assert stats["offloaded_sequences"] + stats["fallback_preemptions"] == row["preemptions"]
                assert stats["offloaded_sequences"] == stats["restored_sequences"] == row["restore_progress"]["restore_events"]
                row["offload_stats"] = stats
            results["trials"].append(row)
            results["medians"] = medians(results["trials"])
            manifest["commands"].append(dict(policy=policy, trial=trial, command=command, log=log.name))
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            render(outdir, results)
            print("DONE", trial, policy, "tok/s", row["output_tokens_per_s"], "preemptions", row["preemptions"],
                  "no-progress", row["restore_progress"]["preemptions_without_token_progress"], flush=True)
    assert source_hashes(ROOT) == manifest["root_hashes"]
    for name, hashes in manifest["source_hashes"].items():
        assert source_hashes(outdir / "variants" / name) == hashes
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), source_checks_passed=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE", outdir / "report.md", flush=True)


if __name__ == "__main__":
    main()
