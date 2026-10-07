"""Paired offload benchmarks: same fast 8-warp kernel, same KV capacity/workload."""

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
from profiling.run_preemption_lock_metrics import measured_benchmark
from profiling.run_kv_metrics_comparison import parse_metrics


def prepare(outdir):
    target = outdir / "variant"
    root_hashes = source_hashes(ROOT)
    archived = ROOT / "profiling/attention_warps_4_vs_8_1024_2026-10-05/variants/warps8"
    kernel_files = {"nanovllm/layers/attention.py", "nanovllm/layers/quantized_attention.py"}
    for rel in root_hashes:
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel == "benchmark_inference_metrics.py":
            code = measured_benchmark()
            marker = "    expected_output_tokens = sum(p.max_tokens for p in params)\n"
            assert code.count(marker) == 1
            code = code.replace(marker, marker +
                "    workload_bytes = (json.dumps(dict(prompts=prompts, max_tokens=[p.max_tokens for p in params], seed=args.seed)) + '\\n').encode()\n"
                "    print('Generated workload SHA256: ' + hashlib.sha256(workload_bytes).hexdigest(), flush=True)\n")
            p.write_text(code)
        elif rel == "nanovllm/engine/model_runner.py":
            code = (ROOT / rel).read_text()
            old = "        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache\n"
            assert code.count(old) == 1
            p.write_text(code.replace(old,
                "        if (cu_seqlens_k[-1] > cu_seqlens_q[-1]\n"
                "                or (self.config.kv_cache_dtype == 'int8_half'\n"
                "                    and all(seq.block_table for seq in seqs))):\n"))
        else:
            shutil.copyfile((archived if rel in kernel_files else ROOT) / rel, p)
    return target, root_hashes


def median_results(trials):
    result = {}
    for enabled in (False, True):
        rows = [v for v in trials if v["offload"] == enabled]
        if rows:
            result[str(enabled)] = aggregate(rows)["int8_half"]
            for key in ("request_preemption_stats", "decode_batch_stats"):
                result[str(enabled)][key] = {
                    k: statistics.median(v[key][k] for v in rows)
                    for k, value in rows[0][key].items() if isinstance(value, (float, int))}
            if enabled:
                def numeric(values):
                    if isinstance(values[0], dict):
                        keys = set.intersection(*(set(v) for v in values))
                        return {k: numeric([v[k] for v in values]) for k in sorted(keys)}
                    return statistics.median(values)
                result[str(enabled)]["offload_stats"] = numeric([v["offload_stats"] for v in rows])
    return result


def render(outdir, manifest, results):
    a, b = (results["medians"][str(flag)] for flag in (False, True))
    lines = ["# 1024 请求：异步 KV 卸载与恢复", "",
             "同一快速双 scale INT8 kernel：prefill BM64、decode BM16、BN64、8 warp、num_stages=1。",
             "两组使用相同推理源码，仅 kv_cpu_offload 开关不同；保留 restore→waiting prefill→running decode 的优先级。",
             f'每组{manifest["runs"]}次独立新进程，交替运行，每项取各轮中位数。KV 容量均固定为{manifest["kv_blocks"]}块。',
             "", "| 指标 | 关闭 | 开启 | 变化 |", "| --- | ---: | ---: | ---: |"]

    def row(label, getter, digits=2):
        x, y = getter(a), getter(b)
        change = f"{(y / x - 1) * 100:+.2f}%" if x else "—"
        lines.append(f"| {label} | {x:.{digits}f} | {y:.{digits}f} | {change} |")

    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for q in ("mean", "p50", "p95", "p99"):
            row(f"{metric} {q.upper()} ms", lambda v, m=metric, k=q: v["latency_ms"][m][k])
    for key in ("elapsed_s", "output_tokens_per_s", "requests_per_s", "input_tokens_per_s", "total_tokens_per_s",
                "preemptions", "blocks", "peak_allocated_gib", "peak_reserved_gib"):
        row(key, lambda v, k=key: v[k], 0 if key in ("preemptions", "blocks") else 3)
    for phase in ("prefill", "decode"):
        for key in ("seconds", "tokens", "tokens_per_s"):
            row(f"{phase} model-run {key}", lambda v, p=phase, k=key: v["stages"][p][k], 0 if key == "tokens" else 3)
    for key in ("preempted_requests", "repeatedly_preempted_requests", "maximum_per_request"):
        row(key, lambda v, k=key: v["request_preemption_stats"][k], 0)
    for key in ("steps", "mean", "p50", "p95", "maximum"):
        row(f"actual decode batch {key}", lambda v, k=key: v["decode_batch_stats"][k])
    lines += ["", "## 每轮结果", "",
              "| 开关 | 轮次 | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for v in results["trials"]:
        lines.append(f'| {v["offload"]} | {v["trial"]} | {v["output_tokens_per_s"]:.2f} | '
                     f'{v["latency_ms"]["TTFT"]["p50"]:.2f} | {v["latency_ms"]["TPOT"]["p50"]:.2f} | {v["preemptions"]} |')
    lines += ["", "## 卸载统计（三轮中位数）", "", "```json",
              json.dumps(b["offload_stats"], indent=2, ensure_ascii=False), "```", "",
              "## 测量范围", "",
              "- RTX 3090 Ti / Qwen3-0.6B / GPU0，权重BF16，双scale INT8 KV，decode CUDA Graph。",
              "- 复用既有完整1024请求负载：输入580663、输出583802 token，长度100–1024、seed0、temperature0.6、ignore_eos=True。",
              "- max_num_seqs512，max_num_batched_tokens16384，max_model_len4096，block256。",
              "- 原仓库 attention 未改；实际测量副本使用先前已校验的快速8warp attention，两组相同。",
              "- pinned CPU 缓冲池上限4GiB，GPU staging 两块共29.75MiB；相同KV容量对比避免容量变化混入策略收益。",
              "- CPU缓冲不足时回退原重新prefill路径，fallback_preemptions单独记录；共享前缀仅在最后GPU引用释放时卸载。",
              "- 初始化、编译与代表性普通/缓存prefill预热在计时外；CPU池第一次分配在实际运行内，池复用，未锁频率。",
              "- TTFT含排队，CPU postprocess首token时间；TPOT每请求首末token差/后续token数，ITL汇总相邻间隔。",
              "- model-run包含准备、模型、采样和同步，拷贝可与计算重叠，阶段总和不等于总墙钟。",
              "- CUDA event统计为各次传输stream时间之和，不能将其与generate时间简单相减来推导被隐藏的时间。",
              "- 随机采样会随调度次序变化；各请求输入与输出长度完全一致，模型正确性另用固定argmax采样和强制压力验证。",
              "", "## 复现", "", "```bash",
              f'{PYTHON} profiling/run_async_kv_offload_metrics.py --outdir /tmp/async-offload-recheck --runs {manifest["runs"]} --kv-blocks {manifest["kv_blocks"]}',
              "```", ""]
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--kv-blocks", type=int, default=1318)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    variant, root_hashes = prepare(outdir)
    shutil.copyfile(ROOT / "profiling/preemption_lock_1024_2026-10-05/workload_1024.json", outdir / "workload_1024.json")
    source = source_hashes(variant)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), root_hashes=root_hashes,
                    source_hashes=source, variant=str(variant), runs=args.runs, kv_blocks=args.kv_blocks,
                    workload_sha256=hashlib.sha256((outdir / "workload_1024.json").read_bytes()).hexdigest(), commands=[])
    results = dict(trials=[], medians={})
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print("PREPARED", variant)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH=str(variant), PYTHONUNBUFFERED="1")
    for trial in range(1, args.runs + 1):
        for enabled in ((False, True) if trial % 2 else (True, False)):
            assert source_hashes(variant) == source and source_hashes(ROOT) == root_hashes
            command = [PYTHON, "benchmark_inference_metrics.py", "--requests", "1024", "--kv-cache-dtype", "int8_half",
                       "--kv-blocks", str(args.kv_blocks)]
            if enabled:
                command.append("--kv-cpu-offload")
            log = outdir / f'offload{int(enabled)}_{trial}.txt'
            print("START", trial, enabled, flush=True)
            started = perf_counter()
            with log.open("w") as fp:
                run = subprocess.run(command, cwd=variant, env=env, stdout=fp, stderr=subprocess.STDOUT, timeout=600)
            assert run.returncode == 0, log
            text = log.read_text()
            row = parse_metrics(text)
            assert row["blocks"] == args.kv_blocks and row["input_tokens"] == 580663 and row["output_tokens"] == 583802
            assert f"Package: {variant}/nanovllm/__init__.py" in text
            assert f'Generated workload SHA256: {manifest["workload_sha256"]}' in text
            row.update(offload=enabled, trial=trial, log=log.name, process_wall_s=perf_counter()-started)

            def json_line(prefix):
                return json.loads(re.search(r"^" + re.escape(prefix) + r": (.*)$", text, re.M)[1])

            row.update(preemptions=int(re.search(r"Preemptions: (\d+)", text)[1]),
                       prefill_batches=json_line("Prefill batches"), request_preemption_stats=json_line("Per-request preemptions"),
                       decode_batch_stats=json_line("Actual decode batch sizes"),
                       kv_tensor_bytes=int(re.search(r"KV tensor bytes: (\d+)", text)[1]),
                       scale_tensor_bytes=int(re.search(r"Scale tensor bytes: (\d+)", text)[1]),
                       output_sha256=re.search(r"Output token SHA256: ([a-f0-9]{64})", text)[1])
            dispatch = json_line("Prefill attention dispatch")
            assert dispatch["flash"] == 0 and dispatch["int8"] > 0
            row["prefill_dispatch"] = dispatch
            if enabled:
                row["offload_stats"] = json_line("KV offload statistics")
                assert row["offload_stats"]["active_handles"] == row["offload_stats"]["restoring"] == 0
            results["trials"].append(row)
            results["medians"] = median_results(results["trials"])
            manifest["commands"].append(dict(command=command, log=log.name))
            (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print("DONE", trial, enabled, row["output_tokens_per_s"], "preemptions", row["preemptions"], flush=True)
    assert source_hashes(variant) == source and source_hashes(ROOT) == root_hashes
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), source_checks_passed=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    render(outdir, manifest, results)
    print("COMPLETE", outdir / "report.md", flush=True)


if __name__ == "__main__":
    main()
