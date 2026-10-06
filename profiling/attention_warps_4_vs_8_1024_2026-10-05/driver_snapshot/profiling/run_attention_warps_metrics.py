"""Paired fresh-process 4/8-warp fast INT8 benchmarks on the saved 1024 workload."""

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


def medians(trials):
    result = {}
    for warps in (4, 8):
        rows = [r for r in trials if r["num_warps"] == warps]
        if not rows:
            continue
        value = aggregate(rows)["int8_half"]
        for key in ("request_preemption_stats", "decode_batch_stats", "lock_stats"):
            value[key] = {k: statistics.median(r[key][k] for r in rows)
                          for k, v in rows[0][key].items() if isinstance(v, (int, float))}
        result[str(warps)] = value
    return result


def render(outdir, manifest, results):
    values = results["medians"]
    lines = ["# 快速双 scale INT8：4 与 8 warp，1024 请求", "",
             "仅修改 INT8 attention kernel 的 num_warps：prefill/decode 同时为 4 或 8。",
             "prefill BM=64、decode BM=16、BN=64、num_stages=1，原 prefill 优先策略，preemption_lock=False。",
             f"两配置各 {manifest['runs_per_case']} 个独立新进程，交替运行；每项指标取各轮中位数。", "",
             "| 指标 | 4 warp | 8 warp | 变化 |", "| --- | ---: | ---: | ---: |"]

    def row(label, getter, digits=2):
        old, new = (getter(values[str(w)]) for w in (4, 8))
        delta = f"{(new / old - 1) * 100:+.2f}%" if old else "—"
        lines.append(f"| {label} | {old:.{digits}f} | {new:.{digits}f} | {delta} |")

    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for q in ("mean", "p50", "p95", "p99"):
            row(f"{metric} {q.upper()} ms", lambda v, k=metric, p=q: v["latency_ms"][k][p])
    for key, label, digits in (("elapsed_s", "整批 generate 秒", 3),
                              ("output_tokens_per_s", "输出 token/s", 2),
                              ("requests_per_s", "请求/s", 2),
                              ("input_tokens_per_s", "输入 token/s", 2),
                              ("total_tokens_per_s", "总 token/s", 2),
                              ("preemptions", "抢占事件次数", 0),
                              ("blocks", "KV blocks", 0),
                              ("peak_allocated_gib", "峰值 allocated GiB", 2),
                              ("peak_reserved_gib", "峰值 reserved GiB", 2)):
        row(label, lambda v, k=key: v[k], digits)
    for key in ("preempted_requests", "repeatedly_preempted_requests", "maximum_per_request"):
        row(key, lambda v, k=key: v["request_preemption_stats"][k], 0)
    for stage in ("prefill", "decode"):
        for key, digits in (("seconds", 3), ("tokens", 0), ("tokens_per_s", 2)):
            row(f"{stage} model-run {key}", lambda v, s=stage, k=key: v["stages"][s][k], digits)
    for key in ("mean", "p50", "p95", "maximum", "steps"):
        row(f"实际 decode batch {key}", lambda v, k=key: v["decode_batch_stats"][k])
    lines += ["", "## 每轮结果", "",
              "| warp | 轮次 | 输出 token/s | generate s | TTFT P50 ms | TPOT P50 ms | 抢占 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results["trials"]:
        lines.append(f"| {r['num_warps']} | {r['trial']} | {r['output_tokens_per_s']:.2f} | {r['elapsed_s']:.3f} | "
                     f"{r['latency_ms']['TTFT']['p50']:.2f} | {r['latency_ms']['TPOT']['p50']:.2f} | {r['preemptions']} |")
    lines += ["", "## 正确性与测量范围", "",
              "- 测量前，4/8 warp 与还原 BF16 KV 后的 FlashAttention 比较；decode、零长度 graph padding、普通 prefill、缓存前缀 prefill 共 5 类用例通过。模型形状 Q heads=16、KV heads=8、head_dim=128，双 scale，rtol=0.02、atol=0.005；详见 correctness.json。",
              f"- Qwen3-0.6B / RTX 3090 Ti / GPU {manifest['gpu']}，权重 BF16，KV 为双 scale INT8。",
              "- 复用上次完整 workload_1024.json，输入 580663、输出 583802 token。输入/输出长度各 100–1024；seed=0、temperature=0.6、ignore_eos=True。每请求输出长度和时间戳数量均验证。",
              "- max_num_seqs=512、max_num_batched_tokens=16384、max_model_len=4096、block size=256、gpu_memory_utilization=0.9，decode CUDA Graph、prefill eager。未锁 GPU 频率。",
              "- 初始化和预热在计时前，普通 prefill 长度 256/512/768/1024（最大 16×1024）；缓存前缀 prefill 页表宽度 2–8、新 Q 长度 128。预热后重置采样 seed 和显存峰值。",
              "- TTFT 包含排队，取 CPU postprocess 首 token 回填时间；TPOT 为每请求首末 token 时间差除以后续 token 数；ITL 汇总相邻 token 间隔。",
              "- model-run 阶段时间包括输入准备、模型、采样和同步，不能当作单 attention kernel 的 GPU 时间。",
              "- CUDA Graph 捕获和正式 prefill/decode 均使用对应 4/8 warp 源码；实际包路径、attention 源码配置、KV dtype、prefill 派发和源码哈希均核对。",
              "- 主仓库推理源码及 reference 源码均未修改；结果只适用于本次固定模型、负载与 kernel 版本。", "",
              "## 复现", "", "```bash",
              f"{PYTHON} profiling/run_attention_warps_metrics.py --outdir /tmp/attention-warps-recheck --runs {manifest['runs_per_case']}", "```", ""]
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "profiling/preemption_lock_1024_2026-10-05")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    assert args.runs > 0
    outdir, reference = args.outdir.resolve(), args.reference.resolve()
    assert not outdir.exists(), "use a fresh output directory"
    prior = json.loads((reference / "manifest.json").read_text())
    source = Path(prior["variant_roots"]["fast_int8_half"])
    source_hash = prior["variant_source_hashes"]["fast_int8_half"]
    assert source_hashes(source) == source_hash
    originals = source_hashes(ROOT)
    assert originals == prior["root_source_hashes"]
    model = Path(prior["model"])
    assert json.loads((model / "config.json").read_text()) == prior["model_config"]
    for item in prior["weight_files"]:
        stat = (model / item["name"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (item["size"], item["mtime_ns"])
    outdir.mkdir(parents=True)
    code = (source / "benchmark_inference_metrics.py").read_text()
    marker = "    print('Package: ' + nanovllm.__file__, flush=True)\n"
    assert code.count(marker) == 1
    code = code.replace(marker, marker +
        "    from pathlib import Path\n"
        "    import nanovllm.layers.quantized_attention as attention_impl\n"
        "    kernel_source = Path(attention_impl.__file__).read_text()\n"
        "    print('Attention num_warps: ' + kernel_source.split('num_warps=', 1)[1].split(',', 1)[0], flush=True)\n")
    compile(code, "benchmark_inference_metrics.py", "exec")
    variants = {}
    kernel_path = "nanovllm/layers/quantized_attention.py"
    for warps in (4, 8):
        target = outdir / f"variants/warps{warps}"
        for relative in source_hash:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == "benchmark_inference_metrics.py":
                destination.write_text(code)
            elif relative == kernel_path and warps == 8:
                kernel = (source / relative).read_text()
                assert kernel.count("num_warps=4, num_stages=1,") == 1
                destination.write_text(kernel.replace("num_warps=4, num_stages=1,", "num_warps=8, num_stages=1,"))
            else:
                shutil.copyfile(source / relative, destination)
        variants[str(warps)] = target
    hashes = {w: source_hashes(path) for w, path in variants.items()}
    assert {k for k in hashes["4"] if hashes["4"][k] != hashes["8"][k]} == {kernel_path}
    assert {k for k in hashes["4"] if hashes["4"][k] != source_hash[k]} == {"benchmark_inference_metrics.py"}
    shutil.copyfile(reference / "workload_1024.json", outdir / "workload_1024.json")
    workload = prior["workload"]
    assert hashlib.sha256((outdir / "workload_1024.json").read_bytes()).hexdigest() == workload["sha256"]
    driver = outdir / "driver_snapshot/profiling"
    driver.mkdir(parents=True)
    for filename in (Path(__file__).name, "check_attention_warps.py", "run_current_int8_metrics.py", "run_kv_metrics_comparison.py"):
        shutil.copyfile(ROOT / "profiling" / filename, driver / filename)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), reference=str(reference),
                    runs_per_case=args.runs, gpu=args.gpu, python=PYTHON,
                    model=str(model), model_config=prior["model_config"], weight_files=prior["weight_files"],
                    config=dict(prior["config"], preemption_lock=False), workload=workload, warmup=prior["warmup"],
                    kernel_config=dict(prefill_BM=64, decode_BM=16, BN=64, num_stages=1, scale_groups=2, num_warps=[4, 8]),
                    root_source_hashes=originals, reference_source_hashes=source_hash,
                    variant_source_hashes=hashes, variant_roots={k: str(v) for k, v in variants.items()},
                    nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True), runs=[])
    results = dict(trials=[], medians={})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")

    def save():
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def unchanged():
        assert source_hashes(ROOT) == originals
        assert source_hashes(source) == source_hash
        assert all(source_hashes(path) == hashes[w] for w, path in variants.items())

    save()
    print("START correctness checks for 4/8 warp", flush=True)
    correctness_command = [PYTHON, str(ROOT / "profiling/check_attention_warps.py"), "--outdir", str(outdir)]
    with (outdir / "correctness.txt").open("w") as fp:
        completed = subprocess.run(correctness_command, env=env, stdout=fp, stderr=subprocess.STDOUT, timeout=180)
    assert completed.returncode == 0, f"correctness/compilation failed: {outdir / 'correctness.txt'}"
    correctness = json.loads((outdir / "correctness.json").read_text())
    assert correctness["status"] == "PASS" and len(correctness["cases"]) == 5
    manifest["correctness_command"] = correctness_command
    manifest["correctness_passed"] = True
    unchanged()
    save()
    print("PASS all 5 correctness cases; START full-model benchmark", flush=True)
    for trial in range(1, args.runs + 1):
        for warps in ((4, 8) if trial % 2 else (8, 4)):
            unchanged()
            path = variants[str(warps)]
            log = outdir / f"warps{warps}_{trial}.txt"
            command = [PYTHON, "benchmark_inference_metrics.py", "--model", str(model), "--requests", "1024", "--kv-cache-dtype", "int8_half"]
            print(f"START trial={trial}/{args.runs} num_warps={warps}", flush=True)
            started = perf_counter()
            with log.open("w") as fp:
                completed = subprocess.run(command, cwd=path, env=dict(env, PYTHONPATH=str(path)),
                                           stdout=fp, stderr=subprocess.STDOUT, timeout=600)
            assert completed.returncode == 0, f"measurement failed: {log}"
            unchanged()
            text = log.read_text()
            assert f"Package: {path}/nanovllm/__init__.py" in text and f"Attention num_warps: {warps}" in text
            assert "Preemption lock enabled: False" in text and "Actual KV dtype: torch.int8" in text
            row = parse_metrics(text)
            assert row["requests"] == 1024 and row["mode"] == "int8_half" and row["graph"]
            assert row["input_tokens"] == workload["input_tokens"] and row["output_tokens"] == workload["output_tokens"]

            def json_line(prefix):
                return json.loads(re.search(r"^" + re.escape(prefix) + r": (.*)$", text, re.M)[1])

            dispatch = json_line("Prefill attention dispatch")
            assert dispatch["flash"] == 0 and dispatch["int8"] > 0
            count = int(re.search(r"Preemptions: (\d+)", text)[1])
            request_stats = json_line("Per-request preemptions")
            assert sum(request_stats["histogram"].values()) == 1024
            assert sum(int(k) * v for k, v in request_stats["histogram"].items()) == count
            row.update(num_warps=warps, trial=trial, log=log.name, process_wall_s=perf_counter()-started,
                       preemptions=count, request_preemption_stats=request_stats,
                       lock_stats=json_line("Lock statistics"), decode_batch_stats=json_line("Actual decode batch sizes"),
                       prefill_batches=json_line("Prefill batches"), prefill_attention_dispatch=dispatch,
                       kv_tensor_bytes=int(re.search(r"KV tensor bytes: (\d+)", text)[1]),
                       scale_tensor_bytes=int(re.search(r"Scale tensor bytes: (\d+)", text)[1]),
                       output_sha256=re.search(r"Output token SHA256: ([a-f0-9]{64})", text)[1])
            results["trials"].append(row)
            results["medians"] = medians(results["trials"])
            manifest["runs"].append(dict(trial=trial, num_warps=warps, command=command, log=log.name))
            save()
            print(f"DONE trial={trial} num_warps={warps} output={row['output_tokens_per_s']:.2f}tok/s "
                  f"TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms TPOT_P50={row['latency_ms']['TPOT']['p50']:.2f}ms "
                  f"preemptions={count}", flush=True)
    unchanged()
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), root_sources_unchanged=True, reference_sources_unchanged=True)
    save()
    render(outdir, manifest, results)
    print("COMPLETE " + str(outdir / "report.md"), flush=True)


if __name__ == "__main__":
    main()
