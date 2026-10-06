"""Measure the verified fast dual-scale INT8 variant against BF16.

Copies the exact saved inference variant and measures fresh paired processes
for each request count. The current inference sources remain unchanged.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_current_int8_metrics import PYTHON, aggregate, instrument_benchmark, source_hashes
from profiling.run_kv_metrics_comparison import parse_metrics


def audited_benchmark(source):
    source = instrument_benchmark(source)
    old = "import nanovllm\n"
    assert source.count(old) == 1
    source = source.replace(old, old + "import nanovllm.layers.attention as attention_module\n")
    old = "    preemptions = 0\n"
    assert source.count(old) == 1
    source = source.replace(old,
        "    attention_dispatch = {'flash': 0, 'int8': 0}\n"
        "    original_int8_attention = attention_module.int8_paged_attention\n"
        "    original_flash_attention = attention_module.flash_attn_varlen_func\n"
        "    def audited_int8_attention(*call_args, **kwargs):\n"
        "        if attention_module.get_context().is_prefill:\n"
        "            attention_dispatch['int8'] += 1\n"
        "        return original_int8_attention(*call_args, **kwargs)\n"
        "    def audited_flash_attention(*call_args, **kwargs):\n"
        "        attention_dispatch['flash'] += 1\n"
        "        return original_flash_attention(*call_args, **kwargs)\n"
        "    attention_module.int8_paged_attention = audited_int8_attention\n"
        "    attention_module.flash_attn_varlen_func = audited_flash_attention\n" + old)
    old = "    print('Preemptions: ' + str(preemptions))\n"
    assert source.count(old) == 1
    source = source.replace(old, old + "    print('Prefill attention dispatch: ' + json.dumps(attention_dispatch))\n")
    compile(source, "benchmark_inference_metrics.py", "exec")
    return source


def render(outdir, manifest, results):
    lines = ["# 更快双 scale INT8 与 BF16：256 / 512 请求重新对比", "",
        "使用此前已保存的 fast_int8 推理源码：prefill BM=64、BN=64；decode BM=16、BN=64；",
        "两阶段均 num_stages=1、num_warps=4。INT8 模式的所有正式 prefill 和 decode 均走自写 kernel。",
        "BF16 模式保持原 FlashAttention prefill/decode。两组模型权重均为 BF16，INT8 仅量化 KV cache。",
        "每轮通过源码哈希、实际包路径、KV dtype、prefill 派发计数及各请求输出长度确认测试版本。", ""]
    for requests in manifest["request_counts"]:
        medians = results["medians"][str(requests)]
        lines += [f"## {requests} 请求（三轮中位数）", "",
            "| 指标 | BF16 | 更快双 scale INT8 | 变化 |", "| --- | ---: | ---: | ---: |"]
        def row(label, getter, digits=2):
            old, new = [getter(medians[m]) for m in ("auto", "int8_half")]
            delta = f"{(new / old - 1) * 100:+.2f}%" if old else "—"
            lines.append(f"| {label} | {old:.{digits}f} | {new:.{digits}f} | {delta} |")
        for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
            for q in ("mean", "p50", "p95", "p99"):
                row(f"{metric} {q.upper()} ms", lambda v, k=metric, p=q: v["latency_ms"][k][p])
        for label, key, digits in (("generate s", "elapsed_s", 3), ("输出 token/s", "output_tokens_per_s", 2),
            ("请求/s", "requests_per_s", 2), ("输入 token/s", "input_tokens_per_s", 2),
            ("总 token/s", "total_tokens_per_s", 2), ("KV blocks", "blocks", 0),
            ("抢占次数", "preemptions", 0), ("峰值 allocated GiB", "peak_allocated_gib", 2),
            ("峰值 reserved GiB", "peak_reserved_gib", 2)):
            row(label, lambda v, k=key: v[k], digits)
        for stage in ("prefill", "decode"):
            for key, digits in (("seconds", 3), ("tokens", 0), ("tokens_per_s", 2)):
                row(f"{stage} model-run {key}", lambda v, s=stage, k=key: v["stages"][s][k], digits)
        row("平均首 token 后未完成请求数", lambda v: v["average_started_unfinished_requests"], 1)
        lines += [""]
    lines += ["## 每轮记录", "", "| 请求数 | 模式 | 轮次 | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in results["trials"]:
        lines.append(f"| {r['requests']} | {r['variant']} | {r['trial']} | {r['output_tokens_per_s']:.2f} | {r['latency_ms']['TTFT']['p50']:.2f} | {r['latency_ms']['TPOT']['p50']:.2f} | {r['preemptions']} |")
    lines += ["", "## 配置和测量范围", "",
        f"- Qwen3-0.6B / {results['trials'][0]['gpu']} / GPU {manifest['gpu']}，decode CUDA Graph、prefill eager；未锁 GPU 频率。",
        "- 每负载、每模式各三个独立新进程；同一负载内 BF16 与更快 INT8 交替运行，全为本次新测量。",
        "- 请求同时到达；输入/输出长度各 100–1024；负载及 PyTorch 采样 seed=0，temperature=0.6，ignore_eos=True。",
        "- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9。",
        "- 正式计时前预热普通 prefill 长度 256/512/768/1024（最大 16×1024），及页表宽度 2–8 的缓存前缀 prefill（新 Q 长度 128）；之后重置采样种子和显存峰值。",
        "- TTFT 为批量提交到 CPU postprocess 首 token 回填完成，包含排队；TPOT 为每请求首末 token 时间差除以后续 token 数，ITL 汇总相邻 token 间隔。",
        "- 阶段时间含输入准备、模型、采样和同步；整模型结果包含 KV 容量与调度效应。双 scale INT8 每 token KV+scale 字节数为 BF16 的 53.125%。",
        "- 平均首 token 后未完成请求数由 ITL_mean × (输出 token 总数−请求数) / generate 时间推导，包含被抢占请求，不能当作实测 decode batch size。",
        "- 同负载的输入、目标输出数量严格相同；随机 token ID 离线负载，这次未重测语言精度。之前同一 fast_int8 源码的 PPL 记录保存在 reference 目录。",
        "- 原工作树和 reference 实验源码均未修改。所有实际运行源码、配置、完整负载、派发计数和日志保存在本目录。", "",
        "## 复现", "", "```bash",
        f"{PYTHON} profiling/run_fast_int8_metrics.py --outdir /tmp/fast-int8-recheck --requests {' '.join(map(str, manifest['request_counts']))} --runs {manifest['runs_per_mode']}", "```", ""]
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--reference", type=Path, default=ROOT / "profiling/fast_int8_vs_bf16_paired_2026-10-05")
    args = parser.parse_args()
    assert args.runs > 0 and all(n > 0 for n in args.requests) and len(set(args.requests)) == len(args.requests)
    outdir, reference = args.outdir.resolve(), args.reference.resolve()
    assert not outdir.exists(), "use a fresh output directory"
    prior = json.loads((reference / "manifest.json").read_text())
    originals = source_hashes(ROOT)
    assert originals == prior["original_source_hashes"]
    fast = reference / "variants/fast_int8"
    assert source_hashes(fast) == prior["variant_source_hashes"]["fast_int8"]
    outdir.mkdir(parents=True)
    for relative in originals:
        target = outdir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    variants = {}
    code = audited_benchmark((ROOT / "benchmark_inference_metrics.py").read_text())
    for name, source in (("bf16", ROOT), ("fast_int8_half", fast)):
        target_root = outdir / "variants" / name
        for relative in originals:
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == "benchmark_inference_metrics.py":
                target.write_text(code)
            else:
                shutil.copyfile(source / relative, target)
        variants[name] = target_root
    hashes = {name: source_hashes(root) for name, root in variants.items()}
    scripts = outdir / "driver_snapshot/profiling"
    scripts.mkdir(parents=True)
    for filename in (Path(__file__).name, "run_current_int8_metrics.py", "run_kv_metrics_comparison.py"):
        shutil.copyfile(ROOT / "profiling" / filename, scripts / filename)
    workloads = {}
    for requests in args.requests:
        rng = random.Random(0)
        prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, 1024))] for _ in range(requests)]
        lengths = [rng.randint(100, 1024) for _ in range(requests)]
        file = outdir / f"workload_{requests}.json"
        file.write_text(json.dumps(dict(prompts=prompts, max_tokens=lengths, seed=0)) + "\n")
        workloads[str(requests)] = dict(input_tokens=sum(map(len, prompts)), output_tokens=sum(lengths),
            file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    model = Path(prior["model"])
    assert json.loads((model / "config.json").read_text()) == prior["model_config"]
    for weight in prior["weight_files"]:
        stat = (model / weight["name"]).stat()
        assert stat.st_size == weight["size"] and stat.st_mtime_ns == weight["mtime_ns"]
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), reference=str(reference),
        request_counts=args.requests, runs_per_mode=args.runs, gpu=args.gpu, python=PYTHON,
        model=str(model), model_config=prior["model_config"], weight_files=prior["weight_files"],
        original_source_hashes=originals, reference_source_hashes=prior["variant_source_hashes"]["fast_int8"],
        variant_source_hashes=hashes, variant_roots={name: str(root) for name, root in variants.items()},
        kernel_config=dict(prefill=dict(BM=64, BN=64, num_stages=1, num_warps=4),
            decode=dict(BM=16, BN=64, num_stages=1, num_warps=4), scale_groups=2,
            first_prefill="int8_paged_attention", cached_prefill="int8_paged_attention"),
        workloads=workloads, config=dict(max_model_len=4096, max_num_batched_tokens=16384,
            max_num_seqs=512, kvcache_block_size=256, gpu_memory_utilization=0.9,
            tensor_parallel_size=1, cuda_graph=True, seed=0, temperature=0.6, ignore_eos=True),
        warmup=dict(fresh_lengths=[256, 512, 768, 1024], fresh_max_batch_tokens=16384,
            cached_prefix_table_widths=list(range(2, 9)), cached_new_tokens=128),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True),
        runs=[])
    results = dict(trials=[], medians={})
    def save():
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    def unchanged():
        assert source_hashes(ROOT) == originals
        assert source_hashes(fast) == prior["variant_source_hashes"]["fast_int8"]
        assert all(source_hashes(root) == hashes[name] for name, root in variants.items())
    save()
    for requests in args.requests:
        for trial in range(1, args.runs + 1):
            for name in (("bf16", "fast_int8_half") if trial % 2 else ("fast_int8_half", "bf16")):
                unchanged()
                mode = "auto" if name == "bf16" else "int8_half"
                root = variants[name]
                log = outdir / f"requests_{requests}_{name}_{trial}.txt"
                command = [PYTHON, "benchmark_inference_metrics.py", "--model", str(model),
                    "--requests", str(requests), "--kv-cache-dtype", mode]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", PYTHONPATH=str(root))
                print(f"START requests={requests} trial={trial} variant={name}", flush=True)
                started = perf_counter()
                with log.open("w") as fp:
                    process = subprocess.run(command, cwd=root, env=env, stdout=fp, stderr=subprocess.STDOUT, timeout=600)
                assert process.returncode == 0, f"measurement failed: {log}"
                unchanged()
                text = log.read_text()
                assert f"Package: {root}/nanovllm/__init__.py" in text
                row = parse_metrics(text)
                workload = workloads[str(requests)]
                assert row["requests"] == requests and row["mode"] == mode and row["graph"]
                assert row["input_tokens"] == workload["input_tokens"] and row["output_tokens"] == workload["output_tokens"]
                dispatch = json.loads(re.search(r"Prefill attention dispatch: (.*)", text)[1])
                assert (dispatch["int8"] == 0 and dispatch["flash"] > 0) if name == "bf16" else (dispatch["flash"] == 0 and dispatch["int8"] > 0)
                assert ("torch.bfloat16" if name == "bf16" else "torch.int8") in re.search(r"Actual KV dtype: (.*)", text)[1]
                row.update(variant=name, trial=trial, log=log.name, process_wall_s=perf_counter()-started,
                    preemptions=int(re.search(r"Preemptions: (\d+)", text)[1]),
                    prefill_batches=json.loads(re.search(r"Prefill batches: (.*)", text)[1]),
                    kv_tensor_bytes=int(re.search(r"KV tensor bytes: (\d+)", text)[1]),
                    scale_tensor_bytes=int(re.search(r"Scale tensor bytes: (\d+)", text)[1]),
                    output_sha256=re.search(r"Output token SHA256: ([a-f0-9]{64})", text)[1], prefill_attention_dispatch=dispatch)
                row["average_started_unfinished_requests"] = row["latency_ms"]["ITL"]["mean"] * (row["output_tokens"] - requests) / (1000 * row["elapsed_s"])
                results["trials"].append(row)
                rows = [r for r in results["trials"] if r["requests"] == requests]
                medians = aggregate(rows)
                for m, value in medians.items():
                    value["average_started_unfinished_requests"] = statistics.median(r["average_started_unfinished_requests"] for r in rows if r["mode"] == m)
                results["medians"][str(requests)] = medians
                manifest["runs"].append(dict(requests=requests, trial=trial, variant=name, command=command, log=log.name))
                save()
                print(f"DONE requests={requests} trial={trial} variant={name} output={row['output_tokens_per_s']:.2f}tok/s TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms TPOT_P50={row['latency_ms']['TPOT']['p50']:.2f}ms prefill_dispatch={dispatch}", flush=True)
    unchanged()
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), original_repository_unchanged=True,
        reference_repository_unchanged=True)
    save()
    render(outdir, manifest, results)
    print("COMPLETE " + str(outdir / "report.md"), flush=True)


if __name__ == "__main__":
    main()
