"""Compare the current INT8 KV kernels against BF16 in fresh processes.

Inference sources are copied without changes. Only the copied benchmark gains
sampling seeds, representative warmup, and counters outside model execution.
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
from profiling.run_kv_metrics_comparison import parse_metrics

PYTHON = "/home/xgd/anaconda3/envs/nanovllm/bin/python"


def source_hashes(root):
    paths = sorted((root / "nanovllm").rglob("*.py")) + [root / "benchmark_inference_metrics.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def instrument_benchmark(source):
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    replace("import argparse\n", "import argparse\nimport hashlib\nimport json\nimport nanovllm\n")
    replace("    args = parser.parse_args()\n", "    args = parser.parse_args()\n"
            "    print('Package: ' + nanovllm.__file__, flush=True)\n"
            "    torch.manual_seed(args.seed)\n")
    replace("    torch.cuda.reset_peak_memory_stats()\n",
            "    # Distinct warmup IDs prevent prefix hits in the measured workload.\n"
            "    for warm_length in (256, 512, 768, 1024):\n"
            "        warm_batch = 16 if warm_length == 1024 else 4\n"
            "        warm_prompts = [[15000 + warm_length + request] * warm_length\n"
            "                        for request in range(warm_batch)]\n"
            "        llm.generate(warm_prompts, SamplingParams(max_tokens=2, ignore_eos=True), use_tqdm=False)\n"
            "    torch.manual_seed(args.seed)\n"
            "    torch.cuda.reset_peak_memory_stats()\n")
    replace("    token_times: dict[int, list[float]] = {}\n",
            "    preemptions = 0\n"
            "    prefill_dispatch = {'fresh': 0, 'cached': 0}\n"
            "    original_preempt = llm.scheduler.preempt\n"
            "    def counted_preempt(seq):\n"
            "        nonlocal preemptions\n"
            "        preemptions += 1\n"
            "        return original_preempt(seq)\n"
            "    llm.scheduler.preempt = counted_preempt\n"
            "    token_times: dict[int, list[float]] = {}\n")
    replace('        count = sum(seq.num_scheduled_tokens for seq in seqs)\n',
            '        count = sum(seq.num_scheduled_tokens for seq in seqs)\n'
            "        if is_prefill:\n"
            "            prefill_dispatch['cached' if any(seq.num_cached_tokens > 0 for seq in seqs) else 'fresh'] += 1\n")
    replace('    print(f"GPU: ',
            "    print('Output token SHA256: ' + hashlib.sha256(json.dumps([o['token_ids'] for o in outputs], separators=(',', ':')).encode()).hexdigest())\n"
            "    print('Preemptions: ' + str(preemptions))\n"
            "    print('Prefill batches: ' + json.dumps(prefill_dispatch))\n"
            "    print('Actual KV dtype: ' + str(llm.model_runner.kv_cache.dtype))\n"
            "    print('KV tensor bytes: ' + str(llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()))\n"
            "    scales = getattr(llm.model_runner, 'kv_scales', None)\n"
            "    print('Scale tensor bytes: ' + str(0 if scales is None else scales.numel() * scales.element_size()))\n"
            '    print(f"GPU: ')
    compile(source, "benchmark_inference_metrics.py", "exec")
    return source


def aggregate(trials):
    medians = {}
    for mode in dict.fromkeys(row["mode"] for row in trials):
        rows = [r for r in trials if r["mode"] == mode]
        def collect(values):
            if isinstance(values[0], dict):
                return {key: collect([v[key] for v in values]) for key in values[0]}
            return statistics.median(values)
        numeric = ("blocks", "elapsed_s", "requests_per_s", "input_tokens_per_s",
                   "output_tokens_per_s", "total_tokens_per_s", "peak_allocated_gib",
                   "peak_reserved_gib", "latency_ms", "stages", "preemptions",
                   "prefill_batches", "kv_tensor_bytes", "scale_tensor_bytes")
        medians[mode] = {key: collect([r[key] for r in rows]) for key in numeric}
        medians[mode]["runs"] = len(rows)
    return medians


def report(outdir, manifest, results):
    modes = manifest["modes"]
    median = results["medians"]
    labels = {"auto": "BF16", "int8": "单 scale INT8", "int8_half": "双 scale INT8"}
    lines = ["# 当前仓库 INT8 kernel 与未量化 BF16 的完整推理对比", "",
             "当前推理源码完整保留，仅在独立副本添加测评预热、固定采样种子和 CPU 计数。",
             "模型权重均为 BF16；INT8 仅量化 KV cache。普通首次 prefill 仍使用 FlashAttention；",
             "INT8 自写 Triton kernel 用于 decode 和带缓存前缀的 prefill（BM=16、BN=64、4 warps，默认流水）。", "",
             "| 指标 | " + " | ".join(labels[m] for m in modes) + " |",
             "| --- | " + " | ".join("---:" for _ in modes) + " |"]
    def row(label, getter, digits=2):
        lines.append("| " + label + " | " + " | ".join(f"{getter(median[m]):.{digits}f}" for m in modes) + " |")
    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for q in ("mean", "p50", "p95", "p99"):
            row(f"{metric} {q.upper()} ms", lambda v, k=metric, p=q: v["latency_ms"][k][p])
    for label, key, digits in (("generate s", "elapsed_s", 3), ("请求/s", "requests_per_s", 2),
                               ("输出 token/s", "output_tokens_per_s", 2), ("输入 token/s", "input_tokens_per_s", 2),
                               ("总 token/s", "total_tokens_per_s", 2), ("KV blocks", "blocks", 0),
                               ("缓存抢占次数", "preemptions", 0), ("峰值 allocated GiB", "peak_allocated_gib", 2),
                               ("峰值 reserved GiB", "peak_reserved_gib", 2)):
        row(label, lambda v, k=key: v[k], digits)
    row("KV cache + scales GiB", lambda v: (v["kv_tensor_bytes"] + v["scale_tensor_bytes"]) / 1024**3, 4)
    for stage in ("prefill", "decode"):
        for key, digits in (("seconds", 3), ("tokens", 0), ("tokens_per_s", 2)):
            row(f"{stage} model-run {key}", lambda v, s=stage, k=key: v["stages"][s][k], digits)
    lines += ["", "## 相对 BF16 的变化", "", "| 模式 | TTFT P50 | TPOT P50 | ITL P50 | 输出吞吐 | generate 时间 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    getters = [lambda v, k=k: v["latency_ms"][k]["p50"] for k in ("TTFT", "TPOT", "ITL")]
    getters += [lambda v: v["output_tokens_per_s"], lambda v: v["elapsed_s"]]
    for mode in modes:
        if mode != "auto":
            lines.append("| " + labels[mode] + " | " + " | ".join(f"{(f(median[mode]) / f(median['auto']) - 1) * 100:+.2f}%" for f in getters) + " |")
    lines += ["", "## 每轮结果", "", "| 模式 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results["trials"]:
        lines.append(f"| {r['mode']} | {r['trial']} | {r['elapsed_s']:.3f} | {r['output_tokens_per_s']:.2f} | {r['latency_ms']['TTFT']['p50']:.2f} | {r['latency_ms']['TPOT']['p50']:.2f} | {r['preemptions']} |")
    c = manifest["workload_config"]
    lines += ["", "## 测量范围", "",
        f"- Qwen3-0.6B / {results['trials'][0]['gpu']} / GPU {manifest['gpu']}，单卡，decode CUDA Graph、prefill eager。未锁 GPU 频率。",
        f"- 每模式 {manifest['runs_per_mode']} 个独立新进程，逐轮循环轮换运行顺序，全部为本次新测量。每轮请求/token 分布先计算 Mean/P50/P95/P99，再逐项取各轮中位数。",
        f"- {c['requests']} 请求同时到达，输入长度 {c['min_input']}–{c['max_input']}，输出长度 {c['min_output']}–{c['max_output']}，负载和 PyTorch 采样 seed={c['seed']}。temperature=0.6、ignore_eos=True。",
        f"- 每轮输入 {manifest['input_tokens']:,}、输出 {manifest['expected_output_tokens']:,} token；每请求输出长度、时间戳数量均验证通过。随机 token ID 离线负载，未评价自然语言质量。",
        "- 初始化、编译、生成预热及代表形状预热均在计时前。代表预热为长度 256/512/768/1024，最后一组 16×1024 token；使用正式输入范围外的 token ID。",
        "- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9。",
        "- TTFT 从批量提交到 CPU Scheduler.postprocess 首 token 回填完成，包含排队。TPOT=(末 token 时间−首 token 时间)/(输出长度−1)；ITL 汇总所有请求相邻 token 间隔。总延迟为末 token 时间−批量提交时间。",
        "- 吞吐分母为完整 generate 墙钟时间；model-run 阶段时间包括输入准备、模型、采样和同步，不能当作单 attention kernel 时间。",
        "- 三种模式使用相同显存预算，INT8 的 KV 容量更大。抢占后的重新 prefill 会影响 BF16 总时间；根据阶段执行 token 和抢占计数解释收益。",
        "- 单 scale INT8 每 token 缓存字节为 BF16 的 51.5625%；双 scale 为 53.125%。节省空间用于分配更多 blocks，完整缓存池与峰值显存不会按比例减少。显存为 PyTorch allocated/reserved 峰值。",
        "- 单 scale 与双 scale 共用当前 _int8_paged_attention_kernel，通过 SCALE_GROUPS=1/2 区分；不使用此前 BM=64 实验副本。",
        "- 生成内容可以因量化与调度轨迹而不同，固定采样种子不保证不同模式输出相同；输入和输出数量严格相同。",
        "- workload.json 保存完整负载；manifest.json 保存源码哈希、模型配置、权重文件元数据、环境和命令；results.json 保存结构化结果，source_snapshot 保存原始源码，measurement_snapshot 保存实际执行源码。", "",
        "## 复现", "", "```bash",
        f"{PYTHON} profiling/run_current_int8_metrics.py --outdir /tmp/current-int8-recheck --modes {' '.join(modes)} --runs {manifest['runs_per_mode']}", "```", ""]
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=("auto", "int8", "int8_half"), default=["auto", "int8", "int8_half"])
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    assert args.runs > 0 and "auto" in args.modes and len(set(args.modes)) == len(args.modes)
    outdir = args.outdir.resolve()
    assert not outdir.exists(), "use a fresh output directory"
    outdir.mkdir(parents=True)
    original_hashes = source_hashes(ROOT)
    for name in ("source_snapshot", "measurement_snapshot"):
        for relative in original_hashes:
            target = outdir / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
    measured_root = outdir / "measurement_snapshot"
    benchmark = measured_root / "benchmark_inference_metrics.py"
    benchmark.write_text(instrument_benchmark(benchmark.read_text()))
    measured_hashes = source_hashes(measured_root)
    shutil.copyfile(Path(__file__), outdir / Path(__file__).name)
    shutil.copyfile(ROOT / "profiling/run_kv_metrics_comparison.py", outdir / "parser_snapshot.py")
    workload_config = dict(requests=256, min_input=100, max_input=1024, min_output=100, max_output=1024, seed=0)
    rng = random.Random(0)
    prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, 1024))] for _ in range(256)]
    lengths = [rng.randint(100, 1024) for _ in range(256)]
    workload_path = outdir / "workload.json"
    workload_path.write_text(json.dumps(dict(prompts=prompts, max_tokens=lengths, seed=0)) + "\n")
    model = Path(args.model)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), model=str(model), python=PYTHON,
        gpu=args.gpu, modes=args.modes, runs_per_mode=args.runs, workload_config=workload_config,
        config=dict(max_model_len=4096, max_num_batched_tokens=16384, max_num_seqs=512,
                    gpu_memory_utilization=0.9, tensor_parallel_size=1, kvcache_block_size=256,
                    temperature=0.6, ignore_eos=True, cuda_graph=True, sampling_seed=0),
        source_hashes=original_hashes, measurement_source_hashes=measured_hashes,
        input_tokens=sum(map(len, prompts)), expected_output_tokens=sum(lengths),
        workload_sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest(),
        model_config=json.loads((model / "config.json").read_text()),
        weight_files=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in model.glob("*.safetensors")],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True),
        runs=[])
    environment_command = [PYTHON, "-c", "import torch,triton,flash_attn,transformers; print('torch',torch.__version__); print('triton',triton.__version__); print('flash_attn',flash_attn.__version__); print('transformers',transformers.__version__)"]
    manifest["environment"] = subprocess.check_output(environment_command, text=True)
    results = dict(trials=[], medians={})
    def save():
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    def unchanged():
        assert source_hashes(ROOT) == original_hashes, "repository source changed"
        assert source_hashes(measured_root) == measured_hashes, "measurement source changed"
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", PYTHONPATH=str(measured_root))
    for trial in range(1, args.runs + 1):
        offset = (trial - 1) % len(args.modes)
        order = args.modes[offset:] + args.modes[:offset]
        for mode in order:
            unchanged()
            log = outdir / f"{mode}_{trial}.txt"
            command = [PYTHON, "benchmark_inference_metrics.py", "--model", str(model), "--kv-cache-dtype", mode]
            print(f"START trial={trial} mode={mode} log={log}", flush=True)
            began = perf_counter()
            with log.open("w") as fp:
                process = subprocess.run(command, cwd=measured_root, env=env, stdout=fp, stderr=subprocess.STDOUT, timeout=600)
            assert process.returncode == 0, f"benchmark failed: {log}"
            unchanged()
            output = log.read_text()
            assert f"Package: {measured_root}/nanovllm/__init__.py" in output
            row = parse_metrics(output)
            assert row["mode"] == mode and row["graph"]
            assert row["input_tokens"] == manifest["input_tokens"] and row["output_tokens"] == manifest["expected_output_tokens"]
            row.update(trial=trial, log=log.name, process_wall_s=perf_counter()-began,
                preemptions=int(re.search(r"Preemptions: (\d+)", output)[1]),
                prefill_batches=json.loads(re.search(r"Prefill batches: (.*)", output)[1]),
                kv_tensor_bytes=int(re.search(r"KV tensor bytes: (\d+)", output)[1]),
                scale_tensor_bytes=int(re.search(r"Scale tensor bytes: (\d+)", output)[1]),
                output_sha256=re.search(r"Output token SHA256: ([a-f0-9]{64})", output)[1])
            assert ("torch.bfloat16" if mode == "auto" else "torch.int8") in re.search(r"Actual KV dtype: (.*)", output)[1]
            results["trials"].append(row)
            results["medians"] = aggregate(results["trials"])
            manifest["runs"].append(dict(trial=trial, mode=mode, command=command, log=log.name))
            save()
            print(f"DONE trial={trial} mode={mode} output={row['output_tokens_per_s']:.2f}tok/s TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms TPOT_P50={row['latency_ms']['TPOT']['p50']:.2f}ms preemptions={row['preemptions']}", flush=True)
    unchanged()
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), original_repository_unchanged=True)
    save()
    report(outdir, manifest, results)
    print("COMPLETE " + str(outdir / "report.md"), flush=True)


if __name__ == "__main__":
    main()
