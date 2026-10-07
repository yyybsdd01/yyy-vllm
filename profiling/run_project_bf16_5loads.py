"""Fresh, resumable comparisons of a frozen checkout against BF16 at five loads."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
from profiling.run_current_int8_metrics import PYTHON, aggregate, source_hashes
from profiling.run_preemption_lock_metrics import measured_benchmark
from profiling.run_kv_offload_decode_reserve import instrument_progress
from profiling.run_kv_metrics_comparison import parse_metrics


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def field(output, name, convert=json.loads):
    found = re.search(r"^" + re.escape(name) + r": (.*)$", output, re.M)
    assert found, name
    return convert(found[1])


def benchmark_code():
    code = instrument_progress(measured_benchmark())
    old = "    input_tokens = sum(map(len, prompts))\n"
    assert code.count(old) == 1
    code = code.replace(old, "    workload_bytes = json.dumps(dict(prompts=prompts, max_tokens=[p.max_tokens for p in params]), separators=(',', ':')).encode()\n"
                        "    print('Workload SHA256: ' + hashlib.sha256(workload_bytes).hexdigest(), flush=True)\n" + old)
    old = "    def timed_postprocess(seqs, token_ids, is_prefill):\n"
    assert code.count(old) == 1
    code = code.replace(old, "    progress_time = perf_counter()\n    progress_tokens = 0\n    progress_finished = 0\n" + old +
                        "        nonlocal progress_time, progress_tokens, progress_finished\n")
    old = "        timestamp = perf_counter()\n"
    assert code.count(old) == 1
    code = code.replace(old, old +
                        "        progress_tokens += sum(seq.num_completion_tokens - old for seq, old in zip(seqs, previous))\n"
                        "        progress_finished += sum(seq.is_finished for seq in seqs)\n"
                        "        if timestamp - progress_time >= 30:\n"
                        "            print(f'BENCH PROGRESS tokens={progress_tokens}/{expected_output_tokens} finished={progress_finished}/{args.requests}', flush=True)\n"
                        "            progress_time = timestamp\n")
    compile(code, "benchmark_inference_metrics.py", "exec")
    return code


def prepare(args, out):
    out.mkdir(parents=True, exist_ok=True)
    checkout = out / "checkout"
    original = source_hashes(ROOT)
    for name in original:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    if args.kernel == "fast8":
        donor = ROOT / "profiling/kv_offload_decode_reserve_2026-10-06/variants/reserved"
        shutil.copyfile(donor / "nanovllm/layers/quantized_attention.py", checkout / "nanovllm/layers/quantized_attention.py")
        runner = checkout / "nanovllm/engine/model_runner.py"
        code = runner.read_text()
        old = "        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache\n"
        assert code.count(old) == 1
        code = code.replace(old, "        if (cu_seqlens_k[-1] > cu_seqlens_q[-1]\n"
                                "                or (self.config.kv_cache_dtype == 'int8_half'\n"
                                "                    and all(seq.block_table for seq in seqs))):\n")
        runner.write_text(code)
    inference_hashes = source_hashes(checkout)
    (checkout / "benchmark_inference_metrics.py").write_text(benchmark_code())
    shutil.copyfile(ROOT / "profiling/kv_cache_perplexity.py", checkout / "kv_cache_perplexity.py")
    shutil.copyfile(ROOT / "profiling/eval_project_kv_quality.py", checkout / "eval_project_kv_quality.py")
    workloads = {}
    for count in args.requests:
        rng = random.Random(0)
        prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, 1024))] for _ in range(count)]
        outputs = [rng.randint(100, 1024) for _ in range(count)]
        raw = json.dumps(dict(prompts=prompts, max_tokens=outputs), separators=(",", ":")).encode()
        filename = f"workload_{count}.json"
        (out / filename).write_bytes(raw)
        workloads[str(count)] = dict(file=filename, sha256=hashlib.sha256(raw).hexdigest(),
                                     requests=count, input_tokens=sum(map(len, prompts)), output_tokens=sum(outputs))
    model = Path(args.model).resolve()
    drivers = out / "driver_snapshot" / "profiling"
    drivers.mkdir(parents=True)
    for filename in (Path(__file__).name, "eval_project_kv_quality.py", "run_current_int8_metrics.py",
                     "run_preemption_lock_metrics.py", "run_fast_int8_metrics.py",
                     "run_kv_offload_decode_reserve.py", "run_kv_metrics_comparison.py",
                     "check_kv_offload_model.py"):
        shutil.copyfile(ROOT / "profiling" / filename, drivers / filename)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), runs=args.runs,
                    request_counts=args.requests, gpu=args.gpu, model=str(model), python=PYTHON,
                    kernel=args.kernel, root_source_hashes=original, inference_source_hashes=inference_hashes,
                    measured_source_hashes=source_hashes(checkout), checkout=str(checkout), workloads=workloads,
                    model_config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
                    weight_files=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                  for p in sorted(model.glob("*.safetensors"))],
                    config=dict(max_num_seqs=512, max_num_batched_tokens=16384, max_model_len=4096,
                                block_size=256, gpu_memory_utilization=0.9, preemption_lock=False,
                                offload_cpu_gb=4, offload_max_inflight=8, cuda_graph=True,
                                seed=0, min_input=100, max_input=1024, min_output=100, max_output=1024),
                    variants=dict(bf16=dict(mode="auto", offload=False),
                                  project=dict(mode="int8_half", offload=True)),
                    commands=[], packages={name: importlib.metadata.version(name) for name in
                                           ("torch", "triton", "flash-attn", "transformers")})
    save(out / "manifest.json", manifest)
    save(out / "results.json", dict(trials=[], medians={}))
    (out / "git_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    return manifest


def verify(manifest):
    checkout = Path(manifest["checkout"])
    assert source_hashes(checkout) == manifest["measured_source_hashes"], "snapshot changed"
    model = Path(manifest["model"])
    assert hashlib.sha256((model / "config.json").read_bytes()).hexdigest() == manifest["model_config_sha256"]
    for item in manifest["weight_files"]:
        stat = (model / item["name"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (item["size"], item["mtime_ns"])


def run_logged(command, logfile, env, cwd, timeout=3600):
    print("RUN " + " ".join(command), flush=True)
    with logfile.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            process.terminate()
            process.wait(timeout=60)
            raise
    assert status == 0, f"exit {status}; see {logfile}"
    return logfile.read_text()


def summarize(trials):
    medians = {}
    for count in dict.fromkeys(t["requests"] for t in trials):
        rows = [t for t in trials if t["requests"] == count]
        medians[str(count)] = aggregate(rows)
        for mode, value in medians[str(count)].items():
            group = [t for t in rows if t["mode"] == mode]
            for key in ("request_preemption_stats", "decode_batch_stats", "restore_progress", "offload_stats"):
                value[key] = {k: statistics.median(t[key][k] for t in group)
                              for k, v in group[0][key].items() if isinstance(v, (int, float))}
            value["output_tokens_per_s_min"] = min(t["output_tokens_per_s"] for t in group)
            value["output_tokens_per_s_max"] = max(t["output_tokens_per_s"] for t in group)
    return medians


def metric_rows(values):
    for label, key in (("输出 token/s", "output_tokens_per_s"), ("请求/s", "requests_per_s"),
                       ("输入 token/s", "input_tokens_per_s"), ("总 token/s", "total_tokens_per_s"),
                       ("整批耗时 s", "elapsed_s"), ("KV blocks", "blocks"),
                       ("峰值 allocated GiB", "peak_allocated_gib"), ("峰值 reserved GiB", "peak_reserved_gib"),
                       ("KV tensor bytes", "kv_tensor_bytes"), ("scale tensor bytes", "scale_tensor_bytes"),
                       ("抢占事件", "preemptions")):
        yield label, [v[key] for v in values]
    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for quantile in ("mean", "p50", "p95", "p99"):
            yield f"{metric} {quantile.upper()} ms", [v["latency_ms"][metric][quantile] for v in values]
    for stage in ("prefill", "decode"):
        for key in ("tokens", "seconds", "tokens_per_s"):
            yield f"{stage} model-run {key}", [v["stages"][stage][key] for v in values]
    for key in ("request_preemption_stats", "decode_batch_stats", "restore_progress"):
        for name in values[0][key]:
            yield f"{key}.{name}", [v[key][name] for v in values]


def render(out, manifest, results):
    lines = ["# 当前项目与 BF16：五档请求负载", "",
             f"快照：{manifest['started_utc']}；内核选择：{manifest['kernel']}。各档两组独立新进程交替运行，各三轮，表内为逐项中位数。",
             "当前项目启用双 scale INT8 KV、异步 CPU 卸载及 decode 块预留；BF16 使用 FlashAttention、关闭卸载。模型权重均为 BF16。", "",
             "## 汇总", "", "| 请求数 | 输出 token/s BF16 → 项目 | 变化 | TTFT P50 ms BF16 → 项目 | TPOT P50 ms BF16 → 项目 | 抢占 BF16 → 项目 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for count in manifest["request_counts"]:
        group = results["medians"].get(str(count), {})
        if set(group) != {"auto", "int8_half"}:
            continue
        a, b = group["auto"], group["int8_half"]
        lines.append(f"| {count} | {a['output_tokens_per_s']:.2f} → {b['output_tokens_per_s']:.2f} | {(b['output_tokens_per_s']/a['output_tokens_per_s']-1)*100:+.2f}% | "
                     f"{a['latency_ms']['TTFT']['p50']:.2f} → {b['latency_ms']['TTFT']['p50']:.2f} | "
                     f"{a['latency_ms']['TPOT']['p50']:.2f} → {b['latency_ms']['TPOT']['p50']:.2f} | {a['preemptions']:.0f} → {b['preemptions']:.0f} |")
    csv_rows = []
    for count in manifest["request_counts"]:
        group = results["medians"].get(str(count), {})
        if set(group) != {"auto", "int8_half"}:
            continue
        a, b = group["auto"], group["int8_half"]
        workload = manifest["workloads"][str(count)]
        lines += ["", f"## {count} 请求", "",
                  f"输入 {workload['input_tokens']}、输出 {workload['output_tokens']} token；已完成 {a['runs']} / {b['runs']} 轮。",
                  "", "| 指标 | BF16 | 当前项目 | 相对变化 |", "| --- | ---: | ---: | ---: |"]
        for label, (old, new) in metric_rows([a, b]):
            delta = (new / old - 1) * 100 if old else None
            change = f"{delta:+.2f}%" if delta is not None else "—"
            lines.append(f"| {label} | {old:.3f} | {new:.3f} | {change} |")
            csv_rows.append(dict(requests=count, metric=label, bf16=old, project=new, change_percent=delta))
        lines += ["", "项目卸载统计（三轮中位数）：", "", "| 指标 | 值 |", "| --- | ---: |"]
        for key, value in b["offload_stats"].items():
            lines.append(f"| {key} | {value:.3f} |")
    lines += ["", "## PPL 与传输正确性", ""]
    quality = results.get("quality", {})
    if set(quality) == {"auto", "int8_half"}:
        a, b = quality["auto"], quality["int8_half"]
        lines += [f"WikiText-2 raw test，共 {a['tokens']} 个相同目标位置；4096 非重叠窗口、256 token 分块，全词表交叉熵。",
                  f"BF16 PPL **{a['ppl']:.6f}**，项目 PPL **{b['ppl']:.6f}**，变化 **{(b['ppl']/a['ppl']-1)*100:+.4f}%**。",
                  "PPL 测的是对应 attention 路径的分块 prefill teacher forcing，不随合成请求数变化；不覆盖生成采样或卸载调度的端到端质量。",
                  "卸载另在 12 块 GPU KV 的压力场景中逐块比较恢复前后有效 KV/scale，完整结果见 validation 文件。"]
    else:
        lines += ["质量评测尚未完成。"]
    lines += ["", "## 口径与复现", "",
              "- GPU0 RTX 3090 Ti，Qwen3-0.6B，max_num_seqs=512、max_num_batched_tokens=16384、max_model_len=4096、page=256、gpu_memory_utilization=0.9。",
              "- 256/512/1024/2048/4096 为一次同时提交的请求数；实际 decode batch 不超过 512。各档输入和输出各 100–1024 token，seed=0、temperature=0.6、ignore_eos=True，两组逐请求长度完全相同。",
              "- 根目录推理源码保持原样；独立快照测量。current 使用根目录 BM16/BN64、4 warps 的 attention，首次无前缀 prefill 走 BF16 FlashAttention；fast8 是显式选择的旧实验内核，所有 INT8 prefill 走量化路径。此次 manifest.kernel 决定实际版本。",
              "- BF16 和项目使用相同显存预算，KV 块数按本版本原生容量分配，包含 INT8 容量优势和调度影响。CPU pinned 池限 4 GiB，最多 8 个 H2D 恢复请求。",
              "- 每轮独立进程；模型加载、CUDA Graph 捕获、代表性普通/缓存 prefill 编译预热在计时外。预热请求 ID 与正式负载不同；正式采样前重置随机种子和显存峰值。CPU 池首次分配计入正式耗时。",
              "- TTFT 从整批提交到 CPU postprocess 回填首 token，含排队；TPOT=(末 token 时间-首 token 时间)/(输出数-1)；ITL 汇总相邻 token 间隔；端到端为每请求完成时间减整批提交时间。",
              "- model-run 时间包含输入准备、模型、采样和同步；输入吞吐按原始输入数计算，prefill tokens 包含抢占后的重算。显存为 PyTorch allocated/reserved 峰值，KV tensor bytes 仅存储张量。",
              "- 抢占是事件次数，同一请求可多次被抢占；恢复后无新 token 再抢占单独计数。随机采样随 batch/调度变化，输出 hash 仅溯源，不要求不同路径生成 token 相同。",
              "- 未锁 GPU 频率；仅 GPU0 串行测量，记录每轮 GPU 状态。每轮检查实际 import 路径、KV dtype、工作负载 hash、每请求输出长度/时间戳数量及 GPU/CPU KV 清理。",
              "", "```bash", f"{PYTHON} profiling/run_project_bf16_5loads.py --outdir {out} --requests 256 512 1024 2048 4096 --runs 3 --kernel {manifest['kernel']}", "```", ""]
    (out / "report.md").write_text("\n".join(lines))
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["requests", "metric", "bf16", "project", "change_percent"])
        writer.writeheader()
        writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--requests", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--kernel", choices=["current", "fast8"], default="current")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.outdir.resolve()
    assert args.runs > 0 and all(n > 0 for n in args.requests)
    manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else prepare(args, out)
    assert (manifest["request_counts"], manifest["runs"], manifest["kernel"], manifest["gpu"]) == (args.requests, args.runs, args.kernel, args.gpu)
    verify(manifest)
    if args.prepare_only:
        print("PREPARED", out, flush=True)
        return
    results = json.loads((out / "results.json").read_text())
    checkout = Path(manifest["checkout"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["gpu"], PYTHONPATH=str(checkout), PYTHONUNBUFFERED="1")
    validation = out / "validation"
    validation.mkdir(exist_ok=True)
    for mode in ("auto", "int8_half"):
        target = validation / f"restored_kv_{mode}.json"
        if not target.exists():
            command = [PYTHON, str(out / "driver_snapshot/profiling/check_kv_offload_model.py"), "--variant", str(checkout),
                       "--offload", "--verify-kv", "--mode", mode, "--output", str(target)]
            run_logged(command, target.with_suffix(".txt"), env, checkout)
    quality = out / "quality"
    quality.mkdir(exist_ok=True)
    for mode in ("auto", "int8_half"):
        target = quality / f"{mode}.json"
        if not target.exists():
            command = [PYTHON, str(checkout / "eval_project_kv_quality.py"), "--model", manifest["model"],
                       "--text", "/home/xgd/.cache/nanovllm_eval/wikitext2_test.txt", "--mode", mode,
                       "--output", str(target)]
            if manifest["kernel"] == "fast8":
                command += ["--quantize-first"]
            run_logged(command, target.with_suffix(".txt"), env, checkout)
        results.setdefault("quality", {})[mode] = json.loads(target.read_text())
        save(out / "results.json", results)
    a, b = (results["quality"][mode] for mode in ("auto", "int8_half"))
    assert all(a[k] == b[k] for k in ("tokens", "text_sha256", "token_ids_sha256", "window_size", "chunk_size", "block_size"))
    for count in manifest["request_counts"]:
        for trial in range(1, manifest["runs"] + 1):
            order = ["bf16", "project"] if trial % 2 else ["project", "bf16"]
            for variant in order:
                if any((r["requests"], r["trial"], r["variant"]) == (count, trial, variant) for r in results["trials"]):
                    continue
                verify(manifest)
                settings = manifest["variants"][variant]
                command = [PYTHON, str(checkout / "benchmark_inference_metrics.py"), "--model", manifest["model"],
                           "--requests", str(count), "--kv-cache-dtype", settings["mode"]]
                if settings["offload"]:
                    command += ["--kv-cpu-offload"]
                log = out / f"{count}_{variant}_{trial}.txt"
                gpu_before = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"], text=True)
                started = perf_counter()
                output = run_logged(command, log, env, checkout)
                row = parse_metrics(output)
                row.update(variant=variant, trial=trial, raw_log=log.name, process_wall_s=perf_counter() - started,
                           gpu_before=gpu_before, preemptions=field(output, "Preemptions", int),
                           prefill_batches=field(output, "Prefill batches"),
                           kv_tensor_bytes=field(output, "KV tensor bytes", int),
                           scale_tensor_bytes=field(output, "Scale tensor bytes", int),
                           package=field(output, "Package", str), actual_kv_dtype=field(output, "Actual KV dtype", str),
                           output_sha256=field(output, "Output token SHA256", str),
                           workload_sha256=field(output, "Workload SHA256", str),
                           attention_dispatch=field(output, "Prefill attention dispatch"),
                           request_preemption_stats=field(output, "Per-request preemptions"),
                           decode_batch_stats=field(output, "Actual decode batch sizes"),
                           restore_progress=field(output, "Restore progress statistics"),
                           offload_stats=field(output, "KV offload statistics") if settings["offload"] else {})
                workload = manifest["workloads"][str(count)]
                assert row["workload_sha256"] == workload["sha256"]
                assert (row["input_tokens"], row["output_tokens"]) == (workload["input_tokens"], workload["output_tokens"])
                assert Path(row["package"]).resolve().is_relative_to(checkout)
                assert row["actual_kv_dtype"] == ("torch.int8" if settings["offload"] else "torch.bfloat16")
                assert row["decode_batch_stats"]["maximum"] <= 512
                if not settings["offload"]:
                    assert row["attention_dispatch"]["int8"] == 0
                manifest["commands"].append(dict(requests=count, trial=trial, variant=variant, command=command))
                results["trials"].append(row)
                results["medians"] = summarize(results["trials"])
                save(out / "manifest.json", manifest)
                save(out / "results.json", results)
                render(out, manifest, results)
                print(f"COMPLETE {count} {variant} {trial} tps={row['output_tokens_per_s']:.2f} TTFTp50={row['latency_ms']['TTFT']['p50']:.2f} TPOTp50={row['latency_ms']['TPOT']['p50']:.2f} preempt={row['preemptions']} ({len(results['trials'])}/{len(manifest['request_counts'])*manifest['runs']*2})", flush=True)
    verify(manifest)
    assert len(results["trials"]) == len(manifest["request_counts"]) * manifest["runs"] * 2
    results.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                   root_inference_unchanged=source_hashes(ROOT) == manifest["root_source_hashes"])
    save(out / "results.json", results)
    render(out, manifest, results)
    print("ALL COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
