"""Compare original prefill priority and the completion-reset preemption lock."""

import argparse
from collections import Counter
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
from profiling.run_fast_int8_metrics import audited_benchmark
from profiling.run_kv_metrics_comparison import parse_metrics


def measured_benchmark():
    code = audited_benchmark((ROOT / "benchmark_inference_metrics.py").read_text())

    def replace(old, new):
        nonlocal code
        assert code.count(old) == 1, old
        code = code.replace(old, new)

    replace("import argparse\n", "import argparse\nfrom collections import Counter\n")
    replace("    preemptions = 0\n", "    preemptions = 0\n"
            "    preemptions_by_seq = {}\n"
            "    decode_batch_sizes = []\n"
            "    lock_stats = {'decode_batches': 0, 'prefill_fallback_batches': 0, 'completion_unlocks': 0}\n"
            "    assert llm.scheduler.preemption_lock == args.preemption_lock\n"
            "    assert llm.scheduler.lock == 0\n")
    replace("        preemptions += 1\n", "        preemptions += 1\n"
            "        preemptions_by_seq[seq.seq_id] = preemptions_by_seq.get(seq.seq_id, 0) + 1\n")
    replace("        original_postprocess(seqs, token_ids, is_prefill)\n",
            "        was_locked = llm.scheduler.preemption_lock and llm.scheduler.lock == 1\n"
            "        original_postprocess(seqs, token_ids, is_prefill)\n"
            "        if was_locked and llm.scheduler.lock == 0:\n"
            "            lock_stats['completion_unlocks'] += 1\n")
    replace("        seqs, is_prefill = call_args\n", "        seqs, is_prefill = call_args\n"
            "        if not is_prefill:\n"
            "            decode_batch_sizes.append(len(seqs))\n"
            "        if llm.scheduler.preemption_lock and llm.scheduler.lock == 1:\n"
            "            lock_stats['prefill_fallback_batches' if is_prefill else 'decode_batches'] += 1\n")
    replace("    print('Preemptions: ' + str(preemptions))\n",
            "    request_counts = [preemptions_by_seq.get(seq_id, 0) for seq_id in token_times]\n"
            "    assert sum(request_counts) == preemptions\n"
            "    assert llm.scheduler.lock == 0 and not llm.scheduler.block_manager.used_block_ids\n"
            "    request_stats = dict(preempted_requests=sum(c > 0 for c in request_counts),\n"
            "                         repeatedly_preempted_requests=sum(c > 1 for c in request_counts),\n"
            "                         maximum_per_request=max(request_counts),\n"
            "                         histogram=dict(sorted(Counter(request_counts).items())))\n"
            "    batch_stats = dict(mean=statistics.mean(decode_batch_sizes),\n"
            "                       p50=percentile(decode_batch_sizes, 50),\n"
            "                       p95=percentile(decode_batch_sizes, 95),\n"
            "                       maximum=max(decode_batch_sizes), steps=len(decode_batch_sizes))\n"
            "    print('Preemption lock enabled: ' + str(llm.scheduler.preemption_lock))\n"
            "    print('Preemptions: ' + str(preemptions))\n"
            "    print('Per-request preemptions: ' + json.dumps(request_stats))\n"
            "    print('Lock statistics: ' + json.dumps(lock_stats))\n"
            "    print('Actual decode batch sizes: ' + json.dumps(batch_stats))\n")
    compile(code, "benchmark_inference_metrics.py", "exec")
    return code


def median_results(trials):
    result = {}
    for variant in dict.fromkeys(t["variant"] for t in trials):
        result[variant] = {}
        for policy in ("original", "lock"):
            rows = [t for t in trials if t["variant"] == variant and t["policy"] == policy]
            if not rows:
                continue
            value = aggregate(rows)[rows[0]["mode"]]
            for key in ("preempted_requests", "repeatedly_preempted_requests", "maximum_per_request"):
                value[key] = statistics.median(t["request_preemption_stats"][key] for t in rows)
            for key in ("decode_batch_stats", "lock_stats"):
                value[key] = {k: statistics.median(t[key][k] for t in rows) for k in rows[0][key]}
            result[variant][policy] = value
    return result


def render(outdir, manifest, results):
    lines = ["# 1024 请求：抢占后优先 decode 的 lock 实验", "",
             "lock 是调度器的全局 0/1 状态：每次抢占设为 1；任一请求完成时设为 0。",
             "开启 preemption_lock 后，lock=1 且 running 非空时跳过 prefill，优先 decode；",
             "running 为空时允许 prefill 恢复 KV。关闭开关保持原来的 prefill 优先策略。", ""]
    for variant, label in (("bf16", "BF16"), ("fast_int8_half", "快速双 scale INT8")):
        values = results["medians"][variant]
        lines += [f"## {label}（各策略三轮中位数）", "",
                  "| 指标 | 原策略 | lock 策略 | 变化 |", "| --- | ---: | ---: | ---: |"]

        def row(label, getter, digits=2):
            old, new = (getter(values[p]) for p in ("original", "lock"))
            delta = f"{(new / old - 1) * 100:+.2f}%" if old else "—"
            lines.append(f"| {label} | {old:.{digits}f} | {new:.{digits}f} | {delta} |")

        for key, label, digits in (("preemptions", "抢占事件次数", 0),
                                  ("preempted_requests", "至少被抢占一次的请求数", 0),
                                  ("repeatedly_preempted_requests", "被抢占多次的请求数", 0),
                                  ("maximum_per_request", "单请求最大抢占次数", 0),
                                  ("elapsed_s", "整批 generate 秒", 3),
                                  ("output_tokens_per_s", "输出 token/s", 2),
                                  ("requests_per_s", "请求/s", 2),
                                  ("input_tokens_per_s", "输入 token/s", 2),
                                  ("total_tokens_per_s", "总 token/s", 2)):
            row(label, lambda v, k=key: v[k], digits)
        for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
            for q in ("mean", "p50", "p95", "p99"):
                row(f"{metric} {q.upper()} ms", lambda v, k=metric, p=q: v["latency_ms"][k][p])
        for key, digits in (("blocks", 0), ("peak_allocated_gib", 2), ("peak_reserved_gib", 2)):
            row(key, lambda v, k=key: v[k], digits)
        for stage in ("prefill", "decode"):
            for key, digits in (("seconds", 3), ("tokens", 0), ("tokens_per_s", 2)):
                row(f"{stage} model-run {key}", lambda v, s=stage, k=key: v["stages"][s][k], digits)
        for key in ("mean", "p50", "p95", "maximum", "steps"):
            row(f"实测 decode batch {key}", lambda v, k=key: v["decode_batch_stats"][k])
        lines += [""]
    lines += ["## 每轮结果", "", "| 版本 | 策略 | 轮次 | 抢占次数 | 被抢占请求数 | 输出 token/s | TTFT P50 ms | TPOT P50 ms |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in results["trials"]:
        lines.append(f"| {t['variant']} | {t['policy']} | {t['trial']} | {t['preemptions']} | "
                     f"{t['request_preemption_stats']['preempted_requests']} | {t['output_tokens_per_s']:.2f} | "
                     f"{t['latency_ms']['TTFT']['p50']:.2f} | {t['latency_ms']['TPOT']['p50']:.2f} |")
    lines += ["", "## 配置与测量范围", "",
              "- RTX 3090 Ti / Qwen3-0.6B / GPU 0；权重均为 BF16，快速 INT8 仅量化 KV。",
              "- 快速 kernel 与先前 1024 请求实验相同：prefill BM=64、decode BM=16，BN=64、num_stages=1、num_warps=4；所有 INT8 prefill/decode 均走自写 kernel。",
              "- 同一版本原策略/lock 使用完全相同的推理源码，仅 preemption_lock 参数不同；各三个独立新进程，交替运行，全为本次测量。",
              "- 完整负载复用上次 workload_1024.json：每轮输入 580663、输出 583802 token；1024 请求同时提交，长度各 100–1024，seed=0、temperature=0.6、ignore_eos=True。",
              "- max_num_seqs=512、max_num_batched_tokens=16384、max_model_len=4096、page size=256、gpu_memory_utilization=0.9；decode CUDA Graph，prefill eager。",
              "- 普通和缓存前缀 prefill 预热与上次相同，预热后重置采样种子和显存峰值；计数只包含正式 generate。未锁 GPU 频率。",
              "- TTFT 包含排队，为批量提交到 CPU postprocess 首 token 回填；TPOT 为每请求首末 token 时间差除以后续 token 数；ITL 汇总所有相邻 token 间隔。",
              "- model-run 时间包含输入准备、模型、采样和同步；整模型比较包含 KV 容量和调度影响。显存为 PyTorch allocated/reserved 峰值。",
              "- 抢占事件可重复发生于同一请求；request_preemption_stats.histogram 记录每请求抢占次数分布（含 0 次）。",
              "- decode batch 直接记录实际 schedule 返回的 decode 请求数，每步等权。lock_stats.decode_batches 为模型执行时 lock=1 的 decode 批次数（包括触发锁的批次）。",
              "- 每个请求的输出长度和时间戳数量均检查；原实验快照未修改；此次仅改变共同的 config/scheduler 和 benchmark 计数，attention/model kernel 未修改。", "",
              "## 复现", "", "```bash",
              f"{PYTHON} profiling/run_preemption_lock_metrics.py --outdir /tmp/preemption-lock-recheck --runs 3", "```", ""]
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "profiling/fast_int8_vs_bf16_1024_2026-10-05")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    assert args.runs > 0
    outdir, reference = args.outdir.resolve(), args.reference.resolve()
    assert not (outdir / "manifest.json").exists(), "use a fresh measurement directory"
    outdir.mkdir(parents=True, exist_ok=True)
    prior = json.loads((reference / "manifest.json").read_text())
    assert prior["request_counts"] == [1024]
    reference_roots = {name: Path(path) for name, path in prior["variant_roots"].items()}
    reference_hashes = prior["variant_source_hashes"]
    assert all(source_hashes(root) == reference_hashes[name] for name, root in reference_roots.items())
    originals = source_hashes(ROOT)
    allowed = {"nanovllm/config.py", "nanovllm/engine/scheduler.py", "benchmark_inference_metrics.py"}
    assert {k for k in originals if originals[k] != prior["original_source_hashes"][k]} == allowed
    model = Path(prior["model"])
    assert json.loads((model / "config.json").read_text()) == prior["model_config"]
    for item in prior["weight_files"]:
        stat = (model / item["name"]).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (item["size"], item["mtime_ns"])
    code = measured_benchmark()
    variants = {}
    for name, source in reference_roots.items():
        target = outdir / "variants" / name
        for relative in reference_hashes[name]:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == "benchmark_inference_metrics.py":
                destination.write_text(code)
            else:
                shutil.copyfile((ROOT if relative in allowed else source) / relative, destination)
        variants[name] = target
    hashes = {name: source_hashes(root) for name, root in variants.items()}
    for name in hashes:
        assert {k for k in hashes[name] if hashes[name][k] != reference_hashes[name][k]} == allowed
    shutil.copyfile(reference / "workload_1024.json", outdir / "workload_1024.json")
    workload = prior["workloads"]["1024"]
    assert hashlib.sha256((outdir / "workload_1024.json").read_bytes()).hexdigest() == workload["sha256"]
    driver = outdir / "driver_snapshot/profiling"
    driver.mkdir(parents=True)
    for filename in (Path(__file__).name, "run_current_int8_metrics.py", "run_fast_int8_metrics.py", "run_kv_metrics_comparison.py"):
        shutil.copyfile(ROOT / "profiling" / filename, driver / filename)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), reference=str(reference),
                    runs_per_case=args.runs, gpu=args.gpu, python=PYTHON,
                    model=str(model), model_config=prior["model_config"], weight_files=prior["weight_files"],
                    config=prior["config"], warmup=prior["warmup"], kernel_config=prior["kernel_config"],
                    workload=workload, root_source_hashes=originals, reference_source_hashes=reference_hashes,
                    variant_source_hashes=hashes, variant_roots={k: str(v) for k, v in variants.items()},
                    lock_rules=dict(preempt=1, any_completion=0, locked="decode priority", empty_running="prefill fallback"),
                    nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"], text=True),
                    runs=[])
    results = dict(trials=[], medians={})

    def save():
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def unchanged():
        assert source_hashes(ROOT) == originals
        assert all(source_hashes(root) == hashes[name] for name, root in variants.items())
        assert all(source_hashes(root) == reference_hashes[name] for name, root in reference_roots.items())

    save()
    for trial in range(1, args.runs + 1):
        order = ("bf16", "fast_int8_half") if trial % 2 else ("fast_int8_half", "bf16")
        policies = ("original", "lock") if trial % 2 else ("lock", "original")
        for name in order:
            for policy in policies:
                unchanged()
                mode = "auto" if name == "bf16" else "int8_half"
                log = outdir / f"{name}_{policy}_{trial}.txt"
                command = [PYTHON, "benchmark_inference_metrics.py", "--model", str(model),
                           "--requests", "1024", "--kv-cache-dtype", mode]
                if policy == "lock":
                    command.append("--preemption-lock")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", PYTHONPATH=str(variants[name]))
                print(f"START trial={trial}/{args.runs} variant={name} policy={policy}", flush=True)
                started = perf_counter()
                with log.open("w") as fp:
                    completed = subprocess.run(command, cwd=variants[name], env=env, stdout=fp, stderr=subprocess.STDOUT, timeout=600)
                assert completed.returncode == 0, f"measurement failed: {log}"
                unchanged()
                text = log.read_text()
                assert f"Package: {variants[name]}/nanovllm/__init__.py" in text
                assert f"Preemption lock enabled: {policy == 'lock'}" in text
                row = parse_metrics(text)
                assert row["requests"] == 1024 and row["mode"] == mode and row["graph"]
                assert row["input_tokens"] == workload["input_tokens"] and row["output_tokens"] == workload["output_tokens"]
                def json_line(prefix):
                    return json.loads(re.search(r"^" + re.escape(prefix) + r": (.*)$", text, re.M)[1])
                dispatch = json_line("Prefill attention dispatch")
                assert (dispatch["int8"] == 0 and dispatch["flash"] > 0) if name == "bf16" else (dispatch["flash"] == 0 and dispatch["int8"] > 0)
                assert ("torch.bfloat16" if name == "bf16" else "torch.int8") in re.search(r"Actual KV dtype: (.*)", text)[1]
                request_stats = json_line("Per-request preemptions")
                count = int(re.search(r"Preemptions: (\d+)", text)[1])
                assert sum(request_stats["histogram"].values()) == 1024
                assert sum(int(k) * v for k, v in request_stats["histogram"].items()) == count
                row.update(variant=name, policy=policy, trial=trial, log=log.name, process_wall_s=perf_counter()-started,
                           preemptions=count, request_preemption_stats=request_stats,
                           lock_stats=json_line("Lock statistics"), decode_batch_stats=json_line("Actual decode batch sizes"),
                           prefill_batches=json_line("Prefill batches"), prefill_attention_dispatch=dispatch,
                           kv_tensor_bytes=int(re.search(r"KV tensor bytes: (\d+)", text)[1]),
                           scale_tensor_bytes=int(re.search(r"Scale tensor bytes: (\d+)", text)[1]),
                           output_sha256=re.search(r"Output token SHA256: ([a-f0-9]{64})", text)[1])
                results["trials"].append(row)
                results["medians"] = median_results(results["trials"])
                manifest["runs"].append(dict(trial=trial, variant=name, policy=policy, command=command, log=log.name))
                save()
                print(f"DONE trial={trial} variant={name} policy={policy} preemptions={count} "
                      f"output={row['output_tokens_per_s']:.2f}tok/s TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms "
                      f"TPOT_P50={row['latency_ms']['TPOT']['p50']:.2f}ms", flush=True)
    unchanged()
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), reference_unchanged=True,
                    sources_unchanged_during_measurement=True)
    save()
    render(outdir, manifest, results)
    print("COMPLETE " + str(outdir / "report.md"), flush=True)


if __name__ == "__main__":
    main()
