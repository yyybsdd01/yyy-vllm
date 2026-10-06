"""Add a fresh, fully unquantized BF16 baseline to the completed prefill experiment."""
import argparse
import csv
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
from profiling.run_scale_layout_metrics import PYTHON, aggregate, hashes
from profiling.run_kv_metrics_comparison import parse_metrics


def render(outdir, manifest, results):
    medians = results["medians"]
    names = ("bf16", "int8_flash_prefill", "int8_token_head", "int8_head_token")
    model = manifest["model_config"]
    dim = model.get("head_dim", model["hidden_size"]//model["num_attention_heads"])
    elements_per_block = 2 * model["num_hidden_layers"] * 256 * model["num_key_value_heads"] * dim
    for name, values in medians.items():
        block_bytes = elements_per_block * (2 if name == "bf16" else 1)
        if name != "bf16":
            block_bytes += 2 * model["num_hidden_layers"] * 256 * model["num_key_value_heads"] * 2 * 4
        values["kv_storage_gib"] = block_bytes * values["blocks"] / 1024**3
    lines = ["# 未量化 BF16 与 INT8 双 scale：完整推理指标", "",
        "BF16 为未量化 KV cache，prefill/decode 都使用项目原 FlashAttention 路径，模型权重为 BF16。"
        "三种 INT8 配置的模型权重也为 BF16，仅量化 KV cache；普通 prefill 分别使用原始 BF16 K/V 的 FlashAttention，或自写 INT8 paged kernel。", "",
        "## 各项指标", "",
        "| 指标 | 未量化 BF16 | INT8 + FA prefill | 全自写 INT8 原 scale | 全自写 INT8 head/token |",
        "| --- | ---: | ---: | ---: | ---: |"]
    csv_rows = []
    def add(title, getter, unit, digits=2):
        values = {name: getter(medians[name]) for name in names}
        lines.append("| "+title+" | "+" | ".join(f"{values[name]:.{digits}f}" for name in names)+" |")
        csv_rows.append(dict(metric=title, unit=unit, **values))
    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for q in ("mean", "p50", "p95", "p99"):
            add(f"{metric} {q.upper()} ms", lambda m, a=metric, b=q: m["latency_ms"][a][b], "ms")
    for title, key, unit, digits in (
        ("generate 时间 s", "elapsed_s", "s", 3),
        ("请求/s", "requests_per_s", "request/s", 2),
        ("输入 token/s", "input_tokens_per_s", "token/s", 2),
        ("输出 token/s", "output_tokens_per_s", "token/s", 2),
        ("输入+输出 token/s", "total_tokens_per_s", "token/s", 2),
        ("KV blocks", "blocks", "block", 0),
        ("KV 张量及 scales GiB", "kv_storage_gib", "GiB", 4),
        ("峰值 allocated GiB", "peak_allocated_gib", "GiB", 2),
        ("峰值 reserved GiB", "peak_reserved_gib", "GiB", 2),
    ):
        add(title, lambda m, k=key: m[k], unit, digits)
    for stage in ("prefill", "decode"):
        for key, unit, digits in (("seconds", "s", 3), ("tokens_per_s", "token/s", 2), ("tokens", "token", 0)):
            add(f"{stage} model-run {key}", lambda m, a=stage, b=key: m["stages"][a][b], unit, digits)
    lines += ["", "## 相对未量化 BF16", "",
        "| 配置 | TTFT P50 | TPOT P50 | ITL P50 | 输出吞吐 | generate 时间 | KV 容量 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    baseline = medians["bf16"]
    delta = lambda a, b: (b/a-1)*100
    for name in names[1:]:
        m = medians[name]
        values = [delta(baseline["latency_ms"][key]["p50"], m["latency_ms"][key]["p50"])
                  for key in ("TTFT", "TPOT", "ITL")]
        values += [delta(baseline[key], m[key]) for key in ("output_tokens_per_s", "elapsed_s", "blocks")]
        lines.append("| "+name+" | "+" | ".join(f"{v:+.2f}%" for v in values)+" |")
    extra = baseline["stages"]["prefill"]["tokens"] - manifest["input_tokens"]
    lines += ["", "## 测量条件和解释", "",
        "- Qwen3-0.6B / RTX 3090 Ti / GPU 0，单卡，CUDA Graph decode，prefill eager。",
        "- 256 请求同时到达，输入/输出长度均为 100–1024。负载 seed=0，采样 torch seed=0、temperature=0.6、ignore_eos=True；每轮输入 142,827，输出 133,966 token，逐请求长度验证通过。",
        "- BF16 本次重新跑三个独立进程；INT8 三轮中位数引用紧邻本次的上一轮完整测量，源码/模型文件/负载/预热配置均校验一致。不是四组交错的新一轮测量；未锁 GPU 频率，微小差异不能据此认定稳定收益。",
        "- BF16 使用上一轮 flash/token_head 的完整源码副本，仅增加 BF16 标签、实际 KV dtype/字节数打印、CPU preemption 计数观察；推理实现未修改。所有副本在计时前预热页表宽度 1/2/3/4及16K-token prefill批量，之后重置采样种子。",
        "- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、block_size=256、gpu_memory_utilization=0.9；模型权重始终 BF16。",
        f'- BF16 KV 容量 {baseline["blocks"]:.0f} 块，INT8 {medians["int8_token_head"]["blocks"]:.0f} 块。BF16 正式测量抢占次数为 {baseline["preemptions"]:.0f} 次；prefill 执行 {baseline["stages"]["prefill"]["tokens"]:.0f} token，较输入额外 {extra:.0f} token。Scheduler.preempt 释放缓存并重新入队，后续 prefill 包含重算。',
        "- 因而整模型阶段时间包含 KV 容量/抢占/重算效应，不能直接解读为单个 INT8 kernel 更快或更慢。INT8 上一轮的 prefill 执行量等于输入量，decode 执行量为输出量减去每请求首 token。",
        "- 按相同显存预算分配缓存，所以 INT8 总峰值显存不会自动减半。每个 token 的 K/V+双scale占 BF16 的 53.125%，节省的空间用于增大可分配 KV 容量；实际完整缓存池大小仍相近。",
        "- TTFT 是整批请求提交至 CPU postprocess 首 token 完成，包含排队。TPOT 是每请求平均相邻 token 时间，ITL 汇总所有相邻时间戳；阶段时间包括输入准备、模型、采样和同步。P95/P99 是每轮请求/token 分布分位数，再对三轮取中位数。",
        "- 不同量化/调度路径的生成内容可不同，输出 token 数固定。此前自写 INT8 kernel 对恢复后 BF16 KV 的正确性检查已通过；这里没有评估语言质量。",
        "- 原仓库 nanovllm 源码及 benchmark_inference_metrics.py 哈希前后一致。INT8 参考结果、配置和日志在 reference_int8/。", "",
        "## BF16 每轮结果", "",
        "| 轮次 | generate s | 输出 token/s | TTFT P50 ms | ITL P50 ms | prefill s | prefill tokens | 抢占次数 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results["bf16_trials"]:
        lines.append(f'| {row["trial"]} | {row["elapsed_s"]:.3f} | {row["output_tokens_per_s"]:.2f} | '
            f'{row["latency_ms"]["TTFT"]["p50"]:.2f} | {row["latency_ms"]["ITL"]["p50"]:.2f} | '
            f'{row["stages"]["prefill"]["seconds"]:.3f} | {row["stages"]["prefill"]["tokens"]} | {row["preemptions"]} |')
    lines += ["", "## 复现", "", "```bash",
        "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_bf16_prefill_comparison.py --outdir /tmp/bf16_prefill_comparison",
        "```", "", "完整模型配置、源码哈希、原始日志、results.json、comparison.csv 和独立 BF16 源码副本均留在该目录。"]
    (outdir/"report.md").write_text("\n".join(lines)+"\n")
    with (outdir/"comparison.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["metric", "unit", *names])
        writer.writeheader()
        writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT/"profiling/prefill_kernel_inference_warm_2026-10-04")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    outdir, reference = args.outdir.resolve(), args.reference.resolve()
    assert args.runs > 0 and not (outdir/"manifest.json").exists(), "use a fresh directory"
    prior = json.loads((reference/"manifest.json").read_text())
    data = json.loads((reference/"results.json").read_text())
    assert "finished_utc" in prior and prior["config"]["prefill_warmup_lengths"] == [256,512,768,1024]
    originals = hashes(ROOT)
    assert originals == prior["original_source_hashes"], "current source differs from the INT8 reference"
    model = prior["model"]
    assert json.loads((Path(model)/"config.json").read_text()) == prior["model_config"]
    for weight in prior["weight_files"]:
        stat = (Path(model)/weight["name"]).stat()
        assert stat.st_size == weight["size"] and stat.st_mtime_ns == weight["mtime_ns"]
    outdir.mkdir(parents=True)
    reference_copy = outdir/"reference_int8"
    reference_copy.mkdir()
    for name in ("manifest.json", "results.json", "workload.json", "report.md"):
        shutil.copyfile(reference/name, reference_copy/name)
    for row in data["trials"]:
        shutil.copyfile(reference/row["log"], reference_copy/row["log"])
    source_root = Path(prior["variant_roots"]["flash_token_head"])
    assert hashes(source_root) == prior["variant_source_hashes"]["flash_token_head"]
    variant = outdir/"variants"/"bf16"
    for relative in originals:
        target = variant/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root/relative, target)
    file = variant/"benchmark_inference_metrics.py"
    source = file.read_text().replace("SCALE_LAYOUT: token_head", "CACHE_FORMAT: bf16")
    source = source.replace("PREFILL_EXPERIMENT: flash_token_head", "PREFILL_EXPERIMENT: bf16")
    old = "    token_times: dict[int, list[float]] = {}\n"
    assert source.count(old) == 1
    source = source.replace(old,
        "    preemption_count = 0\n"
        "    original_preempt = llm.scheduler.preempt\n"
        "    def counted_preempt(seq):\n"
        "        nonlocal preemption_count\n"
        "        preemption_count += 1\n"
        "        return original_preempt(seq)\n"
        "    llm.scheduler.preempt = counted_preempt\n"+old)
    old = "    print('Output token SHA256: ' + output_digest)\n"
    source = source.replace(old, old+
        "    print('Preemptions: ' + str(preemption_count))\n"
        "    print('Actual KV dtype: ' + str(llm.model_runner.kv_cache.dtype))\n"
        "    print('KV cache tensor bytes: ' + str(llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size()))\n")
    file.write_text(source)
    variant_hashes = hashes(variant)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1", PYTHONPATH=str(variant))
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), reference=str(reference),
        reference_manifest_sha256=hashlib.sha256((reference/"manifest.json").read_bytes()).hexdigest(),
        model=model, model_config=prior["model_config"], config={**prior["config"], "kv_cache_dtype":"auto"},
        input_tokens=prior["input_tokens"], output_tokens=prior["output_tokens"],
        workload_sha256=prior["workload_sha256"], python=PYTHON, gpu=args.gpu, runs=args.runs,
        original_source_hashes=originals, variant_source_hashes=variant_hashes, variant_root=str(variant),
        nvidia_smi=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem","--format=csv,noheader"],text=True), runs_completed=[])
    snapshot = outdir/"driver_snapshot"/"profiling"
    snapshot.mkdir(parents=True)
    for name in (Path(__file__).name, "run_scale_layout_metrics.py", "run_kv_metrics_comparison.py"):
        shutil.copyfile(ROOT/"profiling"/name, snapshot/name)
    def save():
        (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save()
    trials = []
    for trial in range(1,args.runs+1):
        assert hashes(ROOT) == originals and hashes(variant) == variant_hashes
        command = [PYTHON, "benchmark_inference_metrics.py", "--model", model, "--kv-cache-dtype", "auto"]
        log = outdir/f"bf16_{trial}.txt"
        print("START", log, flush=True)
        started = perf_counter()
        with log.open("w") as output:
            completed = subprocess.run(command, cwd=variant, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, timeout=600)
        assert completed.returncode == 0, f"BF16 run failed: {log}"
        text = log.read_text()
        assert f"CACHE_FORMAT: bf16 | package: {variant}/nanovllm/__init__.py" in text
        assert "Actual KV dtype: torch.bfloat16" in text
        row = parse_metrics(text)
        assert row["mode"] == "auto" and row["graph"]
        assert row["input_tokens"] == prior["input_tokens"] and row["output_tokens"] == prior["output_tokens"]
        audit = json.loads(re.search(r"Prefill dispatch audit: (.*)",text)[1])
        assert audit["int8"] == 0 and audit["flash"] > 0
        row.update(trial=trial, layout="token_head", variant="bf16", log=log.name,
            output_sha256=re.search(r"Output token SHA256: ([a-f0-9]{64})",text)[1],
            preemptions=int(re.search(r"Preemptions: (\d+)",text)[1]),
            kv_tensor_bytes=int(re.search(r"KV cache tensor bytes: (\d+)",text)[1]),
            prefill_dispatch=audit, process_wall_s=perf_counter()-started)
        trials.append(row)
        manifest["runs_completed"].append(dict(trial=trial, command=command, cwd=str(variant),
                                               log=log.name, process_wall_s=row["process_wall_s"]))
        save()
        medians = {"bf16": aggregate(trials)["token_head"],
                   "int8_flash_prefill": data["medians"]["flash_token_head"],
                   "int8_token_head": data["medians"]["int8_token_head"],
                   "int8_head_token": data["medians"]["int8_head_token"]}
        medians["bf16"]["preemptions"] = statistics.median(r["preemptions"] for r in trials)
        (outdir/"results.json").write_text(json.dumps(dict(bf16_trials=trials,medians=medians),indent=2)+"\n")
        assert hashes(ROOT) == originals and hashes(variant) == variant_hashes
        print(f"DONE BF16 {trial} output={row['output_tokens_per_s']:.2f}tok/s "
              f"TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms blocks={row['blocks']} "
              f"prefill={row['stages']['prefill']} preemptions={row['preemptions']}",flush=True)
    assert len({r["blocks"] for r in trials}) == 1 and len({r["output_sha256"] for r in trials}) == 1
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),original_repository_unchanged=True,
                    bf16_outputs_identical_across_trials=True)
    save()
    result = dict(bf16_trials=trials,medians=medians)
    render(outdir, manifest, result)
    (outdir/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print("COMPLETE",outdir/"report.md",flush=True)


if __name__ == "__main__":
    main()
