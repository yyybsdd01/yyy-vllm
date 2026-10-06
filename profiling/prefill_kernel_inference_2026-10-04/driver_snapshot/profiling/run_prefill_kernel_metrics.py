"""Compare FlashAttention/all-INT8 prefill with both scale layouts in isolated copies."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_scale_layout_metrics import (
    PYTHON, LAYOUTS, aggregate, hashes, prepare_copies,
)
from profiling.run_kv_metrics_comparison import parse_metrics

MODES = ("flash", "int8")
VARIANTS = tuple(f"{mode}_{layout}" for mode in MODES for layout in LAYOUTS)


def prepare_variants(outdir, originals):
    variants = {}
    for mode in MODES:
        for layout, root in prepare_copies(outdir/mode, originals).items():
            key = f"{mode}_{layout}"
            if mode == "int8":
                file = root/"nanovllm/engine/model_runner.py"
                source = file.read_text()
                old = "        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache\n"
                assert source.count(old) == 1
                source = source.replace(old,
                    "        # All allocated INT8 prefills need page tables, including zero-prefix batches.\n"
                    "        # During initialization there is no allocated cache or block table.\n"
                    "        if (cu_seqlens_k[-1] > cu_seqlens_q[-1]\n"
                    "                or (self.config.kv_cache_dtype == 'int8_half'\n"
                    "                    and all(seq.block_table for seq in seqs))):\n")
                file.write_text(source)
                file = root/"nanovllm/layers/attention.py"
                source = file.read_text()
                old = "        if quantized and (not context.is_prefill or context.block_tables is not None):\n"
                assert source.count(old) == 1
                source = source.replace(old, "        if quantized:\n")
                file.write_text(source)
            file = root/"benchmark_inference_metrics.py"
            source = file.read_text()
            source = source.replace("import nanovllm\n", "import nanovllm\nimport nanovllm.layers.attention as attention_module\n")
            old = "    token_times: dict[int, list[float]] = {}\n"
            assert source.count(old) == 1
            source = source.replace(old,
                f"    print('PREFILL_EXPERIMENT: {key}', flush=True)\n"
                "    prefill_dispatch = {'flash': 0, 'int8': 0}\n"
                "    prefill_shapes = []\n"
                "    original_int8_attention = attention_module.int8_paged_attention\n"
                "    original_flash_attention = attention_module.flash_attn_varlen_func\n"
                "    def audited_int8_attention(*call_args, **kwargs):\n"
                "        if attention_module.get_context().is_prefill:\n"
                "            prefill_dispatch['int8'] += 1\n"
                "        return original_int8_attention(*call_args, **kwargs)\n"
                "    def audited_flash_attention(*call_args, **kwargs):\n"
                "        prefill_dispatch['flash'] += 1\n"
                "        return original_flash_attention(*call_args, **kwargs)\n"
                "    attention_module.int8_paged_attention = audited_int8_attention\n"
                "    attention_module.flash_attn_varlen_func = audited_flash_attention\n"
                + old)
            old = "        count = sum(seq.num_scheduled_tokens for seq in seqs)\n"
            assert source.count(old) == 1
            source = source.replace(old, old+
                "        if is_prefill:\n"
                "            prefill_shapes.append([[seq.num_cached_tokens, seq.num_scheduled_tokens]\n"
                "                                   for seq in seqs])\n")
            source = source.replace("    print('Output token SHA256: ' + output_digest)\n",
                "    print('Output token SHA256: ' + output_digest)\n"
                "    print('Prefill dispatch audit: ' + json.dumps(prefill_dispatch))\n"
                "    print('Prefill shape SHA256: ' + hashlib.sha256(json.dumps(prefill_shapes).encode()).hexdigest())\n"
                "    print('Prefill batches: ' + str(len(prefill_shapes)))\n")
            file.write_text(source)
            variants[key] = root
    return variants


def summarize(rows):
    return {f"{mode}_{layout}": values
            for mode in MODES
            for layout, values in aggregate([r for r in rows if r["prefill_mode"] == mode]).items()}


def run(key, variant, outdir, model, env, trial, smoke=False):
    mode, layout = key.split("_", 1)
    command = [PYTHON, "benchmark_inference_metrics.py", "--model", model,
               "--kv-cache-dtype", "int8_half"]
    if smoke:
        command += ["--requests", "4", "--min-input", "257", "--max-input", "257",
                    "--min-output", "16", "--max-output", "16"]
    name = f"{key}_{'smoke' if smoke else trial}.txt"
    log = outdir/name
    print(f"START {name}", flush=True)
    environment = env.copy()
    environment["PYTHONPATH"] = str(variant)
    started = perf_counter()
    with log.open("w") as output:
        result = subprocess.run(command, cwd=variant, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, timeout=600)
    assert result.returncode == 0, f"benchmark failed: {log}"
    text = log.read_text()
    assert f"SCALE_LAYOUT: {layout} | package: {variant}/nanovllm/__init__.py" in text
    assert f"PREFILL_EXPERIMENT: {key}" in text
    row = parse_metrics(text)
    output_hash = re.search(r"Output token SHA256: ([a-f0-9]{64})", text)[1]
    shape_hash = re.search(r"Prefill shape SHA256: ([a-f0-9]{64})", text)[1]
    dispatch = json.loads(re.search(r"Prefill dispatch audit: (.*)", text)[1])
    batches = int(re.search(r"Prefill batches: (\d+)", text)[1])
    layers = json.loads((Path(model)/"config.json").read_text())["num_hidden_layers"]
    assert sum(dispatch.values()) == batches * layers
    if mode == "int8":
        assert dispatch["flash"] == 0 and dispatch["int8"] > 0
    row.update(variant=key, layout=layout, prefill_mode=mode, trial=trial, log=name,
               output_sha256=output_hash, prefill_shape_sha256=shape_hash,
               prefill_dispatch=dispatch, prefill_batches=batches,
               process_wall_s=perf_counter()-started)
    print(f"DONE {key} generate={row['elapsed_s']:.3f}s "
          f"output={row['output_tokens_per_s']:.2f}tok/s "
          f"TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms "
          f"prefill={row['stages']['prefill']['seconds']:.3f}s dispatch={dispatch}", flush=True)
    return row, dict(variant=key, trial=trial, log=name, command=command, cwd=str(variant),
                     process_wall_s=row["process_wall_s"], returncode=result.returncode)


def render_report(outdir, manifest, data):
    medians = data["medians"]
    columns = ("flash_token_head", "flash_head_token", "int8_token_head", "int8_head_token")
    lines = ["# 普通 prefill 切换到自写 INT8 kernel：完整模型对照", "",
        "四组均为 INT8 双 scale KV，BF16 模型权重。Flash 组普通 prefill 使用原始 BF16 K/V；"
        "INT8 组所有分配了缓存的 prefill 使用量化后的 paged K/V。decode 始终使用相应布局的自写 INT8 kernel。", "",
        "## 测量结果", "",
        "| 指标 | Flash prefill 原布局 | Flash prefill head/token | 自写 prefill 原布局 | 自写 prefill head/token |",
        "| --- | ---: | ---: | ---: | ---: |"]
    csv_rows = []
    def add(title, getter, unit, digits=2):
        values = {name: getter(medians[name]) for name in columns}
        lines.append("| "+title+" | "+" | ".join(f"{values[name]:.{digits}f}" for name in columns)+" |")
        csv_rows.append(dict(metric=title, unit=unit, **values))
    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for quantile in ("mean", "p50", "p95", "p99"):
            add(f"{metric} {quantile.upper()} ms", lambda m, a=metric, b=quantile: m["latency_ms"][a][b], "ms")
    for title, key, unit, digits in (
        ("generate 时间 s", "elapsed_s", "s", 3),
        ("请求/s", "requests_per_s", "request/s", 2),
        ("输出 token/s", "output_tokens_per_s", "token/s", 2),
        ("输入 token/s", "input_tokens_per_s", "token/s", 2),
        ("输入+输出 token/s", "total_tokens_per_s", "token/s", 2),
        ("KV blocks", "blocks", "block", 0),
        ("峰值 allocated GiB", "peak_allocated_gib", "GiB", 2),
        ("峰值 reserved GiB", "peak_reserved_gib", "GiB", 2),
    ):
        add(title, lambda m, k=key: m[k], unit, digits)
    for stage in ("prefill", "decode"):
        for key, unit, digits in (("seconds", "s", 3), ("tokens_per_s", "token/s", 2), ("tokens", "token", 0)):
            add(f"{stage} model-run {key}", lambda m, a=stage, b=key: m["stages"][a][b], unit, digits)
    lines += ["", "## 对照变化", ""]
    delta = lambda old, new: (new/old-1)*100
    for layout in LAYOUTS:
        old, new = [medians[f"{mode}_{layout}"] for mode in MODES]
        lines.append(f'- {layout}，将普通 prefill 从 FlashAttention 换成自写 kernel：'
                     f'prefill 总时间 {delta(old["stages"]["prefill"]["seconds"], new["stages"]["prefill"]["seconds"]):+.2f}%，'
                     f'TTFT P50 {delta(old["latency_ms"]["TTFT"]["p50"],new["latency_ms"]["TTFT"]["p50"]):+.2f}%，'
                     f'输出吞吐 {delta(old["output_tokens_per_s"],new["output_tokens_per_s"]):+.2f}%。')
    old, new = [medians[f"int8_{layout}"] for layout in LAYOUTS]
    lines.append(f'- prefill/decode 均自写时，head/token 相对原布局：'
                 f'prefill 总时间 {delta(old["stages"]["prefill"]["seconds"],new["stages"]["prefill"]["seconds"]):+.2f}%，'
                 f'TTFT P50 {delta(old["latency_ms"]["TTFT"]["p50"],new["latency_ms"]["TTFT"]["p50"]):+.2f}%，'
                 f'输出吞吐 {delta(old["output_tokens_per_s"],new["output_tokens_per_s"]):+.2f}%。')
    lines += ["", "## 条件与验证", "",
        "- Qwen3-0.6B，RTX 3090 Ti，单卡 GPU 0；CUDA Graph decode，prefill eager；BM=16、BN=64、BD=128，4 warps。",
        "- 256 请求同时到达；输入/输出长度均 100–1024，seed=0、temperature=0.6、ignore_eos=True；输入 142,827，输出 133,966 token。",
        f'- 每组 {manifest["runs_per_variant"]} 个新进程，顺序轮换；取各项统计量的轮中位数。生成预热和模型初始化不计入，未锁 GPU 频率。',
        "- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、block_size=256、gpu_memory_utilization=0.9；KV 容量和阶段 token 工作量一致。",
        "- 隔离副本里补齐普通 prefill 的 block table，再将已分配 INT8 缓存的 attention 都派发到自写 kernel。初始化时没有缓存，仍以 FlashAttention 预热；容量估算路径相同。",
        "- 集成正确性覆盖零前缀、混合前缀、长度 1/17/255/257、非连续物理页、跨页、部分 Q tile；对恢复后的 BF16 KV 调用 FlashAttention，rtol=0.02、atol=0.005 通过。",
        "- 正式测量记录每批 prefill 的长度摘要及每层实际派发次数，确保自写组没有调用 FlashAttention prefill。",
        "- 每个 prefill 模式内，两种布局和重复轮次的输出 token 摘要一致。Flash 与 INT8 prefill 的输出是否一致见下表；量化后 prefill 引入近似，内容不同不能作为布局错误。没有测语言质量。",
        "- TTFT 从整批请求提交到 CPU postprocess 首 token 完成，含排队；阶段时间为 ModelRunner.run 累计时间，含准备、模型、采样和同步，并非单 attention kernel 时间。",
        "- 原仓库 nanovllm 源码和 benchmark_inference_metrics.py 的 SHA256 前后相同；实验改动仅在四份副本。", "",
        "| prefill 模式 | 正式输出 SHA256 | Flash/INT8 一致 |",
        "| --- | --- | --- |"]
    digests = {mode: next(r["output_sha256"] for r in data["trials"] if r["prefill_mode"] == mode) for mode in MODES}
    for mode in MODES:
        lines.append(f'| {mode} | {digests[mode]} | {digests["flash"] == digests["int8"]} |')
    lines += ["", "## 每轮结果", "",
        "| 组 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | prefill s | Flash/INT8 prefill 调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in data["trials"]:
        audit = row["prefill_dispatch"]
        lines.append(f'| {row["variant"]} | {row["trial"]} | {row["elapsed_s"]:.3f} | '
                     f'{row["output_tokens_per_s"]:.2f} | {row["latency_ms"]["TTFT"]["p50"]:.2f} | '
                     f'{row["stages"]["prefill"]["seconds"]:.3f} | {audit["flash"]}/{audit["int8"]} |')
    lines += ["", "## 复现", "", "```bash",
        "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_prefill_kernel_metrics.py --outdir /tmp/prefill_kernel_metrics",
        "```", "", "产物包括 manifest.json、results.json、comparison.csv、workload.json、四份隔离源码、集成正确性记录、smoke 与正式日志。"]
    (outdir/"report.md").write_text("\n".join(lines)+"\n")
    with (outdir/"comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["metric", "unit", *columns])
        writer.writeheader()
        writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    assert args.runs > 0 and not (outdir/"manifest.json").exists(), "use a fresh output directory"
    outdir.mkdir(parents=True, exist_ok=True)
    originals = hashes(ROOT)
    variants = prepare_variants(outdir, originals)
    variant_hashes = {name: hashes(root) for name, root in variants.items()}
    rng = random.Random(0)
    prompts = [[rng.randint(0,10000) for _ in range(rng.randint(100,1024))] for _ in range(256)]
    max_tokens = [rng.randint(100,1024) for _ in range(256)]
    workload = outdir/"workload.json"
    workload.write_text(json.dumps(dict(seed=0, prompts=prompts, max_tokens=max_tokens))+"\n")
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), model=args.model,
        python=PYTHON, gpu=args.gpu, runs_per_variant=args.runs, variants=list(variants),
        variant_roots={name: str(root) for name, root in variants.items()},
        original_source_hashes=originals, variant_source_hashes=variant_hashes,
        workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        input_tokens=sum(map(len,prompts)), output_tokens=sum(max_tokens),
        config=dict(requests=256, input_length=[100,1024], output_length=[100,1024],
                    python_seed=0, torch_sampling_seed=0, temperature=.6, ignore_eos=True,
                    kv_cache_dtype="int8_half", max_model_len=4096, max_num_batched_tokens=16384,
                    max_num_seqs=512, block_size=256, gpu_memory_utilization=.9,
                    tensor_parallel_size=1, decode_cuda_graph=True),
        model_config=json.loads((Path(args.model)/"config.json").read_text()),
        weight_files=[dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                      for p in Path(args.model).glob("*.safetensors")],
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        nvidia_smi=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem","--format=csv,noheader"],text=True),
        validation=[], smoke_runs=[], runs=[])
    def save():
        (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    def unchanged():
        assert hashes(ROOT) == originals, "original repository source changed"
        assert all(hashes(variants[key]) == variant_hashes[key] for key in VARIANTS)
    save()
    snapshot = outdir/"driver_snapshot"/"profiling"
    snapshot.mkdir(parents=True)
    for name in (Path(__file__).name, "validate_prefill_kernel_route.py",
                 "run_scale_layout_metrics.py", "run_kv_metrics_comparison.py"):
        shutil.copyfile(ROOT/"profiling"/name, snapshot/name)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    for layout in LAYOUTS:
        variant = variants[f"int8_{layout}"]
        env = environment.copy()
        env["PYTHONPATH"] = str(variant)
        log = outdir/f"validate_{layout}.txt"
        output = outdir/f"validation_{layout}.json"
        command = [PYTHON, str(ROOT/"profiling/validate_prefill_kernel_route.py"),
                   "--layout", layout, "--output", str(output)]
        with log.open("w") as fp:
            result = subprocess.run(command, cwd=variant, env=env, stdout=fp,
                                    stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, f"integration correctness failed: {log}"
        record = json.loads(output.read_text())
        assert record["package"] == str(variant/"nanovllm/layers/attention.py")
        manifest["validation"].append(record)
        print(f"CORRECTNESS PASS {layout}: {record['cases']}", flush=True)
        unchanged()
        save()
    smokes = []
    for key in VARIANTS:
        unchanged()
        row, record = run(key, variants[key], outdir, args.model, environment, 0, smoke=True)
        smokes.append(row)
        manifest["smoke_runs"].append(dict(**record, metrics=row))
        save()
    for mode in MODES:
        assert len({row["output_sha256"] for row in smokes if row["prefill_mode"] == mode}) == 1
    assert len({row["blocks"] for row in smokes}) == 1
    rows = []
    for trial in range(1, args.runs+1):
        # Rotate modes and layouts to avoid always measuring a variant first.
        shift = (trial-1) % len(VARIANTS)
        order = VARIANTS[shift:]+VARIANTS[:shift]
        if trial % 2 == 0:
            order = order[::-1]
        for key in order:
            unchanged()
            row, record = run(key, variants[key], outdir, args.model, environment, trial)
            assert row["input_tokens"] == manifest["input_tokens"]
            assert row["output_tokens"] == manifest["output_tokens"]
            rows.append(row)
            manifest["runs"].append(record)
            save()
            (outdir/"results.json").write_text(json.dumps(dict(trials=rows, medians=summarize(rows)),indent=2)+"\n")
            unchanged()
    for mode in MODES:
        assert len({r["output_sha256"] for r in rows if r["prefill_mode"] == mode}) == 1
    assert len({r["blocks"] for r in rows}) == 1
    assert len({r["prefill_shape_sha256"] for r in rows}) == 1
    assert len({r["stages"]["prefill"]["tokens"] for r in rows}) == 1
    assert len({r["stages"]["decode"]["tokens"] for r in rows}) == 1
    manifest.update(original_repository_unchanged=True,
                    outputs_identical_within_each_prefill_mode=True,
                    finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    render_report(outdir, manifest, dict(trials=rows, medians=summarize(rows)))
    print("COMPLETE", outdir/"report.md", flush=True)


if __name__ == "__main__":
    main()
