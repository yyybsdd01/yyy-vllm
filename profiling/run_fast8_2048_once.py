"""One 2048-request fast8 run using the current offload/reservation scheduler."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_current_int8_metrics import PYTHON, source_hashes
from profiling.run_project_bf16_5loads import field, prepare, run_logged, save, verify
from profiling.run_kv_metrics_comparison import parse_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    out = args.outdir.resolve()
    assert not (out / "manifest.json").exists(), "use a fresh output directory"
    options = argparse.Namespace(requests=[2048], runs=1, kernel="fast8", gpu=args.gpu,
                                 model="/home/xgd/huggingface/Qwen3-0.6B")
    manifest = prepare(options, out)
    manifest.update(variants={"project": dict(mode="int8_half", offload=True)},
                    scope="one performance run; comparisons reuse earlier three-run medians")
    checkout = Path(manifest["checkout"])
    donor = ROOT / "profiling/kv_offload_decode_reserve_2026-10-06/variants/reserved"
    for name in ("nanovllm/layers/quantized_attention.py", "nanovllm/engine/model_runner.py"):
        assert (checkout / name).read_bytes() == (donor / name).read_bytes(), name
    changed = {k for k in manifest["root_source_hashes"]
               if manifest["root_source_hashes"][k] != manifest["inference_source_hashes"][k]}
    assert changed == {"nanovllm/layers/quantized_attention.py", "nanovllm/engine/model_runner.py"}
    reference = ROOT / "profiling/current_project_vs_bf16_5loads_2026-10-06"
    prior_manifest = json.loads((reference / "manifest.json").read_text())
    prior_results = json.loads((reference / "results.json").read_text())
    assert prior_manifest["root_source_hashes"] == manifest["root_source_hashes"]
    workload = manifest["workloads"]["2048"]
    assert workload["sha256"] == prior_manifest["workloads"]["2048"]["sha256"]
    assert (out / workload["file"]).read_bytes() == (reference / workload["file"]).read_bytes()
    manifest.update(reference=str(reference), inference_changes=sorted(changed),
                    kernel_config=dict(prefill_BM=64, decode_BM=16, BN=64, num_warps=8, num_stages=1,
                                       first_prefill="int8_paged_attention"))
    shutil.copyfile(Path(__file__), out / "driver_snapshot/profiling" / Path(__file__).name)
    save(out / "manifest.json", manifest)
    verify(manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(checkout), PYTHONUNBUFFERED="1")
    command = [PYTHON, str(checkout / "benchmark_inference_metrics.py"), "--model", manifest["model"],
               "--requests", "2048", "--kv-cache-dtype", "int8_half", "--kv-cpu-offload"]
    log = out / "2048_fast8_1.txt"
    before = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"], text=True)
    start = perf_counter()
    output = run_logged(command, log, env, checkout)
    row = parse_metrics(output)
    row.update(variant="fast8", trial=1, raw_log=log.name, gpu_before=before,
               process_wall_s=perf_counter()-start, preemptions=field(output, "Preemptions", int),
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
               offload_stats=field(output, "KV offload statistics"))
    assert (row["requests"], row["seed"], row["mode"], row["graph"]) == (2048, 0, "int8_half", True)
    assert row["workload_sha256"] == workload["sha256"]
    assert (row["input_tokens"], row["output_tokens"]) == (workload["input_tokens"], workload["output_tokens"])
    assert Path(row["package"]).resolve().is_relative_to(checkout)
    assert row["actual_kv_dtype"] == "torch.int8"
    assert row["attention_dispatch"]["flash"] == 0 and row["attention_dispatch"]["int8"] > 0
    assert row["decode_batch_stats"]["maximum"] <= 512
    histogram = row["request_preemption_stats"]["histogram"]
    assert sum(histogram.values()) == 2048
    assert sum(int(k)*v for k,v in histogram.items()) == row["preemptions"]
    stats = row["offload_stats"]
    assert stats["active_handles"] == stats["restoring"] == 0
    assert stats["cpu_pool_bytes"] <= stats["cpu_budget_bytes"]
    assert stats["offloaded_sequences"] + stats["fallback_preemptions"] == row["preemptions"]
    assert stats["offloaded_sequences"] == stats["restored_sequences"] == row["restore_progress"]["restore_events"]
    for direction in ("d2h", "h2d"):
        assert stats["transport"][direction+"_blocks"] == stats["transport"][direction+"_completed_events"]
    verify(manifest)
    assert source_hashes(ROOT) == manifest["root_source_hashes"]
    baseline = prior_results["medians"]["2048"]
    assert row["blocks"] == baseline["int8_half"]["blocks"]
    manifest.update(commands=[dict(command=command, log=log.name)],
                    finished_utc=datetime.now(timezone.utc).isoformat(), source_checks_passed=True)
    save(out / "manifest.json", manifest)
    results = dict(trials=[row], reference_medians=baseline, source_unchanged=True,
                   completed_utc=manifest["finished_utc"], validation="PASS")
    save(out / "results.json", results)
    lines = ["# fast8 当前卸载调度：2048 请求单轮", "",
             "fast8 为本次新测 1 轮；BF16 和根目录 INT8 为前一轮相同负载的三轮中位数。",
             "GPU0 RTX 3090 Ti / Qwen3-0.6B，模型权重 BF16，fast8 为双 scale INT8 KV。",
             "fast8：prefill BM64、decode BM16、BN64、8 warps、num_stages=1；首次/缓存 prefill 和 decode 使用 INT8 内核。",
             "当前异步卸载＋decode 块预留；CPU pinned 池4GiB，最多8个恢复请求；CUDA Graph decode。",
             f"输入 {row['input_tokens']}、输出 {row['output_tokens']} token；seed0，长度各100–1024，2048请求同时提交，max_num_seqs512。", "",
             "| 指标 | BF16（旧3轮中位数） | 根目录INT8（旧3轮中位数） | fast8（新1轮） |",
             "| --- | ---: | ---: | ---: |"]
    values = [baseline["auto"], baseline["int8_half"], row]
    csv_rows = []
    def emit(label, numbers):
        lines.append("| " + label + " | " + " | ".join(f"{n:.3f}" for n in numbers) + " |")
        csv_rows.append(dict(metric=label, bf16=numbers[0], root_int8=numbers[1], fast8=numbers[2]))
    for key in ("output_tokens_per_s", "requests_per_s", "input_tokens_per_s", "total_tokens_per_s", "elapsed_s",
                "preemptions", "blocks", "peak_allocated_gib", "peak_reserved_gib"):
        emit(key, [v[key] for v in values])
    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        for percentile in ("mean", "p50", "p95", "p99"):
            emit(f"{metric} {percentile} ms", [v["latency_ms"][metric][percentile] for v in values])
    for stage in ("prefill", "decode"):
        for key in ("tokens", "seconds", "tokens_per_s"):
            emit(f"{stage} {key}", [v["stages"][stage][key] for v in values])
    for key in ("request_preemption_stats", "decode_batch_stats", "restore_progress"):
        for name, value in row[key].items():
            if isinstance(value, (int, float)):
                emit(key+"."+name, [v[key][name] for v in values])
    lines += ["", "TTFT 包含排队；TPOT 为每请求首末 token 时间差除以后续 token 数；ITL 汇总相邻 token 间隔。",
              "整模型性能包含 KV 容量、prefill 路径、内核与调度的共同影响；单轮结果用于本次试测。",
              "核验：正式prefill派发flash=0/int8>0；工作负载hash一致；每请求长度/时间戳通过；传输结束、CPU预算、抢占直方图与推理源码hash通过。",
              "本次未重测PPL，性能结果不构成质量评估。", "",
              "```bash", f"{PYTHON} profiling/run_fast8_2048_once.py --outdir /tmp/fast8-2048-recheck", "```", ""]
    (out / "report.md").write_text("\n".join(lines))
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "bf16", "root_int8", "fast8"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"PASS fast8 2048 once tps={row['output_tokens_per_s']:.2f} preempt={row['preemptions']} report={out/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
