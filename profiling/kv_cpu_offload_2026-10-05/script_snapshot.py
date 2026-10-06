"""Measure isolated asynchronous full-sequence KV offload with the real cache layout.

No scheduler or inference source is changed. GPU tensors contain synthetic data;
CPU scheduler replay supplies victim sizes and physical block tables.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from itertools import count
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def source_hashes():
    paths = sorted((ROOT / "nanovllm").rglob("*.py")) + [ROOT / "benchmark_inference_metrics.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def replay_victims(workload, blocks, expected_preemptions, expected_steps):
    Sequence.counter = count()
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=512, max_num_batched_tokens=16384, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=blocks, preemption_lock=False,
    ))
    requests = [Sequence(prompt, SamplingParams(max_tokens=n, ignore_eos=True))
                for prompt, n in zip(workload["prompts"], workload["max_tokens"])]
    for seq in requests:
        scheduler.add(seq)
    sizes = Counter()
    samples = {}
    original = scheduler.preempt

    def record(seq):
        n = len(seq.block_table)
        sizes[n] += 1
        samples.setdefault(n, list(seq.block_table))
        original(seq)

    scheduler.preempt = record
    decode_steps = 0
    while not scheduler.is_finished():
        batch, prefill = scheduler.schedule()
        decode_steps += not prefill
        scheduler.postprocess(batch, [30000 + seq.seq_id for seq in batch], prefill)
    assert sum(sizes.values()) == expected_preemptions
    assert decode_steps == expected_steps
    assert all(seq.num_completion_tokens == seq.max_tokens for seq in requests)
    return dict(scope="CPU replay with synthetic output tokens, matched GPU preemption/step counts",
                histogram=dict(sorted(sizes.items())), total_events=sum(sizes.values()),
                mean_blocks=sum(n * c for n, c in sizes.items()) / sum(sizes.values()),
                sample_block_tables=samples, decode_steps=decode_steps)


def summarize(samples):
    return {k: dict(mean=float(np.mean([s[k] for s in samples])),
                    p50=float(np.percentile([s[k] for s in samples], 50)),
                    p95=float(np.percentile([s[k] for s in samples], 95)))
            for k in samples[0]}


def measure_mode(mode, capacity, victim_data, args, checkpoint):
    dtype = torch.int8 if mode == "int8_half" else torch.bfloat16
    cache = torch.empty((2, 28, capacity, 256, 8, 128), dtype=dtype, device="cuda")
    scales = (torch.empty((2, 28, capacity, 256, 16), dtype=torch.float32, device="cuda")
              if mode == "int8_half" else None)
    # Same leading K/V, layer, physical-block order and strides as ModelRunner.
    # Only selected pages need initialization for this transfer experiment.
    block_bytes = 2 * 28 * 256 * 8 * 128 * cache.element_size()
    if scales is not None:
        block_bytes += 2 * 28 * 256 * 16 * scales.element_size()
    results = []
    stream = torch.cuda.Stream()
    pattern = torch.arange(2 * 28 * 256 * 8 * 128, device="cuda", dtype=torch.int32)
    pattern = ((pattern % 251) - 125).to(dtype).reshape(2, 28, 256, 8, 128)
    scale_pattern = (torch.arange(2 * 28 * 256 * 16, device="cuda", dtype=torch.float32)
                     .reshape(2, 28, 256, 16) / 10000) if scales is not None else None
    for n in range(1, 9):
        ids = victim_data["sample_block_tables"].get(n)
        sampled_victim = ids is not None
        if ids is None:
            ids = [((i * 137 + 11) % capacity) for i in range(n)]
        assert len(set(ids)) == n
        for i, page in enumerate(ids):
            cache[:, :, page].copy_(pattern + i)
            if scales is not None:
                scales[:, :, page].copy_(scale_pattern + i)
        host_ids = torch.tensor(ids, dtype=torch.int64, pin_memory=True)
        device_ids = torch.empty(n, dtype=torch.int64, device="cuda")
        host_kv = torch.empty((2, 28, n, 256, 8, 128), dtype=dtype, pin_memory=True)
        packed_kv = torch.empty_like(host_kv, device="cuda")
        host_scales = (torch.empty((2, 28, n, 256, 16), dtype=torch.float32, pin_memory=True)
                       if scales is not None else None)
        packed_scales = torch.empty_like(host_scales, device="cuda") if scales is not None else None
        torch.cuda.synchronize()
        # Prebuilt views remove Python view construction from the direct-copy lower bound.
        direct_pairs = [(host_kv[k, layer, i], cache[k, layer, page])
                        for k in range(2) for layer in range(28) for i, page in enumerate(ids)]
        if scales is not None:
            direct_pairs += [(host_scales[k, layer, i], scales[k, layer, page])
                             for k in range(2) for layer in range(28) for i, page in enumerate(ids)]

        for method in (["pack_then_d2h", "direct_per_layer"] if n in (1, 4, 8) else ["pack_then_d2h"]):
            raw = []
            start, packed, done = (torch.cuda.Event(enable_timing=True) for _ in range(3))

            def one():
                with torch.cuda.stream(stream):
                    start.record(stream)
                    t0 = time.perf_counter()
                    if method == "pack_then_d2h":
                        device_ids.copy_(host_ids, non_blocking=True)
                        torch.index_select(cache, 2, device_ids, out=packed_kv)
                        if scales is not None:
                            torch.index_select(scales, 2, device_ids, out=packed_scales)
                        packed.record(stream)
                        host_kv.copy_(packed_kv, non_blocking=True)
                        if scales is not None:
                            host_scales.copy_(packed_scales, non_blocking=True)
                    else:
                        packed.record(stream)
                        for dst, src in direct_pairs:
                            dst.copy_(src, non_blocking=True)
                    t1 = time.perf_counter()
                    done.record(stream)
                    pending_after_enqueue = not done.query()
                done.synchronize()
                t2 = time.perf_counter()
                return dict(enqueue_ms=(t1 - t0) * 1000,
                            completion_wall_ms=(t2 - t0) * 1000,
                            pack_stream_ms=start.elapsed_time(packed),
                            d2h_stream_ms=packed.elapsed_time(done),
                            total_stream_ms=start.elapsed_time(done),
                            unfinished_at_enqueue=float(pending_after_enqueue))

            for _ in range(args.warmup):
                one()
            rounds = []
            repeats = args.repeats if method == "pack_then_d2h" else max(10, args.repeats // 2)
            for round_index in range(args.rounds):
                round_samples = [one() for _ in range(repeats)]
                raw.extend(dict(round=round_index + 1, **v) for v in round_samples)
                rounds.append(summarize(round_samples))

            # Exact check after completion, outside all timings, including every scale.
            expected_kv = torch.stack([cache[:, :, page].cpu() for page in ids], dim=2)
            assert torch.equal(host_kv, expected_kv), (mode, n, method, "KV mismatch")
            if scales is not None:
                expected_scales = torch.stack([scales[:, :, page].cpu() for page in ids], dim=2)
                assert torch.equal(host_scales, expected_scales), (mode, n, method, "scale mismatch")
                del expected_scales
            del expected_kv
            value = dict(mode=mode, capacity=capacity, blocks=n, bytes=block_bytes * n,
                         block_tables=ids, block_table_from_replay=sampled_victim,
                         method=method, pinned=True, correctness=True,
                         source_shape=list(cache.shape), source_stride=list(cache.stride()),
                         scale_shape=list(scales.shape) if scales is not None else None,
                         gpu_staging_bytes=block_bytes * n if method == "pack_then_d2h" else 0,
                         d2h_copy_calls=(2 if scales is not None else 1) if method == "pack_then_d2h" else len(direct_pairs),
                         summary=summarize([{k: v for k, v in s.items() if k != "round"} for s in raw]),
                         rounds=rounds, samples=raw)
            results.append(value)
            checkpoint(results)
            print(f'{mode:10s} {n} blocks {method:17s} '
                  f'enqueue={value["summary"]["enqueue_ms"]["p50"]:.4f}ms '
                  f'complete={value["summary"]["completion_wall_ms"]["p50"]:.4f}ms '
                  f'pack={value["summary"]["pack_stream_ms"]["p50"]:.4f}ms '
                  f'D2H={value["summary"]["d2h_stream_ms"]["p50"]:.4f}ms', flush=True)
        del direct_pairs, packed_kv, host_kv, packed_scales, host_scales, device_ids, host_ids
    del cache, scales, pattern, scale_pattern
    gc.collect()
    torch.cuda.empty_cache()
    return results


def decode_reference(warp_dir, preemption_dir):
    warp_results = json.loads((warp_dir / "results.json").read_text())
    old_results = json.loads((preemption_dir / "results.json").read_text())
    cases = {
        "int8_half_warps4": [v for v in warp_results["trials"] if v["num_warps"] == 4],
        "int8_half_warps8": [v for v in warp_results["trials"] if v["num_warps"] == 8],
        "bf16": [v for v in old_results["trials"] if v["variant"] == "bf16" and v["policy"] == "original"],
    }
    return {name: dict(mean_step_ms=statistics.median(v["stages"]["decode"]["seconds"] * 1000 /
                                                    v["decode_batch_stats"]["steps"] for v in trials),
                       steps=trials[0]["decode_batch_stats"]["steps"],
                       mean_batch=trials[0]["decode_batch_stats"]["mean"],
                       decode_seconds=[v["stages"]["decode"]["seconds"] for v in trials],
                       capacity=trials[0]["blocks"], preemptions=trials[0]["preemptions"],
                       scope="complete ModelRunner.run: input preparation, model, sampling and synchronization",
                       source=str((warp_dir if name.startswith("int8") else preemption_dir) / "results.json"))
            for name, trials in cases.items()}


def render(result):
    decode = result["decode_reference"]
    lookup = {(v["mode"], v["blocks"], v["method"]): v for v in result["measurements"]}
    out = ["# 被抢占 sequence 的异步 KV GPU→CPU 卸载耗时", "",
           "RTX 3090 Ti / GPU 0；预分配 pinned CPU 缓冲区，独立 CUDA stream，non_blocking=True。",
           "当前尚未集成 offload：此处测合成 KV 数据的真实传输，保留当前缓存形状和 stride。",
           "全部 28 层 K、V；INT8 还包括双 FP32 scale；每个物理 block=256 token。",
           "", "| 完整模型 decode 对照 | 每步平均 ms | 平均实际 batch | decode 步数 |",
           "| --- | ---: | ---: | ---: |"]
    for name, d in decode.items():
        out.append(f'| {name} | {d["mean_step_ms"]:.3f} | {d["mean_batch"]:.2f} | {d["steps"]} |')
    out += ["", "上述值是三次正式1024请求运行中每次 decode 总秒数/步数，再取中位数。",
            "含输入准备、整模型、采样与同步；不是单 attention kernel 或单请求 TPOT。",
            "", "| blocks | INT8 MiB | 异步提交 P50 ms | GPU 整理 P50 ms | D2H P50 ms | 完成墙钟 P50 ms | 完成 P95 ms | 占 8 warp decode | BF16 完成 P50 ms |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in range(1, 9):
        v = lookup["int8_half", n, "pack_then_d2h"]
        s = v["summary"]
        b = lookup["bf16", n, "pack_then_d2h"]["summary"]
        out.append(f'| {n} | {v["bytes"] / 2**20:.3f} | {s["enqueue_ms"]["p50"]:.4f} | '
                   f'{s["pack_stream_ms"]["p50"]:.4f} | {s["d2h_stream_ms"]["p50"]:.4f} | '
                   f'{s["completion_wall_ms"]["p50"]:.4f} | {s["completion_wall_ms"]["p95"]:.4f} | '
                   f'{s["completion_wall_ms"]["p50"] / decode["int8_half_warps8"]["mean_step_ms"] * 100:.1f}% | '
                   f'{b["completion_wall_ms"]["p50"]:.4f} |')
    out += ["", "GPU 整理包括 block ID 的异步 H2D 和 index_select；随后 INT8 两次连续 D2H，BF16 一次。",
            "完成墙钟包含 Python/CUDA 提交与等待，不能把异步提交时间当作传完时间。",
            "数据大小：INT8 每块14.875 MiB（KV14+scale0.875）；BF16每块28 MiB。",
            "", "| CPU 调度回放统计 | 抢占事件 | 被抢占时 blocks 分布 | 平均 blocks | 加权完成估计 ms |",
            "| --- | ---: | --- | ---: | ---: |"]
    for mode, victim in result["victims"].items():
        estimate = sum(c * lookup[mode, int(n), "pack_then_d2h"]["summary"]["completion_wall_ms"]["p50"]
                       for n, c in victim["histogram"].items()) / victim["total_events"]
        victim["weighted_completion_estimate_ms"] = estimate
        out.append(f'| {mode} | {victim["total_events"]} | {victim["histogram"]} | {victim["mean_blocks"]:.3f} | {estimate:.4f} |')
    out += ["", "blocks 分布为相同 workload 的 CPU 调度回放，使用合成输出 token；抢占总数和 decode 步数匹配正式 GPU 结果。",
            "加权值按事件大小分布和独立卸载测量估计，不是整批推理实测 offload 时间。",
            "", "| 逐层直接拷贝对照 | blocks | D2H 次数 | 提交 P50 ms | 完成 P50 ms |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for v in result["measurements"]:
        if v["method"] == "direct_per_layer":
            s = v["summary"]
            out.append(f'| {v["mode"]} | {v["blocks"]} | {v["d2h_copy_calls"]} | '
                       f'{s["enqueue_ms"]["p50"]:.4f} | {s["completion_wall_ms"]["p50"]:.4f} |')
    out += ["", "此对照预先创建每层视图，因此还未包括调度器中动态构造视图的成本。",
            "", "测量方法与边界：", "",
            f'- 整理路径每大小{result["config"]["rounds"]}轮×{result["config"]["repeats"]}次；预热{result["config"]["warmup"]}次，所有轮的样本统计P50/P95。',
            "- 保留完整缓存容量：INT8 1320 blocks，BF16 701 blocks；源数据布局与 ModelRunner 一致。",
            "- 每次异步提交后以 CUDA event.synchronize 等待完成，再复用缓冲区；所有K/V/scale逐元素校验通过。",
            "- CPU pinned 缓冲和 GPU staging 均预分配，其首次分配成本不在表中。",
            "- 每次只卸载一条 sequence，独立 stream，无并行 decode、无其他 GPU 工作；未测并行竞争或吞吐收益。",
            "- 整理路径原物理页必须保留到GPU整理完成；整理后的GPU staging必须保留到D2H完成。",
            "- 逐层直接拷贝路径原物理页必须保留到D2H完成；当前 preempt 立即 deallocate 的流程不能直接沿用。",
            "- 测整块，包括末块无效槽位；不包括 CPU→GPU 恢复、GPU scatter、队列/元数据处理。",
            "- BF16 与 INT8 完整 decode 的平均 batch 不同，不能按单步时间判定哪种推理模式更快。",
            "- 未锁GPU频率；GPU 0 PCIe最大Gen3×16。",
            "", "复现：", "", "```bash",
            "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/bench_kv_cpu_offload.py --outdir /tmp/kv-offload-recheck",
            "```", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    assert args.rounds > 0 and args.repeats > 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    warp_dir = ROOT / "profiling/attention_warps_4_vs_8_1024_2026-10-05"
    preemption_dir = ROOT / "profiling/preemption_lock_1024_2026-10-05"
    before = source_hashes()
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  config=dict(rounds=args.rounds, repeats=args.repeats, warmup=args.warmup,
                              gpu=0, non_blocking=True, pinned=True, block_size=256,
                              layers=28, kv_heads=8, head_dim=128, int8_scale_groups=2),
                  source_hashes=before, decode_reference=decode_reference(warp_dir, preemption_dir),
                  victims={}, measurements=[])
    workload_path = preemption_dir / "workload_1024.json"
    workload = json.loads(workload_path.read_text())
    result["workload_sha256"] = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    for mode, reference in (("int8_half", result["decode_reference"]["int8_half_warps8"]),
                            ("bf16", result["decode_reference"]["bf16"])):
        print("Replaying scheduler", mode, flush=True)
        result["victims"][mode] = replay_victims(workload, reference["capacity"], reference["preemptions"], reference["steps"])
        print(mode, result["victims"][mode]["histogram"], flush=True)
    torch.cuda.set_device(0)
    result["torch_version"] = torch.__version__
    result["gpu"] = torch.cuda.get_device_name(0)
    result["nvidia_smi"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,pcie.link.gen.max,pcie.link.width.max,memory.used,utilization.gpu",
        "--format=csv,noheader"], text=True)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.outdir / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for mode in ("int8_half", "bf16"):
        completed = list(result["measurements"])

        def checkpoint(current):
            result["measurements"] = completed + current
            (args.outdir / "results.json").write_text(json.dumps(result, indent=2) + "\n")

        measure_mode(mode, result["decode_reference"]["int8_half_warps8" if mode == "int8_half" else "bf16"]["capacity"],
                     result["victims"][mode], args, checkpoint)
    result["root_sources_unchanged"] = source_hashes() == before
    assert result["root_sources_unchanged"]
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.outdir / "report.md").write_text(render(result))
    (args.outdir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", args.outdir / "report.md", flush=True)


if __name__ == "__main__":
    main()
