"""Compare kv_len versus visible_k load masks using identical paged INT8 KV.

Run after preserving before.py and after.py in --snapshot-dir. Timings exclude
compilation, allocation and Python launch overhead (batched CUDA Graph/events).
"""

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import torch
import triton
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_attention_scales import inputs, launch, percentile


def load_snapshot(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module._int8_paged_attention_kernel


def validate_reference(data, output, groups):
    shape = data["k"].shape
    ks = data["ks"] if groups == 1 else data["ks2"]
    vs = data["vs"] if groups == 1 else data["vs2"]

    def restore(cache, scales):
        split_shape = (*shape[:-1], groups, shape[-1] // groups)
        scale_shape = (*shape[:-1], groups, 1)
        return (cache.float().reshape(split_shape) * scales.reshape(scale_shape)).reshape(shape).bfloat16()

    k, v = restore(data["k"], ks), restore(data["v"], vs)
    kwargs = dict(block_table=data["table"], softmax_scale=128 ** -0.5, causal=True)
    if data["mode"] == "decode":
        expected = flash_attn_with_kvcache(
            data["q"][:, None], k, v, cache_seqlens=data["lengths"], **kwargs,
        )[:, 0]
    else:
        expected = flash_attn_varlen_func(
            data["q"], k, v, cu_seqlens_q=data["cu_q"], cu_seqlens_k=data["cu_k"],
            max_seqlen_q=data["query_length"], max_seqlen_k=data["context"], **kwargs,
        )
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
    return (output.float() - expected.float()).abs().max().item()


def run_case(kernels, data, groups, args, rng):
    scale_name = "single" if groups == 1 else "dual"
    outputs = {name: torch.empty_like(data["q"]) for name in kernels}
    compiled = {}
    for name, kernel in kernels.items():
        for _ in range(5):
            compiled[name] = launch(kernel, data, outputs[name], scale_name)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs["after"], outputs["before"], rtol=0, atol=0)
    flash_error = validate_reference(data, outputs["after"], groups)
    graphs = {}
    for name, kernel in kernels.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.calls_per_graph):
                launch(kernel, data, outputs[name], scale_name)
        graphs[name] = graph
    torch.cuda.synchronize()
    for graph in graphs.values():
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    times = {name: [] for name in kernels}
    for round_index in range(args.rounds):
        # Alternate order, with the first order randomized for each shape.
        if round_index == 0:
            order = list(kernels)
            rng.shuffle(order)
        else:
            order.reverse()
        for name in order:
            graphs[name].replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / (args.replays * args.calls_per_graph))
    row = {key: data[key] for key in ("batch", "context", "mode", "query_length")}
    row.update(scale_groups=groups, before_after_exact=True, flash_max_abs_error=flash_error)
    row["variants"] = {}
    for name in kernels:
        kernel = compiled[name]
        row["variants"][name] = dict(
            median_us=statistics.median(times[name]), p10_us=percentile(times[name], .1),
            p90_us=percentile(times[name], .9), round_us=times[name], registers=kernel.n_regs,
            spills=kernel.n_spills, shared_bytes=kernel.metadata.shared,
            cubin_sha256=hashlib.sha256(kernel.asm["cubin"]).hexdigest(),
        )
    before = row["variants"]["before"]["median_us"]
    after = row["variants"]["after"]["median_us"]
    row["latency_reduction_pct"] = (1 - after / before) * 100
    row["paired_reduction_pct"] = [(1 - a / b) * 100 for a, b in zip(times["after"], times["before"])]
    print(f'{data["mode"]:7} G={groups} B={data["batch"]:<3} K={data["context"]:<4} '
          f'Q={data["query_length"]:<4} before={before:9.3f} us after={after:9.3f} us '
          f'change={row["latency_reduction_pct"]:+7.2f}% '
          f'regs={compiled["before"].n_regs}/{compiled["after"].n_regs} '
          f'PASS (exact A/B, Flash max={flash_error:.6f})', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "profiling/attention_visible_mask_2026-10-04")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    assert args.rounds > 0 and args.replays > 0 and args.calls_per_graph > 0
    before_path, after_path = (args.snapshot_dir / f"{name}.py" for name in ("before", "after"))
    before, after = before_path.read_bytes(), after_path.read_bytes()
    assert before.count(b"valid_k = logical_k < kv_len\n") == 1
    assert before.replace(b"valid_k = logical_k < kv_len\n", b"valid_k = logical_k < visible_k\n") == after
    assert after == (ROOT / "nanovllm/layers/quantized_attention.py").read_bytes()
    kernels = {name: load_snapshot(path, f"attention_mask_{name}")
               for name, path in (("before", before_path), ("after", after_path))}
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    # Small/full prefill, cached prefill, partial tiles/pages, decode controls.
    cases = [
        (1, 128, "prefill", 16),
        (1, 128, "prefill", 128), (8, 128, "prefill", 128),
        (1, 1024, "prefill", 1024), (8, 1024, "prefill", 1024),
        (1, 1024, "prefill", 128), (8, 1024, "prefill", 128),
        (8, 1025, "prefill", 129), (8, 4096, "prefill", 128),
        (64, 1024, "decode", 1), (256, 1024, "decode", 1),
        (64, 1025, "decode", 1),
    ]
    result = dict(
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
        cuda=torch.version.cuda, seed=args.seed, rounds=args.rounds, replays=args.replays,
        calls_per_graph=args.calls_per_graph, head_dim=128, query_heads=16, kv_heads=8,
        block_size=256, BM=16, BN=64, BD=128, q_dtype="bfloat16", kv_dtype="int8",
        before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(),
        helper_sha256=hashlib.sha256((ROOT / "profiling/compare_attention_scales.py").read_bytes()).hexdigest(),
        measurement="CUDA Events / batched CUDA Graph; warm repeated inputs; kernel only",
        correctness="exact before/after and FlashAttention restored-KV comparison, rtol=.02 atol=.005",
        cases=[],
    )
    output_path = args.output or args.snapshot_dir / "timings.json"
    rng = random.Random(args.seed)
    print(f'GPU: {result["gpu"]} | rounds={args.rounds} | positive change means faster', flush=True)
    for batch, context, mode, qlen in cases:
        data = inputs(batch, context, mode, qlen, args.seed)
        # Exercise genuinely distinct half-head scales, shared by both variants.
        data["ks2"][..., 1::2] *= .3
        data["vs2"][..., 1::2] *= 1.8
        groups_order = [1, 2]
        rng.shuffle(groups_order)
        for groups in groups_order:
            result["cases"].append(run_case(kernels, data, groups, args, rng))
            output_path.write_text(json.dumps(result, indent=2) + "\n")
        del data
        torch.cuda.empty_cache()
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
