"""Isolated single/two-scale paged-attention timing and compiler ablations.

The production kernel is imported unchanged. Experimental copies change only
the two-scale loading/selection block and live in the output directory.
``dual`` always denotes four separate scale loads, and ``dual_packed`` denotes
the two-dimensional load implementation, including after production switches.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanovllm.layers.quantized_attention import _int8_paged_attention_kernel


def make_variants(outdir):
    source = (ROOT / "nanovllm/layers/quantized_attention.py").read_text()
    kernel_source = source[:source.index("\ndef int8_paged_attention(")]
    offset_line = kernel_source.index("            scale_offset = slot * KV_HEADS * SCALE_GROUPS")
    start = kernel_source.index("\n", offset_line) + 1
    end = kernel_source.index("        # 从 INT8 KV cache", start)
    original = kernel_source[start:end]
    alternatives = {
        "dual": (
            "            k_first = tl.load(ks_ptr + scale_offset, valid_k, other=0)\n"
            "            k_second = tl.load(ks_ptr + scale_offset + 1, valid_k, other=0)\n"
            "            v_first = tl.load(vs_ptr + scale_offset, valid_k, other=0)\n"
            "            v_second = tl.load(vs_ptr + scale_offset + 1, valid_k, other=0)\n"
            "            k_scale = tl.where(dims[None, :] < HEAD_DIM // 2,\n"
            "                               k_first[:, None], k_second[:, None])\n"
            "            v_scale = tl.where(dims[None, :] < HEAD_DIM // 2,\n"
            "                               v_first[:, None], v_second[:, None])\n"
        ),
        "dual_stride_first": (
            "            k_scale = tl.load(ks_ptr + scale_offset, valid_k, other=0)[:, None]\n"
            "            v_scale = tl.load(vs_ptr + scale_offset, valid_k, other=0)[:, None]\n"
        ),
        "dual_load_average": (
            "            k_first = tl.load(ks_ptr + scale_offset, valid_k, other=0)\n"
            "            k_second = tl.load(ks_ptr + scale_offset + 1, valid_k, other=0)\n"
            "            v_first = tl.load(vs_ptr + scale_offset, valid_k, other=0)\n"
            "            v_second = tl.load(vs_ptr + scale_offset + 1, valid_k, other=0)\n"
            "            k_scale = ((k_first + k_second) * 0.5)[:, None]\n"
            "            v_scale = ((v_first + v_second) * 0.5)[:, None]\n"
        ),
        "dual_packed": (
            "            pair_offsets = scale_offset[:, None] + tl.arange(0, 2)[None, :]\n"
            "            k_pair = tl.load(ks_ptr + pair_offsets, valid_k[:, None], other=0)\n"
            "            v_pair = tl.load(vs_ptr + pair_offsets, valid_k[:, None], other=0)\n"
            "            k_first, k_second = tl.split(k_pair)\n"
            "            v_first, v_second = tl.split(v_pair)\n"
            "            k_scale = tl.where(dims[None, :] < HEAD_DIM // 2,\n"
            "                               k_first[:, None], k_second[:, None])\n"
            "            v_scale = tl.where(dims[None, :] < HEAD_DIM // 2,\n"
            "                               v_first[:, None], v_second[:, None])\n"
        ),
    }
    kernels = {"single": _int8_paged_attention_kernel}
    for name, replacement in alternatives.items():
        original_code = "\n".join(line for line in original.splitlines()
                                  if line.strip() and not line.lstrip().startswith("#"))
        if original_code == replacement.rstrip("\n"):
            kernels[name] = _int8_paged_attention_kernel
            continue
        name_in_source = f"_attention_{name}"
        changed = kernel_source.replace(original, replacement).replace(
            "_int8_paged_attention_kernel", name_in_source
        )
        path = outdir / f"{name}.py"
        path.write_text(changed)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        kernels[name] = getattr(module, name_in_source)
    return kernels, hashlib.sha256(source.encode()).hexdigest()


def inputs(batch, context, mode, query_length, seed):
    torch.manual_seed(seed)
    block_size, kv_heads, query_heads, head_dim = 256, 8, 16, 128
    blocks_per_seq = triton.cdiv(context, block_size)
    nblocks = batch * blocks_per_seq
    shape = (nblocks, block_size, kv_heads, head_dim)
    k = torch.randint(-64, 65, shape, device="cuda", dtype=torch.int8)
    v = torch.randint_like(k, low=-64, high=65)
    qlen = 1 if mode == "decode" else query_length
    q = torch.randn((batch * qlen, query_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    table = torch.arange(nblocks, device="cuda", dtype=torch.int32).reshape(batch, blocks_per_seq)
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * qlen
    cu_k = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * context
    ks = torch.rand((nblocks, block_size, kv_heads), device="cuda") * 0.015 + 0.005
    vs = torch.rand_like(ks) * 0.015 + 0.005
    ks2 = ks.repeat_interleave(2, dim=-1)
    vs2 = vs.repeat_interleave(2, dim=-1)
    return dict(q=q, k=k, v=v, table=table, lengths=lengths, cu_q=cu_q, cu_k=cu_k,
                ks=ks, vs=vs, ks2=ks2, vs2=vs2, batch=batch, context=context,
                mode=mode, query_length=qlen)


def launch(kernel, data, output, name):
    q, k = data["q"], data["k"]
    groups = 1 if name == "single" else 2
    decode = data["mode"] == "decode"
    grid = (1 if decode else triton.cdiv(data["query_length"], 16),
            data["batch"], 8 if decode else 16)
    return kernel[grid](
        q, k, data["v"], data["ks"] if groups == 1 else data["ks2"],
        data["vs"] if groups == 1 else data["vs2"], output,
        data["table"], q if decode else data["cu_q"],
        data["lengths"] if decode else data["cu_k"],
        q.stride(0), q.stride(1), output.stride(0), output.stride(1),
        data["table"].stride(0), 256, 8, groups, 128, 2, 128 ** -0.5,
        decode, 16, 64, 128, num_warps=4,
    )


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def run_case(kernels, data, args, outdir):
    outputs, compiled, graphs = {}, {}, {}
    names = list(kernels)
    for name in names:
        outputs[name] = torch.empty_like(data["q"])
        for _ in range(5):
            compiled[name] = launch(kernels[name], data, outputs[name], name)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.calls_per_graph):
                launch(kernels[name], data, outputs[name], name)
        graphs[name] = graph
    for name in names:
        torch.testing.assert_close(outputs[name], outputs["single"], rtol=0, atol=0)
    times = {name: [] for name in names}
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            # Warm the selected graph before its timed interval.
            graphs[name].replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / (args.replays * args.calls_per_graph))
    row = {key: data[key] for key in ("batch", "context", "mode", "query_length")}
    row["variants"] = {}
    for name in names:
        kernel = compiled[name]
        metadata = dict(registers=kernel.n_regs, spills=kernel.n_spills,
                        shared_bytes=kernel.metadata.shared)
        row["variants"][name] = dict(median_us=statistics.median(times[name]),
                                     p10_us=percentile(times[name], 0.1),
                                     p90_us=percentile(times[name], 0.9),
                                     round_us=times[name], **metadata)
        print(f'{data["mode"]:7} B={data["batch"]:<3} K={data["context"]:<4} '
              f'Q={data["query_length"]:<3} {name:19} '
              f'{statistics.median(times[name]):9.3f} us '
              f'p10/p90={percentile(times[name], .1):.3f}/{percentile(times[name], .9):.3f} '
              f'regs={kernel.n_regs} shared={kernel.metadata.shared} spills={kernel.n_spills}', flush=True)
        if data["mode"] == "decode" and data["batch"] == 256 and data["context"] == 1024:
            for extension in ("ptx", "ttgir", "llir", "cubin"):
                content = kernel.asm[extension]
                path = outdir / f"{name}.{extension}"
                path.write_bytes(content) if isinstance(content, bytes) else path.write_text(content)
    return row


def validate_distinct_scales(kernels, args):
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    for mode in ("decode", "prefill"):
        data = inputs(3, 512, mode, 23, args.seed)
        data["ks2"][..., 1::2] *= 0.3
        data["vs2"][..., 1::2] *= 1.8
        shape = data["k"].shape
        def restore(cache, scales):
            return (cache.float().reshape(*shape[:-1], 2, 64)
                    * scales.reshape(*shape[:-1], 2, 1)).reshape(shape).bfloat16()
        k, v = restore(data["k"], data["ks2"]), restore(data["v"], data["vs2"])
        q = data["q"]
        if mode == "decode":
            expected = flash_attn_with_kvcache(q[:, None], k, v, cache_seqlens=data["lengths"],
                                               block_table=data["table"], softmax_scale=128 ** -0.5,
                                               causal=True)[:, 0]
        else:
            expected = flash_attn_varlen_func(q, k, v, cu_seqlens_q=data["cu_q"],
                                              cu_seqlens_k=data["cu_k"], max_seqlen_q=23,
                                              max_seqlen_k=512, block_table=data["table"],
                                              softmax_scale=128 ** -0.5, causal=True)
        for name in ("dual", "dual_packed"):
            if name not in kernels:
                continue
            output = torch.empty_like(q)
            launch(kernels[name], data, output, name)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
        print(f"Distinct half-scale correctness ({mode}): PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "profiling/attention_scales_2026-10-04")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    kernels, source_hash = make_variants(args.outdir)
    if args.variants:
        kernels = {name: kernels[name] for name in args.variants}
    if not args.profile:
        assert "single" in kernels, "timing requires the single-scale reference"
    if args.profile:
        data = inputs(256, 1024, "decode", 1, args.seed)
        output = torch.empty_like(data["q"])
        for name, kernel in kernels.items():
            for _ in range(3):
                launch(kernel, data, output, name)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for name, kernel in kernels.items():
            torch.cuda.nvtx.range_push(name)
            launch(kernel, data, output, name)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        return
    validate_distinct_scales(kernels, args)
    cases = [(256, 1024, "decode", 1)] if args.quick else [
        (1, 1024, "decode", 1), (64, 512, "decode", 1),
        (64, 1024, "decode", 1), (256, 512, "decode", 1),
        (256, 1024, "decode", 1), (64, 4096, "decode", 1),
        (1, 1024, "prefill", 128), (8, 1024, "prefill", 128),
    ]
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  triton=triton.__version__, cuda=torch.version.cuda,
                  source_sha256=source_hash, seed=args.seed, rounds=args.rounds,
                  replays=args.replays, calls_per_graph=args.calls_per_graph,
                  head_dim=128, query_heads=16, kv_heads=8, BM=16, BN=64, BD=128,
                  measurement="CUDA Event / batched CUDA Graph; kernel only; identical restored KV",
                  cases=[])
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    for batch, context, mode, qlen in cases:
        data = inputs(batch, context, mode, qlen, args.seed)
        result["cases"].append(run_case(kernels, data, args, args.outdir))
        del data
        torch.cuda.empty_cache()
    (args.outdir / "timings.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
