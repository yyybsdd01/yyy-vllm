"""Compare original FlashAttention, BF16 Triton control, and current INT8 KV.

All paths see logically identical reconstructed KV. KV construction/restoration,
Python, cache writes, other model layers, and scheduling are outside GPU timing.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys

import torch
import triton
import flash_attn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanovllm.layers.quantized_attention import int8_paged_attention


def bf16_control(outdir):
    path = ROOT / "nanovllm/layers/quantized_attention.py"
    source = path.read_text()
    (outdir / "int8_production_snapshot.py").write_text(source)
    kernel = source[:source.index("\ndef int8_paged_attention(")]
    start = kernel.index("        if SCALE_GROUPS == 1:")
    end = kernel.index("        # 从 INT8 KV cache", start)
    kernel = kernel[:start] + kernel[end:]
    kernel = kernel.replace("k = (k.to(tl.float32) * k_scale).to(q.dtype)", "k = k.to(q.dtype)")
    kernel = kernel.replace("v = (v.to(tl.float32) * v_scale).to(q.dtype)", "v = v.to(q.dtype)")
    kernel = kernel.replace("_int8_paged_attention_kernel", "_bf16_paged_attention_control")
    target = outdir / "bf16_control.py"
    target.write_text(kernel)
    spec = importlib.util.spec_from_file_location("bf16_control", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bf16_control"] = module
    spec.loader.exec_module(module)
    return module._bf16_paged_attention_control, hashlib.sha256(source.encode()).hexdigest()


def make_data(batch, context, mode, qlen, seed):
    torch.manual_seed(seed)
    qlen = 1 if mode == "decode" else qlen
    blocks = batch * triton.cdiv(context, 256)
    shape = (blocks, 256, 8, 128)
    ki = torch.randint(-64, 65, shape, dtype=torch.int8, device="cuda")
    vi = torch.randint_like(ki, low=-64, high=65)
    ks = torch.rand((blocks, 256, 16), device="cuda") * .015 + .005
    vs = torch.rand_like(ks) * .015 + .005
    def restore(cache, scales):
        return (cache.float().reshape(blocks, 256, 8, 2, 64)
                * scales.reshape(blocks, 256, 8, 2, 1)).reshape(shape).bfloat16()
    kb, vb = restore(ki, ks), restore(vi, vs)
    q = torch.randn((batch * qlen, 16, 128), dtype=torch.bfloat16, device="cuda")
    table = torch.arange(blocks, dtype=torch.int32, device="cuda").reshape(batch, -1)
    lengths = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    cuq = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * qlen
    cuk = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * context
    return dict(q=q, ki=ki, vi=vi, ks=ks, vs=vs, kb=kb, vb=vb, table=table,
                lengths=lengths, cuq=cuq, cuk=cuk, batch=batch, context=context,
                mode=mode, qlen=qlen)


def functions(control, data):
    q = data["q"]
    decode = data["mode"] == "decode"
    kwargs = dict(context_lens=data["lengths"]) if decode else dict(
        cu_seqlens_q=data["cuq"], cu_seqlens_k=data["cuk"], max_seqlen_q=data["qlen"])
    if decode:
        original = lambda: flash_attn_with_kvcache(
            q[:, None], data["kb"], data["vb"], cache_seqlens=data["lengths"],
            block_table=data["table"], softmax_scale=128 ** -.5, causal=True)[:, 0]
    else:
        original = lambda: flash_attn_varlen_func(
            q, data["kb"], data["vb"], cu_seqlens_q=data["cuq"], cu_seqlens_k=data["cuk"],
            max_seqlen_q=data["qlen"], max_seqlen_k=data["context"], block_table=data["table"],
            softmax_scale=128 ** -.5, causal=True)
    production = lambda: int8_paged_attention(
        q, data["ki"], data["vi"], data["ks"], data["vs"], data["table"], 128 ** -.5, **kwargs)
    output = torch.empty_like(q)
    def triton_bf16():
        grid = (1 if decode else triton.cdiv(data["qlen"], 16), data["batch"], 8 if decode else 16)
        compiled = control[grid](
            q, data["kb"], data["vb"], data["ks"], data["vs"], output,
            data["table"], q if decode else data["cuq"], data["lengths"] if decode else data["cuk"],
            q.stride(0), q.stride(1), output.stride(0), output.stride(1),
            data["table"].stride(0), 256, 8, 2, 128, 2, 128 ** -.5, decode,
            16, 64, 128, num_warps=4)
        triton_bf16.compiled = compiled
        return output
    return {"flash_bf16": original, "triton_bf16": triton_bf16, "triton_int8_half": production}


def run_case(control, data, args):
    fns = functions(control, data)
    outputs, graphs = {}, {}
    for name, fn in fns.items():
        for _ in range(5):
            outputs[name] = fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.calls_per_graph):
                outputs[name] = fn()
        graphs[name] = graph
        graph.replay()
        torch.cuda.synchronize()
    errors = {}
    for name, value in outputs.items():
        torch.testing.assert_close(value, outputs["flash_bf16"], rtol=.02, atol=.005)
        errors[name] = (value.float() - outputs["flash_bf16"].float()).abs().max().item()
    kernels = {}
    for name, fn in fns.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernels[name] = [dict(name=e.name, duration_us=e.self_device_time_total) for e in events]
    times = {name: [] for name in fns}
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(fns)
        rng.shuffle(order)
        for name in order:
            graphs[name].replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / (args.replays * args.calls_per_graph))
    row = {k: data[k] for k in ("batch", "context", "mode", "qlen")}
    nslots = data["batch"] * data["context"]
    row["logical_bf16_KV_MiB"] = 2 * nslots * 8 * 128 * 2 / 2**20
    row["logical_int8_KV_scales_MiB"] = (2 * nslots * 8 * 128 + 2 * nslots * 8 * 2 * 4) / 2**20
    row["variants"] = {}
    for name in fns:
        samples = times[name]
        row["variants"][name] = dict(median_us=statistics.median(samples),
                                     min_round_us=min(samples), max_round_us=max(samples),
                                     round_us=samples, kernel_count=len(kernels[name]),
                                     kernels=kernels[name], correctness_max_abs=errors[name])
        print(f'{data["mode"]:7} B={data["batch"]:<3} K={data["context"]:<4} Q={data["qlen"]:<3} '
              f'{name:17} {statistics.median(samples):9.3f} us '
              f'kernels={len(kernels[name])} max_error={errors[name]:.6f}', flush=True)
    compiled = fns["triton_bf16"].compiled
    row["bf16_control_resources"] = dict(registers=compiled.n_regs, spills=compiled.n_spills,
                                         shared_bytes=compiled.metadata.shared)
    if data["mode"] == "decode" and data["batch"] == 256 and data["context"] == 1024:
        for ext in ("ptx", "ttgir", "cubin"):
            content = compiled.asm[ext]
            path = args.outdir / f"triton_bf16.{ext}"
            path.write_bytes(content) if isinstance(content, bytes) else path.write_text(content)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    control, source_hash = bf16_control(args.outdir)
    if args.profile:
        data = make_data(256, 1024, "decode", 1, args.seed)
        fns = functions(control, data)
        for fn in fns.values():
            for _ in range(5):
                fn()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for name, fn in fns.items():
            torch.cuda.nvtx.range_push(name)
            fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        return
    cases = [(256, 1024, "decode", 1)] if args.quick else [
        (1, 1024, "decode", 1), (64, 512, "decode", 1), (64, 1024, "decode", 1),
        (256, 512, "decode", 1), (256, 1024, "decode", 1), (64, 4096, "decode", 1),
        (1, 1024, "prefill", 128), (8, 1024, "prefill", 128)]
    result = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  triton=triton.__version__, flash_attn=flash_attn.__version__,
                  cuda=torch.version.cuda, seed=args.seed,
                  source_sha256=source_hash, rounds=args.rounds, replays=args.replays,
                  calls_per_graph=args.calls_per_graph, BM=16, BN=64, BD=128,
                  query_heads=16, kv_heads=8, head_dim=128, cases=[])
    for batch, context, mode, qlen in cases:
        data = make_data(batch, context, mode, qlen, args.seed)
        result["cases"].append(run_case(control, data, args))
        del data
        torch.cuda.empty_cache()
        assert hashlib.sha256((ROOT / "nanovllm/layers/quantized_attention.py").read_bytes()).hexdigest() == source_hash
    (args.outdir / "timings.json").write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", args.outdir, flush=True)


if __name__ == "__main__":
    main()
