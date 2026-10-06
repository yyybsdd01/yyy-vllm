"""Compare unchanged token/head scales with independent head/token kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanovllm.layers.attention import store_kvcache_int8_kernel
from nanovllm.layers.quantized_attention import _int8_paged_attention_kernel
from nanovllm.layers.head_major_scale_attention import (
    store_kvcache_int8_head_major_kernel, _int8_paged_attention_head_major_kernel,
)
from profiling.compare_attention_kernel_formats import make_data


def prepare(batch, context, mode, qlen, seed):
    data = make_data(batch, context, mode, qlen, seed)
    del data["kb"], data["vb"]
    blocks, block_size, heads, dim = data["ki"].shape
    data["kh"], data["vh"] = data["ki"].clone(), data["vi"].clone()
    data["ksh"] = data["ks"].reshape(blocks, block_size, heads, 2).permute(0, 2, 1, 3).contiguous()
    data["vsh"] = data["vs"].reshape(blocks, block_size, heads, 2).permute(0, 2, 1, 3).contiguous()
    qlen = data["qlen"]
    # Each sequence writes its new suffix into its own physical pages.
    data["slots"] = (torch.arange(batch, device="cuda", dtype=torch.int32)[:, None] * context
                     + torch.arange(context-qlen, context, device="cuda", dtype=torch.int32)[None, :]).flatten()
    data["key"] = torch.randn((batch*qlen, heads, dim), device="cuda", dtype=torch.bfloat16)
    data["value"] = torch.randn_like(data["key"])
    data["key"][..., dim//2:] *= .2
    data["value"][..., :dim//2] *= .3
    return data


def paths(data):
    functions, compiled, outputs = {}, {}, {}
    q = data["q"]
    decode = data["mode"] == "decode"
    for layout in ("token_head", "head_token"):
        head_major = layout == "head_token"
        k = data["kh"] if head_major else data["ki"]
        v = data["vh"] if head_major else data["vi"]
        ks = data["ksh"] if head_major else data["ks"]
        vs = data["vsh"] if head_major else data["vs"]
        writer = store_kvcache_int8_head_major_kernel if head_major else store_kvcache_int8_kernel
        attention = _int8_paged_attention_head_major_kernel if head_major else _int8_paged_attention_kernel
        outputs[layout] = torch.empty_like(q)
        def write(writer=writer, k=k, v=v, ks=ks, vs=vs, layout=layout, head_major=head_major):
            arguments = [data["key"], data["key"].stride(0), data["value"], data["value"].stride(0),
                         k, v, ks, vs, data["slots"], 8, 128, 2]
            if head_major:
                arguments.append(256)
            arguments += [8, 128]
            compiled[(layout, "write")] = writer[(data["slots"].numel(),)](*arguments, num_warps=4)
        def attend(attention=attention, k=k, v=v, ks=ks, vs=vs, layout=layout):
            output = outputs[layout]
            compiled[(layout, "attention")] = attention[
                (1 if decode else triton.cdiv(data["qlen"], 16), data["batch"], 8 if decode else 16)
            ](
                q, k, v, ks, vs, output, data["table"],
                q if decode else data["cuq"], data["lengths"] if decode else data["cuk"],
                q.stride(0), q.stride(1), output.stride(0), output.stride(1),
                data["table"].stride(0), 256, 8, 2, 128, 2, 128**-.5, decode,
                16, 64, 128, num_warps=4,
            )
        def together(write=write, attend=attend):
            write()
            attend()
        functions[(layout, "write")] = write
        functions[(layout, "attention")] = attend
        functions[(layout, "write_attention")] = together
    return functions, compiled, outputs


def benchmark(data, args):
    functions, compiled, outputs = paths(data)
    graphs = {}
    for name, fn in functions.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.calls_per_graph):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        graphs[name] = graph
    torch.testing.assert_close(data["ki"], data["kh"], rtol=0, atol=0)
    torch.testing.assert_close(data["vi"], data["vh"], rtol=0, atol=0)
    for old, new in (("ks", "ksh"), ("vs", "vsh")):
        torch.testing.assert_close(data[old].reshape(-1, 256, 8, 2).permute(0, 2, 1, 3),
                                   data[new], rtol=0, atol=0)
    torch.testing.assert_close(outputs["token_head"], outputs["head_token"], rtol=0, atol=0)
    times = {name: [] for name in functions}
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            graphs[name].replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end)*1000/(args.calls_per_graph*args.replays))
    row = {key: data[key] for key in ("batch", "context", "mode", "qlen")}
    row["new_tokens"] = data["slots"].numel()
    row["variants"] = {}
    for layout in ("token_head", "head_token"):
        row["variants"][layout] = {}
        for operation in ("write", "attention", "write_attention"):
            samples = times[(layout, operation)]
            values = dict(median_us=statistics.median(samples), min_round_us=min(samples),
                          max_round_us=max(samples), round_us=samples)
            if operation != "write_attention":
                kernel = compiled[(layout, operation)]
                values.update(registers=kernel.n_regs, spills=kernel.n_spills,
                              shared_bytes=kernel.metadata.shared)
                if data["batch"] == 256 and data["context"] == 1024 and data["mode"] == "decode":
                    for ext in ("ptx", "ttgir", "cubin"):
                        content = kernel.asm[ext]
                        target = args.outdir/f"{layout}_{operation}.{ext}"
                        target.write_bytes(content) if isinstance(content, bytes) else target.write_text(content)
            row["variants"][layout][operation] = values
        v = row["variants"][layout]
        print(f'{data["mode"]:7} B={data["batch"]:<3} K={data["context"]:<4} Q={data["qlen"]:<3} '
              f'{layout:10} write={v["write"]["median_us"]:8.3f} us '
              f'attention={v["attention"]["median_us"]:9.3f} us '
              f'combined={v["write_attention"]["median_us"]:9.3f} us', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT/"profiling/scale_layout_2026-10-04")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    files = ["nanovllm/layers/attention.py", "nanovllm/layers/quantized_attention.py",
             "nanovllm/engine/model_runner.py", "nanovllm/layers/head_major_scale_attention.py"]
    hashes = {file: hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in files}
    if args.profile:
        data = prepare(256, 1024, "decode", 1, args.seed)
        functions, _, _ = paths(data)
        for fn in functions.values():
            for _ in range(5):
                fn()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for operation in ("write", "attention"):
            for layout in ("token_head", "head_token"):
                torch.cuda.nvtx.range_push(f"{layout}_{operation}")
                functions[(layout, operation)]()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        return
    cases = [(256, 1024, "decode", 1)] if args.quick else [
        (1, 1024, "decode", 1), (64, 512, "decode", 1), (64, 1024, "decode", 1),
        (256, 512, "decode", 1), (256, 1024, "decode", 1), (64, 4096, "decode", 1),
        (1, 1024, "prefill", 128), (8, 1024, "prefill", 128)]
    results = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                   triton=triton.__version__, source_hashes=hashes, seed=args.seed,
                   rounds=args.rounds, calls_per_graph=args.calls_per_graph,
                   replays=args.replays, block_size=256, query_heads=16, kv_heads=8,
                   head_dim=128, BM=16, BN=64, BD=128, cases=[])
    for case in cases:
        data = prepare(*case, args.seed)
        results["cases"].append(benchmark(data, args))
        del data
        torch.cuda.empty_cache()
        (args.outdir/"timings.json").write_text(json.dumps(results, indent=2)+"\n")
        assert all(hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == sha
                   for file, sha in hashes.items())
    print("COMPLETE", args.outdir, flush=True)


if __name__ == "__main__":
    main()
