"""Time zero-prefix prefill attention separately from the full model and compilation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import torch
from flash_attn import flash_attn_varlen_func

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.compare_scale_layouts import prepare, paths


def benchmark(batch, length, args):
    data = prepare(batch, length, "prefill", length, args.seed)
    # Match the Q/K/V views produced by the model's fused QKV projection.
    total = batch * length
    qkv = torch.randn((total, 32, 128), device="cuda", dtype=torch.bfloat16)
    data["q"], data["key"], data["value"] = qkv[:, :16], qkv[:, 16:24], qkv[:, 24:]
    blocks = data["ki"].shape[0]
    data["table"] = torch.randperm(blocks, device="cuda", dtype=torch.int32).reshape(batch, -1)
    logical = torch.arange(length, device="cuda", dtype=torch.int32)
    data["slots"] = (data["table"][:, logical//256] * 256 + logical[None, :] % 256).flatten()
    operations, compiled, outputs = paths(data)
    for layout in ("token_head", "head_token"):
        operations[(layout, "write")]()
    torch.testing.assert_close(data["ki"], data["kh"], rtol=0, atol=0)
    torch.testing.assert_close(data["vi"], data["vh"], rtol=0, atol=0)

    restored = torch.empty_like(qkv)
    def restore(cache, scales):
        paged = (cache.float().reshape(blocks, 256, 8, 2, 64)
                 * scales.reshape(blocks, 256, 8, 2, 1)).reshape(blocks, 256, 8, 128)
        return paged[data["table"].long()].reshape(total, 8, 128).bfloat16()
    restored[:, 16:24] = restore(data["ki"], data["ks"])
    restored[:, 24:] = restore(data["vi"], data["vs"])
    def flash(key, value, name):
        outputs[name] = flash_attn_varlen_func(
            data["q"], key, value, cu_seqlens_q=data["cuq"], cu_seqlens_k=data["cuk"],
            max_seqlen_q=length, max_seqlen_k=length, softmax_scale=128**-.5, causal=True,
        )
    functions = dict(
        flash_bf16=lambda: flash(data["key"], data["value"], "flash_bf16"),
        flash_restored=lambda: flash(restored[:, 16:24], restored[:, 24:], "flash_restored"),
        int8_token_head=operations[("token_head", "attention")],
        int8_head_token=operations[("head_token", "attention")],
    )
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
    torch.testing.assert_close(outputs["token_head"], outputs["head_token"], rtol=0, atol=0)
    torch.testing.assert_close(outputs["token_head"], outputs["flash_restored"], rtol=.02, atol=.005)
    times = {name: [] for name in functions}
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            graphs[name].replay()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end)*1000/(args.calls_per_graph*args.replays))
    row = dict(batch=batch, length=length, prefix=0, q_stride=list(data["q"].stride()),
               validation="PASS", max_abs_error_restored=(outputs["token_head"].float()
                   - outputs["flash_restored"].float()).abs().max().item(), variants={})
    for name, samples in times.items():
        values = dict(median_us=statistics.median(samples), round_us=samples)
        if name.startswith("int8"):
            layout = name.removeprefix("int8_")
            kernel = compiled[(layout, "attention")]
            values.update(registers=kernel.n_regs, shared_bytes=kernel.metadata.shared, spills=kernel.n_spills)
        row["variants"][name] = values
        print(f"B={batch:<2} L={length:<4} {name:<16} {values['median_us']:9.3f} us", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--calls-per-graph", type=int, default=8)
    parser.add_argument("--replays", type=int, default=10)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    files = ["nanovllm/layers/attention.py", "nanovllm/layers/quantized_attention.py",
             "nanovllm/layers/head_major_scale_attention.py"]
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, seed=args.seed,
                  source_hashes=source_hashes, q_heads=16, kv_heads=8, dim=128,
                  block_size=256, scattered_pages=True, rounds=args.rounds,
                  calls_per_graph=args.calls_per_graph, replays=args.replays, cases=[])
    for batch, length in ((1,256), (1,1024), (16,1024), (32,512)):
        torch.manual_seed(args.seed)
        result["cases"].append(benchmark(batch, length, args))
        (args.outdir/"timings.json").write_text(json.dumps(result,indent=2)+"\n")
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
                   for name, sha in source_hashes.items())
        torch.cuda.empty_cache()
    lines = ["# 零前缀 prefill attention GPU 时间", "",
        "CUDA Graph + CUDA Event，11 轮随机交错的轮均值中位数；排除写入、编译、模型其他层和 CPU 开销。",
        "Q/K/V 为模型同款 stride=4096 的 BF16 QKV 切片，非连续物理页，BM=16、BN=64、BD=128、4 warps。",
        "Flash BF16 使用原始 K/V；Flash restored 使用恢复成 BF16 的量化 K/V。自写两版输出逐元素一致，并与 restored 参考通过 rtol=0.02、atol=0.005。", "",
        "| B | L | Flash BF16 μs | Flash restored μs | INT8 原布局 μs | INT8 head/token μs |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["cases"]:
        values = [row["variants"][name]["median_us"] for name in
                  ("flash_bf16", "flash_restored", "int8_token_head", "int8_head_token")]
        lines.append(f'| {row["batch"]} | {row["length"]} | '+" | ".join(f"{v:.3f}" for v in values)+" |")
    (args.outdir/"report.md").write_text("\n".join(lines)+"\n")
    print("COMPLETE", args.outdir/"report.md", flush=True)


if __name__ == "__main__":
    main()
