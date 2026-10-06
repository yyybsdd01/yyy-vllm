"""CUDA Graph timing of one Qwen3-0.6B decode attention layer."""

import argparse
import statistics

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.quantized_attention import int8_paged_attention


def capture(fn):
    for _ in range(3):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure(batch, context, rounds, replays, random_data):
    block_size, kv_heads, query_heads, head_dim = 256, 8, 16, 128
    num_blocks = batch * (context // block_size)
    shape = (num_blocks, block_size, kv_heads, head_dim)
    table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).reshape(batch, -1)
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    query = torch.randn((batch, query_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    if random_data:
        bf16_k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        bf16_v = torch.randn_like(bf16_k)
        int8_k = torch.randint(-64, 65, shape, device="cuda", dtype=torch.int8)
        int8_v = torch.randint_like(int8_k, low=-64, high=65)
    else:
        bf16_k = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        bf16_v = torch.zeros_like(bf16_k)
        int8_k = torch.zeros(shape, device="cuda", dtype=torch.int8)
        int8_v = torch.zeros_like(int8_k)
    scale = head_dim ** -0.5

    functions = {
        "auto": lambda: flash_attn_with_kvcache(
            query[:, None], bf16_k, bf16_v, cache_seqlens=lengths,
            block_table=table, softmax_scale=scale, causal=True,
        )[:, 0]
    }
    for groups, name in ((1, "int8"), (2, "int8_half")):
        ks = torch.full((num_blocks, block_size, kv_heads * groups), 0.01, device="cuda")
        vs = torch.full_like(ks, 0.01)
        functions[name] = lambda ks=ks, vs=vs: int8_paged_attention(
            query, int8_k, int8_v, ks, vs, table, scale, context_lens=lengths,
        )

    graphs = {}
    for name, fn in functions.items():
        graph, output = capture(fn)
        if random_data:
            assert torch.isfinite(output).all()
        else:
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        graphs[name] = graph

    times = {name: [] for name in graphs}
    names = tuple(graphs)
    for round_index in range(rounds):
        order = names[round_index % len(names):] + names[:round_index % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / replays)
    for name in names:
        print(f"batch={batch:<3} context={context:<4} mode={name:<9} "
              f"median={statistics.median(times[name]):.2f} us "
              f"rounds={[round(value, 2) for value in times[name]]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=(64, 256))
    parser.add_argument("--contexts", type=int, nargs="+", default=(512, 1024))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--random-data", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name()} | CUDA Graph kernel timing "
          f"| random_data={args.random_data}", flush=True)
    for batch in args.batches:
        for context in args.contexts:
            assert batch > 0 and context > 0 and context % 256 == 0
            measure(batch, context, args.rounds, args.replays, args.random_data)


if __name__ == "__main__":
    main()
