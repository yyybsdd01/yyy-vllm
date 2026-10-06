"""Benchmark the RMSNorm calls used by this project's Qwen3 model.

Run from the repository root with the nanovllm Conda environment:
    /home/xgd/anaconda3/envs/nanovllm/bin/python profiling/rmsnorm.py
    /home/xgd/anaconda3/envs/nanovllm/bin/python profiling/rmsnorm.py --mode benchmark

This script does not modify the model implementation and prints plain terminal
tables. Profiler CPU times include instrumentation overhead; use benchmark mode
for latency comparisons.
"""

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402


def parse_tokens(value: str) -> list[int]:
    values = [int(piece) for piece in value.split(",")]
    if not values or any(token_count <= 0 for token_count in values):
        raise argparse.ArgumentTypeError("--tokens needs positive comma-separated integers")
    return values


def make_case(config, token_count: int, kind: str, device: torch.device):
    dtype = config.dtype
    eps = config.rms_norm_eps
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads

    if kind in ("hidden", "add+hidden"):
        norm = RMSNorm(config.hidden_size, eps=eps).to(device=device, dtype=dtype)
        x = torch.randn(token_count, config.hidden_size, device=device, dtype=dtype)
        if kind == "hidden":
            return kind, (token_count, config.hidden_size), lambda: norm(x)
        residual = torch.randn_like(x)
        return kind, (token_count, config.hidden_size), lambda: norm(x, residual)

    # Q and K in Qwen3Attention are views of the packed QKV projection output.
    # Retaining their original row stride matches the project's actual inputs.
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    qkv = torch.randn(token_count, q_size + 2 * kv_size, device=device, dtype=dtype)
    q, k, _ = qkv.split((q_size, kv_size, kv_size), dim=-1)
    if kind == "q":
        x = q.view(token_count, q_heads, head_dim)
    else:
        x = k.view(token_count, kv_heads, head_dim)
    norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    return kind, tuple(x.shape), lambda: norm(x)


def benchmark(fn, warmup: int, eager_iters: int, graph_ops: int,
              replays: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    tensors = result if isinstance(result, tuple) else (result,)
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise RuntimeError("RMSNorm returned a non-finite value")

    # Host-visible throughput of repeated normal calls. Compilation and data
    # creation are outside the timed interval; synchronization ends it.
    eager_samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = perf_counter()
        for _ in range(eager_iters):
            fn()
        torch.cuda.synchronize()
        eager_samples.append((perf_counter() - started) * 1e6 / eager_iters)

    # A graph containing several calls amortizes Python and graph-replay launch
    # overhead, giving a useful isolated device-side cost per RMSNorm call.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_ops):
            fn()
    graph.replay()
    torch.cuda.synchronize()

    graph_samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        graph_samples.append(start.elapsed_time(end) * 1000 / (graph_ops * replays))
    return statistics.median(eager_samples), statistics.median(graph_samples)


def profile_case(fn, warmup: int, steps: int) -> tuple[float, float, int, str, str]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    label = "RMSNorm_call"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(steps):
            with record_function(label):
                fn()
        torch.cuda.synchronize()

    events = prof.key_averages()
    cpu_range = next(event for event in events
                     if event.key == label
                     and event.device_type == torch.autograd.DeviceType.CPU)
    # The profiler also emits a CUDA event for record_function(label). It is a
    # range containing kernels, not another kernel; exclude it from this sum.
    kernels = [event for event in events
               if event.device_type == torch.autograd.DeviceType.CUDA
               and event.key != label]
    if not kernels:
        raise RuntimeError("torch.profiler captured no CUDA kernels")
    kernel_total_us = sum(event.self_device_time_total for event in kernels)
    launches = sum(event.count for event in kernels)
    top_kernel = max(kernels, key=lambda event: event.self_device_time_total).key
    compiled_ops = [event for event in events
                    if event.device_type == torch.autograd.DeviceType.CPU
                    and event.key.startswith("triton_")
                    and event.self_device_time_total > 0]
    compiled_name = max(compiled_ops,
                        key=lambda event: event.self_device_time_total).key if compiled_ops else "unknown"
    return cpu_range.cpu_time_total / steps, kernel_total_us, launches, top_kernel, compiled_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "torch-profiler"),
                        default="torch-profiler")
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=parse_tokens, default=[1, 32, 256, 1024])
    parser.add_argument("--profile-tokens", type=parse_tokens, default=[1, 32, 256, 1024])
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--eager-iters", type=int, default=100)
    parser.add_argument("--graph-ops", type=int, default=32)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.warmup, args.eager_iters, args.graph_ops, args.replays,
           args.repeats, args.profile_steps) <= 0:
        parser.error("all iteration counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    device = torch.device(f"cuda:{args.device}")
    config = AutoConfig.from_pretrained(args.model)
    print(f"GPU: {torch.cuda.get_device_name(device)} | torch: {torch.__version__}")
    print(f"Model: {args.model} | dtype: {config.dtype} | eps: {config.rms_norm_eps}")
    with torch.inference_mode():
        if args.mode == "benchmark":
            print("Eager loop and CUDA Graph replay results are microseconds per RMSNorm call.")
            print(f"{'Op':<12} {'Input shape':<21} {'Eager loop':>13} {'Graph device':>15}")
            print(f"{'':<12} {'':<21} {'us/call':>13} {'us/call':>15}")
            for token_count in args.tokens:
                for kind in ("hidden", "add+hidden", "q", "k"):
                    name, shape, fn = make_case(config, token_count, kind, device)
                    eager_us, graph_us = benchmark(
                        fn, args.warmup, args.eager_iters, args.graph_ops,
                        args.replays, args.repeats,
                    )
                    print(f"{name:<12} {str(shape):<21} {eager_us:>13.2f} {graph_us:>15.2f}",
                          flush=True)
        else:
            print(f"torch.profiler: {args.profile_steps} calls/case after {args.warmup} warmup calls")
            print("CPU range includes profiler overhead; CUDA kernel time excludes the enclosing range.")
            print("\nCUDA kernel table:")
            print(f"{'Op':<12} {'Input shape':<21} {'Kernel':<12} {'Calls':>6} "
                  f"{'CUDA total':>11} {'CUDA/call':>11} {'CPU/call':>10}")
            print(f"{'':<12} {'':<21} {'':<12} {'':>6} "
                  f"{'us':>11} {'us':>11} {'us':>10}")
            compiled_names = set()
            for token_count in args.profile_tokens:
                for kind in ("hidden", "add+hidden", "q", "k"):
                    name, shape, fn = make_case(config, token_count, kind, device)
                    cpu_us, kernel_total_us, launches, kernel_name, compiled_name = profile_case(
                        fn, args.warmup, args.profile_steps)
                    compiled_names.add(compiled_name)
                    print(f"{name:<12} {str(shape):<21} {kernel_name:<12} {launches:>6} "
                          f"{kernel_total_us:>11.2f} {kernel_total_us / args.profile_steps:>11.2f} "
                          f"{cpu_us:>10.2f}", flush=True)
            print("Inductor compiled op(s): " + ", ".join(sorted(compiled_names)))


if __name__ == "__main__":
    main()
