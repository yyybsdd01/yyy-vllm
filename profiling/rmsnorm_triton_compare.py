"""Compare eager PyTorch, compiled PyTorch, and handwritten Triton RMSNorm.

Run with the existing nanovllm Conda environment from the repository root:
    /home/xgd/anaconda3/envs/nanovllm/bin/python profiling/rmsnorm_triton_compare.py

The eager path calls the original methods via __wrapped__, without torch.compile.
The script checks outputs before measuring. It changes no model code.
"""

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch
import triton
import triton.language as tl
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402
from profiling.rmsnorm import parse_tokens, profile_case  # noqa: E402


@triton.jit
def rmsnorm_kernel(
    X, Residual, Weight, Y, ResidualOut,
    N: tl.constexpr, HEADS: tl.constexpr,
    X_STRIDE_T: tl.constexpr, X_STRIDE_H: tl.constexpr, X_STRIDE_N: tl.constexpr,
    R_STRIDE_T: tl.constexpr, R_STRIDE_N: tl.constexpr,
    EPS: tl.constexpr, ADD_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    col = tl.arange(0, BLOCK)
    mask = col < N

    x = tl.load(X + token * X_STRIDE_T + head * X_STRIDE_H
                + col * X_STRIDE_N, mask, other=0).to(tl.float32)
    if ADD_RESIDUAL:
        residual = tl.load(Residual + token * R_STRIDE_T
                           + col * R_STRIDE_N, mask, other=0).to(tl.float32)
        x = x + residual
        tl.store(ResidualOut + row * N + col, x, mask)

    variance = tl.sum(x * x, 0) / N
    normalized = (x * tl.rsqrt(variance + EPS)).to(Y.dtype.element_ty)
    weight = tl.load(Weight + col, mask, other=0)
    tl.store(Y + row * N + col, normalized * weight, mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                   residual: torch.Tensor | None = None):
    heads = x.shape[1] if x.ndim == 3 else 1
    n = x.shape[-1]
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    residual_out = torch.empty(residual.shape, device=x.device, dtype=x.dtype) if residual is not None else y
    rmsnorm_kernel[(x.shape[0] * heads,)](
        x, residual if residual is not None else x, weight, y, residual_out,
        n, heads,
        x.stride(0), x.stride(1) if x.ndim == 3 else 0, x.stride(-1),
        residual.stride(0) if residual is not None else 0,
        residual.stride(-1) if residual is not None else 0,
        eps, residual is not None, triton.next_power_of_2(n), num_warps=4,
    )
    return (y, residual_out) if residual is not None else y


def make_case(config, tokens: int, kind: str, device: torch.device):
    dtype = config.dtype
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    if kind in ("hidden", "add+hidden"):
        x = torch.randn(tokens, config.hidden_size, device=device, dtype=dtype)
        residual = torch.randn_like(x) if kind == "add+hidden" else None
        n = config.hidden_size
    else:
        q_size = config.num_attention_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim
        qkv = torch.randn(tokens, q_size + 2 * kv_size, device=device, dtype=dtype)
        q, k, _ = qkv.split((q_size, kv_size, kv_size), dim=-1)
        x = (q.view(tokens, config.num_attention_heads, head_dim) if kind == "q"
             else k.view(tokens, config.num_key_value_heads, head_dim))
        residual = None
        n = head_dim

    norm = RMSNorm(n, eps=config.rms_norm_eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn_like(norm.weight))
    eager_method = (RMSNorm.add_rms_forward.__wrapped__ if residual is not None
                    else RMSNorm.rms_forward.__wrapped__)
    eager = lambda: (eager_method(norm, x, residual) if residual is not None
                     else eager_method(norm, x))
    compiled = lambda: norm(x, residual) if residual is not None else norm(x)
    handwritten = lambda: triton_rmsnorm(x, norm.weight, norm.eps, residual)
    return tuple(x.shape), {"eager": eager, "compiled": compiled,
                            "triton": handwritten}


def check_outputs(funcs) -> float:
    reference = funcs["eager"]()
    torch.cuda.synchronize()
    reference = reference if isinstance(reference, tuple) else (reference,)
    max_error = 0.0
    for name in ("compiled", "triton"):
        actual = funcs[name]()
        torch.cuda.synchronize()
        actual = actual if isinstance(actual, tuple) else (actual,)
        if len(reference) != len(actual):
            raise AssertionError(f"{name}: different numbers of outputs")
        for expected, found in zip(reference, actual):
            if expected.shape != found.shape or expected.dtype != found.dtype:
                raise AssertionError(f"{name}: output shape or dtype differs")
            torch.testing.assert_close(found, expected, rtol=0.02, atol=0.02)
            max_error = max(max_error, (found.float() - expected.float()).abs().max().item())
    return max_error


def capture_graph(fn, operations: int) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(operations):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_eager(fn, iterations: int) -> float:
    torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1e6 / iterations


def time_graph(graph: torch.cuda.CUDAGraph, operations: int, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (operations * replays)


def compare(funcs, args):
    for _ in range(args.warmup):
        for fn in funcs.values():
            fn()
    torch.cuda.synchronize()

    eager = {name: [] for name in funcs}
    names = tuple(funcs)
    for repetition in range(args.repeats):
        order = names[repetition % len(names):] + names[:repetition % len(names)]
        for name in order:
            eager[name].append(time_eager(funcs[name], args.eager_iters))

    graphs = {name: capture_graph(fn, args.graph_ops) for name, fn in funcs.items()}
    graph_times = {name: [] for name in funcs}
    for repetition in range(args.repeats):
        order = names[repetition % len(names):] + names[:repetition % len(names)]
        for name in order:
            graph_times[name].append(time_graph(graphs[name], args.graph_ops, args.replays))
    return {name: (statistics.median(eager[name]), statistics.median(graph_times[name]))
            for name in funcs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=parse_tokens, default=[1, 32, 256, 1024])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--eager-iters", type=int, default=100)
    parser.add_argument("--graph-ops", type=int, default=32)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-kernels", action="store_true",
                        help="also print a torch.profiler CUDA-kernel comparison table")
    args = parser.parse_args()
    if min(args.warmup, args.eager_iters, args.graph_ops, args.replays, args.repeats) <= 0:
        parser.error("all iteration counts must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.device}")
    config = AutoConfig.from_pretrained(args.model)
    print(f"GPU: {torch.cuda.get_device_name(device)} | torch: {torch.__version__} | "
          f"Triton: {triton.__version__} | dtype: {config.dtype}")
    print(f"Model: {args.model} | repeats: {args.repeats} | correctness: rtol=atol=0.02")
    print("Eager = original undecorated PyTorch method; compiled = project @torch.compile method.")
    print("All times are us/call. Speedup = uncompiled eager / implementation; >1 is faster.")
    print(f"{'Op':<12} {'Input shape':<21} {'Max diff':>9} "
          f"{'Eager loop: eager/compiled/Triton':>36} "
          f"{'CUDA Graph: eager/compiled/Triton':>36}")

    rows = []
    profiler_rows = []
    with torch.inference_mode():
        for tokens in args.tokens:
            for kind in ("hidden", "add+hidden", "q", "k"):
                shape, funcs = make_case(config, tokens, kind, device)
                max_diff = check_outputs(funcs)
                timing = compare(funcs, args)
                rows.append((kind, str(shape), max_diff, timing))
                eager_times = "/".join(f"{timing[name][0]:.2f}" for name in funcs)
                graph_times = "/".join(f"{timing[name][1]:.2f}" for name in funcs)
                print(f"{kind:<12} {str(shape):<21} {max_diff:>9.5f} "
                      f"{eager_times:>36} {graph_times:>36}", flush=True)

                if args.profile_kernels:
                    for name, fn in funcs.items():
                        _, cuda_total, launches, kernel_name, _ = profile_case(
                            fn, args.warmup, 30)
                        profiler_rows.append((kind, str(shape), name, kernel_name,
                                              launches / 30, cuda_total / 30))

    print("\nSpeedup over uncompiled eager (eager loop / CUDA Graph):")
    print(f"{'Op':<12} {'Input shape':<21} {'Compiled':>18} {'Triton':>18}")
    for kind, shape, _, timing in rows:
        eager_loop, eager_graph = timing["eager"]
        compiled_loop, compiled_graph = timing["compiled"]
        triton_loop, triton_graph = timing["triton"]
        print(f"{kind:<12} {shape:<21} "
              f"{eager_loop / compiled_loop:>7.2f}x / {eager_graph / compiled_graph:>5.2f}x "
              f"{eager_loop / triton_loop:>7.2f}x / {eager_graph / triton_graph:>5.2f}x")

    if profiler_rows:
        print("\ntorch.profiler CUDA kernel table (30 calls per implementation):")
        print(f"{'Op':<12} {'Input shape':<21} {'Implementation':<12} "
              f"{'Top CUDA kernel':<40} {'Kernels/call':>12} {'CUDA us/call':>12}")
        for kind, shape, name, kernel_name, launches, cuda_us in profiler_rows:
            print(f"{kind:<12} {shape:<21} {name:<12} "
                  f"{kernel_name[:40]:<40} {launches:>12.1f} {cuda_us:>12.2f}")


if __name__ == "__main__":
    main()
