"""Compare Triton, eager PyTorch, and compiled PyTorch KV-cache stores.

Example:
    python bench_kvcache_store.py --batch-sizes 1,16,128,1024

Timings are per call, measured with CUDA events after warmup. Both PyTorch
versions use index_copy_; their data-dependent boolean indexing is not
CUDA-Graph-safe. The compiled version gets extra warmup because PyTorch 2.5.1
repeatedly recompiles this particular function before reaching a steady state.
"""

import argparse
import statistics

import torch

from nanovllm.layers.attention import store_kvcache


def store_kvcache_torch(key, value, k_cache, v_cache, slot_mapping):
    """Write each valid token's K/V to its flattened physical cache slot."""
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    k_cache.view(-1, key.size(1), key.size(2)).index_copy_(0, slots, key[valid])  # 先筛选 key
    v_cache.view(-1, value.size(1), value.size(2)).index_copy_(0, slots, value[valid])


@torch.compile
def store_kvcache_torch_compiled(key, value, k_cache, v_cache, slot_mapping):
    return store_kvcache_torch(key, value, k_cache, v_cache, slot_mapping)


def elapsed_ms_per_call(fn, iterations):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark_case(n, args, dtype):
    device = torch.device("cuda", args.device)
    h, d = args.kv_heads, args.head_dim
    total_slots = args.num_blocks * args.block_size

    # QKV projection/split gives the token dimension a stride larger than H*D.
    qkv = torch.randn(n, 3, h, d, dtype=dtype, device=device)
    key, value = qkv[:, 1], qkv[:, 2]
    slots = torch.randperm(total_slots, device=device)[:n].to(torch.int32)
    num_masked = round(n * args.masked_fraction)
    if num_masked:
        slots[-num_masked:] = -1

    # [method, K/V, physical_block, offset, kv_head, head_dim]
    caches = torch.zeros(
        3, 2, args.num_blocks, args.block_size, h, d,
        dtype=dtype, device=device,
    )
    triton_args = (key, value, caches[0, 0], caches[0, 1], slots)
    torch_args = (key, value, caches[1, 0], caches[1, 1], slots)
    compiled_args = (key, value, caches[2, 0], caches[2, 1], slots)
    triton_fn = lambda: store_kvcache(*triton_args)
    torch_fn = lambda: store_kvcache_torch(*torch_args)
    compiled_fn = lambda: store_kvcache_torch_compiled(*compiled_args)
    methods = {"Triton": triton_fn, "PyTorch": torch_fn, "Compiled": compiled_fn}

    # Compare the whole cache, including locations that should remain zero.
    for fn in methods.values():
        fn()
    for i in (1, 2):
        if not torch.equal(caches[0], caches[i]):
            raise AssertionError(f"K/V cache mismatch for N={n}, method={list(methods)[i]}")

    # Check -1 explicitly even when the timed case contains no masked tokens.
    skip_slots = slots.clone()
    skip_slots[0] = -1
    caches.zero_()
    for i, fn in enumerate((store_kvcache, store_kvcache_torch, store_kvcache_torch_compiled)):
        fn(key, value, caches[i, 0], caches[i, 1], skip_slots)
    for i in (1, 2):
        if not torch.equal(caches[0], caches[i]):
            raise AssertionError(f"-1 skip mismatch for N={n}, method={list(methods)[i]}")

    for _ in range(args.compile_warmup):
        compiled_fn()
    for _ in range(args.warmup):
        for fn in methods.values():
            fn()
    torch.cuda.synchronize()

    times = {name: [] for name in methods}
    names = list(methods)
    for repeat in range(args.repeats):
        # Rotate order to reduce systematic clock/temperature bias.
        for name in names[repeat % len(names):] + names[:repeat % len(names)]:
            times[name].append(elapsed_ms_per_call(methods[name], args.iterations))

    return n - num_masked, *(statistics.median(times[name]) for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,16,128,1024")
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--masked-fraction", type=float, default=0.0,
                        help="fraction of slots set to -1 in the timed case")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compile-warmup", type=int, default=300,
                        help="extra compiled calls before timing (default handles PyTorch 2.5.1 recompilation)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    sizes = [int(item) for item in args.batch_sizes.split(",")]
    total_slots = args.num_blocks * args.block_size
    if not sizes or any(n <= 0 or n > total_slots for n in sizes):
        parser.error(f"batch sizes must be in [1, {total_slots}]")
    if args.kv_heads <= 0 or args.head_dim <= 0 or args.block_size <= 0 or args.num_blocks <= 0:
        parser.error("KV dimensions and cache dimensions must be positive")
    width = args.kv_heads * args.head_dim
    if width & (width - 1):
        parser.error("kv_heads * head_dim must be a power of two for this Triton kernel")
    if not 0 <= args.masked_fraction <= 1:
        parser.error("masked-fraction must be between 0 and 1")
    if args.warmup < 0 or args.compile_warmup < 0 or args.repeats <= 0 or args.iterations <= 0:
        parser.error("warmups must be nonnegative; repeats and iterations must be positive")
    if not torch.cuda.is_available():
        parser.exit(1, "CUDA is unavailable; both implementations need a CUDA GPU for this benchmark.\n")
    if args.device >= torch.cuda.device_count() or args.device < 0:
        parser.error(f"CUDA device {args.device} is unavailable")

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    print(f"GPU: {torch.cuda.get_device_name(args.device)} | dtype={args.dtype} | "
          f"cache=[{args.num_blocks}, {args.block_size}, {args.kv_heads}, {args.head_dim}]")
    print("CUDA event median in milliseconds per call after warmup (lower is faster).")
    print(f"{'N':>7} {'valid':>7} {'Triton ms':>12} {'PyTorch ms':>12} {'Compiled ms':>12} "
          f"{'Eager/Triton':>14} {'Compiled/Triton':>16}")
    for n in sizes:
        valid, triton_ms, torch_ms, compiled_ms = benchmark_case(n, args, dtype)
        print(f"{n:7d} {valid:7d} {triton_ms:12.4f} {torch_ms:12.4f} "
              f"{compiled_ms:12.4f} {torch_ms / triton_ms:14.2f}x "
              f"{compiled_ms / triton_ms:16.2f}x", flush=True)


if __name__ == "__main__":
    main()
