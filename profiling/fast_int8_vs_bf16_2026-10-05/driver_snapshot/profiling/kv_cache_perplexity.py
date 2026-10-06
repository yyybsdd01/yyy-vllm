"""Teacher-forced perplexity for the project's BF16 and INT8 KV read paths.

The first chunk of each window uses fresh BF16 K/V in both modes. Subsequent
chunks read the same logical context from the selected paged KV cache.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from nanovllm.layers.attention import Attention
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


def bind_cache(model, mode, layers, block_size, num_blocks, kv_heads, head_dim):
    model_dtype = next(model.parameters()).dtype
    dtype = torch.int8 if mode.startswith("int8") else model_dtype
    scale_groups = 2 if "half" in mode else 1
    cache = torch.empty((2, layers, num_blocks, block_size, kv_heads, head_dim),
                        device="cuda", dtype=dtype)
    scales = (torch.empty((2, layers, num_blocks, block_size, kv_heads * scale_groups),
                          device="cuda", dtype=torch.float32) if dtype == torch.int8 else None)
    scratch = (torch.empty(cache.shape, device="cuda", dtype=model_dtype)
               if mode.endswith("dequant") else None)
    attentions = [module for module in model.modules() if isinstance(module, Attention)]
    assert len(attentions) == layers
    for index, attention in enumerate(attentions):
        attention.k_cache = cache[0, index]
        attention.v_cache = cache[1, index]
        attention.k_scale = scales[0, index] if scales is not None else None
        attention.v_scale = scales[1, index] if scales is not None else None
        attention.k_scratch = scratch[0, index] if scratch is not None else None
        attention.v_scratch = scratch[1, index] if scratch is not None else None
    return cache, scales, scratch


@torch.inference_mode()
def evaluate(model, token_ids, mode, window_size, chunk_size, block_size):
    device = torch.device("cuda")
    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    fresh_nll = torch.zeros_like(total_nll)
    cached_nll = torch.zeros_like(total_nll)
    fresh_count = cached_count = 0
    token_count = len(token_ids) - 1

    for base in range(0, token_count, window_size):
        window_tokens = min(window_size, token_count - base)
        for start in range(0, window_tokens, chunk_size):
            end = min(start + chunk_size, window_tokens)
            length = end - start
            current = torch.tensor(token_ids[base + start:base + end], device=device)
            targets = torch.tensor(token_ids[base + start + 1:base + end + 1], device=device)
            positions = torch.arange(start, end, device=device)
            cu_q = torch.tensor([0, length], device=device, dtype=torch.int32)
            cu_k = torch.tensor([0, end], device=device, dtype=torch.int32)
            slots = torch.arange(start, end, device=device, dtype=torch.int32)
            table = (torch.arange(math.ceil(end / block_size), device=device,
                                  dtype=torch.int32)[None, :] if start else None)
            set_context(True, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                        max_seqlen_q=length, max_seqlen_k=end,
                        slot_mapping=slots, block_tables=table)
            try:
                hidden = model(current, positions)
                # ParallelLMHead.forward selects only the last prefill position;
                # use its tied weight directly to score every next-token target.
                logits = F.linear(hidden, model.lm_head.weight)
                nll = F.cross_entropy(logits.float(), targets, reduction="sum")
            finally:
                reset_context()
            total_nll += nll.double()
            if start:
                cached_nll += nll.double()
                cached_count += length
            else:
                fresh_nll += nll.double()
                fresh_count += length
    torch.cuda.synchronize()
    return {
        "mode": mode,
        "tokens": token_count,
        "nll": total_nll.item(),
        "ppl": math.exp(total_nll.item() / token_count),
        "fresh_tokens": fresh_count,
        "fresh_nll": fresh_nll.item(),
        "fresh_ppl": math.exp(fresh_nll.item() / fresh_count),
        "cached_tokens": cached_count,
        "cached_nll": cached_nll.item(),
        "cached_ppl": math.exp(cached_nll.item() / cached_count) if cached_count else math.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16384,
                        help="Number of next-token predictions to score")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--modes", nargs="+", choices=("auto", "int8", "int8_dequant",
                                                     "int8_half", "int8_half_dequant"),
                        default=("auto", "int8", "int8_dequant"))
    args = parser.parse_args()
    assert 0 < args.chunk_size <= args.window_size
    assert args.window_size % args.block_size == 0
    assert args.chunk_size % args.block_size == 0
    assert args.max_tokens > 0

    config = AutoConfig.from_pretrained(args.model)
    assert args.window_size <= config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = 1_000_000  # tokenize the file; windows limit model input
    ids = tokenizer.encode(args.text.read_text(), add_special_tokens=False)
    ids = ids[:min(len(ids), args.max_tokens + 1)]
    assert len(ids) > 1
    print(f"text={args.text} model={args.model}", flush=True)
    print(f"scored_tokens={len(ids) - 1} window={args.window_size} "
          f"chunk={args.chunk_size} block={args.block_size}", flush=True)

    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29571", rank=0, world_size=1)
    torch.cuda.set_device(0)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    torch.set_default_device("cuda")
    try:
        model = Qwen3ForCausalLM(config)
        load_model(model, args.model)
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(original_dtype)
    model.eval()
    layers = config.num_hidden_layers
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_blocks = args.window_size // args.block_size
    results = {}
    try:
        for mode in args.modes:
            cache, scales, scratch = bind_cache(model, mode, layers, args.block_size,
                                                num_blocks, kv_heads, head_dim)
            result = evaluate(model, ids, mode, args.window_size,
                              args.chunk_size, args.block_size)
            results[mode] = result
            print(f"{mode:<12} NLL={result['nll']:.6f} PPL={result['ppl']:.6f} "
                  f"fresh_tokens={result['fresh_tokens']} "
                  f"cached_tokens={result['cached_tokens']} "
                  f"fresh_PPL={result['fresh_ppl']:.6f} "
                  f"cached_PPL={result['cached_ppl']:.6f}", flush=True)
            del cache, scales, scratch
    finally:
        reset_context()
        dist.destroy_process_group()
    if "auto" in results:
        auto = results["auto"]
        for mode in ("int8", "int8_dequant", "int8_half", "int8_half_dequant"):
            if mode in results:
                quantized = results[mode]
                print(f"{mode}_vs_auto delta_PPL={quantized['ppl'] - auto['ppl']:+.6f} "
                      f"relative={(quantized['ppl'] / auto['ppl'] - 1) * 100:+.4f}% "
                      f"delta_NLL_per_token="
                      f"{(quantized['nll'] - auto['nll']) / auto['tokens']:+.8f}")
    if "int8" in results and "int8_half" in results:
        original, half = results["int8"], results["int8_half"]
        print(f"int8_half_vs_int8 delta_PPL={half['ppl'] - original['ppl']:+.6f} "
              f"relative={(half['ppl'] / original['ppl'] - 1) * 100:+.4f}% "
              f"delta_NLL_per_token="
              f"{(half['nll'] - original['nll']) / original['tokens']:+.8f}")


if __name__ == "__main__":
    main()
