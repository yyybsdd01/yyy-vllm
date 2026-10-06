"""Validate an isolated all-INT8 prefill route against restored-KV FlashAttention."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from flash_attn import flash_attn_varlen_func

import nanovllm.layers.attention as attention_module
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("token_head", "head_token"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(53)
    blocks, block_size, heads, q_heads, dim = 16, 256, 8, 16, 128
    shape = (blocks, block_size, heads, dim)
    layer = attention_module.Attention(q_heads, dim, dim**-.5, heads).cuda()
    layer.k_cache = torch.zeros(shape, device="cuda", dtype=torch.int8)
    layer.v_cache = torch.zeros_like(layer.k_cache)
    scale_shape = ((blocks, block_size, heads * 2) if args.layout == "token_head"
                   else (blocks, heads, block_size, 2))
    layer.k_scale = torch.ones(scale_shape, device="cuda")
    layer.v_scale = torch.ones_like(layer.k_scale)
    writer = (attention_module.store_kvcache_int8 if args.layout == "token_head"
              else attention_module.store_kvcache_int8_head_major)
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = block_size
    runner.config = SimpleNamespace(kv_cache_dtype="int8_half")

    # A populated cache must never enter FlashAttention in this isolated route.
    def unexpected_flash(*unused, **kwargs):
        raise AssertionError("allocated INT8 prefill unexpectedly used FlashAttention")
    attention_module.flash_attn_varlen_func = unexpected_flash
    results = []
    query_lengths = [1, 17, 255, 257]
    for prefixes in ([0, 0, 0, 0], [256, 0, 13, 17]):
        sequences, all_slots, tail_indices = [], [], []
        pages = iter([9, 2, 13, 5, 1, 11, 4, 15, 7, 0, 3, 6, 8, 10, 12, 14])
        offset = 0
        for prefix, q_len in zip(prefixes, query_lengths):
            length = prefix + q_len
            seq = Sequence(list(range(length)))
            seq.num_cached_tokens = prefix
            seq.num_scheduled_tokens = q_len
            seq.block_table = [next(pages) for _ in range((length + block_size - 1)//block_size)]
            sequences.append(seq)
            all_slots.extend(seq.block_table[t//block_size] * block_size + t%block_size
                             for t in range(length))
            tail_indices.extend(range(offset + prefix, offset + length))
            offset += length
        full_k = torch.randn((offset, heads, dim), device="cuda", dtype=torch.bfloat16)
        full_v = torch.randn_like(full_k)
        slots = torch.tensor(all_slots, device="cuda", dtype=torch.int32)
        writer(full_k, full_v, layer.k_cache, layer.v_cache,
               layer.k_scale, layer.v_scale, slots)
        _, positions = runner.prepare_prefill(sequences)
        context = get_context()
        assert context.block_tables is not None
        assert context.slot_mapping.tolist() == [all_slots[i] for i in tail_indices]
        assert positions.tolist() == [position for p, n in zip(prefixes, query_lengths)
                                      for position in range(p, p+n)]
        tail = torch.tensor(tail_indices, device="cuda")
        q = torch.randn((sum(query_lengths), q_heads, dim), device="cuda", dtype=torch.bfloat16)
        actual = layer(q, full_k[tail], full_v[tail])

        def restore(cache, scale):
            if args.layout == "head_token":
                scale = scale.permute(0, 2, 1, 3).contiguous()
            scales = scale.reshape(blocks, block_size, heads, 2, 1)
            return (cache.float().reshape(blocks, block_size, heads, 2, dim//2)
                    * scales).reshape(shape).bfloat16()
        reference = flash_attn_varlen_func(
            q, restore(layer.k_cache, layer.k_scale), restore(layer.v_cache, layer.v_scale),
            cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
            block_table=context.block_tables, softmax_scale=dim**-.5, causal=True,
        )
        torch.testing.assert_close(actual, reference, rtol=.02, atol=.005)
        results.append(dict(prefixes=prefixes, query_lengths=query_lengths,
                            max_abs_error=(actual.float()-reference.float()).abs().max().item(),
                            finite=bool(torch.isfinite(actual).all()),
                            allocated_cache_flash_calls=0))
        reset_context()
    # Model initialization happens before KV allocation and must still be usable.
    empty_layer = attention_module.Attention(q_heads, dim, dim**-.5, heads).cuda()
    seq = Sequence(list(range(17)))
    seq.num_scheduled_tokens = 17
    runner.prepare_prefill([seq])
    assert get_context().block_tables is None
    attention_module.flash_attn_varlen_func = flash_attn_varlen_func
    q = torch.randn((17, q_heads, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((17, heads, dim), device="cuda", dtype=torch.bfloat16)
    output = empty_layer(q, k, torch.randn_like(k))
    assert torch.isfinite(output).all()
    reset_context()
    record = dict(layout=args.layout, package=attention_module.__file__, cases=results,
                  warmup_without_cache="PASS", correctness="PASS", rtol=.02, atol=.005)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
