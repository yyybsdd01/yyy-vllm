"""Check both kernels against the current layout on scattered physical pages."""
import unittest

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.attention import store_kvcache_int8
from nanovllm.layers.quantized_attention import int8_paged_attention
from nanovllm.layers.head_major_scale_attention import (
    store_kvcache_int8_head_major, int8_paged_attention_head_major,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class HeadMajorScaleAttentionTest(unittest.TestCase):
    def test_scattered_store_decode_and_prefill(self):
        for dim in (80, 128):
            with self.subTest(head_dim=dim):
                torch.manual_seed(43)
                blocks, block_size, heads, q_heads = 12, 256, 3, 6
                shape = (blocks, block_size, heads, dim)
                old_k = torch.randint(-100, 101, shape, device="cuda", dtype=torch.int8)
                old_v = torch.randint_like(old_k, low=-100, high=101)
                new_k, new_v = old_k.clone(), old_v.clone()
                old_ks = torch.rand((blocks, block_size, heads * 2), device="cuda") * .02 + .001
                old_vs = torch.rand_like(old_ks) * .02 + .001
                def transpose_scales(scale):
                    return scale.reshape(blocks, block_size, heads, 2).permute(0, 2, 1, 3).contiguous()
                new_ks, new_vs = transpose_scales(old_ks), transpose_scales(old_vs)
                # Include invalid slots, block boundaries, and different physical pages.
                slots = torch.tensor([5*256, 5*256+255, 2*256, 7*256+248, -1,
                                      10*256+17, 0, 4*256+23], device="cuda", dtype=torch.int32)
                key = torch.randn((8, heads, dim), device="cuda", dtype=torch.bfloat16)
                value = torch.randn_like(key)
                key[..., dim//2:] *= .1
                value[..., :dim//2] *= .2
                key[2, 0, :dim//2] = 0
                value[6, 2, dim//2:] = 0
                untouched = old_ks.clone()
                store_kvcache_int8(key, value, old_k, old_v, old_ks, old_vs, slots)
                store_kvcache_int8_head_major(key, value, new_k, new_v, new_ks, new_vs, slots)
                torch.testing.assert_close(new_k, old_k, rtol=0, atol=0)
                torch.testing.assert_close(new_v, old_v, rtol=0, atol=0)
                torch.testing.assert_close(new_ks, transpose_scales(old_ks), rtol=0, atol=0)
                torch.testing.assert_close(new_vs, transpose_scales(old_vs), rtol=0, atol=0)
                changed = torch.zeros((blocks * block_size,), dtype=torch.bool, device="cuda")
                changed[slots[slots >= 0].long()] = True
                torch.testing.assert_close(old_ks.reshape(-1, heads*2)[~changed],
                                           untouched.reshape(-1, heads*2)[~changed], rtol=0, atol=0)
                table = torch.tensor([[5, 2, 1], [7, 0, 3], [4, 10, 8]],
                                     device="cuda", dtype=torch.int32)
                def restore(cache, scales):
                    return (cache.float().reshape(blocks, block_size, heads, 2, dim//2)
                            * scales.reshape(blocks, block_size, heads, 2, 1)).reshape(shape).bfloat16()
                restored_k, restored_v = restore(old_k, old_ks), restore(old_v, old_vs)
                lengths = torch.tensor([1, 249, 512], device="cuda", dtype=torch.int32)
                q = torch.randn((3, q_heads, dim), device="cuda", dtype=torch.bfloat16)
                old = int8_paged_attention(q, old_k, old_v, old_ks, old_vs, table,
                                           dim**-.5, context_lens=lengths)
                new = int8_paged_attention_head_major(q, new_k, new_v, new_ks, new_vs, table,
                                                     dim**-.5, context_lens=lengths)
                reference = flash_attn_with_kvcache(q[:, None], restored_k, restored_v,
                                                    cache_seqlens=lengths, block_table=table,
                                                    softmax_scale=dim**-.5, causal=True)[:, 0]
                torch.testing.assert_close(new, old, rtol=0, atol=0)
                torch.testing.assert_close(new, reference, rtol=.02, atol=.005)
                # Heterogeneous prefixes, a partial Q tile, and a physical page crossing.
                cu_q = torch.tensor([0, 5, 22, 45], device="cuda", dtype=torch.int32)
                cu_k = torch.tensor([0, 5, 285, 797], device="cuda", dtype=torch.int32)
                q = torch.randn((45, q_heads, dim), device="cuda", dtype=torch.bfloat16)
                kwargs = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=23)
                old = int8_paged_attention(q, old_k, old_v, old_ks, old_vs, table, dim**-.5, **kwargs)
                new = int8_paged_attention_head_major(q, new_k, new_v, new_ks, new_vs,
                                                     table, dim**-.5, **kwargs)
                reference = flash_attn_varlen_func(q, restored_k, restored_v,
                                                  max_seqlen_k=512, block_table=table,
                                                  softmax_scale=dim**-.5, causal=True, **kwargs)
                torch.testing.assert_close(new, old, rtol=0, atol=0)
                torch.testing.assert_close(new, reference, rtol=.02, atol=.005)
                zeros = torch.zeros(3, device="cuda", dtype=torch.int32)
                empty = int8_paged_attention_head_major(q[:3], new_k, new_v, new_ks, new_vs,
                                                       table, dim**-.5, context_lens=zeros)
                torch.testing.assert_close(empty, torch.zeros_like(empty), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
