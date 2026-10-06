"""Compare fused INT8 paged attention with FlashAttention on restored KV."""

import unittest

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.quantized_attention import int8_paged_attention


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class QuantizedAttentionTest(unittest.TestCase):
    def test_decode_and_cached_prefill(self):
        for head_dim in (80, 128):
            with self.subTest(head_dim=head_dim):
                torch.manual_seed(7)
                block_size, kv_heads, query_heads, num_blocks = 256, 2, 4, 12
                shape = (num_blocks, block_size, kv_heads, head_dim)
                k = torch.randint(-100, 101, shape, device="cuda", dtype=torch.int8)
                v = torch.randint(-100, 101, shape, device="cuda", dtype=torch.int8)
                scales_shape = shape[:-1]
                ks = torch.rand(scales_shape, device="cuda") * 0.02 + 0.001
                vs = torch.rand(scales_shape, device="cuda") * 0.02 + 0.001
                restored_k = (k.float() * ks[..., None]).bfloat16()
                restored_v = (v.float() * vs[..., None]).bfloat16()
                table = torch.tensor([[5, 2, 1], [7, 0, 3], [4, 10, 8]],
                                     device="cuda", dtype=torch.int32)
                scale = head_dim ** -0.5

                decode_lengths = torch.tensor([1, 249, 512], device="cuda", dtype=torch.int32)
                q = torch.randn((3, query_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                actual = int8_paged_attention(q, k, v, ks, vs, table, scale,
                                              context_lens=decode_lengths)
                expected = flash_attn_with_kvcache(
                    q[:, None], restored_k, restored_v, cache_seqlens=decode_lengths,
                    block_table=table, softmax_scale=scale, causal=True,
                )[:, 0]
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)

                # Sequence 1 has a cached prefix; sequence 2 crosses a KV block.
                cu_q = torch.tensor([0, 5, 22, 45], device="cuda", dtype=torch.int32)
                cu_k = torch.tensor([0, 5, 285, 797], device="cuda", dtype=torch.int32)
                q = torch.randn((45, query_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                actual = int8_paged_attention(q, k, v, ks, vs, table, scale,
                                              cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                              max_seqlen_q=23)
                expected = flash_attn_varlen_func(
                    q, restored_k, restored_v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=23, max_seqlen_k=512, block_table=table,
                    softmax_scale=scale, causal=True,
                )
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)

                # Graph warmup/capture can contain padded rows with no valid KV.
                empty_lengths = torch.zeros(3, device="cuda", dtype=torch.int32)
                empty = int8_paged_attention(q[:3], k, v, ks, vs, table, scale,
                                             context_lens=empty_lengths)
                self.assertTrue(torch.equal(empty, torch.zeros_like(empty)))


if __name__ == "__main__":
    unittest.main()
