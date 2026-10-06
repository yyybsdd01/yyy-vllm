"""Check two INT8 scales per token/KV head against a BF16 reference."""

import unittest

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.attention import store_kvcache_int8, dequantize_kvcache_blocks
from nanovllm.layers.quantized_attention import int8_paged_attention


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class HalfHeadKVCacheTest(unittest.TestCase):
    def test_store_restore_and_attention(self):
        for head_dim in (80, 128):
            with self.subTest(head_dim=head_dim):
                torch.manual_seed(19)
                n, block_size, kv_heads, query_heads = 8, 256, 2, 4
                shape = (1, block_size, kv_heads, head_dim)
                key = torch.randn((n, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                value = torch.randn_like(key)
                key[..., head_dim // 2:] *= 0.1
                value[..., :head_dim // 2] *= 0.2
                k_cache = torch.empty(shape, device="cuda", dtype=torch.int8)
                v_cache = torch.empty_like(k_cache)
                k_scale = torch.empty((1, block_size, kv_heads * 2), device="cuda")
                v_scale = torch.empty_like(k_scale)
                slots = torch.arange(n, device="cuda", dtype=torch.int32)
                store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slots)

                for source, actual_cache, actual_scale in (
                    (key, k_cache, k_scale), (value, v_cache, v_scale)
                ):
                    grouped = source.float().reshape(n, kv_heads, 2, head_dim // 2)
                    absmax = grouped.abs().amax(dim=-1)
                    expected_scale = torch.where(absmax > 0, absmax / 127, 1.0)
                    expected_int8 = torch.round(grouped / expected_scale[..., None]).to(torch.int8)
                    torch.testing.assert_close(actual_scale[0, :n].reshape(n, kv_heads, 2),
                                               expected_scale, rtol=1e-6, atol=0)
                    # Triton and PyTorch can round near half-integer ties differently.
                    error = (actual_cache[0, :n].int()
                             - expected_int8.reshape(n, kv_heads, head_dim).int()).abs()
                    self.assertLessEqual(error.max().item(), 1)
                    self.assertLessEqual((error != 0).sum().item(), error.numel() // 100)

                k_scratch = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
                v_scratch = torch.empty_like(k_scratch)
                table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
                lengths = torch.tensor([n], device="cuda", dtype=torch.int32)
                dequantize_kvcache_blocks(k_cache, v_cache, k_scale, v_scale,
                                          k_scratch, v_scratch, table, lengths)
                for actual, cache, scale in (
                    (k_scratch, k_cache, k_scale), (v_scratch, v_cache, v_scale)
                ):
                    expected = (cache[0, :n].float().reshape(n, kv_heads, 2, head_dim // 2)
                                * scale[0, :n].reshape(n, kv_heads, 2, 1)).reshape(n, kv_heads, head_dim)
                    torch.testing.assert_close(actual[0, :n], expected.bfloat16(), rtol=0, atol=0)

                softmax_scale = head_dim ** -0.5
                q = torch.randn((1, query_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                actual = int8_paged_attention(q, k_cache, v_cache, k_scale, v_scale,
                                              table, softmax_scale, context_lens=lengths)
                expected = flash_attn_with_kvcache(
                    q[:, None], k_scratch, v_scratch, cache_seqlens=lengths,
                    block_table=table, softmax_scale=softmax_scale, causal=True,
                )[:, 0]
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)

                cu_q = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
                cu_k = torch.tensor([0, n], device="cuda", dtype=torch.int32)
                q = torch.randn((4, query_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                actual = int8_paged_attention(q, k_cache, v_cache, k_scale, v_scale,
                                              table, softmax_scale, cu_seqlens_q=cu_q,
                                              cu_seqlens_k=cu_k, max_seqlen_q=4)
                expected = flash_attn_varlen_func(
                    q, k_scratch, v_scratch, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=4, max_seqlen_k=n, block_table=table,
                    softmax_scale=softmax_scale, causal=True,
                )
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


if __name__ == "__main__":
    unittest.main()
