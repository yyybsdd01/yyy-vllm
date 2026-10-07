import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.quantized_attention import int8_paged_attention
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """每个 Triton 程序处理一个 token，按槽位映射将其 K/V 写入物理缓存。"""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """检查 K/V 和缓存布局，并为每个 token 启动一个 KV 写入程序。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def store_kvcache_int8_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, slot_mapping_ptr,
    NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, SCALE_GROUPS: tl.constexpr,
    HEADS_PADDED: tl.constexpr, DIM_PADDED: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    heads = tl.arange(0, HEADS_PADDED)
    dims = tl.arange(0, DIM_PADDED)
    mask = (heads[:, None] < NUM_HEADS) & (dims[None, :] < HEAD_DIM)
    offsets = heads[:, None] * HEAD_DIM + dims[None, :]
    key = tl.load(key_ptr + idx * key_stride + offsets, mask, other=0).to(tl.float32)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask, other=0).to(tl.float32)
    if SCALE_GROUPS == 1:
        k_absmax = tl.max(tl.abs(key), 1)
        v_absmax = tl.max(tl.abs(value), 1)
        k_scale = tl.where(k_absmax > 0, k_absmax / 127.0, 1.0)
        v_scale = tl.where(v_absmax > 0, v_absmax / 127.0, 1.0)
        key_scale = k_scale[:, None]
        value_scale = v_scale[:, None]
    else:
        half = HEAD_DIM // 2
        k_first = tl.max(tl.where(dims[None, :] < half, tl.abs(key), 0.0), 1)
        k_second = tl.max(tl.where(dims[None, :] >= half, tl.abs(key), 0.0), 1)
        v_first = tl.max(tl.where(dims[None, :] < half, tl.abs(value), 0.0), 1)
        v_second = tl.max(tl.where(dims[None, :] >= half, tl.abs(value), 0.0), 1)
        k_scale = tl.where(k_first > 0, k_first / 127.0, 1.0)
        k_scale_second = tl.where(k_second > 0, k_second / 127.0, 1.0)
        v_scale = tl.where(v_first > 0, v_first / 127.0, 1.0)
        v_scale_second = tl.where(v_second > 0, v_second / 127.0, 1.0)
        key_scale = tl.where(dims[None, :] < half, k_scale[:, None], k_scale_second[:, None])
        value_scale = tl.where(dims[None, :] < half, v_scale[:, None], v_scale_second[:, None])
    qkey = tl.extra.cuda.libdevice.nearbyint(key / key_scale).to(tl.int8)
    qvalue = tl.extra.cuda.libdevice.nearbyint(value / value_scale).to(tl.int8)
    cache_offsets = slot * NUM_HEADS * HEAD_DIM + offsets
    tl.store(k_cache_ptr + cache_offsets, qkey, mask)
    tl.store(v_cache_ptr + cache_offsets, qvalue, mask)
    scale_offsets = slot * NUM_HEADS * SCALE_GROUPS + heads * SCALE_GROUPS
    tl.store(k_scale_ptr + scale_offsets, k_scale, heads < NUM_HEADS)
    tl.store(v_scale_ptr + scale_offsets, v_scale, heads < NUM_HEADS)
    if SCALE_GROUPS == 2:
        tl.store(k_scale_ptr + scale_offsets + 1, k_scale_second, heads < NUM_HEADS)
        tl.store(v_scale_ptr + scale_offsets + 1, v_scale_second, heads < NUM_HEADS)


def store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    """Quantize each token/KV head using one or two symmetric FP32 scales."""
    n, num_heads, head_dim = key.shape
    assert value.shape == key.shape and k_cache.dtype == v_cache.dtype == torch.int8
    assert key.stride(-1) == value.stride(-1) == 1
    assert key.stride(1) == value.stride(1) == head_dim
    assert k_cache.stride(1) == v_cache.stride(1) == num_heads * head_dim
    assert k_scale.shape == v_scale.shape and k_scale.shape[:2] == k_cache.shape[:2]
    scale_groups = k_scale.shape[-1] // num_heads
    assert scale_groups in (1, 2) and k_scale.shape[-1] == num_heads * scale_groups
    assert head_dim % scale_groups == 0
    assert k_scale.stride(1) == v_scale.stride(1) == num_heads * scale_groups
    assert slot_mapping.numel() == n
    store_kvcache_int8_kernel[(n,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache,
        k_scale, v_scale, slot_mapping, num_heads, head_dim, scale_groups,
        triton.next_power_of_2(num_heads), triton.next_power_of_2(head_dim),
    )


@triton.jit
def dequantize_kvcache_blocks_kernel(
    k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
    k_out_ptr, v_out_ptr, block_table_ptr, lengths_ptr,
    ELEMENTS_PER_BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr, BLOCK_SIZE: tl.constexpr, SCALE_GROUPS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr, CUMULATIVE_LENGTHS: tl.constexpr,
    TILE: tl.constexpr,
):
    logical_entry = tl.program_id(0)
    physical_block = tl.load(block_table_ptr + logical_entry)
    if physical_block < 0:
        return
    seq_idx = logical_entry // MAX_BLOCKS
    block_idx = logical_entry % MAX_BLOCKS
    if CUMULATIVE_LENGTHS:
        seq_len = tl.load(lengths_ptr + seq_idx + 1) - tl.load(lengths_ptr + seq_idx)
    else:
        seq_len = tl.load(lengths_ptr + seq_idx)
    valid_tokens = tl.minimum(tl.maximum(seq_len - block_idx * BLOCK_SIZE, 0), BLOCK_SIZE)
    valid_elements = valid_tokens * NUM_HEADS * HEAD_DIM
    if valid_elements == 0:
        return
    for start in range(0, valid_elements, TILE):
        offset = start + tl.arange(0, TILE)
        mask = offset < valid_elements
        cache_offset = physical_block * ELEMENTS_PER_BLOCK + offset
        scale_offset = (physical_block * (ELEMENTS_PER_BLOCK // HEAD_DIM) * SCALE_GROUPS
                        + (offset // HEAD_DIM) * SCALE_GROUPS
                        + (offset % HEAD_DIM) // (HEAD_DIM // SCALE_GROUPS))
        key = tl.load(k_cache_ptr + cache_offset, mask, other=0).to(tl.float32)
        value = tl.load(v_cache_ptr + cache_offset, mask, other=0).to(tl.float32)
        k_scale = tl.load(k_scale_ptr + scale_offset, mask, other=0)
        v_scale = tl.load(v_scale_ptr + scale_offset, mask, other=0)
        tl.store(k_out_ptr + cache_offset, key * k_scale, mask)
        tl.store(v_out_ptr + cache_offset, value * v_scale, mask)


def dequantize_kvcache_blocks(k_cache, v_cache, k_scale, v_scale,
                              k_scratch, v_scratch, block_tables, lengths, cumulative_lengths=False):
    """Restore referenced physical blocks into a scratch layer for FlashAttention."""
    assert block_tables is not None and block_tables.dtype == torch.int32
    assert lengths is not None and lengths.dtype == torch.int32
    assert k_cache.shape == v_cache.shape == k_scratch.shape == v_scratch.shape
    assert lengths.numel() == block_tables.shape[0] + int(cumulative_lengths)
    block_size, num_heads, head_dim = k_cache.shape[1:]
    scale_groups = k_scale.shape[-1] // num_heads
    assert scale_groups in (1, 2) and k_scale.shape == v_scale.shape
    assert k_scale.shape[:2] == k_cache.shape[:2]
    assert head_dim % scale_groups == 0
    elements_per_block = block_size * num_heads * head_dim
    dequantize_kvcache_blocks_kernel[(block_tables.numel(),)](
        k_cache, v_cache, k_scale, v_scale, k_scratch, v_scratch, block_tables, lengths,
        elements_per_block, head_dim, num_heads, block_size, scale_groups, block_tables.shape[1],
        cumulative_lengths, 8192,
    )


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """保存注意力头数与缩放系数，并为各层 KV 缓存保留引用。"""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = None
        self.k_scratch = self.v_scratch = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """写入新 K/V，并按缓存格式和阶段选择注意力内核。"""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        quantized = k_cache.dtype == torch.int8
        if k_cache.numel() and v_cache.numel():
            if quantized:
                store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if quantized and (not context.is_prefill or context.block_tables is not None):
            if self.k_scratch is None:
                return int8_paged_attention(
                    q, k_cache, v_cache, self.k_scale, self.v_scale,
                    context.block_tables, self.scale,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    context_lens=context.context_lens,
                    max_seqlen_q=context.max_seqlen_q,
                )
            lengths = context.cu_seqlens_k if context.is_prefill else context.context_lens
            dequantize_kvcache_blocks(k_cache, v_cache, self.k_scale, self.v_scale,
                                      self.k_scratch, self.v_scratch, context.block_tables,
                                      lengths, cumulative_lengths=context.is_prefill)
            k_cache, v_cache = self.k_scratch, self.v_scratch
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
