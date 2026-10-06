"""Paged attention that reads INT8 KV with one or two scales per token/head."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_dual_stride_first(
    q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, out_ptr,
    block_table_ptr, q_starts_ptr, k_lengths_ptr,
    Q_STRIDE_TOKEN: tl.constexpr, Q_STRIDE_HEAD: tl.constexpr,
    O_STRIDE_TOKEN: tl.constexpr, O_STRIDE_HEAD: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, KV_HEADS: tl.constexpr, SCALE_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr, Q_PER_KV: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr, DECODE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
):
    # 网格三维依次是 Q tile、请求、head：decode 按 KV head，prefill 按 Q head 划分。
    q_tile = tl.program_id(0)
    seq = tl.program_id(1)
    head_group = tl.program_id(2)
    # BM 行用于 Q，BN 列用于一轮处理的历史 token，BD 是补齐后的 head_dim。
    rows = tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, BD)

    if DECODE:
        # 每个请求只有一个新 token；同一 KV head 对应的 Q heads 占据不同的行。
        q_start = seq
        q_len = 1
        kv_len = tl.load(k_lengths_ptr + seq)
        kv_head = head_group
        q_heads = kv_head * Q_PER_KV + rows
        valid_rows = rows < Q_PER_KV
        logical_q = tl.full((BM,), 0, tl.int32)
        q_tokens = tl.full((BM,), q_start, tl.int32)
    else:
        # 每个 program 处理一个 Q head 的 BM 个新 token；短请求的多余 tile 直接退出。
        q_start = tl.load(q_starts_ptr + seq)
        q_len = tl.load(q_starts_ptr + seq + 1) - q_start
        kv_len = tl.load(k_lengths_ptr + seq + 1) - tl.load(k_lengths_ptr + seq)
        if q_tile * BM >= q_len:
            return
        kv_head = head_group // Q_PER_KV
        q_heads = tl.full((BM,), head_group, tl.int32)
        logical_q = q_tile * BM + rows
        q_tokens = q_start + logical_q
        valid_rows = logical_q < q_len

    # KV 序列由缓存前缀和本轮新 token 组成；将 Q 行映射到该序列中的绝对位置。
    q_positions = kv_len - q_len + logical_q
    # Q 只读取一次，后续循环反复与不同的 K tile 计算；越界的行和维度填零。
    q = tl.load(
        q_ptr + q_tokens[:, None] * Q_STRIDE_TOKEN
        + q_heads[:, None] * Q_STRIDE_HEAD + dims[None, :],
        valid_rows[:, None] & (dims[None, :] < HEAD_DIM), other=0,
    )
    # 在线 softmax 的跨 tile 状态，避免保存完整的注意力分数矩阵。
    running_max = tl.full((BM,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, BD), tl.float32)

    # prefill 只需遍历到当前 Q tile 的末尾；每个 Q 行的因果边界稍后单独掩码。
    if DECODE:
        visible_k = kv_len
    else:
        visible_k = tl.minimum(kv_len, kv_len - q_len + (q_tile + 1) * BM)
    # 每轮最多读取 BN 个历史 token 的 K/V，处理完后复用片上工作空间。
    for tile in range(tl.cdiv(visible_k, BN)):
        logical_k = tile * BN + cols
        valid_k = logical_k < kv_len
        # 先用页表把逻辑 token 映射到物理 KV block，再算缓存中的槽位和地址。
        physical_block = tl.load(
            block_table_ptr + seq * TABLE_STRIDE + logical_k // BLOCK_SIZE,
            valid_k, other=0,
        )
        slot = physical_block * BLOCK_SIZE + logical_k % BLOCK_SIZE
        kv_offset = slot[:, None] * (KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM + dims[None, :]
        if SCALE_GROUPS == 1:
            # 每个 token、每个 KV head 的 K 和 V 各有一个 scale。
            scale_offset = slot * KV_HEADS + kv_head
            k_scale = tl.load(ks_ptr + scale_offset, valid_k, other=0)[:, None]
            v_scale = tl.load(vs_ptr + scale_offset, valid_k, other=0)[:, None]
        else:
            # 双 scale：前后半个 head_dim 分别使用各自的缩放系数。
            scale_offset = slot * KV_HEADS * SCALE_GROUPS + kv_head * SCALE_GROUPS
            k_scale = tl.load(ks_ptr + scale_offset, valid_k, other=0)[:, None]
            v_scale = tl.load(vs_ptr + scale_offset, valid_k, other=0)[:, None]
        # 从 INT8 KV cache 读取当前 tile；解量化留在本次 attention 内核中完成。
        k = tl.load(k_ptr + kv_offset, valid_k[:, None] & (dims[None, :] < HEAD_DIM), other=0)
        v = tl.load(v_ptr + kv_offset, valid_k[:, None] & (dims[None, :] < HEAD_DIM), other=0)
        k = (k.to(tl.float32) * k_scale).to(q.dtype)
        v = (v.to(tl.float32) * v_scale).to(q.dtype)

        # QK^T 得到当前 tile 的分数，并屏蔽无效行、越界 K 和未来 token。
        scores = tl.dot(q, tl.trans(k)) * SOFTMAX_SCALE
        allowed = valid_rows[:, None] & valid_k[None, :] & (logical_k[None, :] <= q_positions[:, None])
        scores = tl.where(allowed, scores, -1.0e6)
        # 用新的行最大值重标定旧结果，再累加当前 tile 的 softmax 分母和 PV。
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * correction + tl.sum(probabilities, 1)
        accumulator = accumulator * correction[:, None] + tl.dot(probabilities.to(q.dtype), v)
        running_max = new_max

    # CUDA Graph 捕获时 context_len 可能为零；仅写回有效 Q 行的归一化结果。
    result = tl.where(running_sum[:, None] > 0, accumulator / running_sum[:, None], 0.0)
    tl.store(
        out_ptr + q_tokens[:, None] * O_STRIDE_TOKEN
        + q_heads[:, None] * O_STRIDE_HEAD + dims[None, :],
        result, valid_rows[:, None] & (dims[None, :] < HEAD_DIM),
    )

