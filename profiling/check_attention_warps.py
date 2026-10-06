"""Check the 4/8-warp saved INT8 kernels against restored-KV FlashAttention."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.int8_paged_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    kernels = {warps: load_kernel(args.outdir / f"variants/warps{warps}/nanovllm/layers/quantized_attention.py",
                                  f"checked_int8_warps{warps}") for warps in (4, 8)}
    torch.manual_seed(7)
    shape = (24, 256, 8, 128)
    k = torch.randint(-100, 101, shape, device="cuda", dtype=torch.int8)
    v = torch.randint(-100, 101, shape, device="cuda", dtype=torch.int8)
    ks = torch.rand((24, 256, 16), device="cuda") * 0.02 + 0.001
    vs = torch.rand_like(ks) * 0.02 + 0.001
    restored_k = (k.float() * ks.view(24, 256, 8, 2).repeat_interleave(64, dim=-1)).bfloat16()
    restored_v = (v.float() * vs.view(24, 256, 8, 2).repeat_interleave(64, dim=-1)).bfloat16()
    table = torch.randperm(24, device="cuda", dtype=torch.int32).view(3, 8)
    records = []

    def record(name, outputs, expected):
        torch.testing.assert_close(outputs[8], outputs[4], rtol=0.02, atol=0.005)
        for warps in (4, 8):
            torch.testing.assert_close(outputs[warps], expected, rtol=0.02, atol=0.005)
        row = dict(case=name, status="PASS", max_abs_8_vs_4=(outputs[8]-outputs[4]).abs().max().item(),
                   max_abs_4_vs_flash=(outputs[4]-expected).abs().max().item(),
                   max_abs_8_vs_flash=(outputs[8]-expected).abs().max().item())
        records.append(row)
        print(json.dumps(row), flush=True)

    lengths = torch.tensor([1, 249, 2048], device="cuda", dtype=torch.int32)
    q = torch.randn((3, 16, 128), device="cuda", dtype=torch.bfloat16)
    outputs = {w: fn(q, k, v, ks, vs, table, 128 ** -0.5, context_lens=lengths)
               for w, fn in kernels.items()}
    expected = flash_attn_with_kvcache(q[:, None], restored_k, restored_v,
                                     cache_seqlens=lengths, block_table=table,
                                     softmax_scale=128 ** -0.5, causal=True)[:, 0]
    record("decode_lengths_1_249_2048", outputs, expected)

    empty_lengths = torch.zeros(3, device="cuda", dtype=torch.int32)
    for w, fn in kernels.items():
        output = fn(q, k, v, ks, vs, table, 128 ** -0.5, context_lens=empty_lengths)
        assert torch.equal(output, torch.zeros_like(output))
    records.append(dict(case="decode_padded_zero_lengths", status="PASS"))
    print("PASS decode_padded_zero_lengths", flush=True)

    cases = [("mixed_short_cached_prefill", [5, 17, 23], [5, 280, 512]),
             ("fresh_prefill_256_768_1024", [256, 768, 1024], [256, 768, 1024]),
             ("cached_prefill_up_to_2048", [128, 256, 512], [1024, 1280, 2048])]
    for name, q_lengths, k_lengths in cases:
        cu_q = torch.tensor([0] + q_lengths, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        cu_k = torch.tensor([0] + k_lengths, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        q = torch.randn((sum(q_lengths), 16, 128), device="cuda", dtype=torch.bfloat16)
        outputs = {w: fn(q, k, v, ks, vs, table, 128 ** -0.5, cu_seqlens_q=cu_q,
                         cu_seqlens_k=cu_k, max_seqlen_q=max(q_lengths)) for w, fn in kernels.items()}
        expected = flash_attn_varlen_func(q, restored_k, restored_v, cu_seqlens_q=cu_q,
                                         cu_seqlens_k=cu_k, max_seqlen_q=max(q_lengths),
                                         max_seqlen_k=max(k_lengths), block_table=table,
                                         softmax_scale=128 ** -0.5, causal=True)
        record(name, outputs, expected)
    torch.cuda.synchronize()
    (args.outdir / "correctness.json").write_text(json.dumps(dict(status="PASS", cases=records,
        heads=dict(query=16, kv=8, head_dim=128), scale_groups=2, rtol=0.02, atol=0.005), indent=2) + "\n")


if __name__ == "__main__":
    main()
