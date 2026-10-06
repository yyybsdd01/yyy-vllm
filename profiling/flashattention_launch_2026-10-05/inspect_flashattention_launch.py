"""Record actual FlashAttention forward launch grids and thread blocks."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import torch
import flash_attn
import flash_attn_2_cuda
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


def make_case(mode, batch, query_length, context_length, num_splits=0):
    qkv = torch.randn((batch*query_length, 32, 128), device="cuda", dtype=torch.bfloat16)
    q, key, value = qkv[:, :16], qkv[:, 16:24], qkv[:, 24:]
    cuq = torch.arange(batch+1, device="cuda", dtype=torch.int32) * query_length
    cuk = torch.arange(batch+1, device="cuda", dtype=torch.int32) * context_length
    if mode == "prefill":
        return lambda: flash_attn_varlen_func(
            q, key, value, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
            max_seqlen_q=query_length, max_seqlen_k=context_length,
            softmax_scale=128**-.5, causal=True)
    pages_per_sequence = (context_length+255)//256
    k_cache = torch.randn((batch*pages_per_sequence, 256, 8, 128),
                          device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    table = torch.arange(batch*pages_per_sequence, device="cuda", dtype=torch.int32).reshape(batch,-1)
    if mode == "cached_prefill":
        return lambda: flash_attn_varlen_func(
            q, k_cache, v_cache, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
            max_seqlen_q=query_length, max_seqlen_k=context_length,
            block_table=table, softmax_scale=128**-.5, causal=True)
    lengths = torch.full((batch,), context_length, device="cuda", dtype=torch.int32)
    return lambda: flash_attn_with_kvcache(
        q.reshape(batch,query_length,16,128), k_cache, v_cache,
        cache_seqlens=lengths, block_table=table, softmax_scale=128**-.5,
        causal=True, num_splits=num_splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    assert not (args.outdir/"launches.json").exists(), "use a fresh output directory"
    torch.manual_seed(57)
    cases = [
        ("prefill",1,1024,1024,0), ("prefill",16,1024,1024,0),
        ("cached_prefill",8,128,1024,0),
        ("decode",256,1,1024,0), ("decode",64,1,1024,0),
        ("decode",1,1,1024,0), ("decode",1,1,4096,0),
        ("decode",1,1,1024,1),
    ]
    results = dict(torch=torch.__version__, flash_attn=flash_attn.__version__,
        flash_package=flash_attn.__file__, extension=flash_attn_2_cuda.__file__,
        device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        q_heads=16, kv_heads=8, head_dim=128, dtype="bfloat16", block_size=256,
        causal=True, dropout_p=0.0, cases=[])
    print(f"{'Case':<40} {'Kernel':<19} {'Grid':<17} {'Block':<13} {'Warps':>5} {'Tile M/N/D':<12}",flush=True)
    for mode,batch,qlen,klen,splits in cases:
        name = f"{mode}_B{batch}_Q{qlen}_K{klen}_S{splits}"
        fn = make_case(mode,batch,qlen,klen,splits)
        for _ in range(5):
            output = fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function(name):
                output = fn()
            torch.cuda.synchronize()
        trace_path = args.outdir/(name+".json")
        prof.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
        launches = []
        for event in trace["traceEvents"]:
            if event.get("cat") != "kernel" or "flash_fwd" not in event.get("name",""):
                continue
            metadata = event["args"]
            grid, block = metadata["grid"], metadata["block"]
            threads = block[0]*block[1]*block[2]
            assert threads % 32 == 0
            match = re.search(r"Flash_fwd_kernel_traits<([^>]+)",event["name"])
            traits = [int(re.sub(r"\(int\)","",v).strip()) for v in match[1].split(",")[:4]] if match else None
            kernel = ("splitkv_combine" if "splitkv_combine" in event["name"]
                      else "splitkv" if "splitkv" in event["name"] else "regular")
            row = dict(name=event["name"], kind=kernel, grid=grid, block=block,
                       threads=threads, warps=threads//32, traits=traits,
                       shared_memory=metadata.get("shared memory"),
                       registers_per_thread=metadata.get("registers per thread"))
            launches.append(row)
            tile = str((traits[1],traits[2],traits[0])) if traits else "unknown"
            print(f"{name:<40} {kernel:<19} {str(tuple(grid)):<17} {str(tuple(block)):<13} {threads//32:>5} {tile:<12}",flush=True)
        assert launches, "missing FlashAttention CUDA events"
        assert torch.isfinite(output).all()
        results["cases"].append(dict(case=name,mode=mode,batch=batch,query_length=qlen,
            context_length=klen,num_splits_argument=splits,trace=trace_path.name,kernels=launches))
        del fn,output,prof,trace
        torch.cuda.empty_cache()
    (args.outdir/"launches.json").write_text(json.dumps(results,indent=2)+"\n")
    shutil.copyfile(Path(__file__),args.outdir/Path(__file__).name)
    print("COMPLETE",args.outdir/"launches.json",flush=True)


if __name__ == "__main__":
    main()
