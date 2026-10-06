"""Compare larger Q/K tiles in the unchanged dual-scale INT8 attention kernel.

The production sources stay unchanged. CUDA Graph + Events measure attention
alone, after compilation and validation against FlashAttention on restored KV.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import sys

import flash_attn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import torch
import triton
from triton.runtime.errors import OutOfResources

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanovllm.layers.attention import store_kvcache_int8
from nanovllm.layers.quantized_attention import (
    _int8_paged_attention_kernel, int8_paged_attention,
)

TILES = ((16,64), (32,64), (64,64), (128,64),
         (16,128), (32,128), (64,128),
         (16,256), (32,256), (64,256))


def source_hashes():
    paths = sorted((ROOT/"nanovllm").rglob("*.py")) + [ROOT/"benchmark_inference_metrics.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def cumulative(lengths):
    values = [0]
    for n in lengths:
        values.append(values[-1]+n)
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def make_data(q_lengths, kv_lengths, mode, seed):
    torch.manual_seed(seed)
    batch, dim, block_size = len(q_lengths), 128, 256
    assert batch == len(kv_lengths) and all(q <= k for q,k in zip(q_lengths,kv_lengths))
    pages = [triton.cdiv(n, block_size) for n in kv_lengths]
    # Keep an allocated page for padded zero-context decode validation.
    blocks = max(1, sum(pages))
    table = torch.full((batch,max(1,max(pages))), -1, device="cuda", dtype=torch.int32)
    permutation = torch.randperm(blocks, device="cuda", dtype=torch.int32)
    slots, cursor = [], 0
    for b,(length,npages) in enumerate(zip(kv_lengths,pages)):
        table[b,:npages] = permutation[cursor:cursor+npages]
        logical = torch.arange(length, device="cuda", dtype=torch.int32)
        slots.append(table[b,logical//block_size]*block_size + logical%block_size)
        cursor += npages
    slots = torch.cat(slots)
    # Match fused QKV projection views: token stride 4096, head stride 128.
    full_qkv = torch.randn((sum(kv_lengths),32,dim), device="cuda", dtype=torch.bfloat16)
    key, value = full_qkv[:,16:24], full_qkv[:,24:]
    if q_lengths == kv_lengths:
        q = full_qkv[:,:16]
    else:
        qkv = torch.randn((sum(q_lengths),32,dim), device="cuda", dtype=torch.bfloat16)
        q = qkv[:,:16]
    shape = (blocks,block_size,8,dim)
    ki = torch.zeros(shape, device="cuda", dtype=torch.int8)
    vi = torch.zeros_like(ki)
    ks = torch.ones((blocks,block_size,16), device="cuda", dtype=torch.float32)
    vs = torch.ones_like(ks)
    if slots.numel():
        store_kvcache_int8(key,value,ki,vi,ks,vs,slots)
    def restore(cache, scales):
        return (cache.float().reshape(blocks,block_size,8,2,64)
                * scales.reshape(blocks,block_size,8,2,1)).reshape(shape).bfloat16()
    restored_k, restored_v = restore(ki,ks), restore(vi,vs)
    zero_prefix = mode == "prefill" and q_lengths == kv_lengths
    if zero_prefix:
        restored_qkv = torch.empty_like(full_qkv)
        restored_qkv[:,16:24] = restored_k.reshape(-1,8,dim)[slots.long()]
        restored_qkv[:,24:] = restored_v.reshape(-1,8,dim)[slots.long()]
        reference_k, reference_v = restored_qkv[:,16:24], restored_qkv[:,24:]
        original_k, original_v = key,value
    else:
        reference_k, reference_v = restored_k,restored_v
        original_k, original_v = torch.zeros_like(restored_k),torch.zeros_like(restored_v)
        original_k.reshape(-1,8,dim)[slots.long()] = key
        original_v.reshape(-1,8,dim)[slots.long()] = value
    return dict(q=q,ki=ki,vi=vi,ks=ks,vs=vs,table=table,
                reference_k=reference_k,reference_v=reference_v,
                original_k=original_k,original_v=original_v,
                cuq=cumulative(q_lengths),cuk=cumulative(kv_lengths),
                lengths=torch.tensor(kv_lengths, device="cuda", dtype=torch.int32),
                q_lengths=q_lengths,kv_lengths=kv_lengths,batch=batch,
                max_q=max(q_lengths),max_k=max(kv_lengths),mode=mode,zero_prefix=zero_prefix)


def flash_function(data, restored):
    k = data["reference_k"] if restored else data["original_k"]
    v = data["reference_v"] if restored else data["original_v"]
    if data["mode"] == "decode":
        return lambda: flash_attn_with_kvcache(
            data["q"][:,None],k,v,cache_seqlens=data["lengths"],
            block_table=data["table"],softmax_scale=128**-.5,causal=True)[:,0]
    return lambda: flash_attn_varlen_func(
        data["q"],k,v,cu_seqlens_q=data["cuq"],cu_seqlens_k=data["cuk"],
        max_seqlen_q=data["max_q"],max_seqlen_k=data["max_k"],
        block_table=None if data["zero_prefix"] else data["table"],
        softmax_scale=128**-.5,causal=True)


def tile_function(data, bm, bn):
    q = data["q"]
    output = torch.empty_like(q)
    decode = data["mode"] == "decode"
    grid = (1 if decode else triton.cdiv(data["max_q"],bm),data["batch"],8 if decode else 16)
    def run():
        run.compiled = _int8_paged_attention_kernel[grid](
            q,data["ki"],data["vi"],data["ks"],data["vs"],output,
            data["table"],q if decode else data["cuq"],data["lengths"] if decode else data["cuk"],
            q.stride(0),q.stride(1),output.stride(0),output.stride(1),
            data["table"].stride(0),256,8,2,128,2,128**-.5,decode,
            bm,bn,128,num_warps=4,num_stages=3)
        return output
    run.grid = grid
    return run


def validate(args, records, unsupported, save):
    tests = [
        ("prefill",[1,17,63,65,255,257],[1,17,63,65,255,257]),
        ("prefill",[1,17,63,65,255,257],[257,17,319,78,510,514]),
        ("decode",[1]*5,[1,17,255,257,1024]),
    ]
    for mode,qlens,klens in tests:
        data = make_data(qlens,klens,mode,args.seed)
        expected = flash_function(data,True)()
        kwargs = (dict(context_lens=data["lengths"]) if mode == "decode" else
                  dict(cu_seqlens_q=data["cuq"],cu_seqlens_k=data["cuk"],max_seqlen_q=max(qlens)))
        current = int8_paged_attention(data["q"],data["ki"],data["vi"],data["ks"],data["vs"],
                                       data["table"],128**-.5,**kwargs)
        for bm,bn in TILES:
            name = f"M{bm}_N{bn}"
            if (mode,name) in unsupported:
                continue
            print(f"VALIDATE {mode} Q={max(qlens)} K={max(klens)} {name}",flush=True)
            fn = tile_function(data,bm,bn)
            try:
                actual = fn()
            except OutOfResources as error:
                unsupported[(mode,name)] = str(error)
                records.append(dict(mode=mode,tile=name,status="OUT_OF_RESOURCES",error=str(error)))
                save()
                continue
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual,expected,rtol=.02,atol=.005)
            torch.testing.assert_close(actual,current,rtol=.02,atol=.005)
            if (bm,bn) == (16,64):
                torch.testing.assert_close(actual,current,rtol=0,atol=0)
            compiled = fn.compiled
            records.append(dict(mode=mode,tile=name,q_lengths=qlens,kv_lengths=klens,status="PASS",
                max_abs_error_restored=(actual.float()-expected.float()).abs().max().item(),
                max_abs_error_current=(actual.float()-current.float()).abs().max().item(),
                registers=compiled.n_regs,spills=compiled.n_spills,shared_bytes=compiled.metadata.shared))
            save()
        del data,expected,current,actual,fn
        torch.cuda.empty_cache()
    # Padded decode rows can have zero valid KV during CUDA Graph capture.
    data = make_data([1]*3,[1]*3,"decode",args.seed)
    data["lengths"].zero_()
    for bm,bn in TILES:
        name = f"M{bm}_N{bn}"
        if ("decode",name) in unsupported:
            continue
        actual = tile_function(data,bm,bn)()
        assert torch.equal(actual,torch.zeros_like(actual))
        records.append(dict(mode="decode",tile=name,status="PASS",test="zero_context"))
        save()
    print("VALIDATION COMPLETE",flush=True)


def benchmark(data, args, unsupported):
    label = f'{data["mode"]}_B{data["batch"]}_Q{data["max_q"]}_K{data["max_k"]}'
    functions = {"flash_bf16":flash_function(data,False),"flash_restored":flash_function(data,True)}
    functions.update({f"M{bm}_N{bn}":tile_function(data,bm,bn) for bm,bn in TILES
                      if (data["mode"],f"M{bm}_N{bn}") not in unsupported})
    expected = functions["flash_restored"]()
    graphs, values = {}, {}
    for name,fn in functions.items():
        print("CAPTURE",label,name,flush=True)
        try:
            for _ in range(5):
                output = fn()
        except OutOfResources as error:
            assert name.startswith("M")
            unsupported[(data["mode"],name)] = str(error)
            values[name] = dict(status="OUT_OF_RESOURCES",error=str(error))
            continue
        torch.cuda.synchronize()
        if name.startswith("M"):
            torch.testing.assert_close(output,expected,rtol=.02,atol=.005)
            compiled = fn.compiled
            values[name] = dict(status="PASS",grid=fn.grid,num_warps=4,num_stages=3,
                               registers=compiled.n_regs,spills=compiled.n_spills,
                               shared_bytes=compiled.metadata.shared,
                               max_abs_error_restored=(output.float()-expected.float()).abs().max().item())
            if (data["mode"],data["batch"],data["max_q"],data["max_k"]) in (
                    ("prefill",16,1024,1024),("decode",256,1,1024)):
                for suffix in ("ptx","ttgir","cubin"):
                    content = compiled.asm[suffix]
                    target = args.outdir/f"{label}_{name}.{suffix}"
                    target.write_bytes(content) if isinstance(content,bytes) else target.write_text(content)
        else:
            values[name] = dict(status="REFERENCE")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.calls_per_graph):
                output = fn()
        graph.replay()
        torch.cuda.synchronize()
        graphs[name] = graph
    samples = {name:[] for name in graphs}
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            graphs[name].replay()
            start,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/(args.calls_per_graph*args.replays))
    for name,rounds in samples.items():
        values[name].update(median_us=statistics.median(rounds),round_us=rounds,
                            min_round_us=min(rounds),max_round_us=max(rounds))
        print(f'RESULT {label:<31} {name:<15} {values[name]["median_us"]:9.3f} us',flush=True)
    baseline = values["M16_N64"]["median_us"]
    for name,value in values.items():
        if "median_us" in value:
            value["latency_change_percent"] = (value["median_us"]/baseline-1)*100
            value["speedup_vs_current"] = baseline/value["median_us"]
    return dict(case=label,mode=data["mode"],batch=data["batch"],q_length=data["max_q"],
                kv_length=data["max_k"],prefix=data["max_k"]-data["max_q"],
                q_stride=list(data["q"].stride()),variants=values)


def render(outdir, results):
    tiles = [f"M{m}_N{n}" for m,n in TILES]
    lines = ["# 扩大 Q/K tile：INT8 双 scale kernel 对照", "",
        "RTX 3090 Ti；BF16 Q，INT8 KV，token/head 布局，每 token/head 两个 FP32 scale。",
        "当前 kernel 源码未改；仅改变 BM/BN 和对应 Q 网格，BD=128、4 warps、num_stages=3。",
        "CUDA Graph + CUDA Event；11 轮随机交错，每轮 20 次 attention，取轮均值的中位数。写入、解量化参考数据构造、编译和 CPU 开销不计入。",
        "Q 来自模型同款融合 QKV 切片，stride=(4096,128,1)；物理页随机打散。", "",
        "## GPU attention 时间（微秒）", "",
        "| 场景 B/Q/K | Flash BF16 | 当前 16×64 | "+" | ".join(t.replace("M","").replace("_N","×") for t in tiles[1:])+" |",
        "| --- | "+" | ".join(["---:"]*(len(tiles)+1))+" |"]
    for case in results["cases"]:
        names = ["flash_bf16",*tiles]
        row = []
        for name in names:
            value = case["variants"].get(name)
            row.append(f'{value["median_us"]:.3f}' if value and "median_us" in value else "资源超限")
        lines.append(f'| {case["mode"]} {case["batch"]}/{case["q_length"]}/{case["kv_length"]} | '+" | ".join(row)+" |")
    lines += ["", "## 每个场景的最快 tile", "",
              "| 场景 | 当前 μs | 最快 tile | μs | 耗时变化 | 加速比 |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for case in results["cases"]:
        candidates = [(name,value) for name,value in case["variants"].items() if name.startswith("M") and "median_us" in value]
        name,best = min(candidates,key=lambda item:item[1]["median_us"])
        baseline = case["variants"]["M16_N64"]["median_us"]
        lines.append(f'| {case["case"]} | {baseline:.3f} | {name} | {best["median_us"]:.3f} | '
                     f'{best["latency_change_percent"]:+.2f}% | {best["speedup_vs_current"]:.2f}× |')
    lines += ["", "## 编译资源（B=16、Q=KV=1024 的 prefill）", "",
              "| tile | registers/thread | spills | shared bytes | grid |", "| --- | ---: | ---: | ---: | --- |"]
    case = next(c for c in results["cases"] if c["mode"]=="prefill" and c["batch"]==16 and c["q_length"]==1024)
    for name in tiles:
        value = case["variants"].get(name)
        if value and value.get("status")=="PASS":
            lines.append(f'| {name} | {value["registers"]} | {value["spills"]} | {value["shared_bytes"]} | {value["grid"]} |')
    lines += ["", "## 验证和边界", "",
        "- 所有可运行 tile 均与恢复后 BF16 KV 的 FlashAttention 参考通过 rtol=0.02、atol=0.005；当前 16×64 与生产 wrapper 逐元素一致。",
        "- 验证包含长度 1/17/63/65/255/257、混合前缀、跨页、非连续物理页、部分 Q/K tile、GQA decode，以及零上下文的 CUDA Graph 填充行。",
        "- Flash BF16 使用量化前原始 K/V；Flash restored 使用与 INT8 kernel 相同的恢复后数据。初始 prefill 的 FlashAttention 使用非 paged varlen 接口，缓存 prefill 和 decode 使用 paged 接口。",
        "- decode 只有两个有效 Q 行；增大 BM 不增加有效行数。",
        "- num_stages 固定为当前默认值 3，没有同时调整流水级数或 warp 数。资源超限配置不能在这些条件下运行，原始报错记录在 JSON。",
        "- 未锁 GPU 频率；完整轮样本、编译资源、PTX/TTGIR/cubin、源码快照和哈希留在此目录。",
        "- 这是独立 attention kernel 时间，没有在这里测 TTFT/TPOT 或语言质量。正式源码哈希前后相同。", "",
        "## 复现", "", "```bash",
        "CUDA_VISIBLE_DEVICES=0 /home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_attention_tiles.py --outdir /tmp/attention-tiles",
        "```"]
    (outdir/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",required=True,type=Path)
    parser.add_argument("--seed",default=59,type=int)
    parser.add_argument("--rounds",default=11,type=int)
    parser.add_argument("--calls-per-graph",default=4,type=int)
    parser.add_argument("--replays",default=5,type=int)
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    assert not (args.outdir/"results.json").exists(), "use a fresh output directory"
    args.outdir.mkdir(parents=True,exist_ok=True)
    originals = source_hashes()
    for relative in originals:
        destination = args.outdir/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    shutil.copyfile(Path(__file__),args.outdir/Path(__file__).name)
    results = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,triton=triton.__version__,flash_attn=flash_attn.__version__,
        seed=args.seed,rounds=args.rounds,calls_per_graph=args.calls_per_graph,replays=args.replays,
        tiles=TILES,num_warps=4,num_stages=3,BD=128,q_heads=16,kv_heads=8,scale_groups=2,
        scale_dtype="float32",scale_layout="block/token/head/group",original_source_hashes=originals,
        validation=[],unsupported=[],cases=[])
    unsupported = {}
    def save():
        results["unsupported"] = [dict(mode=mode,tile=tile,error=error)
                                   for (mode,tile),error in unsupported.items()]
        (args.outdir/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    save()
    validate(args,results["validation"],unsupported,save)
    cases = [("prefill",1,1024,1024),("prefill",16,1024,1024),("prefill",32,512,512),
             ("prefill",8,128,1024),("prefill",1,128,4096),
             ("decode",1,1,1024),("decode",64,1,1024),("decode",256,1,1024),("decode",64,1,4096)]
    for mode,batch,qlen,klen in cases:
        data = make_data([qlen]*batch,[klen]*batch,mode,args.seed)
        results["cases"].append(benchmark(data,args,unsupported))
        assert source_hashes()==originals, "production sources changed during benchmark"
        save()
        del data
        torch.cuda.empty_cache()
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    results["original_source_hashes_after"] = source_hashes()
    assert results["original_source_hashes_after"]==originals
    save()
    render(args.outdir,results)
    print("COMPLETE",args.outdir/"report.md",flush=True)


if __name__ == "__main__":
    main()
