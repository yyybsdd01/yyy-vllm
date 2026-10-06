"""Fit larger K tiles with fewer pipeline stages and compare with production."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import statistics
import sys

import torch
import triton
from triton.runtime.errors import OutOfResources

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from profiling.compare_attention_tiles import make_data,flash_function,tile_function,source_hashes

CONFIGS = ((16,64,3),(64,64,3),
           (16,64,2),(32,64,2),(64,64,2),(128,64,2),
           (16,128,2),(32,128,2),(64,128,2),
           (16,64,1),(64,64,1),
           (16,128,1),(32,128,1),(64,128,1),
           (16,256,1),(32,256,1),(64,256,1))
CURRENT = "M16_N64_S3"


def functions(data,unsupported):
    return {f"M{m}_N{n}_S{s}":tile_function(data,m,n,s) for m,n,s in CONFIGS
            if (data["mode"],f"M{m}_N{n}_S{s}") not in unsupported}


def validate(args,results,unsupported,save):
    for mode,qs,ks in (
        ("prefill",[1,17,63,65,255,257],[1,17,63,65,255,257]),
        ("prefill",[1,17,63,65,255,257],[257,17,319,78,510,514]),
        ("decode",[1]*5,[1,17,255,257,1024]),
    ):
        data = make_data(qs,ks,mode,args.seed)
        reference = flash_function(data,True)()
        for name,fn in functions(data,unsupported).items():
            print("VALIDATE",mode,max(ks),name,flush=True)
            try:
                output = fn()
            except OutOfResources as error:
                unsupported[(mode,name)] = str(error)
                save()
                continue
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output,reference,rtol=.02,atol=.005)
            results["validation"].append(dict(mode=mode,config=name,q_lengths=qs,kv_lengths=ks,
                status="PASS",max_abs_error_restored=(output.float()-reference.float()).abs().max().item()))
            save()
        del data,reference,output,fn
        torch.cuda.empty_cache()
    data = make_data([1]*3,[1]*3,"decode",args.seed)
    data["lengths"].zero_()
    for name,fn in functions(data,unsupported).items():
        output = fn()
        assert torch.equal(output,torch.zeros_like(output))
        results["validation"].append(dict(mode="decode",config=name,status="PASS",test="zero_context"))
    save()
    print("VALIDATION COMPLETE",flush=True)


def benchmark(data,args,unsupported):
    label = f'{data["mode"]}_B{data["batch"]}_Q{data["max_q"]}_K{data["max_k"]}'
    fns = {"flash_bf16":flash_function(data,False),"flash_restored":flash_function(data,True),
           **functions(data,unsupported)}
    reference = fns["flash_restored"]()
    graphs,values = {},{}
    for name,fn in fns.items():
        print("CAPTURE",label,name,flush=True)
        for _ in range(5):
            output = fn()
        torch.cuda.synchronize()
        values[name] = {}
        if name.startswith("M"):
            torch.testing.assert_close(output,reference,rtol=.02,atol=.005)
            kernel = fn.compiled
            values[name].update(registers=kernel.n_regs,spills=kernel.n_spills,
                               shared_bytes=kernel.metadata.shared,grid=fn.grid,
                               max_abs_error_restored=(output.float()-reference.float()).abs().max().item())
            if (data["mode"],data["batch"],data["max_q"],data["max_k"]) in (
                    ("prefill",16,1024,1024),("decode",256,1,1024)):
                for suffix in ("ptx","ttgir","cubin"):
                    content = kernel.asm[suffix]
                    target = args.outdir/f"{label}_{name}.{suffix}"
                    target.write_bytes(content) if isinstance(content,bytes) else target.write_text(content)
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
    baseline = statistics.median(samples[CURRENT])
    for name,rounds in samples.items():
        median = statistics.median(rounds)
        values[name].update(median_us=median,round_us=rounds,min_round_us=min(rounds),
                            max_round_us=max(rounds),speedup_vs_current=baseline/median,
                            latency_change_percent=(median/baseline-1)*100)
        print(f'RESULT {label:<30} {name:<17} {median:9.3f} us {baseline/median:.2f}x',flush=True)
    return dict(case=label,mode=data["mode"],batch=data["batch"],q_length=data["max_q"],
                kv_length=data["max_k"],q_stride=list(data["q"].stride()),variants=values)


def render(outdir,results):
    names = (CURRENT,"M64_N64_S3","M16_N64_S1","M64_N64_S1",
             "M16_N128_S2","M32_N128_S2","M64_N128_S2",
             "M64_N128_S1","M16_N256_S1","M64_N256_S1")
    lines = ["# 扩大 Q/K tile 并降低流水级数", "",
        "S 是 Triton num_stages。INT8 KV、每 token/head 双 FP32 scale、原 token/head scale 布局、D=128、4 warps。",
        "当前基线为 M16/N64/S3。所有配置在同一轮内随机交错测量，Q stride=(4096,128,1)，物理页随机打散。",
        f'CUDA Graph + Event，{results["rounds"]} 轮、每轮 {results["calls_per_graph"]*results["replays"]} 次 attention，报告轮均值的中位数；排除构造数据、写入、编译和 CPU 开销。', "",
        "## 代表配置的 kernel 时间（微秒）", "",
        "| 场景 B/Q/K | Flash BF16 | "+" | ".join(names)+" |",
        "| --- | "+" | ".join(["---:"]*(len(names)+1))+" |"]
    for case in results["cases"]:
        row = [f'{case["variants"][name]["median_us"]:.3f}' if name in case["variants"] else "资源超限"
               for name in ("flash_bf16",*names)]
        lines.append(f'| {case["mode"]} {case["batch"]}/{case["q_length"]}/{case["kv_length"]} | '+" | ".join(row)+" |")
    lines += ["", "## 每个场景的最快配置", "",
        "| 场景 | 当前 μs | 最快配置 | μs | 耗时变化 | 加速比 |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for case in results["cases"]:
        name,best = min(((name,v) for name,v in case["variants"].items() if name.startswith("M")),
                        key=lambda item:item[1]["median_us"])
        current = case["variants"][CURRENT]["median_us"]
        lines.append(f'| {case["case"]} | {current:.3f} | {name} | {best["median_us"]:.3f} | '
                     f'{best["latency_change_percent"]:+.2f}% | {best["speedup_vs_current"]:.2f}× |')
    for mode,batch in (("prefill",16),("decode",256)):
        case = next(c for c in results["cases"] if c["mode"]==mode and c["batch"]==batch and c["kv_length"]==1024)
        lines += ["",f'## {mode} B={batch} 编译资源',"",
                  "| 配置 | μs | registers/thread | spills | shared bytes | grid |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"]
        for name,v in case["variants"].items():
            if name.startswith("M"):
                lines.append(f'| {name} | {v["median_us"]:.3f} | {v["registers"]} | {v["spills"]} | '
                             f'{v["shared_bytes"]} | {v["grid"]} |')
    lines += ["", "## 条件与验证", "",
        "- 3 级流水下 BN=128/256 共享内存超限的原始记录见 ../attention_tiles_2026-10-05/。本实验尝试 2/1 级：BN=128 在 2 级下仍超限，在 1 级下可以运行；BN=256 在 1 级下仍超限。",
        "- 所有可运行配置在混合前缀、零前缀、长度 1/17/63/65/255/257、跨页、非连续物理页、部分 Q/K tile、GQA decode 和零上下文上通过检查。对恢复后 BF16 KV 的 FlashAttention，rtol=0.02、atol=0.005。",
        "- BF16 FlashAttention 使用量化前原始 K/V；Flash restored 是正确性参考，不包含恢复数据的开销。",
        "- 较大的 Q/K tile 改变归约顺序，输出允许浮点误差。没有评估语言质量；没有将此处的 GPU attention 时间当作 TTFT。",
        "- 未锁 GPU 频率；完整样本、17 个配置、编译资源和汇编保存在 results.json 和对应文件。",
        "- 正式 nanovllm 源码及 benchmark_inference_metrics.py 哈希前后相同。", "",
        "## 复现", "", "```bash",
        "CUDA_VISIBLE_DEVICES=0 /home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_attention_tile_stages.py --outdir /tmp/attention-tile-stages",
        "```"]
    (outdir/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=59)
    parser.add_argument("--rounds",type=int,default=11)
    parser.add_argument("--calls-per-graph",type=int,default=4)
    parser.add_argument("--replays",type=int,default=5)
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    assert not (args.outdir/"results.json").exists(), "use a fresh output directory"
    args.outdir.mkdir(parents=True,exist_ok=True)
    originals = source_hashes()
    for relative in originals:
        destination = args.outdir/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    for name in (Path(__file__).name,"compare_attention_tiles.py"):
        shutil.copyfile(ROOT/"profiling"/name,args.outdir/name)
    results = dict(started_utc=datetime.now(timezone.utc).isoformat(),device=torch.cuda.get_device_name(),
        torch=torch.__version__,triton=triton.__version__,seed=args.seed,configs=CONFIGS,
        rounds=args.rounds,calls_per_graph=args.calls_per_graph,replays=args.replays,
        original_source_hashes=originals,validation=[],unsupported=[],cases=[])
    unsupported = {}
    def save():
        results["unsupported"] = [dict(mode=m,config=n,error=e) for (m,n),e in unsupported.items()]
        (args.outdir/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    save()
    validate(args,results,unsupported,save)
    for mode,batch,q,k in (("prefill",1,1024,1024),("prefill",16,1024,1024),("prefill",32,512,512),
                          ("prefill",8,128,1024),("prefill",1,128,4096),
                          ("decode",1,1,1024),("decode",64,1,1024),("decode",256,1,1024),("decode",64,1,4096)):
        data = make_data([q]*batch,[k]*batch,mode,args.seed)
        results["cases"].append(benchmark(data,args,unsupported))
        assert source_hashes()==originals
        save()
        del data
        torch.cuda.empty_cache()
    results.update(finished_utc=datetime.now(timezone.utc).isoformat(),original_source_hashes_after=source_hashes())
    assert results["original_source_hashes_after"]==originals
    save()
    render(args.outdir,results)
    print("COMPLETE",args.outdir/"report.md",flush=True)


if __name__ == "__main__":
    main()
