"""Measure full-model effects of tile/stage changes in isolated all-INT8 copies."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from profiling.run_scale_layout_metrics import PYTHON,hashes,aggregate
from profiling.run_prefill_kernel_metrics import prepare_variants,run


def render(outdir,manifest,results):
    medians = results["medians"]
    lines = ["# 较大 Q tile 和单级流水：完整模型对照", "",
        "两组都使用 INT8 双 FP32 scale KV、原 token/head scale 布局，自写 kernel 执行已分配缓存的所有 prefill/decode。",
        "当前组：prefill/decode 均 BM=16、BN=64、num_stages=3。优化组：prefill BM=64、BN=64、num_stages=1；decode BM=16、BN=64、num_stages=1。均 BD=128、4 warps。",
        "K tile=128 已在独立 kernel 实验中运行，但整体效果逊于这里选择的 K tile=64。正式源码未修改。", "",
        "## 三轮中位数", "",
        "| 指标 | 当前 | 优化实验 | 变化 |", "| --- | ---: | ---: | ---: |"]
    def add(label,getter,digits=2):
        old,new = [getter(medians[name]) for name in ("current","tiled")]
        lines.append(f"| {label} | {old:.{digits}f} | {new:.{digits}f} | {(new/old-1)*100:+.2f}% |")
    for key in ("TTFT","TPOT","ITL","End-to-end latency"):
        for quantile in ("mean","p50","p95","p99"):
            add(f"{key} {quantile.upper()} ms",lambda m,k=key,q=quantile:m["latency_ms"][k][q])
    for label,key,digits in (("generate s","elapsed_s",3),("输出 token/s","output_tokens_per_s",2),
                             ("请求/s","requests_per_s",2),("输入 token/s","input_tokens_per_s",2),
                             ("输入+输出 token/s","total_tokens_per_s",2),
                             ("KV blocks","blocks",0),("峰值 allocated GiB","peak_allocated_gib",2),
                             ("峰值 reserved GiB","peak_reserved_gib",2)):
        add(label,lambda m,k=key:m[k],digits)
    for stage in ("prefill","decode"):
        add(f"{stage} model-run s",lambda m,s=stage:m["stages"][s]["seconds"],3)
        add(f"{stage} tokens/s",lambda m,s=stage:m["stages"][s]["tokens_per_s"])
    lines += ["", "## 每轮结果", "",
        "| 配置 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | prefill s | decode s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results["trials"]:
        lines.append(f'| {row["variant"]} | {row["trial"]} | {row["elapsed_s"]:.3f} | '
            f'{row["output_tokens_per_s"]:.2f} | {row["latency_ms"]["TTFT"]["p50"]:.2f} | '
            f'{row["latency_ms"]["TPOT"]["p50"]:.2f} | {row["stages"]["prefill"]["seconds"]:.3f} | '
            f'{row["stages"]["decode"]["seconds"]:.3f} |')
    lines += ["", "## 条件与验证", "",
        "- Qwen3-0.6B、RTX 3090 Ti、GPU 0，单卡；decode 使用 CUDA Graph，prefill eager；未锁 GPU 频率。",
        "- 256 请求同时到达，输入/输出长度各 100–1024，seed=0、torch sampling seed=0、temperature=0.6、ignore_eos=True。输入 142,827、输出 133,966 token，逐请求长度检查通过。",
        "- 三轮独立新进程，配置顺序交替。初始化、编译、生成预热和代表 prefill 特化预热不计入正式时间；预热后重设采样种子。",
        "- 页表宽度 1/2/3/4，以及 Q=1024、B=16 的大 prefill 在计时前预热；每批实际 prefill 派发次数与形状摘要均审计。",
        "- 两组采用相同 max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9；KV 容量及阶段 token 工作量一致。",
        "- 集成验证覆盖零前缀、混合前缀、长度 1/17/255/257、跨页、部分 Q tile、随机物理页，并与恢复后 BF16 KV 的 FlashAttention 通过 rtol=0.02、atol=0.005。",
        ("- 本负载两组及重复轮次生成的输出 token SHA256 一致。较大 tile 和不同流水级数可能改变浮点归约顺序，这里没有评估语言质量。"
         if len({r["output_sha256"] for r in results["trials"]})==1 else
         "- 较大 tile 和不同流水级数可能改变浮点归约顺序。输出 token 摘要见下表；这里没有评估语言质量。"),
        "- TTFT 从整批请求提交到 CPU postprocess 完成首 token，包含排队。TPOT/ITL 也基于 CPU postprocess 时间戳；阶段时间包含准备、模型、采样与同步，不是单 attention kernel 时间。",
        "- 优化组同时改变 prefill 的 BM 和 prefill/decode 的 num_stages；整模型收益属于这组组合。纯 BM/BN 的对照另见 kernel 实验报告。",
        "- 原仓库 nanovllm 源码及 benchmark_inference_metrics.py 的哈希前后相同，实验改动仅在独立副本。", "",
        "## 输出摘要", "", "| 配置 | 正式输出 SHA256 |", "| --- | --- |"]
    for name in ("current","tiled"):
        digests = sorted({r["output_sha256"] for r in results["trials"] if r["variant"]==name})
        lines.append(f'| {name} | {", ".join(digests)} |')
    lines += ["", "## 复现", "", "```bash",
        "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_attention_tile_metrics.py --outdir /tmp/attention-tile-metrics",
        "```"]
    (outdir/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--model",default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--runs",type=int,default=3)
    parser.add_argument("--gpu",default="0")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    assert args.runs>0 and not (outdir/"manifest.json").exists(), "use a fresh output directory"
    outdir.mkdir(parents=True,exist_ok=True)
    originals = hashes(ROOT)
    templates = prepare_variants(outdir/"templates",originals)
    variants = {}
    for name in ("current","tiled"):
        target = outdir/"variants"/name
        shutil.copytree(templates["int8_token_head"],target)
        variants[name] = target
    file = variants["tiled"]/"nanovllm/layers/quantized_attention.py"
    source = file.read_text()
    old = "        q_tiles = triton.cdiv(max_seqlen_q, 16)\n"
    assert source.count(old)==1
    source = source.replace(old,"        q_tiles = triton.cdiv(max_seqlen_q, 64)\n")
    old = "        16, 64, triton.next_power_of_2(head_dim), num_warps=4,\n"
    assert source.count(old)==1
    source = source.replace(old,
        "        16 if decode else 64, 64, triton.next_power_of_2(head_dim),\n"
        "        num_warps=4, num_stages=1,\n")
    file.write_text(source)
    configs = {"current":dict(prefill=[16,64,3],decode=[16,64,3]),
               "tiled":dict(prefill=[64,64,1],decode=[16,64,1])}
    for name,root in variants.items():
        file = root/"benchmark_inference_metrics.py"
        source = file.read_text()
        anchor = "    print('PREFILL_EXPERIMENT: int8_token_head', flush=True)\n"
        assert source.count(anchor)==1
        message = f"ATTENTION_TILE_CONFIGURATION: {name} {configs[name]}"
        source = source.replace(anchor,anchor+f"    print({message!r}, flush=True)\n")
        compile(source,str(file),"exec")
        file.write_text(source)
    variant_hashes = {name:hashes(root) for name,root in variants.items()}
    rng = random.Random(0)
    prompts = [[rng.randint(0,10000) for _ in range(rng.randint(100,1024))] for _ in range(256)]
    max_tokens = [rng.randint(100,1024) for _ in range(256)]
    workload = outdir/"workload.json"
    workload.write_text(json.dumps(dict(seed=0,prompts=prompts,max_tokens=max_tokens))+"\n")
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),model=args.model,
        python=PYTHON,gpu=args.gpu,runs=args.runs,configs=configs,
        original_source_hashes=originals,variant_source_hashes=variant_hashes,
        variant_roots={name:str(root) for name,root in variants.items()},
        input_tokens=sum(map(len,prompts)),output_tokens=sum(max_tokens),
        workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        model_config=json.loads((Path(args.model)/"config.json").read_text()),
        weight_files=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                      for p in Path(args.model).glob("*.safetensors")],
        config=dict(requests=256,lengths=[100,1024],seed=0,torch_seed=0,temperature=.6,
                    ignore_eos=True,kv_cache_dtype="int8_half",max_model_len=4096,
                    max_num_batched_tokens=16384,max_num_seqs=512,page_size=256,
                    gpu_memory_utilization=.9,prefill_warmup_lengths=[256,512,768,1024],
                    prefill_warmup_batches=[4,4,4,16],decode_cuda_graph=True),
        validation=[],smoke_runs=[],runs_completed=[])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,PYTHONUNBUFFERED="1")
    snapshot = outdir/"driver_snapshot"/"profiling"
    snapshot.mkdir(parents=True)
    for name in (Path(__file__).name,"run_prefill_kernel_metrics.py","run_scale_layout_metrics.py",
                 "run_kv_metrics_comparison.py","validate_prefill_kernel_route.py"):
        shutil.copyfile(ROOT/"profiling"/name,snapshot/name)
    def save():
        (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    def unchanged():
        assert hashes(ROOT)==originals
        assert all(hashes(root)==variant_hashes[name] for name,root in variants.items())
    save()
    for name,root in variants.items():
        environment = dict(env,PYTHONPATH=str(root))
        log = outdir/f"validation_{name}.txt"
        output = outdir/f"validation_{name}.json"
        command = [PYTHON,str(ROOT/"profiling/validate_prefill_kernel_route.py"),
                   "--layout","token_head","--output",str(output)]
        with log.open("w") as fp:
            completed = subprocess.run(command,cwd=root,env=environment,stdout=fp,
                                       stderr=subprocess.STDOUT,timeout=180)
        assert completed.returncode==0,f"validation failed: {log}"
        record = json.loads(output.read_text())
        assert record["package"]==str(root/"nanovllm/layers/attention.py")
        manifest["validation"].append(dict(variant=name,**record))
        print("INTEGRATION PASS",name,flush=True)
        save()
        unchanged()
    trials = []
    for trial in range(1,args.runs+1):
        order = ("current","tiled") if trial%2 else ("tiled","current")
        for name in order:
            unchanged()
            tempdir = outdir/f"run_{name}_{trial}"
            tempdir.mkdir()
            print("MODEL START",name,trial,flush=True)
            row,record = run("int8_token_head",variants[name],tempdir,args.model,env,trial)
            assert row["input_tokens"]==manifest["input_tokens"]
            assert row["output_tokens"]==manifest["output_tokens"]
            row.update(variant=name,log=str((tempdir/row["log"]).relative_to(outdir)))
            trials.append(row)
            manifest["runs_completed"].append(dict(**record,actual_variant=name))
            medians = {name:aggregate([r for r in trials if r["variant"]==name])["token_head"]
                       for name in ("current","tiled") if any(r["variant"]==name for r in trials)}
            (outdir/"results.json").write_text(json.dumps(dict(trials=trials,medians=medians),indent=2)+"\n")
            save()
            unchanged()
    assert len({r["blocks"] for r in trials})==1
    assert len({r["prefill_shape_sha256"] for r in trials})==1
    assert len({r["stages"]["prefill"]["tokens"] for r in trials})==1
    assert len({r["stages"]["decode"]["tokens"] for r in trials})==1
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),original_repository_unchanged=True)
    save()
    results = dict(trials=trials,medians=medians)
    render(outdir,manifest,results)
    print("COMPLETE",outdir/"report.md",flush=True)


if __name__ == "__main__":
    main()
