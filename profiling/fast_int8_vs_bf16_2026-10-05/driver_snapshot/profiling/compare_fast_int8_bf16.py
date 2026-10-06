"""Fresh paired BF16/fast dual-scale INT8 inference and full-corpus PPL."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from profiling.run_scale_layout_metrics import PYTHON,hashes,aggregate
from profiling.run_kv_metrics_comparison import parse_metrics


def render(outdir,manifest,results):
    medians,quality = results["medians"],results["quality"]
    lines = ["# 较快 INT8 双 scale 组合与未量化 BF16 对照", "",
        "BF16：原 FlashAttention prefill/decode，模型权重与 KV 均 BF16。",
        "较快 INT8：模型权重 BF16；KV INT8，每 token/head 两个 FP32 scale，布局 block/token/head/2；"
        "prefill BM=64、BN=64、num_stages=1，decode BM=16、BN=64、num_stages=1，均 BD=128、4 warps。", "",
        "## 新一轮完整模型结果", "",
        "| 指标 | BF16 | 较快 INT8 | 变化 |", "| --- | ---: | ---: | ---: |"]
    def add(label,getter,digits=2):
        old,new = [getter(medians[name]) for name in ("bf16","fast_int8")]
        change = f"{(new/old-1)*100:+.2f}%" if old else "—"
        lines.append(f"| {label} | {old:.{digits}f} | {new:.{digits}f} | {change} |")
    for key in ("TTFT","TPOT","ITL","End-to-end latency"):
        for quantile in ("mean","p50","p95","p99"):
            add(f"{key} {quantile.upper()} ms",lambda m,k=key,q=quantile:m["latency_ms"][k][q])
    for label,key,digits in (("generate s","elapsed_s",3),("输出 token/s","output_tokens_per_s",2),
                             ("请求/s","requests_per_s",2),("输入 token/s","input_tokens_per_s",2),
                             ("输入+输出 token/s","total_tokens_per_s",2),
                             ("KV blocks","blocks",0),("峰值 allocated GiB","peak_allocated_gib",2),
                             ("峰值 reserved GiB","peak_reserved_gib",2),("缓存抢占次数","preemptions",0)):
        add(label,lambda m,k=key:m[k],digits)
    for stage in ("prefill","decode"):
        for key,digits in (("seconds",3),("tokens",0),("tokens_per_s",2)):
            add(f"{stage} model-run {key}",lambda m,s=stage,k=key:m["stages"][s][k],digits)
    lines += ["", "## 精度：WikiText-2 raw test 全集 PPL", "",
        "相同的 298,938 个下一 token 预测位置，4096-token 非重叠窗口、256-token 分块 teacher forcing。PPL 越低越好。",
        "INT8 在首次分块也读取量化后的 KV，匹配这次所有 prefill 使用自写 kernel 的方案。", "",
        "| 配置 | 全量 PPL | 相对 BF16 | 首块 PPL | 历史缓存部分 PPL |", "| --- | ---: | ---: | ---: | ---: |"]
    base = quality["bf16"]["ppl"]
    for name in ("bf16","original_int8","fast_int8"):
        value = quality[name]
        lines.append(f'| {name} | {value["ppl"]:.6f} | {(value["ppl"]/base-1)*100:+.4f}% | '
                     f'{value["fresh_ppl"]:.6f} | {value["cached_ppl"]:.6f} |')
    original,fast = quality["original_int8"],quality["fast_int8"]
    lines += ["",f'扩大 Q tile/调整流水后，相对原 INT8 配置 ΔPPL={fast["ppl"]-original["ppl"]:+.6f}，'
        f'ΔNLL/token={(fast["nll"]-original["nll"])/fast["tokens"]:+.8f}。', "",
        "这里的 PPL 衡量真实下一 token 的概率，不是问答正确率。PPL 使用多 token 的 prefill 路径；没有单独运行逐 token decode 的全文 PPL。", "",
        "## 每轮性能结果", "",
        "| 配置 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 | prefill tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results["trials"]:
        lines.append(f'| {row["variant"]} | {row["trial"]} | {row["elapsed_s"]:.3f} | '
            f'{row["output_tokens_per_s"]:.2f} | {row["latency_ms"]["TTFT"]["p50"]:.2f} | '
            f'{row["latency_ms"]["TPOT"]["p50"]:.2f} | {row["preemptions"]} | {row["stages"]["prefill"]["tokens"]} |')
    lines += ["", "## 条件和测量范围", "",
        "- Qwen3-0.6B、RTX 3090 Ti、单卡 GPU 0，BF16 模型权重；decode CUDA Graph、prefill eager。未锁 GPU 频率。",
        f'- 两组各 {manifest["runs"]} 个独立新进程，顺序交替，全是这次的新测量；编译、初始化、生成预热和代表 prefill 特化预热不计入正式时间。',
        "- 256 请求同时到达，输入/输出长度各 100–1024，seed=0、torch sampling seed=0、temperature=0.6、ignore_eos=True。输入 142,827、输出 133,966 token，逐请求长度检查通过。",
        "- 页表宽度 1/2/3/4 和 16K-token 大 prefill 在计时前预热。max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9。",
        "- 相同显存预算下 INT8 可分配更多 KV blocks。BF16 的阶段时间包含抢占后的重新 prefill；因此这些整模型结果包含 KV 容量、调度与 kernel 速度的共同影响。",
        "- 每 token 的 INT8 K/V+双 FP32 scale 占 BF16 KV 字节数的 53.125%；节约的空间用于扩大 KV 容量，完整缓存池和峰值显存不会自动减半。",
        "- TTFT 是整批提交到 CPU postprocess 完成首 token，含排队；TPOT/ITL 也由 CPU postprocess 时间戳计算。阶段时间包括准备、模型、采样和同步，不是单 attention kernel 时间。",
        "- PPL 语料 UTF-8 SHA256、token ID 摘要、模型配置与源码哈希、首块/历史块计数及逐层派发次数保存于 quality/*.json；三组使用相同文本和预测位置。",
        "- 该 PPL 是项目固定窗口协议，语料为英文，不能直接当作中文对话或任务准确率；没有聊天模板或额外 EOS。",
        "- 原 INT8 精度对照使用 BM=16、BN=64、num_stages=3，首次 prefill 同样读量化 KV。旧精度测量首块使用原始 BF16 KV，因协议不同不直接复用旧数值。",
        "- 本次 kernel 实现是上一轮通过恢复后 BF16 参考验证的同一份源码；所有正式源码哈希前后一致，改动仅在独立副本。", "",
        "## 复现", "", "```bash",
        "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_fast_int8_bf16.py --outdir /tmp/fast-int8-bf16",
        "```"]
    (outdir/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--reference",type=Path,default=ROOT/"profiling/attention_tile_inference_warm_2026-10-05")
    parser.add_argument("--text",type=Path,default=Path("/home/xgd/.cache/nanovllm_eval/wikitext2_test.txt"))
    parser.add_argument("--runs",type=int,default=3)
    parser.add_argument("--gpu",default="0")
    args = parser.parse_args()
    outdir,reference = args.outdir.resolve(),args.reference.resolve()
    assert args.runs>0 and args.text.is_file() and not (outdir/"manifest.json").exists()
    prior = json.loads((reference/"manifest.json").read_text())
    originals = hashes(ROOT)
    assert originals==prior["original_source_hashes"] and prior["original_repository_unchanged"]
    outdir.mkdir(parents=True,exist_ok=True)
    variants = {}
    for name,source in (("bf16","current"),("fast_int8","tiled"),("original_int8","current")):
        source_root = Path(prior["variant_roots"][source])
        assert hashes(source_root)==prior["variant_source_hashes"][source]
        root = outdir/"variants"/name
        shutil.copytree(source_root,root)
        variants[name] = root
        file = root/"benchmark_inference_metrics.py"
        code = file.read_text()
        old = "    token_times: dict[int, list[float]] = {}\n"
        assert code.count(old)==1
        code = code.replace(old,
            "    preemption_count = 0\n"
            "    original_preempt = llm.scheduler.preempt\n"
            "    def counted_preempt(seq):\n"
            "        nonlocal preemption_count\n"
            "        preemption_count += 1\n"
            "        return original_preempt(seq)\n"
            "    llm.scheduler.preempt = counted_preempt\n"+old)
        old = "    print('Output token SHA256: ' + output_digest)\n"
        assert code.count(old)==1
        code = code.replace(old,old+
            "    print('Preemptions: ' + str(preemption_count))\n"
            "    print('Actual KV dtype: ' + str(llm.model_runner.kv_cache.dtype))\n"
            "    print('KV tensor bytes: ' + str(llm.model_runner.kv_cache.numel()*llm.model_runner.kv_cache.element_size()))\n"
            "    print('Scale tensor bytes: ' + str(0 if llm.model_runner.kv_scales is None else llm.model_runner.kv_scales.numel()*llm.model_runner.kv_scales.element_size()))\n")
        compile(code,str(file),"exec")
        file.write_text(code)
    variant_hashes = {name:hashes(root) for name,root in variants.items()}
    shutil.copyfile(reference/"workload.json",outdir/"workload.json")
    snapshot = outdir/"driver_snapshot"/"profiling"
    snapshot.mkdir(parents=True)
    for name in (Path(__file__).name,"eval_attention_tile_quality.py","kv_cache_perplexity.py",
                 "run_scale_layout_metrics.py","run_kv_metrics_comparison.py"):
        shutil.copyfile(ROOT/"profiling"/name,snapshot/name)
    model = prior["model"]
    assert json.loads((Path(model)/"config.json").read_text())==prior["model_config"]
    for weight in prior["weight_files"]:
        stat = (Path(model)/weight["name"]).stat()
        assert stat.st_size==weight["size"] and stat.st_mtime_ns==weight["mtime_ns"]
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),reference=str(reference),
        model=model,model_config=prior["model_config"],weight_files=prior["weight_files"],
        python=PYTHON,gpu=args.gpu,runs=args.runs,configs=prior["configs"],config=prior["config"],
        original_source_hashes=originals,variant_source_hashes=variant_hashes,
        variant_roots={name:str(root) for name,root in variants.items()},
        input_tokens=prior["input_tokens"],output_tokens=prior["output_tokens"],
        workload_sha256=hashlib.sha256((outdir/"workload.json").read_bytes()).hexdigest(),
        text_sha256=hashlib.sha256(args.text.read_bytes()).hexdigest(),quality_protocol=dict(
            max_tokens=298938,window_size=4096,chunk_size=256,block_size=256,
            int8_first_chunk_quantized=True),runs_completed=[],quality_completed=[])
    results = dict(trials=[],medians={},quality={})
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,PYTHONUNBUFFERED="1")
    def save():
        (outdir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (outdir/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    def unchanged():
        assert hashes(ROOT)==originals
        assert all(hashes(root)==variant_hashes[name] for name,root in variants.items())
    save()
    for trial in range(1,args.runs+1):
        for name in (("bf16","fast_int8") if trial%2 else ("fast_int8","bf16")):
            unchanged()
            mode = "auto" if name=="bf16" else "int8_half"
            root = variants[name]
            log = outdir/f"{name}_{trial}.txt"
            command = [PYTHON,"benchmark_inference_metrics.py","--model",model,"--kv-cache-dtype",mode]
            environment = dict(env,PYTHONPATH=str(root))
            print("PERFORMANCE START",name,trial,flush=True)
            started = perf_counter()
            with log.open("w") as fp:
                completed = subprocess.run(command,cwd=root,env=environment,stdout=fp,
                                           stderr=subprocess.STDOUT,timeout=600)
            assert completed.returncode==0,f"performance failed: {log}"
            text = log.read_text()
            assert f"package: {root}/nanovllm/__init__.py" in text
            row = parse_metrics(text)
            assert row["mode"]==mode and row["graph"]
            assert row["input_tokens"]==prior["input_tokens"] and row["output_tokens"]==prior["output_tokens"]
            dispatch = json.loads(re.search(r"Prefill dispatch audit: (.*)",text)[1])
            assert (dispatch["int8"]==0 and dispatch["flash"]>0) if name=="bf16" else (dispatch["flash"]==0 and dispatch["int8"]>0)
            row.update(variant=name,layout="token_head",trial=trial,log=log.name,
                output_sha256=re.search(r"Output token SHA256: ([a-f0-9]{64})",text)[1],
                preemptions=int(re.search(r"Preemptions: (\d+)",text)[1]),prefill_dispatch=dispatch,
                kv_tensor_bytes=int(re.search(r"KV tensor bytes: (\d+)",text)[1]),
                scale_tensor_bytes=int(re.search(r"Scale tensor bytes: (\d+)",text)[1]),
                process_wall_s=perf_counter()-started)
            results["trials"].append(row)
            manifest["runs_completed"].append(dict(variant=name,trial=trial,command=command,log=log.name))
            for variant in ("bf16","fast_int8"):
                rows = [r for r in results["trials"] if r["variant"]==variant]
                if rows:
                    value = aggregate(rows)["token_head"]
                    value["preemptions"] = statistics.median(r["preemptions"] for r in rows)
                    results["medians"][variant] = value
            save()
            unchanged()
            print(f'PERFORMANCE DONE {name} {trial} output={row["output_tokens_per_s"]:.2f}tok/s '
                  f'TTFT={row["latency_ms"]["TTFT"]["p50"]:.2f}ms TPOT={row["latency_ms"]["TPOT"]["p50"]:.2f}ms '
                  f'blocks={row["blocks"]} preemptions={row["preemptions"]}',flush=True)
    qualitydir = outdir/"quality"
    qualitydir.mkdir()
    for name in ("bf16","original_int8","fast_int8"):
        unchanged()
        mode = "auto" if name=="bf16" else "int8_half"
        output = qualitydir/f"{name}.json"
        log = qualitydir/f"{name}.txt"
        command = [PYTHON,str(ROOT/"profiling/eval_attention_tile_quality.py"),"--model",model,
                   "--text",str(args.text),"--mode",mode,"--label",name,"--output",str(output)]
        print("QUALITY START",name,flush=True)
        with log.open("w") as fp:
            completed = subprocess.run(command,cwd=variants[name],env=dict(env,PYTHONPATH=str(variants[name])),
                                       stdout=fp,stderr=subprocess.STDOUT,timeout=1800)
        assert completed.returncode==0,f"quality failed: {log}"
        value = json.loads(output.read_text())
        assert value["package"]==str(variants[name]/"nanovllm/__init__.py")
        assert value["tokens"]==298938 and value["text_sha256"]==manifest["text_sha256"]
        results["quality"][name] = value
        manifest["quality_completed"].append(dict(variant=name,command=command,log=str(log.relative_to(outdir))))
        save()
        unchanged()
        print(f'QUALITY DONE {name} PPL={value["ppl"]:.6f} time={value["evaluation_wall_s"]:.1f}s',flush=True)
    assert len({v["token_ids_sha256"] for v in results["quality"].values()})==1
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),original_repository_unchanged=True)
    save()
    render(outdir,manifest,results)
    print("COMPLETE",outdir/"report.md",flush=True)


if __name__ == "__main__":
    main()
