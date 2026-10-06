"""Measure full-model metrics using isolated token/head and head/token copies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_kv_metrics_comparison import parse_metrics

PYTHON = "/home/xgd/anaconda3/envs/nanovllm/bin/python"
LAYOUTS = ("token_head", "head_token")


def hashes(root):
    paths = sorted((root/"nanovllm").rglob("*.py"))+[root/"benchmark_inference_metrics.py"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def prepare_copies(outdir, originals):
    roots = {}
    for layout in LAYOUTS:
        target = outdir/"variants"/layout
        for relative in originals:
            destination = target/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, destination)
        # Shared benchmark changes: reproducible sampling and an output digest.
        file = target/"benchmark_inference_metrics.py"
        source = file.read_text().replace("import argparse\n", "import argparse\nimport hashlib\nimport json\nimport nanovllm\n")
        source = source.replace("    args = parser.parse_args()\n",
                                f'    args = parser.parse_args()\n    print("SCALE_LAYOUT: {layout} | package: " + nanovllm.__file__, flush=True)\n    torch.manual_seed(args.seed)\n')
        source = source.replace("    torch.cuda.reset_peak_memory_stats()\n",
                                "    torch.manual_seed(args.seed)\n    torch.cuda.reset_peak_memory_stats()\n")
        source = source.replace("    actual_output_tokens = sum(len(output[\"token_ids\"]) for output in outputs)\n",
                                "    actual_output_tokens = sum(len(output[\"token_ids\"]) for output in outputs)\n"
                                "    output_digest = hashlib.sha256(json.dumps([output['token_ids'] for output in outputs], separators=(',', ':')).encode()).hexdigest()\n")
        source = source.replace("    print(f\"GPU: ",
                                "    print('Output token SHA256: ' + output_digest)\n    print(f\"GPU: ")
        file.write_text(source)
        if layout == "head_token":
            file = target/"nanovllm/layers/attention.py"
            source = file.read_text()
            old_import = "from nanovllm.layers.quantized_attention import int8_paged_attention"
            assert source.count(old_import) == 1
            source = source.replace(old_import,
                                    "from nanovllm.layers.head_major_scale_attention import (\n"
                                    "    int8_paged_attention_head_major as int8_paged_attention,\n"
                                    "    store_kvcache_int8_head_major,\n)")
            old_call = "store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)"
            assert source.count(old_call) == 1
            source = source.replace(old_call, old_call.replace("store_kvcache_int8(", "store_kvcache_int8_head_major("))
            file.write_text(source)
            file = target/"nanovllm/engine/model_runner.py"
            source = file.read_text()
            old_shape = ("self.kv_scales = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,\n"
                         "                                         self.block_size, num_kv_heads * scale_groups, dtype=torch.float32)")
            assert source.count(old_shape) == 1
            source = source.replace(old_shape,
                                    "assert config.kv_cache_dtype == 'int8_half', 'isolated head/token experiment requires dual scales'\n"
                                    "            self.kv_scales = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,\n"
                                    "                                         num_kv_heads, self.block_size, scale_groups, dtype=torch.float32)")
            file.write_text(source)
        roots[layout] = target
    return roots


def aggregate(rows):
    output = {}
    for layout in LAYOUTS:
        trials = [row for row in rows if row["layout"] == layout]
        if not trials:
            continue
        med = lambda fn: statistics.median(fn(row) for row in trials)
        out = dict(runs=len(trials), input_tokens=trials[0]["input_tokens"],
                   output_tokens=trials[0]["output_tokens"])
        for key in ("blocks", "elapsed_s", "requests_per_s", "input_tokens_per_s",
                    "output_tokens_per_s", "total_tokens_per_s", "peak_allocated_gib", "peak_reserved_gib"):
            out[key] = med(lambda row: row[key])
        out["latency_ms"] = {
            metric: {q: med(lambda row: row["latency_ms"][metric][q])
                     for q in ("mean", "p50", "p95", "p99")}
            for metric in trials[0]["latency_ms"]}
        out["stages"] = {stage: {key: med(lambda row: row["stages"][stage][key])
                                 for key in ("tokens_per_s", "tokens", "seconds")}
                          for stage in ("prefill", "decode")}
        output[layout] = out
    return output


def run(layout, variant, outdir, model, env, trial, smoke=False):
    command = [PYTHON, "benchmark_inference_metrics.py", "--model", model,
               "--kv-cache-dtype", "int8_half"]
    if smoke:
        command += ["--requests", "4", "--min-input", "32", "--max-input", "32",
                    "--min-output", "16", "--max-output", "16"]
    name = f"{layout}_smoke.txt" if smoke else f"{layout}_{trial}.txt"
    log = outdir/name
    print(f"START {'smoke' if smoke else 'trial='+str(trial)} layout={layout} log={log}", flush=True)
    environment = env.copy()
    environment["PYTHONPATH"] = str(variant)
    started = perf_counter()
    with log.open("w") as output:
        result = subprocess.run(command, cwd=variant, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, timeout=300)
    assert result.returncode == 0, f"benchmark failed: {log}"
    text = log.read_text()
    assert f"SCALE_LAYOUT: {layout} | package: {variant}/nanovllm/__init__.py" in text
    row = parse_metrics(text)
    match = re.search(r"Output token SHA256: ([a-f0-9]{64})", text)
    assert match, "missing output digest"
    row.update(layout=layout, trial=trial, log=name, output_sha256=match[1],
               process_wall_s=perf_counter()-started)
    record = dict(layout=layout, trial=trial, log=name, cwd=str(variant), command=command,
                  process_wall_s=row["process_wall_s"], returncode=result.returncode)
    print(f"DONE layout={layout} generate={row['elapsed_s']:.3f}s "
          f"output={row['output_tokens_per_s']:.2f}tok/s TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms "
          f"TPOT_P50={row['latency_ms']['TPOT']['p50']:.2f}ms blocks={row['blocks']} length_check=PASS", flush=True)
    return row, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    assert args.runs > 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    assert not (args.outdir/"manifest.json").exists(), "use a fresh output directory"
    original_hashes = hashes(ROOT)
    variants = prepare_copies(args.outdir, original_hashes)
    variant_hashes = {layout: hashes(path) for layout, path in variants.items()}
    rng = random.Random(0)
    prompts = [[rng.randint(0,10000) for _ in range(rng.randint(100,1024))] for _ in range(256)]
    max_tokens = [rng.randint(100,1024) for _ in range(256)]
    workload_path = args.outdir/"workload.json"
    workload_path.write_text(json.dumps(dict(seed=0,prompts=prompts,max_tokens=max_tokens))+"\n")
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), model=args.model,
                    python=PYTHON, gpu=args.gpu, runs_per_layout=args.runs,
                    original_source_hashes=original_hashes, variant_source_hashes=variant_hashes,
                    variant_roots={layout:str(path) for layout,path in variants.items()},
                    workload_sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest(),
                    input_tokens=sum(map(len,prompts)), output_tokens=sum(max_tokens),
                    config=dict(requests=256,min_input=100,max_input=1024,min_output=100,max_output=1024,
                                seed=0,torch_sampling_seed=0,max_model_len=4096,cuda_graph=True,
                                max_num_batched_tokens=16384,max_num_seqs=512,gpu_memory_utilization=.9,
                                tensor_parallel_size=1,kvcache_block_size=256,kv_cache_dtype="int8_half",
                                temperature=.6,ignore_eos=True),
                    model_config=json.loads((Path(args.model)/"config.json").read_text()),
                    weight_files=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                                  for p in Path(args.model).glob("*.safetensors")],
                    git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                    nvidia_smi=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem","--format=csv,noheader"],text=True),
                    smoke_runs=[],runs=[])
    def save():
        (args.outdir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    def unchanged():
        assert hashes(ROOT)==original_hashes,"repository source changed"
        assert all(hashes(variants[layout])==variant_hashes[layout] for layout in LAYOUTS),"isolated source changed"
    save()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,PYTHONUNBUFFERED="1")
    smokes = []
    for layout in LAYOUTS:
        unchanged()
        row,record = run(layout,variants[layout],args.outdir,args.model,env,0,smoke=True)
        manifest["smoke_runs"].append(dict(**record,metrics=row))
        smokes.append(row)
        save()
    assert smokes[0]["output_sha256"]==smokes[1]["output_sha256"],"smoke outputs differ"
    assert smokes[0]["blocks"]==smokes[1]["blocks"],"KV capacity differs"
    results = []
    for trial in range(1,args.runs+1):
        order = LAYOUTS if trial%2 else LAYOUTS[::-1]
        for layout in order:
            unchanged()
            row,record = run(layout,variants[layout],args.outdir,args.model,env,trial)
            assert row["input_tokens"]==manifest["input_tokens"]
            assert row["output_tokens"]==manifest["output_tokens"]
            results.append(row)
            manifest["runs"].append(record)
            save()
            (args.outdir/"results.json").write_text(json.dumps(dict(trials=results,medians=aggregate(results)),indent=2)+"\n")
            unchanged()
    assert len({row["output_sha256"] for row in results})==1,"seeded outputs differ across layouts/trials"
    assert len({row["blocks"] for row in results})==1,"KV capacity differs across trials"
    manifest["all_generated_outputs_identical"] = True
    manifest["original_repository_unchanged"] = True
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("COMPLETE",args.outdir,flush=True)


if __name__=="__main__":
    main()
