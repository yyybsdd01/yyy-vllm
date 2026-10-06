"""Run paired fresh-process BF16/current INT8-half inference measurements."""

import argparse
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
from datetime import datetime, timezone
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xgd/anaconda3/envs/nanovllm/bin/python")


def parse_metrics(text):
    match = re.search(r"GPU: (.*?) \| graph: (\w+) \| KV: (\w+) \| blocks: (\d+) "
                      r"\| requests: (\d+) \| seed: (\d+)", text)
    if not match:
        raise ValueError("missing benchmark summary")
    result = dict(gpu=match[1], graph=match[2] == "True", mode=match[3],
                  blocks=int(match[4]), requests=int(match[5]), seed=int(match[6]))
    match = re.search(r"Input: (\d+) tokens \| Output: (\d+) tokens \| "
                      r"elapsed: ([\d.]+) s \| length check: PASS", text)
    if not match:
        raise ValueError("output length validation failed or missing")
    result.update(input_tokens=int(match[1]), output_tokens=int(match[2]), elapsed_s=float(match[3]))
    result["latency_ms"] = {}
    for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
        match = re.search(r"^" + re.escape(metric) + r"\s+([\d.]+)\s+([\d.]+)\s+"
                          r"([\d.]+)\s+([\d.]+)\s+ms/", text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"missing {metric}")
        result["latency_ms"][metric] = dict(zip(("mean", "p50", "p95", "p99"),
                                                  (float(v) for v in match.groups())))
    for label, key in (("Requests/s", "requests_per_s"), ("Input tokens/s", "input_tokens_per_s"),
                       ("Output tokens/s", "output_tokens_per_s"), ("Total tokens/s", "total_tokens_per_s"),
                       ("Peak PyTorch allocated", "peak_allocated_gib"),
                       ("Peak PyTorch reserved", "peak_reserved_gib")):
        match = re.search(r"^" + re.escape(label) + r": ([\d.]+)", text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"missing {label}")
        result[key] = float(match[1])
    result["stages"] = {}
    for name in ("Prefill", "Decode"):
        match = re.search(name + r" model-run tokens/s: ([\d.]+) \((\d+) tokens / ([\d.]+) s\)", text)
        if not match:
            raise ValueError(f"missing stage {name}")
        result["stages"][name.lower()] = dict(tokens_per_s=float(match[1]),
                                               tokens=int(match[2]), seconds=float(match[3]))
    return result


def source_hashes():
    paths = sorted((ROOT / "nanovllm").rglob("*.py")) + [ROOT / "benchmark_inference_metrics.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def aggregate(results):
    output = {}
    for mode in ("auto", "int8_half"):
        trials = [row for row in results if row["mode"] == mode]
        if not trials:
            continue
        med = lambda getter: statistics.median(getter(row) for row in trials)
        output[mode] = dict(runs=len(trials), blocks=med(lambda row: row["blocks"]),
                            input_tokens=trials[0]["input_tokens"], output_tokens=trials[0]["output_tokens"])
        for key in ("elapsed_s", "requests_per_s", "input_tokens_per_s", "output_tokens_per_s",
                    "total_tokens_per_s", "peak_allocated_gib", "peak_reserved_gib"):
            output[mode][key] = med(lambda row: row[key])
        output[mode]["latency_ms"] = {
            metric: {p: med(lambda row: row["latency_ms"][metric][p])
                     for p in ("mean", "p50", "p95", "p99")}
            for metric in trials[0]["latency_ms"]
        }
        output[mode]["stages"] = {
            stage: {key: med(lambda row: row["stages"][stage][key])
                    for key in ("tokens_per_s", "tokens", "seconds")}
            for stage in ("prefill", "decode")
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    assert args.runs > 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    assert not (args.outdir / "manifest.json").exists(), "use a fresh output directory"
    hashes = source_hashes()
    snapshot = args.outdir / "source_snapshot"
    for relative_path in hashes:
        destination = snapshot / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative_path, destination)
    rng = random.Random(0)
    prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(100, 1024))] for _ in range(256)]
    max_tokens = [rng.randint(100, 1024) for _ in range(256)]
    workload = dict(seed=0, prompts=prompts, max_tokens=max_tokens)
    workload_path = args.outdir / "workload.json"
    workload_path.write_text(json.dumps(workload) + "\n")
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), model=args.model,
                    python=str(PYTHON), gpu=args.gpu, runs_per_mode=args.runs,
                    modes=["auto", "int8_half"], source_hashes=hashes,
                    workload_sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest(),
                    config=dict(requests=256, min_input=100, max_input=1024, min_output=100,
                                max_output=1024, seed=0, max_model_len=4096, cuda_graph=True,
                                max_num_batched_tokens=16384, max_num_seqs=512,
                                gpu_memory_utilization=0.9, tensor_parallel_size=1,
                                kvcache_block_size=256, temperature=0.6, ignore_eos=True),
                    input_tokens=sum(map(len, prompts)), expected_output_tokens=sum(max_tokens),
                    pytorch_sampling_seed="not set, matching the existing benchmark protocol",
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"],text=True),
                    runs=[])
    model_config = Path(args.model) / "config.json"
    manifest["model_config"] = json.loads(model_config.read_text())
    manifest["weight_files"] = [dict(name=p.name, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                for p in Path(args.model).glob("*.safetensors")]
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    results = []
    for trial in range(1, args.runs + 1):
        order = ("auto", "int8_half") if trial % 2 else ("int8_half", "auto")
        for mode in order:
            assert source_hashes() == hashes, "source changed while benchmarking"
            command = [str(PYTHON), "benchmark_inference_metrics.py", "--model", args.model,
                       "--kv-cache-dtype", mode]
            log = args.outdir / f"{mode}_{trial}.txt"
            print(f"START trial={trial}/{args.runs} mode={mode} log={log}", flush=True)
            started = perf_counter()
            with log.open("w") as output:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, timeout=300)
            wall = perf_counter() - started
            assert completed.returncode == 0, f"trial failed: {log}"
            assert source_hashes() == hashes, "source changed while benchmarking"
            row = parse_metrics(log.read_text())
            assert row["input_tokens"] == manifest["input_tokens"]
            assert row["output_tokens"] == manifest["expected_output_tokens"]
            row.update(trial=trial, process_wall_s=wall, log=log.name)
            results.append(row)
            manifest["runs"].append(dict(trial=trial, mode=mode, command=command,
                                         log=log.name, process_wall_s=wall, returncode=completed.returncode))
            (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (args.outdir / "results.json").write_text(json.dumps(dict(trials=results,
                                                                     medians=aggregate(results)), indent=2) + "\n")
            print(f"DONE  trial={trial} mode={mode} generate={row['elapsed_s']:.3f}s "
                  f"output={row['output_tokens_per_s']:.2f}tok/s "
                  f"TTFT_P50={row['latency_ms']['TTFT']['p50']:.2f}ms "
                  f"TPOT_P50={row['latency_ms']['TPOT']['p50']:.2f}ms "
                  f"length_check=PASS", flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE: " + str(args.outdir.resolve()), flush=True)


if __name__ == "__main__":
    main()
