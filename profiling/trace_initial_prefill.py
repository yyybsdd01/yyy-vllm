"""Trace initial prefill batches in existing isolated BF16/INT8 variants.

Only benchmark instrumentation is changed in memory. The prompt workload and
warmup match the paired comparison; output is shortened to two tokens because
later decoding does not affect the initial prefill phase.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_scale_layout_metrics import PYTHON, hashes
from profiling.run_kv_metrics_comparison import parse_metrics


CHILD = r'''
import os
from pathlib import Path

path = Path.cwd() / "benchmark_inference_metrics.py"
source = path.read_text()
old = "    token_times: dict[int, list[float]] = {}\n"
assert source.count(old) == 1
source = source.replace(old, old + "    initial_prefill_trace = []\n")
old = "        started = perf_counter()\n        result = original_call(method_name, *call_args)\n        stage_seconds[stage] += perf_counter() - started\n"
assert source.count(old) == 1
source = source.replace(old, """        step_started = perf_counter()
        result = original_call(method_name, *call_args)
        duration = perf_counter() - step_started
        stage_seconds[stage] += duration
        if is_prefill and all(seq.num_completion_tokens == 0 for seq in seqs):
            initial_prefill_trace.append(dict(
                batch=len(initial_prefill_trace) + 1,
                requests=len(seqs), tokens=count,
                model_run_ms=duration * 1000,
                initial_cached_tokens=sum(seq.num_cached_tokens for seq in seqs)))
""")
old = "        timestamp = perf_counter()\n"
assert source.count(old) == 1
source = source.replace(old, old + """        if is_prefill and all(old_count == 0 for old_count in previous):
            record = initial_prefill_trace[-1]
            record['first_token_elapsed_ms'] = (timestamp - started) * 1000
            record['cumulative_requests'] = sum(r['requests'] for r in initial_prefill_trace)
""")
old = "    print('Output token SHA256: ' + output_digest)\n"
assert source.count(old) == 1
source = source.replace(old, old + "    print('INITIAL_PREFILL_TRACE: ' + json.dumps(initial_prefill_trace))\n")
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'profiling/fast_int8_vs_bf16_paired_2026-10-05')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    prior = json.loads((args.reference / 'manifest.json').read_text())
    original_hashes = hashes(ROOT)
    assert original_hashes == prior['original_source_hashes']
    variants = {name: Path(prior['variant_roots'][name]) for name in ('bf16', 'fast_int8')}
    for name, root in variants.items():
        assert hashes(root) == prior['variant_source_hashes'][name]
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    child = outdir / 'trace_trial.py'
    child.write_text(CHILD)
    compile(CHILD, str(child), 'exec')
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), reference=str(args.reference.resolve()),
                    rounds=args.rounds, output_tokens_per_request=2,
                    original_source_hashes=original_hashes, variant_source_hashes=prior['variant_source_hashes'])
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for trial in range(1, args.rounds + 1):
        order = ('bf16', 'fast_int8') if trial % 2 else ('fast_int8', 'bf16')
        for name in order:
            print(f'START {name} {trial}', flush=True)
            root = variants[name]
            command = [PYTHON, str(child), '--model', prior['model'], '--kv-cache-dtype',
                       'auto' if name == 'bf16' else 'int8_half', '--min-output', '2', '--max-output', '2']
            log = outdir / f'{name}_{trial}.txt'
            with log.open('w') as output:
                result = subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root),
                                        CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1'),
                                        stdout=output, stderr=subprocess.STDOUT, timeout=300)
            assert result.returncode == 0, f'Failed: {log}'
            text = log.read_text()
            assert f'package: {root}/nanovllm/__init__.py' in text
            trace = json.loads(re.search(r'INITIAL_PREFILL_TRACE: (.*)', text)[1])
            assert len(trace) == 9 and sum(r['requests'] for r in trace) == 256
            assert sum(r['tokens'] for r in trace) == prior['input_tokens']
            assert all(r['initial_cached_tokens'] == 0 for r in trace)
            row = dict(variant=name, trial=trial, trace=trace, metrics=parse_metrics(text), command=command)
            rows.append(row)
            (outdir / 'results.json').write_text(json.dumps(rows, indent=2))
            print(f"DONE {name} {trial}: P50={row['metrics']['latency_ms']['TTFT']['p50']:.2f}ms "
                  f"initial model-run={sum(r['model_run_ms'] for r in trace):.2f}ms", flush=True)
    assert hashes(ROOT) == original_hashes
    for name, root in variants.items():
        assert hashes(root) == prior['variant_source_hashes'][name]
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), original_repository_unchanged=True)
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    lines = ['# 首次 prefill 分批耗时', '',
             '与完整对照相同的 256 个输入、同样预热，输出缩短为每请求 2 token。计时不含初始化和预热。',
             'Model-run 包含输入准备、模型、采样和同步；完成时间从整批提交开始。两轮顺序交替，取各项中位数。', '',
             '| 批次 | 请求数 | 累计请求 | BF16 model-run ms | INT8 model-run ms | BF16 首 token 完成 ms | INT8 首 token 完成 ms | 累计差 ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i in range(9):
        grouped = {name: [row['trace'][i] for row in rows if row['variant'] == name] for name in variants}
        med = lambda name, key: statistics.median(r[key] for r in grouped[name])
        a, b = [med(name, 'first_token_elapsed_ms') for name in variants]
        reference = grouped['bf16'][0]
        lines.append(f"| {i+1} | {reference['requests']} | {reference['cumulative_requests']} | "
                     f"{med('bf16', 'model_run_ms'):.2f} | {med('fast_int8', 'model_run_ms'):.2f} | "
                     f"{a:.2f} | {b:.2f} | {b-a:+.2f} |")
    lines += ['', '该诊断只分析首次 prefill；不能用缩短后的生成时间或吞吐替代此前完整 100–1024 token 输出的结果。',
              '正式源码和对照副本均未修改。']
    (outdir / 'report.md').write_text('\n'.join(lines) + '\n')
    print('COMPLETE', outdir / 'report.md', flush=True)


if __name__ == '__main__':
    main()
