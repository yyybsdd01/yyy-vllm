
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
