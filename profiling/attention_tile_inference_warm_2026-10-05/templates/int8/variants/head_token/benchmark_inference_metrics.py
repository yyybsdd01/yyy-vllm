"""Measure the offline inference baseline used by bench.py, including latency.

Run each trial in a fresh process so prefix-cache state is identical.
"""

import argparse
import hashlib
import json
import nanovllm
import nanovllm.layers.attention as attention_module
import random
import statistics
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/xgd/huggingface/Qwen3-0.6B")
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--min-input", type=int, default=100)
    parser.add_argument("--max-input", type=int, default=1024)
    parser.add_argument("--min-output", type=int, default=100)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8", "int8_dequant",
                                                     "int8_half", "int8_half_dequant"), default="auto")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    print("SCALE_LAYOUT: head_token | package: " + nanovllm.__file__, flush=True)
    torch.manual_seed(args.seed)
    assert args.requests > 0 and 0 < args.min_input <= args.max_input
    assert 1 < args.min_output <= args.max_output
    assert args.max_input + args.max_output <= args.max_model_len

    rng = random.Random(args.seed)
    prompts = [
        [rng.randint(0, 10000) for _ in range(rng.randint(args.min_input, args.max_input))]
        for _ in range(args.requests)
    ]
    params = [
        SamplingParams(temperature=0.6, ignore_eos=True,
                       max_tokens=rng.randint(args.min_output, args.max_output))
        for _ in range(args.requests)
    ]
    input_tokens = sum(map(len, prompts))
    expected_output_tokens = sum(p.max_tokens for p in params)

    llm = LLM(args.model, enforce_eager=args.eager, max_model_len=args.max_model_len,
              kv_cache_dtype=args.kv_cache_dtype)
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    # Warm page-table strides 1/2/3/4 and a full 16K-token prefill batch.
    # Warmup IDs exceed the formal workload range, preventing prefix hits.
    for warm_length in (256, 512, 768, 1024):
        warm_batch = 16 if warm_length == 1024 else 4
        warm_prompts = [[15000 + warm_length + request] * warm_length
                        for request in range(warm_batch)]
        llm.generate(warm_prompts, SamplingParams(max_tokens=2, ignore_eos=True), use_tqdm=False)
    print('Representative prefill warmup: page-table strides 1/2/3/4 and 16384 tokens', flush=True)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    print('PREFILL_EXPERIMENT: int8_head_token', flush=True)
    prefill_dispatch = {'flash': 0, 'int8': 0}
    prefill_shapes = []
    original_int8_attention = attention_module.int8_paged_attention
    original_flash_attention = attention_module.flash_attn_varlen_func
    def audited_int8_attention(*call_args, **kwargs):
        if attention_module.get_context().is_prefill:
            prefill_dispatch['int8'] += 1
        return original_int8_attention(*call_args, **kwargs)
    def audited_flash_attention(*call_args, **kwargs):
        prefill_dispatch['flash'] += 1
        return original_flash_attention(*call_args, **kwargs)
    attention_module.int8_paged_attention = audited_int8_attention
    attention_module.flash_attn_varlen_func = audited_flash_attention
    token_times: dict[int, list[float]] = {}
    original_add_request = llm.add_request

    def timed_add_request(prompt, sampling_params):
        original_add_request(prompt, sampling_params)
        token_times[llm.scheduler.waiting[-1].seq_id] = []

    llm.add_request = timed_add_request

    # Timestamps are taken after each synchronous model step has returned its
    # sampled tokens and the scheduler has applied them to the request states.
    original_postprocess = llm.scheduler.postprocess

    def timed_postprocess(seqs, token_ids, is_prefill):
        previous = [seq.num_completion_tokens for seq in seqs]
        original_postprocess(seqs, token_ids, is_prefill)
        timestamp = perf_counter()
        for seq, old_count in zip(seqs, previous):
            if seq.num_completion_tokens > old_count:
                token_times[seq.seq_id].append(timestamp)

    llm.scheduler.postprocess = timed_postprocess

    stage_seconds = {"prefill": 0.0, "decode": 0.0}
    stage_tokens = {"prefill": 0, "decode": 0}
    original_call = llm.model_runner.call

    def timed_call(method_name, *call_args):
        if method_name != "run":
            return original_call(method_name, *call_args)
        seqs, is_prefill = call_args
        stage = "prefill" if is_prefill else "decode"
        count = sum(seq.num_scheduled_tokens for seq in seqs)
        if is_prefill:
            prefill_shapes.append([[seq.num_cached_tokens, seq.num_scheduled_tokens]
                                   for seq in seqs])
        started = perf_counter()
        result = original_call(method_name, *call_args)
        stage_seconds[stage] += perf_counter() - started
        stage_tokens[stage] += count
        return result

    llm.model_runner.call = timed_call

    # All requests arrive together, as in the repository's original bench.py.
    started = perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = perf_counter() - started
    actual_output_tokens = sum(len(output["token_ids"]) for output in outputs)
    output_digest = hashlib.sha256(json.dumps([output['token_ids'] for output in outputs], separators=(',', ':')).encode()).hexdigest()
    assert len(outputs) == args.requests
    assert actual_output_tokens == expected_output_tokens
    assert [len(output["token_ids"]) for output in outputs] == [p.max_tokens for p in params]
    assert all(len(times) == p.max_tokens for times, p in zip(token_times.values(), params))

    ttft_ms = [(times[0] - started) * 1000 for times in token_times.values()]
    latency_ms = [(times[-1] - started) * 1000 for times in token_times.values()]
    tpot_ms = [(times[-1] - times[0]) * 1000 / (len(times) - 1)
               for times in token_times.values()]
    itl_ms = [(later - earlier) * 1000
              for times in token_times.values()
              for earlier, later in zip(times, times[1:])]

    def row(name: str, values: list[float], unit: str) -> None:
        print(f"{name:<28} {statistics.mean(values):>10.2f} {percentile(values, 50):>10.2f} "
              f"{percentile(values, 95):>10.2f} {percentile(values, 99):>10.2f} {unit}")

    print('Output token SHA256: ' + output_digest)
    print('Prefill dispatch audit: ' + json.dumps(prefill_dispatch))
    print('Prefill shape SHA256: ' + hashlib.sha256(json.dumps(prefill_shapes).encode()).hexdigest())
    print('Prefill batches: ' + str(len(prefill_shapes)))
    print(f"GPU: {torch.cuda.get_device_name()} | graph: {not args.eager} | "
          f"KV: {args.kv_cache_dtype} | blocks: {llm.model_runner.config.num_kvcache_blocks} | "
          f"requests: {args.requests} | seed: {args.seed}")
    print(f"Input: {input_tokens} tokens | Output: {actual_output_tokens} tokens | "
          f"elapsed: {elapsed:.3f} s | length check: PASS")
    print(f"{'Metric':<28} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} Unit")
    row("TTFT", ttft_ms, "ms/request")
    row("TPOT", tpot_ms, "ms/request/token")
    row("ITL", itl_ms, "ms/token")
    row("End-to-end latency", latency_ms, "ms/request")
    print(f"Requests/s: {args.requests / elapsed:.2f}")
    print(f"Input tokens/s: {input_tokens / elapsed:.2f}")
    print(f"Output tokens/s: {actual_output_tokens / elapsed:.2f}")
    print(f"Total tokens/s: {(input_tokens + actual_output_tokens) / elapsed:.2f}")
    for stage in ("prefill", "decode"):
        print(f"{stage.capitalize()} model-run tokens/s: "
              f"{stage_tokens[stage] / stage_seconds[stage]:.2f} "
              f"({stage_tokens[stage]} tokens / {stage_seconds[stage]:.3f} s)")
    gib = 1024 ** 3
    print(f"Peak PyTorch allocated: {torch.cuda.max_memory_allocated() / gib:.2f} GiB")
    print(f"Peak PyTorch reserved: {torch.cuda.max_memory_reserved() / gib:.2f} GiB")


if __name__ == "__main__":
    main()
