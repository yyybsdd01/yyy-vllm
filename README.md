<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

Set `kv_cache_dtype="int8"` to store K/V as INT8 with one FP32 scale per token
and KV head. Decode and cached-prefix prefill use a Triton attention kernel
that restores K/V inside the kernel. Ordinary prefill still attends to the
newly computed FP16/BF16 K/V with FlashAttention. The default `"auto"` keeps
the model's FP16/BF16 cache. `"int8_dequant"` retains the earlier path that
restores the referenced cache into a shared FP16/BF16 buffer before calling
FlashAttention, for direct comparison. `"int8_half"` stores two FP32 scales
per token and KV head, one for each half of the head dimension, and reads them
inside the fused attention kernel. `"int8_half_dequant"` uses the same cache
format but restores BF16 K/V before FlashAttention. All five modes support
eager execution and CUDA Graph decoding.

```bash
python benchmark_inference_metrics.py --kv-cache-dtype auto
python benchmark_inference_metrics.py --kv-cache-dtype int8
python benchmark_inference_metrics.py --kv-cache-dtype int8_dequant
python benchmark_inference_metrics.py --kv-cache-dtype int8_half
python benchmark_inference_metrics.py --kv-cache-dtype int8_half_dequant
```

## Benchmark

See `bench.py` for benchmark.

Set `preemption_lock=True` when constructing `LLM` to try decode priority after
KV-cache preemption. `Scheduler.lock` becomes 1 whenever a request is preempted,
and returns to 0 when any request finishes. While locked, running requests decode
before waiting requests prefill. If no request can decode, prefill resumes to
restore progress. The option defaults to `False`, preserving prefill priority.

```python
llm = LLM("/YOUR/MODEL/PATH", preemption_lock=True)
```

```bash
python benchmark_inference_metrics.py --requests 1024 --preemption-lock
```

Set `kv_cpu_offload=True` to preserve preempted KV instead of recomputing it.
Restore admission precedes waiting-prefill admission; normal prefill still
precedes decode. Both restore and waiting-prefill admission leave enough free
blocks for the next decode batch's current block-growth requirements. This
reservation is recomputed each round; requests can still need additional blocks
while an asynchronous restore is in flight.
Requests in `OFFLOADING`, `WAITING_RESTORE`, or `RESTORING`
cannot decode. CUDA events promote restored requests to `RUNNING` and prevent
prefill, decode, or restore from overwriting a block before its pending KV read.
Shared prefix blocks stay on GPU until their GPU reference count reaches zero;
pending restore dependencies are tracked separately and share CPU snapshots.

```python
llm = LLM("/YOUR/MODEL/PATH", kv_cache_dtype="int8_half",
          kv_cpu_offload=True, offload_cpu_gb=4.0, offload_max_inflight=8)
```

```bash
python benchmark_inference_metrics.py --requests 1024 --kv-cache-dtype int8_half --kv-cpu-offload
python -m unittest discover -s tests -p 'test_kv_offload*.py' -v
```

Offload defaults to disabled and currently supports one GPU. It is mutually
exclusive with `preemption_lock`, which changes phase priority. BF16 and all
INT8 cache formats include every layer's K/V and their scales. The pinned CPU
pool is bounded by `offload_cpu_gb`; exhaustion falls back to recomputation and
is reported in offload statistics. Each copy direction reserves one GPU block
for staging, accounted for before KV cache allocation. A positive
`num_kvcache_blocks` caps cache capacity; the benchmark exposes it as `--kv-blocks`.
The existing attention kernel and its default warp count are unchanged.

Implementation tests and limitations are recorded in
[the offload validation report](profiling/async_kv_offload_2026-10-06/validation.md).
The original admission comparison uses the archived fast 8-warp kernel in an
isolated copy: throughput improves by 4.78%, while preemption count and TPOT
P95/P99 increase. Whole-model outputs are not token-for-token identical.
The matched three-trial comparison of original admission, decode-block
reservation, and offload disabled is recorded in
[the decode reservation report](profiling/kv_offload_decode_reserve_2026-10-06/report.md).

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
