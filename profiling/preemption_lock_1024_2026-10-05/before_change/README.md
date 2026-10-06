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
