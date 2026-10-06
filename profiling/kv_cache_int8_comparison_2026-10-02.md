# nano-vLLM KV Cache：默认精度与 INT8 对比（2026-10-02）

本报告记录的是旧版“先解量化到 BF16 缓冲区”的 INT8 路径；当前可用
`--kv-cache-dtype int8_dequant` 复现。融合 Triton 注意力内核的结果见
[后续对比](kv_cache_fused_comparison_2026-10-02.md)。

## 条件与口径

- 模型：本地 Qwen3-0.6B；GPU 0：NVIDIA GeForce RTX 3090 Ti；张量并行 1；CUDA Graph 开启。
- 每轮在全新进程运行 `benchmark_inference_metrics.py --kv-cache-dtype auto` 或 `--kv-cache-dtype int8`。顺序为 auto、int8、int8、auto、auto、int8；每种模式各 3 轮。
- 与 [2026-10-01 原版基线](inference_baseline_2026-10-01.md)使用相同工作负载：种子 0，一次性提交 256 个请求，输入长度 100–1024，输出长度 100–1024，`temperature=0.6`，`ignore_eos=True`。每轮唯一输入 142,827 token、实际输出 133,966 token；6 轮输出长度校验均通过。
- `max_model_len=4096`、`max_num_batched_tokens=16384`、`max_num_seqs=512`、`gpu_memory_utilization=0.9`、KV 块大小 256。模型加载和预热不计入计时。
- 表中数值是每种模式 3 轮的**逐指标中位数**；每轮的 P50/P95/P99 在 256 个请求上计算，ITL 在该轮全部相邻 token 间隔上计算。

## 吞吐量、容量与显存

| 指标 | 默认精度 auto | INT8 | INT8 相对 auto |
| --- | ---: | ---: | ---: |
| KV 块数 | 701 | 1,272 | +81.5% |
| 完成 256 请求的墙钟时间 | 26.380 s | 48.216 s | +82.8% |
| 请求吞吐 | 9.70 请求/s | 5.31 请求/s | −45.3% |
| 唯一输入 token / 墙钟时间 | 5,414.21 token/s | 2,962.25 token/s | −45.3% |
| 实际输出 token / 墙钟时间 | 5,078.31 token/s | 2,778.47 token/s | −45.3% |
| 输入加输出 token / 墙钟时间 | 10,492.53 token/s | 5,740.72 token/s | −45.3% |
| Prefill 模型执行吞吐 | 46,795.05 处理 token/s | 63,315.21 处理 token/s | +35.3% |
| Prefill 实际处理 token 数 | 181,836 | 142,827 | −21.5% |
| Decode 模型执行吞吐 | 6,011.15 处理 token/s | 2,924.78 处理 token/s | −51.3% |
| PyTorch 峰值 allocated | 20.75 GiB | 20.76 GiB | 基本相同 |
| PyTorch 峰值 reserved | 21.21 GiB | 21.21 GiB | 相同 |

三轮实际输出吞吐分别为：

| 模式 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| auto | 5,125.46 | 5,075.85 | 5,078.31 token/s |
| int8 | 2,784.60 | 2,778.47 | 2,775.87 token/s |

## 延迟

单位均为毫秒；TPOT、ITL 是每个后续 token 的时间。

| 指标 | auto 均值 | auto P50 | auto P95 | auto P99 | INT8 均值 | INT8 P50 | INT8 P95 | INT8 P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT | 1,305.98 | 1,344.04 | 2,280.29 | 2,280.29 | 1,296.08 | 1,332.30 | 2,265.88 | 2,265.88 |
| TPOT | 32.43 | 32.00 | 42.23 | 57.83 | 59.12 | 60.48 | 68.79 | 73.24 |
| ITL | 29.85 | 26.84 | 53.85 | 55.61 | 55.78 | 57.90 | 62.51 | 62.68 |
| 请求总延迟 | 16,895.10 | 17,739.45 | 25,165.80 | 25,480.09 | 30,429.28 | 31,528.68 | 47,885.12 | 48,129.96 |

## 解读

- INT8 将可分配 KV 块从 701 增至 1,272，当前负载下没有发生默认精度模式的 39,009 token 额外 prefill 重算。这使 INT8 的 prefill 阶段更短，但整个任务仍由 decode 主导。
- 当前 INT8 实现每轮 decode 都把本轮所需的量化 KV 块反量化到一层共享的 BF16 临时缓冲，再调用 FlashAttention。Decode 模型执行吞吐约减半，抵消了容量收益；输出吞吐下降约 45.3%。
- TTFT P50 基本相同（1,344.04 vs 1,332.30 ms）；TPOT P50 从 32.00 增至 60.48 ms/token，请求总延迟 P50 从 17.74 增至 31.53 s。
- `Total tokens/s` 同时计入 142,827 个输入 token 和 133,966 个输出 token；它约为 `Output tokens/s` 的 2.07 倍，不表示生成吞吐翻倍。三轮 INT8 输出吞吐为 2,784.60、2,778.47、2,775.87 token/s，没有第三轮翻倍。
- 两种模式都按照同一显存使用比例预分配 KV 容量；因此 PyTorch 峰值显存接近，INT8 的收益体现在块数上。

## 测量边界与原始记录

TTFT 从批量提交开始，到该请求首个 token 完成调度回填；包含排队时间。TPOT 是每个请求的 `(最后 token 时间 − 首 token 时间)/(输出 token 数 − 1)`；ITL 汇总所有相邻 token 间隔。Prefill/Decode 模型执行吞吐以 `ModelRunner.run` 调用时间为分母，与完整 `generate()` 墙钟吞吐的口径不同。显存数字来自 PyTorch 分配器，不是整卡 NVML 峰值。

这是随机 token、离线批处理测试；没有评估量化后的回答质量。昨天原版默认模式的输出吞吐中位数为 5,114.18 token/s，今天同条件复测为 5,078.31 token/s，差约 0.7%。6 份完整终端输出见 [原始日志目录](kv_cache_int8_2026-10-02/)。
