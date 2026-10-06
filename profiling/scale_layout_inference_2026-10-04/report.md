# 双 scale 存储布局的完整模型指标：token/head 与 head/token

新 `[block, head, token, 2]` 布局相对当前 `[block, token, head, 2]`：输出吞吐 **-0.47%**，TTFT P50 **+0.05%**，TPOT P50 **-0.40%**。本次没有观察到完整推理的吞吐收益。

## 实验实现与条件

- 原仓库所有 `nanovllm/*.py` 和原 `benchmark_inference_metrics.py` 未修改，逐次启动/结束均校验 SHA256。
- 两份独立源码位于 `variants/token_head/` 和 `variants/head_token/`。后者仅改变 scale 分配形状，并在 `Attention.forward` 调用新增的写入与 attention 接口。原量化、原模型结构和调度配置保留；没有把新布局接入当前生产工作树。
- 两者都是 INT8 K/V、每 token/head 的 K/V 各两个 FP32 scale；模型权重为 BF16。
- Qwen3-0.6B，RTX 3090 Ti，GPU 0，单卡；CUDA Graph 开启；max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、block_size=256、gpu_memory_utilization=0.9。
- 256 个请求同时到达，随机 token-ID 输入/输出长度都在 100–1024。每轮输入 142,827 token、输出 133,966 token；temperature=0.6、ignore_eos=True，逐请求输出长度校验通过。
- 每种布局各 3 个独立进程，按 token/head→head/token / head/token→token/head / token/head→head/token 交替。模型初始化与生成预热不计入正式 generate 时间；GPU 没有锁频。
- Python 负载 seed=0。两份基准副本均额外固定 torch seed=0，生成预热之后再次设 seed=0；这是相对原基准的共同测量改动，原基准文件未修改。六轮输出 token 的 SHA256 完全相同，因此比较相同生成内容和相同工作量。
- 先按每轮请求计算分布，再对每项统计量取三轮中位数；P95/P99 是请求/token 分布的分位数，不是三轮结果的分位数。
- KV 容量均为 1320 块，prefill 执行 token 均为 142827，decode 均为 133710。与 BF16/INT8 容量对照不同，本次没有容量和重算工作量差异。
- 正式测量前，两种布局各用 4 请求的完整模型 smoke 检查通过，生成 token 一致。完整源码、运行配置和日志均留在此目录。

## 延迟分布

单位 ms。TTFT 与请求总延迟为 ms/request；TPOT 为每请求平均 ms/token；ITL 为相邻 token 的 ms/token。

| 指标 | 统计量 | 当前 token/head | 新 head/token | 相对变化 |
| --- | --- | ---: | ---: | ---: |
| TTFT | MEAN | 1300.37 | 1300.92 | +0.04% |
| TTFT | P50 | 1336.99 | 1337.66 | +0.05% |
| TTFT | P95 | 2275.10 | 2274.36 | -0.03% |
| TTFT | P99 | 2275.10 | 2274.36 | -0.03% |
| TPOT | MEAN | 29.95 | 29.90 | -0.17% |
| TPOT | P50 | 29.75 | 29.63 | -0.40% |
| TPOT | P95 | 37.65 | 37.31 | -0.90% |
| TPOT | P99 | 42.32 | 41.96 | -0.85% |
| ITL | MEAN | 27.96 | 27.99 | +0.11% |
| ITL | P50 | 27.93 | 27.95 | +0.07% |
| ITL | P95 | 29.80 | 29.64 | -0.54% |
| ITL | P99 | 30.06 | 29.78 | -0.93% |
| End-to-end latency | MEAN | 15901.30 | 15916.43 | +0.10% |
| End-to-end latency | P50 | 16427.35 | 16425.86 | -0.01% |
| End-to-end latency | P95 | 24580.91 | 24701.31 | +0.49% |
| End-to-end latency | P99 | 24759.00 | 24877.73 | +0.48% |

## 吞吐、阶段执行与显存

| 指标 | 当前 token/head | 新 head/token | 相对变化 |
| --- | ---: | ---: | ---: |
| 总 generate 时间 s | 24.839 | 24.957 | +0.48% |
| 请求吞吐 request/s | 10.31 | 10.26 | -0.48% |
| 输入 token/s | 5750.07 | 5722.85 | -0.47% |
| 输出 token/s | 5393.33 | 5367.80 | -0.47% |
| 输入+输出 token/s | 11143.40 | 11090.65 | -0.47% |
| KV blocks | 1320 | 1320 | +0.00% |
| 峰值 PyTorch allocated GiB | 20.76 | 20.76 | +0.00% |
| 峰值 PyTorch reserved GiB | 21.21 | 21.21 | +0.00% |
| prefill model-run tokens_per_s | 63059.94 | 63072.58 | +0.02% |
| prefill model-run tokens | 142827 | 142827 | +0.00% |
| prefill model-run seconds | 2.265 | 2.264 | -0.04% |
| decode model-run tokens_per_s | 5985.58 | 5952.88 | -0.55% |
| decode model-run tokens | 133710 | 133710 | +0.00% |
| decode model-run seconds | 22.339 | 22.461 | +0.55% |

## 每轮结果与波动

| 布局 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 日志 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| token_head | 1 | 24.827 | 5395.91 | 1328.92 | 29.75 | [token_head_1.txt](token_head_1.txt) |
| head_token | 1 | 24.957 | 5367.80 | 1334.73 | 29.63 | [head_token_1.txt](head_token_1.txt) |
| head_token | 2 | 24.945 | 5370.53 | 1337.66 | 29.60 | [head_token_2.txt](head_token_2.txt) |
| token_head | 2 | 24.839 | 5393.33 | 1344.04 | 29.72 | [token_head_2.txt](token_head_2.txt) |
| token_head | 3 | 24.880 | 5384.48 | 1336.99 | 29.82 | [token_head_3.txt](token_head_3.txt) |
| head_token | 3 | 24.978 | 5363.41 | 1343.78 | 29.65 | [head_token_3.txt](head_token_3.txt) |

token_head 三轮输出吞吐范围：5384.48–5395.91 token/s。

head_token 三轮输出吞吐范围：5363.41–5370.53 token/s。

## 指标边界与解释

1. **TTFT 主要体现 prefill 和排队。** 普通首次 prefill 两种布局都调用 FlashAttention，布局只影响它的缓存写入；新增 attention kernel 主要在 decode 和带缓存前缀的 prefill 使用。TTFT 不会直接对应上一轮单个 decode kernel 的速度变化。
2. **吞吐接近，未观察到布局带来的整体收益。** 两种布局的 KV 容量、生成 token 和阶段 token 数一致，隔离了前次 BF16/INT8 对比中的容量差异。约 1% 或更小的变化需结合三轮波动解释，当前结果只适用于此模型、GPU 和离线负载。
3. **ITL/TPOT/总时间的分位数可以朝不同方向变化。** TPOT 先逐请求平均，ITL 混合所有 token 间隔，请求总时间还含首 token 等待；少量不同 batch/上下文阶段的变化会被不同权重放大。
4. **显存基本相同。** 两种布局存储相同数量的 INT8 K/V 和 FP32 scale，布局变化不改变容量。记录的是 PyTorch 分配器峰值，没有采集 NVML 进程峰值。
5. **延续项目基准的测量边界。** TTFT 从整批请求提交到 CPU `Scheduler.postprocess` 回填首 token 完成，包含排队/准备；TPOT=(最后−首 token)/(输出长度−1)，ITL 为相邻时间戳差。吞吐以完整 generate 墙钟时间计算。阶段吞吐以 `ModelRunner.run` 累计调用时间计算，含准备、模型、采样和同步。该负载为离线随机 token-ID 批量测量，不是在线服务延迟。
6. **模型行为校验。** 六轮生成 token 摘要一致，说明本次负载中新布局保持当前 INT8 模型行为；它不评价 INT8 相对 BF16 的语言质量。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_scale_layout_metrics.py --outdir /tmp/scale_layout_inference_recheck
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/summarize_scale_layout_metrics.py --outdir /tmp/scale_layout_inference_recheck
```

产物：`manifest.json`、`workload.json`、`results.json`、`comparison.csv`、两份独立 `variants/` 源码、2 个 smoke 日志和 6 个正式日志。原仓库源码和两份副本的哈希都在 manifest 中。
