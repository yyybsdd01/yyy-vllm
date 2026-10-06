# 更快双 scale INT8 与 BF16：256 / 512 请求重新对比

使用此前已保存的 fast_int8 推理源码：prefill BM=64、BN=64；decode BM=16、BN=64；
两阶段均 num_stages=1、num_warps=4。INT8 模式的所有正式 prefill 和 decode 均走自写 kernel。
BF16 模式保持原 FlashAttention prefill/decode。两组模型权重均为 BF16，INT8 仅量化 KV cache。
每轮通过源码哈希、实际包路径、KV dtype、prefill 派发计数及各请求输出长度确认测试版本。

## 512 请求（三轮中位数）

| 指标 | BF16 | 更快双 scale INT8 | 变化 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 15115.32 | 3333.87 | -77.94% |
| TTFT P50 ms | 2184.21 | 2317.95 | +6.12% |
| TTFT P95 ms | 39610.31 | 4507.90 | -88.62% |
| TTFT P99 ms | 43572.53 | 28335.52 | -34.97% |
| TPOT MEAN ms | 35.01 | 53.44 | +52.64% |
| TPOT P50 ms | 34.81 | 50.55 | +45.22% |
| TPOT P95 ms | 45.86 | 71.79 | +56.54% |
| TPOT P99 ms | 73.49 | 167.83 | +128.37% |
| ITL MEAN ms | 33.43 | 48.71 | +45.71% |
| ITL P50 ms | 27.16 | 39.29 | +44.66% |
| ITL P95 ms | 55.25 | 68.40 | +23.80% |
| ITL P99 ms | 68.29 | 79.79 | +16.84% |
| End-to-end latency MEAN ms | 33651.09 | 30338.66 | -9.84% |
| End-to-end latency P50 ms | 33904.35 | 32832.03 | -3.16% |
| End-to-end latency P95 ms | 55426.17 | 43420.53 | -21.66% |
| End-to-end latency P99 ms | 56363.11 | 44779.33 | -20.55% |
| generate s | 57.236 | 45.338 | -20.79% |
| 输出 token/s | 4969.11 | 6273.14 | +26.24% |
| 请求/s | 8.95 | 11.29 | +26.15% |
| 输入 token/s | 5100.60 | 6439.13 | +26.24% |
| 总 token/s | 10069.72 | 12712.27 | +26.24% |
| KV blocks | 701 | 1320 | +88.30% |
| 抢占次数 | 322 | 220 | -31.68% |
| 峰值 allocated GiB | 20.75 | 20.76 | +0.05% |
| 峰值 reserved GiB | 21.05 | 21.06 | +0.05% |
| prefill model-run seconds | 10.763 | 8.880 | -17.50% |
| prefill model-run tokens | 426682 | 408495 | -4.26% |
| prefill model-run tokens_per_s | 39642.45 | 46000.37 | +16.04% |
| decode model-run seconds | 45.825 | 35.931 | -21.59% |
| decode model-run tokens | 283579 | 283681 | +0.04% |
| decode model-run tokens_per_s | 6188.27 | 7895.24 | +27.58% |
| 平均首 token 后未完成请求数 | 165.8 | 305.0 | +83.96% |

## 256 请求（三轮中位数）

| 指标 | BF16 | 更快双 scale INT8 | 变化 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 1185.01 | 1259.25 | +6.26% |
| TTFT P50 ms | 1223.98 | 1300.31 | +6.24% |
| TTFT P95 ms | 2166.99 | 2303.06 | +6.28% |
| TTFT P99 ms | 2166.99 | 2303.06 | +6.28% |
| TPOT MEAN ms | 32.50 | 24.94 | -23.26% |
| TPOT P50 ms | 32.05 | 24.39 | -23.90% |
| TPOT P95 ms | 42.38 | 33.10 | -21.90% |
| TPOT P99 ms | 57.87 | 38.07 | -34.21% |
| ITL MEAN ms | 29.89 | 23.00 | -23.05% |
| ITL P50 ms | 26.87 | 22.79 | -15.18% |
| ITL P95 ms | 54.12 | 24.26 | -55.17% |
| ITL P99 ms | 55.15 | 24.39 | -55.78% |
| End-to-end latency MEAN ms | 16799.20 | 13274.63 | -20.98% |
| End-to-end latency P50 ms | 17641.91 | 13735.09 | -22.15% |
| End-to-end latency P95 ms | 25078.95 | 20166.61 | -19.59% |
| End-to-end latency P99 ms | 25392.99 | 20350.85 | -19.86% |
| generate s | 26.290 | 20.434 | -22.27% |
| 输出 token/s | 5095.63 | 6555.92 | +28.66% |
| 请求/s | 9.74 | 12.53 | +28.64% |
| 输入 token/s | 5432.68 | 6989.55 | +28.66% |
| 总 token/s | 10528.31 | 13545.47 | +28.66% |
| KV blocks | 701 | 1320 | +88.30% |
| 抢占次数 | 84 | 0 | -100.00% |
| 峰值 allocated GiB | 20.75 | 20.76 | +0.05% |
| 峰值 reserved GiB | 21.05 | 21.06 | +0.05% |
| prefill model-run seconds | 3.773 | 2.293 | -39.23% |
| prefill model-run tokens | 181836 | 142827 | -21.45% |
| prefill model-run tokens_per_s | 48199.44 | 62275.46 | +29.20% |
| decode model-run seconds | 22.260 | 17.908 | -19.55% |
| decode model-run tokens | 133626 | 133710 | +0.06% |
| decode model-run tokens_per_s | 6002.92 | 7466.48 | +24.38% |
| 平均首 token 后未完成请求数 | 152.0 | 150.5 | -1.00% |

## 每轮记录

| 请求数 | 模式 | 轮次 | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | bf16 | 1 | 4980.38 | 2185.16 | 34.72 | 322 |
| 512 | fast_int8_half | 1 | 6282.53 | 2320.81 | 50.46 | 220 |
| 512 | fast_int8_half | 2 | 6270.66 | 2317.95 | 50.58 | 220 |
| 512 | bf16 | 2 | 4965.77 | 2179.94 | 34.85 | 322 |
| 512 | bf16 | 3 | 4969.11 | 2184.21 | 34.81 | 322 |
| 512 | fast_int8_half | 3 | 6273.14 | 2314.28 | 50.55 | 220 |
| 256 | bf16 | 1 | 5104.52 | 1220.44 | 31.97 | 84 |
| 256 | fast_int8_half | 1 | 6555.92 | 1300.31 | 24.39 | 0 |
| 256 | fast_int8_half | 2 | 6567.18 | 1295.73 | 24.33 | 0 |
| 256 | bf16 | 2 | 5095.63 | 1225.20 | 32.05 | 84 |
| 256 | bf16 | 3 | 5089.16 | 1223.98 | 32.09 | 84 |
| 256 | fast_int8_half | 3 | 6553.35 | 1301.91 | 24.42 | 0 |

## 配置和测量范围

- Qwen3-0.6B / NVIDIA GeForce RTX 3090 Ti / GPU 0，decode CUDA Graph、prefill eager；未锁 GPU 频率。
- 每负载、每模式各三个独立新进程；同一负载内 BF16 与更快 INT8 交替运行，全为本次新测量。
- 请求同时到达；输入/输出长度各 100–1024；负载及 PyTorch 采样 seed=0，temperature=0.6，ignore_eos=True。
- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9。
- 正式计时前预热普通 prefill 长度 256/512/768/1024（最大 16×1024），及页表宽度 2–8 的缓存前缀 prefill（新 Q 长度 128）；之后重置采样种子和显存峰值。
- TTFT 为批量提交到 CPU postprocess 首 token 回填完成，包含排队；TPOT 为每请求首末 token 时间差除以后续 token 数，ITL 汇总相邻 token 间隔。
- 阶段时间含输入准备、模型、采样和同步；整模型结果包含 KV 容量与调度效应。双 scale INT8 每 token KV+scale 字节数为 BF16 的 53.125%。
- 平均首 token 后未完成请求数由 ITL_mean × (输出 token 总数−请求数) / generate 时间推导，包含被抢占请求，不能当作实测 decode batch size。
- 同负载的输入、目标输出数量严格相同；随机 token ID 离线负载，这次未重测语言精度。之前同一 fast_int8 源码的 PPL 记录保存在 reference 目录。
- 原工作树和 reference 实验源码均未修改。所有实际运行源码、配置、完整负载、派发计数和日志保存在本目录。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_fast_int8_metrics.py --outdir /tmp/fast-int8-recheck --requests 512 256 --runs 3
```
