# 更快双 scale INT8 与 BF16：1024 请求对比

使用此前已保存的 fast_int8 推理源码：prefill BM=64、BN=64；decode BM=16、BN=64；
两阶段均 num_stages=1、num_warps=4。INT8 模式的所有正式 prefill 和 decode 均走自写 kernel。
BF16 模式保持原 FlashAttention prefill/decode。两组模型权重均为 BF16，INT8 仅量化 KV cache。
每轮通过源码哈希、实际包路径、KV dtype、prefill 派发计数及各请求输出长度确认测试版本。

## 1024 请求（三轮中位数）

每轮输入 580663 token、输出 583802 token；两组负载相同。

| 指标 | BF16 | 更快双 scale INT8 | 变化 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 46247.24 | 28032.82 | -39.38% |
| TTFT P50 ms | 48018.70 | 34062.41 | -29.06% |
| TTFT P95 ms | 101035.01 | 73424.46 | -27.33% |
| TTFT P99 ms | 104587.90 | 77647.05 | -25.76% |
| TPOT MEAN ms | 34.98 | 53.41 | +52.69% |
| TPOT P50 ms | 34.75 | 52.24 | +50.33% |
| TPOT P95 ms | 39.89 | 77.06 | +93.18% |
| TPOT P99 ms | 70.94 | 153.63 | +116.56% |
| ITL MEAN ms | 34.26 | 50.80 | +48.28% |
| ITL P50 ms | 27.26 | 39.35 | +44.35% |
| ITL P95 ms | 56.77 | 73.63 | +29.70% |
| ITL P99 ms | 69.19 | 97.90 | +41.49% |
| End-to-end latency MEAN ms | 65745.96 | 56938.48 | -13.40% |
| End-to-end latency P50 ms | 66064.78 | 56865.75 | -13.92% |
| End-to-end latency P95 ms | 114067.83 | 91111.09 | -20.13% |
| End-to-end latency P99 ms | 117128.31 | 93324.25 | -20.32% |
| generate s | 117.840 | 94.174 | -20.08% |
| 输出 token/s | 4954.20 | 6199.17 | +25.13% |
| 请求/s | 8.69 | 10.87 | +25.09% |
| 输入 token/s | 4927.57 | 6165.84 | +25.13% |
| 总 token/s | 9881.77 | 12365.01 | +25.13% |
| KV blocks | 701 | 1320 | +88.30% |
| 抢占次数 | 769 | 653 | -15.08% |
| 峰值 allocated GiB | 20.75 | 20.76 | +0.05% |
| 峰值 reserved GiB | 21.05 | 21.06 | +0.05% |
| prefill model-run seconds | 23.244 | 21.262 | -8.53% |
| prefill model-run tokens | 871974 | 877980 | +0.69% |
| prefill model-run tokens_per_s | 37514.73 | 41292.77 | +10.07% |
| decode model-run seconds | 93.379 | 71.807 | -23.10% |
| decode model-run tokens | 582009 | 582125 | +0.02% |
| decode model-run tokens_per_s | 6232.73 | 8106.80 | +30.07% |
| 平均首 token 后未完成请求数 | 169.4 | 314.4 | +85.57% |

## 每轮记录

| 请求数 | 模式 | 轮次 | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | bf16 | 1 | 4960.78 | 47893.06 | 34.69 | 769 |
| 1024 | fast_int8_half | 1 | 6200.44 | 34062.41 | 52.24 | 653 |
| 1024 | fast_int8_half | 2 | 6199.17 | 34039.98 | 52.22 | 653 |
| 1024 | bf16 | 2 | 4952.72 | 48018.70 | 34.75 | 769 |
| 1024 | bf16 | 3 | 4954.20 | 48072.03 | 34.75 | 769 |
| 1024 | fast_int8_half | 3 | 6184.60 | 34162.61 | 52.38 | 653 |

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
- 原仓库推理源码和 reference 实验推理源码均未修改。所有实际运行源码、配置、完整负载、派发计数和日志保存在本目录。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_fast_int8_metrics.py --outdir /tmp/fast-int8-recheck --requests 1024 --runs 3
```
