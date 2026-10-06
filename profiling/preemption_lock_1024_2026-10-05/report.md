# 1024 请求：抢占后优先 decode 的 lock 实验

lock 是调度器的全局 0/1 状态：每次抢占设为 1；任一请求完成时设为 0。
开启 preemption_lock 后，lock=1 且 running 非空时跳过 prefill，优先 decode；
running 为空时允许 prefill 恢复 KV。关闭开关保持原来的 prefill 优先策略。

## BF16（各策略三轮中位数）

| 指标 | 原策略 | lock 策略 | 变化 |
| --- | ---: | ---: | ---: |
| 抢占事件次数 | 769 | 769 | +0.00% |
| 至少被抢占一次的请求数 | 427 | 427 | +0.00% |
| 被抢占多次的请求数 | 178 | 178 | +0.00% |
| 单请求最大抢占次数 | 10 | 10 | +0.00% |
| 整批 generate 秒 | 117.604 | 117.517 | -0.07% |
| 输出 token/s | 4964.14 | 4967.83 | +0.07% |
| 请求/s | 8.71 | 8.71 | +0.00% |
| 输入 token/s | 4937.45 | 4941.11 | +0.07% |
| 总 token/s | 9901.58 | 9908.94 | +0.07% |
| TTFT MEAN ms | 46120.15 | 46103.41 | -0.04% |
| TTFT P50 ms | 47895.92 | 47883.77 | -0.03% |
| TTFT P95 ms | 100747.18 | 100711.91 | -0.04% |
| TTFT P99 ms | 104287.20 | 104254.37 | -0.03% |
| TPOT MEAN ms | 34.89 | 34.87 | -0.06% |
| TPOT P50 ms | 34.64 | 34.64 | +0.00% |
| TPOT P95 ms | 39.79 | 39.75 | -0.10% |
| TPOT P99 ms | 70.75 | 70.73 | -0.03% |
| ITL MEAN ms | 34.18 | 34.16 | -0.06% |
| ITL P50 ms | 27.23 | 27.23 | +0.00% |
| ITL P95 ms | 57.02 | 56.90 | -0.21% |
| ITL P99 ms | 69.12 | 68.98 | -0.20% |
| End-to-end latency MEAN ms | 65562.56 | 65541.93 | -0.03% |
| End-to-end latency P50 ms | 65880.40 | 65863.49 | -0.03% |
| End-to-end latency P95 ms | 113815.27 | 113736.63 | -0.07% |
| End-to-end latency P99 ms | 116892.27 | 116806.57 | -0.07% |
| blocks | 701 | 701 | +0.00% |
| peak_allocated_gib | 20.75 | 20.75 | +0.00% |
| peak_reserved_gib | 21.05 | 21.05 | +0.00% |
| prefill model-run seconds | 23.072 | 23.079 | +0.03% |
| prefill model-run tokens | 871974 | 871974 | +0.00% |
| prefill model-run tokens_per_s | 37793.56 | 37781.94 | -0.03% |
| decode model-run seconds | 93.300 | 93.321 | +0.02% |
| decode model-run tokens | 582009 | 582009 | +0.00% |
| decode model-run tokens_per_s | 6238.03 | 6236.63 | -0.02% |
| 实测 decode batch mean | 147.72 | 147.72 | +0.00% |
| 实测 decode batch p50 | 167.00 | 167.00 | +0.00% |
| 实测 decode batch p95 | 203.00 | 203.00 | +0.00% |
| 实测 decode batch maximum | 265.00 | 265.00 | +0.00% |
| 实测 decode batch steps | 3940.00 | 3940.00 | +0.00% |

## 快速双 scale INT8（各策略三轮中位数）

| 指标 | 原策略 | lock 策略 | 变化 |
| --- | ---: | ---: | ---: |
| 抢占事件次数 | 653 | 653 | +0.00% |
| 至少被抢占一次的请求数 | 305 | 305 | +0.00% |
| 被抢占多次的请求数 | 132 | 132 | +0.00% |
| 单请求最大抢占次数 | 17 | 17 | +0.00% |
| 整批 generate 秒 | 94.285 | 94.532 | +0.26% |
| 输出 token/s | 6191.87 | 6175.74 | -0.26% |
| 请求/s | 10.86 | 10.83 | -0.28% |
| 输入 token/s | 6158.57 | 6142.53 | -0.26% |
| 总 token/s | 12350.44 | 12318.27 | -0.26% |
| TTFT MEAN ms | 28030.66 | 28030.04 | -0.00% |
| TTFT P50 ms | 34035.30 | 34063.08 | +0.08% |
| TTFT P95 ms | 73491.01 | 73662.00 | +0.23% |
| TTFT P99 ms | 77723.05 | 77961.23 | +0.31% |
| TPOT MEAN ms | 53.45 | 53.57 | +0.22% |
| TPOT P50 ms | 52.24 | 52.41 | +0.33% |
| TPOT P95 ms | 77.12 | 77.08 | -0.05% |
| TPOT P99 ms | 153.74 | 153.74 | +0.00% |
| ITL MEAN ms | 50.85 | 50.98 | +0.26% |
| ITL P50 ms | 39.40 | 39.49 | +0.23% |
| ITL P95 ms | 73.50 | 73.72 | +0.30% |
| ITL P99 ms | 98.45 | 97.32 | -1.15% |
| End-to-end latency MEAN ms | 56971.06 | 57035.64 | +0.11% |
| End-to-end latency P50 ms | 56875.04 | 56857.79 | -0.03% |
| End-to-end latency P95 ms | 91219.40 | 91465.56 | +0.27% |
| End-to-end latency P99 ms | 93437.65 | 93685.43 | +0.27% |
| blocks | 1320 | 1320 | +0.00% |
| peak_allocated_gib | 20.76 | 20.76 | +0.00% |
| peak_reserved_gib | 21.06 | 21.06 | +0.00% |
| prefill model-run seconds | 21.235 | 21.317 | +0.39% |
| prefill model-run tokens | 877980 | 877980 | +0.00% |
| prefill model-run tokens_per_s | 41345.78 | 41187.12 | -0.38% |
| decode model-run seconds | 71.922 | 72.015 | +0.13% |
| decode model-run tokens | 582125 | 582125 | +0.00% |
| decode model-run tokens_per_s | 8093.81 | 8083.36 | -0.13% |
| 实测 decode batch mean | 244.28 | 244.28 | +0.00% |
| 实测 decode batch p50 | 302.00 | 302.00 | +0.00% |
| 实测 decode batch p95 | 417.00 | 417.00 | +0.00% |
| 实测 decode batch maximum | 490.00 | 490.00 | +0.00% |
| 实测 decode batch steps | 2383.00 | 2383.00 | +0.00% |

## 每轮结果

| 版本 | 策略 | 轮次 | 抢占次数 | 被抢占请求数 | 输出 token/s | TTFT P50 ms | TPOT P50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | original | 1 | 769 | 427 | 4951.87 | 48144.19 | 34.73 |
| bf16 | lock | 1 | 769 | 427 | 4970.97 | 47863.60 | 34.62 |
| fast_int8_half | original | 1 | 653 | 305 | 6211.99 | 33963.13 | 52.10 |
| fast_int8_half | lock | 1 | 653 | 305 | 6196.65 | 34063.08 | 52.24 |
| fast_int8_half | lock | 2 | 653 | 305 | 6175.74 | 33995.30 | 52.41 |
| fast_int8_half | original | 2 | 653 | 305 | 6191.87 | 34035.30 | 52.24 |
| bf16 | lock | 2 | 769 | 427 | 4967.83 | 47883.77 | 34.64 |
| bf16 | original | 2 | 769 | 427 | 4967.95 | 47895.92 | 34.64 |
| bf16 | original | 3 | 769 | 427 | 4964.14 | 47877.10 | 34.63 |
| bf16 | lock | 3 | 769 | 427 | 4960.89 | 47993.72 | 34.67 |
| fast_int8_half | original | 3 | 653 | 305 | 6184.19 | 34105.35 | 52.31 |
| fast_int8_half | lock | 3 | 653 | 305 | 6172.04 | 34153.76 | 52.43 |

## 配置与测量范围

- RTX 3090 Ti / Qwen3-0.6B / GPU 0；权重均为 BF16，快速 INT8 仅量化 KV。
- 快速 kernel 与先前 1024 请求实验相同：prefill BM=64、decode BM=16，BN=64、num_stages=1、num_warps=4；所有 INT8 prefill/decode 均走自写 kernel。
- 同一版本原策略/lock 使用完全相同的推理源码，仅 preemption_lock 参数不同；各三个独立新进程，交替运行，全为本次测量。
- 完整负载复用上次 workload_1024.json：每轮输入 580663、输出 583802 token；1024 请求同时提交，长度各 100–1024，seed=0、temperature=0.6、ignore_eos=True。
- max_num_seqs=512、max_num_batched_tokens=16384、max_model_len=4096、page size=256、gpu_memory_utilization=0.9；decode CUDA Graph，prefill eager。
- 普通和缓存前缀 prefill 预热与上次相同，预热后重置采样种子和显存峰值；计数只包含正式 generate。未锁 GPU 频率。
- TTFT 包含排队，为批量提交到 CPU postprocess 首 token 回填；TPOT 为每请求首末 token 时间差除以后续 token 数；ITL 汇总所有相邻 token 间隔。
- model-run 时间包含输入准备、模型、采样和同步；整模型比较包含 KV 容量和调度影响。显存为 PyTorch allocated/reserved 峰值。
- 抢占事件可重复发生于同一请求；request_preemption_stats.histogram 记录每请求抢占次数分布（含 0 次）。
- decode batch 直接记录实际 schedule 返回的 decode 请求数，每步等权。lock_stats.decode_batches 为模型执行时 lock=1 的 decode 批次数（包括触发锁的批次）。
- 每个请求的输出长度和时间戳数量均检查；原实验快照未修改；此次仅改变共同的 config/scheduler 和 benchmark 计数，attention/model kernel 未修改。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_preemption_lock_metrics.py --outdir /tmp/preemption-lock-recheck --runs 3
```

## 为什么 lock 没有减少抢占

12 次 GPU 实测中，同一版本开关前后的输出 token 摘要、抢占分布和 decode batch 统计完全一致。
CPU 调度复现使用同一输入负载与 KV 容量，输出 token 为每请求固定的合成 token；不用于测量模型性能。
复现的抢占事件数和每请求分布与所有 GPU 轮次一致，两种策略的调度轨迹摘要也相同。

| 版本 | lock=1 且存在运行/等待请求的步数 | 其中队首可接纳 prefill 的步数 | 两策略轨迹 |
| --- | ---: | ---: | --- |
| BF16 | 1319 | 0 | 相同 |
| 快速 INT8 | 595 | 0 | 相同 |

原策略在这些 lock=1 时段也会因为 can_allocate=-1 跳过 prefill。完成请求释放更多块时，lock 恰好清零，两个策略同时恢复 prefill；因此该条件没有改变实际调度顺序。

当前 chunked prefill 拆分计算量，KV 分配仍覆盖整个当前上下文：Scheduler.schedule 先 can_allocate，再 allocate，最后才设置 num_scheduled_tokens=min(num_tokens,remaining)。
BlockManager.can_allocate 从 seq.num_blocks 起算所需块数（只扣除已被其他请求引用的可复用前缀块）；allocate 同样分配到 seq.num_blocks。空闲块不足时会提前 break，尚未进入切 chunk 的阶段。
本负载单请求总上下文不超过 2048，单步 token 预算为 16384；当前代码只允许本轮第一个请求拆分 chunk，所以本次也没有触发单请求的 chunked prefill。

重复抢占循环确实存在。CPU 复现里 BF16 请求索引 223（第 224 个）在第 196 步 prefill、第 197 步被抢占、第 198 步再次 prefill；再次 prefill 时 lock 已为 0。这一请求共被抢占 10 次，两个策略相同。
完整事件记录见 scheduler_replay.json；7 项调度单元测试通过，测试含锁状态、EOS 解锁、空 running 回退和混合请求完成/KV 回收。
