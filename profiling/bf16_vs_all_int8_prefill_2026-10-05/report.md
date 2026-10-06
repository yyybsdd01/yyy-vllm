# 未量化 BF16 与 INT8 双 scale：完整推理指标

BF16 为未量化 KV cache，prefill/decode 都使用项目原 FlashAttention 路径，模型权重为 BF16。三种 INT8 配置的模型权重也为 BF16，仅量化 KV cache；普通 prefill 分别使用原始 BF16 K/V 的 FlashAttention，或自写 INT8 paged kernel。

## 各项指标

| 指标 | 未量化 BF16 | INT8 + FA prefill | 全自写 INT8 原 scale | 全自写 INT8 head/token |
| --- | ---: | ---: | ---: | ---: |
| TTFT MEAN ms | 1181.02 | 1179.74 | 1507.36 | 1499.13 |
| TTFT P50 ms | 1218.74 | 1217.92 | 1557.69 | 1549.60 |
| TTFT P95 ms | 2161.14 | 2156.19 | 2762.17 | 2746.19 |
| TTFT P99 ms | 2161.14 | 2156.19 | 2762.17 | 2746.19 |
| TPOT MEAN ms | 32.47 | 29.99 | 30.97 | 30.82 |
| TPOT P50 ms | 32.02 | 29.79 | 30.28 | 30.08 |
| TPOT P95 ms | 42.35 | 37.70 | 40.46 | 40.00 |
| TPOT P99 ms | 57.83 | 42.39 | 46.45 | 45.95 |
| ITL MEAN ms | 29.87 | 27.99 | 28.69 | 28.65 |
| ITL P50 ms | 26.86 | 27.97 | 28.08 | 28.07 |
| ITL P95 ms | 53.78 | 29.86 | 30.07 | 29.82 |
| ITL P99 ms | 55.59 | 30.10 | 30.33 | 29.94 |
| End-to-end latency MEAN ms | 16781.81 | 15799.78 | 16496.14 | 16463.99 |
| End-to-end latency P50 ms | 17622.26 | 16335.30 | 17032.92 | 16973.18 |
| End-to-end latency P95 ms | 25052.51 | 24478.33 | 25223.78 | 25302.67 |
| End-to-end latency P99 ms | 25366.14 | 24656.34 | 25402.12 | 25480.04 |
| generate 时间 s | 26.263 | 24.736 | 25.482 | 25.560 |
| 请求/s | 9.75 | 10.35 | 10.05 | 10.02 |
| 输入 token/s | 5438.44 | 5774.00 | 5605.03 | 5588.01 |
| 输出 token/s | 5101.04 | 5415.78 | 5257.29 | 5241.33 |
| 输入+输出 token/s | 10539.48 | 11189.77 | 10862.32 | 10829.34 |
| KV blocks | 701 | 1320 | 1320 | 1320 |
| KV 张量及 scales GiB | 19.1680 | 19.1748 | 19.1748 | 19.1748 |
| 峰值 allocated GiB | 20.75 | 20.76 | 20.76 | 20.76 |
| 峰值 reserved GiB | 21.05 | 21.06 | 21.06 | 21.06 |
| prefill model-run seconds | 3.757 | 2.146 | 2.752 | 2.736 |
| prefill model-run tokens_per_s | 48399.81 | 66555.68 | 51897.13 | 52202.86 |
| prefill model-run tokens | 181836 | 142827 | 142827 | 142827 |
| decode model-run seconds | 22.251 | 22.357 | 22.488 | 22.591 |
| decode model-run tokens_per_s | 6005.31 | 5980.75 | 5945.87 | 5918.81 |
| decode model-run tokens | 133626 | 133710 | 133710 | 133710 |

## 相对未量化 BF16

| 配置 | TTFT P50 | TPOT P50 | ITL P50 | 输出吞吐 | generate 时间 | KV 容量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| int8_flash_prefill | -0.07% | -6.96% | +4.13% | +6.17% | -5.81% | +88.30% |
| int8_token_head | +27.81% | -5.43% | +4.54% | +3.06% | -2.97% | +88.30% |
| int8_head_token | +27.15% | -6.06% | +4.50% | +2.75% | -2.68% | +88.30% |

## 测量条件和解释

- Qwen3-0.6B / RTX 3090 Ti / GPU 0，单卡，CUDA Graph decode，prefill eager。
- 256 请求同时到达，输入/输出长度均为 100–1024。负载 seed=0，采样 torch seed=0、temperature=0.6、ignore_eos=True；每轮输入 142,827，输出 133,966 token，逐请求长度验证通过。
- BF16 本次重新跑三个独立进程；INT8 三轮中位数引用紧邻本次的上一轮完整测量，源码/模型文件/负载/预热配置均校验一致。不是四组交错的新一轮测量；未锁 GPU 频率，微小差异不能据此认定稳定收益。
- BF16 使用上一轮 flash/token_head 的完整源码副本，仅增加 BF16 标签、实际 KV dtype/字节数打印、CPU preemption 计数观察；推理实现未修改。所有副本在计时前预热页表宽度 1/2/3/4及16K-token prefill批量，之后重置采样种子。
- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、block_size=256、gpu_memory_utilization=0.9；模型权重始终 BF16。
- BF16 KV 容量 701 块，INT8 1320 块。BF16 正式测量抢占次数为 84 次；prefill 执行 181836 token，较输入额外 39009 token。Scheduler.preempt 释放缓存并重新入队，后续 prefill 包含重算。
- 因而整模型阶段时间包含 KV 容量/抢占/重算效应，不能直接解读为单个 INT8 kernel 更快或更慢。INT8 上一轮的 prefill 执行量等于输入量，decode 执行量为输出量减去每请求首 token。
- 按相同显存预算分配缓存，所以 INT8 总峰值显存不会自动减半。每个 token 的 K/V+双scale占 BF16 的 53.125%，节省的空间用于增大可分配 KV 容量；实际完整缓存池大小仍相近。
- TTFT 是整批请求提交至 CPU postprocess 首 token 完成，包含排队。TPOT 是每请求平均相邻 token 时间，ITL 汇总所有相邻时间戳；阶段时间包括输入准备、模型、采样和同步。P95/P99 是每轮请求/token 分布分位数，再对三轮取中位数。
- 不同量化/调度路径的生成内容可不同，输出 token 数固定。此前自写 INT8 kernel 对恢复后 BF16 KV 的正确性检查已通过；这里没有评估语言质量。
- 原仓库 nanovllm 源码及 benchmark_inference_metrics.py 哈希前后一致。INT8 参考结果、配置和日志在 reference_int8/。

## BF16 每轮结果

| 轮次 | generate s | 输出 token/s | TTFT P50 ms | ITL P50 ms | prefill s | prefill tokens | 抢占次数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.255 | 5102.40 | 1217.16 | 26.85 | 3.761 | 181836 | 84 |
| 2 | 26.263 | 5101.04 | 1218.74 | 26.86 | 3.755 | 181836 | 84 |
| 3 | 26.283 | 5097.00 | 1223.65 | 26.89 | 3.757 | 181836 | 84 |

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_bf16_prefill_comparison.py --outdir /tmp/bf16_prefill_comparison
```

完整模型配置、源码哈希、原始日志、results.json、comparison.csv 和独立 BF16 源码副本均留在该目录。
