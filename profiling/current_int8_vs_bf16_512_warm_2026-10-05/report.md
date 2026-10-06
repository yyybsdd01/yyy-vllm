# 当前仓库 INT8 kernel 与未量化 BF16 的完整推理对比

当前推理源码完整保留，仅在独立副本添加测评预热、固定采样种子和 CPU 计数。
模型权重均为 BF16；INT8 仅量化 KV cache。普通首次 prefill 仍使用 FlashAttention；
INT8 自写 Triton kernel 用于 decode 和带缓存前缀的 prefill（BM=16、BN=64、4 warps，默认流水）。

| 指标 | BF16 | 单 scale INT8 | 双 scale INT8 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 15114.33 | 2585.08 | 3397.90 |
| TTFT P50 ms | 2192.46 | 2174.21 | 2180.76 |
| TTFT P95 ms | 39593.01 | 4341.76 | 4233.41 |
| TTFT P99 ms | 43551.40 | 27843.38 | 33459.03 |
| TPOT MEAN ms | 35.00 | 62.72 | 64.53 |
| TPOT P50 ms | 34.78 | 58.65 | 61.18 |
| TPOT P95 ms | 45.83 | 83.64 | 84.21 |
| TPOT P99 ms | 73.45 | 196.63 | 207.79 |
| ITL MEAN ms | 33.41 | 57.10 | 59.34 |
| ITL P50 ms | 27.26 | 47.29 | 50.26 |
| ITL P95 ms | 54.85 | 77.12 | 79.39 |
| ITL P99 ms | 68.70 | 88.79 | 89.63 |
| End-to-end latency MEAN ms | 33640.97 | 34243.15 | 36299.02 |
| End-to-end latency P50 ms | 33899.28 | 36736.33 | 39253.12 |
| End-to-end latency P95 ms | 55401.83 | 49076.27 | 52851.22 |
| End-to-end latency P99 ms | 56338.12 | 50704.66 | 54440.70 |
| generate s | 57.210 | 51.220 | 54.966 |
| 请求/s | 8.95 | 10.00 | 9.31 |
| 输出 token/s | 4971.36 | 5552.78 | 5174.39 |
| 输入 token/s | 5102.91 | 5699.71 | 5311.31 |
| 总 token/s | 10074.26 | 11252.49 | 10485.70 |
| KV blocks | 701 | 1360 | 1320 |
| 缓存抢占次数 | 322 | 224 | 220 |
| 峰值 allocated GiB | 20.75 | 20.76 | 20.76 |
| 峰值 reserved GiB | 21.05 | 21.06 | 21.06 |
| KV cache + scales GiB | 19.1680 | 19.1748 | 19.1748 |
| prefill model-run seconds | 10.706 | 8.508 | 8.612 |
| prefill model-run tokens | 426682 | 410912 | 408495 |
| prefill model-run tokens_per_s | 39856.28 | 48295.66 | 47434.62 |
| decode model-run seconds | 45.963 | 42.208 | 45.852 |
| decode model-run tokens | 283579 | 283677 | 283681 |
| decode model-run tokens_per_s | 6169.78 | 6720.90 | 6186.90 |

## 相对 BF16 的变化

| 模式 | TTFT P50 | TPOT P50 | ITL P50 | 输出吞吐 | generate 时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单 scale INT8 | -0.83% | +68.63% | +73.48% | +11.70% | -10.47% |
| 双 scale INT8 | -0.53% | +75.91% | +84.37% | +4.08% | -3.92% |

## 每轮结果

| 模式 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto | 1 | 57.338 | 4960.29 | 2196.05 | 34.87 | 322 |
| int8 | 1 | 51.220 | 5552.78 | 2173.35 | 58.69 | 224 |
| int8_half | 1 | 54.966 | 5174.39 | 2182.21 | 61.16 | 220 |
| int8 | 2 | 51.194 | 5555.62 | 2184.58 | 58.58 | 224 |
| int8_half | 2 | 54.998 | 5171.33 | 2180.76 | 61.28 | 220 |
| auto | 2 | 57.209 | 4971.50 | 2191.55 | 34.78 | 322 |
| int8_half | 3 | 54.940 | 5176.80 | 2179.07 | 61.18 | 220 |
| auto | 3 | 57.210 | 4971.36 | 2192.46 | 34.78 | 322 |
| int8 | 3 | 51.239 | 5550.66 | 2174.21 | 58.65 | 224 |

## 测量范围

- Qwen3-0.6B / NVIDIA GeForce RTX 3090 Ti / GPU 0，单卡，decode CUDA Graph、prefill eager。未锁 GPU 频率。
- 每模式 3 个独立新进程，逐轮循环轮换运行顺序，全部为本次新测量。每轮请求/token 分布先计算 Mean/P50/P95/P99，再逐项取各轮中位数。
- 512 请求同时到达，输入长度 100–1024，输出长度 100–1024，负载和 PyTorch 采样 seed=0。temperature=0.6、ignore_eos=True。
- 每轮输入 291,939、输出 284,413 token；每请求输出长度、时间戳数量均验证通过。随机 token ID 离线负载，未评价自然语言质量。
- 初始化、CUDA Graph 捕获、生成预热及代表形状编译/预热均在计时前。普通 prefill 预热长度 256/512/768/1024，最后一组 16×1024 token；另预热页表宽度 2–8 的缓存前缀 prefill（新 Q 长度 128）。使用正式输入范围外的 token ID。
- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9。
- TTFT 从批量提交到 CPU Scheduler.postprocess 首 token 回填完成，包含排队。TPOT=(末 token 时间−首 token 时间)/(输出长度−1)；ITL 汇总所有请求相邻 token 间隔。总延迟为末 token 时间−批量提交时间。
- 吞吐分母为完整 generate 墙钟时间；model-run 阶段时间包括输入准备、模型、采样和同步，不能当作单 attention kernel 时间。
- 三种模式使用相同显存预算，INT8 的 KV 容量更大。抢占后的重新 prefill 会影响 BF16 总时间；根据阶段执行 token 和抢占计数解释收益。
- 单 scale INT8 每 token 缓存字节为 BF16 的 51.5625%；双 scale 为 53.125%。节省空间用于分配更多 blocks，完整缓存池与峰值显存不会按比例减少。显存为 PyTorch allocated/reserved 峰值。
- 单 scale 与双 scale 共用当前 _int8_paged_attention_kernel，通过 SCALE_GROUPS=1/2 区分；不使用此前 BM=64 实验副本。
- 生成内容可以因量化与调度轨迹而不同，固定采样种子不保证不同模式输出相同；输入和输出数量严格相同。
- workload.json 保存完整负载；manifest.json 保存源码哈希、模型配置、权重文件元数据、环境和命令；results.json 保存结构化结果，source_snapshot 保存原始源码，measurement_snapshot 保存实际执行源码。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_current_int8_metrics.py --outdir /tmp/current-int8-recheck --requests 512 --modes auto int8 int8_half --runs 3
```

## 首 token 之后的未完成请求数量

INT8 的 TTFT 尾部下降，而 TPOT/ITL 上升，需要结合请求的起止时间理解。
对每个请求，所有 ITL 之和等于末 token 时间减首 token 时间；把所有请求的这个时间相加，再除以整批 generate 时间，
可以估算平均有多少请求已经生成首 token、但还没有生成末 token。这里先逐轮计算，再取三轮中位数。

`平均请求数 = ITL_mean_ms × (输出 token 总数 − 请求数) / (1000 × generate_seconds)`

| 模式 | 平均已出首 token、尚未完成的请求数 |
| --- | ---: |
| BF16 | 165.8 |
| 单 scale INT8 | 316.4 |
| 双 scale INT8 | 306.5 |

这个数量包含被抢占后等待重新 prefill 的请求；没有记录每次 decode 的实际 batch size。
原始 ITL 和 generate 时间来自打印值，推导值有四舍五入误差。
当前 Scheduler 优先 prefill，并受可分配 KV 块数限制；INT8 更大的 KV 池让更多请求较早得到首 token。
这些数据与更多请求进入首 token 后的生成阶段、共享 GPU 执行时间的解释一致。
本表反映整个推理引擎的容量和调度行为；单个 attention kernel 的性能需要固定相同 batch、上下文和输入后单独测量。
