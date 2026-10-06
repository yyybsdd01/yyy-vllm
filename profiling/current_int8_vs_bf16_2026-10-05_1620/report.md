# 当前仓库 INT8 kernel 与未量化 BF16 的完整推理对比

当前推理源码完整保留，仅在独立副本添加测评预热、固定采样种子和 CPU 计数。
模型权重均为 BF16；INT8 仅量化 KV cache。普通首次 prefill 仍使用 FlashAttention；
INT8 自写 Triton kernel 用于 decode 和带缓存前缀的 prefill（BM=16、BN=64、4 warps，默认流水）。

| 指标 | BF16 | 单 scale INT8 | 双 scale INT8 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 1178.79 | 1179.62 | 1179.81 |
| TTFT P50 ms | 1216.90 | 1217.79 | 1217.65 |
| TTFT P95 ms | 2154.38 | 2156.18 | 2157.41 |
| TTFT P99 ms | 2154.38 | 2156.18 | 2157.41 |
| TPOT MEAN ms | 32.39 | 28.07 | 30.04 |
| TPOT P50 ms | 31.95 | 27.74 | 29.83 |
| TPOT P95 ms | 42.18 | 35.78 | 37.72 |
| TPOT P99 ms | 57.75 | 40.46 | 42.40 |
| ITL MEAN ms | 29.81 | 26.13 | 28.03 |
| ITL P50 ms | 26.74 | 26.01 | 27.97 |
| ITL P95 ms | 54.08 | 27.78 | 29.89 |
| ITL P99 ms | 55.56 | 27.94 | 30.09 |
| End-to-end latency MEAN ms | 16750.56 | 14829.82 | 15820.66 |
| End-to-end latency P50 ms | 17594.01 | 15293.73 | 16359.38 |
| End-to-end latency P95 ms | 25017.26 | 22924.99 | 24510.22 |
| End-to-end latency P99 ms | 25331.44 | 23092.91 | 24688.37 |
| generate s | 26.229 | 23.170 | 24.768 |
| 请求/s | 9.76 | 11.05 | 10.34 |
| 输出 token/s | 5107.59 | 5781.86 | 5408.78 |
| 输入 token/s | 5445.43 | 6164.29 | 5766.54 |
| 总 token/s | 10553.02 | 11946.15 | 11175.32 |
| KV blocks | 701 | 1360 | 1320 |
| 缓存抢占次数 | 84 | 0 | 0 |
| 峰值 allocated GiB | 20.75 | 20.76 | 20.76 |
| 峰值 reserved GiB | 21.05 | 21.06 | 21.06 |
| KV cache + scales GiB | 19.1680 | 19.1748 | 19.1748 |
| prefill model-run seconds | 3.765 | 2.146 | 2.148 |
| prefill model-run tokens | 181836 | 142827 | 142827 |
| prefill model-run tokens_per_s | 48295.98 | 66542.65 | 66507.41 |
| decode model-run seconds | 22.209 | 20.784 | 22.388 |
| decode model-run tokens | 133626 | 133710 | 133710 |
| decode model-run tokens_per_s | 6016.79 | 6433.21 | 5972.44 |

## 相对 BF16 的变化

| 模式 | TTFT P50 | TPOT P50 | ITL P50 | 输出吞吐 | generate 时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单 scale INT8 | +0.07% | -13.18% | -2.73% | +13.20% | -11.66% |
| 双 scale INT8 | +0.06% | -6.64% | +4.60% | +5.90% | -5.57% |

## 每轮结果

| 模式 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto | 1 | 26.086 | 5135.61 | 1201.02 | 31.76 | 84 |
| int8 | 1 | 23.082 | 5803.86 | 1208.30 | 27.63 | 0 |
| int8_half | 1 | 24.698 | 5424.27 | 1213.54 | 29.73 | 0 |
| int8 | 2 | 23.171 | 5781.66 | 1217.79 | 27.74 | 0 |
| int8_half | 2 | 24.792 | 5403.56 | 1220.44 | 29.83 | 0 |
| auto | 2 | 26.229 | 5107.59 | 1216.90 | 31.95 | 84 |
| int8_half | 3 | 24.768 | 5408.78 | 1217.65 | 29.84 | 0 |
| auto | 3 | 26.300 | 5093.74 | 1225.63 | 32.06 | 84 |
| int8 | 3 | 23.170 | 5781.86 | 1220.26 | 27.80 | 0 |

## 测量范围

- Qwen3-0.6B / NVIDIA GeForce RTX 3090 Ti / GPU 0，单卡，decode CUDA Graph、prefill eager。未锁 GPU 频率。
- 每模式 3 个独立新进程，逐轮循环轮换运行顺序，全部为本次新测量。每轮请求/token 分布先计算 Mean/P50/P95/P99，再逐项取各轮中位数。
- 256 请求同时到达，输入长度 100–1024，输出长度 100–1024，负载和 PyTorch 采样 seed=0。temperature=0.6、ignore_eos=True。
- 每轮输入 142,827、输出 133,966 token；每请求输出长度、时间戳数量均验证通过。随机 token ID 离线负载，未评价自然语言质量。
- 初始化、编译、生成预热及代表形状预热均在计时前。代表预热为长度 256/512/768/1024，最后一组 16×1024 token；使用正式输入范围外的 token ID。
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
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_current_int8_metrics.py --outdir /tmp/current-int8-recheck --modes auto int8 int8_half --runs 3
```
