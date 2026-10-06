# 双 scale 融合 INT8 的推理指标（2026-10-02）

## 测评条件

Qwen3-0.6B、RTX 3090 Ti、单卡、CUDA Graph。与[原融合 INT8 测评](kv_cache_fused_comparison_2026-10-02.md)使用同一 `benchmark_inference_metrics.py` 负载：随机种子 0、256 个请求同时提交、输入 142,827 token、固定输出 133,966 token、`temperature=0.6`、`ignore_eos=True`、长度校验通过；模型加载和一次预热不计入测量。每个模式运行 3 个独立进程，下表对每项指标取三轮中位数。延迟均先在一轮内按请求或 token 计算，再取三轮对应统计量的中位数。

先测得双 scale 原始内核（每个 head 前后 64 维各一个 scale，融合 attention 使用 32-key tile）吞吐明显下降。随后把重复的逐维 scale 加载改为每个 KV token 读取两个 scale，并恢复 64-key tile。此变更只影响 `int8_half` 融合分支，下面将改前和改后的原始日志分开保留。当前工作树使用优化后的实现。

## 吞吐与资源

| 指标 | `auto` BF16 | `int8` 单 scale | `int8_half` 原始 | `int8_half` 当前 |
| --- | ---: | ---: | ---: | ---: |
| PPL（同一 WikiText-2 协议） | 18.810083 | 19.056158 | 18.938513 | **18.944100** |
| 可分配 KV 块 | 701 | 1,360 | 1,320 | 1,320 |
| 256 请求总时间，秒 | 26.375 | 23.258 | 57.883 | 26.988 |
| 请求吞吐，请求/s | 9.71 | **11.01** | 4.42 | 9.49 |
| 输入 token/s | 5,415.32 | 6,141.10 | 2,467.51 | 5,292.19 |
| 输出 token/s | 5,079.35 | **5,760.11** | 2,314.42 | 4,963.87 |
| 输入+输出 token/s | 10,494.67 | 11,901.21 | 4,781.93 | 10,256.06 |
| Prefill 模型执行 token/s | 46,811.65 | 63,195.87 | 63,206.65 | 63,387.05 |
| Decode 模型执行 token/s | 6,015.63 | **6,442.13** | 2,414.76 | 5,456.92 |
| PyTorch 峰值 allocated，GiB | 20.75 | 20.76 | 20.76 | 20.76 |
| PyTorch 峰值 reserved，GiB | 21.21 | 21.21 | 21.21 | 21.21 |

当前双 scale 输出吞吐相对单 scale **低 13.82%**，相对 BF16 **低 2.27%**；但 PPL 比单 scale 低 **0.112058**（19.056158 → 18.944100）。原始双 scale 内核仅 2,314.42 token/s；改为向量加载 scale 并恢复 64-key tile 后达到 4,963.87 token/s。三轮当前输出吞吐分别为 **4,965.14 / 4,963.87 / 4,942.46 token/s**。这两个内核调整同时发生，不能从端到端测量单独归因其各自贡献。

`auto` 的 prefill 模型运行处理了 181,836 token，而两个 INT8 模式均处理 142,827 token；`auto` 在此负载下发生了额外的 prefill 重算。因此阶段执行吞吐和端到端吞吐不能直接解释为单个 attention 内核的速度。两种 INT8 模式的 prefill 执行吞吐接近，性能差异主要出现在 decode。可分配块数的变化来自不同 scale 字节数和缓存预算；PyTorch 峰值几乎相同，因为该脚本把显存预算用于更多 KV 块。

### 单层 decode attention 计时

为核对“缓存字节更少却更慢”，另以同一 Qwen3-0.6B head 布局（16 个 query head、8 个 KV head、128 维）对单层 decode attention 做 CUDA Graph + CUDA Event 计时。K/V 为非零随机数据，batch 256，所有序列分别具有 512 或 1,024 token 的历史；每项 5 轮、每轮 20 次 graph replay，以下为每次调用耗时的中位数。这是 attention 调用本身的微基准，不包含 KV 写入、其他模型层或调度。

| 单层 attention | BF16 FlashAttention | 单 scale INT8 Triton | 双 scale INT8 Triton |
| --- | ---: | ---: | ---: |
| batch 256，context 512 | **591.51 µs** | 633.91 µs | 778.75 µs |
| batch 256，context 1,024 | **1,154.85 µs** | 1,175.08 µs | 1,454.89 µs |

在 1,024-token 行，按缓存张量的逻辑大小计算，BF16 K/V 为 1,024 MiB，双 scale INT8 的 K/V 加 scale 为 544 MiB；双 scale 的 attention 调用仍慢约 26%。逻辑大小不等于实测 DRAM 事务量。当前 Triton 内核每个 KV token 读取两组 K/V scale、把 INT8 转成浮点并执行在线 softmax；decode 时每个 KV head 只有 2 个有效 query head，但 `BM=16` 的点积 tile 仍按 16 行计算。这些是明确存在的额外工作；微基准证明 slowdown 在 attention 调用内，但尚未把各指令、访存事务和占用率的贡献分开。全零 KV 的同形状复测也得到相同的排序。

微基准脚本：[`bench_paged_attention_read.py`](bench_paged_attention_read.py)；[随机 K/V 日志](bench_paged_attention_read_random_2026-10-02.txt)、[全零 K/V 对照日志](bench_paged_attention_read_2026-10-02.txt)。

## 延迟分布

每个单元格按「均值 / P50 / P95 / P99」排列，单位毫秒。

| 指标 | `auto` BF16 | `int8` 单 scale | `int8_half` 原始 | `int8_half` 当前 |
| --- | --- | --- | --- | --- |
| TTFT | 1305.43 / 1341.17 / 2281.49 / 2281.49 | 1298.25 / 1334.99 / 2270.38 / 2270.38 | 1300.59 / 1338.18 / 2270.10 / 2270.10 | **1294.79 / 1331.69 / 2263.24 / 2263.24** |
| TPOT | 32.41 / 31.99 / 42.17 / 57.79 | **28.04 / 27.73 / 35.75 / 40.38** | 70.37 / 72.29 / 80.68 / 84.95 | 32.69 / 32.57 / 40.46 / 45.10 |
| ITL | 29.83 / 26.77 / 53.99 / 56.15 | **26.11 / 25.98 / 27.73 / 27.92** | 66.65 / 68.98 / 75.00 / 75.37 | 30.57 / 30.69 / 33.02 / 33.20 |
| 请求总延迟 | 16891.47 / 17736.78 / 25161.18 / 25475.26 | **14936.45 / 15413.12 / 23010.03 / 23177.99** | 36112.05 / 37323.97 / 57380.28 / 57748.04 | 17260.80 / 17850.24 / 26724.01 / 26905.88 |

当前双 scale 的 TTFT P50 为 **1,331.69 ms**，与单 scale 的 **1,334.99 ms** 接近；TPOT P50 为 **32.57 vs 27.73 ms/token**，请求总延迟 P50 为 **17.85 vs 15.41 s**。TTFT 从批量提交到首 token 在 CPU `Scheduler.postprocess()` 回填完成，包含排队；它不是 GPU 上首 token 的独立事件时间。TPOT 是每个请求首末 token 时间差除以后续 token 数；ITL 汇总相邻 token 间隔。阶段吞吐以 `ModelRunner.run` 调用时间为分母，包含其准备和同步，与完整 `generate()` 墙钟吞吐口径不同。PyTorch 显存数字不是 NVML 进程峰值。

## 运行记录

- 当前双 scale 三轮：[第 1 轮](kv_cache_half_perf_vector_2026-10-02/int8_half_1.txt)、[第 2 轮](kv_cache_half_perf_vector_2026-10-02/int8_half_2.txt)、[第 3 轮](kv_cache_half_perf_vector_2026-10-02/int8_half_3.txt)。当前代码重算的[全量 PPL](kv_cache_perplexity_half_vector_2026-10-02.txt)。
- 同负载重测的 `auto`、单 scale `int8`、原始双 scale 各三轮：[原始日志目录](kv_cache_half_perf_2026-10-02/)。这些运行均校验了输出 token 数。`auto` 和单 scale `int8` 的路径在双 scale 内核优化时未改动。
- 复现：`/home/xgd/anaconda3/envs/nanovllm/bin/python benchmark_inference_metrics.py --kv-cache-dtype int8_half`。脚本固定以上默认负载；每轮使用全新进程。原始双 scale 内核的 32-key tile 性能数字仅保留作这次优化的对照，当前代码无法直接复现那一行。
