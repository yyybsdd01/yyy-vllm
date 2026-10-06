# 当前二维加载双 scale INT8 与 BF16 KV cache 的推理指标（2026-10-04）

当前双 scale INT8 的输出吞吐比 BF16 提升 **6.18%**；TTFT P50 变化 **-0.43%**，差异很小；TPOT P50 变化 **-6.99%**。ITL 的典型值与尾部变化不同，见完整分布。

## 测量条件

- Qwen3-0.6B，RTX 3090 Ti，单卡 GPU 0；模型权重均为 BF16。
- `auto`：BF16 KV cache + FlashAttention。`int8_half`：INT8 KV cache，K/V 各按 head 前后半维保留两个 FP32 scale，融合 Triton attention；使用已接入的 `[BN,2]` 二维 scale 加载。
- CUDA Graph 开启；max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、KV block size=256、gpu_memory_utilization=0.9。
- 256 个请求同时提交；随机 token ID 输入长度 100–1024，目标输出长度 100–1024；seed=0、temperature=0.6、ignore_eos=True。每轮输入 142,827 token、输出 133,966 token，各请求长度校验通过。
- 两种模式各 3 个独立进程，顺序 auto→INT8 / INT8→auto / auto→INT8。每进程加载与一次生成预热不计入正式测量。两种模式使用同一工作树，已逐轮核对源码哈希。
- 先计算每轮分布指标，再逐项取三轮中位数；没有把六轮请求混在一起计算一个分位数。
- 沿用原基准协议：seed 固定 Python 负载生成，未设置 PyTorch 采样随机种子。固定输出长度保证同一工作量；输出内容因采样与量化可以不同。
- 环境：PyTorch 2.5.1+cu121、Triton 3.1.0、FlashAttention 2.7.4.post1、Transformers 4.57.3、驱动 535.183.01。
- 完整负载见 `workload.json`，配置和逐轮命令见 `manifest.json`，源码见 `source_snapshot/`，结构化结果见 `results.json`。

## 延迟分布

单位毫秒。TTFT 和请求总延迟为 ms/request，TPOT 为每请求平均 ms/token，ITL 为相邻 token 的 ms/token。

| 指标 | 分位/统计量 | BF16 | 当前双 scale INT8 | 相对变化 |
| --- | --- | ---: | ---: | ---: |
| TTFT | MEAN | 1304.58 | 1299.31 | -0.40% |
| TTFT | P50 | 1342.11 | 1336.31 | -0.43% |
| TTFT | P95 | 2281.41 | 2274.45 | -0.31% |
| TTFT | P99 | 2281.41 | 2274.45 | -0.31% |
| TPOT | MEAN | 32.48 | 29.98 | -7.70% |
| TPOT | P50 | 32.03 | 29.79 | -6.99% |
| TPOT | P95 | 42.32 | 37.68 | -10.96% |
| TPOT | P99 | 57.86 | 42.34 | -26.82% |
| ITL | MEAN | 29.88 | 27.99 | -6.33% |
| ITL | P50 | 26.84 | 27.98 | +4.25% |
| ITL | P95 | 53.78 | 29.82 | -44.55% |
| ITL | P99 | 55.93 | 30.06 | -46.25% |
| End-to-end latency | MEAN | 16910.47 | 15915.91 | -5.88% |
| End-to-end latency | P50 | 17755.38 | 16455.04 | -7.32% |
| End-to-end latency | P95 | 25181.66 | 24598.10 | -2.32% |
| End-to-end latency | P99 | 25495.51 | 24775.81 | -2.82% |

## 吞吐、阶段执行与容量

| 指标 | BF16 | 当前双 scale INT8 | 相对变化 |
| --- | ---: | ---: | ---: |
| 256 请求总时间，s | 26.393 | 24.857 | -5.82% |
| 请求吞吐，request/s | 9.70 | 10.30 | +6.19% |
| 输入 token/s | 5411.58 | 5745.84 | +6.18% |
| 输出 token/s | 5075.84 | 5389.37 | +6.18% |
| 输入+输出 token/s | 10487.42 | 11135.21 | +6.18% |
| 可分配 KV 块 | 701 | 1320 | +88.30% |
| 峰值 PyTorch allocated，GiB | 20.75 | 20.76 | +0.05% |
| 峰值 PyTorch reserved，GiB | 21.21 | 21.21 | +0.00% |
| prefill model-run token/s | 46907.97 | 63077.44 | +34.47% |
| prefill 实际执行 token | 181836 | 142827 | -21.45% |
| prefill model-run 秒 | 3.876 | 2.264 | -41.59% |
| decode model-run token/s | 6005.88 | 5981.85 | -0.40% |
| decode 实际执行 token | 133626 | 133710 | +0.06% |
| decode model-run 秒 | 22.249 | 22.353 | +0.47% |

## 如何解释

1. **TTFT 基本不变。** 普通首次 prefill 两种模式都走 FlashAttention；INT8 额外量化并写缓存。当前二维 scale 读取主要影响 decode 和带缓存前缀的 prefill，不能据此期待首次 prefill 的 TTFT 同比例下降。
2. **端到端吞吐收益包含容量与调度差异。** BF16 prefill 实际执行 181,836 token，INT8 执行 142,827 token。BF16 有额外 prefill 重算；INT8 更大的 KV 容量减少了本负载的重算。因此 prefill 执行吞吐不可直接当成同一形状 attention 的速度比，端到端增益也不能全部归因于 scale 加载优化。
3. **Decode 执行吞吐变化较小。** BF16 为 6005.88 token/s，INT8 为 5981.85 token/s。阶段时间以 `ModelRunner.run` 调用为边界，含准备、模型、采样和同步。两种模式的调度轨迹不相同。
4. **ITL P50 与 P95 应同时看。** INT8 的典型 token 间隔略高，但尾部间隔下降；TPOT 是每请求首末 token 时间差除以后续 token 数，ITL 汇总全部请求的相邻间隔，两种统计口径和权重不同。容量更大的模式可以维持不同的活跃请求数量，尾部也包含重算、调度造成的间隔。
5. **峰值显存接近来自固定的容量预算。** BF16 每个 token 全 28 层 KV 为 112 KiB，双 scale INT8（包括 FP32 scales）为 59.5 KiB，减少 46.875%。项目把同一 90% 显存预算用于更多 KV 块，所以主要表现为 701→1320 块（+88.30%），而非已分配显存减半。显存数字是 PyTorch 分配器的峰值，没有测 NVML 进程峰值。

## 指标边界

- `LLMEngine.generate → step → Scheduler.schedule → ModelRunner.run → Scheduler.postprocess`。
- TTFT 从批量提交到首 token 在 CPU postprocess 回填完成，包含排队和准备；不是 GPU 内核的首 token event 时间。
- TPOT=`(t_last−t_first)/(output_tokens−1)`，先逐请求计算再取分布；ITL 是所有相邻 token 时间差；请求总延迟=`t_last−t_batch_start`。
- 吞吐统一使用完整 generate 墙钟时间，模型加载、初始化与预热均排除。阶段执行吞吐使用各阶段 model-run 的累计时间。
- 这是随机 token ID 的离线批量负载，不是在线服务测量；本轮测性能与长度，不评价量化后的自然语言回答质量。

## 每轮原始记录

| 模式 | 轮次 | 总时间 s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 日志 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| auto | 1 | 26.237 | 5105.90 | 1342.11 | 31.76 | [auto_1.txt](auto_1.txt) |
| int8_half | 1 | 24.852 | 5390.52 | 1335.76 | 29.79 | [int8_half_1.txt](int8_half_1.txt) |
| int8_half | 2 | 24.857 | 5389.37 | 1336.31 | 29.78 | [int8_half_2.txt](int8_half_2.txt) |
| auto | 2 | 26.393 | 5075.84 | 1340.64 | 32.05 | [auto_2.txt](auto_2.txt) |
| auto | 3 | 26.404 | 5073.62 | 1345.36 | 32.03 | [auto_3.txt](auto_3.txt) |
| int8_half | 3 | 24.866 | 5387.56 | 1343.99 | 29.80 | [int8_half_3.txt](int8_half_3.txt) |

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_kv_metrics_comparison.py --outdir /tmp/kv_metrics_packed_vs_bf16_recheck
```

复现应使用本报告 source_snapshot 对应的代码状态；当前实验的每轮启动和结束均检查了 21 个源文件的哈希。
当前 attention 内核 SHA256：`310be258172cfcadc8ca728f3c0ea0de0427c7569ce59f35e308bbd20e3dff2a`。
