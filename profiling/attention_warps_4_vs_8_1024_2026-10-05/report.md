# 快速双 scale INT8：4 与 8 warp，1024 请求

仅修改 INT8 attention kernel 的 num_warps：prefill/decode 同时为 4 或 8。
prefill BM=64、decode BM=16、BN=64、num_stages=1，原 prefill 优先策略，preemption_lock=False。
两配置各 3 个独立新进程，交替运行；每项指标取各轮中位数。

| 指标 | 4 warp | 8 warp | 变化 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 28011.49 | 27123.24 | -3.17% |
| TTFT P50 ms | 34017.07 | 32897.23 | -3.29% |
| TTFT P95 ms | 73449.17 | 70736.81 | -3.69% |
| TTFT P99 ms | 77682.41 | 74776.95 | -3.74% |
| TPOT MEAN ms | 53.43 | 51.31 | -3.97% |
| TPOT P50 ms | 52.22 | 50.20 | -3.87% |
| TPOT P95 ms | 77.09 | 74.08 | -3.90% |
| TPOT P99 ms | 153.61 | 146.91 | -4.36% |
| ITL MEAN ms | 50.83 | 48.75 | -4.09% |
| ITL P50 ms | 39.40 | 37.11 | -5.81% |
| ITL P95 ms | 73.58 | 72.56 | -1.39% |
| ITL P99 ms | 97.49 | 97.93 | +0.45% |
| End-to-end latency MEAN ms | 56940.67 | 54869.73 | -3.64% |
| End-to-end latency P50 ms | 56828.49 | 54874.20 | -3.44% |
| End-to-end latency P95 ms | 91180.35 | 87529.42 | -4.00% |
| End-to-end latency P99 ms | 93399.67 | 89647.20 | -4.02% |
| 整批 generate 秒 | 94.249 | 90.468 | -4.01% |
| 输出 token/s | 6194.27 | 6453.16 | +4.18% |
| 请求/s | 10.86 | 11.32 | +4.24% |
| 输入 token/s | 6160.97 | 6418.46 | +4.18% |
| 总 token/s | 12355.24 | 12871.62 | +4.18% |
| 抢占事件次数 | 653 | 653 | +0.00% |
| KV blocks | 1320 | 1320 | +0.00% |
| 峰值 allocated GiB | 20.76 | 20.76 | +0.00% |
| 峰值 reserved GiB | 21.06 | 21.06 | +0.00% |
| preempted_requests | 305 | 305 | +0.00% |
| repeatedly_preempted_requests | 132 | 132 | +0.00% |
| maximum_per_request | 17 | 17 | +0.00% |
| prefill model-run seconds | 21.224 | 21.682 | +2.16% |
| prefill model-run tokens | 877980 | 877980 | +0.00% |
| prefill model-run tokens_per_s | 41366.44 | 40493.29 | -2.11% |
| decode model-run seconds | 71.909 | 67.716 | -5.83% |
| decode model-run tokens | 582125 | 582125 | +0.00% |
| decode model-run tokens_per_s | 8095.34 | 8596.61 | +6.19% |
| 实际 decode batch mean | 244.28 | 244.28 | +0.00% |
| 实际 decode batch p50 | 302.00 | 302.00 | +0.00% |
| 实际 decode batch p95 | 417.00 | 417.00 | +0.00% |
| 实际 decode batch maximum | 490.00 | 490.00 | +0.00% |
| 实际 decode batch steps | 2383.00 | 2383.00 | +0.00% |

## 每轮结果

| warp | 轮次 | 输出 token/s | generate s | TTFT P50 ms | TPOT P50 ms | 抢占 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 6218.47 | 93.882 | 33896.73 | 51.99 | 653 |
| 8 | 1 | 6459.17 | 90.383 | 32841.39 | 50.14 | 653 |
| 8 | 2 | 6453.16 | 90.468 | 32897.23 | 50.20 | 653 |
| 4 | 2 | 6194.27 | 94.249 | 34017.07 | 52.22 | 653 |
| 4 | 3 | 6189.29 | 94.324 | 34063.30 | 52.27 | 653 |
| 8 | 3 | 6428.12 | 90.820 | 33004.99 | 50.40 | 653 |

## 正确性与测量范围

- 测量前，4/8 warp 与还原 BF16 KV 后的 FlashAttention 比较；decode、零长度 graph padding、普通 prefill、缓存前缀 prefill 共 5 类用例通过。模型形状 Q heads=16、KV heads=8、head_dim=128，双 scale，rtol=0.02、atol=0.005；详见 correctness.json。
- Qwen3-0.6B / RTX 3090 Ti / GPU 0，权重 BF16，KV 为双 scale INT8。
- 复用上次完整 workload_1024.json，输入 580663、输出 583802 token。输入/输出长度各 100–1024；seed=0、temperature=0.6、ignore_eos=True。每请求输出长度和时间戳数量均验证。
- max_num_seqs=512、max_num_batched_tokens=16384、max_model_len=4096、block size=256、gpu_memory_utilization=0.9，decode CUDA Graph、prefill eager。未锁 GPU 频率。
- 初始化和预热在计时前，普通 prefill 长度 256/512/768/1024（最大 16×1024）；缓存前缀 prefill 页表宽度 2–8、新 Q 长度 128。预热后重置采样 seed 和显存峰值。
- TTFT 包含排队，取 CPU postprocess 首 token 回填时间；TPOT 为每请求首末 token 时间差除以后续 token 数；ITL 汇总相邻 token 间隔。
- model-run 阶段时间包括输入准备、模型、采样和同步，不能当作单 attention kernel 的 GPU 时间。
- CUDA Graph 捕获和正式 prefill/decode 均使用对应 4/8 warp 源码；实际包路径、attention 源码配置、KV dtype、prefill 派发和源码哈希均核对。
- 主仓库推理源码及 reference 源码均未修改；结果只适用于本次固定模型、负载与 kernel 版本。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_attention_warps_metrics.py --outdir /tmp/attention-warps-recheck --runs 3
```
