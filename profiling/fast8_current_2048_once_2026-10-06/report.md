# fast8 当前卸载调度：2048 请求单轮

fast8 为本次新测 1 轮；BF16 和根目录 INT8 为前一轮相同负载的三轮中位数。
GPU0 RTX 3090 Ti / Qwen3-0.6B，模型权重 BF16，fast8 为双 scale INT8 KV。
fast8：prefill BM64、decode BM16、BN64、8 warps、num_stages=1；首次/缓存 prefill 和 decode 使用 INT8 内核。
当前异步卸载＋decode 块预留；CPU pinned 池4GiB，最多8个恢复请求；CUDA Graph decode。
输入 1150273、输出 1128906 token；seed0，长度各100–1024，2048请求同时提交，max_num_seqs512。

| 指标 | BF16（旧3轮中位数） | 根目录INT8（旧3轮中位数） | fast8（新1轮） |
| --- | ---: | ---: | ---: |
| output_tokens_per_s | 4980.220 | 5428.800 | 6968.840 |
| requests_per_s | 9.030 | 9.850 | 12.640 |
| input_tokens_per_s | 5074.480 | 5531.560 | 7100.740 |
| total_tokens_per_s | 10054.700 | 10960.360 | 14069.580 |
| elapsed_s | 226.678 | 207.947 | 161.993 |
| preemptions | 1566.000 | 900.000 | 900.000 |
| blocks | 701.000 | 1318.000 | 1318.000 |
| peak_allocated_gib | 20.750 | 20.760 | 20.760 |
| peak_reserved_gib | 21.050 | 21.060 | 21.060 |
| TTFT mean ms | 102081.990 | 83134.260 | 65368.790 |
| TTFT p50 ms | 102963.680 | 86490.780 | 67796.210 |
| TTFT p95 ms | 202223.590 | 175117.050 | 137060.650 |
| TTFT p99 ms | 210774.880 | 184270.850 | 144162.650 |
| TPOT mean ms | 35.140 | 60.990 | 47.610 |
| TPOT p50 ms | 35.140 | 60.440 | 47.120 |
| TPOT p95 ms | 37.640 | 72.590 | 60.400 |
| TPOT p99 ms | 52.490 | 112.280 | 86.760 |
| ITL mean ms | 34.780 | 59.590 | 46.340 |
| ITL p50 ms | 27.290 | 51.420 | 37.730 |
| ITL p95 ms | 57.380 | 82.320 | 71.550 |
| ITL p99 ms | 69.250 | 101.220 | 94.320 |
| End-to-end latency mean ms | 121218.090 | 115920.580 | 90866.090 |
| End-to-end latency p50 ms | 121917.570 | 116677.600 | 91490.380 |
| End-to-end latency p95 ms | 220055.460 | 202655.480 | 157761.220 |
| End-to-end latency p99 ms | 225092.840 | 206688.540 | 160831.550 |
| prefill tokens | 1758555.000 | 1150273.000 | 1150273.000 |
| prefill seconds | 48.272 | 27.785 | 29.351 |
| prefill tokens_per_s | 36430.310 | 41399.660 | 39189.700 |
| decode tokens | 1125292.000 | 1126858.000 | 1126858.000 |
| decode seconds | 175.536 | 174.569 | 127.041 |
| decode tokens_per_s | 6410.620 | 6455.090 | 8870.000 |
| request_preemption_stats.preempted_requests | 923.000 | 527.000 | 527.000 |
| request_preemption_stats.repeatedly_preempted_requests | 360.000 | 220.000 | 220.000 |
| request_preemption_stats.maximum_per_request | 9.000 | 8.000 | 8.000 |
| decode_batch_stats.mean | 161.078 | 285.064 | 285.064 |
| decode_batch_stats.p50 | 173.000 | 322.000 | 322.000 |
| decode_batch_stats.p95 | 190.000 | 385.000 | 385.000 |
| decode_batch_stats.maximum | 265.000 | 489.000 | 489.000 |
| decode_batch_stats.steps | 6986.000 | 3953.000 | 3953.000 |
| restore_progress.restore_events | 0.000 | 900.000 | 900.000 |
| restore_progress.preemptions_without_token_progress | 0.000 | 124.000 | 124.000 |
| restore_progress.same_schedule_preemptions | 0.000 | 124.000 | 124.000 |

TTFT 包含排队；TPOT 为每请求首末 token 时间差除以后续 token 数；ITL 汇总相邻 token 间隔。
整模型性能包含 KV 容量、prefill 路径、内核与调度的共同影响；单轮结果用于本次试测。
核验：正式prefill派发flash=0/int8>0；工作负载hash一致；每请求长度/时间戳通过；传输结束、CPU预算、抢占直方图与推理源码hash通过。
本次未重测PPL，性能结果不构成质量评估。

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_fast8_2048_once.py --outdir /tmp/fast8-2048-recheck
```
