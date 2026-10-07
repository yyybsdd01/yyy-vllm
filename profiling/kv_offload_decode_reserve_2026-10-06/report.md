# KV 卸载：预留下一批 decode 扩块空间

相同 Qwen3-0.6B、GPU0 RTX 3090 Ti、快速双 scale INT8 kernel、1318 个 KV 块。
相同 1024 请求（输入580663、输出583802 token），独立进程交替测量，各项取三轮中位数。
预留量为 running 队首 max_num_seqs 个请求当前缺失的逻辑块数之和；
恢复和 waiting prefill 都需在分配后保留这部分空闲块。关闭卸载的调度行为不变。

| 指标 | 关闭卸载 | 原卸载 | 预留 decode 块 |
| --- | ---: | ---: | ---: |
| TTFT mean ms | 27253.69 | 25084.98 | 25080.80 |
| TTFT p50 ms | 33012.96 | 17780.16 | 30452.23 |
| TTFT p95 ms | 71085.70 | 64388.91 | 65174.21 |
| TTFT p99 ms | 75233.51 | 70162.11 | 69188.11 |
| TPOT mean ms | 51.57 | 50.97 | 49.19 |
| TPOT p50 ms | 50.41 | 47.38 | 46.56 |
| TPOT p95 ms | 74.41 | 81.86 | 73.73 |
| TPOT p99 ms | 147.82 | 182.60 | 142.86 |
| ITL mean ms | 48.94 | 47.16 | 45.98 |
| ITL p50 ms | 37.21 | 37.79 | 37.87 |
| ITL p95 ms | 72.50 | 70.11 | 69.03 |
| ITL p99 ms | 98.35 | 95.21 | 99.75 |
| End-to-end latency mean ms | 55108.25 | 51926.71 | 51251.37 |
| End-to-end latency p50 ms | 54965.15 | 51888.27 | 51167.46 |
| End-to-end latency p95 ms | 87969.20 | 82820.09 | 81955.38 |
| End-to-end latency p99 ms | 90137.05 | 84916.30 | 84106.35 |
| output_tokens_per_s | 6418.460 | 6806.030 | 6864.970 |
| requests_per_s | 11.260 | 11.940 | 12.040 |
| elapsed_s | 90.957 | 85.777 | 85.041 |
| preemptions | 657 | 716 | 455 |
| peak_allocated_gib | 20.730 | 20.760 | 20.760 |
| peak_reserved_gib | 21.030 | 21.060 | 21.060 |
| preempted_requests | 302 | 282 | 221 |
| repeatedly_preempted_requests | 122 | 163 | 111 |
| maximum_per_request | 16 | 14 | 9 |
| restore_events | 0 | 708 | 455 |
| preemptions_without_token_progress | 0 | 183 | 66 |
| same_schedule_preemptions | 0 | 182 | 66 |
| prefill tokens | 879176 | 586886 | 580663 |
| prefill seconds | 21.756 | 13.958 | 13.333 |
| prefill tokens_per_s | 40411.590 | 42047.910 | 43549.350 |
| decode tokens | 582121 | 582770 | 582778 |
| decode seconds | 67.880 | 68.507 | 68.403 |
| decode tokens_per_s | 8575.770 | 8506.750 | 8519.730 |

每轮结果：

| 策略 | 轮次 | 输出 token/s | 抢占 | 恢复后无进度再抢占 | 同一次调度内 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原卸载 | 1 | 6856.82 | 716 | 183 | 182 |
| 预留 decode 块 | 1 | 6865.28 | 455 | 66 | 66 |
| 关闭卸载 | 1 | 6414.61 | 657 | 0 | 0 |
| 关闭卸载 | 2 | 6433.48 | 657 | 0 | 0 |
| 预留 decode 块 | 2 | 6823.95 | 455 | 67 | 67 |
| 原卸载 | 2 | 6806.03 | 716 | 183 | 182 |
| 原卸载 | 3 | 6798.36 | 716 | 183 | 182 |
| 预留 decode 块 | 3 | 6864.97 | 455 | 66 | 66 |
| 关闭卸载 | 3 | 6418.46 | 657 | 0 | 0 |

测量边界：

- max_num_seqs512、max_num_batched_tokens16384、block256、max_model_len4096。
- CPU pinned 池4GiB、最多8个同时恢复请求；两种卸载策略的 staging 和 GPU KV 容量相同。
- attention 使用既有 fast8 测量副本，根目录 attention 未修改；两组差异仅在三个调度/准入文件。
- 初始化、编译、代表性普通/缓存 prefill 预热在计时外；CPU池首次分配在计时内。
- TTFT包含排队；TPOT是每请求首末 token 间隔除以后续 token 数。
- 恢复后无进度再抢占表示恢复完成后尚未生成任何新token即被抢占；同一次调度内是其子集。
- 所有策略使用相同 CPU 计数探针；未锁GPU频率，异步事件完成时机可改变调度。
- 检查每请求输出长度、时间戳数量、完整负载hash、抢占直方图总和、传输结束和CPU池预算。
- 随机采样随 batch/调度变化；本次性能对照不构成PPL或生成质量评估。

## 结论与正确性

相对原卸载，抢占 716 → 455（-36.45%），输出吞吐 6806.03 → 6864.97 token/s（+0.87%）。

恢复后未生成新 token 即再次抢占 183 → 66（-63.93%）；同一次 schedule 内 182 → 66。三轮预留方案抢占均455，无进度再抢占分别66、67、66。

TPOT P99 182.60 → 142.86 ms（-21.76%），TTFT P50 17.780 → 30.452 s（+71.27%）。TTFT均值基本不变；减少抢占和生成尾延迟的同时，首token排队分布发生变化。

20 个测试方法通过：11个CPU卸载测试、2个真实CUDA搬运测试、7个旧抢占锁测试。BF16和双scale INT8整模型各逐元素检查14个恢复逻辑块的全部有效KV行和scales，共享前缀验证通过；完整1024请求输出长度和时间戳检查通过。未做正式生成质量评测。

一轮预留无法保证异步恢复期间一直有余量。CPU小复现表明：A在256-token长度时无需扩块，恢复B获准；B的H2D尚未完成时A生成至257-token，新增1块需求；B恢复完成后仍可能被立即抢占。这个例子证明剩余循环的可能路径，不能将本次66次残余全部归因于它。见 [复现脚本](reproduce_inflight_boundary.py) 与 [结果](reproduce_inflight_boundary.json)。

原源码保存在 `before_change/`，两组完整测量源码保存在 `variants/`。实现差异见 [implementation.diff](implementation.diff)，结构化正确性检查见 [validation.json](validation.json)，传输和CPU池中位数见 [comparison_summary.json](comparison_summary.json)。

复现使用新的输出目录：

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_kv_offload_decode_reserve.py --outdir /tmp/kv-offload-reserve-recheck --runs 3 --include-disabled
```
