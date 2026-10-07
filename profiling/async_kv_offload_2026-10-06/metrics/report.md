# 1024 请求：异步 KV 卸载与恢复

同一快速双 scale INT8 kernel：prefill BM64、decode BM16、BN64、8 warp、num_stages=1。
两组使用相同推理源码，仅 kv_cpu_offload 开关不同；保留 restore→waiting prefill→running decode 的优先级。
每组3次独立新进程，交替运行，每项取各轮中位数。KV 容量均固定为1318块。

| 指标 | 关闭 | 开启 | 变化 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 27160.71 | 25476.87 | -6.20% |
| TTFT P50 ms | 32913.85 | 18081.73 | -45.06% |
| TTFT P95 ms | 70829.99 | 65542.56 | -7.46% |
| TTFT P99 ms | 74972.07 | 70851.76 | -5.50% |
| TPOT MEAN ms | 51.39 | 51.51 | +0.23% |
| TPOT P50 ms | 50.25 | 47.94 | -4.60% |
| TPOT P95 ms | 74.14 | 82.34 | +11.06% |
| TPOT P99 ms | 147.22 | 183.80 | +24.85% |
| ITL MEAN ms | 48.78 | 47.68 | -2.26% |
| ITL P50 ms | 37.18 | 37.97 | +2.12% |
| ITL P95 ms | 72.22 | 72.03 | -0.26% |
| ITL P99 ms | 98.19 | 98.02 | -0.17% |
| End-to-end latency MEAN ms | 54921.59 | 52613.54 | -4.20% |
| End-to-end latency P50 ms | 54752.74 | 52497.80 | -4.12% |
| End-to-end latency P95 ms | 87693.08 | 83531.47 | -4.75% |
| End-to-end latency P99 ms | 89858.91 | 85666.27 | -4.67% |
| elapsed_s | 90.681 | 86.541 | -4.57% |
| output_tokens_per_s | 6437.980 | 6745.970 | +4.78% |
| requests_per_s | 11.290 | 11.830 | +4.78% |
| input_tokens_per_s | 6403.360 | 6709.700 | +4.78% |
| total_tokens_per_s | 12841.340 | 13455.670 | +4.78% |
| preemptions | 657 | 728 | +10.81% |
| blocks | 1318 | 1318 | +0.00% |
| peak_allocated_gib | 20.730 | 20.760 | +0.14% |
| peak_reserved_gib | 21.030 | 21.060 | +0.14% |
| prefill model-run seconds | 21.629 | 14.298 | -33.89% |
| prefill model-run tokens | 879176 | 586886 | -33.25% |
| prefill model-run tokens_per_s | 40648.830 | 40922.800 | +0.67% |
| decode model-run seconds | 67.852 | 68.636 | +1.16% |
| decode model-run tokens | 582121 | 582770 | +0.11% |
| decode model-run tokens_per_s | 8579.290 | 8490.770 | -1.03% |
| preempted_requests | 302 | 299 | -0.99% |
| repeatedly_preempted_requests | 122 | 160 | +31.15% |
| maximum_per_request | 16 | 15 | -6.25% |
| actual decode batch steps | 2387.00 | 2392.00 | +0.21% |
| actual decode batch mean | 243.87 | 243.63 | -0.10% |
| actual decode batch p50 | 302.00 | 302.00 | +0.00% |
| actual decode batch p95 | 416.00 | 415.45 | -0.13% |
| actual decode batch maximum | 489.00 | 489.00 | +0.00% |

## 每轮结果

| 开关 | 轮次 | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 |
| --- | ---: | ---: | ---: | ---: | ---: |
| False | 1 | 6466.24 | 32633.72 | 49.91 | 657 |
| True | 1 | 6803.62 | 18081.73 | 47.52 | 728 |
| True | 2 | 6728.63 | 18405.39 | 48.19 | 716 |
| False | 2 | 6437.98 | 32913.85 | 50.25 | 657 |
| False | 3 | 6428.72 | 33022.77 | 50.40 | 657 |
| True | 3 | 6745.97 | 16035.92 | 47.94 | 758 |

## 卸载统计（三轮中位数）

```json
{
  "active_handles": 0,
  "cpu_budget_bytes": 4289331200,
  "cpu_pool_bytes": 4273733632,
  "fallback_preemptions": 8,
  "gpu_reused_blocks": 454,
  "idle_copy_wait_ms": 0,
  "idle_copy_waits": 0,
  "offloaded_sequences": 720,
  "peak_cpu_buffers": 274,
  "peak_restoring_sequences": 6,
  "restored_sequences": 720,
  "restoring": 0,
  "shared_snapshot_references": 0,
  "transport": {
    "d2h_blocks": 2092,
    "d2h_bytes": 32630112256,
    "d2h_completed_events": 2092,
    "d2h_enqueue_ms": 337.4454132281244,
    "d2h_first_phase_ms": 620.9243839681149,
    "d2h_second_phase_ms": 2615.1856323480606,
    "d2h_stream_ms": 3265.250912427902,
    "h2d_blocks": 1648,
    "h2d_bytes": 25704792064,
    "h2d_completed_events": 1648,
    "h2d_enqueue_ms": 243.68826998397708,
    "h2d_first_phase_ms": 2212.757183790207,
    "h2d_second_phase_ms": 528.9725756794214,
    "h2d_stream_ms": 2736.331809401512,
    "write_lock_waits": 417
  }
}
```

## 测量范围

- RTX 3090 Ti / Qwen3-0.6B / GPU0，权重BF16，双scale INT8 KV，decode CUDA Graph。
- 复用既有完整1024请求负载：输入580663、输出583802 token，长度100–1024、seed0、temperature0.6、ignore_eos=True。
- max_num_seqs512，max_num_batched_tokens16384，max_model_len4096，block256。
- 原仓库 attention 未改；实际测量副本使用先前已校验的快速8warp attention，两组相同。
- pinned CPU 缓冲池上限4GiB，GPU staging 两块共29.75MiB；相同KV容量对比避免容量变化混入策略收益。
- CPU缓冲不足时回退原重新prefill路径，fallback_preemptions单独记录；共享前缀仅在最后GPU引用释放时卸载。
- 初始化、编译与代表性普通/缓存prefill预热在计时外；CPU池第一次分配在实际运行内，池复用，未锁频率。
- TTFT含排队，CPU postprocess首token时间；TPOT每请求首末token差/后续token数，ITL汇总相邻间隔。
- model-run包含准备、模型、采样和同步，拷贝可与计算重叠，阶段总和不等于总墙钟。
- CUDA event统计为各次传输stream时间之和，不能将其与generate时间简单相减来推导被隐藏的时间。
- 随机采样会随调度次序变化；各请求输入与输出长度完全一致，模型正确性另用固定argmax采样和强制压力验证。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_async_kv_offload_metrics.py --outdir /tmp/async-offload-recheck --runs 3 --kv-blocks 1318
```

## 正确性与结论

实现、17 个测试方法、整模型 KV 逐元素检查、输出差异和 profiler 重叠的完整记录见 [验证报告](../validation.md)。

**吞吐提高 4.78%，收益主要对应减少重新 prefill；抢占增加 10.81%，TPOT P95/P99 变差，decode 阶段耗时增加 1.16%。** 开启三轮 CPU 池不足回退分别为 8、8、7 次。整模型输出未达到逐 token 一致，本次未做正式质量评测。
