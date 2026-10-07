# 当前项目与 BF16：五档请求负载

快照：2026-10-06T11:25:11.763476+00:00；内核选择：current。各档两组独立新进程交替运行，各三轮，表内为逐项中位数。
当前项目启用双 scale INT8 KV、异步 CPU 卸载及 decode 块预留；BF16 使用 FlashAttention、关闭卸载。模型权重均为 BF16。

## 汇总

| 请求数 | 输出 token/s BF16 → 项目 | 变化 | TTFT P50 ms BF16 → 项目 | TPOT P50 ms BF16 → 项目 | 抢占 BF16 → 项目 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 5101.07 → 5393.74 | +5.74% | 1217.29 → 1212.90 | 31.98 → 29.95 | 84 → 0 |
| 512 | 4957.17 → 5400.04 | +8.93% | 2186.90 → 2180.55 | 34.88 → 58.86 | 322 → 188 |
| 1024 | 4948.65 → 5347.86 | +8.07% | 48130.02 → 38128.59 | 34.78 → 59.80 | 769 → 455 |
| 2048 | 4980.22 → 5428.80 | +9.01% | 102963.68 → 86490.78 | 35.14 → 60.44 | 1566 → 900 |
| 4096 | 4887.52 → 5418.01 | +10.85% | 221018.11 → 188884.95 | 35.57 → 60.09 | 3271 → 1895 |

## 256 请求

输入 142827、输出 133966 token；已完成 3 / 3 轮。

| 指标 | BF16 | 当前项目 | 相对变化 |
| --- | ---: | ---: | ---: |
| 输出 token/s | 5101.070 | 5393.740 | +5.74% |
| 请求/s | 9.750 | 10.310 | +5.74% |
| 输入 token/s | 5438.480 | 5750.500 | +5.74% |
| 总 token/s | 10539.550 | 11144.230 | +5.74% |
| 整批耗时 s | 26.262 | 24.837 | -5.43% |
| KV blocks | 701.000 | 1318.000 | +88.02% |
| 峰值 allocated GiB | 20.750 | 20.760 | +0.05% |
| 峰值 reserved GiB | 21.050 | 21.060 | +0.05% |
| KV tensor bytes | 20581449728.000 | 19348324352.000 | -5.99% |
| scale tensor bytes | 0.000 | 1209270272.000 | — |
| 抢占事件 | 84.000 | 0.000 | -100.00% |
| TTFT MEAN ms | 1178.450 | 1175.070 | -0.29% |
| TTFT P50 ms | 1217.290 | 1212.900 | -0.36% |
| TTFT P95 ms | 2154.870 | 2151.300 | -0.17% |
| TTFT P99 ms | 2154.870 | 2151.300 | -0.17% |
| TPOT MEAN ms | 32.430 | 30.140 | -7.06% |
| TPOT P50 ms | 31.980 | 29.950 | -6.35% |
| TPOT P95 ms | 42.240 | 37.860 | -10.37% |
| TPOT P99 ms | 57.820 | 42.520 | -26.46% |
| ITL MEAN ms | 29.850 | 28.130 | -5.76% |
| ITL P50 ms | 26.770 | 28.100 | +4.97% |
| ITL P95 ms | 54.210 | 30.010 | -44.64% |
| ITL P99 ms | 55.710 | 30.230 | -45.74% |
| End-to-end latency MEAN ms | 16769.790 | 15865.800 | -5.39% |
| End-to-end latency P50 ms | 17614.730 | 16410.320 | -6.84% |
| End-to-end latency P95 ms | 25049.350 | 24577.780 | -1.88% |
| End-to-end latency P99 ms | 25363.750 | 24756.600 | -2.39% |
| prefill model-run tokens | 181836.000 | 142827.000 | -21.45% |
| prefill model-run seconds | 3.760 | 2.134 | -43.24% |
| prefill model-run tokens_per_s | 48361.600 | 66940.390 | +38.42% |
| decode model-run tokens | 133626.000 | 133710.000 | +0.06% |
| decode model-run seconds | 22.174 | 22.389 | +0.97% |
| decode model-run tokens_per_s | 6026.130 | 5972.250 | -0.89% |
| request_preemption_stats.preempted_requests | 27.000 | 0.000 | -100.00% |
| request_preemption_stats.repeatedly_preempted_requests | 16.000 | 0.000 | -100.00% |
| request_preemption_stats.maximum_per_request | 11.000 | 0.000 | -100.00% |
| decode_batch_stats.mean | 100.850 | 130.704 | +29.60% |
| decode_batch_stats.p50 | 90.000 | 121.000 | +34.44% |
| decode_batch_stats.p95 | 238.000 | 256.000 | +7.56% |
| decode_batch_stats.maximum | 256.000 | 256.000 | +0.00% |
| decode_batch_stats.steps | 1325.000 | 1023.000 | -22.79% |
| restore_progress.restore_events | 0.000 | 0.000 | — |
| restore_progress.preemptions_without_token_progress | 0.000 | 0.000 | — |
| restore_progress.same_schedule_preemptions | 0.000 | 0.000 | — |

项目卸载统计（三轮中位数）：

| 指标 | 值 |
| --- | ---: |
| fallback_preemptions | 0.000 |
| shared_snapshot_references | 0.000 |
| offloaded_sequences | 0.000 |
| restored_sequences | 0.000 |
| gpu_reused_blocks | 0.000 |
| peak_cpu_buffers | 0.000 |
| peak_restoring_sequences | 0.000 |
| idle_copy_waits | 0.000 |
| idle_copy_wait_ms | 0.000 |
| cpu_budget_bytes | 4289331200.000 |
| cpu_pool_bytes | 0.000 |
| active_handles | 0.000 |
| restoring | 0.000 |

## 512 请求

输入 291939、输出 284413 token；已完成 3 / 3 轮。

| 指标 | BF16 | 当前项目 | 相对变化 |
| --- | ---: | ---: | ---: |
| 输出 token/s | 4957.170 | 5400.040 | +8.93% |
| 请求/s | 8.920 | 9.720 | +8.97% |
| 输入 token/s | 5088.340 | 5542.930 | +8.93% |
| 总 token/s | 10045.510 | 10942.970 | +8.93% |
| 整批耗时 s | 57.374 | 52.669 | -8.20% |
| KV blocks | 701.000 | 1318.000 | +88.02% |
| 峰值 allocated GiB | 20.750 | 20.760 | +0.05% |
| 峰值 reserved GiB | 21.050 | 21.060 | +0.05% |
| KV tensor bytes | 20581449728.000 | 19348324352.000 | -5.99% |
| scale tensor bytes | 0.000 | 1209270272.000 | — |
| 抢占事件 | 322.000 | 188.000 | -41.61% |
| TTFT MEAN ms | 15145.010 | 2712.670 | -82.09% |
| TTFT P50 ms | 2186.900 | 2180.550 | -0.29% |
| TTFT P95 ms | 39703.920 | 4230.790 | -89.34% |
| TTFT P99 ms | 43678.670 | 17995.350 | -58.80% |
| TPOT MEAN ms | 35.090 | 64.880 | +84.90% |
| TPOT P50 ms | 34.880 | 58.860 | +68.75% |
| TPOT P95 ms | 46.010 | 90.650 | +97.02% |
| TPOT P99 ms | 73.600 | 209.840 | +185.11% |
| ITL MEAN ms | 33.510 | 58.340 | +74.10% |
| ITL P50 ms | 27.290 | 50.930 | +86.63% |
| ITL P95 ms | 55.360 | 73.240 | +32.30% |
| ITL P99 ms | 68.590 | 82.070 | +19.65% |
| End-to-end latency MEAN ms | 33724.730 | 35064.490 | +3.97% |
| End-to-end latency P50 ms | 33978.990 | 37298.070 | +9.77% |
| End-to-end latency P95 ms | 55559.950 | 50923.380 | -8.35% |
| End-to-end latency P99 ms | 56497.840 | 52470.850 | -7.13% |
| prefill model-run tokens | 426682.000 | 291939.000 | -31.58% |
| prefill model-run seconds | 10.770 | 4.814 | -55.30% |
| prefill model-run tokens_per_s | 39617.930 | 60641.290 | +53.07% |
| decode model-run tokens | 283579.000 | 283901.000 | +0.11% |
| decode model-run seconds | 45.873 | 45.617 | -0.56% |
| decode model-run tokens_per_s | 6181.770 | 6223.550 | +0.68% |
| request_preemption_stats.preempted_requests | 152.000 | 92.000 | -39.47% |
| request_preemption_stats.repeatedly_preempted_requests | 66.000 | 49.000 | -25.76% |
| request_preemption_stats.maximum_per_request | 10.000 | 7.000 | -30.00% |
| decode_batch_stats.mean | 130.742 | 202.497 | +54.88% |
| decode_batch_stats.p50 | 161.000 | 202.500 | +25.78% |
| decode_batch_stats.p95 | 228.000 | 442.000 | +93.86% |
| decode_batch_stats.maximum | 265.000 | 489.000 | +84.53% |
| decode_batch_stats.steps | 2169.000 | 1402.000 | -35.36% |
| restore_progress.restore_events | 0.000 | 188.000 | — |
| restore_progress.preemptions_without_token_progress | 0.000 | 35.000 | — |
| restore_progress.same_schedule_preemptions | 0.000 | 35.000 | — |

项目卸载统计（三轮中位数）：

| 指标 | 值 |
| --- | ---: |
| fallback_preemptions | 0.000 |
| shared_snapshot_references | 0.000 |
| offloaded_sequences | 188.000 |
| restored_sequences | 188.000 |
| gpu_reused_blocks | 88.000 |
| peak_cpu_buffers | 257.000 |
| peak_restoring_sequences | 5.000 |
| idle_copy_waits | 0.000 |
| idle_copy_wait_ms | 0.000 |
| cpu_budget_bytes | 4289331200.000 |
| cpu_pool_bytes | 4008574976.000 |
| active_handles | 0.000 |
| restoring | 0.000 |

## 1024 请求

输入 580663、输出 583802 token；已完成 3 / 3 轮。

| 指标 | BF16 | 当前项目 | 相对变化 |
| --- | ---: | ---: | ---: |
| 输出 token/s | 4948.650 | 5347.860 | +8.07% |
| 请求/s | 8.680 | 9.380 | +8.06% |
| 输入 token/s | 4922.040 | 5319.100 | +8.07% |
| 总 token/s | 9870.690 | 10666.960 | +8.07% |
| 整批耗时 s | 117.972 | 109.166 | -7.46% |
| KV blocks | 701.000 | 1318.000 | +88.02% |
| 峰值 allocated GiB | 20.750 | 20.760 | +0.05% |
| 峰值 reserved GiB | 21.050 | 21.060 | +0.05% |
| KV tensor bytes | 20581449728.000 | 19348324352.000 | -5.99% |
| scale tensor bytes | 0.000 | 1209270272.000 | — |
| 抢占事件 | 769.000 | 455.000 | -40.83% |
| TTFT MEAN ms | 46321.520 | 31183.550 | -32.68% |
| TTFT P50 ms | 48130.020 | 38128.590 | -20.78% |
| TTFT P95 ms | 101144.980 | 82866.300 | -18.07% |
| TTFT P99 ms | 104696.610 | 88109.160 | -15.84% |
| TPOT MEAN ms | 35.020 | 62.960 | +79.78% |
| TPOT P50 ms | 34.780 | 59.800 | +71.94% |
| TPOT P95 ms | 39.970 | 92.950 | +132.55% |
| TPOT P99 ms | 71.130 | 186.250 | +161.84% |
| ITL MEAN ms | 34.300 | 59.290 | +72.86% |
| ITL P50 ms | 27.310 | 51.570 | +88.83% |
| ITL P95 ms | 56.910 | 82.090 | +44.25% |
| ITL P99 ms | 69.040 | 106.010 | +53.55% |
| End-to-end latency MEAN ms | 65841.080 | 64924.750 | -1.39% |
| End-to-end latency P50 ms | 66170.340 | 64316.490 | -2.80% |
| End-to-end latency P95 ms | 114195.540 | 105521.030 | -7.60% |
| End-to-end latency P99 ms | 117258.590 | 108248.050 | -7.68% |
| prefill model-run tokens | 871974.000 | 580663.000 | -33.41% |
| prefill model-run seconds | 23.224 | 12.473 | -46.29% |
| prefill model-run tokens_per_s | 37546.000 | 46552.050 | +23.99% |
| decode model-run tokens | 582009.000 | 582778.000 | +0.13% |
| decode model-run seconds | 93.261 | 93.257 | -0.00% |
| decode model-run tokens_per_s | 6240.650 | 6249.190 | +0.14% |
| request_preemption_stats.preempted_requests | 427.000 | 220.000 | -48.48% |
| request_preemption_stats.repeatedly_preempted_requests | 178.000 | 111.000 | -37.64% |
| request_preemption_stats.maximum_per_request | 10.000 | 9.000 | -10.00% |
| decode_batch_stats.mean | 147.718 | 241.817 | +63.70% |
| decode_batch_stats.p50 | 167.000 | 301.000 | +80.24% |
| decode_batch_stats.p95 | 203.000 | 414.550 | +104.21% |
| decode_batch_stats.maximum | 265.000 | 489.000 | +84.53% |
| decode_batch_stats.steps | 3940.000 | 2410.000 | -38.83% |
| restore_progress.restore_events | 0.000 | 455.000 | — |
| restore_progress.preemptions_without_token_progress | 0.000 | 67.000 | — |
| restore_progress.same_schedule_preemptions | 0.000 | 67.000 | — |

项目卸载统计（三轮中位数）：

| 指标 | 值 |
| --- | ---: |
| fallback_preemptions | 0.000 |
| shared_snapshot_references | 0.000 |
| offloaded_sequences | 455.000 |
| restored_sequences | 455.000 |
| gpu_reused_blocks | 297.000 |
| peak_cpu_buffers | 246.000 |
| peak_restoring_sequences | 5.000 |
| idle_copy_waits | 0.000 |
| idle_copy_wait_ms | 0.000 |
| cpu_budget_bytes | 4289331200.000 |
| cpu_pool_bytes | 3837001728.000 |
| active_handles | 0.000 |
| restoring | 0.000 |

## 2048 请求

输入 1150273、输出 1128906 token；已完成 3 / 3 轮。

| 指标 | BF16 | 当前项目 | 相对变化 |
| --- | ---: | ---: | ---: |
| 输出 token/s | 4980.220 | 5428.800 | +9.01% |
| 请求/s | 9.030 | 9.850 | +9.08% |
| 输入 token/s | 5074.480 | 5531.560 | +9.01% |
| 总 token/s | 10054.700 | 10960.360 | +9.01% |
| 整批耗时 s | 226.678 | 207.947 | -8.26% |
| KV blocks | 701.000 | 1318.000 | +88.02% |
| 峰值 allocated GiB | 20.750 | 20.760 | +0.05% |
| 峰值 reserved GiB | 21.050 | 21.060 | +0.05% |
| KV tensor bytes | 20581449728.000 | 19348324352.000 | -5.99% |
| scale tensor bytes | 0.000 | 1209270272.000 | — |
| 抢占事件 | 1566.000 | 900.000 | -42.53% |
| TTFT MEAN ms | 102081.990 | 83134.260 | -18.56% |
| TTFT P50 ms | 102963.680 | 86490.780 | -16.00% |
| TTFT P95 ms | 202223.590 | 175117.050 | -13.40% |
| TTFT P99 ms | 210774.880 | 184270.850 | -12.57% |
| TPOT MEAN ms | 35.140 | 60.990 | +73.56% |
| TPOT P50 ms | 35.140 | 60.440 | +72.00% |
| TPOT P95 ms | 37.640 | 72.590 | +92.85% |
| TPOT P99 ms | 52.490 | 112.280 | +113.91% |
| ITL MEAN ms | 34.780 | 59.590 | +71.33% |
| ITL P50 ms | 27.290 | 51.420 | +88.42% |
| ITL P95 ms | 57.380 | 82.320 | +43.46% |
| ITL P99 ms | 69.250 | 101.220 | +46.17% |
| End-to-end latency MEAN ms | 121218.090 | 115920.580 | -4.37% |
| End-to-end latency P50 ms | 121917.570 | 116677.600 | -4.30% |
| End-to-end latency P95 ms | 220055.460 | 202655.480 | -7.91% |
| End-to-end latency P99 ms | 225092.840 | 206688.540 | -8.18% |
| prefill model-run tokens | 1758555.000 | 1150273.000 | -34.59% |
| prefill model-run seconds | 48.272 | 27.785 | -42.44% |
| prefill model-run tokens_per_s | 36430.310 | 41399.660 | +13.64% |
| decode model-run tokens | 1125292.000 | 1126858.000 | +0.14% |
| decode model-run seconds | 175.536 | 174.569 | -0.55% |
| decode model-run tokens_per_s | 6410.620 | 6455.090 | +0.69% |
| request_preemption_stats.preempted_requests | 923.000 | 527.000 | -42.90% |
| request_preemption_stats.repeatedly_preempted_requests | 360.000 | 220.000 | -38.89% |
| request_preemption_stats.maximum_per_request | 9.000 | 8.000 | -11.11% |
| decode_batch_stats.mean | 161.078 | 285.064 | +76.97% |
| decode_batch_stats.p50 | 173.000 | 322.000 | +86.13% |
| decode_batch_stats.p95 | 190.000 | 385.000 | +102.63% |
| decode_batch_stats.maximum | 265.000 | 489.000 | +84.53% |
| decode_batch_stats.steps | 6986.000 | 3953.000 | -43.42% |
| restore_progress.restore_events | 0.000 | 900.000 | — |
| restore_progress.preemptions_without_token_progress | 0.000 | 124.000 | — |
| restore_progress.same_schedule_preemptions | 0.000 | 124.000 | — |

项目卸载统计（三轮中位数）：

| 指标 | 值 |
| --- | ---: |
| fallback_preemptions | 0.000 |
| shared_snapshot_references | 0.000 |
| offloaded_sequences | 900.000 |
| restored_sequences | 900.000 |
| gpu_reused_blocks | 521.000 |
| peak_cpu_buffers | 262.000 |
| peak_restoring_sequences | 7.000 |
| idle_copy_waits | 0.000 |
| idle_copy_wait_ms | 0.000 |
| cpu_budget_bytes | 4289331200.000 |
| cpu_pool_bytes | 4086562816.000 |
| active_handles | 0.000 |
| restoring | 0.000 |

## 4096 请求

输入 2303424、输出 2307852 token；已完成 3 / 3 轮。

| 指标 | BF16 | 当前项目 | 相对变化 |
| --- | ---: | ---: | ---: |
| 输出 token/s | 4887.520 | 5418.010 | +10.85% |
| 请求/s | 8.670 | 9.620 | +10.96% |
| 输入 token/s | 4878.150 | 5407.620 | +10.85% |
| 总 token/s | 9765.670 | 10825.630 | +10.85% |
| 整批耗时 s | 472.192 | 425.959 | -9.79% |
| KV blocks | 701.000 | 1318.000 | +88.02% |
| 峰值 allocated GiB | 20.750 | 20.760 | +0.05% |
| 峰值 reserved GiB | 21.050 | 21.060 | +0.05% |
| KV tensor bytes | 20581449728.000 | 19348324352.000 | -5.99% |
| scale tensor bytes | 0.000 | 1209270272.000 | — |
| 抢占事件 | 3271.000 | 1895.000 | -42.07% |
| TTFT MEAN ms | 221245.020 | 189973.240 | -14.13% |
| TTFT P50 ms | 221018.110 | 188884.950 | -14.54% |
| TTFT P95 ms | 436203.380 | 385109.210 | -11.71% |
| TTFT P99 ms | 454053.990 | 400305.180 | -11.84% |
| TPOT MEAN ms | 35.550 | 60.340 | +69.73% |
| TPOT P50 ms | 35.570 | 60.090 | +68.93% |
| TPOT P95 ms | 37.350 | 64.500 | +72.69% |
| TPOT P99 ms | 41.650 | 90.530 | +117.36% |
| ITL MEAN ms | 35.320 | 59.500 | +68.46% |
| ITL P50 ms | 27.260 | 51.370 | +88.44% |
| ITL P95 ms | 61.240 | 82.650 | +34.96% |
| ITL P99 ms | 75.800 | 100.200 | +32.19% |
| End-to-end latency MEAN ms | 241108.500 | 223436.020 | -7.33% |
| End-to-end latency P50 ms | 241013.690 | 223841.810 | -7.12% |
| End-to-end latency P95 ms | 455922.240 | 412451.840 | -9.53% |
| End-to-end latency P99 ms | 468992.590 | 423319.400 | -9.74% |
| prefill model-run tokens | 3544801.000 | 2303424.000 | -35.02% |
| prefill model-run seconds | 107.553 | 59.384 | -44.79% |
| prefill model-run tokens_per_s | 32958.700 | 38788.560 | +17.69% |
| decode model-run tokens | 2300485.000 | 2303756.000 | +0.14% |
| decode model-run seconds | 358.781 | 356.903 | -0.52% |
| decode model-run tokens_per_s | 6411.940 | 6454.850 | +0.67% |
| request_preemption_stats.preempted_requests | 1966.000 | 1084.000 | -44.86% |
| request_preemption_stats.repeatedly_preempted_requests | 792.000 | 459.000 | -42.05% |
| request_preemption_stats.maximum_per_request | 8.000 | 9.000 | +12.50% |
| decode_batch_stats.mean | 165.586 | 301.105 | +81.84% |
| decode_batch_stats.p50 | 172.000 | 323.000 | +87.79% |
| decode_batch_stats.p95 | 183.000 | 350.000 | +91.26% |
| decode_batch_stats.maximum | 265.000 | 489.000 | +84.53% |
| decode_batch_stats.steps | 13893.000 | 7651.000 | -44.93% |
| restore_progress.restore_events | 0.000 | 1895.000 | — |
| restore_progress.preemptions_without_token_progress | 0.000 | 230.000 | — |
| restore_progress.same_schedule_preemptions | 0.000 | 230.000 | — |

项目卸载统计（三轮中位数）：

| 指标 | 值 |
| --- | ---: |
| fallback_preemptions | 0.000 |
| shared_snapshot_references | 0.000 |
| offloaded_sequences | 1895.000 |
| restored_sequences | 1895.000 |
| gpu_reused_blocks | 1134.000 |
| peak_cpu_buffers | 207.000 |
| peak_restoring_sequences | 7.000 |
| idle_copy_waits | 0.000 |
| idle_copy_wait_ms | 0.000 |
| cpu_budget_bytes | 4289331200.000 |
| cpu_pool_bytes | 3228696576.000 |
| active_handles | 0.000 |
| restoring | 0.000 |

## PPL 与传输正确性

WikiText-2 raw test，共 298938 个相同目标位置；4096 非重叠窗口、256 token 分块，全词表交叉熵。
BF16 PPL **18.810820**，项目 PPL **18.939454**，变化 **+0.6838%**。
PPL 测的是对应 attention 路径的分块 prefill teacher forcing，不随合成请求数变化；不覆盖生成采样或卸载调度的端到端质量。
卸载另在 12 块 GPU KV 的压力场景中逐块比较恢复前后有效 KV/scale，完整结果见 validation 文件。

## 口径与复现

- GPU0 RTX 3090 Ti，Qwen3-0.6B，max_num_seqs=512、max_num_batched_tokens=16384、max_model_len=4096、page=256、gpu_memory_utilization=0.9。
- 256/512/1024/2048/4096 为一次同时提交的请求数；实际 decode batch 不超过 512。各档输入和输出各 100–1024 token，seed=0、temperature=0.6、ignore_eos=True，两组逐请求长度完全相同。
- 根目录推理源码保持原样；独立快照测量。current 使用根目录 BM16/BN64、4 warps 的 attention，首次无前缀 prefill 走 BF16 FlashAttention；fast8 是显式选择的旧实验内核，所有 INT8 prefill 走量化路径。此次 manifest.kernel 决定实际版本。
- BF16 和项目使用相同显存预算，KV 块数按本版本原生容量分配，包含 INT8 容量优势和调度影响。CPU pinned 池限 4 GiB，最多 8 个 H2D 恢复请求。
- 每轮独立进程；模型加载、CUDA Graph 捕获、代表性普通/缓存 prefill 编译预热在计时外。预热请求 ID 与正式负载不同；正式采样前重置随机种子和显存峰值。CPU 池首次分配计入正式耗时。
- TTFT 从整批提交到 CPU postprocess 回填首 token，含排队；TPOT=(末 token 时间-首 token 时间)/(输出数-1)；ITL 汇总相邻 token 间隔；端到端为每请求完成时间减整批提交时间。
- model-run 时间包含输入准备、模型、采样和同步；输入吞吐按原始输入数计算，prefill tokens 包含抢占后的重算。显存为 PyTorch allocated/reserved 峰值，KV tensor bytes 仅存储张量。
- 抢占是事件次数，同一请求可多次被抢占；恢复后无新 token 再抢占单独计数。随机采样随 batch/调度变化，输出 hash 仅溯源，不要求不同路径生成 token 相同。
- 未锁 GPU 频率；仅 GPU0 串行测量，记录每轮 GPU 状态。每轮检查实际 import 路径、KV dtype、工作负载 hash、每请求输出长度/时间戳数量及 GPU/CPU KV 清理。

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_project_bf16_5loads.py --outdir /home/xgd/yyy/nano-vllm/profiling/current_project_vs_bf16_5loads_2026-10-06 --requests 256 512 1024 2048 4096 --runs 3 --kernel current
```

## 结果解读与传输统计

这组结果比较的是当前完整推理系统：INT8 KV 容量、attention 路径与 CPU 卸载调度共同影响性能，不能把吞吐差直接归因于单个 kernel。
显存预算相同，压缩后用于增加 KV 块数，所以整体 GPU 显存峰值不会按 KV 每 token 字节数同比下降。

| 请求数 | 吞吐变化 | TTFT P95 变化 | TPOT P50 变化 | E2E P50 变化 | 抢占变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | +5.74% | -0.17% | -6.35% | -6.84% | -100.00% |
| 512 | +8.93% | -89.34% | +68.75% | +9.77% | -41.61% |
| 1024 | +8.07% | -18.07% | +71.94% | -2.80% | -40.83% |
| 2048 | +9.01% | -13.40% | +72.00% | -4.30% | -42.53% |
| 4096 | +10.85% | -11.71% | +68.93% | -7.12% | -42.07% |

CPU 卸载传输统计（三轮中位数）：

| 请求数 | D2H blocks | H2D blocks | D2H GiB | H2D GiB | D2H stream ms | H2D stream ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 512 | 601 | 513 | 8.730 | 7.452 | 892.491 | 847.040 |
| 1024 | 1329 | 1032 | 19.306 | 14.991 | 1967.115 | 1672.312 |
| 2048 | 2415 | 1894 | 35.081 | 27.513 | 3543.554 | 3039.891 |
| 4096 | 5160 | 4042 | 74.956 | 58.716 | 7586.980 | 6432.184 |

stream 时间为各复制事件的 CUDA event 时长之和，复制可能与模型重叠，不能直接从整批耗时相减。

## Torch 编译缓存事件

本轮保留 PyTorch 2.5.1 的默认 Dynamo 编译缓存上限 8；未修改推理源码或提高该上限。
以下轮次在正式计时阶段出现 RMSNorm.rms_forward 达到缓存上限的提示；新形状的编译/回退耗时计入原生端到端结果。
decode CUDA Graph 在计时前已捕获，正式过程重放已保存的图；该提示的发生不能直接换算为整个模型均转为 eager。
这组结果用于描述当前系统行为；不能将 4096 档性能差全部归因于 INT8 attention 或卸载。

| 请求数 | 版本 | 轮次 | 原始日志 |
| ---: | --- | ---: | --- |
| 4096 | bf16 | 1 | 4096_bf16_1.txt |
| 4096 | bf16 | 2 | 4096_bf16_2.txt |
| 4096 | bf16 | 3 | 4096_bf16_3.txt |
