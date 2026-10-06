# INT8 双 scale 与 BF16：DRAM 忙碌时间补充测量

本轮直接采集 `dram__cycles_active_read.avg`、`dram__cycles_active_write.avg`、`dram__cycles_active.avg` 和 `dram__cycles_elapsed.avg`。B=256、K=1024 decode；相同输入、源码、生产 packed 双 scale 及前次 BF16 对照。GPU 无其他计算进程；每版预热 5 次后只采集一次 launch，1 pass/版，不锁频、不清缓存。

**显存处理读取的平均忙碌时间减少了：同结构 BF16 约 1.093 ms，INT8+双 scale 约 0.684 ms，减少 37.46%。** 该值按各 DRAM 实例的 active cycles 求平均后换算；表示显存实际忙着处理数据的时间，不等同于所有访存指令在 kernel 关键路径上的耗时，也不等于 CPU 等待时间。

| 指标 | 原 Flash BF16 | 同结构 Triton BF16 | 当前 INT8 双 scale |
| --- | ---: | ---: | ---: |
| DRAM 读取 MiB | 1025.093994 | 1025.062744 | 641.086426 |
| DRAM 平均读取忙碌时间 ms | 1.093266 | 1.093244 | 0.683723 |
| DRAM 平均写入忙碌时间 ms | 0.003077 | 0.003011 | 0.002948 |
| DRAM 平均读写忙碌时间 ms | 1.096343 | 1.096256 | 0.686672 |
| DRAM 平均忙碌占比 % | 94.768503 | 95.548590 | 52.908158 |
| 此次单次 kernel 时间 ms | 1.156864 | 1.147328 | 1.297856 |

换算公式：

```text
DRAM 平均读取忙碌时间 = kernel duration
                       × dram__cycles_active_read.avg
                       ÷ dram__cycles_elapsed.avg
```

NCU 本机官方 Profiling Guide 的 Cycle Metrics 定义：`cycles_active` 是单元处理数据的周期数，`cycles_elapsed` 是采集区间内该单元时钟域的总周期数；avg 在单元实例间平均。文档路径：`/usr/local/cuda-12.4/nsight-compute-2024.1.0/docs/ProfilingGuide/index.html#cycle-metrics`。也用本机 `ncu --query-metrics --chips ga102` 验证了 DRAM read/write active 计数器定义。

访存包括地址生成、页表和 scale 加载、缓存层级、等待数据和实际 DRAM 传输，不能用一个 busy 指标覆盖全部环节。融合内核中计算与访存重叠，而且不同 DRAM 分区并行，因此不能把平均 busy 时间直接从 kernel wall time 中扣除，得到解量化耗时。

当前 INT8 的 DRAM 平均 busy 占比约 52.91%，同结构 BF16 约 95.55%；INT8 有更长的显存空闲间隔，读取量和忙碌时间减少，同时内核总耗时增加。前次多轮 CUDA Graph 中位数仍为 BF16 对照 1141.089 µs、INT8 1245.115 µs，本次单 launch 的 INT8 时间 1297.856 µs 受运行波动影响，不替代多轮性能结果。

long-scoreboard 原始累计 warp-cycle：原 FlashAttention 79,686,000，同结构 BF16 1,034,448,156，INT8 202,345,773。它表示驻留 warps 等待 L1TEX scoreboard 依赖的累计统计，与驻留 warp 数、异步加载和编译调度相关，不能当作 kernel 的独占访存时间；不同实现之间该统计差异很大。

原始报告：`ncu_memory_cycles.ncu-rep`；计数器与换算值：`memory_time.json`；采集命令：`ncu_memory_command.txt`。
