# 单、双 scale INT8 paged attention 耗时与瓶颈（2026-10-04）

当前代码的双 scale 主要增加了 scale 加载与消费依赖带来的访存等待。decode 的主测例中，单 scale **1.191 ms**，双 scale **1.458 ms**，增加 **22.44%**。两者的矩阵乘、INT8 转换数量相同；占用率都受相同的共享内存配置限制。保持双 scale 数学语义、只改变 scale 加载布局的独立副本降到 **1.239 ms**，支持优先检查 scale load 的指令组织和调度。

## 测量范围和控制条件

- RTX 3090 Ti，PyTorch 2.5.1+cu121，Triton 3.1.0；单卡，无模型权重加载。
- 当前生产文件 `nanovllm/layers/quantized_attention.py` 未修改。快照：`production_snapshot.py`；SHA256：`02a9a36c499ed979f21e8053a1d33260c1ba474a7dd6c5509f2c52ee64c80ca9`。
- Qwen3-0.6B head 布局：Q heads=16，KV heads=8，head_dim=128；BM=16、BN=64、BD=128，4 warps，block_size=256。
- 只测 `_int8_paged_attention_kernel`。输入分配、KV 写入、调度、其他层均不计入；预分配输出。
- CUDA Event 计时，每个 CUDA Graph 放 8 次内核，每轮回放 10 次；11 轮随机打乱模式顺序。报告每轮 80 次内核的平均耗时之中位数，P10/P90 是轮均值的分位数，不是单次调用尾延迟。
- seed=31，非零随机 INT8 K/V，BF16 Q，scale 在 [0.005,0.02) 随机分布。双 scale 把每个单 scale 重复两次，两种模式恢复出的 K/V 完全相同；所有计时副本与单 scale 输出逐元素一致。编译器不知道 scale 的数值相等，双 scale 的读取仍实际执行。
- 另用前后半维不同 scale 检查生产双 scale 和成对加载副本的 decode、cached prefill，对解量化 BF16 FlashAttention 参考通过 rtol=0.02、atol=0.005 校验。定位用 first-only/average 副本只在两个 scale 相等的数据上保持结果；不能用于实际双 scale 推理。
- prefill 的 Q=128 为本轮新增 token，K=1024 包括 896 token 的缓存前缀。此处是融合 cached prefill 路径。

## 当前生产内核的计时

| 阶段 | batch | KV 长度 | 新 Q token/序列 | 单 scale µs | 双 scale µs | 双 scale 增幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 1024 | 1 | 36.070 | 40.973 | 13.59% |
| decode | 64 | 512 | 1 | 168.704 | 203.763 | 20.78% |
| decode | 64 | 1024 | 1 | 341.120 | 402.112 | 17.88% |
| decode | 256 | 512 | 1 | 617.370 | 744.448 | 20.58% |
| decode | 256 | 1024 | 1 | 1191.091 | 1458.355 | 22.44% |
| decode | 64 | 4096 | 1 | 1370.099 | 1585.920 | 15.75% |
| prefill | 1 | 1024 | 128 | 78.554 | 94.515 | 20.32% |
| prefill | 8 | 1024 | 128 | 472.576 | 555.405 | 17.53% |

完整轮次、P10/P90、资源记录：`timings.json`。

## 定位用对照：decode batch=256、context=1024

| 版本 | 改动 | 中位数 µs | P10 / P90 µs | 寄存器/线程 | spills |
| --- | --- | ---: | ---: | ---: | ---: |
| `single` | 生产单 scale，每个 K/V 各读一个值 | 1191.091 | 1182.938 / 1196.288 | 147 | 0 |
| `dual` | 生产双 scale，分别读 first/second 后按维度选择 | 1458.355 | 1457.830 / 1458.611 | 146 | 0 |
| `dual_stride_first` | 双 scale 的数组/地址跨度，但每个 K/V 只读 first | 1248.371 | 1247.398 / 1251.379 | 128 | 0 |
| `dual_load_average` | 四次 scale load 保留；first/second 求平均后统一广播 | 1108.659 | 1107.994 / 1109.760 | 128 | 0 |
| `dual_packed` | 以 [BN,2] tile 加载相邻 scale，tl.split 后保留按维度选择 | 1239.462 | 1237.914 / 1240.691 | 191 | 0 |

成对加载副本比原双 scale 减少 15.01% 的调用耗时，去掉单、双 scale 差距的 81.90%。这是主测形状的微基准结果。batch=1 时它反而比原双 scale 慢；prefill batch=8 时仅改善约 3.45%，不能直接作为全负载优化结论。
两个定位副本 first-only/average 会改变不同 half scale 的数学语义。average 仍保留四次 scalar scale load，却快于原双 scale，说明指令数量与逻辑字节数不足以解释差距；scale 的消费者、寄存器分配和全局/异步加载调度同样重要。对照改变了编译生成的内核，几项耗时差不能作为可相加的阶段成本。

## Nsight Compute 证据

使用 Nsight Compute 2024.1.1，CUDA profiler API 只圈住已预热的各版本一次直接 launch，17 passes/版本。报告为 `ncu_scales.ncu-rep`，导出 `ncu_raw.csv`、`ncu_details.txt`、`ncu_source.txt`。普通用户受到 ERR_NVGPUCTRPERM 限制，使用管理员权限完成采集；没有改驱动配置。

计数器采集没有锁定频率，cache-control=none；其 duration 为 profiler 单次 launch 的测量，和上面的多轮 CUDA Graph 中位数分开。各指标用于定位方向，不用于把流水线中各环节的耗时相加。

| 指标 | 单 scale | 双 scale | 四次加载求平均 | 成对加载双 scale |
| --- | ---: | ---: | ---: | ---: |
| NCU duration，ms | 1.180 | 1.452 | 1.099 | 1.212 |
| DRAM throughput，GB/s | 572.59 | 465.59 | 619.75 | 557.14 |
| Achieved occupancy，% | 8.39 | 8.32 | 8.35 | 8.38 |
| 每条发射指令的 long scoreboard 等待，cycle | 2.964 | 4.532 | 2.074 | 2.198 |
| 每条执行指令的平均 warp cycle | 8.124 | 9.750 | 7.284 | 8.596 |
| 实际执行的 warp 指令数，million | 97.382 | 100.880 | 102.699 | 97.149 |
| DRAM 总事务量，MiB，rate×duration 推导 | 644.19 | 644.68 | 649.45 | 644.13 |

1. **主要增长是等待 global/local memory 数据的 long scoreboard。** 2.964 → 4.532 cycle/issued instruction；该原因占平均 warp cycle 的比例由约 36.48% 升到 46.48%。总指令数仅增加约 3.59%，矩阵乘相关指令未增加。
2. **scale 的逻辑容量翻倍没有对应为 DRAM 事务量翻倍。** 本例 INT8 K/V 合计 512 MiB，单 scale 的 K/V scales 合计 16 MiB，双 scale 合计 32 MiB，逻辑总量 528 → 544 MiB（+3.03%）。计数器 rate×duration 推导的实际 DRAM 事务总量约 644.19 → 644.68 MiB（+0.08%）；实际吞吐下降约 18.69%。这里的事务量包含读取过量及写出，与张量逻辑字节数不同。
3. **两者的并发都受共享内存限制。** Triton 动态 shared memory=54,272 B，NCU 还记录 1,024 B driver shared memory；GA102 每个 SM 配置 102,400 B，只能驻留一个 CTA=4 warps，理论 occupancy=8.33%。生产单/双 scale 寄存器是 147/146，decode spills 都是 0；没有双 scale 因寄存器溢出而降低 occupancy 的证据。驻留 warp 少，额外的 global load 依赖难以被其他 warp 隐藏。

## 定位到源码和机器指令

生产源码的 scale 读取位于 89–92 行，按前后半维选择位于 93–96 行，解量化位于 100–101 行：

```python
k_first = tl.load(ks_ptr + scale_offset, valid_k, other=0)
k_second = tl.load(ks_ptr + scale_offset + 1, valid_k, other=0)
v_first = tl.load(vs_ptr + scale_offset, valid_k, other=0)
v_second = tl.load(vs_ptr + scale_offset + 1, valid_k, other=0)
k_scale = tl.where(dims[None, :] < HEAD_DIM // 2,
                   k_first[:, None], k_second[:, None])
```

当前编译产物对这几次读取使用分开的 32-bit `LDG.E`，并未自动合成“每个线程一条 64-bit 成对加载”。单 scale 的 K/V scale load 静态指令为 8 条，双 scale 为 16 条；消费者跟在加载之后，形成 scoreboard 等待。成对加载副本让两个值作为二维 tile 分布到线程，再进行 tl.split/layout 转换，生成更少 global load 指令，同时增加共享内存转换和 barrier；它不是保证发出一条 64-bit vector load。

| 整个编译内核的静态 SASS 指令数 | 单 scale | 双 scale | 求平均 | 成对加载 |
| --- | ---: | ---: | ---: | ---: |
| `LDG.E` | 9 | 17 | 17 | 4 |
| `SEL` | 28 | 43 | 36 | 26 |
| `I2F.S8` | 128 | 128 | 128 | 128 |
| `FMUL` | 221 | 221 | 229 | 221 |
| `HMMA.16816.F32.BF16` | 32 | 32 | 32 | 32 |
| `SHFL.BFLY` | 10 | 10 | 10 | 10 |
| `BAR.SYNC.DEFER_BLOCKING` | 17 | 17 | 17 | 29 |

单 scale、原双 scale 和求平均版本的 `LDG.E` 行包含一个初始化 context_len load，减去这一条后分别是 8、16、16 条 scale load。成对加载版本还用 `LDG.E` 读取页表，不能套用这一减法。这里是静态指令数，不能当成每次实际执行的指令数，也不能与源码 tl.load 次数一一对应。每个程序在多个 KV tile 上重复这些指令。

双 scale 的 long scoreboard 共 111,750 个采样，其中 scale 源码 89–96 行合计 67,677（60.56%）。第 89 行单独有 62,547（55.97%），主要采在读到 scale 后的 `SEL` 消费者，而非 load 发射时。循环推进第 72 行有 38,527（34.48%），其邻近 SASS 存在异步 K/V load 的 `DEPBAR.LE`；这是访存依赖/调度的等待位置，不能据源码行号认定为索引乘法耗时。
这几个百分比是 **long scoreboard 等待采样内的分布**，不是总 kernel 时间百分比。融合内核有重排、预取与指令重叠，源码行注释也不等价于精确计时区间。原始 PC、SASS 和源码位置见 `long_scoreboard_sources.json`。

## 结论和后续方向

当前测例的双 scale 额外耗时，主要表现为 scale global load 消费依赖增加，低驻留 warp 数导致延迟难以隐藏，内存有效吞吐下降。额外的按维选择及编译指令调度参与其中，但不能根据这次对照单独宣称 `tl.where` 本身消耗固定多少微秒。

优先实验 scale 的线程布局、把前后两个 scale 的加载与 K/V 解量化结合、以及减少 shared memory 使更多 CTA 驻留。BM=16 而 decode 只有 2 个有效 Q head 行的冗余计算是两种模式共同的开销，不是双 scale 特有原因。成对加载在大 batch 的结果值得继续验证，但需要跨形状、异质长度、正确性和完整模型测量后才能判断是否应接入推理路径。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_attention_scales.py
```

仅复测主例可加 `--quick`，另存结果可加 `--outdir /tmp/attention_scales_recheck`。脚本复制当前生产源文件到输出目录生成实验副本；原始结果对应上面记录的 source SHA256。计数器报告的采集命令和采样来自本轮运行，不依赖 2026-10-02 的历史耗时。
