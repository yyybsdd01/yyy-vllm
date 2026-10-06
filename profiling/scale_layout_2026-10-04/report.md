# scale 布局对照：[block, token, head, 2] 与 [block, head, token, 2]

已新增两个独立 Triton kernel 和对应 Python 调用接口，放在 `nanovllm/layers/head_major_scale_attention.py`。原写入、原 attention 和 model runner 未修改，SHA256 前后校验一致。新实现仅更改 scale 物理地址，INT8 K/V 仍按 `[block, token, head, dim]` 保存。

主测例 decode B=256、K=1024，新布局确实减少了读取事务和 DRAM 读取量，但 attention 耗时增加约 2.15%。不同形状有小幅改善或回退，本次没有观察到稳定的速度收益。

## 新增接口

- `store_kvcache_int8_head_major_kernel` / `store_kvcache_int8_head_major`：写入 INT8 K/V 和 `[blocks, KV heads, block size, 2]` FP32 scales；保持原来的一个 program 写一个 token 的线程组织。
- `_int8_paged_attention_head_major_kernel` / `int8_paged_attention_head_major`：读取上述布局，支持 decode 和 cached prefill；BM=16、BN=64、BD=128、4 warps 与原实现一致。
- Python 接口要求新 scale 张量物理连续，当前提供双 scale。两个 scale 前后半维的量化算法和 `tl.split`/广播/解量化/矩阵乘全部保留。attention AST 校验确认与当前 kernel 仅函数名、scale_offset 表达式不同。

```python
from nanovllm.layers.head_major_scale_attention import (
    store_kvcache_int8_head_major, int8_paged_attention_head_major,
)
k_scale = torch.empty((num_blocks, kv_heads, block_size, 2),
                      device='cuda', dtype=torch.float32)
v_scale = torch.empty_like(k_scale)
store_kvcache_int8_head_major(key, value, k_cache, v_cache,
                            k_scale, v_scale, slot_mapping)
out = int8_paged_attention_head_major(q, k_cache, v_cache, k_scale, v_scale,
                                      block_tables, softmax_scale,
                                      context_lens=context_lens)
```

## 验证和计时范围

RTX 3090 Ti，torch 2.5.1+cu121，Triton 3.1.0，Q heads=16、KV heads=8、D=128、block_size=256、双独立 FP32 scales，seed=47。

新增测试覆盖 D=80/128、非 2 次幂 head 数（3 KV/6 Q heads）、非连续物理页、跨页、异质长度、部分 Q tile、slot=-1、全零 half 的 scale、context_len=0，decode 和 cached prefill；两种布局的量化 K/V、scale 转置及 attention 输出逐元素完全一致，并与恢复后 BF16 FlashAttention 通过 rtol=0.02、atol=0.005。八个基准形状也全部通过新旧逐元素一致校验。

每种布局都真实调用各自的新旧写入 kernel，再调用匹配的 attention kernel。CUDA Graph 每图 8 次操作、每轮回放 10 次、11 轮随机交错顺序；表中为轮均值的中位数。数据生成、布局转置、CPU/Python 和模型其他层不计入。组合时间是两次 GPU launch 的独立实测，不是两项中位数相加；缓存和运行波动可能让它小于两项独立测量之和，约 1% 的差异不应外推成稳定收益。未锁定 GPU 频率。

## attention GPU 时间

| 阶段 | B | K | Q/序列 | 原 token/head µs | 新 head/token µs | 耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 1024 | 1 | 43.546 | 42.662 | -2.03% |
| decode | 64 | 512 | 1 | 192.179 | 192.525 | +0.18% |
| decode | 64 | 1024 | 1 | 383.270 | 380.698 | -0.67% |
| decode | 256 | 512 | 1 | 637.146 | 643.443 | +0.99% |
| decode | 256 | 1024 | 1 | 1236.531 | 1263.078 | +2.15% |
| decode | 64 | 4096 | 1 | 1492.403 | 1496.550 | +0.28% |
| prefill | 1 | 1024 | 128 | 83.482 | 80.960 | -3.02% |
| prefill | 8 | 1024 | 128 | 536.269 | 535.757 | -0.10% |

## 写入与写入后 attention

| 阶段 | B | K | 本轮写 token 数 | 原写入 µs | 新写入 µs | 原组合 µs | 新组合 µs | 组合耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 1024 | 1 | 2.150 | 2.163 | 45.107 | 44.262 | -1.87% |
| decode | 64 | 512 | 64 | 2.112 | 2.138 | 191.168 | 194.765 | +1.88% |
| decode | 64 | 1024 | 64 | 2.125 | 2.138 | 383.091 | 382.771 | -0.08% |
| decode | 256 | 512 | 256 | 2.714 | 2.739 | 639.526 | 647.910 | +1.31% |
| decode | 256 | 1024 | 256 | 2.714 | 2.726 | 1238.400 | 1263.744 | +2.05% |
| decode | 64 | 4096 | 64 | 2.138 | 2.150 | 1492.582 | 1499.162 | +0.44% |
| prefill | 1 | 1024 | 128 | 2.368 | 2.394 | 85.350 | 82.816 | -2.97% |
| prefill | 8 | 1024 | 1024 | 5.235 | 5.606 | 546.432 | 543.283 | -0.58% |

cached prefill 含 896 token 前缀、本轮新增 128 token。写入只写本轮 token，不重写整个历史 KV。

## 合并访存的硬件证据：B=256、K=1024 decode

NCU 2024.1.1，已预热的四个 kernel 各一个直接 launch、17 passes。cache-control=none、clock-control=none。计数器 duration 与 CUDA Graph 中位数分别记录；尤其几微秒的写入 duration 不取代上述多轮计时。L1TEX global load/store sectors 按 32 B sector 计，覆盖整个 kernel。

| attention 指标 | 原 token/head | 新 head/token |
| --- | ---: | ---: |
| NCU duration µs | 1221.056 | 1289.856 |
| DRAM read MiB | 641.060 | 545.754 |
| DRAM write MiB | 3.206 | 3.187 |
| L1TEX global load sectors million | 21.209 | 18.063 |
| L1TEX global load requests million | 1.518 | 1.518 |
| DRAM 平均读写忙碌时间 µs | 687.108 | 585.444 |
| 实际 DRAM 吞吐 GB/s | 553.260 | 446.256 |
| warp 指令 million | 97.149 | 97.157 |
| long scoreboard / issue | 2.188 | 2.656 |
| 寄存器/线程 | 191.000 | 191.000 |
| 动态共享内存 KiB | 53.000 | 53.000 |
| occupancy % | 8.322 | 8.387 |

写入端：global store requests 都为 3,072，global store sectors 从 **18,432 → 24,576（+33.33%）**；包含 K/V 与 scale 全部写入。这证明写入合并程度变差，与相邻 heads 的 scale 间距 8 B → 2048 B 一致。寄存器都是 33、动态共享内存都是 32 B。实测写入 B=256 只增加约 0.013 µs，B=8 cached prefill（写 1024 token）增加约 0.371 µs（7.09%）。

attention load requests 数量相同，global load sectors 减少约 14.83%，DRAM read 减少约 14.87%（约 95.31 MiB）。当前编译的 `[64,2]` scale load 在线程布局上仍为每 warp 16 token × 2 scale，跨 token 的实际地址从 64 B 间距改为 8 B 间距；从地址分析看，每个完整 warp 的 scale 请求从约 16 个 sector 降为 4 个。由于 K/V loads 仍占多数，整个 kernel 的 sectors 只下降约 15%。不同 heads 的缓存复用会影响实际 DRAM 事务量，不能直接把每条 scale 请求的比例套用到全 kernel。

## 结果的含义

读端合并访存得到改善，但没有改变 `[BN,2] → tl.split → 广播` 的布局转换、解量化或 tile 算法；两版都用 191 registers/thread、53 KiB 动态共享内存、一个 CTA/SM，occupancy 约 8.3%。总 warp 指令几乎相同，long scoreboard/issue 在本次采集中却从 2.188 增至 2.656。当前回退表现为加载依赖等待增加和有效吞吐下降，具体缓存局部性/指令调度原因尚未通过消融隔离，不能归因于新增解量化算术。

本实验只实现布局变更，不重写 writer 的线程组织，也不新增流水线/异步预取优化。改善 transactions 不保证降低关键路径延迟。两个新 kernel 保持独立，当前推理路径继续使用原版本。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python -m unittest discover -s tests -p test_head_major_scale_attention.py -v
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_scale_layouts.py --outdir /tmp/scale_layout_recheck
```

`--quick` 只测主形状；`--profile` 只圈住四个直接 launch，计数器命令见 `ncu_command.txt`。完整轮次/资源：`timings.json`；汇总：`comparison.csv`；硬件原始报告：`ncu_layouts.ncu-rep`；`ncu_summary.json` 和 `long_scoreboard_sources.json`。源码及测试快照在 `source_snapshot/`。
