# 融合 INT8 KV Attention 对比（2026-10-02）

## 实现与条件

- `auto`：FP16/BF16 KV cache，FlashAttention。
- `int8_dequant`：逐 token、逐 KV head 量化为 INT8，读缓存时先写入 BF16 临时缓冲，再调用 FlashAttention。保留作为旧实现对照。
- `int8`：相同 INT8 KV 与 FP32 scale 布局；Triton 注意力内核按 `block_table` 直接读取、片上反量化并计算在线 softmax，无整块 BF16 KV 临时缓冲。普通首轮 prefill 仍使用原来的 FlashAttention；融合路径处理 decode 和带缓存前缀的 prefill。
- 环境：Qwen3-0.6B，RTX 3090 Ti，单卡，CUDA Graph 开启；256 个一次性提交的请求；固定随机种子 0；输入 142,827 token，实际输出 133,966 token；KV block size 256；预热和加载不计时。其余参数与[前次对比](kv_cache_int8_comparison_2026-10-02.md)相同。
- `auto` 和旧 INT8 列使用前次每种各三轮的逐指标中位数；融合 INT8 列使用本次前三轮的逐指标中位数。本次还复测了旧路径一轮和融合路径在最后一处 prefix 优化后的一轮，见原始日志。

## 容量与吞吐

| 指标 | auto | int8_dequant | int8 融合 |
| --- | ---: | ---: | ---: |
| 可分配 KV 块 | 701 | 1,272 | 1,360 |
| 256 请求墙钟时间 | 26.380 s | 48.216 s | 23.196 s |
| 请求吞吐 | 9.70 请求/s | 5.31 请求/s | 11.04 请求/s |
| 输入 token / 墙钟时间 | 5,414.21 token/s | 2,962.25 token/s | 6,157.40 token/s |
| 实际输出 token / 墙钟时间 | 5,078.31 token/s | 2,778.47 token/s | 5,775.39 token/s |
| 输入加输出 token / 墙钟时间 | 10,492.53 token/s | 5,740.72 token/s | 11,932.79 token/s |
| Prefill 模型执行吞吐 | 46,795.05 token/s | 63,315.21 token/s | 63,417.27 token/s |
| Decode 模型执行吞吐 | 6,011.15 token/s | 2,924.78 token/s | 6,457.97 token/s |
| PyTorch 峰值 allocated | 20.75 GiB | 20.76 GiB | 20.76 GiB |
| PyTorch 峰值 reserved | 21.21 GiB | 21.21 GiB | 21.21 GiB |

融合 INT8 的输出吞吐相对旧 INT8 提升 **107.9%**，相对 `auto` 提升 **13.7%**；decode 模型执行吞吐相应提升 **120.8%** 和 **7.4%**。去掉 BF16 临时缓冲后，KV 块数相对旧 INT8 增加 **6.9%**。本负载下，INT8 模式没有发生 `auto` 模式的额外 prefill 重算，因此端到端收益同时包含容量差异，不能把全部提升归因于内核。

融合 INT8 三轮输出吞吐为 **5,803.01 / 5,775.39 / 5,762.88 token/s**；最后一处 prefix 优化后的复测为 **5,781.30 token/s**。旧路径本次复测为 **2,772.02 token/s**，与前次三轮中位数接近。

## 延迟分布

每个单元格按「均值 / P50 / P95 / P99」排列，单位毫秒。跨轮先算各轮的请求分布统计量，再对三轮逐指标取中位数。

| 指标 | auto | int8_dequant | int8 融合 |
| --- | --- | --- | --- |
| TTFT | 1,305.98 / 1,344.04 / 2,280.29 / 2,280.29 | 1,296.08 / 1,332.30 / 2,265.88 / 2,265.88 | 1,296.78 / 1,333.24 / 2,262.32 / 2,262.32 |
| TPOT | 32.43 / 32.00 / 42.23 / 57.83 | 59.12 / 60.48 / 68.79 / 73.24 | 27.96 / 27.66 / 35.63 / 40.24 |
| ITL | 29.85 / 26.84 / 53.85 / 55.61 | 55.78 / 57.90 / 62.51 / 62.68 | 26.04 / 25.94 / 27.68 / 27.86 |
| 请求总延迟 | 16,895.10 / 17,739.45 / 25,165.80 / 25,480.09 | 30,429.28 / 31,528.68 / 47,885.12 / 48,129.96 | 14,897.03 / 15,372.91 / 22,949.03 / 23,117.19 |

TTFT 包括队列等待和首 token 完成后的 CPU 回填；TPOT 是请求内首末 token 时间差除以后续 token 数；ITL 汇总相邻 token 时间间隔。阶段执行吞吐以 `ModelRunner.run` 时间为分母，和完整 `generate()` 墙钟吞吐口径不同。显存是 PyTorch 分配器统计，INT8 的主要显存收益体现为可分配块数。

## 正确性与原始记录

`python -m unittest discover -s tests -v` 通过：融合内核的 decode 和带缓存前缀的 prefill，与“恢复 BF16 KV 后调用 FlashAttention”的结果在测试容差内一致；覆盖 GQA、跨 block、head dim 80/128 和 CUDA Graph 空长度占位。端到端还连续提交两个共享首个 256-token block 的请求，第二个请求确认复用了 1 个缓存块并成功生成 8 token。完整 256 请求的四轮融合运行都通过输出长度校验。尚未用真实问答评价 INT8 量化造成的回答质量变化。

- [融合 INT8 第 1 轮](kv_cache_fused_2026-10-02/int8_1.txt)、[第 2 轮](kv_cache_fused_2026-10-02/int8_2.txt)、[第 3 轮](kv_cache_fused_2026-10-02/int8_3.txt)、[最后优化后复测](kv_cache_fused_2026-10-02/int8_4.txt)
- [旧路径当前代码复测](kv_cache_fused_2026-10-02/int8_dequant_1.txt)
- [前次 `auto` 与旧 INT8 六轮原始日志](kv_cache_int8_2026-10-02/)
