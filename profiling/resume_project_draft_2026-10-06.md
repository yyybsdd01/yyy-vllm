# 简历项目草稿：基于 nano-vLLM 的推理性能优化

技术栈：Python、PyTorch、Triton、CUDA Graph

- 基于 nano-vLLM 实现可切换的单/双 scale INT8 KV cache 与 Triton paged attention，融合解量化、QK、在线 softmax 与 PV 计算，支持 prefill/decode 及前缀缓存；双 scale 方案将每 token KV 与 scale 存储开销较 BF16 降低 46.9%，在既有相同显存预算对照中将 KV 块容量由 701 提升至 1320。
- 针对 prefill/decode 分别调优 tile、warp 数与流水参数；在 RTX 3090 Ti、Qwen3-0.6B、256 请求离线配对测试中，量化优化版本较 BF16 基线输出吞吐提升 28.8%（5106→6575 token/s），TPOT P50 降低 24.0%（31.97→24.30 ms），结果取三轮独立进程中位数。
- 实现 pinned CPU 缓冲池、独立 D2H/H2D stream 与 CUDA event 驱动的异步 KV 卸载/恢复，处理共享前缀引用、物理块复用与传输状态；按下一批 decode 的新增块需求限制恢复和 prefill 准入，在 1024 请求消融测试中较原卸载策略减少 36.5% 抢占，TPOT P99 降低 21.8%。
- 建立延迟、吞吐、KV 容量和 PPL 评测；在 WikiText-2 固定 teacher-forcing 协议下，既有量化版本 PPL 为 18.9306，BF16 为 18.8108，相对增加 0.64%；20 项调度与 CUDA 测试通过，BF16/INT8 整模型恢复后的有效 KV 与 scales 逐元素一致。

上述数字属于两个独立性能实验与一个量化质量实验，不是最终组合版本的单一 BF16 对照结果。

## 指标及证据

| 实验 | 对照及条件 | 证据 |
| --- | --- | --- |
| 量化存储、吞吐和延迟 | BF16 vs 优化 INT8，256请求，4-warp版本，Qwen3-0.6B，RTX3090Ti，三轮中位数 | [配对报告](fast_int8_vs_bf16_paired_2026-10-05/report.md)、[结构化结果](fast_int8_vs_bf16_paired_2026-10-05/results.json) |
| 量化 PPL | WikiText-2 raw test，298938个预测位置，4096-token非重叠窗口，256-token分块；多token prefill评分 | [BF16](fast_int8_vs_bf16_paired_2026-10-05/quality/bf16.json)、[INT8](fast_int8_vs_bf16_paired_2026-10-05/quality/fast_int8.json) |
| 恢复准入策略 | 原卸载 vs 预留decode块；相同8-warp INT8，1024请求，固定1318 GPU块；三轮中位数 | [报告](kv_offload_decode_reserve_2026-10-06/report.md)、[结果](kv_offload_decode_reserve_2026-10-06/results.json)、[验证](kv_offload_decode_reserve_2026-10-06/validation.json) |

## 最终版本的主对照仍需补齐

最终“8-warp INT8＋异步卸载＋decode块预留”组合还没有同轮运行的原始 BF16 主对照，也没有对应的正式 PPL/任务质量评测。不能把历史 BF16 的吞吐与最终版的吞吐直接相除，作为同轮配对提升。

建议统一比较原始 BF16、INT8无卸载、INT8原卸载、INT8预留四组。使用同一模型、输入与输出长度、负载摘要、采样协议、预热及GPU显存预算，在256/512/1024请求等负载下记录：

- TTFT、TPOT、ITL和端到端延迟的均值及P50/P95/P99；TTFT明确包含排队。
- 输出token/s、请求/s、总耗时及实际decode batch分布。
- 每token KV字节数、GPU KV块容量、GPU峰值、pinned CPU内存和传输量。
- 抢占、重复抢占、恢复后无生成进度再抢占及重算prefill token数。
- 相同语料、tokenizer、预测位置与窗口协议的PPL；如验证卸载影响，另测实际decode/卸载路径，不能只沿用prefill量化PPL。

当前可以准确陈述的取舍：既有256请求量化对照的TTFT P50增加6.43%；最新1024请求预留策略较原卸载的TTFT P50增加71.27%，TTFT均值基本不变。每token KV开销降低46.9%不等于整个进程显存降低46.9%；PPL相对增加0.64%不等于任务准确率降低0.64%。
