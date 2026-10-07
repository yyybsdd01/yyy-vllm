# fast8 对 BF16：WikiText-2 PPL

BF16：18.81081954；fast8：18.93411603；相对增加 0.6555%。PPL 越低越好。

Qwen3-0.6B / RTX 3090 Ti；298,938 个预测位置；4096-token 非重叠窗口、256-token 分块；fast8 首块与缓存块均走量化内核（BM64/BN64，8 warps，num_stages=1）。两组语料、tokenizer 与模型 config 哈希相同。

本指标为分块 prefill teacher forcing，不覆盖采样生成或 CPU 卸载调度的端到端质量。完整结果、派发统计、命令与源码哈希见 JSON。

## 与历史 PPL 的差异核验

保持本次 fast8 源码、语料和评测协议，只在独立副本将 num_warps 从 8 改为 4：PPL=18.930554834033，NLL 与历史 fast4 结果逐值完全一致。8-warps fast8 PPL=18.934116030607，绝对差 0.003561196575。该单参数控制实验确认 warp 配置足以引起此次数值差异，量化读写规则未变；未进一步拆分编译布局和浮点规约顺序的贡献。见 warps_control_comparison.json。
