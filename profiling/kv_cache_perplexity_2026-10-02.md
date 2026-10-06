# KV cache 量化前后困惑度测评（2026-10-02）

## 结果

模型为本机 `Qwen3-0.6B`，语料为 WikiText-2 raw test 全集。对相同的 298,938 个下一 token 位置做 teacher forcing，PPL 越低越好。

| KV 路径 | 总 NLL | 全量 PPL | 相对 auto | 读取缓存部分 PPL |
| --- | ---: | ---: | ---: | ---: |
| `auto`：BF16 KV + FlashAttention | 877,201.595184 | **18.810083** | 基线 | 18.050919 |
| `int8_dequant`：INT8 KV → BF16 暂存 → FlashAttention | 881,105.884247 | 19.057364 | +1.3146% | 18.304154 |
| `int8`：融合 Triton 内核中解量化 | 881,086.955719 | **19.056158** | **+1.3082%** | 18.302918 |

融合 `int8` 相比 `auto` 增加 **0.246074 PPL**，平均每个预测位置的 NLL 增加 **0.01299721**。`int8` 与 `int8_dequant` 的全量 PPL 仅差 **0.001207**。三种模式首次填充块的 PPL 均为 **34.889511**，因为这时没有读取历史缓存。零历史缓存的 256-token 对照中，`auto` 与 `int8` 都是 **28.460446**。

`int8_dequant` 与 `auto` 都通过 FlashAttention 读取 BF16 KV；两者的差异主要反映 INT8 写入与解量化带来的数值误差。`int8` 与 `int8_dequant` 的差异还包括两个 attention 内核的浮点运算顺序。上述归因是基于当前实现的近似隔离，不能视为严格的逐项误差分解。

## 语料和计算方式

- 数据：Salesforce [WikiText-2 raw test Parquet](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/wikitext-2-raw-v1/test-00000-of-00001.parquet)，4,358 个 `text` 行，按原顺序用单个换行符连接；Parquet SHA256 为 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`，连接后的 UTF-8 文本 SHA256 为 `aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc`。
- Tokenizer：本机 `/home/xgd/huggingface/Qwen3-0.6B`，`add_special_tokens=False`；文本共 298,939 个 token，评分 298,938 个下一 token 位置；不加聊天模板或 EOS。
- 模型：同一份 BF16 权重、单张 RTX 3090 Ti、张量并行度 1。模型配置文件 SHA256 为 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`。
- 上下文：每 4,096 个预测位置重置一次位置和 KV cache，共 73 个窗口；每次送入 256 个 token，KV block size 256。首次块共有 18,688 个位置（6.25%），读取历史缓存的后续块共有 280,250 个位置（93.75%）。同一窗口内，各模式处理完全相同的 token 序列。
- 评分：对每个位置从模型最终隐藏状态计算完整词表 logits，求真实下一 token 的交叉熵并相加。`PPL = exp(总 NLL / 298938)`。`fresh_PPL` 与 `cached_PPL` 分别用各自位置的 NLL 和位置数计算。模型正常推理接口在 prefill 时仅输出每段最后一个位置的 logits，因此测评脚本单独对所有隐藏状态使用同一 LM head 权重投影。

这是一种固定 **4,096-token 非重叠窗口、256-token 分块** 的项目内对比协议。窗口边界会重置历史，故数值不能直接和其他 tokenizer、语料拼接、窗口或 stride 规则下公布的 WikiText PPL 对比。语料是英文；结果只衡量下一 token 概率，不直接衡量中文对话质量。

## 复现

本机环境：`torch 2.5.1+cu121`、`transformers 4.57.3`、`triton 3.1.0`、`flash-attn 2.7.4.post1`。测评时工作树包含本次 INT8 KV 实现，基底提交为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。

```bash
mkdir -p /home/xgd/.cache/nanovllm_eval
curl -L --fail \
  'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet' \
  -o /home/xgd/.cache/nanovllm_eval/wikitext2_test.parquet

/home/xgd/anaconda3/envs/llama-factory/bin/python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
rows = pq.read_table('/home/xgd/.cache/nanovllm_eval/wikitext2_test.parquet').column('text').to_pylist()
Path('/home/xgd/.cache/nanovllm_eval/wikitext2_test.txt').write_text('\n'.join(rows), encoding='utf-8')
PY

/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/kv_cache_perplexity.py \
  --text /home/xgd/.cache/nanovllm_eval/wikitext2_test.txt \
  --max-tokens 298938
```

测评脚本：[`kv_cache_perplexity.py`](kv_cache_perplexity.py)。原版与融合 INT8 的完整运行日志：[`kv_cache_perplexity_full_2026-10-02.txt`](kv_cache_perplexity_full_2026-10-02.txt)；INT8 解量化对照日志：[`kv_cache_perplexity_dequant_2026-10-02.txt`](kv_cache_perplexity_dequant_2026-10-02.txt)。另有前 16,384 个预测位置的先导运行日志：[`kv_cache_perplexity_2026-10-02.txt`](kv_cache_perplexity_2026-10-02.txt)，其 PPL 分别为 16.505059 和 16.779630；前缀结果不代表全集。
