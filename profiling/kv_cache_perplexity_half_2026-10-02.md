# 每半个 KV head 一个 scale 的困惑度对比（2026-10-02）

**后续更新：**为减少重复的 scale 读取，融合双 scale 内核改为向量加载并使用 64-key tile。当前代码重测的全量 PPL 为 **18.944100**；下表的 **18.938513** 是修改前 32-key tile 内核的结果。两版代码的吞吐和延迟见[性能对比](kv_cache_half_performance_2026-10-02.md)。

## 全量结果

本机 Qwen3-0.6B 的 KV head_dim 为 128。原 `int8` 对每个 token 的每个 K/V head 分别使用一个 FP32 scale（覆盖 128 维）；`int8_half` 把每个 head 分为前后各 64 维，分别计算 scale。二者保留同样的 INT8 K/V 数据和融合 attention 路径。

对同一份 [WikiText-2 raw test](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/wikitext-2-raw-v1/test-00000-of-00001.parquet) 的 298,938 个下一 token 位置做 teacher forcing。4,096-token 窗口、256-token 分块、KV block size 256、同一 BF16 模型权重和 tokenizer；PPL 越低越好。

| KV 路径 | 全量 NLL | 全量 PPL | 相对 BF16 | 读取缓存部分 PPL |
| --- | ---: | ---: | ---: | ---: |
| `auto`：BF16 KV | 877,201.595184 | **18.810083** | 基线 | 18.050919 |
| `int8`：每 head 一个 scale，融合读取 | 881,086.955719 | 19.056158 | +1.3082% | 18.302918 |
| `int8_half`：每半 head 一个 scale，融合读取 | 879,235.727936 | **18.938513** | **+0.6828%** | 18.182414 |
| `int8_half_dequant`：双 scale，恢复 BF16 后读取 | 879,167.063293 | 18.934164 | +0.6596% | 18.177959 |

把 scale 数量翻倍后，**PPL 降低 0.117644**（相对原 `int8` 降低 **0.6174%**）。相对 BF16 的 PPL 增量从 **0.246074** 降到 **0.128430**，缩小约 **47.8%**。平均每个预测位置的 NLL 比原 `int8` 低 **0.00619268**。首次填充块在四种模式下 PPL 均为 **34.889511**，因为它不读取历史 KV；后续 280,250 个位置会读取选定格式的缓存。

`int8_half_dequant` 的 PPL 为 18.934164，与融合双 scale 路径相差 0.004349。原单 scale 的解量化路径在[前次测评](kv_cache_perplexity_2026-10-02.md)中为 19.057364；两个解量化路径使用相同的 FlashAttention，因此也支持“更细的 scale 降低量化误差”这一判断。融合内核对半 head scale 采用 32-key tile，原单 scale 内核采用 64-key tile，所以融合两行之间的差值也包含浮点运算顺序的影响。

## 空间与验证

以该模型的 8 个 KV head、128 维、BF16 权重为例，每层每 token 的 K+V 缓存从单 scale INT8 的 `2048B 数据 + 64B scale = 2112B` 变为双 scale 的 `2048B 数据 + 128B scale = 2176B`，增加 **3.03%**；仍比 BF16 KV 的 4096B 少 **46.875%**。这是缓存张量的理论字节数，不是进程峰值显存。

新增模式 `int8_half` 与 `int8_half_dequant` 可通过 `kv_cache_dtype` 或 `example.py --kv-cache-dtype` 选择。`tests.test_half_head_kvcache` 验证了 80/128 维下的量化 scale、INT8 值、解量化结果和 decode/prefix attention；原单 scale attention 测试也通过。另用 `LLM.generate` 分别在 eager 和 CUDA Graph 模式完成了 8-token 短请求。

完整运行日志：[`kv_cache_perplexity_half_2026-10-02.txt`](kv_cache_perplexity_half_2026-10-02.txt)。复现命令：

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/kv_cache_perplexity.py \
  --text /home/xgd/.cache/nanovllm_eval/wikitext2_test.txt \
  --max-tokens 298938 \
  --modes auto int8 int8_half int8_half_dequant
```

文本下载、连接方式与校验值见[前次测评](kv_cache_perplexity_2026-10-02.md)。此结果属于项目内固定分块协议，不宜直接与其他 WikiText PPL 数值比较；语料为英文，不直接代表中文对话质量。
