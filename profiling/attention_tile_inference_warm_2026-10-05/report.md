# 较大 Q tile 和单级流水：完整模型对照

两组都使用 INT8 双 FP32 scale KV、原 token/head scale 布局，自写 kernel 执行已分配缓存的所有 prefill/decode。
当前组：prefill/decode 均 BM=16、BN=64、num_stages=3。优化组：prefill BM=64、BN=64、num_stages=1；decode BM=16、BN=64、num_stages=1。均 BD=128、4 warps。
K tile=128 已在独立 kernel 实验中运行，但整体效果逊于这里选择的 K tile=64。正式源码未修改。

## 三轮中位数

| 指标 | 当前 | 优化实验 | 变化 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 1506.75 | 1254.59 | -16.74% |
| TTFT P50 ms | 1558.84 | 1294.66 | -16.95% |
| TTFT P95 ms | 2759.43 | 2292.76 | -16.91% |
| TTFT P99 ms | 2759.43 | 2292.76 | -16.91% |
| TPOT MEAN ms | 30.90 | 24.78 | -19.81% |
| TPOT P50 ms | 30.23 | 24.23 | -19.85% |
| TPOT P95 ms | 40.38 | 32.89 | -18.55% |
| TPOT P99 ms | 46.37 | 37.83 | -18.42% |
| ITL MEAN ms | 28.63 | 22.85 | -20.19% |
| ITL P50 ms | 28.09 | 22.63 | -19.44% |
| ITL P95 ms | 30.04 | 24.12 | -19.71% |
| ITL P99 ms | 30.23 | 24.30 | -19.62% |
| End-to-end latency MEAN ms | 16461.03 | 13190.89 | -19.87% |
| End-to-end latency P50 ms | 17003.41 | 13648.44 | -19.73% |
| End-to-end latency P95 ms | 25164.34 | 20036.72 | -20.38% |
| End-to-end latency P99 ms | 25342.97 | 20220.03 | -20.21% |
| generate s | 25.423 | 20.303 | -20.14% |
| 输出 token/s | 5269.55 | 6598.27 | +25.22% |
| 请求/s | 10.07 | 12.61 | +25.22% |
| 输入 token/s | 5618.10 | 7034.70 | +25.21% |
| 输入+输出 token/s | 10887.65 | 13632.97 | +25.21% |
| KV blocks | 1320 | 1320 | +0.00% |
| 峰值 allocated GiB | 20.76 | 20.76 | +0.00% |
| 峰值 reserved GiB | 21.06 | 21.06 | +0.00% |
| prefill model-run s | 2.750 | 2.283 | -16.98% |
| prefill tokens/s | 51946.26 | 62572.35 | +20.46% |
| decode model-run s | 22.440 | 17.789 | -20.73% |
| decode tokens/s | 5958.49 | 7516.26 | +26.14% |

## 每轮结果

| 配置 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | prefill s | decode s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 1 | 25.331 | 5288.69 | 1535.60 | 30.15 | 2.715 | 22.319 |
| tiled | 1 | 20.264 | 6611.17 | 1288.84 | 24.19 | 2.272 | 17.757 |
| tiled | 2 | 20.303 | 6598.27 | 1294.66 | 24.23 | 2.283 | 17.789 |
| current | 2 | 25.423 | 5269.55 | 1558.84 | 30.23 | 2.750 | 22.440 |
| current | 3 | 25.465 | 5260.74 | 1561.22 | 30.28 | 2.755 | 22.474 |
| tiled | 3 | 20.378 | 6574.17 | 1300.78 | 24.31 | 2.293 | 17.854 |

## 条件与验证

- Qwen3-0.6B、RTX 3090 Ti、GPU 0，单卡；decode 使用 CUDA Graph，prefill eager；未锁 GPU 频率。
- 256 请求同时到达，输入/输出长度各 100–1024，seed=0、torch sampling seed=0、temperature=0.6、ignore_eos=True。输入 142,827、输出 133,966 token，逐请求长度检查通过。
- 三轮独立新进程，配置顺序交替。初始化、编译、生成预热和代表 prefill 特化预热不计入正式时间；预热后重设采样种子。
- 页表宽度 1/2/3/4，以及 Q=1024、B=16 的大 prefill 在计时前预热；每批实际 prefill 派发次数与形状摘要均审计。
- 两组采用相同 max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9；KV 容量及阶段 token 工作量一致。
- 集成验证覆盖零前缀、混合前缀、长度 1/17/255/257、跨页、部分 Q tile、随机物理页，并与恢复后 BF16 KV 的 FlashAttention 通过 rtol=0.02、atol=0.005。
- 本负载两组及重复轮次生成的输出 token SHA256 一致。较大 tile 和不同流水级数可能改变浮点归约顺序，这里没有评估语言质量。
- TTFT 从整批请求提交到 CPU postprocess 完成首 token，包含排队。TPOT/ITL 也基于 CPU postprocess 时间戳；阶段时间包含准备、模型、采样与同步，不是单 attention kernel 时间。
- 优化组同时改变 prefill 的 BM 和 prefill/decode 的 num_stages；整模型收益属于这组组合。纯 BM/BN 的对照另见 kernel 实验报告。
- 原仓库 nanovllm 源码及 benchmark_inference_metrics.py 的哈希前后相同，实验改动仅在独立副本。

## 输出摘要

| 配置 | 正式输出 SHA256 |
| --- | --- |
| current | 355a78e9f408de388b2508287f777cb546d83685ce75022aaa1914dd34f45781 |
| tiled | 355a78e9f408de388b2508287f777cb546d83685ce75022aaa1914dd34f45781 |

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_attention_tile_metrics.py --outdir /tmp/attention-tile-metrics
```
