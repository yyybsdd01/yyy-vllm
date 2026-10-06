# 普通 prefill 切换到自写 INT8 kernel：完整模型对照

四组均为 INT8 双 scale KV，BF16 模型权重。Flash 组普通 prefill 使用原始 BF16 K/V；INT8 组所有分配了缓存的 prefill 使用量化后的 paged K/V。decode 始终使用相应布局的自写 INT8 kernel。

## 测量结果

| 指标 | Flash prefill 原布局 | Flash prefill head/token | 自写 prefill 原布局 | 自写 prefill head/token |
| --- | ---: | ---: | ---: | ---: |
| TTFT MEAN ms | 1179.74 | 1178.82 | 1507.36 | 1499.13 |
| TTFT P50 ms | 1217.92 | 1216.51 | 1557.69 | 1549.60 |
| TTFT P95 ms | 2156.19 | 2155.88 | 2762.17 | 2746.19 |
| TTFT P99 ms | 2156.19 | 2155.88 | 2762.17 | 2746.19 |
| TPOT MEAN ms | 29.99 | 29.87 | 30.97 | 30.82 |
| TPOT P50 ms | 29.79 | 29.58 | 30.28 | 30.08 |
| TPOT P95 ms | 37.70 | 37.33 | 40.46 | 40.00 |
| TPOT P99 ms | 42.39 | 42.01 | 46.45 | 45.95 |
| ITL MEAN ms | 27.99 | 27.96 | 28.69 | 28.65 |
| ITL P50 ms | 27.97 | 27.92 | 28.08 | 28.07 |
| ITL P95 ms | 29.86 | 29.57 | 30.07 | 29.82 |
| ITL P99 ms | 30.10 | 29.78 | 30.33 | 29.94 |
| End-to-end latency MEAN ms | 15799.78 | 15779.88 | 16496.14 | 16463.99 |
| End-to-end latency P50 ms | 16335.30 | 16286.93 | 17032.92 | 16973.18 |
| End-to-end latency P95 ms | 24478.33 | 24553.93 | 25223.78 | 25302.67 |
| End-to-end latency P99 ms | 24656.34 | 24730.58 | 25402.12 | 25480.04 |
| generate 时间 s | 24.736 | 24.811 | 25.482 | 25.560 |
| 请求/s | 10.35 | 10.32 | 10.05 | 10.02 |
| 输出 token/s | 5415.78 | 5399.56 | 5257.29 | 5241.33 |
| 输入 token/s | 5774.00 | 5756.71 | 5605.03 | 5588.01 |
| 输入+输出 token/s | 11189.77 | 11156.27 | 10862.32 | 10829.34 |
| KV blocks | 1320 | 1320 | 1320 | 1320 |
| 峰值 allocated GiB | 20.76 | 20.76 | 20.76 | 20.76 |
| 峰值 reserved GiB | 21.06 | 21.06 | 21.06 | 21.06 |
| prefill model-run seconds | 2.146 | 2.146 | 2.752 | 2.736 |
| prefill model-run tokens_per_s | 66555.68 | 66558.85 | 51897.13 | 52202.86 |
| prefill model-run tokens | 142827 | 142827 | 142827 | 142827 |
| decode model-run seconds | 22.357 | 22.436 | 22.488 | 22.591 |
| decode model-run tokens_per_s | 5980.75 | 5959.70 | 5945.87 | 5918.81 |
| decode model-run tokens | 133710 | 133710 | 133710 | 133710 |

## 对照变化

- token_head，将普通 prefill 从 FlashAttention 换成自写 kernel：prefill 总时间 +28.24%，TTFT P50 +27.90%，输出吞吐 -2.93%。
- head_token，将普通 prefill 从 FlashAttention 换成自写 kernel：prefill 总时间 +27.49%，TTFT P50 +27.38%，输出吞吐 -2.93%。
- prefill/decode 均自写时，head/token 相对原布局：prefill 总时间 -0.58%，TTFT P50 -0.52%，输出吞吐 -0.30%。

## 条件与验证

- Qwen3-0.6B，RTX 3090 Ti，单卡 GPU 0；CUDA Graph decode，prefill eager；BM=16、BN=64、BD=128，4 warps。
- 256 请求同时到达；输入/输出长度均 100–1024，seed=0、temperature=0.6、ignore_eos=True；输入 142,827，输出 133,966 token。
- 每组 3 个新进程，顺序轮换；取各项统计量的轮中位数。生成预热和模型初始化不计入，未锁 GPU 频率。
- 四组均在计时前额外预热页表宽度 1/2/3/4（对应 256/512/768/1024 token）和 16K-token prefill 大批量，以排除新 prefill 特化首次编译。预热后重设 torch seed=0。
- max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、block_size=256、gpu_memory_utilization=0.9；KV 容量和阶段 token 工作量一致。
- 隔离副本里补齐普通 prefill 的 block table，再将已分配 INT8 缓存的 attention 都派发到自写 kernel。初始化时没有缓存，仍以 FlashAttention 预热；容量估算路径相同。
- 集成正确性覆盖零前缀、混合前缀、长度 1/17/255/257、非连续物理页、跨页、部分 Q tile；对恢复后的 BF16 KV 调用 FlashAttention，rtol=0.02、atol=0.005 通过。
- 正式测量记录每批 prefill 的长度摘要及每层实际派发次数，确保自写组没有调用 FlashAttention prefill。
- 每个 prefill 模式内，两种布局和重复轮次的输出 token 摘要一致。Flash 与 INT8 prefill 的输出是否一致见下表；量化后 prefill 引入近似，内容不同不能作为布局错误。没有测语言质量。
- TTFT 从整批请求提交到 CPU postprocess 首 token 完成，含排队；阶段时间为 ModelRunner.run 累计时间，含准备、模型、采样和同步，并非单 attention kernel 时间。
- 原仓库 nanovllm 源码和 benchmark_inference_metrics.py 的 SHA256 前后相同；实验改动仅在四份副本。

| prefill 模式 | 正式输出 SHA256 | Flash/INT8 一致 |
| --- | --- | --- |
| flash | a13925efdc6a8426adab2c0531ac77270aff5b9e23a9b8997152eab8976acafb | False |
| int8 | 355a78e9f408de388b2508287f777cb546d83685ce75022aaa1914dd34f45781 | False |

## 每轮结果

| 组 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | prefill s | Flash/INT8 prefill 调用 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| flash_token_head | 1 | 24.714 | 5420.64 | 1201.02 | 2.123 | 252/0 |
| flash_head_token | 1 | 24.805 | 5400.70 | 1211.39 | 2.135 | 252/0 |
| int8_token_head | 1 | 25.362 | 5282.24 | 1557.05 | 2.750 | 0/252 |
| int8_head_token | 1 | 25.536 | 5246.20 | 1549.60 | 2.736 | 0/252 |
| flash_token_head | 2 | 24.893 | 5381.59 | 1217.92 | 2.146 | 252/0 |
| int8_head_token | 2 | 25.560 | 5241.33 | 1547.79 | 2.735 | 0/252 |
| int8_token_head | 2 | 25.502 | 5253.07 | 1557.69 | 2.752 | 0/252 |
| flash_head_token | 2 | 24.836 | 5393.93 | 1217.84 | 2.146 | 252/0 |
| int8_token_head | 3 | 25.482 | 5257.29 | 1560.33 | 2.752 | 0/252 |
| int8_head_token | 3 | 25.562 | 5240.84 | 1550.97 | 2.737 | 0/252 |
| flash_token_head | 3 | 24.736 | 5415.78 | 1219.85 | 2.150 | 252/0 |
| flash_head_token | 3 | 24.811 | 5399.56 | 1216.51 | 2.147 | 252/0 |

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_prefill_kernel_metrics.py --outdir /tmp/prefill_kernel_metrics
```

产物包括 manifest.json、results.json、comparison.csv、workload.json、四份隔离源码、集成正确性记录、smoke 与正式日志。

## 单 attention GPU 时间

另行排除 CPU、首次编译、KV 写入和模型其他层，使用 CUDA Graph + CUDA Event 测零前缀 prefill。Q/K/V stride 与模型一致，并使用非连续物理页。完整轮次和正确性见 [kernel report](kernel_comparison/report.md)。

| B | 长度 | Flash BF16 ms | Flash restored ms | INT8 原布局 ms | INT8 head/token ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 0.0135 | 0.0135 | 0.0323 | 0.0319 |
| 1 | 1024 | 0.0977 | 0.0971 | 0.3047 | 0.2985 |
| 16 | 1024 | 0.9840 | 0.9888 | 4.5312 | 4.4288 |
| 32 | 512 | 0.5421 | 0.5533 | 2.5559 | 2.5020 |

B=16、长度=1024 时，自写原布局 attention 耗时约为 FlashAttention 的 4.60 倍，head/token 约为 4.50 倍。相同量化 K/V 恢复成 BF16 后由 FlashAttention 计算的耗时仍相近。这说明本次 prefill 回退在 attention GPU 执行中也存在，无法只归因于 CPU 准备或首次编译；这里比较的是整套 kernel 实现，尚未隔离分块、流水线与解量化各自的贡献。

单 kernel 复现：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_initial_prefill_kernels.py --outdir /tmp/initial_prefill_kernels
```
