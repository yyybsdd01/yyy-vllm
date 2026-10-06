# 较快 INT8 双 scale 组合与未量化 BF16 对照

BF16：原 FlashAttention prefill/decode，模型权重与 KV 均 BF16。
较快 INT8：模型权重 BF16；KV INT8，每 token/head 两个 FP32 scale，布局 block/token/head/2；prefill BM=64、BN=64、num_stages=1，decode BM=16、BN=64、num_stages=1，均 BD=128、4 warps。

## 新一轮完整模型结果

| 指标 | BF16 | 较快 INT8 | 变化 |
| --- | ---: | ---: | ---: |
| TTFT MEAN ms | 1184.05 | 1259.52 | +6.37% |
| TTFT P50 ms | 1222.24 | 1300.80 | +6.43% |
| TTFT P95 ms | 2165.03 | 2302.19 | +6.34% |
| TTFT P99 ms | 2165.03 | 2302.19 | +6.34% |
| TPOT MEAN ms | 32.43 | 24.86 | -23.34% |
| TPOT P50 ms | 31.97 | 24.30 | -23.99% |
| TPOT P95 ms | 42.30 | 32.99 | -22.01% |
| TPOT P99 ms | 57.75 | 37.96 | -34.27% |
| ITL MEAN ms | 29.83 | 22.93 | -23.13% |
| ITL P50 ms | 26.84 | 22.68 | -15.50% |
| ITL P95 ms | 53.71 | 24.16 | -55.02% |
| ITL P99 ms | 55.65 | 24.29 | -56.35% |
| End-to-end latency MEAN ms | 16764.60 | 13235.12 | -21.05% |
| End-to-end latency P50 ms | 17606.37 | 13689.88 | -22.24% |
| End-to-end latency P95 ms | 25026.61 | 20107.98 | -19.65% |
| End-to-end latency P99 ms | 25340.03 | 20292.05 | -19.92% |
| generate s | 26.236 | 20.375 | -22.34% |
| 输出 token/s | 5106.21 | 6574.97 | +28.76% |
| 请求/s | 9.76 | 12.56 | +28.69% |
| 输入 token/s | 5443.96 | 7009.86 | +28.76% |
| 输入+输出 token/s | 10550.17 | 13584.84 | +28.76% |
| KV blocks | 701 | 1320 | +88.30% |
| 峰值 allocated GiB | 20.75 | 20.76 | +0.05% |
| 峰值 reserved GiB | 21.05 | 21.06 | +0.05% |
| 缓存抢占次数 | 84 | 0 | -100.00% |
| prefill model-run seconds | 3.753 | 2.292 | -38.93% |
| prefill model-run tokens | 181836 | 142827 | -21.45% |
| prefill model-run tokens_per_s | 48456.25 | 62314.67 | +28.60% |
| decode model-run seconds | 22.228 | 17.850 | -19.70% |
| decode model-run tokens | 133626 | 133710 | +0.06% |
| decode model-run tokens_per_s | 6011.49 | 7490.85 | +24.61% |

## 精度：WikiText-2 raw test 全集 PPL

相同的 298,938 个下一 token 预测位置，4096-token 非重叠窗口、256-token 分块 teacher forcing。PPL 越低越好。
INT8 在首次分块也读取量化后的 KV，匹配这次所有 prefill 使用自写 kernel 的方案。

| 配置 | 全量 PPL | 相对 BF16 | 首块 PPL | 历史缓存部分 PPL |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 18.810820 | +0.0000% | 34.892483 | 18.051570 |
| original_int8 | 18.930555 | +0.6365% | 35.487352 | 18.153684 |
| fast_int8 | 18.930555 | +0.6365% | 35.487352 | 18.153684 |

扩大 Q tile/调整流水后，相对原 INT8 配置 ΔPPL=+0.000000，ΔNLL/token=+0.00000000。

这里的 PPL 衡量真实下一 token 的概率，不是问答正确率。PPL 使用多 token 的 prefill 路径；没有单独运行逐 token decode 的全文 PPL。

## 每轮性能结果

| 配置 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 抢占 | prefill tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1 | 26.142 | 5124.61 | 1203.33 | 31.89 | 84 | 181836 |
| fast_int8 | 1 | 20.321 | 6592.48 | 1294.96 | 24.25 | 0 | 142827 |
| fast_int8 | 2 | 20.375 | 6574.97 | 1301.44 | 24.30 | 0 | 142827 |
| bf16 | 2 | 26.246 | 5104.28 | 1222.24 | 31.97 | 84 | 181836 |
| bf16 | 3 | 26.236 | 5106.21 | 1222.62 | 31.97 | 84 | 181836 |
| fast_int8 | 3 | 20.404 | 6565.67 | 1300.80 | 24.36 | 0 | 142827 |

## 条件和测量范围

- Qwen3-0.6B、RTX 3090 Ti、单卡 GPU 0，BF16 模型权重；decode CUDA Graph、prefill eager。未锁 GPU 频率。
- 两组各 3 个独立新进程，顺序交替，全是这次的新测量；编译、初始化、生成预热和代表 prefill 特化预热不计入正式时间。
- 256 请求同时到达，输入/输出长度各 100–1024，seed=0、torch sampling seed=0、temperature=0.6、ignore_eos=True。输入 142,827、输出 133,966 token，逐请求长度检查通过。
- 页表宽度 1/2/3/4 和 16K-token 大 prefill 在计时前预热。max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、page size=256、gpu_memory_utilization=0.9。
- 相同显存预算下 INT8 可分配更多 KV blocks。BF16 的阶段时间包含抢占后的重新 prefill；因此这些整模型结果包含 KV 容量、调度与 kernel 速度的共同影响。
- 每 token 的 INT8 K/V+双 FP32 scale 占 BF16 KV 字节数的 53.125%；节约的空间用于扩大 KV 容量，完整缓存池和峰值显存不会自动减半。
- TTFT 是整批提交到 CPU postprocess 完成首 token，含排队；TPOT/ITL 也由 CPU postprocess 时间戳计算。阶段时间包括准备、模型、采样和同步，不是单 attention kernel 时间。
- PPL 语料 UTF-8 SHA256、token ID 摘要、模型配置与源码哈希、首块/历史块计数及逐层派发次数保存于 quality/*.json；三组使用相同文本和预测位置。
- 该 PPL 是项目固定窗口协议，语料为英文，不能直接当作中文对话或任务准确率；没有聊天模板或额外 EOS。
- 原 INT8 精度对照使用 BM=16、BN=64、num_stages=3，首次 prefill 同样读量化 KV。旧精度测量首块使用原始 BF16 KV，因协议不同不直接复用旧数值。
- 本次 kernel 实现是上一轮通过恢复后 BF16 参考验证的同一份源码；所有正式源码哈希前后一致，改动仅在独立副本。

## 复现

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_fast_int8_bf16.py --outdir /tmp/fast-int8-bf16
```
