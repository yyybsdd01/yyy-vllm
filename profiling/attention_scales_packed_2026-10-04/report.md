# 二维 scale 加载接入验证（2026-10-04）

双 scale 分支现在用两次 `[BN,2]` 的 tl.load 读取 K/V scale，再用 tl.split 得到前后两组。单 scale 分支、半维选择、解量化和 attention 算法保持原实现。逐字对比确认保留了修改前的其他源码和用户注释，差异见 change.diff，原文件见 before.py。

现有两项 GPU 测试通过，覆盖 head_dim=80/128、单/双 scale、decode/cached prefill。对比脚本另外校验了不同 half scale 与 BF16 FlashAttention 参考，以及相同恢复 K/V 时三个版本的逐元素一致性。

RTX 3090 Ti，decode batch=256、context=1024，Q heads=16、KV heads=8、head_dim=128；BM=16、BN=64、BD=128，4 warps。CUDA Event + 每图8次kernel、每轮10次回放、11轮随机顺序，单位为每次调用的轮均值中位数。只测 attention 内核，不含 KV 写入、其他模型层或调度。

| 版本 | 中位数 µs | P10 µs | P90 µs |
| --- | ---: | ---: | ---: |
| 单 scale | 1209.894 | 1208.845 | 1216.602 |
| 原四次加载双 scale | 1467.226 | 1466.739 | 1473.907 |
| 当前二维加载双 scale | 1251.187 | 1250.445 | 1259.738 |

当前版本在此测例耗时减少 14.72%。这里未测完整模型吞吐；此前 batch=1 时二维加载略慢，见 ../attention_scales_2026-10-04/report.md。

已修复 compare_attention_scales.py 的版本构造逻辑，使其在接入后仍保留原四次加载作为 dual 基线；dual_packed 在本次计时中直接使用生产内核。AST 对比确认四次加载基线与修改前内核一致。

复现：

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_attention_scales.py --quick --variants single dual dual_packed --outdir /tmp/attention_scales_packed_recheck
```

完整轮次见 timings.json，检查记录见 validation.json。当前生产源码 SHA256：310be258172cfcadc8ca728f3c0ea0de0427c7569ce59f35e308bbd20e3dff2a。
