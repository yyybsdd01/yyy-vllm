# 异步 KV 卸载：实现与验证

已实现可选的异步 KV 卸载/恢复。默认关闭，当前支持单 GPU，不能与旧 preemption_lock 同时启用。

## 调度与数据生命周期

- 顺序为恢复队列分配与提交 H2D → waiting prefill → running decode。恢复提交不执行模型；仅 RUNNING 序列参与 decode。
- 状态为 RUNNING → OFFLOADING → WAITING_RESTORE → RESTORING → RUNNING，恢复事件完成后才进入运行队列。
- BlockManager 继续管理块表、分配与 GPU ref_count。CPU 快照依赖计数与 GPU 引用独立；共享前缀仅在最后 GPU 引用释放且仍有恢复依赖时复制到 CPU。普通完成释放不需要卸载。
- 保存有效 num_cached_tokens，仅备份有效历史涉及的块；恢复可复用仍在 GPU 的有效同代块，缺失块才 H2D。完整前缀 hash 仅在恢复完成后发布。
- 使用独立 D2H/H2D stream、pinned CPU 池和每方向一个连续 GPU staging 块。D2H 先 gather；原物理块的写事件等待 gather 完成。CPU DMA 此后读 staging，原块可继续写。
- prefill、decode、H2D scatter 都遵守原块写事件；RESTORING 持有目标块引用，防止被重分配。若目标 batch 包含等待写的块，当前计算 stream 会等待，不能承诺完全无影响。
- CPU 池上限 4 GiB，最大同时恢复 8 个序列；不足时回退原重新 prefill，单独记录 fallback_preemptions。

## 自动与整模型验证

8 个 CPU 卸载测试、2 个真实 CUDA 测试、7 个原 preemption_lock 测试通过，共 17 个测试方法。
覆盖物理 ID 复用与内容代数、共享前缀最后引用释放、CPU 快照去重、恢复优先级、传输状态隔离、CPU 池不足回退、块边界不重复分配、事件完成后发布 hash、混合负载完整 KV 和输出、真实复制期间覆盖源块的事件保护。
CUDA 数据测试覆盖 28 层 BF16、单 scale INT8 和双 scale INT8 的全部 K/V 与 scales。

Qwen3-0.6B 整模型使用 12 个块强制抢占，共享 256-token 前缀，16 请求，每请求 24 输出 token，CUDA Graph decode。
分别在 BF16 和双 scale INT8 的每次恢复后逐元素检查全部层的每个有效 KV 行和 scales，各检查 16 个恢复逻辑块，完全一致；有效缓存长度一致。
新增单 scale INT8、单/双 scale dequant 三个根目录 CUDA Graph 冒烟测试，每种 24 请求、12 个块、3 次真实卸载和恢复，长度和时间戳数量通过，结束没有传输中的请求。

**整模型输出不保证逐 token 相同。** argmax 固定采样的关闭/开启对照存在不同输出：保存的随机输入 INT8 对照 9/16 请求完全相同，自然复制文本 15/16 请求完全相同。固定每个请求相同后续 token 的 teacher-forcing 对照中，381/384 步 argmax 相同（99.21875%），全 vocab BF16 logits 最大绝对差 3.84375，平均绝对差 0.102546。数据搬运逐元素相同，但不同调度与重新计算路径的整模型数值没有达到完全一致；本次未做 PPL 或自然语言质量评测。

teacher-forcing 的 64-block 无抢占参考还改变了预热前缀在缓存中的驻留情况，因此其差异不能单独归因于卸载。主要对照为相同 12-block 压力下的 off/on。

## 复制和 decode 的真实重叠

独立 16 请求共享前缀压力测试用 torch.profiler 跟踪 GPU DMA 与实际 decode kernel 的时间区间。
H2D 共 14 个 KV/scale DMA，8.988 ms，其中 4.298 ms 与 decode kernel 重叠（47.82%）；D2H 共 18 个 DMA，10.800 ms，其中 0.777 ms 重叠（7.20%）。
此结果仅证明该小负载发生真实并发，不是 1024 请求测试的延迟隐藏比例。不能将 CUDA event 累计时间从 generate 墙钟时间中相减作为隐藏时间。

## 1024 请求性能结果

见 [完整对照报告](metrics/report.md)、[结构化结果](metrics/results.json) 和 [源码、负载与环境记录](metrics/manifest.json)。
相同 8-warp 快速双 scale INT8 kernel、相同 1318 个 KV 块，三轮独立进程交替测量。根目录原 attention kernel 保留；性能副本切换到既有 fast8 kernel，两组完全相同。

- 输出吞吐 6437.98 → 6745.97 token/s，+4.78%；generate 90.681 → 86.541 s，-4.57%。
- TTFT 含排队，P50 32.914 → 18.082 s（-45.06%），均值 27.161 → 25.477 s（-6.20%），P95 70.830 → 65.543 s（-7.46%）。
- TPOT P50 50.25 → 47.94 ms（-4.60%），P95 74.14 → 82.34 ms（+11.06%），P99 147.22 → 183.80 ms（+24.85%）。ITL P50 37.18 → 37.97 ms（+2.12%）。
- 抢占三轮：关闭均为 657；开启 728、716、758，中位数 728（+10.81%）。发生至少两次抢占的请求中位数 122 → 160。策略没有减少抢占或重复抢占。
- prefill 实际执行 token 879176 → 586886（-33.25%），阶段耗时 21.629 → 14.298 s（-33.89%）；decode 阶段 67.852 → 68.636 s（+1.16%）。收益主要对应减少重新 prefill，不能声称 decode kernel 变快。
- GPU staging 额外 29.75 MiB，峰值 allocated 20.73 → 20.76 GiB；CPU pinned 池中位数 3.980 GiB。三轮 CPU 容量不足回退分别为 8、8、7 次，中位数 8 次。
- 六轮均验证 1024 个输出长度和时间戳数量，完整负载 SHA256 相同；前后源码 hash 一致；每请求抢占直方图与总计数一致；所有异步传输结束且池预算满足限制。

## 运行

```python
llm = LLM('/home/xgd/huggingface/Qwen3-0.6B', kv_cache_dtype='int8_half',
          kv_cpu_offload=True, offload_cpu_gb=4.0, offload_max_inflight=8)
```

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_kv_offload*.py' -v
/home/xgd/anaconda3/envs/nanovllm/bin/python -m unittest discover -s tests -p 'test_scheduler_preemption_lock.py' -v
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_async_kv_offload_metrics.py --outdir /tmp/async-offload-recheck --runs 3 --kv-blocks 1318
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/check_kv_offload_model.py --variant profiling/async_kv_offload_2026-10-06/metrics/variant --output /tmp/offload-kv-check.json --offload --verify-kv
```

数据与限制记录：[验证 JSON](validation.json)、[模型 logits 对照](teacher_comparison.json)、[重叠统计](overlap.json)、[原始 profiler trace](overlap_trace.json)、[其余 KV 模式冒烟](mode_smokes.json)。
