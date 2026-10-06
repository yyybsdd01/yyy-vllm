# nano-vLLM 推理基线（2026-10-01）

## 复测条件

- 代码：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`。现有 19 个已修改的 Python 文件与该提交相比，仅有注释和文档字符串变化（去掉文档字符串后的 AST 相同）。本次未修改推理实现。
- 模型：本地 `Qwen3-0.6B`；硬件：GPU 0，NVIDIA GeForce RTX 3090 Ti；张量并行 1；CUDA Graph 开启；`max_model_len=4096`、`max_num_batched_tokens=16384`、`max_num_seqs=512`、`gpu_memory_utilization=0.9`、KV 块大小 256（其余为项目默认值）。
- 环境：Python Conda 环境 `nanovllm`；PyTorch 2.5.1+cu121；Transformers 4.57.3；Triton 3.1.0；FlashAttention 2.7.4.post1。
- 工作负载：与仓库 `bench.py` 一致，随机种子 0，256 个请求一次性提交；每个输入 100–1024 token，每个输出 100–1024 token；`temperature=0.6`，`ignore_eos=True`。总计 142,827 个唯一输入 token 和 133,966 个实际输出 token。
- 命令：`CUDA_VISIBLE_DEVICES=0 /home/xgd/anaconda3/envs/nanovllm/bin/python benchmark_inference_metrics.py`。每次在全新进程中运行，同一配置共 3 次；模型加载和预热不计入耗时。三次输出长度校验均通过。

## 结果

下表的每个数值是三次运行对应统计量的中位数。延迟分位数在单次运行的 256 个请求上计算；ITL 分位数在单次运行的全部相邻 token 间隔上计算。

| 指标 | 均值 | P50 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: |
| 首 token 延迟 TTFT（ms/请求） | 1,291.92 | 1,328.33 | 2,255.41 | 2,255.41 |
| 每请求平均后续 token 时间 TPOT（ms/token） | 32.16 | 31.74 | 41.79 | 57.38 |
| 相邻 token 间隔 ITL（ms/token） | 29.62 | 26.55 | 53.71 | 55.52 |
| 请求总延迟（ms/请求） | 16,760.49 | 17,595.84 | 24,986.35 | 25,299.43 |

| 吞吐与资源指标 | 三次运行中位数 |
| --- | ---: |
| 完成 256 请求的墙钟时间 | 26.195 s |
| 请求吞吐量 | 9.77 请求/s |
| 唯一输入 token / 墙钟时间 | 5,452.45 token/s |
| 实际输出 token / 墙钟时间 | 5,114.18 token/s |
| 输入加输出 token / 墙钟时间 | 10,566.64 token/s |
| Prefill 模型执行吞吐量 | 47,187.10 处理 token/s |
| Decode 模型执行吞吐量 | 6,051.58 处理 token/s |
| PyTorch 峰值 allocated 显存 | 20.75 GiB |
| PyTorch 峰值 reserved 显存 | 21.21 GiB |

三次输出吞吐量依次为 **5,150.08 / 5,114.18 / 5,107.74 token/s**（范围约 0.8%）。仓库原有 `bench.py` 独立运行一次为 **133,966 token / 26.87 s = 4,986.38 token/s**。

## 测量口径

- 所有请求在计时开始时一起进入队列。TTFT 是计时开始至请求第一个 token 完成调度回填；请求总延迟是至最后一个 token 回填；TPOT 是每个请求的 `(最后 token 时间 - 首 token 时间) / (输出 token 数 - 1)`；ITL 是相邻两个 token 的时间差。
- 请求/输入/输出吞吐量以完整 `generate()` 墙钟时间为分母，包含调度、GPU 执行及返回文本解码。Prefill/Decode 的“模型执行吞吐量”只以 `ModelRunner.run` 调用时间为分母，包含该调用内的准备和同步，不含外层调度与回填，不能与端到端吞吐量直接比较。
- Prefill 执行共处理 181,836 个 token，高于 142,827 个唯一输入 token；该计数包含重新执行的 prefill。PyTorch 显存统计反映其分配器的 allocated/reserved 峰值，不等同于整卡 NVML 峰值。
- 这是 **nano-vLLM 的离线批处理基线**，随机 token 工作负载不代表在线到达流量或真实文本分布。当前 `nanovllm` 环境未安装原版 `vllm`，因此本报告没有原版 vLLM 的对照数据。
