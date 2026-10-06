"""Render the completed six-run comparison from its recorded metrics."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results.json").read_text())
manifest = json.loads((ROOT / "manifest.json").read_text())
assert len(data["trials"]) == 6 and "finished_utc" in manifest
assert all(row["input_tokens"] == 142827 and row["output_tokens"] == 133966 for row in data["trials"])
assert all(row["graph"] for row in data["trials"])
for relative, expected in manifest["source_hashes"].items():
    assert hashlib.sha256((ROOT / "source_snapshot" / relative).read_bytes()).hexdigest() == expected

bf16 = data["medians"]["auto"]
int8 = data["medians"]["int8_half"]
delta = lambda a, b: (b / a - 1) * 100
output_delta = delta(bf16["output_tokens_per_s"], int8["output_tokens_per_s"])
ttft_delta = delta(bf16["latency_ms"]["TTFT"]["p50"], int8["latency_ms"]["TTFT"]["p50"])
tpot_delta = delta(bf16["latency_ms"]["TPOT"]["p50"], int8["latency_ms"]["TPOT"]["p50"])

lines = [
    "# 当前二维加载双 scale INT8 与 BF16 KV cache 的推理指标（2026-10-04）", "",
    f"当前双 scale INT8 的输出吞吐比 BF16 提升 **{output_delta:.2f}%**；"
    f"TTFT P50 变化 **{ttft_delta:+.2f}%**，差异很小；"
    f"TPOT P50 变化 **{tpot_delta:+.2f}%**。ITL 的典型值与尾部变化不同，见完整分布。", "",
    "## 测量条件", "",
    "- Qwen3-0.6B，RTX 3090 Ti，单卡 GPU 0；模型权重均为 BF16。",
    "- `auto`：BF16 KV cache + FlashAttention。`int8_half`：INT8 KV cache，K/V 各按 head 前后半维保留两个 FP32 scale，融合 Triton attention；使用已接入的 `[BN,2]` 二维 scale 加载。",
    "- CUDA Graph 开启；max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、KV block size=256、gpu_memory_utilization=0.9。",
    "- 256 个请求同时提交；随机 token ID 输入长度 100–1024，目标输出长度 100–1024；seed=0、temperature=0.6、ignore_eos=True。每轮输入 142,827 token、输出 133,966 token，各请求长度校验通过。",
    "- 两种模式各 3 个独立进程，顺序 auto→INT8 / INT8→auto / auto→INT8。每进程加载与一次生成预热不计入正式测量。两种模式使用同一工作树，已逐轮核对源码哈希。",
    "- 先计算每轮分布指标，再逐项取三轮中位数；没有把六轮请求混在一起计算一个分位数。",
    "- 沿用原基准协议：seed 固定 Python 负载生成，未设置 PyTorch 采样随机种子。固定输出长度保证同一工作量；输出内容因采样与量化可以不同。",
    "- 环境：PyTorch 2.5.1+cu121、Triton 3.1.0、FlashAttention 2.7.4.post1、Transformers 4.57.3、驱动 535.183.01。",
    "- 完整负载见 `workload.json`，配置和逐轮命令见 `manifest.json`，源码见 `source_snapshot/`，结构化结果见 `results.json`。", "",
    "## 延迟分布", "",
    "单位毫秒。TTFT 和请求总延迟为 ms/request，TPOT 为每请求平均 ms/token，ITL 为相邻 token 的 ms/token。", "",
    "| 指标 | 分位/统计量 | BF16 | 当前双 scale INT8 | 相对变化 |",
    "| --- | --- | ---: | ---: | ---: |",
]
csv_rows = []
for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
    for quantile in ("mean", "p50", "p95", "p99"):
        a, b = bf16["latency_ms"][metric][quantile], int8["latency_ms"][metric][quantile]
        lines.append(f"| {metric} | {quantile.upper()} | {a:.2f} | {b:.2f} | {delta(a,b):+.2f}% |")
        csv_rows.append(dict(metric=metric, statistic=quantile, unit="ms", bf16=a, int8_half=b,
                             change_percent=delta(a,b)))

lines += ["", "## 吞吐、阶段执行与容量", "",
          "| 指标 | BF16 | 当前双 scale INT8 | 相对变化 |",
          "| --- | ---: | ---: | ---: |"]
for title, key, digits, unit in (
    ("256 请求总时间，s", "elapsed_s", 3, "s"),
    ("请求吞吐，request/s", "requests_per_s", 2, "request/s"),
    ("输入 token/s", "input_tokens_per_s", 2, "token/s"),
    ("输出 token/s", "output_tokens_per_s", 2, "token/s"),
    ("输入+输出 token/s", "total_tokens_per_s", 2, "token/s"),
    ("可分配 KV 块", "blocks", 0, "block"),
    ("峰值 PyTorch allocated，GiB", "peak_allocated_gib", 2, "GiB"),
    ("峰值 PyTorch reserved，GiB", "peak_reserved_gib", 2, "GiB"),
):
    a, b = bf16[key], int8[key]
    lines.append(f"| {title} | {a:.{digits}f} | {b:.{digits}f} | {delta(a,b):+.2f}% |")
    csv_rows.append(dict(metric=key, statistic="run_median", unit=unit, bf16=a, int8_half=b,
                         change_percent=delta(a,b)))
for stage in ("prefill", "decode"):
    for title, key, digits in (("model-run token/s", "tokens_per_s", 2),
                              ("实际执行 token", "tokens", 0), ("model-run 秒", "seconds", 3)):
        a, b = bf16["stages"][stage][key], int8["stages"][stage][key]
        lines.append(f"| {stage} {title} | {a:.{digits}f} | {b:.{digits}f} | {delta(a,b):+.2f}% |")

lines += ["", "## 如何解释", "",
    "1. **TTFT 基本不变。** 普通首次 prefill 两种模式都走 FlashAttention；INT8 额外量化并写缓存。当前二维 scale 读取主要影响 decode 和带缓存前缀的 prefill，不能据此期待首次 prefill 的 TTFT 同比例下降。",
    f"2. **端到端吞吐收益包含容量与调度差异。** BF16 prefill 实际执行 {bf16['stages']['prefill']['tokens']:,} token，"
    f"INT8 执行 {int8['stages']['prefill']['tokens']:,} token。BF16 有额外 prefill 重算；INT8 更大的 KV 容量减少了本负载的重算。"
    "因此 prefill 执行吞吐不可直接当成同一形状 attention 的速度比，端到端增益也不能全部归因于 scale 加载优化。",
    f"3. **Decode 执行吞吐变化较小。** BF16 为 {bf16['stages']['decode']['tokens_per_s']:.2f} token/s，"
    f"INT8 为 {int8['stages']['decode']['tokens_per_s']:.2f} token/s。阶段时间以 `ModelRunner.run` 调用为边界，含准备、模型、采样和同步。两种模式的调度轨迹不相同。",
    "4. **ITL P50 与 P95 应同时看。** INT8 的典型 token 间隔略高，但尾部间隔下降；TPOT 是每请求首末 token 时间差除以后续 token 数，ITL 汇总全部请求的相邻间隔，两种统计口径和权重不同。容量更大的模式可以维持不同的活跃请求数量，尾部也包含重算、调度造成的间隔。",
    "5. **峰值显存接近来自固定的容量预算。** BF16 每个 token 全 28 层 KV 为 112 KiB，双 scale INT8（包括 FP32 scales）为 59.5 KiB，减少 46.875%。项目把同一 90% 显存预算用于更多 KV 块，所以主要表现为 701→1320 块（+88.30%），而非已分配显存减半。显存数字是 PyTorch 分配器的峰值，没有测 NVML 进程峰值。", "",
    "## 指标边界", "",
    "- `LLMEngine.generate → step → Scheduler.schedule → ModelRunner.run → Scheduler.postprocess`。",
    "- TTFT 从批量提交到首 token 在 CPU postprocess 回填完成，包含排队和准备；不是 GPU 内核的首 token event 时间。",
    "- TPOT=`(t_last−t_first)/(output_tokens−1)`，先逐请求计算再取分布；ITL 是所有相邻 token 时间差；请求总延迟=`t_last−t_batch_start`。",
    "- 吞吐统一使用完整 generate 墙钟时间，模型加载、初始化与预热均排除。阶段执行吞吐使用各阶段 model-run 的累计时间。",
    "- 这是随机 token ID 的离线批量负载，不是在线服务测量；本轮测性能与长度，不评价量化后的自然语言回答质量。", "",
    "## 每轮原始记录", "",
    "| 模式 | 轮次 | 总时间 s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 日志 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in data["trials"]:
    lines.append(f"| {row['mode']} | {row['trial']} | {row['elapsed_s']:.3f} | {row['output_tokens_per_s']:.2f} | "
                 f"{row['latency_ms']['TTFT']['p50']:.2f} | {row['latency_ms']['TPOT']['p50']:.2f} | [{row['log']}]({row['log']}) |")
lines += ["", "## 复现", "", "```bash",
    "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_kv_metrics_comparison.py --outdir /tmp/kv_metrics_packed_vs_bf16_recheck",
    "```", "", "复现应使用本报告 source_snapshot 对应的代码状态；当前实验的每轮启动和结束均检查了 21 个源文件的哈希。",
    "当前 attention 内核 SHA256：`" + manifest["source_hashes"]["nanovllm/layers/quantized_attention.py"] + "`。", "",
]
(ROOT / "report.md").write_text("\n".join(lines))
with (ROOT / "comparison.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=["metric", "statistic", "unit", "bf16", "int8_half", "change_percent"])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Output throughput: {bf16['output_tokens_per_s']:.2f} -> {int8['output_tokens_per_s']:.2f}, {output_delta:+.2f}%")
for metric in ("TTFT", "TPOT", "ITL", "End-to-end latency"):
    a, b = bf16["latency_ms"][metric], int8["latency_ms"][metric]
    print(metric, "BF16", a, "INT8", b)
print("Report:", ROOT / "report.md")
