"""Derive average outstanding requests after their first token from ITL sums."""

import json
from pathlib import Path
import statistics

outdir = Path(__file__).resolve().parent
results = json.loads((outdir / "results.json").read_text())
assert len(results["trials"]) == 9
labels = {"auto": "BF16", "int8": "单 scale INT8", "int8_half": "双 scale INT8"}
derived = {}
for mode in labels:
    rows = [r for r in results["trials"] if r["mode"] == mode]
    assert len(rows) == 3
    values = [r["latency_ms"]["ITL"]["mean"] * (r["output_tokens"] - r["requests"])
              / (1000 * r["elapsed_s"]) for r in rows]
    derived[mode] = dict(trials=values, median=statistics.median(values))

(outdir / "interpretation.json").write_text(json.dumps(derived, indent=2) + "\n")
heading = "## 首 token 之后的未完成请求数量"
lines = [heading, "",
    "INT8 的 TTFT 尾部下降，而 TPOT/ITL 上升，需要结合请求的起止时间理解。",
    "对每个请求，所有 ITL 之和等于末 token 时间减首 token 时间；把所有请求的这个时间相加，再除以整批 generate 时间，",
    "可以估算平均有多少请求已经生成首 token、但还没有生成末 token。这里先逐轮计算，再取三轮中位数。", "",
    "`平均请求数 = ITL_mean_ms × (输出 token 总数 − 请求数) / (1000 × generate_seconds)`", "",
    "| 模式 | 平均已出首 token、尚未完成的请求数 |", "| --- | ---: |"]
for mode, value in derived.items():
    lines.append(f"| {labels[mode]} | {value['median']:.1f} |")
lines += ["", "这个数量包含被抢占后等待重新 prefill 的请求；没有记录每次 decode 的实际 batch size。",
    "原始 ITL 和 generate 时间来自打印值，推导值有四舍五入误差。",
    "当前 Scheduler 优先 prefill，并受可分配 KV 块数限制；INT8 更大的 KV 池让更多请求较早得到首 token。",
    "这些数据与更多请求进入首 token 后的生成阶段、共享 GPU 执行时间的解释一致。",
    "本表反映整个推理引擎的容量和调度行为；单个 attention kernel 的性能需要固定相同 batch、上下文和输入后单独测量。", ""]
report = outdir / "report.md"
body = report.read_text().split(heading)[0].rstrip()
report.write_text(body + "\n\n" + "\n".join(lines))
print(json.dumps({mode: round(v["median"], 1) for mode, v in derived.items()}))
