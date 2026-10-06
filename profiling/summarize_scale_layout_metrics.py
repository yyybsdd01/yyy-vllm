"""Render completed full-model metrics for the two dual-scale layouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    out = args.outdir.resolve()
    data = json.loads((out/"results.json").read_text())
    manifest = json.loads((out/"manifest.json").read_text())
    assert "finished_utc" in manifest
    assert len(data["trials"])==2*manifest["runs_per_layout"]
    assert len({row["output_sha256"] for row in data["trials"]})==1
    assert len({row["blocks"] for row in data["trials"]})==1
    assert all(row["graph"] and row["mode"]=="int8_half" for row in data["trials"])
    assert all(row["input_tokens"]==manifest["input_tokens"] and
               row["output_tokens"]==manifest["output_tokens"] for row in data["trials"])
    for layout, hashes in manifest["variant_source_hashes"].items():
        for relative, sha in hashes.items():
            assert hashlib.sha256((out/"variants"/layout/relative).read_bytes()).hexdigest()==sha
    a,b = [data["medians"][name] for name in ("token_head","head_token")]
    delta = lambda x,y: (y/x-1)*100
    lines = [
        "# 双 scale 存储布局的完整模型指标：token/head 与 head/token", "",
        f'新 `[block, head, token, 2]` 布局相对当前 `[block, token, head, 2]`：'
        f'输出吞吐 **{delta(a["output_tokens_per_s"],b["output_tokens_per_s"]):+.2f}%**，'
        f'TTFT P50 **{delta(a["latency_ms"]["TTFT"]["p50"],b["latency_ms"]["TTFT"]["p50"]):+.2f}%**，'
        f'TPOT P50 **{delta(a["latency_ms"]["TPOT"]["p50"],b["latency_ms"]["TPOT"]["p50"]):+.2f}%**。'
        '本次没有观察到完整推理的吞吐收益。', "",
        "## 实验实现与条件", "",
        "- 原仓库所有 `nanovllm/*.py` 和原 `benchmark_inference_metrics.py` 未修改，逐次启动/结束均校验 SHA256。",
        "- 两份独立源码位于 `variants/token_head/` 和 `variants/head_token/`。后者仅改变 scale 分配形状，并在 `Attention.forward` 调用新增的写入与 attention 接口。原量化、原模型结构和调度配置保留；没有把新布局接入当前生产工作树。",
        "- 两者都是 INT8 K/V、每 token/head 的 K/V 各两个 FP32 scale；模型权重为 BF16。",
        "- Qwen3-0.6B，RTX 3090 Ti，GPU 0，单卡；CUDA Graph 开启；max_model_len=4096、max_num_batched_tokens=16384、max_num_seqs=512、block_size=256、gpu_memory_utilization=0.9。",
        "- 256 个请求同时到达，随机 token-ID 输入/输出长度都在 100–1024。每轮输入 142,827 token、输出 133,966 token；temperature=0.6、ignore_eos=True，逐请求输出长度校验通过。",
        f'- 每种布局各 {manifest["runs_per_layout"]} 个独立进程，按 token/head→head/token / head/token→token/head / token/head→head/token 交替。模型初始化与生成预热不计入正式 generate 时间；GPU 没有锁频。',
        "- Python 负载 seed=0。两份基准副本均额外固定 torch seed=0，生成预热之后再次设 seed=0；这是相对原基准的共同测量改动，原基准文件未修改。六轮输出 token 的 SHA256 完全相同，因此比较相同生成内容和相同工作量。",
        "- 先按每轮请求计算分布，再对每项统计量取三轮中位数；P95/P99 是请求/token 分布的分位数，不是三轮结果的分位数。",
        f'- KV 容量均为 {a["blocks"]:.0f} 块，prefill 执行 token 均为 {a["stages"]["prefill"]["tokens"]:.0f}，decode 均为 {a["stages"]["decode"]["tokens"]:.0f}。与 BF16/INT8 容量对照不同，本次没有容量和重算工作量差异。',
        "- 正式测量前，两种布局各用 4 请求的完整模型 smoke 检查通过，生成 token 一致。完整源码、运行配置和日志均留在此目录。", "",
        "## 延迟分布", "",
        "单位 ms。TTFT 与请求总延迟为 ms/request；TPOT 为每请求平均 ms/token；ITL 为相邻 token 的 ms/token。", "",
        "| 指标 | 统计量 | 当前 token/head | 新 head/token | 相对变化 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    csv_rows = []
    for metric in ("TTFT","TPOT","ITL","End-to-end latency"):
        for q in ("mean","p50","p95","p99"):
            x,y = a["latency_ms"][metric][q],b["latency_ms"][metric][q]
            lines.append(f"| {metric} | {q.upper()} | {x:.2f} | {y:.2f} | {delta(x,y):+.2f}% |")
            csv_rows.append(dict(metric=metric,statistic=q,unit="ms",token_head=x,head_token=y,change_pct=delta(x,y)))
    lines += ["","## 吞吐、阶段执行与显存","",
              "| 指标 | 当前 token/head | 新 head/token | 相对变化 |",
              "| --- | ---: | ---: | ---: |"]
    for title,key,digits,unit in [
        ("总 generate 时间 s","elapsed_s",3,"s"),
        ("请求吞吐 request/s","requests_per_s",2,"request/s"),
        ("输入 token/s","input_tokens_per_s",2,"token/s"),
        ("输出 token/s","output_tokens_per_s",2,"token/s"),
        ("输入+输出 token/s","total_tokens_per_s",2,"token/s"),
        ("KV blocks","blocks",0,"block"),
        ("峰值 PyTorch allocated GiB","peak_allocated_gib",2,"GiB"),
        ("峰值 PyTorch reserved GiB","peak_reserved_gib",2,"GiB"),
    ]:
        x,y=a[key],b[key]
        lines.append(f"| {title} | {x:.{digits}f} | {y:.{digits}f} | {delta(x,y):+.2f}% |")
        csv_rows.append(dict(metric=key,statistic="run_median",unit=unit,token_head=x,head_token=y,change_pct=delta(x,y)))
    for stage in ("prefill","decode"):
        for key,unit,digits in [("tokens_per_s","token/s",2),("tokens","token",0),("seconds","s",3)]:
            x,y=a["stages"][stage][key],b["stages"][stage][key]
            lines.append(f"| {stage} model-run {key} | {x:.{digits}f} | {y:.{digits}f} | {delta(x,y):+.2f}% |")
            csv_rows.append(dict(metric=stage+"_"+key,statistic="run_median",unit=unit,token_head=x,head_token=y,change_pct=delta(x,y)))
    lines += ["","## 每轮结果与波动","",
              "| 布局 | 轮次 | generate s | 输出 token/s | TTFT P50 ms | TPOT P50 ms | 日志 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in data["trials"]:
        lines.append(f'| {row["layout"]} | {row["trial"]} | {row["elapsed_s"]:.3f} | '
                     f'{row["output_tokens_per_s"]:.2f} | {row["latency_ms"]["TTFT"]["p50"]:.2f} | '
                     f'{row["latency_ms"]["TPOT"]["p50"]:.2f} | [{row["log"]}]({row["log"]}) |')
    for layout in ("token_head","head_token"):
        rows=[r for r in data["trials"] if r["layout"]==layout]
        values=[r["output_tokens_per_s"] for r in rows]
        lines += ["",f'{layout} 三轮输出吞吐范围：{min(values):.2f}–{max(values):.2f} token/s。']
    lines += ["","## 指标边界与解释","",
        "1. **TTFT 主要体现 prefill 和排队。** 普通首次 prefill 两种布局都调用 FlashAttention，布局只影响它的缓存写入；新增 attention kernel 主要在 decode 和带缓存前缀的 prefill 使用。TTFT 不会直接对应上一轮单个 decode kernel 的速度变化。",
        "2. **吞吐接近，未观察到布局带来的整体收益。** 两种布局的 KV 容量、生成 token 和阶段 token 数一致，隔离了前次 BF16/INT8 对比中的容量差异。约 1% 或更小的变化需结合三轮波动解释，当前结果只适用于此模型、GPU 和离线负载。",
        "3. **ITL/TPOT/总时间的分位数可以朝不同方向变化。** TPOT 先逐请求平均，ITL 混合所有 token 间隔，请求总时间还含首 token 等待；少量不同 batch/上下文阶段的变化会被不同权重放大。",
        "4. **显存基本相同。** 两种布局存储相同数量的 INT8 K/V 和 FP32 scale，布局变化不改变容量。记录的是 PyTorch 分配器峰值，没有采集 NVML 进程峰值。",
        "5. **延续项目基准的测量边界。** TTFT 从整批请求提交到 CPU `Scheduler.postprocess` 回填首 token 完成，包含排队/准备；TPOT=(最后−首 token)/(输出长度−1)，ITL 为相邻时间戳差。吞吐以完整 generate 墙钟时间计算。阶段吞吐以 `ModelRunner.run` 累计调用时间计算，含准备、模型、采样和同步。该负载为离线随机 token-ID 批量测量，不是在线服务延迟。",
        "6. **模型行为校验。** 六轮生成 token 摘要一致，说明本次负载中新布局保持当前 INT8 模型行为；它不评价 INT8 相对 BF16 的语言质量。", "",
        "## 复现","","```bash",
        "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/run_scale_layout_metrics.py --outdir /tmp/scale_layout_inference_recheck",
        "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/summarize_scale_layout_metrics.py --outdir /tmp/scale_layout_inference_recheck",
        "```","",
        "产物：`manifest.json`、`workload.json`、`results.json`、`comparison.csv`、两份独立 `variants/` 源码、2 个 smoke 日志和 6 个正式日志。原仓库源码和两份副本的哈希都在 manifest 中。",
    ]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    with (out/"comparison.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=["metric","statistic","unit","token_head","head_token","change_pct"])
        writer.writeheader();writer.writerows(csv_rows)
    for name in ("token_head","head_token"):
        print(name,json.dumps(data["medians"][name],ensure_ascii=False))
    print("REPORT",out/"report.md")


if __name__=="__main__":
    main()
