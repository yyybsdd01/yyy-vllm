"""Export measured kernel counters, source samples, and a reproducible report."""
import collections
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, "/usr/local/cuda-12.4/nsight-compute-2024.1.0/extras/python")
import ncu_report

OUT = Path(__file__).resolve().parent
timings = json.loads((OUT / "timings.json").read_text())
report = ncu_report.load_report(str(OUT / "ncu_formats.ncu-rep"))
names = ["flash_bf16", "triton_bf16", "triton_int8_half"]
metrics = {
    "duration_ns": "gpu__time_duration.sum",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
    "dram_bytes": "dram__bytes.sum",
    "warp_instructions": "smsp__inst_executed.sum",
    "conversion_thread_instructions": "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
    "fp32_thread_instructions": "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_dynamic_bytes": "launch__shared_mem_per_block_dynamic",
    "shared_driver_bytes": "launch__shared_mem_per_block_driver",
    "cta_limit_shared": "launch__occupancy_limit_shared_mem",
    "long_scoreboard_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "barrier_per_issue": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "short_scoreboard_per_issue": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "warp_latency_per_issue": "smsp__average_warp_latency_per_inst_issued.ratio",
}
counters = {}
samples = {}
for idx, name in enumerate(names):
    action = report.range_by_idx(0).action_by_idx(idx)
    counters[name] = {label: action.metric_by_name(metric).as_double()
                      for label, metric in metrics.items()}
    row = counters[name]
    row["dram_GB_per_second"] = row["dram_bytes"] / row["duration_ns"]
    row["dram_read_MiB"] = row["dram_read_bytes"] / 2**20
    metric = action.metric_by_name("smsp__pcsamp_warps_issue_stalled_long_scoreboard")
    ids = metric.correlation_ids()
    locations = collections.Counter()
    ops = collections.Counter()
    pcs = []
    for inst in range(metric.num_instances()):
        pc = ids.as_uint64(inst)
        info = action.source_info(pc)
        sass = action.sass_by_pc(pc)
        count = metric.as_uint64(inst)
        if sass:
            tokens = sass.split()
            op = tokens[1] if tokens[0].startswith("@") else tokens[0]
            ops[op] += 1
        if count:
            file, line = (info.file_name(), info.line()) if info else ("", -1)
            locations[(file, line)] += count
            pcs.append(dict(pc=pc, samples=count, file=file, line=line, sass=sass))
    samples[name] = dict(total=metric.as_uint64(),
                         by_source=[dict(file=f, line=l, samples=n)
                                    for (f, l), n in locations.most_common()],
                         pcs=sorted(pcs, key=lambda x: -x["samples"]),
                         static_sass_opcodes=dict(ops))
(OUT / "ncu_summary.json").write_text(json.dumps(counters, indent=2) + "\n")
(OUT / "long_scoreboard_sources.json").write_text(json.dumps(samples, indent=2) + "\n")

with (OUT / "comparison.csv").open("w") as fp:
    writer = csv.writer(fp)
    writer.writerow(["mode", "batch", "context", "qlen", *[n + "_us" for n in names],
                     "int8_vs_flash_pct", "int8_vs_triton_bf16_pct"])
    for case in timings["cases"]:
        values = [case["variants"][n]["median_us"] for n in names]
        writer.writerow([case[k] for k in ("mode", "batch", "context", "qlen")]
                        + values + [(values[2] / values[j] - 1) * 100 for j in (0, 1)])

lines = [
    "# 原 BF16 FlashAttention 与当前 INT8 paged attention 内核对比（2026-10-04）", "",
    "当前双 scale、成对加载的 INT8 内核，在本次 8 个测例中均慢于项目原 BF16 FlashAttention。主测例 decode B=256、K=1024：原内核 **1152.184 µs**，INT8 **1245.115 µs**，慢 **8.07%**。硬件计数器确认减少了 DRAM 读取，但新增转换、乘法及数据布局开销降低有效吞吐；在同一 Triton attention 结构的 BF16 对照中也观察到回退。", "",
    "## 比较范围与正确性", "",
    "- RTX 3090 Ti；torch 2.5.1+cu121、Triton 3.1.0、FlashAttention 2.7.4.post1。Q heads=16、KV heads=8、D=128，block_size=256，BM=16、BN=64、BD=128，Triton 4 warps。",
    "- 原实现为 `nanovllm/layers/attention.py` 调用的 FlashAttention：decode 使用 `flash_attn_with_kvcache`，cached prefill 使用 `flash_attn_varlen_func`，参数与项目默认一致，保留自动 split-KV。",
    "- INT8 是当前生产 `_int8_paged_attention_kernel`，每个 token/head 的 K、V 各两个独立 FP32 scale，使用合并后的 `[BN,2]` 加载。",
    "- 增设 BF16 Triton 对照：从当前源码生成独立副本，保持页表、tile、在线 softmax、GQA 和矩阵乘结构，直接加载 BF16 K/V，删除 scale 和解量化。它只用于测量，不接入推理。修改输入类型后编译器也会改变布局和调度，因此时间差是整套 quantized 路径的净成本，不能作为独立解量化阶段耗时。",
    "- seed=37；INT8 K/V 取 [-64,64]，两个 scale 独立取 [0.005,0.02)，BF16 参考先按生产内核相同的 FP32 乘法/BF16 舍入恢复。三种路径使用相同 Q 和逻辑上相同的 K/V。8 个形状全部通过 rtol=0.02、atol=0.005；全局最大绝对差 0.00048828125。这是内核算子正确性校验，不是模型量化精度测评。",
    "- 统一长度、连续物理页；cached prefill Q=128、K=1024，含 896 token 前缀。首次无前缀 prefill 仍使用 FlashAttention，不属于此 INT8 kernel 比较。",
    "- CUDA Event + CUDA Graph，每图 8 次 attention，每轮回放 10 次，11 轮随机交错顺序；表中为轮均值的中位数。输入生成、恢复、KV 写入、Python 调用、调度、模型其他层均不计入 GPU 耗时。B=1 decode 的 FlashAttention 自动产生 split 与 combine 两个内核，表中计入两者总耗时；其余测例各路径均为一个内核。",
    f'- 生产源码快照 `int8_production_snapshot.py`，SHA256 `{timings["source_sha256"]}`；本轮没有修改生产 attention 源码。', "",
    "## CUDA Graph GPU 时间", "",
    "| 阶段 | B | K | Q/请求 | 原 Flash BF16 µs | 同结构 Triton BF16 µs | 当前 INT8 µs | INT8 比原内核 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for case in timings["cases"]:
    t = [case["variants"][n]["median_us"] for n in names]
    lines.append(f'| {case["mode"]} | {case["batch"]} | {case["context"]} | {case["qlen"]} | '
                 f'{t[0]:.3f} | {t[1]:.3f} | {t[2]:.3f} | +{(t[2]/t[0]-1)*100:.2f}% |')
lines += ["", "完整轮次、最小/最大轮均值、CUDA kernel 名称和校验误差见 `timings.json`。",
          "", "## 主测例的实际访存和指令", "",
          "Nsight Compute 2024.1.1，B=256、K=1024 decode，每个预热后的版本采集一次直接 launch，17 replay passes。cache-control=none、clock-control=none。使用 `sudo -n` 获取硬件计数器权限，未修改驱动配置；计数器采样的 duration 与上述多轮计时分别报告。实际 bytes 来自显式计数器，包含访问事务、Q/页表/结果，不能等同于张量逻辑容量。", "",
          "| 指标 | 原 Flash BF16 | 同结构 Triton BF16 | 当前 INT8 |",
          "| --- | ---: | ---: | ---: |"]
display = [
    ("NCU 单次时间 µs", "duration_ns", 1e-3),
    ("DRAM 读取 MiB", "dram_read_MiB", 1),
    ("DRAM 写入 MiB", "dram_write_bytes", 2**-20),
    ("DRAM 总量 MiB", "dram_bytes", 2**-20),
    ("实际 DRAM 吞吐 GB/s", "dram_GB_per_second", 1),
    ("执行 warp 指令 million", "warp_instructions", 1e-6),
    ("转换类 thread 指令 million", "conversion_thread_instructions", 1e-6),
    ("FP32 thread 指令 million", "fp32_thread_instructions", 1e-6),
    ("动态共享内存 KiB", "shared_dynamic_bytes", 2**-10),
    ("寄存器/线程", "registers_per_thread", 1),
    ("共享内存允许 CTA/SM", "cta_limit_shared", 1),
    ("实际 occupancy %", "occupancy_pct", 1),
    ("long scoreboard / issue", "long_scoreboard_per_issue", 1),
    ("barrier stall / issue", "barrier_per_issue", 1),
]
for label, key, factor in display:
    lines.append(f'| {label} | ' + " | ".join(f'{counters[n][key]*factor:.3f}' for n in names) + " |")
bf, ti = counters["triton_bf16"], counters["triton_int8_half"]
lines += ["", "## 为什么减少读取仍然更慢", "",
    "1. **逻辑容量减少 46.875%，实际读取减少约 37.46%。** BF16 的 K/V 共 1024 MiB；INT8 K/V 512 MiB，加 K/V 两组 FP32 scales 共 32 MiB，即 544 MiB。实际 DRAM read 为 1025.044 → 641.071 MiB。缓存命中、事务粒度、读取放大使实际流量不同于逻辑容量，不能直接按 dtype 比例推算速度。",
    f'2. **不是 INT8 矩阵乘。** 生产代码先 `INT8 → FP32`，乘 scale，再转 BF16，`tl.dot` 使用 BF16 Tensor Core。相对同结构 BF16 对照，执行 warp 指令增加 {(ti["warp_instructions"]/bf["warp_instructions"]-1)*100:.2f}%；转换 thread 指令 18.874 → 824.181 million，FP32 thread 指令 359.662 → 896.532 million。FP32 增量恰为 536.871 million，与本例 K/V 元素数一致。硬件分类包含整个内核指令，不能全当成独立解量化阶段。',
    "3. **解量化和 scale 布局扩大共享内存，削弱访存延迟隐藏。** 同结构 BF16 动态共享内存 39 KiB，INT8 53 KiB；加 1 KiB driver shared 后分别 40/54 KiB。SM 共享内存配置为 100 KiB，因此 CTA 驻留上限 2 → 1，实际 occupancy 16.55% → 8.35%。原 FlashAttention 也只有一个 CTA/SM，却达到更高吞吐，说明 occupancy 本身不能独立解释速度；指令、加载与计算的调度同样重要。",
    "4. **内存吞吐明显下降。** 原 FlashAttention 约 930 GB/s，同结构 BF16 约 940 GB/s，INT8 约 552 GB/s。流量虽减小，但本实现持续读取数据的能力也下降，最终未换到延迟收益。按原内核约 930 GB/s、INT8 实测总事务量估算，仅流量项约需 0.73 ms，而实测为 1.225 ms；两者差值不能作为解量化时间，因为流水线阶段重叠、有效吞吐与编译实现也不同。",
    "5. **当前 packed scale 的等待仍存在。** INT8 long-scoreboard 共 52,922 个采样，归属 scale 加载 91/92 行的共 28,049（53.0%），K load 的 100 行 22,475（42.5%）。源码映射反映 scale/KV load 消费依赖的等待位置，不能解释为这几行占总时间对应比例。相比同结构 BF16，barrier stall/issue 反而下降，long-scoreboard/issue 也下降；INT8 绝对指令数量更大，不能只看某项归一化 stall 指标就判定变快。",
    "6. **形状影响很大。** B=1 decode，当前 Triton 只有 8 个 CTA，原 FlashAttention 自动 split-KV 提供更多并行，故 12.40 vs 43.35 µs。cached prefill 在同一 KV head 的多个 Q tile/head 中重复读取和解量化 K/V；tile、GQA 复用及缓存局部性影响更大，B=8 时原 FlashAttention 164.45 µs、BF16 Triton 262.54 µs、INT8 538.41 µs。这里包含实现差异，不能全部归因于 dtype。", "",
    "主测例中，同结构 BF16 1141.089 µs、INT8 1245.115 µs，净增加 104.026 µs（9.12%）；这是‘节省读取 + 增加解量化 + 布局/并发改变’后的合计差值，而非解量化独占耗时。当前内核的优先优化方向是降低解量化中间布局/共享内存开销、改善 scale 与 KV 的加载调度；小 batch 另需评估 split-KV。", "",
    "## 复现与产物", "", "```bash",
    "/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/compare_attention_kernel_formats.py --outdir /tmp/attention_formats_recheck",
    "```", "",
    "`--quick` 只测 B=256、K=1024 decode；`--profile` 在 CUDA profiler API 区间只启动各版本一次。计数器命令见 `ncu_command.txt`。",
    "原始产物：`timings.json`、`comparison.csv`、`ncu_formats.ncu-rep`、`ncu_raw.csv`、`ncu_details.txt`、`ncu_source.txt`、`ncu_summary.json`、`long_scoreboard_sources.json`、生产源码与 BF16 对照快照。",
]
(OUT / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(counters, indent=2))
