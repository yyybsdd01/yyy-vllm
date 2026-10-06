# MLP 训练步骤的基准测试与性能剖析

脚本测量完整训练步骤：`zero_grad → forward → MSE loss → backward → AdamW.step`。默认是 20 个 Linear 层，batch=1024，前 19 层后接 ReLU；预热 20 步，采集 5 步。输入和固定目标在 GPU 上生成一次，均为 FP32。`--dtype float16` 表示前向使用 FP16 autocast，参数和 AdamW 状态保持 FP32，并用 GradScaler 处理反向；`--dtype float32` 不启用 autocast 和 GradScaler。

在仓库根目录运行，使用现有 `nanovllm` Conda 环境：

```bash
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/mlp_profile.py --mode benchmark
/home/xgd/anaconda3/envs/nanovllm/bin/python profiling/mlp_profile.py --mode torch-profiler
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  /opt/nvidia/nsight-systems/2023.4.4/target-linux-x64/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=nvtx --capture-range-end=stop --nvtx-capture=MLP_CAPTURE \
  --output=profiling/output/mlp_training_nsys \
  /home/xgd/anaconda3/envs/nanovllm/bin/python profiling/mlp_profile.py --mode nsys
```

| 模式 | 输出 |
| --- | --- |
| `benchmark` | `profiling/output/training_benchmark.json`：整步 CUDA event、墙钟耗时、吞吐量、末步 loss，以及 Linear 前向与反向矩阵乘法 FLOPs 估计 |
| `torch-profiler` | 终端直接打印阶段 CPU 耗时表和按 CUDA 自耗时排序的算子表；不生成 JSON 或表格文件 |
| `nsys` | `profiling/output/mlp_training_nsys.nsys-rep`：整步 CPU NVTX、PyTorch 算子、CUDA API 与 GPU kernel 时间线 |

`torch-profiler` 的阶段表按 CPU 范围统计。CUDA kernel 异步执行，不要把阶段的 CPU 耗时当成 GPU 耗时；GPU 启动和执行的对应关系请看 nsys。算子表中的嵌套行也不能直接相加。Profiler 和 NVTX 都会增加开销，比较性能时使用 `benchmark` 模式。

## nsys 中看哪些标记

在 GUI 中找到 `MLP_CAPTURE → MLP_train_step_01`，再展开 `zero_grad`、`forward`、`loss`、`backward`、`optimizer_step`、`scaler_update` 和末尾的 `wait_for_GPU`。`forward` 内有 `layer_01` 等逐层范围，每层有带输入和权重形状的 `linear_XX` 与 `relu_XX`。PyTorch 的 `emit_nvtx(record_shapes=True)` 还会标记前向算子、反向 autograd 算子及 AdamW 的 foreach 算子，并附带输入形状。把这些 CPU 范围与 **CUDA API**、**GPU Kernels** 轨道对齐，才能看到发射延迟与实际 GPU 执行。末尾的同步确保所有采集步骤的 GPU 工作进入报告。

`emit_nvtx` 的首次算子可能有一次性初始化开销。脚本现在把预热放在 `emit_nvtx` 内、正式 `MLP_CAPTURE` 前；默认预热 20 步。设置 `--warmup 0` 时，这部分开销可能重新落入正式采集。

## 为什么时间线开头约 15 ms 是空白

仓库原有的 `profiling/output/mlp_nsys.nsys-rep` 是**旧版前向专用报告**。从它导出的 SQLite 中，`MLP_CAPTURE` 在时间轴 **15.690 ms** 才开始，首个 CUDA kernel 在 **16.066 ms** 开始。nsys 命令使用 `--capture-range=nvtx`，因此开始标记之前的设置与预热没有被记录；GUI 的前约 15 ms 看起来就是空白。这段空白不表示 GPU 执行了 15 ms 的空操作，也不是某个 Linear 层的耗时。选中 `MLP_CAPTURE` 并缩放到该范围即可看正式采样。

如果要查看初始化和预热实际做了什么，运行 nsys 时去掉 `--capture-range=nvtx`、`--capture-range-end=stop` 和 `--nvtx-capture=MLP_CAPTURE` 三个参数；这样会采集整个进程，报告也会更大。

原有 `profiling/output/benchmark.json` 中约 10 ms/step 的数值同样来自旧版前向专用脚本，不可与现在的完整训练步骤比较。新结果分别写入 `training_benchmark.json` 和 `mlp_training_nsys.nsys-rep`，保留旧结果。

可用 `--layers`、`--steps`、`--batch`、`--in-features`、`--hidden-features`、`--out-features`、`--dtype`、`--device` 和 `--warmup` 改变实验；比较两次性能时保持形状、精度、GPU 和系统负载一致。
