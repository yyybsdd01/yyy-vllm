# FlashAttention 实际启动网格和 warp 数

采集时间：2026-10-05。环境：RTX 3090 Ti（sm86），PyTorch 2.5.1+cu121，flash-attn 2.7.4.post1。

使用与项目一致的接口和张量布局构造独立调用：BF16、Q heads=16、KV heads=8、head_dim=128、page size=256、causal=True、dropout=0。Q/K/V 来自融合 QKV 张量切片，其 token stride 为 4096。普通 prefill 和缓存 prefill 调用 `flash_attn_varlen_func`，decode 调用 `flash_attn_with_kvcache`。

网格和线程数取自 PyTorch profiler 导出的实际 CUDA kernel 事件；tile 参数取自事件中的编译模板名，并与对应版本源码核对。这里测量启动配置，没有进行延迟比较，也没有采集完整模型运行。

## 实际结果

下表 grid 和 block 均按 CUDA 的 `(x,y,z)` 顺序表示。每个 kernel 都使用 `block=(128,1,1)`，即 128 个线程、4 个 warp。

| 调用 | B | Q 长度 | KV 长度 | num_splits 参数 | 主 kernel grid | 额外 combine grid | 主计算 tile (M,N,D) |
|---|---:|---:|---:|---:|---|---|---|
| 普通 prefill | 1 | 1024 | 1024 | 接口无此参数 | `(16,1,16)` | 无 | `(64,64,128)` |
| 普通 prefill | 16 | 1024 | 1024 | 接口无此参数 | `(16,16,16)` | 无 | `(64,64,128)` |
| 缓存 prefill | 8 | 128 | 1024 | 接口无此参数 | `(2,8,16)` | 无 | `(64,128,128)` |
| decode | 256 | 1 | 1024 | 0，自动 | `(1,256,8)` | 无 | `(64,128,128)` |
| decode | 64 | 1 | 1024 | 0，自动 | `(1,64,8)` | 无 | `(64,128,128)` |
| decode | 1 | 1 | 1024 | 0，自动 | `(1,8,8)` | `(4,1,1)` | `(64,128,128)` |
| decode | 1 | 1 | 4096 | 0，自动 | `(1,16,8)` | `(4,1,1)` | `(64,128,128)` |
| decode | 1 | 1 | 1024 | 1，禁用拆分 | `(1,1,8)` | 无 | `(64,128,128)` |

普通 prefill 使用 `flash_fwd_kernel`；缓存 prefill 和 decode 使用 `flash_fwd_splitkv_kernel`。kernel 名称包含 splitkv 不代表一定发生 KV 拆分：是否拆分由 `num_splits > 1` 决定，未拆分时也使用同一族 kernel。

## 网格轴的含义

普通 causal prefill 在当前 GPU 和 D=128 下使用 M=64、N=64：

```text
grid = (ceil(max_seqlen_q / 64), B, Q_HEADS)
        Q 行分块                  请求 Q head
```

缓存 prefill 的 paged KV 路径使用 M=64、N=128，未拆分时网格公式相同。M 是一个 block 的 Q 行 tile；N 是每次循环读取的 K/V token tile。KV 长度通常影响 block 内的循环次数，不直接成为普通 prefill 的网格轴。

decode 在当前 GQA 配置下，将 Q 的两个 group 变成两行：

```text
Q: [B,1,16,128] -> [B,2,8,128]
```

因此 CUDA 参数中的有效 head 数是 8，有效 Q 长度是 2。未拆分时：

```text
grid = (ceil(2 / 64), B, 8) = (1,B,8)
```

每个 block 处理一个请求、一个 KV head 及其对应的两个 Q head。GQA 变换有条件：这里满足单 token Q、无限制窗口、无 ALiBi 等源码条件。

自动选择 S>1 份 KV 时：

```text
grid = (ceil(effective_seqlen_q / 64), S, B * effective_heads)
        Q 行分块                       KV 分片 请求/head 合并索引
```

小 batch decode 的实测 S 分别是 8（KV=1024）和 16（KV=4096）。随后归并每一行的各分片结果。D=128 时 combine 每个 block 处理 4 行：

```text
combine_grid = (ceil(B * effective_heads * effective_seqlen_q / 4), 1, 1)
```

这里 B=1、effective_heads=8、effective_seqlen_q=2，所以 combine grid 是 `(4,1,1)`，同样每 block 4 个 warp。JSON 中 combine 的 `traits` 是继承的主 kernel traits，不能将其中的 M=64 当作 combine 的行 tile。

## 与当前 Triton kernel 的配置对照

当前 `_int8_paged_attention_kernel` 使用 M=16、N=64、D=128、4 个 warp。普通长度 1024、B=16 的 Q 网格若走该 Triton prefill 路径是 `(64,16,16)`，对应 FlashAttention 为 `(16,16,16)`：Q 分块数相差 4 倍。两者每个 block 的 warp 数相同，但 tile、寄存器、共享内存和计算安排不同，不能仅凭 block 数判断性能。

当前 Triton decode 网格 `(1,B,8)` 与未拆分的 FlashAttention 一致；它没有这里的小 batch 自动 KV 拆分与归并。

## 源码和复现

对应版本官方源码下载在 `source/`，URL 与 SHA256 记录在 `source/manifest.json`：

- `flash_fwd_launch_template.h:63-64`：普通 grid；`:91`：线程数。
- `flash_fwd_launch_template.h:106-107`：splitkv grid；`:136-156`：combine；`:163-174`：paged/splitkv tile；`:225-241`：当前 GPU 的 D=128 普通 prefill tile。
- `kernel_traits.h:63-64`：线程数 = warp 数 × 32。
- `flash_api.cpp:1272-1284`：decode 单 token 的 causal 处理与 GQA reshape。
- `flash_api.cpp:263-321`：自动 split 数的选择。

官方下载基址：`https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1/csrc/flash_attn`。

采集命令（使用新的输出目录）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/xgd/anaconda3/envs/nanovllm/bin/python \
  profiling/inspect_flashattention_launch.py --outdir /tmp/flashattention-launch-check
```

`launches.json` 保存解析结果；每个调用的原始 Chrome trace 保存在对应的 JSON 文件中。所有采集调用的输出均通过有限值检查。报告中的配置限于所列版本、GPU 与输入条件，FlashAttention 的其他 head_dim 或架构可能选择其他 tile 和 warp 数。
