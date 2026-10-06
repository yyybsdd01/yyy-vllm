# example.py 计时（2026-10-02）

运行原有两条聊天提示（`introduce yourself`、`list all prime numbers within 100`），Qwen3-0.6B，RTX 3090 Ti，单卡，eager 模式，temperature 0.6，max_tokens 256，seed 0。`auto` 和融合 `int8` 各在全新进程运行两次；未在计时前额外预热请求。

`example.py` 使用 `perf_counter` 测量 tokenizer 加载、模型初始化与预热、`generate()` 墙钟时间，并在生成前后调用 `torch.cuda.synchronize()`。进程墙钟使用 `/usr/bin/time`，包含 Python 启动、导入和退出。输出吞吐是实际输出 token 数除以 `generate()` 时间。

| 模式 | 轮次 | 模型初始化与预热 | 生成耗时 | 实际输出 token | 输出吞吐 | 完整进程墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto | 1 | 3.986 s | 8.324 s | 450 | 54.06 token/s | 未单独记录 |
| 融合 int8 | 1 | 3.877 s | 8.841 s | 426 | 48.18 token/s | 未单独记录 |
| auto | 2 | 见日志 | 8.193 s | 450 | 54.93 token/s | 16.34 s |
| 融合 int8 | 2 | 见日志 | 8.833 s | 426 | 48.23 token/s | 16.98 s |

两个模式都按 EOS 正常停止，输出 token 数不同；这是只有两个请求、未额外预热的交互式示例，结果不能直接作为等长度或大批量吞吐对比。此前固定 256 请求、输出长度一致的对比见[融合 KV 基准](kv_cache_fused_comparison_2026-10-02.md)。

- [auto 第 1 轮](example_timing_2026-10-02/auto.txt)、[融合 int8 第 1 轮](example_timing_2026-10-02/int8.txt)
- [auto 第 2 轮及进程墙钟](example_timing_2026-10-02/auto_process_wall.txt)、[融合 int8 第 2 轮及进程墙钟](example_timing_2026-10-02/int8_process_wall.txt)
