# 零前缀 prefill attention GPU 时间

CUDA Graph + CUDA Event，11 轮随机交错的轮均值中位数；排除写入、编译、模型其他层和 CPU 开销。
Q/K/V 为模型同款 stride=4096 的 BF16 QKV 切片，非连续物理页，BM=16、BN=64、BD=128、4 warps。
Flash BF16 使用原始 K/V；Flash restored 使用恢复成 BF16 的量化 K/V。自写两版输出逐元素一致，并与 restored 参考通过 rtol=0.02、atol=0.005。

| B | L | Flash BF16 μs | Flash restored μs | INT8 原布局 μs | INT8 head/token μs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 13.530 | 13.504 | 32.294 | 31.859 |
| 1 | 1024 | 97.690 | 97.062 | 304.691 | 298.458 |
| 16 | 1024 | 984.013 | 988.774 | 4531.187 | 4428.800 |
| 32 | 512 | 542.131 | 553.254 | 2555.930 | 2502.029 |
