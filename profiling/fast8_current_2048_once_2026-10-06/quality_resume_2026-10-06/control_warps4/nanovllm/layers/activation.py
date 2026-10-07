import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将输入均分为两半，对前半应用 SiLU 后与后半逐元素相乘。"""
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
