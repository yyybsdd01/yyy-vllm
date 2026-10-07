from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """校验采样温度为正，并拒绝近零温度对应的贪心采样。"""
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
