"""A small, repeatable CUDA MLP benchmark and profiling workload.

See README.md in this directory for the three separate runs.
"""

import argparse
import json
import shutil
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


PHASES = ("zero_grad", "forward", "loss", "backward", "optimizer_step", "scaler_update")


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 num_layers: int):
        super().__init__()
        dimensions = [in_features] + [hidden_features] * (num_layers - 1) + [out_features]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(num_layers))
        self.relu = nn.ReLU()
        self.annotate_nvtx = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.annotate_nvtx:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i + 1 < len(self.layers):
                    x = self.relu(x)
            return x

        for i, layer in enumerate(self.layers):
            with torch.cuda.nvtx.range(
                f"layer_{i + 1:02d} [batch={x.shape[0]}, "
                f"{layer.in_features}->{layer.out_features}]"):
                with torch.cuda.nvtx.range(
                    f"linear_{i + 1:02d} [input={tuple(x.shape)}, "
                    f"weight={tuple(layer.weight.shape)}, bias={layer.bias is not None}]"):
                    x = layer(x)
                if i + 1 < len(self.layers):
                    with torch.cuda.nvtx.range(
                        f"relu_{i + 1:02d} [shape={tuple(x.shape)}]"):
                        x = self.relu(x)
        return x


def train_step(model: MLP, optimizer: torch.optim.Optimizer,
               scaler: torch.amp.GradScaler, x: torch.Tensor,
               target: torch.Tensor, dtype: torch.dtype,
               scope=nullcontext) -> torch.Tensor:
    with scope("zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with scope("forward"):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
            output = model(x)
    with scope("loss"):
        loss = torch.nn.functional.mse_loss(output.float(), target)
    with scope("backward"):
        scaler.scale(loss).backward()
    with scope("optimizer_step"):
        scaler.step(optimizer)
    if scaler.is_enabled():
        with scope("scaler_update"):
            scaler.update()
    return loss


def warmup(model: MLP, optimizer: torch.optim.Optimizer,
           scaler: torch.amp.GradScaler, x: torch.Tensor,
           target: torch.Tensor, dtype: torch.dtype, steps: int) -> None:
    for _ in range(steps):
        train_step(model, optimizer, scaler, x, target, dtype)
    torch.cuda.synchronize()


def run_benchmark(model: MLP, optimizer: torch.optim.Optimizer,
                  scaler: torch.amp.GradScaler, x: torch.Tensor,
                  target: torch.Tensor, dtype: torch.dtype,
                  args: argparse.Namespace) -> None:
    warmup(model, optimizer, scaler, x, target, dtype, args.warmup)
    gpu_ms, wall_ms = [], []
    for _ in range(args.steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        loss = train_step(model, optimizer, scaler, x, target, dtype)
        end.record()
        end.synchronize()
        gpu_ms.append(start.elapsed_time(end))
        wall_ms.append((time.perf_counter() - wall_start) * 1000)

    batch = x.shape[0]
    dimensions = [args.in_features] + [args.hidden_features] * (args.layers - 1) + [args.out_features]
    # Three matrix multiplies per Linear: forward, input gradient, weight gradient.
    linear_flops = 6 * batch * sum(a * b for a, b in zip(dimensions[:-1], dimensions[1:]))
    result = {
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "compute_dtype": args.dtype,
        "parameter_dtype": "float32",
        "optimizer": "AdamW (foreach=True)",
        "shape": [batch, args.in_features, args.hidden_features, args.out_features],
        "linear_layers": args.layers,
        "warmup": args.warmup,
        "measured_steps": args.steps,
        "cuda_event_ms_per_step": gpu_ms,
        "cuda_event_median_ms": statistics.median(gpu_ms),
        "wall_ms_per_step": wall_ms,
        "wall_median_ms": statistics.median(wall_ms),
        "samples_per_second": batch * 1000 / statistics.median(wall_ms),
        "final_loss": loss.detach().item(),
        "estimated_linear_train_flops_per_step": linear_flops,
        "estimated_linear_train_tflops_from_cuda_event":
            linear_flops / statistics.median(gpu_ms) / 1e9,
    }
    path = args.outdir / "training_benchmark.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"CUDA event median: {result['cuda_event_median_ms']:.4f} ms/train step")
    print(f"Wall median:       {result['wall_median_ms']:.4f} ms/train step")
    print(f"Throughput:        {result['samples_per_second']:.0f} samples/s")
    print(f"Final loss:        {result['final_loss']:.6f}")
    print(f"Result:            {path}")


def run_torch_profiler(model: MLP, optimizer: torch.optim.Optimizer,
                       scaler: torch.amp.GradScaler, x: torch.Tensor,
                       target: torch.Tensor, dtype: torch.dtype,
                       args: argparse.Namespace) -> None:
    warmup(model, optimizer, scaler, x, target, dtype, args.warmup)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(args.steps):
            with record_function("MLP_train_step"):
                train_step(model, optimizer, scaler, x, target, dtype, record_function)
    torch.cuda.synchronize()

    print_profiler_table(prof)


def duration(us: float) -> str:
    return f"{us / 1000:.3f} ms" if us >= 1000 else f"{us:.2f} us"


def print_terminal_table(headers: tuple[str, ...], rows: list[tuple[str, ...]],
                         fill_width: bool = True) -> None:
    columns = list(range(len(headers)))
    terminal_width = max(40, min(shutil.get_terminal_size((100, 24)).columns, 160))

    while True:
        widths = {i: max(len(headers[i]), *(len(row[i]) for row in rows))
                  for i in columns if i != 0}
        name_width = terminal_width - sum(widths.values()) - (3 * len(columns) + 1)
        if name_width >= 16 or len(columns) == 3:
            break
        columns.pop()
    natural_name_width = max(len(headers[0]), *(len(row[0]) for row in rows))
    widths[0] = max(8, name_width if fill_width else min(name_width, natural_name_width))

    def fit_name(name: str) -> str:
        width = widths[0]
        return name if len(name) <= width else name[:width - 3] + "..."

    def line(values: tuple[str, ...]) -> str:
        cells = [f"{fit_name(values[i]):<{widths[i]}}" if i == 0
                 else f"{values[i]:>{widths[i]}}" for i in columns]
        return "| " + " | ".join(cells) + " |"

    border = "+" + "+".join("-" * (widths[i] + 2) for i in columns) + "+"
    for text in (border, line(headers), border, *(line(row) for row in rows), border):
        print(text)


def print_profiler_table(prof: profile, row_limit: int = 25) -> None:
    averages = prof.key_averages()
    phase_events = {event.key: event for event in averages
                    if event.key in ("MLP_train_step", *PHASES)
                    and event.device_type == torch.autograd.DeviceType.CPU}
    phase_rows = [
        (name, str(phase_events[name].count),
         duration(phase_events[name].cpu_time_total),
         duration(phase_events[name].cpu_time_total / phase_events[name].count))
        for name in ("MLP_train_step", *PHASES) if name in phase_events
    ]
    print("Training phases (CPU ranges; GPU work is asynchronous):")
    print_terminal_table(("Phase", "Calls", "CPU total", "CPU / step"),
                         phase_rows, fill_width=False)

    events = sorted((event for event in averages
                     if event.key not in ("MLP_train_step", *PHASES)),
                    key=lambda event: event.self_device_time_total,
                    reverse=True)[:row_limit]
    rows = [
        (event.key, str(event.count), duration(event.self_device_time_total),
         duration(event.self_cpu_time_total), duration(event.device_time_total))
        for event in events
    ]
    print("\nTop profiler events (ranges, operators, CUDA kernels):")
    print_terminal_table(
        ("Operation / kernel", "Calls", "CUDA self", "CPU self", "CUDA total"), rows)
    print(f"Top {len(rows)} by self CUDA time (profiler adds overhead).")


def run_nsys_region(model: MLP, optimizer: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, x: torch.Tensor,
                    target: torch.Tensor, dtype: torch.dtype,
                    args: argparse.Namespace) -> None:
    # Warm up inside emit_nvtx so its first-op setup stays outside MLP_CAPTURE.
    with torch.autograd.profiler.emit_nvtx(record_shapes=True):
        warmup(model, optimizer, scaler, x, target, dtype, args.warmup)
        model.annotate_nvtx = True
        with torch.cuda.nvtx.range("MLP_CAPTURE"):
            for step in range(args.steps):
                with torch.cuda.nvtx.range(f"MLP_train_step_{step + 1:02d}"):
                    train_step(model, optimizer, scaler, x, target, dtype,
                               torch.cuda.nvtx.range)
            with torch.cuda.nvtx.range("wait_for_GPU"):
                torch.cuda.synchronize()
    print(f"Captured {args.steps} full training steps inside MLP_CAPTURE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "torch-profiler", "nsys"),
                        required=True)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--hidden-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=20,
                        help="number of Linear layers; ReLU follows every layer except the last")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5,
                        help="measured or captured MLP training steps")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()

    if min(args.batch, args.in_features, args.hidden_features, args.out_features,
           args.steps) <= 0 or args.layers < 2 or args.warmup < 0:
        parser.error("dimensions and steps must be positive; layers >= 2; warmup >= 0")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error(f"CUDA device {args.device} is unavailable")

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    device = torch.device("cuda", args.device)
    dtype = getattr(torch, args.dtype)
    model = MLP(args.in_features, args.hidden_features, args.out_features,
                args.layers).to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=128, enabled=dtype == torch.float16)
    x = torch.randn(args.batch, args.in_features, device=device, dtype=torch.float32)
    target = torch.randn(args.batch, args.out_features, device=device, dtype=torch.float32)
    if args.mode == "benchmark":
        args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"GPU={torch.cuda.get_device_name(args.device)}, compute={args.dtype}, "
          f"optimizer=AdamW, layers={args.layers}, steps={args.steps}")
    print(f"shape=[{args.batch}, {args.in_features}, "
          f"{args.hidden_features}, {args.out_features}]")

    if args.mode == "benchmark":
        run_benchmark(model, optimizer, scaler, x, target, dtype, args)
    elif args.mode == "torch-profiler":
        run_torch_profiler(model, optimizer, scaler, x, target, dtype, args)
    else:
        run_nsys_region(model, optimizer, scaler, x, target, dtype, args)


if __name__ == "__main__":
    main()
