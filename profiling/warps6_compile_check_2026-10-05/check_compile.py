"""Launch the saved fast INT8 kernel with num_warps=6 without changing the repo."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import torch
import triton

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from profiling.run_current_int8_metrics import source_hashes

manifest = json.loads((HERE / "manifest.json").read_text())
spec = importlib.util.spec_from_file_location("int8_warps6_check", HERE / "quantized_attention_warps6.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

k = torch.zeros((1, 256, 8, 128), device="cuda", dtype=torch.int8)
v = torch.zeros_like(k)
ks = torch.ones((1, 256, 16), device="cuda", dtype=torch.float32)
vs = torch.ones_like(ks)
table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
results = dict(triton_version=triton.__version__, num_warps=6, stages={}, performance_metrics=None)
for stage in ("decode", "prefill"):
    print("ATTEMPT", stage, "num_warps=6", flush=True)
    if stage == "decode":
        q = torch.zeros((1, 16, 128), device="cuda", dtype=torch.bfloat16)
        kwargs = dict(context_lens=torch.tensor([64], device="cuda", dtype=torch.int32))
    else:
        q = torch.zeros((32, 16, 128), device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
        kwargs = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=32)
    try:
        module.int8_paged_attention(q, k, v, ks, vs, table, 128 ** -0.5, **kwargs)
        torch.cuda.synchronize()
    except AssertionError as exc:
        assert str(exc) == "num_warps must be a power of 2", str(exc)
        traceback.print_exc(file=sys.stdout)
        results["stages"][stage] = dict(status="compiler_options_rejected", error=str(exc))
    else:
        raise AssertionError("unexpectedly accepted num_warps=6")

assert source_hashes(ROOT) == manifest["root_source_hashes"]
assert hashlib.sha256(Path(manifest["reference"]).read_bytes()).hexdigest() == manifest["reference_sha256"]
results.update(root_sources_unchanged=True, reference_kernel_unchanged=True)
(HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print("CONFIRMED: both decode and prefill reject num_warps=6; no performance measurements.", flush=True)
