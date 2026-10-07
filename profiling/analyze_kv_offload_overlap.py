"""Compute observed DMA overlap with model kernels from a Kineto Chrome trace."""

import argparse
import json
from pathlib import Path


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def intersection(a, b):
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    events = json.loads(args.trace.read_text())["traceEvents"]
    large = [e for e in events if e.get("cat") == "gpu_memcpy" and e["args"].get("bytes", 0) >= 14 * 2**20]
    streams = {e["args"]["stream"] for e in large}
    copies = [e for e in events if e.get("cat") == "gpu_memcpy" and e["args"]["stream"] in streams]
    kernels = [e for e in events if e.get("cat") == "kernel" and e["args"]["stream"] not in streams]
    decode_scopes = [(e["ts"], e["ts"] + e["dur"]) for e in events if e.get("name") == "MODEL_DECODE" and e.get("ph") == "X"]
    decode = [e for e in kernels if any(a <= e["ts"] and e["ts"] + e["dur"] <= b for a, b in decode_scopes)]
    interval = lambda rows: merge([(e["ts"], e["ts"] + e["dur"]) for e in rows])
    all_kernels, decode_kernels = interval(kernels), interval(decode)
    result = dict(scope="profiled 16-request shared-prefix pressure test; GPU DMA intersects actual kernels",
                  copy_streams=sorted(streams), model_kernel_events=len(kernels), decode_kernel_events=len(decode))
    for direction in ("HtoD", "DtoH"):
        rows = [e for e in copies if direction in e["name"]]
        intervals = interval(rows)
        duration = sum(b - a for a, b in intervals)
        overlap = intersection(intervals, decode_kernels)
        result[direction] = dict(copy_events=len(rows), bytes=sum(e["args"]["bytes"] for e in rows),
                                 dma_ms=duration / 1000, overlap_model_kernel_ms=intersection(intervals, all_kernels) / 1000,
                                 overlap_decode_kernel_ms=overlap / 1000,
                                 overlap_decode_fraction=overlap / duration if duration else 0)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
