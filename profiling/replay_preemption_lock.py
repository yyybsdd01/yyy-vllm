"""CPU replay of scheduling decisions with synthetic per-request output tokens.

This diagnoses admission and KV allocation; it does not measure model latency
or reproduce the model's sampled output tokens.
"""

import argparse
from collections import Counter
import hashlib
from itertools import count
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def replay(workload, blocks, enabled):
    Sequence.counter = count()
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=512, max_num_batched_tokens=16384, eos=-1,
        kvcache_block_size=256, num_kvcache_blocks=blocks, preemption_lock=enabled,
    ))
    requests = [Sequence(prompt, SamplingParams(max_tokens=n, ignore_eos=True))
                for prompt, n in zip(workload["prompts"], workload["max_tokens"])]
    for seq in requests:
        scheduler.add(seq)
    preemptions = Counter()
    request_events = {seq.seq_id: [] for seq in requests}
    step = 0
    new_preemptions = []
    original_preempt = scheduler.preempt

    def record_preempt(seq):
        preemptions[seq.seq_id] += 1
        event = dict(step=step, event="preempt", held_blocks=len(seq.block_table),
                     needed_blocks=seq.num_blocks,
                     free_before=len(scheduler.block_manager.free_block_ids))
        request_events[seq.seq_id].append(event)
        new_preemptions.append(event)
        original_preempt(seq)
        event["free_after_release"] = len(scheduler.block_manager.free_block_ids)

    scheduler.preempt = record_preempt
    trace = hashlib.sha256()
    stats = Counter()
    while not scheduler.is_finished():
        step += 1
        new_preemptions.clear()
        lock_before = scheduler.lock
        if scheduler.lock == 1 and scheduler.running and scheduler.waiting:
            stats["locked_steps_with_running_and_waiting"] += 1
            head = scheduler.waiting[0]
            eligible = bool(head.block_table) or scheduler.block_manager.can_allocate(head) >= 0
            stats["locked_steps_with_eligible_prefill"] += int(eligible)
        batch, prefill = scheduler.schedule()
        for event in new_preemptions:
            event["free_after_schedule"] = len(scheduler.block_manager.free_block_ids)
        if prefill:
            for seq in batch:
                request_events[seq.seq_id].append(dict(step=step, event="prefill",
                                                       lock_before=lock_before,
                                                       cached_tokens=seq.num_cached_tokens,
                                                       new_tokens=seq.num_scheduled_tokens))
        stats["prefill_steps" if prefill else "decode_steps"] += 1
        record = [prefill, [[seq.seq_id, seq.num_cached_tokens, seq.num_scheduled_tokens,
                            seq.block_table] for seq in batch]]
        trace.update(json.dumps(record, separators=(",", ":")).encode() + b"\n")
        scheduler.postprocess(batch, [30000 + seq.seq_id for seq in batch], prefill)
        for seq in batch:
            if seq.is_finished:
                request_events[seq.seq_id].append(dict(step=step, event="finish"))
        assert sum(stats[k] for k in ("prefill_steps", "decode_steps")) < 100000
    assert all(seq.num_completion_tokens == seq.max_tokens for seq in requests)
    assert not scheduler.block_manager.used_block_ids and scheduler.lock == 0
    most = max(requests, key=lambda seq: preemptions[seq.seq_id])
    return dict(stats, preemptions=sum(preemptions.values()),
                per_request_histogram=dict(sorted(Counter(preemptions[seq.seq_id] for seq in requests).items())),
                most_preempted_request=dict(request_index=most.seq_id, count=preemptions[most.seq_id],
                                            events=request_events[most.seq_id]),
                schedule_sha256=trace.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.outdir / "manifest.json").read_text())
    measurements = json.loads((args.outdir / "results.json").read_text())
    workload = json.loads((args.outdir / "workload_1024.json").read_text())
    results = dict(scope="CPU scheduler replay; synthetic output tokens; no model timing", variants={})
    for variant in ("bf16", "fast_int8_half"):
        medians = measurements["medians"][variant]
        blocks = int(medians["original"]["blocks"])
        values = {policy: replay(workload, blocks, policy == "lock") for policy in ("original", "lock")}
        for policy in values:
            assert values[policy]["preemptions"] == medians[policy]["preemptions"]
            for trial in measurements["trials"]:
                if trial["variant"] == variant and trial["policy"] == policy:
                    assert {str(k): v for k, v in values[policy]["per_request_histogram"].items()} == trial["request_preemption_stats"]["histogram"]
        results["variants"][variant] = dict(blocks=blocks, policies=values,
                                             identical_schedule=values["original"]["schedule_sha256"] == values["lock"]["schedule_sha256"])
    results["measured_source_hashes"] = manifest["variant_source_hashes"]
    (args.outdir / "scheduler_replay.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, value in results["variants"].items():
        lock = value["policies"]["lock"]
        print(name, "identical_schedule", value["identical_schedule"],
              "locked_steps", lock["locked_steps_with_running_and_waiting"],
              "eligible_prefill", lock["locked_steps_with_eligible_prefill"])
        print("most_preempted_request", json.dumps(lock["most_preempted_request"]))


if __name__ == "__main__":
    main()
