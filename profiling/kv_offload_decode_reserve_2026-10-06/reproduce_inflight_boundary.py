"""Reproduce the one-round reservation limit without using a GPU."""

import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nanovllm.engine.kv_offload import KVOffloadManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

FakeTransport = runpy.run_path(str(ROOT / "tests/test_kv_offload.py"))["FakeTransport"]
s = Scheduler(SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=4096,
                             eos=-1, kvcache_block_size=256,
                             num_kvcache_blocks=4, preemption_lock=False))
t = FakeTransport()
s.offload = KVOffloadManager(s.block_manager, t, cpu_gb=32 * 1024 / 1024**3)
a, c, b = [Sequence([value] * length, SamplingParams(max_tokens=20, ignore_eos=True))
           for value, length in ((1, 255), (2, 1), (3, 300))]


def compute(batch, prefill):
    for seq in batch:
        end = seq.num_cached_tokens + seq.num_scheduled_tokens
        for i in range(seq.num_cached_tokens // 256, (end + 255) // 256):
            t.memory[seq.block_table[i]] = tuple(seq.token_ids[i * 256:min(end, (i + 1) * 256)])
    s.postprocess(batch, [11 + seq.seq_id for seq in batch], prefill)


for seq in (a, c, b):
    s.add(seq)
batch, prefill = s.schedule()
compute(batch, prefill)
s.running.remove(b)
s.preempt(b)
t.advance()
# Invalidate B's old physical pages to require asynchronous H2D restoration.
ids = [s.block_manager._allocate_block() for _ in range(2)]
for gid in ids:
    t.memory[gid] = (999,)
    s.block_manager.blocks[gid].ref_count -= 1
    s.block_manager._deallocate_block(gid)

reserve_at_submission = s._decode_block_reserve()
assert reserve_at_submission == 0 and a.num_tokens == 256
batch, prefill = s.schedule()
assert not prefill and b.status == SequenceStatus.RESTORING and b not in batch
compute(batch, prefill)
reserve_after_decode = s._decode_block_reserve()
assert reserve_after_decode == 1 and a.num_tokens == 257
t.advance()
batch, prefill = s.schedule()
assert not prefill and b not in batch and b.status == SequenceStatus.OFFLOADING
assert b.num_completion_tokens == 1
result = dict(scope="CPU state-machine reproduction; no performance estimate",
              reserve_at_restore_submission=reserve_at_submission,
              reserve_after_intervening_decode=reserve_after_decode,
              restored_B_generated_new_tokens=False,
              B_preempted_again_before_model_step=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
