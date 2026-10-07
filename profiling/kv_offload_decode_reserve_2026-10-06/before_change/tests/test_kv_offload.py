"""Exercise ownership, async state transitions and scheduler progress on CPU."""

from types import SimpleNamespace
import unittest

from nanovllm.engine.kv_offload import KVOffloadManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class ManualEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


class FakeTransport:
    block_bytes = 1024

    def __init__(self):
        self.memory = {}
        self.events = []
        self.out_ids = []
        self.in_ids = []

    def new_buffer(self):
        return {}

    def offload(self, gid, buffer):
        buffer["kv"] = self.memory[gid]
        event = ManualEvent()
        self.events.append(event)
        self.out_ids.append(gid)
        return event

    def restore_block(self, gid, buffer, ready_event):
        assert ready_event.query()
        self.memory[gid] = buffer["kv"]
        self.in_ids.append(gid)

    def finish_restore(self, copied):
        event = ManualEvent()
        if not copied:
            event.done = True
        self.events.append(event)
        return event

    def advance(self):
        for event in self.events:
            event.done = True

    def summary(self):
        return dict(d2h_blocks=len(self.out_ids), h2d_blocks=len(self.in_ids))


class KVOffloadTest(unittest.TestCase):
    def scheduler(self, blocks=8, buffers=32, inflight=8, budget=4096):
        scheduler = Scheduler(SimpleNamespace(
            max_num_seqs=16, max_num_batched_tokens=budget, eos=-1,
            kvcache_block_size=256, num_kvcache_blocks=blocks, preemption_lock=False,
        ))
        transport = FakeTransport()
        scheduler.offload = KVOffloadManager(scheduler.block_manager, transport,
                                            cpu_gb=buffers * 1024 / 1024**3, max_inflight=inflight)
        return scheduler, transport

    def sequence(self, tokens, output=20):
        return Sequence(tokens, SamplingParams(max_tokens=output, ignore_eos=True))

    def compute(self, scheduler, transport, batch, prefill):
        for seq in batch:
            # Check every historical valid KV token before this synthetic model step.
            for i in range((seq.num_cached_tokens + 255) // 256):
                length = min(256, seq.num_cached_tokens - i * 256)
                self.assertEqual(transport.memory[seq.block_table[i]][:length],
                                 tuple(seq.token_ids[i * 256:i * 256 + length]))
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            for i in range(seq.num_cached_tokens // 256, (end + 255) // 256):
                transport.memory[seq.block_table[i]] = tuple(seq.token_ids[i * 256:min(end, (i + 1) * 256)])
        scheduler.postprocess(batch, [10000 + seq.token_ids[0] + seq.num_completion_tokens for seq in batch], prefill)

    def prefill(self, scheduler, transport, sequences):
        for seq in sequences:
            scheduler.add(seq)
        batch, phase = scheduler.schedule()
        self.assertTrue(phase)
        self.compute(scheduler, transport, batch, phase)

    def preempt(self, scheduler, seq):
        scheduler.running.remove(seq)
        scheduler.preempt(seq)

    def test_transfer_states_and_no_duplicate_boundary_block(self):
        scheduler, transport = self.scheduler(blocks=4)
        seq = self.sequence([3] * 256)
        self.prefill(scheduler, transport, [seq])
        self.preempt(scheduler, seq)
        self.assertEqual(seq.status, SequenceStatus.OFFLOADING)
        self.assertFalse(scheduler.is_finished())
        self.assertEqual(scheduler.schedule(), ([], False))
        transport.advance()
        batch, phase = scheduler.schedule()
        self.assertFalse(phase)
        self.assertEqual(batch, [seq])
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertEqual(seq.num_cached_tokens, 256)
        self.assertEqual(len(seq.block_table), 2)
        self.compute(scheduler, transport, batch, phase)

    def test_reused_physical_id_does_not_reuse_stale_generation(self):
        scheduler, transport = self.scheduler(blocks=1)
        seq = self.sequence([3])
        self.prefill(scheduler, transport, [seq])
        old_generation = scheduler.block_manager.blocks[0].generation
        self.preempt(scheduler, seq)
        gid = scheduler.block_manager._allocate_block()
        transport.memory[gid] = (999,)
        self.assertGreater(scheduler.block_manager.blocks[0].generation, old_generation)
        scheduler.block_manager.blocks[gid].ref_count -= 1
        scheduler.block_manager._deallocate_block(gid)
        transport.advance()
        self.assertEqual(scheduler.schedule(), ([], False))
        self.assertEqual(seq.status, SequenceStatus.RESTORING)
        self.assertEqual(transport.memory[seq.block_table[0]], (3,))
        self.assertFalse(scheduler.running)
        transport.advance()
        batch, phase = scheduler.schedule()
        self.assertEqual(batch, [seq])
        self.compute(scheduler, transport, batch, phase)

    def test_shared_prefix_copied_only_when_last_gpu_reference_leaves(self):
        scheduler, transport = self.scheduler()
        a = self.sequence([11] * 256 + [12] * 256 + [13] * 256)
        b = self.sequence([11] * 256 + [12] * 256 + [14] * 256)
        self.prefill(scheduler, transport, [a])
        self.prefill(scheduler, transport, [b])
        shared = list(a.block_table[:2])
        self.assertEqual(shared, b.block_table[:2])
        private = a.block_table[-1]
        self.preempt(scheduler, a)
        self.assertEqual(transport.out_ids, [private])
        self.assertTrue(all(scheduler.block_manager.blocks[p].ref_count == 1 for p in shared))
        # Finishing B, rather than preempting B, must preserve A's deferred prefix.
        scheduler.running.remove(b)
        scheduler.block_manager.deallocate(b)
        self.assertEqual(set(transport.out_ids), set(shared + [private]))

    def test_two_offloaded_sequences_share_one_snapshot_and_cpu_copy(self):
        scheduler, transport = self.scheduler()
        a = self.sequence([11] * 256 + [12] * 256 + [13] * 256)
        b = self.sequence([11] * 256 + [12] * 256 + [14] * 256)
        self.prefill(scheduler, transport, [a])
        self.prefill(scheduler, transport, [b])
        self.preempt(scheduler, a)
        self.preempt(scheduler, b)
        ah, bh = (scheduler.offload.handles[q.seq_id] for q in (a, b))
        self.assertIs(ah.blocks[0], bh.blocks[0])
        self.assertEqual(ah.blocks[0].refs, 2)
        self.assertEqual(len(transport.out_ids), 4)
        self.assertEqual(scheduler.offload.allocated_buffers, 4)

    def test_restore_admission_precedes_waiting_prefill_without_running_transfer(self):
        scheduler, transport = self.scheduler(blocks=4)
        a, b = self.sequence([3]), self.sequence([4])
        self.prefill(scheduler, transport, [a])
        self.preempt(scheduler, a)
        # Occupy/overwrite the old page to force actual CPU restoration.
        bm = scheduler.block_manager
        while bm.free_block_ids[0] != 0:
            gid = bm._allocate_block()
            bm.blocks[gid].ref_count -= 1
            bm._deallocate_block(gid)
        gid = bm._allocate_block()
        transport.memory[gid] = (999,)
        bm.blocks[gid].ref_count -= 1
        bm._deallocate_block(gid)
        transport.advance()
        scheduler.add(b)
        batch, phase = scheduler.schedule()
        self.assertTrue(phase)
        self.assertEqual(batch, [b])
        self.assertEqual(a.status, SequenceStatus.RESTORING)
        self.assertNotIn(a, scheduler.running)
        self.compute(scheduler, transport, batch, phase)

    def test_cpu_budget_falls_back_without_leaking_snapshots(self):
        scheduler, transport = self.scheduler(buffers=1)
        seq = self.sequence([3] * 300)
        self.prefill(scheduler, transport, [seq])
        self.preempt(scheduler, seq)
        self.assertEqual(seq.status, SequenceStatus.WAITING)
        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertFalse(scheduler.offload.handles)
        self.assertFalse(transport.out_ids)
        self.assertEqual(scheduler.offload.stats["fallback_preemptions"], 1)

    def test_restore_publishes_hash_only_after_transfer_completes(self):
        scheduler, transport = self.scheduler(blocks=3)
        seq = self.sequence([3] * 256)
        self.prefill(scheduler, transport, [seq])
        self.preempt(scheduler, seq)
        bm = scheduler.block_manager
        # Evict every free page, including the snapshot's original physical page.
        ids = [bm._allocate_block() for _ in range(3)]
        for gid in ids:
            transport.memory[gid] = (999,)
            bm.blocks[gid].ref_count -= 1
            bm._deallocate_block(gid)
        transport.advance()
        scheduler.schedule()
        self.assertEqual(seq.status, SequenceStatus.RESTORING)
        self.assertEqual(bm.blocks[seq.block_table[0]].hash, -1)
        transport.advance()
        scheduler.schedule()
        self.assertNotEqual(bm.blocks[seq.block_table[0]].hash, -1)

    def test_mixed_workload_preserves_all_kv_and_generated_tokens(self):
        scheduler, transport = self.scheduler(blocks=10, buffers=128, budget=512, inflight=2)
        sequences = [self.sequence([100 + i] * (100 + i * 17 % 413), output=100 + i * 29 % 401)
                     for i in range(24)]
        for seq in sequences:
            scheduler.add(seq)
        steps = 0
        while not scheduler.is_finished():
            transport.advance()
            batch, phase = scheduler.schedule()
            if batch:
                self.compute(scheduler, transport, batch, phase)
            else:
                scheduler.offload.wait_for_progress()
            steps += 1
            self.assertLess(steps, 20000)
            bm = scheduler.block_manager
            self.assertEqual(len(bm.used_block_ids) + len(bm.free_block_ids), 10)
            self.assertTrue(all(q.status == SequenceStatus.RUNNING for q in scheduler.running))
        self.assertGreater(scheduler.offload.stats["offloaded_sequences"], 0)
        self.assertFalse(scheduler.offload.handles)
        self.assertFalse(scheduler.block_manager.used_block_ids)
        for seq in sequences:
            self.assertEqual(seq.completion_token_ids,
                             [10000 + seq.token_ids[0] + i for i in range(seq.max_tokens)])


if __name__ == "__main__":
    unittest.main()
