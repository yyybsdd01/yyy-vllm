"""Exercise the real scheduler and KV block manager without running a model."""

from types import SimpleNamespace
import unittest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class SchedulerPreemptionLockTest(unittest.TestCase):
    def scheduler(self, enabled=True, blocks=4, token_budget=4096):
        return Scheduler(SimpleNamespace(
            max_num_seqs=16, max_num_batched_tokens=token_budget, eos=2,
            kvcache_block_size=256, num_kvcache_blocks=blocks,
            preemption_lock=enabled,
        ))

    def sequence(self, value, length=1, output=3, ignore_eos=True):
        return Sequence([value] * length, SamplingParams(
            max_tokens=output, ignore_eos=ignore_eos,
        ))

    def step(self, scheduler, token=10):
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [token] * len(seqs), prefill)
        return seqs, prefill

    def preempt_last(self, scheduler):
        seq = scheduler.running.pop()
        scheduler.preempt(seq)
        return seq

    def test_memory_pressure_sets_lock_and_releases_victim_kv(self):
        scheduler = self.scheduler(blocks=2)
        first, second = self.sequence(3, 256), self.sequence(4, 256)
        scheduler.add(first)
        scheduler.add(second)
        self.step(scheduler)
        seqs, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        self.assertEqual(seqs, [first])
        self.assertEqual(scheduler.lock, 1)
        self.assertEqual(list(scheduler.waiting), [second])
        self.assertEqual(second.block_table, [])
        self.assertEqual(second.num_cached_tokens, 0)

    def test_locked_decode_waits_for_completion_then_restores_prefill(self):
        scheduler = self.scheduler()
        first, second = self.sequence(3), self.sequence(4)
        scheduler.add(first)
        scheduler.add(second)
        self.step(scheduler)
        self.preempt_last(scheduler)
        self.assertGreaterEqual(scheduler.block_manager.can_allocate(second), 0)
        seqs, prefill = self.step(scheduler)
        self.assertFalse(prefill)
        self.assertEqual(seqs, [first])
        self.assertEqual(scheduler.lock, 1)
        self.assertEqual(second.num_completion_tokens, 1)
        self.step(scheduler)
        self.assertTrue(first.is_finished)
        self.assertEqual(scheduler.lock, 0)
        seqs, prefill = self.step(scheduler)
        self.assertTrue(prefill)
        self.assertEqual(seqs, [second])

    def test_switch_off_keeps_prefill_priority(self):
        scheduler = self.scheduler(enabled=False)
        first, second = self.sequence(3), self.sequence(4)
        scheduler.add(first)
        scheduler.add(second)
        self.step(scheduler)
        self.preempt_last(scheduler)
        seqs, prefill = scheduler.schedule()
        self.assertEqual(scheduler.lock, 1)
        self.assertTrue(prefill)
        self.assertEqual(seqs, [second])

    def test_empty_running_queue_can_restore_preempted_request(self):
        scheduler = self.scheduler()
        seq = self.sequence(3)
        scheduler.add(seq)
        self.step(scheduler)
        self.preempt_last(scheduler)
        self.assertEqual(scheduler.lock, 1)
        self.assertFalse(scheduler.running)
        seqs, prefill = self.step(scheduler)
        self.assertTrue(prefill)
        self.assertEqual(seqs, [seq])

    def test_eos_completion_also_unlocks(self):
        scheduler = self.scheduler()
        first = self.sequence(3, output=99, ignore_eos=False)
        scheduler.add(first)
        scheduler.add(self.sequence(4))
        self.step(scheduler)
        self.preempt_last(scheduler)
        self.step(scheduler, token=2)
        self.assertTrue(first.is_finished)
        self.assertEqual(scheduler.lock, 0)

    def test_repeated_preemptions_keep_lock_at_one(self):
        scheduler = self.scheduler()
        for value in (3, 4):
            scheduler.add(self.sequence(value))
        self.step(scheduler)
        self.preempt_last(scheduler)
        self.preempt_last(scheduler)
        self.assertEqual(scheduler.lock, 1)
        self.assertEqual(len(scheduler.waiting), 2)

    def test_mixed_workload_completes_with_valid_cache_and_output_lengths(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                scheduler = self.scheduler(enabled=enabled, blocks=12, token_budget=512)
                seqs = [self.sequence(100 + i, length=100 + i * 17 % 413,
                                      output=100 + i * 29 % 401) for i in range(24)]
                for seq in seqs:
                    scheduler.add(seq)
                steps = 0
                while not scheduler.is_finished():
                    batch, prefill = scheduler.schedule()
                    tokens = [10000 + seq[0] + seq.num_completion_tokens for seq in batch]
                    scheduler.postprocess(batch, tokens, prefill)
                    steps += 1
                    self.assertLess(steps, 20000, "scheduler did not make progress")
                    bm = scheduler.block_manager
                    self.assertEqual(len(bm.used_block_ids) + len(bm.free_block_ids), 12)
                    self.assertFalse(bm.used_block_ids.intersection(bm.free_block_ids))
                    for seq in seqs:
                        self.assertLessEqual(seq.num_cached_tokens, seq.num_tokens)
                self.assertEqual(scheduler.lock, 0)
                self.assertFalse(scheduler.block_manager.used_block_ids)
                for seq in seqs:
                    self.assertEqual(seq.completion_token_ids,
                                     [10000 + seq[0] + i for i in range(seq.max_tokens)])


if __name__ == "__main__":
    unittest.main()
