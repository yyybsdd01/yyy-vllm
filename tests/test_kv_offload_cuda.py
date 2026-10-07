"""Real pinned transfers, source overwrite protection and restore gating."""

from types import SimpleNamespace
import unittest

import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_offload import CudaKVTransport, KVOffloadManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KVOffloadCudaTest(unittest.TestCase):
    def test_all_layers_roundtrip_and_locked_source_overwrite(self):
        for dtype, groups in ((torch.bfloat16, 0), (torch.int8, 1), (torch.int8, 2)):
            with self.subTest(dtype=dtype, groups=groups):
                kv = torch.randint(-100, 100, (2, 28, 4, 256, 8, 128), device="cuda", dtype=torch.int8).to(dtype)
                scales = (torch.rand((2, 28, 4, 256, 8 * groups), device="cuda") if groups else None)
                transport = CudaKVTransport(kv, scales)
                expected = kv[:, :, 0].cpu().clone()
                expected_scale = scales[:, :, 0].cpu().clone() if groups else None
                buffer = transport.new_buffer()
                # The original page is logically available before the queued pack executes.
                with torch.cuda.stream(transport.d2h_stream):
                    torch.cuda._sleep(30_000_000)
                out_done = transport.offload(0, buffer)
                self.assertFalse(out_done.query())
                seq = SimpleNamespace(block_table=[0], num_cached_tokens=0, num_scheduled_tokens=256)
                transport.wait_for_compute_writes([seq], True)
                kv[:, :, 0].fill_(-7)
                if groups:
                    scales[:, :, 0].fill_(-7)
                out_done.synchronize()
                self.assertTrue(torch.equal(buffer[0], expected))
                if groups:
                    self.assertTrue(torch.equal(buffer[1], expected_scale))
                done = transport.restore_block(2, buffer, out_done)
                done.synchronize()
                self.assertTrue(torch.equal(kv[:, :, 2].cpu(), expected))
                if groups:
                    self.assertTrue(torch.equal(scales[:, :, 2].cpu(), expected_scale))
                transport.close()
                self.assertGreaterEqual(transport.stats["write_lock_waits"], 1)

    def test_restore_state_blocks_admission_until_real_h2d_finishes(self):
        kv = torch.randint(-100, 100, (2, 28, 1, 256, 8, 128), device="cuda", dtype=torch.int8)
        scales = torch.rand((2, 28, 1, 256, 16), device="cuda")
        transport = CudaKVTransport(kv, scales)
        bm = BlockManager(1, 256)
        manager = KVOffloadManager(bm, transport, cpu_gb=0.1)
        seq = Sequence([7] * 10)
        bm.allocate(seq, 0)
        seq.num_cached_tokens = 9
        expected = kv[:, :, 0].cpu().clone()
        self.assertTrue(manager.capture(seq))
        bm.deallocate(seq)
        manager.handles[seq.seq_id].blocks[0].ready_event.synchronize()
        # Reallocate the physical ID for another generation before restoration.
        gid = bm._allocate_block()
        kv[:, :, gid].fill_(-9)
        bm.blocks[gid].ref_count -= 1
        bm._deallocate_block(gid)
        with torch.cuda.stream(transport.h2d_stream):
            torch.cuda._sleep(30_000_000)
        manager.restore(seq)
        self.assertEqual(seq.status, SequenceStatus.RESTORING)
        self.assertEqual(manager.poll(), [])
        manager.restoring[seq.seq_id].restore_event.synchronize()
        self.assertEqual(manager.poll(), [seq])
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertEqual(seq.num_cached_tokens, 9)
        self.assertTrue(torch.equal(kv[:, :, 0].cpu(), expected))
        transport.close()


if __name__ == "__main__":
    unittest.main()
