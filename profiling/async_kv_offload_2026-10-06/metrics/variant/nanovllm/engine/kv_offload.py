"""Block snapshots, deferred shared-prefix eviction, and asynchronous KV copies."""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import torch

from nanovllm.engine.sequence import SequenceStatus


class CompletedEvent:
    def query(self):
        return True

    def synchronize(self):
        pass


class CudaKVTransport:
    """One reusable GPU block per direction; CPU buffers belong to snapshots.

    D2H first gathers a strided physical block into staging. Its pack event
    protects the old physical bytes, while the CPU buffer needs the D2H event.
    All subsequent writes (including restoration) observe the pack event.
    """

    def __init__(self, kv_cache, kv_scales=None):
        self.kv = kv_cache
        self.scales = kv_scales
        self.device = kv_cache.device
        self.block_shape = tuple(kv_cache[:, :, 0].shape)
        self.scale_shape = tuple(kv_scales[:, :, 0].shape) if kv_scales is not None else None
        self.block_bytes = kv_cache[:, :, 0].numel() * kv_cache.element_size()
        if kv_scales is not None:
            self.block_bytes += kv_scales[:, :, 0].numel() * kv_scales.element_size()
        self.d2h_stream = torch.cuda.Stream(device=self.device)
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self.out_kv = torch.empty(self.block_shape, device=self.device, dtype=kv_cache.dtype)
        self.in_kv = torch.empty_like(self.out_kv)
        self.out_scale = self.in_scale = None
        if kv_scales is not None:
            self.out_scale = torch.empty(self.scale_shape, device=self.device, dtype=kv_scales.dtype)
            self.in_scale = torch.empty_like(self.out_scale)
        self.write_locks = {}
        self.stats = Counter({key: 0 for key in (
            "write_lock_waits", "d2h_blocks", "h2d_blocks", "d2h_bytes", "h2d_bytes",
            "d2h_enqueue_ms", "h2d_enqueue_ms", "d2h_completed_events", "h2d_completed_events",
            "d2h_stream_ms", "h2d_stream_ms", "d2h_first_phase_ms", "h2d_first_phase_ms",
            "d2h_second_phase_ms", "h2d_second_phase_ms")})
        self.timings = []

    def new_buffer(self):
        kv = torch.empty(self.block_shape, dtype=self.kv.dtype, device="cpu", pin_memory=True)
        scale = (torch.empty(self.scale_shape, dtype=self.scales.dtype, device="cpu", pin_memory=True)
                 if self.scales is not None else None)
        return kv, scale

    def _wait_writes(self, stream, block_ids):
        seen = set()
        for block_id in block_ids:
            event = self.write_locks.get(block_id)
            if event is None or id(event) in seen:
                continue
            seen.add(id(event))
            if event.query():
                self.write_locks.pop(block_id, None)
            else:
                stream.wait_event(event)
                self.stats["write_lock_waits"] += 1

    def wait_for_compute_writes(self, seqs, is_prefill):
        pages = []
        block_size = self.kv.shape[3]
        for seq in seqs:
            if is_prefill:
                start = seq.num_cached_tokens // block_size
                end = (seq.num_cached_tokens + seq.num_scheduled_tokens + block_size - 1) // block_size
                pages.extend(seq.block_table[start:end])
            else:
                pages.append(seq.block_table[-1])
        self._wait_writes(torch.cuda.current_stream(self.device), pages)

    def offload(self, block_id, buffer):
        host_kv, host_scale = buffer
        producer = torch.cuda.current_stream(self.device)
        start, packed, done = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        submitted = perf_counter()
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_stream(producer)
            self._wait_writes(self.d2h_stream, [block_id])
            start.record(self.d2h_stream)
            self.out_kv.copy_(self.kv[:, :, block_id], non_blocking=True)
            if self.scales is not None:
                self.out_scale.copy_(self.scales[:, :, block_id], non_blocking=True)
            packed.record(self.d2h_stream)
            host_kv.copy_(self.out_kv, non_blocking=True)
            if self.scales is not None:
                host_scale.copy_(self.out_scale, non_blocking=True)
            done.record(self.d2h_stream)
        self.write_locks[block_id] = packed
        self.stats["d2h_blocks"] += 1
        self.stats["d2h_bytes"] += self.block_bytes
        self.stats["d2h_enqueue_ms"] += (perf_counter() - submitted) * 1000
        self.timings.append(("d2h", start, packed, done))
        return done

    def restore_block(self, block_id, buffer, ready_event):
        host_kv, host_scale = buffer
        start, copied, done = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        submitted = perf_counter()
        with torch.cuda.stream(self.h2d_stream):
            if ready_event is not None:
                self.h2d_stream.wait_event(ready_event)
            self._wait_writes(self.h2d_stream, [block_id])
            start.record(self.h2d_stream)
            self.in_kv.copy_(host_kv, non_blocking=True)
            if self.scales is not None:
                self.in_scale.copy_(host_scale, non_blocking=True)
            copied.record(self.h2d_stream)
            self.kv[:, :, block_id].copy_(self.in_kv, non_blocking=True)
            if self.scales is not None:
                self.scales[:, :, block_id].copy_(self.in_scale, non_blocking=True)
            done.record(self.h2d_stream)
        self.stats["h2d_blocks"] += 1
        self.stats["h2d_bytes"] += self.block_bytes
        self.stats["h2d_enqueue_ms"] += (perf_counter() - submitted) * 1000
        self.timings.append(("h2d", start, copied, done))
        return done

    def finish_restore(self, copied):
        if not copied:
            return CompletedEvent()
        event = torch.cuda.Event()
        event.record(self.h2d_stream)
        return event

    def collect_completed(self):
        pending = []
        for mode, start, middle, end in self.timings:
            if end.query():
                self.stats[mode + "_completed_events"] += 1
                self.stats[mode + "_stream_ms"] += start.elapsed_time(end)
                self.stats[mode + "_first_phase_ms"] += start.elapsed_time(middle)
                self.stats[mode + "_second_phase_ms"] += middle.elapsed_time(end)
            else:
                pending.append((mode, start, middle, end))
        self.timings = pending

    def summary(self):
        self.collect_completed()
        return dict(self.stats)

    def close(self):
        self.d2h_stream.synchronize()
        self.h2d_stream.synchronize()
        self.collect_completed()


@dataclass(eq=False)
class BlockSnapshot:
    gpu_id: int | None
    generation: int
    block_hash: int
    token_ids: list[int]
    buffer: object
    refs: int = 0
    ready_event: object = None


@dataclass
class SequenceSnapshot:
    seq: object
    cached_tokens: int
    blocks: list[BlockSnapshot]
    restore_event: object = None


class KVOffloadManager:
    """CPU-side ownership independent of GPU ref_count; transport is injectable."""

    def __init__(self, block_manager, transport, cpu_gb=4.0, max_inflight=8):
        self.bm = block_manager
        self.transport = transport
        self.max_buffers = int(cpu_gb * 1024**3) // transport.block_bytes
        if self.max_buffers < 1:
            raise ValueError("offload CPU budget is smaller than one KV block")
        self.max_inflight = max_inflight
        self.by_gpu = {}
        self.handles = {}
        self.restoring = {}
        self.free_buffers = []
        self.retired = []
        self.allocated_buffers = 0
        self.stats = Counter({key: 0 for key in (
            "fallback_preemptions", "shared_snapshot_references", "offloaded_sequences",
            "restored_sequences", "gpu_reused_blocks", "peak_cpu_buffers",
            "peak_restoring_sequences", "idle_copy_waits", "idle_copy_wait_ms")})
        self.bm.on_free = self.on_free
        self.bm.on_reuse = self.on_reuse

    def _gpu_valid(self, snapshot):
        gid = snapshot.gpu_id
        return gid is not None and self.bm.blocks[gid].generation == snapshot.generation

    def _collect_retired(self):
        remaining = []
        for snapshot in self.retired:
            if snapshot.ready_event is None or snapshot.ready_event.query():
                self.free_buffers.append(snapshot.buffer)
            else:
                remaining.append(snapshot)
        self.retired = remaining

    def capture(self, seq):
        self._collect_retired()
        count = (seq.num_cached_tokens + self.bm.block_size - 1) // self.bm.block_size
        pages = seq.block_table[:count]
        missing = sum(gid not in self.by_gpu for gid in pages)
        available = len(self.free_buffers) + self.max_buffers - self.allocated_buffers
        if missing > available:
            self.stats["fallback_preemptions"] += 1
            return False
        snapshots = []
        for gid in pages:
            snapshot = self.by_gpu.get(gid)
            if snapshot is None:
                if self.free_buffers:
                    buffer = self.free_buffers.pop()
                else:
                    buffer = self.transport.new_buffer()
                    self.allocated_buffers += 1
                block = self.bm.blocks[gid]
                snapshot = BlockSnapshot(gid, block.generation, block.hash, list(block.token_ids), buffer)
                self.by_gpu[gid] = snapshot
            snapshot.refs += 1
            snapshots.append(snapshot)
            if self.bm.blocks[gid].ref_count > 1:
                self.stats["shared_snapshot_references"] += 1
        self.handles[seq.seq_id] = SequenceSnapshot(seq, seq.num_cached_tokens, snapshots)
        self.stats["offloaded_sequences"] += 1
        self.stats["peak_cpu_buffers"] = max(self.stats["peak_cpu_buffers"], self.allocated_buffers)
        return True

    def on_free(self, gid):
        snapshot = self.by_gpu.get(gid)
        if snapshot is not None and snapshot.refs and snapshot.ready_event is None:
            assert self.bm.blocks[gid].ref_count == 0
            snapshot.ready_event = self.transport.offload(gid, snapshot.buffer)

    def on_reuse(self, gid):
        snapshot = self.by_gpu.pop(gid, None)
        if snapshot is not None:
            assert snapshot.ready_event is not None, "reusing a snapshot without preserving its KV"
            snapshot.gpu_id = None

    def offload_ready(self, seq):
        handle = self.handles[seq.seq_id]
        ready = all(s.ready_event is None or s.ready_event.query() for s in handle.blocks)
        seq.status = SequenceStatus.WAITING_RESTORE if ready else SequenceStatus.OFFLOADING
        return ready

    def can_restore(self, seq):
        handle = self.handles[seq.seq_id]
        resident_used = sum(self._gpu_valid(s) and self.bm.blocks[s.gpu_id].ref_count > 0 for s in handle.blocks)
        return len(self.bm.free_block_ids) >= seq.num_blocks - resident_used

    def restore(self, seq):
        handle = self.handles[seq.seq_id]
        assert not seq.block_table and self.can_restore(seq)
        targets = [None] * len(handle.blocks)
        # Retain every resident page first, so new allocations cannot evict them.
        for i, snapshot in enumerate(handle.blocks):
            if self._gpu_valid(snapshot):
                self.bm.retain(snapshot.gpu_id)
                targets[i] = snapshot.gpu_id
                self.stats["gpu_reused_blocks"] += 1
        copied = 0
        for i, snapshot in enumerate(handle.blocks):
            if targets[i] is None:
                assert snapshot.ready_event is not None
                target = self.bm._allocate_block()
                targets[i] = target
                self.transport.restore_block(target, snapshot.buffer, snapshot.ready_event)
                copied += 1
        seq.block_table = targets
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.bm._allocate_block())
        seq.status = SequenceStatus.RESTORING
        seq.num_scheduled_tokens = 0
        seq.is_prefill = False
        handle.restore_event = self.transport.finish_restore(copied)
        self.restoring[seq.seq_id] = handle
        self.stats["peak_restoring_sequences"] = max(self.stats["peak_restoring_sequences"], len(self.restoring))

    def poll(self):
        if hasattr(self.transport, "collect_completed"):
            self.transport.collect_completed()
        ready = []
        for seq_id, handle in list(self.restoring.items()):
            if not handle.restore_event.query():
                continue
            seq = handle.seq
            seq.num_cached_tokens = handle.cached_tokens
            # Publish reusable prefix metadata only after all restored bytes exist.
            for i, snapshot in enumerate(handle.blocks):
                if not self._gpu_valid(snapshot):
                    gid = seq.block_table[i]
                    snapshot.gpu_id = gid
                    snapshot.generation = self.bm.blocks[gid].generation
                    self.by_gpu[gid] = snapshot
                if snapshot.block_hash != -1:
                    gid = seq.block_table[i]
                    self.bm.blocks[gid].update(snapshot.block_hash, list(snapshot.token_ids))
                    self.bm.hash_to_block_id[snapshot.block_hash] = gid
            seq.status = SequenceStatus.RUNNING
            ready.append(seq)
            for snapshot in handle.blocks:
                snapshot.refs -= 1
                if snapshot.refs == 0:
                    if snapshot.gpu_id is not None and self.by_gpu.get(snapshot.gpu_id) is snapshot:
                        del self.by_gpu[snapshot.gpu_id]
                    self.retired.append(snapshot)
            del self.restoring[seq_id]
            del self.handles[seq_id]
            self.stats["restored_sequences"] += 1
        self._collect_retired()
        return ready

    def wait_for_progress(self):
        events = [h.restore_event for h in self.restoring.values()]
        events += [s.ready_event for h in self.handles.values() for s in h.blocks if s.ready_event is not None]
        for event in events:
            if not event.query():
                start = perf_counter()
                event.synchronize()
                self.stats["idle_copy_wait_ms"] += (perf_counter() - start) * 1000
                self.stats["idle_copy_waits"] += 1
                return
        raise RuntimeError("offload scheduler has no runnable sequence and no pending transfer")

    def summary(self):
        return dict(self.stats, cpu_budget_bytes=self.max_buffers * self.transport.block_bytes,
                    cpu_pool_bytes=self.allocated_buffers * self.transport.block_bytes,
                    active_handles=len(self.handles), restoring=len(self.restoring),
                    transport=self.transport.summary())
