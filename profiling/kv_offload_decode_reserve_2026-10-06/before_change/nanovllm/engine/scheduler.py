from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        """初始化调度上限、KV 块管理器及等待和运行队列。"""
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.preemption_lock = config.preemption_lock
        self.lock = 0
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.offload = None
        self.offload_queue: deque[Sequence] = deque()

    def is_finished(self):
        """判断等待队列和运行队列是否都已清空。"""
        return (not self.waiting and not self.running and not self.offload_queue
                and (self.offload is None or not self.offload.restoring))

    def add(self, seq: Sequence):
        """将新序列加入等待队列。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """默认优先 prefill；启用抢占锁后，锁住期间优先推进运行中的 decode。"""
        scheduled_seqs = []
        num_batched_tokens = 0
        if self.offload is not None:
            self.running.extend(self.offload.poll())
            # Restore admission precedes normal prefill, without changing phase priority.
            while self.offload_queue and len(self.offload.restoring) < self.offload.max_inflight:
                seq = self.offload_queue[0]
                if not self.offload.offload_ready(seq) or not self.offload.can_restore(seq):
                    break
                self.offload_queue.popleft()
                self.offload.restore(seq)
            self.running.extend(self.offload.poll())
        # 抢占后暂停接纳 prefill，直到任一请求完成并释放 KV。
        # 没有 running 请求时仍允许 prefill，以恢复被抢占请求的 KV。
        prefer_decode = self.preemption_lock and self.lock == 1 and bool(self.running)

        # prefill
        while not prefer_decode and self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            assert seq.status == SequenceStatus.RUNNING
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs and prefer_decode and self.waiting:
            # 若本轮最后一个 running 请求也被抢占，下一次选择允许 prefill。
            return self.schedule()
        if not scheduled_seqs and self.offload is not None and (self.offload_queue or self.offload.restoring):
            return [], False
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占序列，释放其 KV 块并放回等待队列以便重新预填充。"""
        if self.offload is not None and self.offload.capture(seq):
            seq.status = SequenceStatus.OFFLOADING
            self.block_manager.deallocate(seq)
            self.offload_queue.appendleft(seq)
            return
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.lock = 1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """记录新完成的块与 token，按 EOS 或生成长度结束序列并回收 KV 块。"""
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)#一个块只有填满了才会算hash
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self.lock = 0
