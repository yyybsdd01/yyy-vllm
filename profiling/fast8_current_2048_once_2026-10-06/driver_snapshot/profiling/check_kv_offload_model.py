"""Fresh-process deterministic model checks under forced KV pressure."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--mode", default="int8_half")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--verify-kv", action="store_true")
    parser.add_argument("--kv-blocks", type=int, default=12)
    parser.add_argument("--natural", action="store_true")
    parser.add_argument("--teacher-force", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.variant.resolve()))
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import SequenceStatus
    torch.manual_seed(0)
    llm = LLM("/home/xgd/huggingface/Qwen3-0.6B", kv_cache_dtype=args.mode,
              kv_cpu_offload=args.offload, num_kvcache_blocks=args.kv_blocks,
              max_num_seqs=16, max_num_batched_tokens=2048, max_model_len=1024)
    # Validation only: remove sampling dependence on batch membership/order.
    llm.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
    rng = random.Random(0)
    if args.natural:
        text = llm.tokenizer.encode("这是文本复制测试。请严格按题目要求复制指定内容，不添加解释。")
        prefix = (text * (256 // len(text) + 1))[:256]
    else:
        prefix = [rng.randrange(10000) for _ in range(256)]
    llm.generate([prefix], SamplingParams(max_tokens=2, ignore_eos=True), use_tqdm=False)
    # Prewarm the actual attention variants before profiler capture.
    llm.generate([[11000 + i] * 512 for i in range(4)],
                 SamplingParams(max_tokens=2, ignore_eos=True), use_tqdm=False)
    preemptions = Counter()
    expected_kv = {}
    kv_checks = 0
    current_batch = []
    original_preempt = llm.scheduler.preempt

    def preempt(seq):
        preemptions[seq.seq_id] += 1
        if args.verify_kv and args.offload:
            expected_kv[seq.seq_id] = (
                seq.num_cached_tokens,
                [llm.model_runner.kv_cache[:, :, gid].cpu().clone() for gid in seq.block_table],
                [llm.model_runner.kv_scales[:, :, gid].cpu().clone() for gid in seq.block_table]
                if hasattr(llm.model_runner, "kv_scales") else None,
            )
        original_preempt(seq)

    llm.scheduler.preempt = preempt
    original_run = llm.model_runner.run

    def checked_run(seqs, prefill):
        nonlocal kv_checks, current_batch
        current_batch = seqs
        if not prefill:
            assert all(seq.status == SequenceStatus.RUNNING for seq in seqs)
            for seq in seqs:
                if seq.seq_id in expected_kv:
                    length, cache, scales = expected_kv.pop(seq.seq_id)
                    assert seq.num_cached_tokens == length
                    for i in range((length + 255) // 256):
                        valid = min(256, length - i * 256)
                        actual = llm.model_runner.kv_cache[:, :, seq.block_table[i], :valid].cpu()
                        assert torch.equal(actual, cache[i][:, :, :valid]), (seq.seq_id, i, "KV differs")
                        if scales is not None:
                            actual = llm.model_runner.kv_scales[:, :, seq.block_table[i], :valid].cpu()
                            assert torch.equal(actual, scales[i][:, :, :valid]), (seq.seq_id, i, "scale differs")
                        kv_checks += 1
        if args.profile:
            with torch.profiler.record_function("MODEL_PREFILL" if prefill else "MODEL_DECODE"):
                return original_run(seqs, prefill)
        return original_run(seqs, prefill)

    llm.model_runner.run = checked_run
    if args.natural:
        prompts = []
        for i in range(16):
            filler = llm.tokenizer.encode(f"样例编号{i}。材料与答案无关，请阅读最后的问题。")
            question = llm.tokenizer.encode("\n请原样输出：春天来了，花开了，鸟儿在树上唱歌。\n答案：")
            middle = (filler * (256 // len(filler) + 1))[:256 - len(question)]
            prompts.append(prefix + middle + question)
    else:
        prompts = [prefix + [rng.randrange(10000) for _ in range(256)] for _ in range(16)]
    if args.teacher_force:
        request_index = {tuple(prompt): i for i, prompt in enumerate(prompts)}
        scores = torch.empty((16, 24, llm.model_runner.config.hf_config.vocab_size),
                             dtype=torch.bfloat16, device="cpu")
        forced = llm.tokenizer.encode("春天来了，花开了，鸟儿在树上唱歌。")

        def teacher(logits, temperatures):
            ids = []
            cpu_logits = logits.detach().cpu()
            for row, seq in enumerate(current_batch):
                step = seq.num_completion_tokens
                scores[request_index[tuple(seq.prompt_token_ids)], step].copy_(cpu_logits[row])
                ids.append(forced[step % len(forced)])
            return torch.tensor(ids, dtype=torch.int64, device=logits.device)

        llm.model_runner.sampler.forward = teacher
    params = SamplingParams(max_tokens=24, ignore_eos=True)
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        profiler.export_chrome_trace(str(args.profile))
    else:
        outputs = llm.generate(prompts, params, use_tqdm=False)
    assert len(outputs) == 16 and all(len(o["token_ids"]) == 24 for o in outputs)
    assert sum(preemptions.values()) > 0 or args.kv_blocks > 32
    assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
    stats = llm.scheduler.offload.summary() if args.offload else None
    if args.offload:
        assert stats["active_handles"] == 0 and stats["restoring"] == 0
        assert stats["shared_snapshot_references"] > 0
        assert stats["transport"]["h2d_blocks"] > 0
    result = dict(mode=args.mode, offload=args.offload, natural=args.natural, scope="validation-only argmax sampler",
                  blocks=llm.model_runner.config.num_kvcache_blocks,
                  preemptions=sum(preemptions.values()), stats=stats,
                  restored_block_exact_checks=kv_checks,
                  outputs=[o["token_ids"] for o in outputs])
    result["output_sha256"] = hashlib.sha256(json.dumps(result["outputs"]).encode()).hexdigest()
    if args.teacher_force:
        torch.save(scores, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", args.mode, args.offload, "preemptions", result["preemptions"],
          "output_sha256", result["output_sha256"], flush=True)


if __name__ == "__main__":
    main()
