import argparse
import os
from time import perf_counter

import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """将示例问题包装为聊天提示，运行 Qwen3 推理并打印生成结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8", "int8_dequant",
                                                     "int8_half", "int8_half_dequant"), default="auto")
    parser.add_argument("--cuda-graph", action="store_true", help="启用 CUDA Graph；默认保持原示例的 eager 模式")
    parser.add_argument("--seed", type=int, help="设置采样随机种子，便于重复计时")
    args = parser.parse_args()

    started = perf_counter()
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer_ready = perf_counter()
    llm = LLM(path, enforce_eager=not args.cuda_graph, tensor_parallel_size=1,
              kv_cache_dtype=args.kv_cache_dtype)
    model_ready = perf_counter()
    # llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, kv_cache_dtype="int8")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # prompts = [
    #         "介绍你自己",
    #         "今天星期几",
    #     ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    prompt_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.cuda.synchronize()
    generate_started = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    generate_finished = perf_counter()

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    output_tokens = sum(len(output["token_ids"]) for output in outputs)
    generate_seconds = generate_finished - generate_started
    print(f"\nKV cache: {args.kv_cache_dtype} | CUDA Graph: {args.cuda_graph} | seed: {args.seed}")
    print(f"Tokenizer load: {tokenizer_ready - started:.3f} s")
    print(f"Model init + warmup: {model_ready - tokenizer_ready:.3f} s")
    print(f"Generate ({len(prompts)} requests): {generate_seconds:.3f} s")
    print(f"Prompt tokens: {prompt_tokens} | output tokens: {output_tokens} | "
          f"output throughput: {output_tokens / generate_seconds:.2f} token/s")
    print(f"Startup through generation: {generate_finished - started:.3f} s")


if __name__ == "__main__":
    main()
