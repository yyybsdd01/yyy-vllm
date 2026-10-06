"""Full WikiText-2 PPL for an isolated BF16 or all-INT8 prefill implementation.

Use PYTHONPATH to select the source copy. The established 4096-window,
256-chunk protocol is retained; INT8 also quantizes the first chunk.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from transformers import AutoConfig,AutoTokenizer
import nanovllm
import nanovllm.layers.attention as attention_module
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import reset_context
from nanovllm.utils.loader import load_model
import kv_cache_perplexity as protocol


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--text",required=True,type=Path)
    parser.add_argument("--mode",required=True,choices=("auto","int8_half"))
    parser.add_argument("--label",required=True)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--max-tokens",type=int,default=298938)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = 1000000
    ids = tokenizer.encode(args.text.read_text(),add_special_tokens=False)[:args.max_tokens+1]
    assert len(ids)>1
    print(f"QUALITY START {args.label} mode={args.mode} tokens={len(ids)-1} package={nanovllm.__file__}",flush=True)
    dist.init_process_group("nccl",init_method="tcp://127.0.0.1:29571",rank=0,world_size=1)
    torch.cuda.set_device(0)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.dtype)
    torch.set_default_device("cuda")
    try:
        model = Qwen3ForCausalLM(config)
        load_model(model,args.model)
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
    model.eval()
    dim = getattr(config,"head_dim",config.hidden_size//config.num_attention_heads)
    cache,scales,scratch = protocol.bind_cache(model,args.mode,config.num_hidden_layers,
                                             256,16,config.num_key_value_heads,dim)
    dispatch = {"flash":0,"int8":0}
    original_flash = attention_module.flash_attn_varlen_func
    original_int8 = attention_module.int8_paged_attention
    def counted_flash(*a,**kw):
        dispatch["flash"] += 1
        return original_flash(*a,**kw)
    def counted_int8(*a,**kw):
        dispatch["int8"] += 1
        return original_int8(*a,**kw)
    attention_module.flash_attn_varlen_func = counted_flash
    attention_module.int8_paged_attention = counted_int8
    original_context = protocol.set_context
    chunks = 0
    def selected_context(is_prefill,**kwargs):
        nonlocal chunks
        # The optimized inference route reads INT8 KV even with no prefix.
        if args.mode.startswith("int8") and kwargs["block_tables"] is None:
            kwargs["block_tables"] = torch.arange(math.ceil(kwargs["max_seqlen_k"]/256),
                                                   device="cuda",dtype=torch.int32)[None,:]
        original_context(is_prefill,**kwargs)
        chunks += 1
        if chunks%128==0:
            print(f"QUALITY PROGRESS {args.label} chunks={chunks}",flush=True)
    protocol.set_context = selected_context
    started = perf_counter()
    try:
        result = protocol.evaluate(model,ids,args.mode,4096,256,256)
    finally:
        reset_context()
        dist.destroy_process_group()
    assert sum(dispatch.values())==chunks*config.num_hidden_layers
    assert dispatch["int8"]==0 if args.mode=="auto" else dispatch["flash"]==0
    result.update(label=args.label,package=nanovllm.__file__,attention_module=attention_module.__file__,
        dispatch=dispatch,chunks=chunks,window_size=4096,chunk_size=256,block_size=256,
        first_chunk_route="flash_original_bf16" if args.mode=="auto" else "quantized_int8",
        text=str(args.text),text_sha256=hashlib.sha256(args.text.read_bytes()).hexdigest(),
        token_ids_sha256=hashlib.sha256(array("I",ids).tobytes()).hexdigest(),
        model=args.model,model_config_sha256=hashlib.sha256((Path(args.model)/"config.json").read_bytes()).hexdigest(),
        evaluation_wall_s=perf_counter()-started)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f'QUALITY COMPLETE {args.label} NLL={result["nll"]:.6f} PPL={result["ppl"]:.6f} '
          f'fresh_PPL={result["fresh_ppl"]:.6f} cached_PPL={result["cached_ppl"]:.6f} dispatch={dispatch}',flush=True)


if __name__ == "__main__":
    main()
