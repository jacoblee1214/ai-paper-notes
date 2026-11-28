"""Numerics only: paged attention over a block table equals dense attention on real GPT-2 K/V.

For each of 12 layers, q/k/v are recomputed from the hidden states (ln_1 + c_attn). The first T-1
tokens' K/V go into a PagedKVCache whose physical blocks are shuffled; the last token's query
attends through the block table (3 kernels) and is compared with F.scaled_dot_product_attention
on the same q/k/v (causal, last row). Also checks that the recomputed K/V equal HF's own
past_key_values. 20 documents x 12 layers x 12 heads, T=200, block sizes 1/16/64. No speed claim.
"""
import json
import random

import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from minimal_impl import BlockManager, PagedKVCache

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", local_files_only=True).eval()
p = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                    repo_type="dataset", local_files_only=True)
docs = pd.read_parquet(p)["text"].tolist()
T, ND = 200, 20
rng = random.Random(0)
res = {"T": T, "docs": ND, "max_abs_err": {}, "max_kv_vs_hf": 0.0}
errs = {(B, m): 0.0 for B in (1, 16, 64) for m in ("gather", "blockwise", "online")}
used = 0
for d in docs:
    if used >= ND:
        break
    ids = tok(d[:3000], return_tensors="pt").input_ids[:, :T]
    if ids.shape[1] < T:
        continue
    used += 1
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=True)
    for li, blk in enumerate(model.transformer.h):
        with torch.no_grad():
            h = blk.ln_1(out.hidden_states[li])
            q, k, v = blk.attn.c_attn(h).split(768, dim=-1)
        sh = lambda x: x.view(T, 12, 64)
        q, k, v = sh(q[0]), sh(k[0]), sh(v[0])
        pkv = out.past_key_values.layers[li] if hasattr(out.past_key_values, "layers") else None
        if pkv is not None:
            res["max_kv_vs_hf"] = max(res["max_kv_vs_hf"], (pkv.keys[0].transpose(0, 1) - k).abs().max().item())
        ref = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                             v.transpose(0, 1)[None], is_causal=True)[0, :, -1]
        for B in (1, 16, 64):
            nb = -(-T // B) + 3
            mgr, cache = BlockManager(nb, B), PagedKVCache(nb, B, 12, 64)
            perm = list(range(nb)); rng.shuffle(perm)
            mgr.gpu.free_list.extend(perm); mgr.gpu._fresh = nb
            s = mgr.new_sequence(); mgr.allocate(s, T)
            cache.write_many(s, 0, k, v)
            for mode in ("gather", "blockwise", "online"):
                e = (cache.paged_attention(q[-1], s, mode) - ref).abs().max().item()
                errs[(B, mode)] = max(errs[(B, mode)], e)
res["docs_used"] = used
res["max_abs_err"] = {f"B={B},{m}": v for (B, m), v in errs.items()}
print(json.dumps(res, indent=1))
json.dump(res, open("gpt2_paged_check_results.json", "w"), indent=1)
