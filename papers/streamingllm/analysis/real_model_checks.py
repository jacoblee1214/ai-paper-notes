"""실제 가중치에서의 정합성: (1) 우리 MiniLM이 HF 구현과 같은 값을 내는가, (2) 토큰 하나씩 넣는 RollingKVCache가
한 번에 계산하는 attend()와 같은가. Pythia-70M, GPT-2 small, float32, CPU. 결과 real_model_checks_results.json.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_streaming_ppl as R
from minimal_impl import MiniLM, RollingKVCache

HERE = Path(__file__).resolve().parent
out = {}
docs = R.load_docs()
for key, T_ref, spec in [("p70", 512, (4, 124)), ("gpt2", 512, (4, 124))]:
    name = R.PLAN[key][0]
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    m = MiniLM.from_hf(hf)
    ids = R.make_streams(tok, docs, T_ref, 4)
    with torch.no_grad():
        lg_hf = hf(ids).logits
        nll_hf = F.cross_entropy(lg_hf[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
        nll_me = m.nll(ids, bs=4)
        lg_me = m.logits(ids)
    # 롤링 캐시 대 병렬 구현 (캐시 128 = 4 + 124, 길이 512 -> 캐시를 여러 번 비운다)
    for mode in (["orig"] if key == "p70" else ["orig", "slot"]):
        par = m.logits(ids, spec=spec, pos_mode=mode)
        c = RollingKVCache(m, spec[0], spec[1], mode)
        step = torch.stack([c.step(ids[:, t]) for t in range(T_ref)], 1)
        out[f"{key}|rolling_vs_parallel|{mode}"] = {"max_abs_logit_diff": float((step - par).abs().max()),
                                                    "max_abs_logit": float(par.abs().max()), "spec": list(spec), "T": T_ref}
    out[f"{key}|mini_vs_hf"] = {"max_abs_logit_diff": float((lg_hf - lg_me).abs().max()), "max_abs_logit": float(lg_hf.abs().max()),
                                "mean_abs_nll_diff": float((nll_hf - nll_me).abs().mean()),
                                "ppl_hf": float(nll_hf.mean().exp()), "ppl_mini": float(nll_me.mean().exp()), "n_streams": 4, "T": T_ref}
    print(key, out[f"{key}|mini_vs_hf"], flush=True)
json.dump(out, open(HERE / "real_model_checks_results.json", "w"), indent=1)
