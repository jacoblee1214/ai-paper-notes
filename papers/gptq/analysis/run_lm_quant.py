"""GPT-2 small / Pythia-160M에 RTN과 GPTQ(층별 순차)를 적용하고 held-out pile-10k 문서별 NLL을 저장한다.

설정 이름: fp | <rtn|gptq>_b<bits>_g<그룹|0>[_act][_d<damp>][_s<보정 시드>]
  예) gptq_b3_g128, gptq_b4_g0_act, gptq_b3_g0_d0.001, gptq_b3_g0_s1
보정: pile-10k 앞쪽 문서(섞은 순서 처음 5000개)에서 문서당 한 구간, 길이 512, N_CALIB개(기본 64). 평가: 나머지 문서에서 길이 ≥ 512인 문서의 첫 512토큰 N_EVAL개(기본 150).
실행: python3 run_lm_quant.py <gpt2|p160> <설정 ...>   결과는 lm_quant_nll.json에 이어 쓴다.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

from lm_quant import doc_nll, load_docs, load_model, make_sets, quantize_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "lm_quant_nll.json"
MODELS = {"gpt2": "openai-community/gpt2", "p160": "EleutherAI/pythia-160m"}
T, N_CALIB, N_EVAL = 512, 64, 150


def parse(cfg):
    if cfg == "fp":
        return dict(method="fp")
    m = re.fullmatch(r"(rtn|gptq)_b(\d+)_g(\d+)((?:_act)?)((?:_d[0-9.]+)?)((?:_s\d+)?)", cfg)
    assert m, cfg
    return dict(method=m[1], bits=int(m[2]), group=int(m[3]) or None, act_order=bool(m[4]),
                damp=float(m[5][2:]) if m[5] else 0.01, seed=int(m[6][2:]) if m[6] else 0)


def main():
    key, cfgs = sys.argv[1], sys.argv[2:]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    docs = load_docs()
    for cfg in cfgs:
        if f"{key}/{cfg}" in res:
            continue
        torch.manual_seed(0)
        p = parse(cfg)
        model, tok = load_model(MODELS[key])
        calib, ev, ev_docs = make_sets(tok, docs, T, N_CALIB, N_EVAL, p.get("seed", 0))
        t0 = time.time()
        if p["method"] != "fp":
            quantize_model(model, calib, p["method"], p["bits"], p["group"], p["damp"], p["act_order"]) \
                if p["method"] == "gptq" else quantize_model(model, calib, "rtn", p["bits"], p["group"])
        tq = time.time() - t0
        nll = doc_nll(model, ev)
        res[f"{key}/{cfg}"] = {"nll": nll.tolist(), "ppl": float(np.exp(nll.mean())), "quant_s": tq,
                               "eval_docs": ev_docs if "eval_docs" not in res.get(f"{key}/fp", {}) else None}
        OUT.write_text(json.dumps(res))
        print(f"{key}/{cfg}: ppl {res[f'{key}/{cfg}']['ppl']:.3f} (quant {tq:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
