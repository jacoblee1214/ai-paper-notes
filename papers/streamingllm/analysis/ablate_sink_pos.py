"""소거: 질의의 회전 위치. 논문 3.2절은 "캐시 안 위치로 매기는 것이 결정적"이라고 쓴다.
RoPE 모델(Pythia)에서 싱크 키는 그대로(위치 0..s-1), 질의만 (a) 캐시 안 위치 min(t, C-1) (논문) 또는 (b) 원래 위치 t로 회전해 비교한다.
(b)는 질의-싱크 거리가 t-k로 계속 커져 사전학습 길이를 넘는다. 윈도 키와의 거리는 두 경우 모두 t-j로 같다.
실행: python3 ablate_sink_pos.py   -> ablate_sink_pos_results.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_streaming_ppl as R
from minimal_impl import MiniLM

HERE = Path(__file__).resolve().parent
C = 512


def main():
    docs = R.load_docs()
    out = {"C": C, "T": 4096, "n_streams": R.N_STREAMS, "seed": R.SEED, "results": {}}
    for key in ("p70", "p160"):
        name, T, _, _, _, pre_ctx = R.PLAN[key]
        tok = AutoTokenizer.from_pretrained(name)
        m = MiniLM.from_hf(AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()).to(R.DEV)
        ids = R.make_streams(tok, docs, T, R.N_STREAMS).to(R.DEV)
        rng = np.random.RandomState(0)
        bk = R.buckets(T, C, pre_ctx)
        for s in (1, 4):
            for mode in ("cache", "orig"):
                m.sink_pos = mode
                d = m.nll(ids, spec=(s, C - s), bs=2).cpu().numpy()
                out["results"][f"{key}|s{s}|{mode}"] = R.summarize(d, bk, rng)
                print(key, s, mode, {b: round(v["ppl"], 2) for b, v in out["results"][f"{key}|s{s}|{mode}"].items()}, flush=True)
                if R.DEV == "mps":
                    torch.mps.empty_cache()
        del m
    json.dump(out, open(HERE / "ablate_sink_pos_results.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
