"""윈도만 쓸 때 perplexity가 얼마나 터지는가(0+C 대 4+(C-4))와 처음 위치가 받는 attention 질량 α₁의 순서가 같은가.

α₁은 별도 분석(attention_sink_emerges)이 공개 가중치로 잰 값을 옮겨 적은 것이다: Pile 문서 200개, 길이 T=1024,
처음 위치가 받는 attention을 head 평균(균일 기준선 1/136 = 0.00733). 모델 4개라 상관 추정이 아니라 순서 확인이다.
실행: python3 sink_alpha_link.py -> sink_alpha_link_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "streaming_ppl_results.json"))
ALPHA1 = {"p14": 0.01729, "p70": 0.02861, "p160": 0.07291, "gpt2": 0.35480}
CASES = {"p14": (512, "orig"), "p70": (512, "orig"), "p160": (512, "orig"), "gpt2": (512, "orig")}
rows = {}
for k, (C, mode) in CASES.items():
    a = np.array(R[k]["configs"][f"C{C}|s0|{mode}"]["buckets"]["post_all"]["per_stream_nll"])
    b = np.array(R[k]["configs"][f"C{C}|s4|{mode}"]["buckets"]["post_all"]["per_stream_nll"])
    rows[k] = {"alpha1_head_mean_T1024": ALPHA1[k], "C": C, "ppl_ratio_window_over_4sinks": float(np.exp((a - b).mean())),
               "log_ratio_nll": float((a - b).mean())}
rho = spearmanr([v["alpha1_head_mean_T1024"] for v in rows.values()], [v["log_ratio_nll"] for v in rows.values()])
out = {"rows": rows, "spearman_rho": float(rho.statistic), "n_models": len(rows),
       "note": "모델 4개, 순서만 본다. GPT-2는 학습된 절대 위치 모델이고 Pythia는 RoPE라 모델군이 섞여 있다."}
json.dump(out, open(HERE / "sink_alpha_link_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
