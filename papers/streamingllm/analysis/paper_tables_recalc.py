"""논문 표·그림에 인쇄된 숫자의 재계산과 교차 확인(실행 없이 계산만). 결과는 paper_tables_recalc_results.json.

  1. 그림 10: 지연 시간 막대의 값에서 속도비(최대 22.2x?)를 다시 계산하고, 재계산 기준선이 L²로 느는지 최소제곱으로 적합
  2. 표 2 vs 표 6: 같은 구성(4+y)이 두 표에 나오면 값이 같아야 한다
  3. 표 2: 윈도만(0+y) 대 4+y의 비율(모델별로 "perplexity 폭발"의 크기)
  4. 표 3/10: 표본 1개 평가
  5. 표 4: 싱크 토큰 모델과 vanilla의 제로샷 정확도 차이가 표본 오차 안인가
  6. 표 7: 정확도가 0이 되는 거리가 캐시 크기와 맞는가
논문 값은 원문 표를 눈으로 옮긴 것(PDF 쪽 5, 6, 9, 16, 21).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
out = {}

# ---- 1. 그림 10 (막대 위 숫자, ms). 캐시 크기 256 512 1024 2048 4096
L = np.array([256, 512, 1024, 2048, 4096], float)
fig10 = {"Llama-2-7B": {"recompute": [63, 103, 223, 523, 1411], "streaming": [31, 31, 35, 45, 65]},
         "Llama-2-13B": {"recompute": [99, 169, 361, 860, 2355], "streaming": [48, 52, 60, 75, 106]}}
r1 = {}
for k, v in fig10.items():
    rc, st = np.array(v["recompute"], float), np.array(v["streaming"], float)
    # 재계산: a*L + b*L^2 + c (c는 고정 오버헤드), 스트리밍: a*L + c
    A = np.stack([L, L ** 2, np.ones_like(L)], 1)
    coef, *_ = np.linalg.lstsq(A, rc, rcond=None)
    pred = A @ coef
    sl = np.polyfit(L, st, 1)
    r1[k] = {"speedup": [round(float(x), 2) for x in rc / st], "max_speedup": round(float((rc / st).max()), 2),
             "recompute_fit_aL_bL2_c": [float(x) for x in coef], "recompute_fit_max_abs_err_ms": round(float(np.abs(pred - rc).max()), 1),
             "recompute_quadratic_share_at_4096": round(float(coef[1] * 4096 ** 2 / rc[-1]), 3),
             "streaming_slope_ms_per_cache_token": float(sl[0]), "streaming_intercept_ms": float(sl[1]),
             "streaming_linear_r2": float(1 - ((np.polyval(sl, L) - st) ** 2).sum() / ((st - st.mean()) ** 2).sum())}
out["fig10"] = r1

# ---- 2. 표 2와 표 6의 같은 구성
t2 = {"Falcon-7B": {"4+2044": 12.12}, "MPT-7B": {"4+2044": 14.99}, "Pythia-12B": {"4+2044": 12.09}, "Llama-2-7B": {"4+4092": 9.59}}
t6 = {"Falcon-7B": {"4+2044": 12.84}, "MPT-7B": {"4+2044": 14.99}, "Pythia-12B": {"4+2044": 12.09}, "Llama-2-7B": {"4+4092": 9.59}}
out["table2_vs_table6"] = {m: {c: {"table2": t2[m][c], "table6": t6[m][c], "same": t2[m][c] == t6[m][c]} for c in t2[m]} for m in t2}
out["table6_falcon_note"] = "표 6 Falcon-7B 행은 4+508과 4+2044가 모두 12.84이고, 굵게 표시된 최솟값 12.34(4+1020)는 표 2의 12.12(4+2044)보다 크다"

# ---- 3. 표 2의 폭발 크기 (윈도만 / 4개 싱크)
t2full = {"Falcon-7B": (17.90, 12.12), "MPT-7B": (460.29, 14.99), "Pythia-12B": (21.62, 12.09), "Llama-2-7B": (3359.95, 9.59)}
out["table2_blowup_ratio_window_over_4sinks"] = {k: round(a / b, 2) for k, (a, b) in t2full.items()}
out["table1_llama13b_window_over_4sinks"] = round(5158.07 / 5.40, 1)

# ---- 4. 표 3 (PG19 첫 표본 1개)
out["table3_vanilla_ratios"] = {"window_only_over_1sink": round(27.87 / 18.49, 3), "1sink_over_2sinks": round(18.49 / 18.05, 3),
                                "note": "평가는 PG19 테스트 첫 표본 하나라 신뢰구간을 만들 수 없다"}

# ---- 5. 표 4: 이항 표본 오차. 문항 수는 논문에 없고 lm-evaluation-harness 기본 분할을 가정
n_items = {"ARC-c": 1172, "ARC-e": 2376, "HellaSwag": 10042, "LAMBADA": 5153, "OBQA": 500, "PIQA": 1838, "Winogrande": 1267}
van = {"ARC-c": 18.6, "ARC-e": 45.2, "HellaSwag": 29.4, "LAMBADA": 39.6, "OBQA": 16.0, "PIQA": 62.2, "Winogrande": 50.1}
snk = {"ARC-c": 19.6, "ARC-e": 45.6, "HellaSwag": 29.8, "LAMBADA": 39.9, "OBQA": 16.6, "PIQA": 62.6, "Winogrande": 50.8}
rows, ps = {}, []
for k in van:
    p1, p2, n = van[k] / 100, snk[k] / 100, n_items[k]
    se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)      # 짝 없는 두 비율(짝을 지으면 SE가 더 작을 수 있음)
    z = (p2 - p1) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    rows[k] = {"diff_pt": round(snk[k] - van[k], 2), "n_assumed": n, "se_pt": round(100 * se, 2), "z": round(float(z), 2), "p": round(float(p), 3)}
    ps.append(p)
order = np.argsort(ps)
holm, run = [0] * len(ps), 0
for r, i in enumerate(order):
    run = max(run, (len(ps) - r) * ps[i])
    holm[i] = min(1, run)
for (k, v), h in zip(rows.items(), holm):
    v["p_holm"] = round(float(h), 3)
out["table4"] = {"rows": rows, "n_positive_of_7": 7, "sign_test_p": round(2 * 0.5 ** 7, 4),
                 "note": "7개 모두 +0.3~+1.0pt이고 부호검정은 p=0.016이지만 벤치마크가 같은 모델 쌍의 점수라 독립이 아니다. 개별로는 표본 오차 안(Holm 후 전부 p>0.5)"}

# ---- 6. 표 7: 정확도 0이 되는 거리 vs 캐시 크기
lines = [20, 40, 60, 80, 100, 150, 200, 400, 600, 800, 1000]
t7 = {2048: [85.80, 80.35, 79.15, 75.30, 0, 0, 0, 0, 0, 0, 0], 4096: [84.60, 83.80, 82.80, 77.15, 61.60, 68.20, 0, 0, 0, 0, 0],
      8192: [81.15, 81.25, 81.50, 76.40, 50.10, 58.30, 62.75, 0, 0, 0, 0], 16384: [77.65, 77.50, 78.50, 73.80, 40.50, 38.45, 46.90, 45.70, 28.50, 0, 0]}
chk = {}
for C, acc in t7.items():
    pred_zero = [23 * l > C for l in lines]       # 줄 하나가 23토큰
    chk[C] = {"zero_iff_distance_exceeds_cache": all((a == 0) == z for a, z in zip(acc, pred_zero))}
out["table7"] = chk
out["table7_note"] = "정확도 0이 되는 경계는 4개 캐시 모두 23*줄 수 > 캐시 크기와 일치. 경계 안에서 거리에 대해 단조롭지 않다(4+4092: 100줄 61.6 < 150줄 68.2, 4+16380: 100줄 40.5, 200줄 46.9). 표본은 100개라 정확도 SE는 5pt 안팎"
json.dump(out, open(HERE / "paper_tables_recalc_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:3500])
