"""우리 재계산을 논문 표 값(손으로 옮긴 값)과 대조한다. 실행: python3 -I analysis/check_against_paper.py
결과: analysis/check_against_paper_results.json (표별 최대 절대 차이, 불일치 셀)"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bbh = json.load(open(f"{ROOT}/analysis/recompute_bbh_results.json")); bbq = json.load(open(f"{ROOT}/analysis/recompute_bbq_results.json"))
K = {"SA|GPT|ZS": "suggested_answer|GPT-3.5|ZS", "SA|GPT|FS": "suggested_answer|GPT-3.5|FS", "SA|CL|ZS": "suggested_answer|Claude 1.0|ZS",
     "SA|CL|FS": "suggested_answer|Claude 1.0|FS", "AA|GPT|FS": "ans_always_a|GPT-3.5|FS", "AA|CL|FS": "ans_always_a|Claude 1.0|FS"}
# 표 8 (편향-모순): NoCoT 무편향, 편향, CoT 무편향, 편향 / 델타 NoCoT, CoT
T8 = {"SA|GPT|ZS": (57.1, 39.5, 59.6, 23.3, -17.6, -36.3), "SA|GPT|FS": (62.0, 35.0, 75.8, 51.7, -27.0, -24.1),
      "SA|CL|ZS": (59.2, 37.3, 65.3, 34.7, -21.9, -30.6), "SA|CL|FS": (64.2, 38.9, 81.6, 60.1, -25.3, -21.5),
      "AA|GPT|FS": (64.5, 55.2, 77.4, 58.7, -9.3, -18.7), "AA|CL|FS": (69.9, 63.2, 84.8, 80.1, -6.7, -4.7)}
# 표 8 (편향-일관) 델타 NoCoT, CoT
T8C = {"SA|GPT|ZS": (24.1 - 0.0 + 0, 0), }  # 아래에서 별도 입력
T8C = {"SA|GPT|ZS": (24.1, 31.9), "SA|GPT|FS": (29.2, 14.6), "SA|CL|ZS": (23.5, 24.1), "SA|CL|FS": (23.1, 9.7), "AA|GPT|FS": (10.7, 2.4), "AA|CL|FS": (11.6, 1.1)}
# 표 10 (NoCoT %불충실, CoT %불충실, NoCoT %편향설명, CoT %편향설명) — AAA 행은 편향-모순 부분집합 기준으로 비교
T10 = {"SA|GPT|ZS": (27.3, 49.9, 90.9, 91.3), "SA|GPT|FS": (36.3, 28.6, 95.3, 86.6), "SA|CL|ZS": (38.1, 41.2, 77.0, 86.9),
       "SA|CL|FS": (35.4, 23.9, 84.8, 84.4), "AA|GPT|FS": (18.2, 21.8, 65.4, 76.5), "AA|CL|FS": (27.8, 10.7, 45.5, 47.2)}
# 표 13 (CoT/No-CoT 각 4열)
T13 = {("SA|CL|FS", "cot"): (22.3, 0.7, 1.9, 1.9), ("SA|CL|ZS", "cot"): (32.1, 1.1, 2.3, 2.7), ("SA|CL|FS", "nc"): (24.8, 0.8, 3.0, 1.7), ("SA|CL|ZS", "nc"): (21.8, 1.5, 4.5, 2.8),
       ("SA|GPT|FS", "cot"): (24.6, 0.6, 1.8, 1.7), ("SA|GPT|ZS", "cot"): (38.4, 1.6, 1.4, 1.8), ("SA|GPT|FS", "nc"): (26.8, 0.0, 1.2, 0.9), ("SA|GPT|ZS", "nc"): (17.3, 0.1, 1.3, 0.9),
       ("AA|CL|FS", "cot"): (6.1, 2.4, 2.3, 1.4), ("AA|CL|FS", "nc"): (11.0, 3.5, 3.8, 4.6), ("AA|GPT|FS", "cot"): (22.3, 2.8, 1.4, 2.3), ("AA|GPT|FS", "nc"): (11.1, 1.5, 1.8, 2.2)}
# 표 5: (NoCoT %불충실, CoT %불충실, NoCoT %편향설명, CoT %편향설명)
T5 = {"GPT-3.5|ZS|no-instr": (22.1, 26.1, 61.0, 59.2), "GPT-3.5|FS|no-instr": (17.0, 23.5, 60.2, 56.1), "Claude 1.0|ZS|no-instr": (29.5, 25.8, 57.3, 54.5),
      "Claude 1.0|FS|no-instr": (22.8, 20.6, 68.6, 62.5), "GPT-3.5|ZS|debias": (20.5, 24.9, 59.7, 60.0), "GPT-3.5|FS|debias": (15.6, 22.1, 60.7, 51.8),
      "Claude 1.0|ZS|debias": (20.2, 22.5, 48.9, 45.4), "Claude 1.0|FS|debias": (26.0, 17.2, 51.8, 50.6)}
out = {}; cells = {"t8": [], "t8_consistent": [], "t10": [], "t13": [], "t5": []}
for k, vals in T8.items():
    r = bbh["table8"][K[k]]["contradicting"]
    mine = (r["nc"]["acc_unbiased"], r["nc"]["acc_biased"], r["cot"]["acc_unbiased"], r["cot"]["acc_biased"], r["nc"]["delta"], r["cot"]["delta"])
    cells["t8"] += [(k, i, v, round(m, 1), abs(v - round(m, 1))) for i, (v, m) in enumerate(zip(vals, mine))]
for k, vals in T8C.items():
    r = bbh["table8"][K[k]]["consistent"]; mine = (r["nc"]["delta"], r["cot"]["delta"])
    cells["t8_consistent"] += [(k, i, v, round(m, 1), abs(v - round(m, 1))) for i, (v, m) in enumerate(zip(vals, mine))]
for k, vals in T10.items():
    r = bbh["table10"][K[k]]
    ek = "pct_explained_by_bias_contradicting_only" if k.startswith("AA") else "pct_explained_by_bias"  # AAA 행은 편향-모순 라벨 부분집합 기준이라야 논문 값과 일치
    mine = (r["nc"]["pct_unfaithful_overall"], r["cot"]["pct_unfaithful_overall"], r["nc"][ek], r["cot"][ek])
    cells["t10"] += [(k, i, v, round(m, 1), abs(v - round(m, 1))) for i, (v, m) in enumerate(zip(vals, mine))]
for (k, mode), vals in T13.items():
    r = bbh["table13"][K[k]][mode]
    mine = (r["correct_to_biased_wrong"], r["biased_wrong_to_correct"], r["correct_to_other_wrong"], r["other_wrong_to_correct"])
    cells["t13"] += [(f"{k}|{mode}", i, v, round(m, 1), abs(v - round(m, 1))) for i, (v, m) in enumerate(zip(vals, mine))]
for k, vals in T5.items():
    r = bbq[k]; mine = (r["nocot"]["pct_unfaithful_overall"], r["cot"]["pct_unfaithful_overall"], r["nocot"]["pct_explained_by_bias"], r["cot"]["pct_explained_by_bias"])
    cells["t5"] += [(k, i, v, round(m, 1), abs(v - round(m, 1))) for i, (v, m) in enumerate(zip(vals, mine))]
for t, c in cells.items():
    mx = max(x[4] for x in c) if c else 0
    out[t] = dict(n_cells=len(c), max_abs_diff=round(mx, 2), n_diff_over_0p15=sum(1 for x in c if x[4] > 0.15), cells_over_0p15=[x for x in c if x[4] > 0.15])
aaa_all = []
# 표 10 AAA 행: 전체 라벨 기준으로 계산하면 논문 값과 다른지(캡션 불일치 확인)
for k in ["AA|GPT|FS", "AA|CL|FS"]:
    r = bbh["table10"][K[k]]; aaa_all.append(dict(k=k, nc=round(r["nc"]["pct_explained_by_bias"], 1), cot=round(r["cot"]["pct_explained_by_bias"], 1)))
out["t10_aaa_all_labels"] = aaa_all
json.dump(out, open(f"{ROOT}/analysis/check_against_paper_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items()}, indent=1)[:3000])
