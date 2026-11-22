"""AgenTracer 논문 표 1·2의 재계산 검사.
(a) 셀 값이 k/N(2자리 반올림)으로 나오는 분모 N을 찾는다. 표 3의 테스트 크기와 맞는지 본다.
(b) 본문·초록의 개선폭 주장(18.18, 12.21, 9.21, 14.21, 17.57, 26.0, 12.2, 22.68)이 표 셀 차이로 재현되는지 찾는다.
(c) 주요 비교를 Fisher 정확 검정으로 본다. N은 (a)에서 추정한 값과 2배 값을 쓴다(추정이라 민감도 확인).
입력: 논문 표 값을 아래에 옮겨 적은 것(출처: paper/paper.txt의 Table 1, Table 2).
실행: python3 -I analysis/recompute_tables.py -> analysis/recompute_results.json
"""
import json, math, os
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "recompute_results.json")
MODELS = ["Qwen3-8B", "Llama-3.2-3B", "Qwen3-32B", "Qwen3-Coder", "GPT-4.1", "DeepSeek-R1", "Gemini-2.5-Pro", "Claude-Sonnet-4", "AgenTracer-8B"]
# 각 열: 9개 모델 x (w/G, w/o G). Table 1 (Who&When) 과 Table 2 (TracerTraj)
T = {
 "WW-hand-agent": [(42.10,39.50),(37.93,50.00),(44.80,44.80),(51.72,60.35),(43.10,37.93),(56.90,53.44),(51.72,51.72),(56.90,50.00),(69.10,63.82)],
 "WW-hand-step":  [(1.72,3.45),(1.72,3.45),(1.72,1.72),(8.62,13.79),(3.44,3.44),(13.29,6.90),(9.72,6.90),(17.24,18.97),(20.68,20.68)],
 "WW-auto-agent": [(58.73,60.32),(37.30,45.23),(63.49,57.93),(42.86,36.50),(55.55,59.52),(66.67,65.08),(61.11,57.14),(57.93,51.11),(69.62,63.73)],
 "WW-auto-step":  [(3.97,5.56),(2.38,8.73),(9.52,8.73),(34.13,32.54),(29.52,21.90),(31.32,29.52),(29.52,25.86),(40.65,38.83),(42.86,37.30)],
 "Code-agent": [(45.35,32.99),(13.38,11.81),(62.99,63.78),(69.29,66.92),(59.84,53.54),(11.81,11.81),(70.07,66.92),(65.98,63.78),(72.95,72.21)],
 "Code-step": [(2.36,1.15),(2.36,3.93),(2.36,1.75),(14.96,14.17),(12.59,11.02),(10.23,10.23),(11.02,6.29),(15.51,11.02),(18.85,18.85)],
 "MATH-agent": [(31.74,33.58),(15.87,14.28),(17.46,17.46),(28.57,33.58),(39.55,35.18),(42.68,38.58),(58.79,58.79),(46.03,50.79),(59.32,66.10)],
 "MATH-step": [(9.52,12.96),(4.76,3.17),(4.76,7.93),(11.11,22.22),(35.81,26.63),(29.52,18.51),(32.22,27.40),(38.10,46.03),(57.63,57.63)],
 "Agentic-agent": [(27.93,30.16),(8.11,13.15),(30.55,30.55),(40.67,43.12),(43.61,40.67),(45.12,46.19),(37.16,32.18),(55.20,49.13),(53.28,50.61)],
 "Agentic-step": [(13.49,15.31),(2.18,5.09),(18.60,15.31),(24.99,25.69),(24.99,25.69),(27.13,21.80),(18.60,17.04),(30.33,29.80),(36.17,35.55)],
}
# 표 3의 테스트 크기(논문), 공개 릴리스 테스트 크기, 논문 Table 2 열이 가장 잘 맞는 N
PAPER_N = {"WW-hand": 58, "WW-auto": 63, "Code": 147, "MATH": 63, "Agentic": 56}
RELEASE_N = {"Code": 127, "MATH": 63, "Agentic": 600}

def fits(v, N):
    return any(abs(round(100 * k / N, 2) - v) < 1e-9 for k in range(N + 1))

def fit_scan(name, rows):
    vals = [x for r in rows for x in r]
    all_fit = [N for N in range(20, 1001) if all(fits(v, N) for v in vals)]
    best = sorted(range(20, 401), key=lambda N: -sum(fits(v, N) for v in vals))[:3]
    return {"cells": len(vals), "N_all_cells_fit_20_1000": all_fit[:10],
            "top_N_by_cells_fit": [[N, sum(fits(v, N) for v in vals)] for N in best]}

def gap(name, rows, N):
    vals = [x for r in rows for x in r]
    g = [abs(100 * round(v * N / 100) / N - v) for v in vals]
    return {"N": N, "max_gap_pp": round(max(g), 2), "cells_exact": sum(x < 0.006 for x in g), "cells": len(g)}

res = {"fit_scan": {}, "gap_at_stated_N": {}, "claims": {}, "fisher": []}
for n, rows in T.items():
    res["fit_scan"][n] = fit_scan(n, rows)
res["gap_at_stated_N"]["WW-hand-agent@58"] = gap("", T["WW-hand-agent"], 58)
res["gap_at_stated_N"]["WW-hand-step@58"] = gap("", T["WW-hand-step"], 58)
res["gap_at_stated_N"]["WW-auto-agent@126"] = gap("", T["WW-auto-agent"], 126)
res["gap_at_stated_N"]["Code-agent@147(paper Table3)"] = gap("", T["Code-agent"], 147)
res["gap_at_stated_N"]["Code-agent@127(release)"] = gap("", T["Code-agent"], 127)
res["gap_at_stated_N"]["Agentic-agent@56(paper Table3)"] = gap("", T["Agentic-agent"], 56)
res["gap_at_stated_N"]["Agentic-agent@600(release)"] = gap("", T["Agentic-agent"], 600)

# (b) 주장 수치 탐색: AgenTracer(index 8)와 각 모델의 차이(pp)와 상대차(%)
targets = {"18.18": 18.18, "12.21": 12.21, "9.21": 9.21, "14.21": 14.21, "17.57": 17.57, "26.0": 26.0, "12.2": 12.2, "22.68": 22.68, "9.04": 9.04}
hits = []
for n, rows in T.items():
    for side, sname in ((0, "w/G"), (1, "w/oG")):
        for i in range(9):
            for j in range(9):
                if i == j: continue
                x, y = rows[i][side], rows[j][side]
                for tag, val in (("pp", x - y), ("rel%", (x - y) / y * 100 if y else float("nan"))):
                    for t, tv in targets.items():
                        if abs(val - tv) < 0.006:
                            hits.append({"cell": n, "setting": sname, "A": MODELS[i], "B": MODELS[j], "metric": tag, "value": round(val, 3), "claim": t})
res["claims"]["matching_cell_differences"] = hits
res["claims"]["max_AgenTracer_minus_Gemini_pp_WhoWhen"] = round(max(T[c][8][s] - T[c][6][s] for c in ["WW-hand-agent", "WW-hand-step", "WW-auto-agent", "WW-auto-step"] for s in (0, 1)), 2)
res["claims"]["max_AgenTracer_minus_Gemini_pp_all_cells"] = round(max(T[c][8][s] - T[c][6][s] for c in T for s in (0, 1)), 2)
res["claims"]["R1_drop_wo_G_MATH_agent_step"] = [round(T["MATH-agent"][5][1] - T["MATH-agent"][5][0], 2), round(T["MATH-step"][5][1] - T["MATH-step"][5][0], 2)]

# (c) Fisher 정확 검정. 분모는 추정치(1배, 2배 민감도)
def fisher(cell, a_idx, b_idx, side, n):
    a = T[cell][a_idx][side]; b = T[cell][b_idx][side]
    ka, kb = round(a * n / 100), round(b * n / 100)
    _, p = fisher_exact([[ka, n - ka], [kb, n - kb]])
    return {"cell": cell, "setting": "w/G" if side == 0 else "w/oG", "A": MODELS[a_idx], "B": MODELS[b_idx], "N_assumed": n,
            "A_pct": a, "B_pct": b, "diff_pp": round(a - b, 2), "p_fisher": round(p, 3)}
for n_mult in (1, 2):
    for cell, b_idx, side, n in [
        ("WW-hand-agent", 7, 0, 58), ("WW-hand-agent", 6, 0, 58), ("WW-hand-agent", 7, 1, 58),
        ("WW-hand-step", 7, 0, 58), ("WW-auto-step", 7, 0, 63), ("WW-auto-step", 5, 0, 63),
        ("WW-auto-agent", 5, 0, 63), ("MATH-step", 6, 0, 63), ("MATH-step", 7, 1, 63),
        ("Agentic-step", 7, 0, 56), ("Agentic-step", 0, 0, 56), ("Code-agent", 3, 0, 127), ("Code-step", 7, 0, 127)]:
        res["fisher"].append(fisher(cell, 8, b_idx, side, n * n_mult) | {"n_multiplier": n_mult})

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ["claims"]}, ensure_ascii=False, indent=1)[:3000])
for f in res["fisher"]:
    print(f)
for k, v in res["fit_scan"].items():
    print(k, v["N_all_cells_fit_20_1000"], v["top_N_by_cells_fit"][:1])
for k, v in res["gap_at_stated_N"].items():
    print(k, v)
