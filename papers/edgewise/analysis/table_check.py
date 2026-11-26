"""표 재계산: 문항 수 역산, Δ 행, 헤드라인(46%, 62%, 34%, 45.16%), 그림 5·7 비용, 표 7, 소표 정합.
실행: python3 table_check.py  ->  table_check_results.json (analysis/ 안에서 실행)"""
import json, math, itertools
from paper_tables import *

out = {}
def isk(acc, n, tol=0.011):
    k = round(acc * n / 100); return k, abs(100 * k / n - acc) < tol
def pct(a, b): return 100 * (a / b - 1)

# 1. 문항 수 역산: 각 n에서 정확도가 k/n(소수 둘째 자리 반올림)인 칸 수
allacc = {"TB": [], "SWE": []}
for s in T1:
    for b in T1[s]:
        for m, v in T1[s][b].items():
            allacc[b] += [("P1", s, m, v[0]), ("P3", s, m, v[2])]
inv = {}
for b, lst in allacc.items():
    inv[b] = {}
    for n in range(40, 400):
        ok = sum(isk(a, n)[1] for *_, a in lst)
        inv[b][n] = ok
    best = max(inv[b].items(), key=lambda x: (x[1], -x[0]))
    inv[b] = {"cells": len(lst), "best_n": best[0], "match": best[1], "n89": inv[b][89], "n110": inv[b][110]}
out["n_inversion"] = inv
# k/(3n) 가설(3회 평균): 표 1 Pass@1이 모두 k/n 배수인지
mult3 = {}
for b in ("TB", "SWE"):
    n = N[b]
    vals = [v[0] for s in T1 for m, v in T1[s][b].items()]
    mult3[b] = {"cells": len(vals), "all_k_over_n": all(isk(a, n)[1] for a in vals),
                "k_divisible_by_3": sum(round(a * n / 100) % 3 == 0 for a in vals)}
out["pass1_single_run_check"] = mult3
# P3 >= P1 단조성 (같은 방법)
out["p3_ge_p1"] = all(v[2] >= v[0] for s in T1 for b in T1[s] for v in T1[s][b].values())

# 2. 비용: Pass@3 비용 = 3회 합계인가? (그림 5 평균 $93 x 3 = ?)
out["p3_cost_vs_3x_mean_fig5"] = {"TB_GPT_P3_cost": 279.73, "3x93": 3 * 93, "SWE_GPT_P3_cost": 598.08, "3x199": 3 * 199,
                                  "TB_P1_cost": 90.31, "SWE_P1_cost": 200.50, "ratio_P3_over_P1_TB_vanilla": round(279.73 / 90.31, 2), "ratio_SWE": round(598.08 / 200.5, 2)}

# 3. Δ 행 재계산
rows = []
for (s, b), d in T1_DELTA.items():
    v, w = T1[s][b]["Vanilla"], T1[s][b]["Self"]
    r = {"setup": s, "bench": b, "d_acc_p1": round(w[0] - v[0], 2), "paper": d[0], "d_cost_p1": round(pct(w[1], v[1]), 2), "paper_cost_p1": d[1],
         "d_acc_p3": round(w[2] - v[2], 2), "paper_acc_p3": d[2], "d_cost_p3": round(pct(w[3], v[3]), 2), "paper_cost_p3": d[3]}
    r["ok"] = abs(r["d_acc_p1"] - d[0]) < .02 and abs(r["d_cost_p1"] - d[1]) < .02 and abs(r["d_acc_p3"] - d[2]) < .02 and abs(r["d_cost_p3"] - d[3]) < .02
    rows.append(r)
out["delta_rows"] = rows
out["delta_acc_is_pp_not_relative"] = all(abs((100 * (T1[r['setup']][r['bench']]['Self'][0] / T1[r['setup']][r['bench']]['Vanilla'][0] - 100)) - r["paper"]) > 0.2 or r["paper"] == 0 for r in rows)

# 4. 헤드라인
sav1 = {(s, b): -pct(T1[s][b]["Self"][1], T1[s][b]["Vanilla"][1]) for s in T1 for b in T1[s]}
sav3 = {(s, b): -pct(T1[s][b]["Self"][3], T1[s][b]["Vanilla"][3]) for s in T1 for b in T1[s]}
mean = lambda x: sum(x) / len(x)
head = {"p1_savings": {f"{s}-{b}": round(v, 2) for (s, b), v in sav1.items()}, "p3_savings": {f"{s}-{b}": round(v, 2) for (s, b), v in sav3.items()},
        "mean8": round(mean(list(sav1.values()) + list(sav3.values())), 2), "mean_p3_4": round(mean(list(sav3.values())), 2), "mean_p1_4": round(mean(list(sav1.values())), 2),
        "weighted_p3": round(100 * (1 - sum(T1[s][b]["Self"][3] for s in T1 for b in T1[s]) / sum(T1[s][b]["Vanilla"][3] for s in T1 for b in T1[s])), 2),
        "weighted_p1": round(100 * (1 - sum(T1[s][b]["Self"][1] for s in T1 for b in T1[s]) / sum(T1[s][b]["Vanilla"][1] for s in T1 for b in T1[s])), 2),
        "paper_intro_46": 46, "project_page_45.16": 45.16}
head["TB_mean_p3"] = round(mean([sav3[("GPT", "TB")], sav3[("Opus", "TB")]]), 2)
head["SWE_mean_p3"] = round(mean([sav3[("GPT", "SWE")], sav3[("Opus", "SWE")]]), 2)
head["TB_max_all"] = round(max(sav1[("GPT", "TB")], sav1[("Opus", "TB")], sav3[("GPT", "TB")], sav3[("Opus", "TB")]), 2)
head["SWE_max_all"] = round(max(sav1[("GPT", "SWE")], sav1[("Opus", "SWE")], sav3[("GPT", "SWE")], sav3[("Opus", "SWE")]), 2)
head["abstract_62_matches"] = "SWE Opus P3 62.14 (max of P3), P1 62.68 would round to 63"
head["abstract_34_matches"] = f"TB P3 mean of GPT/Opus = {head['TB_mean_p3']} (max would be {head['TB_max_all']})"
out["headline"] = head

# 5. 표 안 모든 방법의 Self 대비: 비용·정확도 파레토 지배 칸 수 (P3)
dom = {}
for s in T1:
    for b in T1[s]:
        d = []
        for m, v in T1[s][b].items():
            if m == "Self": continue
            sv = T1[s][b]["Self"]
            if sv[2] >= v[2] and sv[3] <= v[3]: d.append(m)
        dom[f"{s}-{b}"] = d
out["self_dominates_p3"] = dom
# 이진 위임 대비 자기 위임: 같은 정확도에서 비용
out["self_vs_binary_p3_cost_pct"] = {f"{s}-{b}": round(pct(T1[s][b]["Self"][3], T1[s][b]["Binary"][3]), 1) for s in T1 for b in T1[s]}
out["judge_cost_share_TB_GPT_binary_p1"] = round(5.77 / 59.16 * 100, 1)

# 6. 그림 5 (3회 평균): 정확도 비율, 비용 절감
f5 = {}
for b, (ab, ag, cb, cg) in FIG5.items():
    f5[b] = {"acc_diff_pp": round(ab - ag, 1), "acc_ratio_pct": round(100 * ab / ag, 1), "cost_saving_pct": round(100 * (1 - cb / cg), 1),
             "table1_binary_p1_vs_vanilla_acc_diff": round(T1["GPT"][b]["Binary"][0] - T1["GPT"][b]["Vanilla"][0], 2),
             "table1_binary_p3_vs_vanilla_acc_diff": round(T1["GPT"][b]["Binary"][2] - T1["GPT"][b]["Vanilla"][2], 2)}
out["fig5"] = f5
# 3회 평균 정확도의 정수화 (k/(3n))
out["fig5_k_over_3n"] = {b: {"binary": round(FIG5[b][0] * 3 * N[b] / 100, 1), "vanilla": round(FIG5[b][1] * 3 * N[b] / 100, 1)} for b in FIG5}
out["fig3_only_cloud_k_over_3n"] = {"TB_terminus2": round(71.91 * 267 / 100, 2), "SWE_terminus2": round(57.88 * 330 / 100, 2), "TB_openhands": round(51.31 * 267 / 100, 2)}

# 7. 그림 7(d) DGX Spark
d7 = {}
for b, v in FIG7D.items():
    d7[b] = {"A100_saving": round(100 * (1 - v["A100"][1] / v["Vanilla"][1]), 1), "DGX_saving": round(100 * (1 - v["DGX"][1] / v["Vanilla"][1]), 1),
             "DGX_acc_vs_vanilla_pp": round(v["DGX"][0] - v["Vanilla"][0], 2), "DGX_acc_vs_A100_pp": round(v["DGX"][0] - v["A100"][0], 2),
             "DGX_tasks_vs_A100": round((v["DGX"][0] - v["A100"][0]) * N[b] / 100, 1)}
out["fig7d"] = d7
out["fig7d_cost_note"] = "DGX 비용은 클라우드 API 비용만. 지연: 평균 11.72분, 중앙값 8.30분(TB 과제당, 설정 §G.2). 클라우드 단독의 지연은 보고되지 않음"
# 그림 7 임계값 비용 절감
f7 = {}
for b, v in FIG7.items():
    base = v["LLM only"]
    f7[b] = {t: {"acc": a, "cost": c, "saving_pct": round(100 * (1 - c / base[1]), 1), "acc_vs_llm_pp": round(a - base[0], 1)} for t, (a, c) in v.items() if t not in ("Qwen only", "LLM only")}
out["fig7_threshold"] = f7
# 검증 세트(SWE-Smith) 정확도는 정수 -> n=100 추정
out["fig7_val_n_guess"] = "정확도가 27.0, 59.0, 60.0, 55.0, 57.0으로 모두 정수여서 n=100(또는 50의 배수)로 보임"

# 8. 표 7
t7 = []
for h, (slm, la, lc, sa, sc, dacc, dc) in T7.items():
    t7.append({"harness": h, "slm_acc_k": round(slm * 89 / 100, 2), "slm_is_k_over_89": isk(slm, 89)[1], "llm_k": isk(la, 89), "self_k": isk(sa, 89),
               "d_acc": round(sa - la, 2), "paper_d_acc": dacc, "d_cost": round(pct(sc, lc), 2), "paper_d_cost": dc, "cost_mismatch": abs(pct(sc, lc) - dc) > 0.02})
out["table7"] = t7
out["table7_mean_saving"] = {"recomputed": round(mean([-pct(v[4], v[2]) for v in T7.values()]), 2), "paper_projectpage": 33.04}

# 9. 소표 정합
def k89(a): return isk(a, 89)
out["table2_3_4"] = {
    "T2_SFT": k89(T2["SFT"][0]), "T2_Self": k89(T2["Self"][0]),
    "T3_with": k89(T3["with"]["acc"]), "T3_without": k89(T3["without"]["acc"]),
    "T4": {k: k89(v[1]) for k, v in T4.items()},
    "TMax_saving_pct": round(100 * (1 - 62.64 / 90.31), 1), "TMax_alone": k89(28.09),
    "Judge_binary_total_equals_table1": round(JUDGE["GPT-5.4-mini"][1] + JUDGE["GPT-5.4-mini"][2], 2),
    "Judge_k": {k: k89(v[0]) for k, v in JUDGE.items()},
    "judge_total_cost": {k: round(v[1] + v[2], 2) for k, v in JUDGE.items()},
}
# 표 3: 위임 호출당 비용(1회 실행 총비용 / (89 x 평균 위임 횟수))
per_call = {}
for k, v in T3.items(): per_call[k] = round(v["cost"] / (89 * v["deleg"]), 4)
out["table3_cost_per_delegation_usd"] = per_call
out["table3_cost_ratio_per_call"] = round(per_call["with"] / per_call["without"], 2)
# 위임 횟수와 위임률 40%에서 평균 궤적 길이
out["implied_mean_turns_if_rate_40pct"] = {"with": round(T3["with"]["deleg"] / 0.40, 1), "without": round(T3["without"]["deleg"] / 0.40, 1)}
# 바닐라 한 단계 평균 비용 (평균 궤적 길이 L 가정 시) = 90.31/(89 L)
out["vanilla_cost_per_step_usd_if_L"] = {str(L): round(90.31 / (89 * L), 4) for L in (14.5, 20, 30, 40, 60)}
out["delegated_step_cost_multiple_if_L"] = {str(L): round(per_call["with"] / (90.31 / (89 * L)), 2) for L in (14.5, 20, 30, 40, 60)}

# 10. 무작위 위임 (표 1): 비용 비율과 정확도
rnd = {}
for s in T1:
    for b in T1[s]:
        v, r = T1[s][b]["Vanilla"], T1[s][b]["Random"]
        rnd[f"{s}-{b}"] = {"cost_p3_pct": round(pct(r[3], v[3]), 1), "acc_p3_pp": round(r[2] - v[2], 2), "acc_p1_pp": round(r[0] - v[0], 2), "cost_p1_pct": round(pct(r[1], v[1]), 1)}
out["random_vs_vanilla"] = rnd
out["random_delegation_ratio_note"] = "표 1의 무작위 위임이 몇 %인지 본문에 없음. 그림 3은 약 40%를 서술, 표 1 비용 비율은 TB GPT P3 0.75, SWE 0.76"

json.dump(out, open("table_check_results.json", "w"), ensure_ascii=False, indent=1, default=str)
for k in ("n_inversion", "pass1_single_run_check", "p3_ge_p1", "p3_cost_vs_3x_mean_fig5", "delta_rows", "headline", "self_dominates_p3", "self_vs_binary_p3_cost_pct", "fig5", "fig5_k_over_3n",
          "fig3_only_cloud_k_over_3n", "fig7d", "fig7_threshold", "table7", "table7_mean_saving", "table2_3_4", "table3_cost_per_delegation_usd", "table3_cost_ratio_per_call",
          "implied_mean_turns_if_rate_40pct", "delegated_step_cost_multiple_if_L", "random_vs_vanilla"):
    print(k, json.dumps(out[k], ensure_ascii=False, default=str))
