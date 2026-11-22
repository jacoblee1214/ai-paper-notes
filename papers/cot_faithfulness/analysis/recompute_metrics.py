"""Lanham et al. (2023) 표 2와 그림 3, 4, 6, 7, 10, 11의 재계산·검정.

입력: analysis/figure_values.json (extract_figures.py가 PDF 벡터 그림에서 추출).
출력: analysis/recompute_results.json
실행: python3 -I analysis/recompute_metrics.py   (cot_faithfulness/ 에서)
공개 값만 사용한다. 논문은 문항 수·시드·오차막대 정의를 적지 않았으므로 표본 오차는 오차막대 길이에서 역산한다.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
FV = json.loads((ROOT / "analysis" / "figure_values.json").read_text())

# ---- 표 2 (논문 p.5): 과제, early answering AOC, adding mistakes AOC, 정확도(CoT 없이, CoT 있게), 차이(%p)
TABLE2 = {
    "AQuA": (0.44, 0.52, 28, 43, 15.32),
    "LogiQA": (0.26, 0.31, 42, 43, 1.02),
    "MMLU": (0.12, 0.21, 68, 71, 3.77),
    "HellaSwag": (0.12, 0.23, 71, 66, -4.69),
    "TruthfulQA": (0.11, 0.20, 59, 63, 4.38),
    "OpenBookQA": (0.07, 0.15, 82, 84, 2.71),
    "ARC (Challenge)": (0.05, 0.11, 88, 90, 2.28),
    "ARC (Easy)": (0.02, 0.07, 96, 96, 0.77),
}
TASKS = list(TABLE2)
res = {}

# =========================================================== A. 표 2 내부 일관성과 순위 상관
a = {}
a["accuracy_diff_rounding_check"] = {
    t: {"with_minus_without_rounded": v[3] - v[2], "reported_diff": v[4], "abs_gap": round(abs((v[3] - v[2]) - v[4]), 2)}
    for t, v in TABLE2.items()}
ea = [TABLE2[t][0] for t in TASKS]; am = [TABLE2[t][1] for t in TASKS]
noc = [TABLE2[t][2] for t in TASKS]; cot = [TABLE2[t][3] for t in TASKS]; gain = [TABLE2[t][4] for t in TASKS]

def sp(x, y):
    r = stats.spearmanr(x, y)
    k = stats.kendalltau(x, y)
    return {"spearman_rho": round(float(r.statistic), 3), "spearman_p": float(f"{r.pvalue:.2g}"),
            "kendall_tau": round(float(k.statistic), 3), "kendall_p": float(f"{k.pvalue:.2g}"), "n": len(x)}

a["spearman_AOC_EA_vs_AOC_AM"] = sp(ea, am)
a["spearman_AOC_EA_vs_noCoT_accuracy"] = sp(ea, noc)
a["spearman_AOC_EA_vs_CoT_gain"] = sp(ea, gain)
a["spearman_AOC_AM_vs_CoT_gain"] = sp(am, gain)
# AOC 순서가 CoT 없는 정확도의 역순과 몇 쌍에서 어긋나는가
inv_pairs = sum(1 for i in range(8) for j in range(i + 1, 8) if (ea[i] - ea[j]) * (noc[i] - noc[j]) < 0)
a["discordant_pairs_EA_vs_noCoT_acc"] = {"discordant": inv_pairs, "of": 28, "note": "28쌍 중 AOC가 높은 과제가 CoT 없는 정확도도 높은 쌍의 수"}
# 최소제곱 직선: AOC_EA ~ noCoT 정확도
sl = stats.linregress(noc, ea)
a["linreg_AOC_EA_on_noCoT_acc"] = {"slope": round(sl.slope, 4), "intercept": round(sl.intercept, 3), "r2": round(sl.rvalue ** 2, 3)}
res["A_table2"] = a

# =========================================================== B. 그림 3, 4 곡선에서 AOC 재계산
# AOC_n = 1 - mean(p_i) (i=0..n, 그림 3은 마지막 점이 100%). 그림 4는 실수를 단계 1..n 앞에 넣은 점들만 있다.
def aoc_len(y):
    return 1 - float(np.mean(y)) / 100

lens = ("3-step", "4-step", "5-step", "6-step")
B = {"per_task": {}}
for fig, idx, name in (("fig3_early_answering", 0, "EA"), ("fig4_adding_mistakes", 1, "AM")):
    M = np.array([[aoc_len(FV[fig][t][l]["y"]) for l in lens] for t in TASKS])  # 8x4
    target = np.array([TABLE2[t][idx] for t in TASKS])
    eq = M.mean(axis=1)  # 길이 가중 동일
    # 길이 가중치를 모르므로 합이 1인 비음수 가중으로 최소제곱 (제약: 큰 가중으로 합=1 행 추가)
    A = np.vstack([M, 100 * np.ones(4)]); b = np.concatenate([target, [100.0]])
    w, _ = nnls(A, b)
    fit = M @ w
    B[name] = {
        "equal_weight_vs_table": {t: [round(float(eq[i]), 3), target[i]] for i, t in enumerate(TASKS)},
        "max_abs_gap_equal_weight": round(float(np.abs(eq - target).max()), 3),
        "fitted_length_weights_3to6": [round(float(x), 3) for x in w],
        "max_abs_gap_fitted_weights": round(float(np.abs(fit - target).max()), 3),
        "aoc_by_length": {t: [round(float(M[i, j]), 3) for j in range(4)] for i, t in enumerate(TASKS)},
    }
res["B_aoc_from_figures"] = B

# =========================================================== C. 그림 6 패러프레이즈: 원본 CoT 정확도와의 차이
dev = []
for t in TASKS:
    for l in lens:
        p = np.array(FV["fig6_paraphrase_accuracy"][t][l]["paraphrased"]["y"])
        o = FV["fig6_paraphrase_accuracy"][t][l]["dashed_nonparaphrased"]["y"][0]
        dev.append((t, l, float(np.abs(p - o).max()), float((p - o).mean())))
mean_dev = np.array([d[3] for d in dev])
res["C_paraphrase"] = {
    "cells_task_x_length": len(dev),
    "median_of_max_abs_gap_pts": round(float(np.median([d[2] for d in dev])), 2),
    "max_of_max_abs_gap_pts": [round(max(d[2] for d in dev), 2), max(dev, key=lambda d: d[2])[:2]],
    "mean_signed_gap_pts_all_cells": round(float(mean_dev.mean()), 2),
    "frac_cells_mean_gap_negative": round(float((mean_dev < 0).mean()), 2),
    "note": "패러프레이즈 곡선 점들과 점선(원본 CoT 정확도)의 차이. 오차막대는 논문이 정의하지 않는다.",
}


# =========================================================== C2. 그림 5 필러 토큰 (x=0..100 백분위, 21점)
fil = {}
for t in TASKS:
    y = np.array(FV["fig5_filler_accuracy"][t]["filler_curve"]["y"])
    no_c, with_c = TABLE2[t][2], TABLE2[t][3]
    fil[t] = {"filler_mean": round(float(y.mean()), 1), "filler_at_0pct": round(float(y[0]), 1), "filler_at_100pct": round(float(y[-1]), 1),
             "noCoT_table2": no_c, "withCoT_table2": with_c,
             "filler_minus_noCoT": round(float(y.mean() - no_c), 1),
             "share_of_CoT_gain_recovered": round(float((y.mean() - no_c) / (with_c - no_c)), 2) if with_c - no_c >= 3 else None}
res["C2_filler"] = {"per_task": fil, "note": "필러 곡선 평균을 표 2의 CoT 없는 정확도와 비교. 이득 3%p 이상인 과제만 회복 비율을 적는다."}

# =========================================================== D. 모델 크기 (그림 7, 10, 11)
SZ = [0.81, 1.6, 3.5, 6.4, 13, 22, 52, 175]
F7 = {t: np.array(FV["fig7_same_answer_with_without_cot"][t]["solid"]["y"]) for t in TASKS}
E7 = {t: np.array(FV["fig7_same_answer_with_without_cot"][t]["err_halfwidth_by_x"]) for t in TASKS}
ACC_NO = {t: np.array(FV["fig10_accuracy_by_model_size"][t]["dashed"]["y"]) for t in TASKS}
ACC_CO = {t: np.array(FV["fig10_accuracy_by_model_size"][t]["solid"]["y"]) for t in TASKS}
E10 = {t: np.array(FV["fig10_accuracy_by_model_size"][t]["err_halfwidth_by_x"]) for t in TASKS}
GAIN = {t: np.array(FV["fig11_cot_gain_by_model_size"][t]["solid"]["y"]) for t in TASKS}
i13, i175 = SZ.index(13), SZ.index(175)
D = {}
# D1 "7/8 과제에서 13B가 175B보다 CoT에 더 의존"
d1 = {t: {"13B": round(float(F7[t][i13]), 1), "175B": round(float(F7[t][i175]), 1)} for t in TASKS}
D["claim_13B_lower_than_175B"] = {"per_task": d1, "n_tasks_13B_lower": sum(F7[t][i13] < F7[t][i175] for t in TASKS)}
# D2 13B->175B 단조 증가(= 신뢰성 단조 감소)
mono = {}
for t in TASKS:
    seq = F7[t][i13:]
    mono[t] = {"seq_13_22_52_175": [round(float(x), 1) for x in seq],
               "monotone_nondecreasing": bool(np.all(np.diff(seq) >= 0)),
               "largest_decrease_pts": round(float(-min(0, np.diff(seq).min())), 1)}
D["monotone_13B_to_175B"] = mono
# D3 가장 충실한(같은 답 비율이 가장 낮은) 모델 크기
argmin = {}
for t in TASKS:
    order = np.argsort(F7[t])
    argmin[t] = {"most_faithful_size_B": SZ[order[0]], "value": round(float(F7[t][order[0]]), 1),
                 "runner_up_B": SZ[order[1]], "runner_up_value": round(float(F7[t][order[1]]), 1)}
D["most_faithful_size"] = argmin
# D4 13B 대 175B 차이와 오차막대
z = {}
for t in TASKS:
    diff = F7[t][i175] - F7[t][i13]
    se = np.sqrt(E7[t][i13] ** 2 + E7[t][i175] ** 2)
    z[t] = {"diff_pts_175B_minus_13B": round(float(diff), 1), "errbar_halfwidths": [float(E7[t][i13]), float(E7[t][i175])],
            "ratio_diff_to_errbar_if_bar_is_1SE": round(float(diff / se), 1)}
D["diff_vs_errbars_13B_175B"] = z
# D5 U자형: 작은 모델(0.81B, 1.6B)의 같은 답 비율
D["small_models_same_answer"] = {t: [round(float(F7[t][0]), 1), round(float(F7[t][1]), 1)] for t in TASKS}
D["small_models_accuracy_with_CoT_pts"] = {t: [round(float(ACC_CO[t][0]), 1), round(float(ACC_CO[t][1]), 1)] for t in TASKS}
D["rank_trend_13B_to_175B"] = {"note": "4개 크기에 대한 순위 상관의 정확 양측 p는 최대 단조일 때 0.083이라 순위 검정만으로는 5% 유의에 못 미친다. 이 주장은 쌍별 차이(D4)에 기댄다.",
                              "exact_two_sided_p_perfect_monotone_n4": float(stats.kendalltau([1, 2, 3, 4], [1, 2, 3, 4]).pvalue)}
res["D_model_size"] = D

# =========================================================== E. 정확도 바닥: P(같은 답) >= P(둘 다 정답) >= a + b - 1
floor = {}
rows = []
for t in TASKS:
    for k, s in enumerate(SZ):
        a_, b_ = ACC_NO[t][k], ACC_CO[t][k]
        fl = max(0.0, a_ + b_ - 100)
        rows.append((t, s, F7[t][k], fl, a_, b_))
for t in TASKS:
    rr = [r for r in rows if r[0] == t and r[1] >= 13]
    floor[t] = {str(r[1]) + "B": {"same": round(float(r[2]), 1), "floor_from_accuracy": round(float(r[3]), 1),
                                  "excess_over_floor": round(float(r[2] - r[3]), 1)} for r in rr}
# 175B에서 요약
at175 = {t: floor[t]["175B"] for t in TASKS}
viol = [(r[0], r[1], round(float(r[3] - r[2]), 1)) for r in rows if r[3] - r[2] > 1.0]  # 바닥이 관측값을 1%p 넘게 초과하면 데이터 해석 문제
E = {"floor_definition": "P(같은 답) >= P(CoT 없는 답과 CoT 답이 모두 정답) >= a + b - 1 (a, b = CoT 없는/있는 정확도). 정확도는 그림 10, 같은 답 비율은 그림 7.",
     "at_175B": at175, "violations_floor_gt_observed_by_over_1pt": viol, "per_task_13B_and_up": floor}
# 정확도와 같은 답 비율의 상관 (13B 이상, 과제×크기 32점)
same_ge13 = [r[2] for r in rows if r[1] >= 13]; nocot_ge13 = [r[4] for r in rows if r[1] >= 13]
gain_ge13 = [GAIN[r[0]][SZ.index(r[1])] for r in rows if r[1] >= 13]
E["spearman_same_vs_noCoT_accuracy_13B_up_n32"] = sp(same_ge13, nocot_ge13)
E["spearman_same_vs_CoT_gain_13B_up_n32"] = sp(same_ge13, gain_ge13)
# 선형 R^2: 같은 답 비율 ~ CoT 없는 정확도
sl2 = stats.linregress(nocot_ge13, same_ge13)
E["linreg_same_on_noCoT_acc_13B_up"] = {"slope": round(sl2.slope, 3), "intercept": round(sl2.intercept, 2), "r2": round(sl2.rvalue ** 2, 3)}
# 정확도 바닥 정규화: (same - floor)/(100 - floor) -> 바닥 이상에서 남는 불일치 여유 대비 같은 답 비율

# 정확도로 가능한 범위 [lo, hi] 안에서의 위치: lo = max(0, a+b-100), hi = 100 - |a-b| (P(다른 답) >= |a-b|)
norm = {}
for t in TASKS:
    norm[t] = {}
    for k, s_ in enumerate(SZ):
        a_, b_ = ACC_NO[t][k], ACC_CO[t][k]
        lo, hi = max(0.0, a_ + b_ - 100), 100 - abs(a_ - b_)
        norm[t][f"{s_}B"] = {"lo": round(float(lo), 1), "hi": round(float(hi), 1), "same": round(float(F7[t][k]), 1),
                             "position_in_range": round(float((F7[t][k] - lo) / (hi - lo)), 2) if hi > lo else None}
E["feasible_range_position"] = norm
E["position_13B_vs_175B"] = {t: [norm[t]["13B"]["position_in_range"], norm[t]["175B"]["position_in_range"]] for t in TASKS}
E["floor_share_of_observed_at_175B"] = {t: round(float(at175[t]["floor_from_accuracy"] / at175[t]["same"]), 2) for t in TASKS}
res["E_accuracy_floor"] = E

# 논문 부록 C의 주장: 4/8 과제에서 최대 CoT 이득 크기 = 최대 충실성 크기
agree = {}
for t in TASKS:
    agree[t] = {"max_gain_size_B": SZ[int(np.argmax(GAIN[t]))], "most_faithful_size_B": argmin[t]["most_faithful_size_B"]}
    agree[t]["same"] = agree[t]["max_gain_size_B"] == agree[t]["most_faithful_size_B"]
res["E2_appendixC_claim"] = {"per_task": agree, "n_same": sum(v["same"] for v in agree.values()),
                             "note": "논문은 4/8(ARC E, ARC C, HellaSwag, AQuA)이라 적음. 우리 추출은 LogiQA에서 13B 47.1 대 3.5B 47.3(0.2%p 차)라 5/8."}

# =========================================================== F. 오차막대에서 역산한 유효 문항 수 (그림 10, 오차막대=1SE 가정)
imp = {}
for t in TASKS:
    p = ACC_CO[t] / 100
    n1 = p * (1 - p) / (E10[t] / 100) ** 2
    imp[t] = {"median_n_if_bar_is_1SE": int(np.median(n1)), "range": [int(n1.min()), int(n1.max())],
              "median_n_if_bar_is_1.96SE": int(np.median(n1 * 1.96 ** 2))}
res["F_implied_n_from_fig10_errbars"] = {"per_task": imp,
    "note": "논문이 문항 수와 오차막대 정의를 적지 않아 역산 추정이다. 모델 크기가 달라도 n이 비슷하면 이항 오차막대로 읽는 것이 자연스럽다."}

# =========================================================== G. 그림 간 일관성: 그림 3의 0% 절단 = 그림 7의 175B (논문 p.7)
cons = {}
for t in TASKS:
    x0 = [FV["fig3_early_answering"][t][l]["y"][0] for l in lens]
    cons[t] = {"fig3_0pct_by_length": [round(v, 1) for v in x0], "fig7_175B": round(float(F7[t][i175]), 1),
               "gap_to_mean_pts": round(float(np.mean(x0) - F7[t][i175]), 1)}
res["G_cross_figure"] = cons

# =========================================================== H. 그림 4의 셀별 표본 수 역산
def fits(ys, tol=0.012, maxN=700):
    return [N for N in range(2, maxN) if all(abs(round(y * N / 100) * 100 / N - y) <= tol for y in ys)]
H = {}
for t in TASKS:
    for l in lens:
        f = fits(FV["fig4_adding_mistakes"][t][l]["y"])
        if f:
            H[f"{t} {l}"] = {"smallest_N_consistent": f[0], "binomial_SE_at_p0.5_pts": round(50 / np.sqrt(f[0]), 1)}
res["H_fig4_cell_N"] = {"cells_with_N_below_700": H,
    "note": "곡선 값이 k/N 격자 위에 있다고 보고 N을 역산한다(허용 0.012%p). N이 700 미만으로 식별되는 셀만 적었다. 나머지는 N이 더 크다(식별 불가)."}

(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
# ----- 요약 출력
print("A", json.dumps({k: v for k, v in res["A_table2"].items() if k.startswith(("spearman", "linreg", "discordant"))}, ensure_ascii=False))
print("B EA gap", B["EA"]["max_abs_gap_equal_weight"], B["EA"]["max_abs_gap_fitted_weights"], B["EA"]["fitted_length_weights_3to6"])
print("B AM gap", B["AM"]["max_abs_gap_equal_weight"], B["AM"]["max_abs_gap_fitted_weights"], B["AM"]["fitted_length_weights_3to6"])
print("C", res["C_paraphrase"]); print("C2", {t:(v["filler_mean"],v["noCoT_table2"],v["withCoT_table2"],v["share_of_CoT_gain_recovered"]) for t,v in fil.items()})
print("D1", D["claim_13B_lower_than_175B"]["n_tasks_13B_lower"])
print("D2", {t: v["monotone_nondecreasing"] for t, v in mono.items()})
print("D3", {t: v["most_faithful_size_B"] for t, v in argmin.items()})
print("D4", {t: v["ratio_diff_to_errbar_if_bar_is_1SE"] for t, v in z.items()})
print("E175", at175)
print("Eviol", viol)
print("E corr", E["spearman_same_vs_noCoT_accuracy_13B_up_n32"], E["spearman_same_vs_CoT_gain_13B_up_n32"], E["linreg_same_on_noCoT_acc_13B_up"])
print("Epos", E["position_13B_vs_175B"]); print("Eshare", E["floor_share_of_observed_at_175B"])
print("E2", res["E2_appendixC_claim"]["n_same"])
print("F", {t: v["median_n_if_bar_is_1SE"] for t, v in imp.items()})
print("G", {t: v["gap_to_mean_pts"] for t, v in cons.items()})
print("H", H)
