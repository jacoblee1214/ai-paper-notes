"""그림 복원값(figure_values.json)과 공개 데이터셋 크기로 논문 주장을 재계산합니다.
논문과 공개 저장소에는 모델 출력, 채점 결과, hh-rlhf 특징 라벨이 없으므로 문항 단위 검정은 못 하고,
(1) 그림에서 복원한 전환 비율 x 데이터셋 문항 수로 짝지은 검정을 근사하고
(2) 문항 수로 Wilson 구간과 최악 경우 오차를 붙이고 (3) 그림 사이의 산술 일치를 확인합니다.
입력: figure_values.json (extract_figures.py가 만듦). 결과: recompute_results.json
"""
import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "figure_values.json")))
OUT = {}

M5 = ["Claude 1.3", "Claude 2", "GPT-3.5", "GPT-4", "LLaMA 2"]
M5B = ["GPT-3.5", "GPT-4", "Claude 1.3", "Claude 2", "LLaMA 2"]
N_AYS = {"TruthfulQA MC": 817, "AQuA": 254, "MATH (CoT)": 1000, "MMLU (CoT)": 1000, "TruthfulQA freeform": 817, "TriviaQA": 1000}
N_FB = {"math": 1000, "arguments": 300, "poems": 400}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def mcnemar_exact(b, c):
    """b=오답->정답, c=정답->오답. 양측 정확 검정(이항 p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5).pvalue)


# ---------------------------------------------------------------- 1. 피드백 (그림 1, 10)
f1, f10 = R["fig1_feedback_positivity"], R["fig10_feedback_by_dataset_axes_in_page_order"]
doms = ["math", "arguments", "poems"]
S1 = {"macro_check": {}, "metric": {}, "per_domain": {}}
for m in M5:
    S1["metric"][m] = {
        "like_minus_dislike": round(f1["really_like"][m]["value"] - f1["really_dislike"][m]["value"], 1),
        "wrote_minus_not_wrote": round(f1["wrote"][m]["value"] - f1["did_not_write"][m]["value"], 1),
    }
    for ser in ["really_like", "really_dislike", "wrote", "did_not_write"]:
        v = [f10[i][ser][m]["value"] for i in range(3)]
        mean = float(np.mean(v)); se_dom = float(np.std(v, ddof=1) / math.sqrt(3))
        S1["macro_check"][f"{m}|{ser}"] = {"fig1": f1[ser][m]["value"], "mean_of_fig10_domains": round(mean, 2), "reported_se": f1[ser][m]["se"],
                                          "se_across_3_domains": round(se_dom, 2)}
    S1["per_domain"][m] = {d: {"like_minus_dislike": round(f10[i]["really_like"][m]["value"] - f10[i]["really_dislike"][m]["value"], 1),
                               "wrote_minus_not_wrote": round(f10[i]["wrote"][m]["value"] - f10[i]["did_not_write"][m]["value"], 1),
                               "did_not_write_positivity": f10[i]["did_not_write"][m]["value"]} for i, d in enumerate(doms)}
# 단일 도메인 이항 오차(참고): did_not_write가 50과 구분되는지 (n = 도메인 문항 수)
S1["did_not_write_vs_50"] = {}
for m in M5:
    for i, d in enumerate(doms):
        v = f10[i]["did_not_write"][m]["value"]; n = N_FB[d]
        z = (v - 50) / (100 * math.sqrt(0.25 / n))
        S1["did_not_write_vs_50"][f"{m}|{d}"] = {"positivity": v, "n": n, "z_vs_50": round(z, 2)}
rat = []
for ser in ["really_like", "really_dislike", "wrote", "did_not_write"]:
    for m in M5:
        v = f1[ser][m]; p = v["value"] / 100
        rat.append(v["se"] / (100 * math.sqrt(p * (1 - p) / 1700)))
S1["reported_se_over_pooled_binomial_se_n1700"] = {"mean": round(float(np.mean(rat)), 2), "min": round(min(rat), 2), "max": round(max(rat), 2), "sqrt3": round(math.sqrt(3), 3)}
OUT["feedback"] = S1

# ---------------------------------------------------------------- 2. "확실해요?" (그림 13-17)
OFF = -0.14  # 100% 막대가 100.14로 복원되므로 보정
T = R["fig17_transition_fractions"]
S2 = {"cells": {}, "model_macro": {}, "grid_check": {}}
sig = {"drop_sig": 0, "rise_sig": 0, "ns": 0}
for d, nd in N_AYS.items():
    for m in M5B:
        t = T[d][m]
        c2c, c2i, i2c, i2i = (t.get(k, 0.0) for k in ["c_to_c", "c_to_i", "i_to_c", "i_to_i"])
        k_c2i, k_i2c = round(c2i * nd), round(i2c * nd)
        k_init, k_second = round((c2c + c2i) * nd), round((c2c + i2c) * nd)
        p = mcnemar_exact(k_i2c, k_c2i)
        delta = 100 * (k_second - k_init) / nd
        status = "ns" if p >= 0.05 else ("drop" if delta < 0 else "rise")
        sig[{"ns": "ns", "drop": "drop_sig", "rise": "rise_sig"}[status]] += 1
        a1 = R["fig13_accuracy_initial_vs_second"][d]["initial"][m]["value"] + OFF
        a2 = R["fig13_accuracy_initial_vs_second"][d]["second"][m]["value"] + OFF
        S2["cells"][f"{d}|{m}"] = {
            "n": nd, "init_acc_fig13": round(a1, 1), "second_acc_fig13": round(a2, 1), "init_acc_from_fig17": round(100 * k_init / nd, 1),
            "second_acc_from_fig17": round(100 * k_second / nd, 1), "init_wilson95": wilson(k_init, nd), "second_wilson95": wilson(k_second, nd),
            "c_to_i": k_c2i, "i_to_c": k_i2c, "net_change_pp": round(delta, 1), "mcnemar_exact_p": float(f"{p:.3g}"),
            "p_change_given_correct": round(100 * c2i / (c2c + c2i), 1) if (c2c + c2i) else None,
            "p_change_given_incorrect": round(100 * i2c / (i2c + i2i), 1) if (i2c + i2i) else None,
            "churn_pct_of_questions": round(100 * (k_c2i + k_i2c) / nd, 1), "status": status}
S2["significance_counts_30_cells_alpha_0.05"] = sig
bonf = sum(1 for c in S2["cells"].values() if c["mcnemar_exact_p"] < 0.05 / 30)
S2["significant_after_bonferroni_30"] = bonf
for m in M5B:
    drops = [R["fig13_accuracy_initial_vs_second"][d]["initial"][m]["value"] - R["fig13_accuracy_initial_vs_second"][d]["second"][m]["value"] for d in N_AYS]
    pcc = [S2["cells"][f"{d}|{m}"]["p_change_given_correct"] for d in N_AYS]
    ch = [R["fig16_changes_answer_all_questions"][d][m]["value"] + OFF for d in N_AYS]
    ad = [R["fig15_admits_mistake_all_questions"][d][m]["value"] + OFF for d in N_AYS]
    S2["model_macro"][m] = {"mean_acc_drop_6ds": round(float(np.mean(drops)), 1), "mean_drop_excl_AQuA": round(float(np.mean([x for x, d in zip(drops, N_AYS) if d != "AQuA"])), 1),
                           "mean_P_c_to_i_given_correct_from_fig17": round(float(np.mean(pcc)), 1), "fig2b_reported": R["fig2_are_you_sure"]["b_changes_to_incorrect_after_correct"][m]["value"],
                           "fig2b_se_reported": R["fig2_are_you_sure"]["b_changes_to_incorrect_after_correct"][m]["se"],
                           "se_across_6_datasets_of_fig17_values": round(float(np.std(pcc, ddof=1) / math.sqrt(6)), 1),
                           "mean_admits_all_q_fig15": round(float(np.mean(ad)), 1), "mean_changes_answer_all_q_fig16": round(float(np.mean(ch)), 1)}
# 정오 격자 점검: AQuA(n=254)는 1/254=0.39%라 격자가 거칠어 n이 맞는지 알 수 있다
g = []
for m in M5B:
    for k in ("initial", "second"):
        v = R["fig13_accuracy_initial_vs_second"]["AQuA"][k][m]["value"] + OFF
        for n in (200, 254, 300):
            kk = v * n / 100
            g.append((n, abs(kk - round(kk))))
S2["grid_check_aqua"] = {str(n): round(float(np.mean([e for nn, e in g if nn == n])), 3) for n in (200, 254, 300)}
# 2차: MATH(CoT)는 2지선다라 우연 수준 50%
S2["math_below_chance"] = {m: {"init_acc": S2["cells"][f"MATH (CoT)|{m}"]["init_acc_fig13"], "wilson95": S2["cells"][f"MATH (CoT)|{m}"]["init_wilson95"]} for m in M5B}
OUT["are_you_sure"] = S2

# ---------------------------------------------------------------- 3. 답 아첨 (그림 3, 11)
f11 = R["fig11_answer_sycophancy_by_dataset"]
S3 = {"macro_check": {}, "per_cell": {}}
NA = {"TriviaQA": 1000, "TruthfulQA": 817}
for m in M5:
    for cond in ["think_correct", "dont_think_correct", "think_incorrect"]:
        vals = [f11[d][cond][m]["value"] for d in ("TriviaQA", "TruthfulQA")]
        S3["macro_check"][f"{m}|{cond}"] = {"fig3": R["fig3_answer_sycophancy_delta_acc"][cond][m]["value"], "mean_of_fig11": round(float(np.mean(vals)), 2),
                                           "fig3_se": R["fig3_answer_sycophancy_delta_acc"][cond][m]["se"],
                                           "se_across_2_datasets": round(abs(vals[0] - vals[1]) / 2 / math.sqrt(1) * 1.0, 2)}
        for d, v in zip(("TriviaQA", "TruthfulQA"), vals):
            n = NA[d]
            se_wc = 100 * math.sqrt(2 * 0.25 / n)  # 기준선과 조건 정확도가 독립이고 p=0.5일 때의 최악 경우 표준오차
            S3["per_cell"][f"{m}|{cond}|{d}"] = {"delta_pp": v, "n": n, "worst_case_unpaired_se_pp": round(se_wc, 2), "z_worst_case": round(v / se_wc, 2),
                                                "exceeds_1.96_worst_case": bool(abs(v) > 1.96 * se_wc)}
S3["count_exceeding_by_condition"] = {c: sum(1 for k, v in S3["per_cell"].items() if k.split("|")[1] == c and v["exceeds_1.96_worst_case"]) for c in ["think_correct", "dont_think_correct", "think_incorrect"]}
OUT["answer"] = S3

# ---------------------------------------------------------------- 4. 흉내 (그림 4, 12): n=300 격자 복원
f12 = R["fig12_mimicry_other_outcomes"]
S4 = {"counts": {}}
for m in M5:
    cnt = {}
    for key in ("a_neither", "b_both", "c_only_correct"):
        v = f12[key].get(m, {"value": 0.0})["value"] - (0.06 if m in f12[key] else 0.0)
        cnt[key] = round(v * 3)
    k_inc = 300 - sum(cnt.values())
    fig4 = R["fig4_mimicry_only_incorrect_attribution"][m]["value"] - 0.06
    ment = 300 - cnt["a_neither"]
    S4["counts"][m] = {**cnt, "mentions_correct_attribution_pct": round(100 * (cnt["b_both"] + cnt["c_only_correct"]) / 300, 1),
                       "only_incorrect_given_any_attribution_pct": round(100 * k_inc / ment, 1), "only_incorrect": k_inc, "only_incorrect_pct": round(100 * k_inc / 300, 1), "fig4_value": round(fig4, 2), "wilson95_n300": wilson(k_inc, 300),
                       "wilson95_if_only_15_poems_independent": wilson(round(k_inc / 20), 15)}
ks = {m: S4["counts"][m]["only_incorrect"] for m in M5}
S4["pairwise_fisher_p"] = {f"{a} vs {b}": float(f"{stats.fisher_exact([[ks[a], 300 - ks[a]], [ks[b], 300 - ks[b]]])[1]:.3g}") for a in M5 for b in M5 if a < b}
OUT["mimicry"] = S4

# ---------------------------------------------------------------- 5. 사람 선호 분석 (그림 5)
F5 = R["fig5_feature_effects"]
S5 = {"features": [], "ci95_excludes_50": 0, "ci95_excludes_50_names": [], "identical_pairs": []}
for r in F5:
    ex = (r["ci95"][0] > 50) or (r["ci95"][1] < 50)
    S5["features"].append({**r, "ci95_excludes_50": bool(ex), "ci95_halfwidth_pp": round((r["ci95"][1] - r["ci95"][0]) / 2, 2)})
    if ex:
        S5["ci95_excludes_50"] += 1; S5["ci95_excludes_50_names"].append(r["feature"])
for a, b in zip(F5, F5[1:]):
    if a["ci95"] == b["ci95"] and a["median"] == b["median"]:
        S5["identical_pairs"].append([a["feature"], b["feature"]])
sd = lambda r: (r["ci95"][1] - r["ci95"][0]) / 3.92
mb, au, tr = F5[0], F5[1], F5[4]
S5["match_vs_authoritative_z_independent"] = round((mb["median"] - au["median"]) / math.hypot(sd(mb), sd(au)), 2)
S5["match_vs_truthful_z_independent"] = round((mb["median"] - tr["median"]) / math.hypot(sd(mb), sd(tr)), 2)
S5["holdout_accuracy_71.3_n1000_wilson95"] = wilson(713, 1000)
S5["logit_of_top_effect"] = round(math.log(mb["median"] / (100 - mb["median"])), 3)
S5["laplace_prior_scale_b"] = 0.01
S5["prior_logdensity_penalty_at_top_effect_nats"] = round(S5["logit_of_top_effect"] / 0.01, 1)
OUT["preference_regression"] = S5

# ---------------------------------------------------------------- 6. BoN, RL (그림 6)
f6 = R["fig6_bon_and_rl"]
S6 = {}
for nm, n in [("feedback", 300), ("answer", 1817), ("mimicry", 300)]:
    c, s = f6[nm]["claude2_pm"], f6[nm]["non_syc_pm"]
    # 지표 하나의 최악 경우 표준오차: feedback은 like-dislike 두 긍정도의 차(각 n=300) -> sqrt(2*.25/n), answer는 3조건 평균 정확도 변화, mimicry는 비율
    se1 = 100 * math.sqrt(0.25 / n) if nm == "mimicry" else 100 * math.sqrt(2 * 0.25 / n)
    S6[nm] = {"n_prompts_assumed": n, "claude2_pm_N1_to_N32": [c[0], c[-1]], "non_syc_pm_N1_to_N32": [s[0], s[-1]], "gap_at_N32_pp": round(c[-1] - s[-1], 1),
              "worst_case_se_single_metric_pp": round(se1, 1), "gap_over_se_unpaired_diff": round((c[-1] - s[-1]) / (se1 * math.sqrt(2)), 2),
              "claude2_pm_monotone_decrease_or_increase_steps": int(sum(np.sign(np.diff(c)) == np.sign(c[-1] - c[0]))), "non_syc_steps_same_sign": int(sum(np.sign(np.diff(s)) == np.sign(s[-1] - s[0])))}
rl = f6["rl_points_fraction_pct"]
S6["rl"] = {k: {"start": v[0][1], "end": v[-1][1], "peak": max(y for _, y in v), "n_points": len(v)} for k, v in rl.items()}
OUT["bon_rl"] = S6

# ---------------------------------------------------------------- 7. 오해 선호 (그림 7)
f7 = R["fig7_misconceptions"]
S7 = {}
pm = f7["a_pm_prefers_sycophantic_by_difficulty"]["vs_helpful_truthful"]
hm = f7["b_humans_majority_vote_by_difficulty"]
best_n = {}
for n in (31, 32, 33, 34, 35):
    err = [abs(v * n / 100 - round(v * n / 100)) for v in pm + hm]
    best_n[n] = round(float(np.mean(err)), 3)
S7["grid_fit_mean_abs_error_by_assumed_n"] = best_n
N = 33
S7["assumed_n_per_level"] = N
kpm = [round(v * N / 100) for v in pm]; khm = [round(v * N / 100) for v in hm]
S7["pm_vs_helpful_counts"] = kpm; S7["human_majority_counts"] = khm
S7["pm_level8_wilson95"] = wilson(kpm[-1], N); S7["human_majority_level8_wilson95"] = wilson(khm[-1], N)
S7["fisher_pm_vs_human_majority_level8_p"] = float(f"{stats.fisher_exact([[kpm[-1], N - kpm[-1]], [khm[-1], N - khm[-1]]])[1]:.3g}")


def trend(ks, n):
    x = np.arange(1, 9); k = np.array(ks); tot = n * 8
    pbar = k.sum() / tot
    t = (x * (k - n * pbar)).sum()
    var = pbar * (1 - pbar) * n * ((x ** 2).sum() - (x.sum() ** 2) / 8)
    z = t / math.sqrt(var)
    return round(float(z), 2), float(f"{2 * stats.norm.sf(abs(z)):.3g}")


S7["cochran_armitage_trend_pm_helpful"] = trend(kpm, N)
S7["cochran_armitage_trend_human_majority"] = trend(khm, N)
S7["levels_1_to_8_pm_vs_baseline_truthful_pct"] = f7["a_pm_prefers_sycophantic_by_difficulty"]["vs_baseline_truthful"]
S7["bon_all_misconceptions_N4096"] = {k: v[-1] for k, v in f7["c_all_misconceptions"].items()}
S7["bon_hardest_N4096"] = {k: v[-1] for k, v in f7["d_hardest_misconceptions"].items()}
S7["bon_all_misconceptions_N1"] = f7["c_all_misconceptions"]["claude2_pm"][0]
S7["human_comparisons_total"] = 266 * 5
OUT["misconceptions"] = S7

json.dump(OUT, open(os.path.join(HERE, "recompute_results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
