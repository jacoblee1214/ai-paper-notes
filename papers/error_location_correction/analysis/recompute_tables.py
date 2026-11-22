"""Recompute checkable numbers from the released BIG-Bench Mistake data (arXiv:2311.08516 v3).

Reads data/*.jsonl (upstream WHGTyen/BIG-Bench-Mistake, fetched 2026-11-22).
Model predictions are NOT released, so Table 4/5 cell values cannot be recomputed;
we check counts (Table 2), denominators behind Table 6, macro-averages (Table 4 'Overall'),
the 78 baseline (Table 5), Table 7 differences, and simple tests on the reported numbers.
Writes analysis/recompute_results.json.
"""
import json, math, os
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TASKS = ["word_sorting", "tracking_shuffled_objects", "logical_deduction",
         "multistep_arithmetic", "dyck_languages"]

out = {"table2_counts": {}, "table6_denominators": {}, "table4_overall_macro": {},
       "table5_baseline_f1": {}, "table7_diff_check": {}, "table6_tests": {}}

rows_by_task = {}
for t in TASKS:
    with open(os.path.join(DATA, t + ".jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    rows_by_task[t] = rows
    corr = [r for r in rows if str(r["answer"]).strip() == str(r["target"]).strip()]
    inc = [r for r in rows if str(r["answer"]).strip() != str(r["target"]).strip()]
    cm = lambda rs: sum(1 for r in rs if r["mistake_index"] is not None)
    out["table2_counts"][t] = {
        "total": len(rows), "correct_ans": len(corr), "incorrect_ans": len(inc),
        "incorrect_mis": sum(1 for r in rows if r["mistake_index"] is not None),
        "correct_ans_with_mistake": cm(corr), "incorrect_ans_with_mistake": cm(inc)}

# Table 6: reported delta accuracy (percent) and implied counts over candidate denominators
t6 = {  # task: (oracle_check, oracle_inc, random_check, random_inc)
    "word_sorting": (-11.11, 23.53, -15.56, 11.76),
    "tracking_shuffled_objects": (-6.67, 43.92, -6.67, 20.39),
    "logical_deduction": (-11.43, 36.86, -13.33, 21.57),
    "multistep_arithmetic": (-0.00, 18.04, -8.89, 10.59),
    "dyck_languages": (-6.82, 18.06, -15.91, 5.16),
}
def fits(pct, k, n):
    return abs(100.0 * k / n - abs(pct)) < 0.006 if n else False
for t, (oc, oi, rc, ri) in t6.items():
    c = out["table2_counts"][t]
    n_c = c["correct_ans"]; n_i = c["incorrect_ans"]
    if t == "dyck_languages":  # paper samples Dyck to 88 correctans / 504 incorrectans for Tables 4-6
        n_c = 88
    n_c_m = c["correct_ans_with_mistake"]; n_i_m = c["incorrect_ans_with_mistake"]
    res = {}
    for label, pct, n_all, n_m in [("oracle_correct", oc, n_c, n_c_m), ("oracle_incorrect", oi, n_i, n_i_m),
                                    ("random_correct", rc, n_c, n_c_m), ("random_incorrect", ri, n_i, n_i_m)]:
        ks_all = [k for k in range(0, n_all + 1) if fits(pct, k, n_all)]
        ks_m = [k for k in range(0, n_m + 1) if n_m and fits(pct, k, n_m)]
        res[label] = {"reported_pct": pct, "n_all_group": n_all, "k_fits_all_group": ks_all,
                      "n_group_with_mistake": n_m, "k_fits_with_mistake": ks_m}
    out["table6_denominators"][t] = res

# Table 4 'Overall' = unweighted mean over the 5 tasks? (checked for GPT-4 direct-step, GPT-4-Turbo direct-trace)
# Table 4 full check: reported 'Overall' vs unweighted mean of the five task columns (Direct trace, Direct step, CoT step)
T4 = {
 "GPT-4-Turbo": [[36.33,39.33,21.33,38.33,15.33],[33.00,61.67,75.00,43.33,28.67],[None]*5],
 "GPT-4": [[35.00,62.29,40.67,44.00,17.06],[44.33,65.33,67.67,42.67,44.33],[34.00,90.67,10.33,41.00,41.00]],
 "GPT-3.5-Turbo": [[11.33,10.10,2.00,20.00,8.78],[15.00,1.67,25.33,26.00,5.91],[15.67,19.00,9.67,25.33,1.86]],
 "PaLM 2 Unicorn": [[11.67,18.00,6.67,22.00,10.98],[16.33,28.00,38.00,21.67,14.36],[14.00,55.67,12.00,23.67,17.91]],
}
REP = {"GPT-4-Turbo": [30.13,48.33,None], "GPT-4": [39.80,52.87,43.40],
       "GPT-3.5-Turbo": [10.44,14.78,14.31], "PaLM 2 Unicorn": [17.09,23.67,24.65]}
for m, cols in T4.items():
    for j, vals in enumerate(cols):
        if None in vals: continue
        out["table4_overall_macro"][f"{m} col{j}"] = {"macro_mean": round(sum(vals)/5, 2), "reported_overall": REP[m][j]}

# Table 5 baseline (predict all incorrect): weighted F1 over correct/incorrect labels
def baseline_wf1(n_pos, n_neg):  # positive = incorrect (pred all positive), negatives = correct
    n = n_pos + n_neg
    f1_inc = 2 * (n_pos / n) * 1.0 / ((n_pos / n) + 1.0)
    f1_cor = 0.0
    return (n_pos * f1_inc + n_neg * f1_cor) / n * 100
pooled = [(out["table2_counts"][t]["incorrect_ans"], out["table2_counts"][t]["correct_ans"]) for t in TASKS[:4]]
ni = sum(p[0] for p in pooled); nc = sum(p[1] for p in pooled)
out["table5_baseline_f1"] = {"pooled_4_tasks_300each": round(baseline_wf1(ni, nc), 2),
                             "dyck_sampled_504_88": round(baseline_wf1(504, 88), 2),
                             "reported": 78}

# Table 7: reported difference vs its own two columns
t7 = [("word_sorting", 22.33, 11.67, 11.66), ("tracking_shuffled_objects", 37.67, 18.00, 19.67),
      ("logical_deduction", 6.00, 6.67, -0.67), ("multistep_arithmetic", 26.00, 22.00, 4.00),
      ("dyck_languages", 33.57, 10.98, 22.59)]
for t, a, b, d in t7:
    out["table7_diff_check"][t] = {"computed": round(a - b, 2), "reported": d,
                                   "mismatch": abs((a - b) - d) > 0.005}
# Denominator check for Dyck in Table 7: 33.57% of 592 vs 986; 10.98% of 592 vs 986
for n in (592, 986):  # 592 = 88+504 sampled Dyck, 986 = full Dyck
    out["table7_diff_check"].setdefault("dyck_denominator_check", {})[str(n)] = {
        "33.57_is_k_over_n": [k for k in range(n + 1) if abs(100 * k / n - 33.57) < 0.006],
        "10.98_is_k_over_n": [k for k in range(n + 1) if abs(100 * k / n - 10.98) < 0.006]}

# Simple tests on the reported Table 6 'oracle vs random' in the incorrect-answer group (counts from implied k)
# word sorting: 60/255 oracle vs 30/255 random (11.76%)
def two_prop(k1, n1, k2, n2):
    tab = [[k1, n1 - k1], [k2, n2 - k2]]
    return stats.fisher_exact(tab)[1]
out["table6_tests"]["word_sorting_incorrect_oracle_vs_random_fisher_p"] = round(two_prop(60, 255, 30, 255), 4)
out["table6_tests"]["note"] = ("Single random-location draw, no seeds reported: the random arm has one realization, "
                               "so this p-value is indicative only.")
out["table7_tests"] = {}
for t, a, b, d in t7:
    n = out["table2_counts"][t]["total"]
    if t == "dyck_languages":
        n = 592
    out["table7_tests"][t] = {"n_assumed": n, "fisher_p_two_sided_for_reported_diff": round(
        two_prop(round(a * n / 100), n, round(b * n / 100), n), 4)}


# Wilson 95% CI for the reported incorrectans gains (oracle and random), and oracle vs random Fisher per task
def wilson(k, n, z=1.96):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))
out["table6_incorrect_ci"] = {}
for t, (oc, oi, rc, ri) in t6.items():
    n_i = 504 if t == "dyck_languages" else 255
    ko = round(oi * n_i / 100); kr = round(ri * n_i / 100)
    out["table6_incorrect_ci"][t] = {"oracle_k": ko, "oracle_wilson95": wilson(ko, n_i),
                                     "random_k": kr, "random_wilson95": wilson(kr, n_i),
                                     "fisher_p_oracle_vs_random": round(two_prop(ko, n_i, kr, n_i), 4)}

with open(os.path.join(HERE, "recompute_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
