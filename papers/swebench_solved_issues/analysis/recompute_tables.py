"""논문 표 1·2·4·7·8과 본문 추정치를 공개 결과 파일에서 다시 계산한다 (재현 아님, 재계산).

입력: data/unz/PatchDiff/ (Zenodo 10.5281/zenodo.17074796 의 PatchDiff_0908.7z 를 풀어 둔 곳)
출력: analysis/recompute_results.json
실행: python3 -I analysis/recompute_tables.py   (논문 폴더에서)
"""
import csv, glob, json, math, os, sys
from collections import Counter
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = os.path.join(ROOT, "data", "unz", "PatchDiff")
R = os.path.join(P, "results")
TOOLS = {"CodeStory": "20241221_codestory_midwit_claude-3-5-sonnet_swe-search",
         "LearnByInteract": "20250110_learn_by_interact_claude3.5",
         "OpenHands": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022_verified"}
N_INST = 500


def wilson(k, n):
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return [round(100 * ci.low, 1), round(100 * ci.high, 1)]


def mcnemar_exact(b, c):
    """짝지은 이진 결과의 정확 McNemar(양측). b,c = 불일치 쌍의 두 방향 개수."""
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5).pvalue)


def load(path):
    with open(path) as f:
        return json.load(f)


def nonempty(d):
    return {i for i, l in d.items() if any(e["differential_tests"] for e in l)}


out = {}
resolved, rq1_all, rq1_func, susp = {}, {}, {}, {}
for t, folder in TOOLS.items():
    resolved[t] = set(load(os.path.join(P, "data", "tool_results", folder, "results.json"))["resolved"])
    r1 = load(os.path.join(R, f"RQ1_{t}_runall.json"))
    rq1_all[t] = set(r1)
    rq1_func[t] = {i for i, v in r1.items() if v["difference"] == "functionality"}
    susp[t] = nonempty(load(os.path.join(R, f"RQ2_{t}_difftests.json")))

# ---- 표 1: 모든 개발자 테스트 실행 ----
t1 = {}
for t in TOOLS:
    n = len(resolved[t]); k = len(rq1_func[t])
    t1[t] = dict(resolved=n, resolved_pct=round(100 * n / N_INST, 1),
                 rq1_files=len(rq1_all[t]), rq1_difference_counts=dict(Counter(v["difference"] for v in load(os.path.join(R, f"RQ1_{t}_runall.json")).values())),
                 incorrect=k, incorrect_pct=round(100 * k / n, 1), incorrect_ci95=wilson(k, n),
                 inc_subset_of_resolved=rq1_func[t] <= resolved[t],
                 updated_pct=round(100 * (n - k) / N_INST, 1), drop_pp=round(100 * k / N_INST, 1))
tot_n = sum(len(resolved[t]) for t in TOOLS); tot_k = sum(len(rq1_func[t]) for t in TOOLS)
out["table1"] = t1
out["table1_pooled"] = dict(incorrect=tot_k, plausible=tot_n, pct=round(100 * tot_k / tot_n, 2), ci95=wilson(tot_k, tot_n),
                            mean_drop_pp=round(sum(100 * len(rq1_func[t]) / N_INST for t in TOOLS) / 3, 2))

# ---- 표 2: PatchDiff 의심 패치 ----
t2 = {}
for t in TOOLS:
    n = len(resolved[t]); s = susp[t]
    uniq = s - rq1_func[t]
    t2[t] = dict(susp=len(s), susp_pct=round(100 * len(s) / n, 1), susp_ci95=wilson(len(s), n),
                 susp_subset_of_resolved=s <= resolved[t], uniq=len(uniq), overlap_with_rq1=len(s & rq1_func[t]),
                 wo_susp_pct=round(100 * (n - len(s)) / N_INST, 1), drop_pp=round(100 * len(s) / N_INST, 1),
                 wo_susp_and_rq1_pct=round(100 * len(resolved[t] - s - rq1_func[t]) / N_INST, 1))
tot_s = sum(len(susp[t]) for t in TOOLS)
out["table2"] = t2
out["table2_pooled"] = dict(susp=tot_s, plausible=tot_n, pct=round(100 * tot_s / tot_n, 2), ci95=wilson(tot_s, tot_n),
                            mean_drop_pp=round(sum(t2[t]["drop_pp"] for t in TOOLS) / 3, 2),
                            mean_uniq_share=round(100 * sum(t2[t]["uniq"] / t2[t]["susp"] for t in TOOLS) / 3, 1),
                            rq1_nonsusp=sum(len(rq1_func[t] - susp[t]) for t in TOOLS))
# 도구 간 의심 비율 차이: 3x2 카이제곱, 쌍별 Fisher
tab = [[len(susp[t]), len(resolved[t]) - len(susp[t])] for t in TOOLS]
chi2, p, dof, _ = stats.chi2_contingency(tab)
pair = {}
names = list(TOOLS)
for i in range(3):
    for j in range(i + 1, 3):
        a, b = names[i], names[j]
        pair[f"{a}_vs_{b}"] = round(float(stats.fisher_exact([tab[i], tab[j]])[1]), 3)
out["susp_rate_across_tools"] = dict(chi2=round(float(chi2), 2), dof=int(dof), p=round(float(p), 3), fisher_pairs=pair)

# ---- 해결률 순위: LearnByInteract vs OpenHands (같은 500 문항, 짝지은 검정) ----
def paired(a, b):
    only_a = len(a - b); only_b = len(b - a)
    return dict(only_first=only_a, only_second=only_b, mcnemar_p=round(mcnemar_exact(only_a, only_b), 3))
L, O, C = "LearnByInteract", "OpenHands", "CodeStory"
out["paired_resolution"] = {
    "LBI_vs_OH_before": dict(pct=[round(100 * len(resolved[L]) / 500, 1), round(100 * len(resolved[O]) / 500, 1)], **paired(resolved[L], resolved[O])),
    "LBI_vs_OH_wo_susp": dict(pct=[round(100 * len(resolved[L] - susp[L]) / 500, 1), round(100 * len(resolved[O] - susp[O]) / 500, 1)], **paired(resolved[L] - susp[L], resolved[O] - susp[O])),
    "LBI_vs_OH_wo_susp_rq1": dict(pct=[round(100 * len(resolved[L] - susp[L] - rq1_func[L]) / 500, 1), round(100 * len(resolved[O] - susp[O] - rq1_func[O]) / 500, 1)],
                                  **paired(resolved[L] - susp[L] - rq1_func[L], resolved[O] - susp[O] - rq1_func[O])),
    "CS_vs_LBI_before": paired(resolved[C], resolved[L]),
    "CS_vs_OH_before": paired(resolved[C], resolved[O]),
    "CS_vs_LBI_wo_susp": paired(resolved[C] - susp[C], resolved[L] - susp[L]),
    "CS_vs_OH_wo_susp": paired(resolved[C] - susp[C], resolved[O] - susp[O]),
}
# 해결 여부와 의심 여부의 도구 간 일치 (같은 인스턴스를 여러 도구가 푼 경우)
allres = resolved[C] & resolved[L] & resolved[O]
out["cross_tool"] = dict(resolved_by_all3=len(allres),
                         susp_by_any_of_3_among_all3=len([i for i in allres if any(i in susp[t] for t in TOOLS)]),
                         susp_by_all_of_3_among_all3=len([i for i in allres if all(i in susp[t] for t in TOOLS)]))

# ---- 표 4: 다른 LLM으로 같은 100개 표본 ----
t4 = {}
sample = {}
for t in TOOLS:
    ds = load(os.path.join(R, f"RQ2_{t}_deepseek100.json")); qw = load(os.path.join(R, f"RQ2_{t}_qwen100.json"))
    assert set(ds) == set(qw)
    sample[t] = set(ds)
    gpt = susp[t] & set(ds)
    t4[t] = dict(gpt4omini=len(gpt), deepseek=len(nonempty(ds)), qwen=len(nonempty(qw)), sample_subset_of_resolved=set(ds) <= resolved[t])
    t4[t]["_sets"] = (gpt, nonempty(ds), nonempty(qw))
overall = {m: sum(len(t4[t]["_sets"][k]) for t in TOOLS) for k, m in enumerate(["gpt4omini", "deepseek", "qwen"])}
# 짝지은 McNemar: 같은 300 패치에서 모델별 의심 판정
def union(k):
    return {(t, i) for t in TOOLS for i in t4[t]["_sets"][k]}
G, D, Q = union(0), union(1), union(2)
out["table4"] = {t: {k: v for k, v in t4[t].items() if k != "_sets"} for t in TOOLS}
out["table4_overall"] = dict(n=300, **overall, pct={m: round(100 * v / 300, 1) for m, v in overall.items()},
                             mcnemar_gpt_vs_deepseek=paired(G, D), mcnemar_gpt_vs_qwen=paired(G, Q), mcnemar_deepseek_vs_qwen=paired(D, Q),
                             gpt_found_but_not_qwen=len(G - Q), union_all3=len(G | D | Q))

# ---- RQ3/RQ4: 수작업 표본 (RQ34.csv) ----
rows = list(csv.DictReader(open(os.path.join(R, "RQ34.csv"))))
n = len(rows)
pat = Counter(r["diff_pattern"] for r in rows); cor = Counter(r["correctness"] for r in rows)
inc = {k: v for k, v in cor.items() if k.startswith("incorrect")}
n_inc = sum(inc.values()); n_cor = sum(v for k, v in cor.items() if k.startswith("correct")); n_unc = cor["uncertain"]
by_tool = Counter(r["tool"] for r in rows)
out["rq34_counts"] = dict(n=n, pattern=dict(pat), correctness=dict(cor), by_tool=dict(by_tool),
                          expected_by_tool_if_30pct={t: round(0.3 * len(susp[t]), 1) for t in TOOLS})
# 표 7과 표 8 대조
out["table7_check"] = dict(absent=pat["absent"], supplementary=pat["supplementary_more"] + pat["supplementary_change"],
                           sup_more=pat["supplementary_more"], sup_change=pat["supplementary_change"],
                           divergent=pat["divergent"], noalign=pat["noalignment"],
                           pct={k: round(100 * v / n, 1) for k, v in dict(divergent=pat["divergent"], supp=pat["supplementary_more"] + pat["supplementary_change"]).items()})
in_rq1 = [r for r in rows if r["instance_id"] in rq1_func[r["tool"]]]
out["table8_check"] = dict(incorrect=n_inc, incorrect_by_type=inc, correct=n_cor, uncertain=n_unc,
                           incorrect_pct=round(100 * n_inc / n, 1), incorrect_wilson95=wilson(n_inc, n),
                           uncertain_pct=round(100 * n_unc / n, 1),
                           sample_in_rq1_incorrect=len(in_rq1),
                           sample_rq1_share_pct=round(100 * len(in_rq1) / n, 1),
                           population_rq1_share_pct=round(100 * sum(len(susp[t] & rq1_func[t]) for t in TOOLS) / tot_s, 1),
                           fisher_rq1_share_p=round(float(stats.fisher_exact([[len(in_rq1), n - len(in_rq1)],
                                                    [sum(len(susp[t] & rq1_func[t]) for t in TOOLS) - len(in_rq1), tot_s - n - (sum(len(susp[t] & rq1_func[t]) for t in TOOLS) - len(in_rq1))]])[1]), 3))
# 표본 오차 9.39%: 모집단 260, 표본 77, p=0.5, 유한 모집단 보정
fpc = math.sqrt((tot_s - n) / (tot_s - 1))
out["margin_check"] = dict(paper_claim=9.39, computed_with_fpc=round(100 * 1.96 * math.sqrt(0.25 / n) * fpc, 2),
                           computed_without_fpc=round(100 * 1.96 * math.sqrt(0.25 / n), 2))

# ---- 외삽: 11.0%와 6.4%p ----
ratio = n_inc / n
non_susp_rq1 = {t: len(rq1_func[t] - susp[t]) for t in TOOLS}
def extrapolate(frac):
    est = {t: frac * len(susp[t]) + non_susp_rq1[t] for t in TOOLS}
    pooled = sum(est.values())
    infl = {t: round(100 * est[t] / N_INST, 2) for t in TOOLS}
    return dict(est_incorrect=round(pooled, 1), rate_pct=round(100 * pooled / tot_n, 2), inflation_pp_by_tool=infl,
                inflation_mean_pp=round(sum(infl.values()) / 3, 2))
ci = stats.binomtest(n_inc, n).proportion_ci(confidence_level=0.95, method="wilson")
out["extrapolation"] = dict(
    point_22_of_77=extrapolate(ratio),
    wilson_low=extrapolate(ci.low), wilson_high=extrapolate(ci.high),
    lower_bound_only_RQ1=extrapolate(0.0),
    upper_if_all_uncertain_incorrect=extrapolate((n_inc + n_unc) / n),
    note="upper 는 '불확실' 51건을 모두 오답으로 본 극단값. 하한 0.0 은 PatchDiff 의심분을 전부 정답으로 본 값(RQ1만).")
# 외삽 없이 직접 확인된 오답만 센 하한: RQ1 오답 전부 + 수작업 표본에서 새로 확인된 오답
conf = {}
for t in TOOLS:
    new = {r["instance_id"] for r in rows if r["tool"] == t and r["correctness"].startswith("incorrect")} - rq1_func[t]
    conf[t] = len(rq1_func[t]) + len(new)
out["confirmed_only"] = dict(by_tool=conf, total=sum(conf.values()), rate_pct=round(100 * sum(conf.values()) / tot_n, 2),
                             inflation_mean_pp=round(sum(100 * conf[t] / N_INST for t in TOOLS) / 3, 2))
# 도구별 층화 외삽 (도구별 표본의 오답 비율 사용)
strat = {}
for t in TOOLS:
    rt = [r for r in rows if r["tool"] == t]
    k = sum(r["correctness"].startswith("incorrect") for r in rt)
    strat[t] = dict(sample=len(rt), incorrect=k, rate=round(k / len(rt), 3), wilson95=wilson(k, len(rt)))
est_s = {t: strat[t]["rate"] * len(susp[t]) + non_susp_rq1[t] for t in TOOLS}
out["stratified_extrapolation"] = dict(by_tool=strat, rate_pct=round(100 * sum(est_s.values()) / tot_n, 2),
                                       inflation_mean_pp=round(sum(100 * est_s[t] / N_INST for t in TOOLS) / 3, 2))
# 논문 식 그대로: 11.0% x 평균 해결률
out["inflation_identity"] = dict(avg_resolved_pct=round(100 * tot_n / 1500, 2), rate11_times_avg=round(0.110 * 100 * tot_n / 1500, 2))

# ---- 4.5 논의: "추가 변경(supplementary)이 오답 징후" ----
sup = [r for r in rows if r["diff_pattern"].startswith("supplementary")]
oth = [r for r in rows if not r["diff_pattern"].startswith("supplementary")]
ks = sum(r["correctness"].startswith("incorrect") for r in sup); ko = sum(r["correctness"].startswith("incorrect") for r in oth)
pat_inc = {}
for p_ in ["divergent", "supplementary_more", "supplementary_change", "noalignment", "absent"]:
    rp = [r for r in rows if r["diff_pattern"] == p_]
    pat_inc[p_] = dict(n=len(rp), incorrect=sum(r["correctness"].startswith("incorrect") for r in rp),
                       uncertain=sum(r["correctness"] == "uncertain" for r in rp))
out["supplementary_claim"] = dict(supp_incorrect=f"{ks}/{len(sup)}", other_incorrect=f"{ko}/{len(oth)}",
                                  share_of_incorrect_pct=round(100 * ks / n_inc, 1), share_of_sample_pct=round(100 * len(sup) / n, 1),
                                  fisher_p=round(float(stats.fisher_exact([[ks, len(sup) - ks], [ko, len(oth) - ko]])[1]), 3),
                                  by_pattern=pat_inc,
                                  determinable_only=dict(supp=f"{ks}/{ks + sum(r['correctness'].startswith('correct') for r in sup)}"))

with open(os.path.join(HERE, "recompute_results.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=lambda x: sorted(x) if isinstance(x, set) else str(x))
print(json.dumps(out, ensure_ascii=False, indent=1, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
