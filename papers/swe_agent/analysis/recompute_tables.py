"""SWE-agent (arXiv:2405.15793v3) 표 재계산과 통계 검정.

입력
  - 논문 표의 숫자(아래 상수, 출처는 주석의 표 번호)
  - data/exp/{lite,test}/{gpt4,claude3opus}/results.json  (SWE-bench/experiments 저장소의 공개 결과, 총 약 0.7 MB)
출력: analysis/recompute_results.json

실행: python3 -I analysis/recompute_tables.py   (저장소 루트 swe_agent/ 에서)
LLM 호출·Docker·패치 실행 없음. 공개 표와 공개 결과 파일만 계산한다.
"""
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data" / "exp"
OUT = ROOT / "analysis" / "recompute_results.json"


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def mcnemar_exact(b10, b01):
    """b10 = A만 해결, b01 = B만 해결. 양측 exact (이항 검정)."""
    n = b10 + b01
    return 1.0 if n == 0 else float(stats.binomtest(b10, n, 0.5).pvalue)


def nearest_k(pct, n):
    """퍼센트가 k/n 꼴인지: 가장 가까운 k와 그 k/n(%) 및 오차."""
    k = round(pct * n / 100)
    return k, round(100 * k / n, 3), round(pct - 100 * k / n, 3)


res = {}

# ---------------------------------------------------------------- 1. 표 1: k/n 일치와 구간
T1 = {  # (% Resolved, n): 표 1 (n=2294 전체, 300 Lite)
    "full/RAG GPT-4 Turbo": (1.31, 2294), "full/RAG Claude 3 Opus": (3.79, 2294),
    "full/SWE-agent GPT-4 Turbo": (12.47, 2294), "full/SWE-agent Claude 3 Opus": (10.46, 2294),
    "lite/RAG GPT-4 Turbo": (2.67, 300), "lite/RAG Claude 3 Opus": (4.33, 300),
    "lite/Shell-only GPT-4 Turbo": (11.00, 300), "lite/Shell-only GPT-4 Turbo w/o demo": (7.33, 300),
    "lite/SWE-agent GPT-4 Turbo": (18.00, 300), "lite/SWE-agent Claude 3 Opus": (13.00, 300),
}
t1 = {}
for name, (pct, n) in T1.items():
    k, back, err = nearest_k(pct, n)
    t1[name] = {"reported_pct": pct, "n": n, "k": k, "k_over_n_pct": back, "rounding_err": err,
                "wilson95_pct": wilson(k, n)}
res["table1"] = t1

# 본문 주장의 산술
K = {name: v["k"] for name, v in t1.items()}
claims = {}
claims["rel_increase_vs_shell_with_demo"] = round(18.00 / 11.00 - 1, 4)  # '64% 상대 증가'
claims["fold_vs_RAG_gpt4_lite"] = round(18.00 / 2.67, 2)  # '6.7배'
claims["cost_ratio_vs_RAG"] = {"gpt4": round(1.67 / 0.13, 1), "opus": round(2.18 / 0.25, 1)}  # '8–13배'
claims["intro_10.7pp"] = {"as_stated": round(18.00 - 7.33, 2), "note": "SWE-agent(데모 있음) 대 Shell-only(데모 없음)"}
claims["matched_with_demo_pp"] = round(18.00 - 11.00, 2)
claims["matched_without_demo_pp"] = round(16.3 - 7.33, 2)  # 표 3 'w/o demo' 16.3 대 표 1 Shell-only w/o demo 7.33
# 일치 비교의 짝지은 검정은 불가(문항별 결과 없음) → 범위만
def mcnemar_range(ka, kb, n):
    """주변 합(ka, kb, n)만으로 가능한 McNemar 양측 p의 범위. A가 더 큰 쪽이라 가정(ka>=kb)."""
    diff = ka - kb
    # 최소 불일치: b01=0, b10=diff
    p_min_disc = mcnemar_exact(diff, 0)
    # 최대 불일치: b11=max(0, ka+kb-n) 이 최소가 되게 → b10=ka-b11, b01=kb-b11
    b11 = max(0, ka + kb - n)
    p_max_disc = mcnemar_exact(ka - b11, kb - b11)
    return {"diff": diff, "p_if_min_discordance": p_min_disc, "p_if_max_discordance": p_max_disc,
            "b10_max": ka - b11, "b01_max": kb - b11}

claims["shell_vs_sweagent_with_demo"] = mcnemar_range(54, 33, 300)
claims["shell_vs_sweagent_without_demo"] = mcnemar_range(49, 22, 300)
claims["sweagent_vs_RAG_lite"] = mcnemar_range(54, 8, 300)
res["table1_claims"] = claims

# ---------------------------------------------------------------- 2. 표 2: HumanEvalFix
T2 = {"Python": 87.7, "JS": 89.7, "Java": 87.9}
t2 = {}
for lang, pct in T2.items():
    k, back, err = nearest_k(pct, 164)
    t2[lang] = {"reported_pct": pct, "nearest_k_of_164": k, "k_over_164_pct": back, "gap_pp": err,
                "wilson95_pct": wilson(k, 164)}
res["table2"] = t2
res["table2_text_claim"] = {"text_pass_at_1": 88.3, "mean_of_three_languages": round(sum(T2.values()) / 3, 2),
                            "python_only": 87.7, "note": "5.절 본문은 88.3%, 표 2 Python은 87.7%, 세 언어 평균 88.43%"}
# 표 2에서 k/164 꼴이 안 되는 이유 후보: 한 언어를 다른 분모로 계산했거나 여러 번 평균. 가능한 분모 탐색
cands = []
for n in range(100, 400):
    if all(abs(100 * round(p * n / 100) / n - p) < 0.051 for p in T2.values()):
        cands.append(n)
res["table2_possible_denominators_all_three_fit_within_0.05pp"] = cands[:10]

# ---------------------------------------------------------------- 3. 표 3: 어블레이션
BASE = 54  # 18.0% = 표 1 주 실행과 같은 실행
T3 = {
    "Editor: edit action (no linting)": 15.0, "Editor: No edit": 10.3,
    "Search: Iterative": 12.0, "Search: No search": 15.7,
    "File viewer: 30 lines": 14.3, "File viewer: Full file": 12.7,
    "Context: Full history": 15.0, "Context: w/o demo": 16.3,
}
# 표 10에서 얻는 불일치 추정(아래 5.절)을 먼저 계산
P6 = {1: 17.94, 2: 23.89, 3: 27.35, 4: 29.67, 5: 31.33, 6: 32.67}  # 표 10 pass@k (6회 실행, Lite 300)
RUNS = [17.33, 18.00, 18.00, 18.67, 17.33, 18.33]  # 표 10 각 실행
N6, NI = 6, 300


def solve_histogram():
    """pass@k (불편 추정) 6개와 합계 300에서 'c번(0..6) 푼 문항 수' 7개를 푼다."""
    A, b = [], []
    for k in range(1, 7):
        row = [math.comb(N6 - j, k) / math.comb(N6, k) for j in range(7)]
        A.append(row)
        b.append(NI * (1 - P6[k] / 100))
    A.append([1] * 7)
    b.append(NI)
    sol = np.linalg.solve(np.array(A), np.array(b))
    return sol


hist = solve_histogram()
hist_round = [float(round(x, 2)) for x in hist]
# 소수 둘째 자리 반올림 오차에 대한 민감도: pass@k 값을 ±0.005 흔들어 최솟값/최댓값
rng = np.random.default_rng(0)
sens = []
for _ in range(2000):
    for k in P6:
        pass
    P6j = {k: v + rng.uniform(-0.005, 0.005) for k, v in P6.items()}
    A, b = [], []
    for k in range(1, 7):
        A.append([math.comb(N6 - j, k) / math.comb(N6, k) for j in range(7)])
        b.append(NI * (1 - P6j[k] / 100))
    A.append([1] * 7)
    b.append(NI)
    sens.append(np.linalg.solve(np.array(A), np.array(b)))
sens = np.array(sens)
hist_lo = [float(round(x, 2)) for x in sens.min(0)]
hist_hi = [float(round(x, 2)) for x in sens.max(0)]

p_hat = np.arange(7) / 6
# 불편 추정: Σ p(1-p) ≈ n/(n-1) Σ ĉ(1-ĉ) (ĉ = c/6, 문항별 성공률의 표본 추정)
sum_pq = float(N6 / (N6 - 1) * np.sum(hist * p_hat * (1 - p_hat)))
sd_run_indep_pp = 100 * math.sqrt(sum_pq) / NI
disc_between_two_runs = 2 * sum_pq  # 독립 두 실행에서 불일치 문항 수의 기대값 = 2Σp(1-p)
obs_sd_pop = float(np.std(RUNS))
obs_sd_samp = float(np.std(RUNS, ddof=1))
# 관측 SD가 독립 가정 SD보다 작은 정도: (n-1)s²/σ² ~ χ²_{5}
chi = 5 * obs_sd_samp ** 2 / sd_run_indep_pp ** 2
res["table10"] = {
    "runs_pct": RUNS, "mean": round(float(np.mean(RUNS)), 3), "sd_population": round(obs_sd_pop, 3),
    "sd_sample": round(obs_sd_samp, 3), "reported_pm": 0.49,
    "runs_as_counts": [round(r * 3) for r in RUNS],
    "pass_at_k_reported": P6,
    "histogram_n_solved_c_of_6_runs": {"c=%d" % j: hist_round[j] for j in range(7)},
    "histogram_sensitivity_min": hist_lo, "histogram_sensitivity_max": hist_hi,
    "n_never_solved": hist_round[0], "n_always_solved": hist_round[6],
    "n_unstable_0<c<6": round(float(300 - hist[0] - hist[6]), 1),
    "sum_p_one_minus_p": round(sum_pq, 2),
    "expected_sd_single_run_pp_if_independent": round(sd_run_indep_pp, 2),
    "expected_discordant_pairs_between_two_runs": round(disc_between_two_runs, 1),
    "chi2_stat_obs_vs_indep": round(chi, 3),
    "p_lower_tail_chi2_df5": round(float(stats.chi2.cdf(chi, 5)), 4),
}

abl = {}
for name, pct in T3.items():
    k, back, err = nearest_k(pct, 300)
    diff = BASE - k
    rng_ = mcnemar_range(BASE, k, 300)
    # (a) 독립 실행 사이 불일치 수준(D)에서 이 차이가 나올 확률: b10=(D+diff)/2, b01=(D-diff)/2
    D = disc_between_two_runs
    b10 = (D + diff) / 2
    b01 = (D - diff) / 2
    z_indep = diff / math.sqrt(D)
    p_indep = float(2 * stats.norm.sf(abs(z_indep)))
    # (b) 표 10의 관측 실행 SD(0.49pp, 모집단 SD)만 쓰면 두 실행 차이 SD = 0.49*sqrt(2)*3 문항
    sd_diff_counts = obs_sd_samp * math.sqrt(2) * 3  # pp*3 = 문항
    z_obs = diff / sd_diff_counts
    abl[name] = {
        "reported_pct": pct, "k": k, "k_over_300_pct": back, "wilson95_pct": wilson(k, 300),
        "baseline_minus_variant_instances": diff, "diff_pp": round(diff / 3, 2),
        "mcnemar_p_if_min_discordance": rng_["p_if_min_discordance"],
        "mcnemar_p_if_max_discordance": rng_["p_if_max_discordance"],
        "z_if_discordance_like_between_runs": round(z_indep, 2),
        "p_if_discordance_like_between_runs_normal_approx": round(p_indep, 4),
        "z_if_run_noise_is_observed_sd": round(z_obs, 2),
    }
res["table3_baseline"] = {"k": BASE, "pct": 18.0, "wilson95_pct": wilson(BASE, 300),
                          "mean_of_6_runs_pct": round(float(np.mean(RUNS)), 2),
                          "note": "표 3의 기준(18.0%)은 표 1 주 실행과 같고 6회 평균(17.94)과 거의 같다(고른 결과가 아님)."}
res["table3"] = abl

# ---------------------------------------------------------------- 4. 부록 표 5(탐색), 표 13, 본문 수치
T5_best = {"GPT-4 Turbo": [14.1, 15.1, 9.2, 10.8, 10.8, 12.4, 8.7, 10.8]}
res["table5"] = {"n_dev_instances": 37, "samples_per_config": 5, "n_configs_per_model": 8,
                 "one_instance_pp": round(100 / 37, 2),
                 "gpt4_best_minus_second_pp": 15.1 - 14.1,
                 "gpt4_best_minus_second_in_instances": round((15.1 - 14.1) * 37 / 100, 2),
                 "wilson95_one_run_at_15pct_n37": wilson(round(0.151 * 37), 37)}

t13 = {
    "full_gpt4": {"submit": 1589, "exit_cost_submit": 630, "exit_cost_nosubmit": 48, "early_exit": 1,
                  "resolved_submit": 266, "resolved_exit_cost_submit": 20},
    "full_opus": {"submit": 882, "exit_cost_submit": 1048, "exit_cost_nosubmit": 73, "early_exit": 1,
                  "resolved_submit": 206, "resolved_exit_cost_submit": 35},
    "lite_gpt4": {"submit": 203, "exit_cost_submit": 95, "exit_cost_nosubmit": 2, "early_exit": 0,
                  "resolved_submit": 50, "resolved_exit_cost_submit": 4},
    "lite_opus": {"submit": 133, "exit_cost_submit": 156, "exit_cost_nosubmit": 11, "early_exit": 0,
                  "resolved_submit": 32, "resolved_exit_cost_submit": 3},
}
out13 = {}
for key, d in t13.items():
    total = d["submit"] + d["exit_cost_submit"] + d["exit_cost_nosubmit"] + d["early_exit"]
    resolved = d["resolved_submit"] + d["resolved_exit_cost_submit"]
    nominal = 2294 if key.startswith("full") else 300
    out13[key] = {
        "n_trajectories_in_table13": total, "n_nominal": nominal, "missing": nominal - total,
        "resolved_in_table13": resolved, "resolved_pct_of_nominal": round(100 * resolved / nominal, 2),
        "resolved_rate_submit_pct": round(100 * d["resolved_submit"] / d["submit"], 2),
        "resolved_rate_exit_cost_submit_pct": round(100 * d["resolved_exit_cost_submit"] / d["exit_cost_submit"], 2),
        "share_resolved_before_budget_pct": round(100 * d["resolved_submit"] / resolved, 2),
        "share_all_submitted_pct_of_total": round(100 * d["submit"] / total, 2),
        "share_all_submitted_pct_of_nominal": round(100 * d["submit"] / nominal, 2),
    }
res["table13"] = out13
res["table13_paper_claims"] = {
    "resolved_submit_rate_text_pct": 14.3, "exit_cost_rate_text_pct": 3.1,
    "share_resolved_before_budget_text_pct": 93.0, "share_all_submitted_text_pct": 69.0,
    "our_resolved_submit_rate_pct": out13["full_gpt4"]["resolved_rate_submit_pct"],
    "our_exit_cost_rate_pct": out13["full_gpt4"]["resolved_rate_exit_cost_submit_pct"],
    "our_share_resolved_before_budget_pct": out13["full_gpt4"]["share_resolved_before_budget_pct"],
    "our_share_all_submitted_pct": out13["full_gpt4"]["share_all_submitted_pct_of_total"],
}
# 예산 소진이 성공을 막았는지: 소진 실행의 해결률 대 제출 실행의 해결률(관찰 비교이고 인과가 아님)
g = t13["full_gpt4"]
exh = g["exit_cost_submit"] + g["exit_cost_nosubmit"]
res["budget_exhaustion_full_gpt4"] = {
    "n_exhausted": exh, "share_of_trajectories_pct": round(100 * exh / 2268, 1),
    "resolved_among_exhausted": g["resolved_exit_cost_submit"],
    "resolved_rate_pct": round(100 * g["resolved_exit_cost_submit"] / exh, 2),
    "wilson95_pct": wilson(g["resolved_exit_cost_submit"], exh),
    "note": "소진 실행은 이미 막힌 문항일 가능성이 커서(선택 편향) '예산을 늘려도 소용없다'는 증거로는 약하다. 예산을 바꾼 대조 실행이 없다.",
}

# 그림 8: 실패 범주(Lite, GPT-4 Turbo, n=248)
FIG8 = {"Incorrect Implementation": 39.9, "Overly Specific": 12.1, "Failed to Recover from Edit": 23.4,
        "Failed to Find Edit Location": 12.9, "Failed to Find Relevant File": 2.0, "Gave Up Prematurely": 2.4,
        "Can't Reproduce": 4.8, "Ran Out of Time": 2.4}
cnt = {k: round(v * 248 / 100) for k, v in FIG8.items()}
res["figure8"] = {"percent": FIG8, "sum_pct": round(sum(FIG8.values()), 1), "counts_if_n248": cnt,
                  "sum_counts": sum(cnt.values()), "lite_unresolved_by_table1": 300 - 54,
                  "note": "본문은 n=248로 쓰지만 표 1 기준 Lite 미해결은 246개",
                  "llm_judge_agreement": {"agree": 13, "n": 15, "pct": round(100 * 13 / 15, 1), "wilson95_pct": wilson(13, 15),
                                          "paper_says_pct": 87}}

# 편집 실패와 복구(5.2절)
res["edit_failure_text"] = {"trajectories_with_failed_edit": 1185, "n": 2294, "pct": round(100 * 1185 / 2294, 2),
                            "paper_pct": 51.7, "wilson95_pct": wilson(1185, 2294)}

# ---------------------------------------------------------------- 5. 표 6·7 합계 (본문 표 1과 일관한가)
T6_n = {"astropy": 6, "django": 114, "matplotlib": 23, "seaborn": 4, "flask": 3, "requests": 6, "xarray": 5,
        "pylint": 6, "pytest": 17, "sklearn": 23, "sphinx": 16, "sympy": 77}
T6_gpt4 = [16.67, 26.32, 13.04, 25.00, 0.00, 33.33, 0.00, 16.67, 17.65, 17.39, 6.25, 10.39]
T6_opus = [33.33, 16.67, 13.04, 0.00, 0.00, 16.67, 0.00, 0.00, 5.88, 17.39, 6.25, 5.19]
ns = list(T6_n.values())
res["table6_sums"] = {
    "n_total": sum(ns),
    "gpt4_resolved": sum(round(p * n / 100) for p, n in zip(T6_gpt4, ns)),
    "opus_resolved": sum(round(p * n / 100) for p, n in zip(T6_opus, ns)),
}
T7_n = [30, 57, 42, 66, 105]
T7_gpt4 = [23.33, 21.05, 23.81, 10.61, 17.14]
T7_opus = [13.33, 17.54, 11.90, 7.58, 10.48]
res["table7_sums"] = {"n_total": sum(T7_n),
                      "gpt4_resolved": sum(round(p * n / 100) for p, n in zip(T7_gpt4, T7_n)),
                      "opus_resolved": sum(round(p * n / 100) for p, n in zip(T7_opus, T7_n))}

# ---------------------------------------------------------------- 6. 공개 결과 파일(문항별)
def load(split, model):
    r = json.loads((EXP / split / model / "results.json").read_text())
    return r


R = {(s, m): load(s, m) for s in ("lite", "test") for m in ("gpt4", "claude3opus")}
pub = {}
for (s, m), r in R.items():
    lst = r["resolved"]
    cnt = Counter(lst)
    ids = set(r["generated"]) | set(r["no_generation"])
    pub[f"{s}/{m}"] = {
        "resolved_list_len": len(lst), "resolved_unique": len(cnt),
        "duplicated_ids_in_resolved": sum(1 for v in cnt.values() if v > 1),
        "extra_entries_from_duplicates": sum(v - 1 for v in cnt.values()),
        "ids_seen_in_generated_or_no_generation": len(ids),
        "generated_len": len(r["generated"]), "generated_unique": len(set(r["generated"])),
        "no_generation_len": len(r["no_generation"]), "no_generation_unique": len(set(r["no_generation"])),
        "generated_and_no_generation_overlap": len(set(r["generated"]) & set(r["no_generation"])),
    }
res["public_results_integrity"] = pub

N = {"lite": 300, "test": 2294}
sets = {(s, m): set(R[s, m]["resolved"]) for (s, m) in R}
paper_pct = {("lite", "gpt4"): 18.00, ("lite", "claude3opus"): 13.00, ("test", "gpt4"): 12.47, ("test", "claude3opus"): 10.46}
cmp_paper = {}
for key, S in sets.items():
    s, m = key
    n = N[s]
    k = len(S)
    cmp_paper[f"{s}/{m}"] = {
        "public_unique_resolved": k, "public_pct": round(100 * k / n, 2), "paper_table1_pct": paper_pct[key],
        "paper_k": round(paper_pct[key] * n / 100), "wilson95_pct": wilson(k, n),
        "raw_list_len": len(R[key]["resolved"]), "raw_list_pct": round(100 * len(R[key]["resolved"]) / n, 2),
    }
res["public_vs_paper_table1"] = cmp_paper

# 짝지은 비교: GPT-4 대 Opus (같은 문항)
def paired(A, B, n, label):
    a, b = len(A - B), len(B - A)
    both = len(A & B)
    neither = n - len(A | B)
    p = mcnemar_exact(a, b)
    # 짝지은 차이의 Wilson이 아닌 Newcombe 근사 대신, 불일치 문항의 이항 비율 구간
    ci = [round(x, 4) for x in (stats.binomtest(a, a + b, 0.5).proportion_ci(method="wilson"))] if a + b else None
    return {"label": label, "A_only": a, "B_only": b, "both": both, "neither": neither,
            "A_total": len(A), "B_total": len(B), "diff_pp": round(100 * (len(A) - len(B)) / n, 2),
            "mcnemar_exact_p": p, "A_share_of_discordant_wilson95": ci,
            "cohen_kappa_like_overlap_jaccard": round(both / len(A | B), 3)}


res["paired"] = {
    "lite_gpt4_vs_opus": paired(sets["lite", "gpt4"], sets["lite", "claude3opus"], 300, "Lite: GPT-4 대 Opus"),
    "full_gpt4_vs_opus_dedup": paired(sets["test", "gpt4"], sets["test", "claude3opus"], 2294, "Full: GPT-4 대 Opus(중복 제거)"),
}
# 논문 표 1의 Opus 수치(주변 합)만 쓸 때 GPT-4와의 McNemar 범위
res["paper_marginals_gpt4_vs_opus"] = {"lite_54_vs_39": mcnemar_range(54, 39, 300), "full_286_vs_240": mcnemar_range(286, 240, 2294)}
# Opus의 Lite 실행과 Full 실행(Lite 문항만)은 같은 모델·설정의 별개 실행인가
lite_ids = set(R["lite", "gpt4"]["generated"]) | set(R["lite", "gpt4"]["no_generation"]) | \
           set(R["lite", "claude3opus"]["generated"]) | set(R["lite", "claude3opus"]["no_generation"])
res["lite_id_universe_seen"] = len(lite_ids)
full_opus_on_lite = sets["test", "claude3opus"] & lite_ids
res["opus_lite_run_vs_full_run_on_lite_ids"] = paired(sets["lite", "claude3opus"], full_opus_on_lite, 300,
                                                       "Opus: Lite 실행 대 Full 실행(Lite 문항)")
full_gpt4_on_lite = sets["test", "gpt4"] & lite_ids
res["gpt4_lite_run_vs_full_run_on_lite_ids"] = paired(sets["lite", "gpt4"], full_gpt4_on_lite, 300,
                                                       "GPT-4: Lite 실행 대 Full 실행(Lite 문항)")
# 위 Opus 두 실행의 불일치는 표 10의 6회 실행 사이 불일치(기대 약 36쌍)와 비슷한 크기인가
res["table10_expected_discordance_for_comparison"] = round(disc_between_two_runs, 1)

# 저장소별 합계(공개 resolved_by_repo)가 표 6과 맞는가
byrepo = {}
for m in ("gpt4", "claude3opus"):
    d = json.loads((EXP / "lite" / m / "resolved_by_repo.json").read_text())
    byrepo[m] = {"resolved": sum(v["resolved"] for v in d.values()), "total": sum(v["total"] for v in d.values())}
res["lite_resolved_by_repo_sums_public"] = byrepo
d = json.loads((EXP / "test" / "claude3opus" / "resolved_by_repo.json").read_text())
res["full_opus_resolved_by_repo_public"] = {"resolved_sum": sum(v["resolved"] for v in d.values()),
                                            "total_sum": sum(v["total"] for v in d.values())}
d = json.loads((EXP / "test" / "claude3opus" / "resolved_by_time.json").read_text())
res["full_opus_resolved_by_time_public"] = {"resolved_sum": sum(v["resolved"] for v in d.values()),
                                            "total_sum": sum(v["total"] for v in d.values())}

# 파일 F1(본문 B.9: 59.05%) - 공개 file_f1.json 과 대조. 본문은 분할을 적지 않았다.
f1res = {}
for sp, m in (("lite", "gpt4"), ("test", "gpt4"), ("lite", "claude3opus")):
    f1 = json.loads((EXP / sp / m / "file_f1.json").read_text())
    vals = f1["f1_scores"]
    f1res[f"{sp}/{m}"] = {"n_entries": len(vals), "mean_pct": round(100 * float(np.mean(vals)), 2),
                          "share_f1_zero_pct": round(100 * float(np.mean(np.array(vals) == 0)), 1)}
f1res["paper_says_pct"] = 59.05
f1res["note"] = "59.05%는 Lite GPT-4(302개 항목, 중복 2개 포함)에서 재현된다. 본문 B.9는 분할을 적지 않았고 전체 2,294 기준은 50.94%."
res["file_f1_public"] = f1res

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
print("wrote", OUT.relative_to(ROOT))
