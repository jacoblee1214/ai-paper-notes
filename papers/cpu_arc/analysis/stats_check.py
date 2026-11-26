"""CPU-ARC 정확도 주장 재계산: 표본 수 추정, Wilson 구간, 동등성(차이 구간·MDE), 교란 통과율의 구간.

문항별 결과가 공개되지 않아 표 2의 백분율만 쓴다. 표본 수는 논문에 없으므로 벤치마크 정의와 백분율 격자에서 추정한다.
실행: python3 -I stats_check.py  (결과: stats_check_results.json)
"""
import json, math, os, sys
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_tables import TABLE2, T2_KEYS

here = os.path.dirname(os.path.abspath(__file__))
res = {}

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def consistent_ns(pcts, lo=1, hi=13000):
    """모든 백분율(소수 첫째 자리 반올림)을 정수 개수로 만들 수 있는 n"""
    out = []
    for n in range(lo, hi + 1):
        ok = True
        for p in pcts:
            if not any(abs(round(100 * k / n, 1) - p) < 1e-9 for k in (round(p * n / 100) - 1, round(p * n / 100), round(p * n / 100) + 1) if 0 <= k <= n):
                ok = False
                break
        if ok:
            out.append(n)
    return out

# 벤치마크 정의상 알려진 크기 (논문 3.3절 + 공개 정의)
KNOWN = {"MATH-500": 500, "AIME24": 30, "AIME25": 30, "AIME26": 30, "GPQA-Diamond": 198}
res["n_inference"] = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    ns = consistent_ns([r["s_acc"], r["c_acc"], r["fid"]], 1, 1500)
    entry = {"consistent_n_up_to_1500": ns[:12], "count": len(ns)}
    if k in KNOWN:
        entry["known_n"] = KNOWN[k]
        entry["known_n_consistent"] = KNOWN[k] in ns
    res["n_inference"][k] = entry
# GSM8K 테스트 1,319, MMLU-Pro 테스트 12,032, SciBench 텍스트 문항(논문 표본 수 미기재)
for k, n in (("GSM8K", 1319), ("MMLU-Pro", 12032)):
    r = dict(zip(T2_KEYS, TABLE2[k]))
    res["n_inference"][k]["candidate_full_set_n"] = n
    res["n_inference"][k]["candidate_consistent"] = n in consistent_ns([r["s_acc"], r["c_acc"], r["fid"]], n, n)

# 사용할 n: 알려진 n, 없으면 (a) 가장 작은 격자 일치 n, (b) 후보 전체 n 둘 다 계산
N_USE = {"MATH-500": 500, "AIME24": 30, "AIME25": 30, "AIME26": 30, "GPQA-Diamond": 198, "GSM8K": 1319}
# MMLU-Pro와 SciBench는 표본 수가 논문에 없다: 격자와 맞는 가장 작은 n(하한 시나리오)을 쓰고, 전체 집합 n을 대안으로 따로 본다.
N_ALT = {"MMLU-Pro": 12032, "SciBench": 869}
for k in ("MMLU-Pro", "SciBench"):
    ns = res["n_inference"][k]["consistent_n_up_to_1500"]
    N_USE[k] = ns[0] if ns else None
for k, n in N_ALT.items():
    r = dict(zip(T2_KEYS, TABLE2[k]))
    res["n_inference"][k]["alt_full_set_n"] = n
    res["n_inference"][k]["alt_consistent"] = n in consistent_ns([r["s_acc"], r["c_acc"], r["fid"]], n, n)

def cp_upper(k, n, alpha=0.05):
    if k >= n:
        return 1.0
    return stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

# 정의상 일관성: 정답 불일치 문항 수 <= 답 불일치 문항 수  =>  |C Acc - S Acc| <= 100 - Fidelity
res["definition_consistency"] = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    gap = abs(r["c_acc"] - r["s_acc"])
    cap = 100 - r["fid"]
    res["definition_consistency"][k] = {"|C-S|_pp": round(gap, 1), "100-fidelity_pp": round(cap, 1), "consistent": bool(gap <= cap + 0.05)}

res["accuracy_intervals"] = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    n = N_USE[k]
    row = {"n_used": n, "n_source": "benchmark definition" if k in KNOWN or k == "GSM8K" else "smallest n consistent with the three percentages"}
    for name in ("s_acc", "c_acc", "fid"):
        kk = round(r[name] * n / 100)
        lo, hi = wilson(kk, n)
        row[name] = {"k": kk, "pct": r[name], "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
    # 짝지은 차이: 정답 불일치 문항 수 <= 비일치 문항(1 - fid) * n. 이 상한으로 McNemar 가능 범위
    n_disagree_max = n - round(r["fid"] * n / 100)
    diff_k = round(r["c_acc"] * n / 100) - round(r["s_acc"] * n / 100)
    row["fidelity_disagreements_max"] = n_disagree_max
    row["net_diff_items"] = diff_k
    row["net_diff_pp"] = round(100 * diff_k / n, 2)
    # 비짝 차이 95% 구간 (정규 근사, 불일치율을 모르는 보수적 상한 = 비짝)
    p1, p2 = r["c_acc"] / 100, r["s_acc"] / 100
    se_unpaired = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    row["diff_unpaired_95pp"] = [round(100 * (p1 - p2 - 1.96 * se_unpaired), 1), round(100 * (p1 - p2 + 1.96 * se_unpaired), 1)]
    # 짝지은 비교: 정답 정오가 갈린 문항은 답이 갈린 문항의 부분집합이므로 d <= 100 - fidelity (관측).
    # 관측 불일치율이 0이어도 모집단 불일치율은 0이 아니므로 Clopper-Pearson 95% 상한을 쓴다.
    d_obs = max(n_disagree_max, abs(diff_k))
    d_up = cp_upper(d_obs, n)
    row["discordance_obs_upper_items"] = d_obs
    row["discordance_rate_95_upper_pct"] = round(100 * d_up, 1)
    row["MDE_pp_80power(2.80*sqrt(d_up/n))"] = round(100 * 2.80 * math.sqrt(d_up / n), 1)
    se_p = math.sqrt(d_up / n)
    ci90 = (diff_k / n - 1.645 * se_p, diff_k / n + 1.645 * se_p)
    row["tost_90ci_pp(d=CP upper)"] = [round(100 * ci90[0], 1), round(100 * ci90[1], 1)]
    row["equiv_within_2pp"] = bool(ci90[0] > -0.02 and ci90[1] < 0.02)
    row["equiv_within_5pp"] = bool(ci90[0] > -0.05 and ci90[1] < 0.05)
    res["accuracy_intervals"][k] = row

# 교란 통과율: 분모(변형 수)가 논문에 없다. 가능한 분모별 구간 폭을 보여 준다.
res["perturbation_pass_rate_intervals"] = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    base_n = round(r["c_acc"] * N_USE[k] / 100)  # 교란 시험 대상 문항 수 = 골든 일치 문항
    row = {"items_eligible(c_acc*n)": base_n}
    for nvar in (1, 3, 5, 10):
        n_tot = base_n * nvar
        for mode in ("phrasal", "param", "comb"):
            kk = round(r[mode] * n_tot / 100)
            lo, hi = wilson(kk, n_tot)
            row[f"{mode}_x{nvar}"] = [round(100 * lo, 1), round(100 * hi, 1)]
    # 문항 단위 군집(변형이 같은 문항에서 나옴): 문항 수 기준 구간이 하한 (변형 1개/문항)
    res["perturbation_pass_rate_intervals"][k] = row
    # 소스 대비: 소스 정확도 대 교란 통과율
    row["source_error_pct"] = round(100 - r["s_acc"], 1)
    row["combined_fail_pct"] = round(100 - r["comb"], 1)
    row["combined_fail_over_source_error"] = round((100 - r["comb"]) / max(100 - r["s_acc"], 0.1), 1)

# Holm 보정: 8개 벤치마크 C-S 차이를 McNemar 정확 검정 최선 사례(불일치 = |net|)에서
res["holm"] = {}
pvals = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    n = N_USE[k]
    diff_k = round(r["c_acc"] * n / 100) - round(r["s_acc"] * n / 100)
    n_dis = n - round(r["fid"] * n / 100)
    # 최선(가장 작은 p): 불일치 문항이 모두 한쪽 방향. 최악: 불일치 n_dis 중 균형
    b_best = abs(diff_k)  # 한쪽 승리 문항 수
    m_best = abs(diff_k)
    p_best = stats.binomtest(b_best, m_best, 0.5).pvalue if m_best > 0 else 1.0
    m_worst = max(n_dis, abs(diff_k))
    b_worst = (m_worst + abs(diff_k)) // 2
    p_worst = stats.binomtest(b_worst, m_worst, 0.5).pvalue if m_worst > 0 else 1.0
    pvals[k] = (p_best, p_worst)
order = sorted(pvals, key=lambda k: pvals[k][0])
m = len(order)
holm = {}
run = 0.0
for i, k in enumerate(order):
    adj = min(1.0, (m - i) * pvals[k][0])
    run = max(run, adj)
    holm[k] = round(run, 4)
res["holm"] = {"p_best_case(all discordant one-sided)": {k: round(pvals[k][0], 5) for k in order},
               "p_worst_case(balanced discordant up to 1-fid)": {k: round(pvals[k][1], 4) for k in order},
               "holm_adjusted_best_case": holm}

# 교란 통과율은 "원본에서 이미 정답인 문항"만 대상이다(3.2절). 원본에서 틀린 문항이 교란에서 모두 틀린다고 두고(하한 가정)
# 문항 전체에 대한 조건 없는 정확도를 추정하고 소스 원본 정확도와 비교한다. 소스의 교란 문항 정확도는 논문에 없다.
res["unconditional_variant_accuracy_estimate"] = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    row = {m: round(r["c_acc"] * r[m] / 100, 1) for m in ("phrasal", "param", "comb")}
    row["source_acc_on_originals"] = r["s_acc"]
    row["comb_minus_source_pp"] = round(row["comb"] - r["s_acc"], 1)
    res["unconditional_variant_accuracy_estimate"][k] = row

with open(os.path.join(here, "stats_check_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
for k in TABLE2:
    a = res["accuracy_intervals"][k]
    print(k, "n", a["n_used"], "| S", a["s_acc"]["wilson95"], "C", a["c_acc"]["wilson95"], "diff", a["net_diff_pp"], a["diff_unpaired_95pp"],
          "d_up", a["discordance_rate_95_upper_pct"], "MDE", a["MDE_pp_80power(2.80*sqrt(d_up/n))"], "tost", a["tost_90ci_pp(d=CP upper)"], "eq2", a["equiv_within_2pp"], "eq5", a["equiv_within_5pp"])
print(res["definition_consistency"])
for k in ("MMLU-Pro","SciBench"):
    print(k, res["n_inference"][k])
print(json.dumps(res["holm"]["holm_adjusted_best_case"]))

print(json.dumps(res["unconditional_variant_accuracy_estimate"], ensure_ascii=False))
