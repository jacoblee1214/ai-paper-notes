"""표본 수 기반 통계: Wilson, MDE, 같은 구성 실행 간 불일치(역산), 짝지은 McNemar 시나리오와 Holm, 동등성, 소표 절제의 유의성,
핵심 단계 분석(그림 4)의 p 재구성, 무작위 위임 선택 효과 모의.
과제별 결과가 공개되지 않아 짝지은 불일치율을 시나리오로 둔다.
실행: python3 power_check.py  ->  power_check_results.json (analysis/ 안에서 실행)"""
import json, math
import numpy as np
from scipy import stats
from paper_tables import *

rng = np.random.default_rng(0)
out = {}
Z = 1.959964


def kk(acc, n): return round(acc * n / 100)


def wilson(k, n, z=Z):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def mcnemar_p(D, m):
    """순차이 D(과제 수, Self - 비교 대상), 불일치 쌍 수 m (D와 같은 홀짝으로 맞춤)."""
    if (m - D) % 2: m += 1
    b = (m + D) // 2
    return stats.binomtest(b, m, 0.5).pvalue, m


def paired_ci(D, m, n, z=Z):
    se = math.sqrt(max(m - D * D / n, 1e-9)) / n
    return 100 * (D / n - z * se), 100 * (D / n + z * se), 100 * se


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj


# 1. Wilson 구간
cells = []
for s in T1:
    for b in T1[s]:
        for m in ("Vanilla", "Self", "Binary", "Random"):
            v = T1[s][b][m]
            for tag, i in (("P1", 0), ("P3", 2)):
                k = kk(v[i], N[b]); cells.append({"setup": s, "bench": b, "method": m, "metric": tag, "k": k, "n": N[b], "acc": v[i], "wilson95": wilson(k, N[b])})
out["wilson_cells"] = cells
out["wilson_halfwidth_pp"] = {b: round((wilson(round(0.7 * N[b]), N[b])[1] - wilson(round(0.7 * N[b]), N[b])[0]) / 2, 1) for b in N}

# 2. MDE (짝지은, 80% 검정력, α=.05): 2.80*sqrt(d/n)
out["mde_pp"] = {b: {f"disc{r}": round(100 * 2.80 * math.sqrt(r / N[b]), 1) for r in (0.05, 0.10, 0.20, 0.30)} for b in N}


# 3. 같은 구성 실행 간 불일치 역산. 3회 평균 정확도의 성공 합 S = 3a+2b+c, Pass@3 과제 수 k3 = a+b+c
#    (a: 3/3 성공, b: 2/3, c: 1/3). 두 실행이 다를 확률 d = (2/3)(b+c)/n.
def flaky_range(S, k3, n):
    res = []
    for a in range(0, k3 + 1):
        b = S - 2 * a - k3
        c = k3 - a - b
        if b >= 0 and c >= 0: res.append((a, b, c))
    if not res: return None
    ds = [(2 / 3) * (b + c) / n for a, b, c in res]
    return {"S": S, "k3": k3, "n": n, "a_range": (min(r[0] for r in res), max(r[0] for r in res)), "disc_range": (round(min(ds), 3), round(max(ds), 3))}


out["same_config_discordance"] = {"TB_GPT_vanilla": flaky_range(192, kk(82.02, 89), 89), "SWE_GPT_vanilla": flaky_range(191, kk(63.64, 110), 110)}

# 4. Self 대 Vanilla 8칸 (P1, P3 각 4칸)
rates = (0.05, 0.10, 0.20, 0.30)
rows = []
for s in T1:
    for b in T1[s]:
        n = N[b]
        for tag, i in (("P1", 0), ("P3", 2)):
            ka, kb = kk(T1[s][b]["Self"][i], n), kk(T1[s][b]["Vanilla"][i], n); D = ka - kb
            r = {"setup": s, "bench": b, "metric": tag, "k_self": ka, "k_vanilla": kb, "D_tasks": D, "D_pp": round(100 * D / n, 2),
                 "fisher_p": round(stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1], 3)}
            for rate in rates:
                m0 = max(abs(D), round(rate * n)); p, m = mcnemar_p(D, m0)
                lo, hi, se = paired_ci(D, m, n)
                r[f"disc{rate}"] = {"m": m, "p": round(p, 4), "ci95": (round(lo, 1), round(hi, 1)),
                                    "ni5": bool(100 * D / n - 1.644854 * se > -5), "ni10": bool(100 * D / n - 1.644854 * se > -10)}
            rows.append(r)
out["self_vs_vanilla"] = rows
for tag in ("P1", "P3"):
    sub = [r for r in rows if r["metric"] == tag]
    out[f"summary_{tag}"] = {}
    for rate in rates:
        key = f"disc{rate}"
        ps = np.array([r[key]["p"] for r in sub]); adj = holm(ps)
        out[f"summary_{tag}"][key] = {"raw_p_lt_0.05": [(r["setup"], r["bench"], r["D_pp"]) for r in sub if r[key]["p"] < 0.05],
                                      "holm_lt_0.05": int((adj < 0.05).sum()), "ni5": sum(r[key]["ni5"] for r in sub), "ni10": sum(r[key]["ni10"] for r in sub), "cells": len(sub)}

# 5. Self 대 다른 방법
cmp_ = []
for tag, i in (("P1", 0), ("P3", 2)):
    for s in T1:
        for b in T1[s]:
            n = N[b]
            for m in ("Random", "RSD", "STEER", "MTRouter", "Binary"):
                ka, kb = kk(T1[s][b]["Self"][i], n), kk(T1[s][b][m][i], n); D = ka - kb
                e = {"metric": tag, "setup": s, "bench": b, "vs": m, "D_tasks": D, "D_pp": round(100 * D / n, 1)}
                for rate in (0.10, 0.20):
                    p, _ = mcnemar_p(D, max(abs(D), round(rate * n))); e[f"p{rate}"] = round(p, 4)
                cmp_.append(e)
out["self_vs_baselines"] = cmp_
sumc = {}
for tag in ("P1", "P3"):
    for rate in (0.10, 0.20):
        sub = [e for e in cmp_ if e["metric"] == tag]
        ps = np.array([e[f"p{rate}"] for e in sub]); adj = holm(ps)
        sumc[f"{tag}_disc{rate}"] = {"comparisons": len(sub), "raw_lt_0.05": int((ps < 0.05).sum()), "holm_lt_0.05": int((adj < 0.05).sum()),
                                     "holm_pairs": [(e["setup"], e["bench"], e["vs"], e["D_pp"]) for e, a in zip(sub, adj) if a < 0.05]}
out["self_vs_baselines_summary"] = sumc
rnd = [e for e in cmp_ if e["vs"] == "Random"]

# 6. 소표 절제 (단일 실행, TB n=89)
abl = {}
for name, (ka, kb) in {"RL_vs_SFT (표2)": (64, 51), "token_adv_with_vs_without (표3)": (64, 58), "self_vs_GRPO+SFT (표4)": (64, 29),
                       "judge_27B_vs_9B (표5)": (63, 45), "judge_mini_vs_27B (표5)": (64, 63)}.items():
    D = ka - kb; n = 89
    e = {"D_tasks": D, "D_pp": round(100 * D / n, 1), "fisher_p": round(stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1], 4)}
    for rate in (0.10, 0.20, 0.30):
        p, m = mcnemar_p(D, max(abs(D), round(rate * n))); lo, hi, _ = paired_ci(D, m, n); e[f"disc{rate}"] = {"p": round(p, 4), "ci95": (round(lo, 1), round(hi, 1))}
    abl[name] = e
out["ablations"] = abl
n1, n2 = round(89 * 5.80), round(89 * 8.97)
p1, p2 = 0.4282, 0.2920
se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
z = (p1 - p2) / se
out["table3_write_edit_share"] = {"n_with": n1, "n_without": n2, "diff_pp": round(100 * (p1 - p2), 1), "z_iid": round(z, 2),
                                  "z_cluster_deff1.5": round(z / math.sqrt(1.5), 2), "z_cluster_deff2.5": round(z / math.sqrt(2.5), 2),
                                  "p_iid": float(2 * stats.norm.sf(abs(z))), "p_deff2.5": float(2 * stats.norm.sf(abs(z) / math.sqrt(2.5)))}

# 7. 핵심 단계 분석 (그림 4): 구간에서 SE·p 재구성, Holm (6개 유형)
cr = []
for name, (r, lo, hi) in FIG4_R.items():
    se = (hi - lo) / (2 * Z); zz = r / se
    cr.append({"type": name, "r": r, "se_from_ci": round(se, 4), "z": round(zz, 2), "p": float(2 * stats.norm.sf(abs(zz))), "n_eff_if_se_1_over_sqrt_n": round(1 / se ** 2)})
adj = holm([c["p"] for c in cr])
for c, a in zip(cr, adj): c["holm_p"] = round(float(a), 4); c["p"] = round(c["p"], 5)
out["fig4_r"] = cr
gaps = {}
for k, (g1, g2, d, lo, hi) in FIG4_GAP.items():
    se = (hi - lo) / (2 * Z); gaps[k] = {"delta_pp": d, "se": round(se, 2), "p": round(float(2 * stats.norm.sf(abs(d / se))), 4)}
out["fig4_gap"] = gaps
out["fig4_trajectory_count_check"] = {"tasks": 89 + 110, "budgets": 4, "runs": 3, "product": (89 + 110) * 4 * 3, "paper": "2.4k"}


# 8. 무작위 위임 선택 효과: 3회 평균 정확도, 무작위 배치 K개 중 최댓값이 "전부 클라우드"에 근접할 확률
def sim_select(n, d_run, gap_true_pp, Ks=(1, 3, 5, 10, 20), near_pp=1.0, reps=40000):
    sd = math.sqrt((d_run / 2) / (3 * n))      # 3회 평균의 표준편차(과제 고정, 실행 간 변동만)
    res = {}
    for K in Ks:
        x = rng.normal(-gap_true_pp / 100, sd * math.sqrt(2), size=(reps, K))
        mx = x.max(1) * 100
        res[K] = {"E_max_diff_pp": round(float(mx.mean()), 2), "P(max>=-1pp)": round(float((mx >= -near_pp).mean()), 3), "P(max>=0)": round(float((mx >= 0).mean()), 3)}
    return {"sd_mean3_pp": round(100 * sd, 2), "gap_true_pp": gap_true_pp, "by_K": res}


out["random_alloc_selection"] = {"TB_n89_d0.15_gap3": sim_select(89, 0.15, 3.0), "TB_n89_d0.15_gap5": sim_select(89, 0.15, 5.0), "SWE_n110_d0.09_gap3": sim_select(110, 0.09, 3.0)}

json.dump(out, open("power_check_results.json", "w"), ensure_ascii=False, indent=1, default=str)
for k in ("wilson_halfwidth_pp", "mde_pp", "same_config_discordance", "summary_P1", "summary_P3", "self_vs_baselines_summary", "ablations", "table3_write_edit_share", "fig4_r", "fig4_gap",
          "fig4_trajectory_count_check", "random_alloc_selection"):
    print(k, json.dumps(out[k], ensure_ascii=False, default=str))
print("vs vanilla P3:")
for r in rows:
    if r["metric"] == "P3": print(r["setup"], r["bench"], r["D_tasks"], r["D_pp"], "ci.1", r["disc0.1"]["ci95"], "ci.2", r["disc0.2"]["ci95"], "p.2", r["disc0.2"]["p"], "ni5@.05/.1/.2", r["disc0.05"]["ni5"], r["disc0.1"]["ni5"], r["disc0.2"]["ni5"], "ni10@.2", r["disc0.2"]["ni10"])
print("vs vanilla P1:")
for r in rows:
    if r["metric"] == "P1": print(r["setup"], r["bench"], r["D_tasks"], r["D_pp"], r["disc0.2"]["ci95"], r["disc0.2"]["p"], r["disc0.2"]["ni5"], r["disc0.2"]["ni10"])
print("vs random")
for e in rnd: print(e)
