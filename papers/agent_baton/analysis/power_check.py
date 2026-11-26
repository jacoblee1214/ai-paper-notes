"""표본 수 기반 통계: Wilson 구간, 짝지은 McNemar 시나리오, MDE, 선택 효과 시뮬레이션, 동등성(TOST), Holm.
과제별 결과가 공개되지 않아(공식 코드·데이터 없음) 짝지은 불일치율을 시나리오로 둔다.
실행: python3 power_check.py  ->  power_check_results.json"""
import json, math
import numpy as np
from scipy import stats
from paper_tables import *

rng = np.random.default_rng(0)
out = {}

def kk(acc, n): return round(acc * n / 100)
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))
def mcnemar_p(D, m):
    """순차이 D(과제 수, 인계 - 기준선), 불일치 쌍 수 m. b = 인계만 성공."""
    if (m - D) % 2: m += 1
    b = (m + D) // 2
    return stats.binomtest(b, m, 0.5).pvalue, m
def paired_ci(D, m, n, z=1.959964):
    se = math.sqrt(max(m - D * D / n, 1e-9)) / n
    return 100 * (D / n - z * se), 100 * (D / n + z * se), 100 * se
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj

# 1. 기준선·AgentBaton 셀별 Wilson 구간
cells = []
for m, d in BASE.items():
    for b in ("DS", "TB"):
        k = kk(d[b][0], N[b]); cells.append({"name": f"{m} 단독", "bench": b, "k": k, "n": N[b], "acc": d[b][0], "wilson95": wilson(k, N[b])})
for key, d in AB.items():
    for b in ("DS", "TB"):
        k = kk(d[b][0], N[b]); cells.append({"name": "->".join(key), "bench": b, "k": k, "n": N[b], "acc": d[b][0], "wilson95": wilson(k, N[b])})
out["wilson_cells"] = cells
out["wilson_halfwidth_pp"] = {b: round((wilson(N[b] // 2, N[b])[1] - wilson(N[b] // 2, N[b])[0]) / 2, 1) for b in ("DS", "TB")}

# 2. MDE (짝지은, 80% 검정력, α=.05): 2.80*sqrt(d/n)
out["mde_pp"] = {b: {f"disc{r}": round(100 * 2.80 * math.sqrt(r / N[b]), 1) for r in (0.05, 0.10, 0.20, 0.30)} for b in ("DS", "TB")}
# 비대응(독립 실행) MDE, p0=0.65
out["mde_unpaired_pp"] = {b: round(100 * 2.80 * math.sqrt(2 * 0.65 * 0.35 / N[b]), 1) for b in ("DS", "TB")}

# 3. 22개 설정: 짝지은 시나리오 p, Holm, 동등성(비열등 하한)
rows = []
for key, d in AB.items():
    for b in ("DS", "TB"):
        n = N[b]; ka = kk(d[b][0], n); kb = kk(BASE[key[0]][b][0], n); D = ka - kb
        r = {"pair": "->".join(key), "bench": b, "k_handoff": ka, "k_capable": kb, "D_tasks": D, "D_pp": round(100 * D / n, 1)}
        # 비대응 Fisher
        r["fisher_p"] = round(stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1], 3)
        for rate in (0.10, 0.20, 0.30):
            m0 = max(abs(D), round(rate * n)); p, m = mcnemar_p(D, m0)
            lo, hi, se = paired_ci(D, m, n)
            r[f"disc{rate}"] = {"m": m, "p": round(p, 4), "ci95": (round(lo, 1), round(hi, 1)),
                                "noninferior_5pp": bool(100 * D / n - 1.644854 * se > -5), "noninferior_10pp": bool(100 * D / n - 1.644854 * se > -10)}
        rows.append(r)
out["rows22"] = rows
for rate in (0.10, 0.20, 0.30):
    key = f"disc{rate}"
    ps = np.array([r[key]["p"] for r in rows]); adj = holm(ps)
    sig_raw = [(r["pair"], r["bench"], r["D_pp"]) for r in rows if r[key]["p"] < 0.05]
    sig_holm = [(r["pair"], r["bench"], r["D_pp"]) for r, a in zip(rows, adj) if a < 0.05]
    out[f"summary_{key}"] = {"raw_p_lt_0.05": len(sig_raw), "holm_lt_0.05": len(sig_holm), "holm_list": sig_holm, "raw_list": sig_raw,
                              "gains_raw_sig": [(r["pair"], r["bench"], r["D_pp"], r[key]["p"]) for r in rows if r["D_pp"] > 0 and r[key]["p"] < 0.05],
                              "noninferior_5pp": sum(r[key]["noninferior_5pp"] for r in rows), "noninferior_10pp": sum(r[key]["noninferior_10pp"] for r in rows)}
out["fisher_unpaired_sig_raw"] = [(r["pair"], r["bench"], r["D_pp"], r["fisher_p"]) for r in rows if r["fisher_p"] < 0.05]
out["headline_8.8pp"] = next(r for r in rows if r["pair"] == "Opus->Luna" and r["bench"] == "DS")

# 4. 선택 효과: 귀무(참 차이 0)에서 최댓값. 짝지은 SD = sqrt(rate/n). 같은 capable 기준선을 쓰는 2개 비교는 상관 0.5.
def sim_max(rate, n, ncap=5, nrecv=2, reps=40000, benches=("DS",)):
    sd = math.sqrt(rate / n)
    base = rng.normal(size=(reps, ncap, 1)) * sd * math.sqrt(0.5)
    run = rng.normal(size=(reps, ncap, nrecv)) * sd * math.sqrt(0.5)
    diff = (run - base).reshape(reps, -1)
    return diff.max(1) * 100
sel = {}
for rate in (0.10, 0.20, 0.30):
    mx = sim_max(rate, N["DS"])
    sel[f"disc{rate}"] = {"DS_E_max_pp": round(float(mx.mean()), 1), "DS_P(max>=8.8)": round(float((mx >= 8.8).mean()), 3),
                         "DS_95pct_of_max": round(float(np.percentile(mx, 95)), 1)}
    # 두 벤치마크 20개 중 최댓값
    mx2 = np.maximum(sim_max(rate, N["DS"]), sim_max(rate, N["TB"]))
    sel[f"disc{rate}"]["both_E_max_pp"] = round(float(mx2.mean()), 1)
    sel[f"disc{rate}"]["both_P(max>=8.8)"] = round(float((mx2 >= 8.8).mean()), 3)
out["selection_effect_null"] = sel

# 5. 같은 구성의 두 실행 간 불일치 (Table 3 vs Table 6, DeepSWE): 과제 수 차이
out["rerun_gap_tasks"] = {"GLM_single_max": (kk(61.1, 113), kk(69.0, 113)), "Sol_single_max": (kk(68.1, 113), kk(71.7, 113)),
                          "GLM->Luna": (kk(66.4, 113), kk(72.6, 113))}
# 한 구성이 8.8%p(10과제) 차이날 최소 불일치율
out["tasks_for_8.8pp"] = round(0.088 * 113, 2)
# 10과제 순차이가 McNemar p<.05가 되려면 불일치 쌍 수 상한
mm = None
for m in range(10, 200, 2):
    if mcnemar_p(10, m)[0] < 0.05: mm = m
out["max_discordant_for_sig_D10_n113"] = mm

# 6. 표 1: 전송 방식 차이 (같은 칸, 비대응 Fisher + 불일치율 0.2 시나리오)
t1 = []
for (src, rcv), vals in T1.items():
    for i in range(0, 8, 2):
        b, pct, _ = T1_COLS[i]; n = N[b]
        ka, kb = kk(vals[i + 1][0], n), kk(vals[i][0], n); D = ka - kb
        p_f = stats.fisher_exact([[ka, n - ka], [kb, n - kb]])[1]
        p_m, _ = mcnemar_p(D, max(abs(D), round(0.2 * n)))
        t1.append({"dir": "E->C" if src == "Qwen" else "C->E", "pair": f"{src}->{rcv}", "bench": b, "pct": pct, "log_minus_in_tasks": D, "fisher_p": round(p_f, 3), "mcnemar_disc0.2_p": round(p_m, 3)})
out["table1_transfer_tests"] = t1
out["table1_transfer_sig"] = {d: sum(1 for x in t1 if x["dir"] == d and x["mcnemar_disc0.2_p"] < 0.05) for d in ("E->C", "C->E")}
out["table1_transfer_sig_fisher"] = {d: sum(1 for x in t1 if x["dir"] == d and x["fisher_p"] < 0.05) for d in ("E->C", "C->E")}

# 7. 파레토 확장 주장의 쌍: Fig.1·Table 6 (DeepSWE n=113)
pairs = [("Kimi->Luna high 79.6/1.60", 79.6, "Opus xhigh 75.2/8.90", 75.2), ("GLM->Luna max 72.6/0.80", 72.6, "Sol xhigh 71.7/2.79", 71.7),
         ("GLM->Luna max 72.6/0.80", 72.6, "GLM max 69.0/3.06", 69.0), ("GLM->Luna max(Table3) 66.4/0.83", 66.4, "GLM max(Table3) 61.1/3.16", 61.1),
         ("Luna->Qwen xhigh 56.6/0.06", 56.6, "Luna xhigh 56.6/0.21", 56.6)]
pf = []
for na, a, nb, b in pairs:
    ka, kb = kk(a, 113), kk(b, 113); D = ka - kb
    pf.append({"A": na, "B": nb, "D_tasks": D, "fisher_p": round(stats.fisher_exact([[ka, 113 - ka], [kb, 113 - kb]])[1], 3),
               "mcnemar_disc0.2_p": round(mcnemar_p(D, max(abs(D), 23))[0], 3),
               "ci95_paired_disc0.2": tuple(round(x, 1) for x in paired_ci(D, max(abs(D), 23) + ((max(abs(D), 23) - D) % 2), 113)[:2])})
out["pareto_pairs"] = pf

json.dump(out, open("power_check_results.json", "w"), ensure_ascii=False, indent=1)
for k in ("wilson_halfwidth_pp", "mde_pp", "mde_unpaired_pp", "summary_disc0.1", "summary_disc0.2", "summary_disc0.3", "fisher_unpaired_sig_raw", "headline_8.8pp", "selection_effect_null",
          "rerun_gap_tasks", "tasks_for_8.8pp", "max_discordant_for_sig_D10_n113", "table1_transfer_sig", "table1_transfer_sig_fisher", "pareto_pairs"):
    print(k, json.dumps(out[k], ensure_ascii=False))
