"""표 재계산: 문항 수 역산, 부제(상대 변화) 대조, 헤드라인 65.9%·+8.8%p·12/22, 표 간 불일치, 파레토 우위.
실행: python3 table_check.py  ->  table_check_results.json"""
import json, itertools
import numpy as np
from paper_tables import *

out = {}

# 1. 문항 수 역산: 보고된 정확도가 k/n(소수 첫째 자리 반올림)으로 설명되는 n 후보
def fits(n, accs):
    return all(any(abs(round(100 * k / n, 1) - a) < 1e-9 or abs(100 * k / n - a) < 0.051 for k in range(n + 1)) for a in accs)
all_acc = {"DS": [], "TB": []}
for m in (BASE, ):
    for v in m.values():
        for b in ("DS", "TB"): all_acc[b].append(v[b][0])
for v in AB.values():
    for b in ("DS", "TB"): all_acc[b].append(v[b][0])
for k, v in T1.items():
    for (b, _, _), (a, c) in zip(T1_COLS, v): all_acc[b].append(a)
for v in T4.values():
    all_acc["DS"] += [v[0]]; all_acc["TB"] += [v[2]]
for r in T6: all_acc["DS"].append(r[2][0]); [all_acc["DS"].append(x[0]) for x in r[3:] if x]
def fit_rate(n, accs):
    ok = [any(abs(100 * k / n - a) < 0.0501 for k in range(n + 1)) for a in accs]
    return sum(ok), len(ok)
out["n_inference"] = {}
for b, rng in (("DS", range(60, 200)), ("TB", range(40, 150))):
    res = {n: fit_rate(n, all_acc[b]) for n in rng}
    best = sorted(res.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:4]
    out["n_inference"][b] = {"values": len(all_acc[b]), "best": [(n, f"{a}/{t}") for n, (a, t) in best],
                              "paper_n_assumed": N[b], f"fit_at_{N[b]}": fit_rate(N[b], all_acc[b])}

# 2. Table 3 부제 대조
sub = []
for key, d in AB.items():
    for b in ("DS", "TB"):
        base = BASE[key[0]][b]; acc, cost = d[b]
        rel_a = 100 * (acc / base[0] - 1); rel_c = 100 * (cost / base[1] - 1)
        sa, sc = AB_SUB[key][b]
        sub.append({"pair": "->".join(key), "bench": b, "rel_acc": round(rel_a, 1), "paper_rel_acc": sa, "rel_cost": round(rel_c, 1), "paper_rel_cost": sc,
                    "acc_mismatch": abs(rel_a - sa) > 0.15, "cost_mismatch": abs(rel_c - sc) > 0.15, "pp": round(acc - base[0], 1)})
out["table3_subscripts"] = sub
out["table3_subscript_mismatch"] = [s for s in sub if s["acc_mismatch"] or s["cost_mismatch"]]

# 3. 헤드라인
red = [-s["rel_cost"] for s in sub]
out["avg_cost_reduction_22"] = round(float(np.mean(red)), 2)
out["avg_cost_reduction_by_bench"] = {b: round(float(np.mean([-s["rel_cost"] for s in sub if s["bench"] == b])), 1) for b in ("DS", "TB")}
out["avg_cost_reduction_by_receiver"] = {r: round(float(np.mean([-s["rel_cost"] for s in sub if s["pair"].endswith(r)])), 1) for r in ("Qwen", "Luna")}
out["avg_cost_reduction_capable_sources_only_20"] = round(float(np.mean([-s["rel_cost"] for s in sub if not s["pair"].startswith("Luna")])), 1)
# 총비용 가중 절감 (합계 비용 기준)
tot_b = tot_a = 0
for key, d in AB.items():
    for b in ("DS", "TB"):
        tot_b += BASE[key[0]][b][1]; tot_a += d[b][1]
out["cost_weighted_reduction"] = round(100 * (1 - tot_a / tot_b), 1)
gains = [(s["pp"], s["pair"], s["bench"]) for s in sub]
out["max_gain_pp"] = max(gains)
n_ok = sum(1 for s in sub if s["rel_acc"] > -5 or s["rel_acc"] >= 0)
out["settings_improve_or_lose_lt5pct_relative"] = {"count": n_ok, "of": len(sub), "paper": "12 of 22",
    "note": "논문 본문은 이를 22개 (모델 쌍, 벤치마크) 중 12개로 적었는데 Luna 소스 포함 여부는 명시 안 함"}
out["settings_improve_or_lose_lt5pct_relative"]["excluding_Luna_source"] = sum(1 for s in sub if not s["pair"].startswith("Luna") and s["rel_acc"] > -5)
out["settings_with_acc_gain_pp_positive"] = sum(1 for s in sub if s["pp"] > 0)
out["settings_with_pp_loss_ge_5"] = sum(1 for s in sub if s["pp"] <= -5)
out["pp_diffs_all"] = sorted(round(s["pp"], 1) for s in sub)

# 4. 표 간 불일치: Table 3 vs Table 6 (DeepSWE, 같은 구성 max 노력)
t6 = {(r[0], r[1]): r for r in T6}
inc = []
for s in ("Sol", "Opus", "GLM", "Kimi", "Luna"):
    eff = "max"
    r = t6[(s, eff)]
    inc.append({"item": f"{s} 단독 max", "table3": BASE[s]["DS"], "table6": r[2], "diff_pp": round(r[2][0] - BASE[s]["DS"][0], 1),
                "diff_tasks": round((r[2][0] - BASE[s]["DS"][0]) * 1.13)})
    for tag, idx, rec in (("Qwen", 3, "Qwen"), ("Luna", 4, "Luna")):
        if r[idx] and (s, rec) in AB:
            inc.append({"item": f"{s}->{rec} (AgentBaton)", "table3": AB[(s, rec)]["DS"], "table6": r[idx],
                        "diff_pp": round(r[idx][0] - AB[(s, rec)]["DS"][0], 1), "diff_tasks": round((r[idx][0] - AB[(s, rec)]["DS"][0]) * 1.13)})
out["table3_vs_table6_deepswe_same_config"] = inc

# 5. 노력 스윕의 비단조성 (같은 모델, 노력만 올림)
nm = []
for s in ("Sol", "Opus", "Kimi"):
    rows = [(r[1], r[2][0]) for r in T6 if r[0] == s]
    nm.append({"model": s, "efforts": rows,
               "decreases": [(rows[i][0], rows[i + 1][0], round(rows[i + 1][1] - rows[i][1], 1)) for i in range(len(rows) - 1) if rows[i + 1][1] < rows[i][1]]})
out["effort_sweep_nonmonotone"] = nm

# 6. 파레토: Table 3의 단일 모델 기준선이 AgentBaton 행을 지배하는가 (acc >=, cost <=), DeepSWE·TB
dom = []
for key, d in AB.items():
    for b in ("DS", "TB"):
        acc, cost = d[b]
        dominators = [m for m, v in BASE.items() if m != key[0] and v[b][0] >= acc and v[b][1] <= cost]
        dom.append({"pair": "->".join(key), "bench": b, "acc": acc, "cost": cost, "dominated_by_single": dominators})
out["ab_rows_dominated_by_single_model_table3"] = [x for x in dom if x["dominated_by_single"]]
out["ab_rows_dominated_count"] = f"{sum(1 for x in dom if x['dominated_by_single'])}/{len(dom)}"
# 같은 표에서 Luna 단독 대비
luna = []
for key, d in AB.items():
    if key[1] != "Luna" or key[0] == "Luna": continue
    for b in ("DS", "TB"):
        la, lc = BASE["Luna"][b]
        luna.append({"pair": "->".join(key), "bench": b, "vs_luna_alone_pp": round(d[b][0] - la, 1), "cost_ratio_vs_luna_alone": round(d[b][1] / lc, 1)})
out["ab_to_luna_vs_luna_alone"] = luna
# capable 단독 vs Luna 단독 격차
out["capable_minus_luna_alone_pp"] = {b: {m: round(BASE[m][b][0] - BASE["Luna"][b][0], 1) for m in ("Sol", "Opus", "GLM", "Kimi", "Gem")} for b in ("DS", "TB")}

# 7. Table 6 파레토 집합: 단일 모델만의 프론티어에 AgentBaton 점이 들어가는가
pts = []
for r in T6:
    pts.append(("single", r[0], r[1], *r[2]))
    if r[3]: pts.append(("AB->Qwen", r[0], r[1], *r[3]))
    if r[4]: pts.append(("AB->Luna", r[0], r[1], *r[4]))
def frontier(P):
    return [p for p in P if not any(q[3] >= p[3] and q[4] <= p[4] and (q[3] > p[3] or q[4] < p[4]) for q in P)]
fs = frontier([p for p in pts if p[0] == "single"]); fa = frontier(pts)
out["pareto_single_only"] = [(p[1], p[2], p[3], p[4]) for p in sorted(fs, key=lambda p: p[4])]
out["pareto_with_agentbaton"] = [(p[0], p[1], p[2], p[3], p[4]) for p in sorted(fa, key=lambda p: p[4])]

# 8. Table 1 방향별 비교: 인계 결과 vs capable 단독(Table 3 기준선)
t1c = []
for (src, rcv), vals in T1.items():
    cap = rcv if src == "Qwen" else src
    cap = {"GLM": "GLM", "Kimi": "Kimi", "Sol": "Sol"}[cap]
    for (b, pct, tr), (a, c) in zip(T1_COLS, vals):
        ba, bc = BASE[cap][b]
        t1c.append({"dir": "E->C" if src == "Qwen" else "C->E", "cap": cap, "bench": b, "pct": pct, "transfer": tr, "acc": a, "cost": c,
                    "d_pp": round(a - ba, 1), "cost_change_pct": round(100 * (c / bc - 1), 1)})
out["table1_vs_capable_only"] = t1c
for d in ("E->C", "C->E"):
    rows = [x for x in t1c if x["dir"] == d]
    out[f"table1_{d}_summary"] = {"cells": len(rows), "acc_ge_capable": sum(1 for x in rows if x["d_pp"] >= 0), "acc_loss_ge_5pp": sum(1 for x in rows if x["d_pp"] <= -5),
                                  "mean_d_pp": round(float(np.mean([x["d_pp"] for x in rows])), 1), "mean_cost_change_pct": round(float(np.mean([x["cost_change_pct"] for x in rows])), 1),
                                  "cost_higher_than_capable_only": sum(1 for x in rows if x["cost_change_pct"] > 0)}
# 전송 방식 비교: log - in
tr = []
for (src, rcv), vals in T1.items():
    for i in range(0, 8, 2):
        b, pct, _ = T1_COLS[i]
        tr.append({"dir": "E->C" if src == "Qwen" else "C->E", "pair": f"{src}->{rcv}", "bench": b, "pct": pct, "log_minus_in_pp": round(vals[i + 1][0] - vals[i][0], 1),
                   "log_cost_ratio": round(vals[i + 1][1] / vals[i][1], 2)})
out["table1_logfile_minus_incontext"] = tr
for d in ("E->C", "C->E"):
    rows = [x for x in tr if x["dir"] == d]
    out[f"transfer_{d}"] = {"cells": len(rows), "log_better": sum(1 for x in rows if x["log_minus_in_pp"] > 0), "log_worse": sum(1 for x in rows if x["log_minus_in_pp"] < 0),
                            "tie": sum(1 for x in rows if x["log_minus_in_pp"] == 0), "mean_pp": round(float(np.mean([x["log_minus_in_pp"] for x in rows])), 1),
                            "max_abs_pp": max(abs(x["log_minus_in_pp"]) for x in rows)}

# 9. Table 4: AgentBaton vs MTRouter 비용 감소율
mt = {}
for cap in ("GLM", "Kimi"):
    m = T4[(cap, "MTRouter")]; a = T4[(cap, "AgentBaton")]
    mt[cap] = {"DS_cost_reduction": round(100 * (1 - a[1] / m[1]), 1), "TB_cost_reduction": round(100 * (1 - a[3] / m[3]), 1),
               "DS_acc_diff_pp": round(a[0] - m[0], 1), "TB_acc_diff_pp": round(a[2] - m[2], 1)}
out["table4_vs_mtrouter"] = mt
fp = {}
for cap in ("GLM", "Kimi"):
    a = T4[(cap, "AgentBaton")]; f = T4[(cap, "Fixed progress")]; t = T4[(cap, "Fixed turn")]
    fp[cap] = {"AB_minus_fixedprogress_pp": (round(a[0] - f[0], 1), round(a[2] - f[2], 1)), "AB_minus_fixedturn_pp": (round(a[0] - t[0], 1), round(a[2] - t[2], 1)),
               "AB_cost_vs_fixedprogress": (a[1], f[1], a[3], f[3]), "AB_cost_vs_fixedturn": (a[1], t[1], a[3], t[3])}
out["table4_vs_fixed"] = fp

json.dump(out, open("table_check_results.json", "w"), ensure_ascii=False, indent=1)
for k in ("n_inference", "table3_subscript_mismatch", "avg_cost_reduction_22", "avg_cost_reduction_by_bench", "avg_cost_reduction_by_receiver", "avg_cost_reduction_capable_sources_only_20",
          "cost_weighted_reduction", "max_gain_pp", "settings_improve_or_lose_lt5pct_relative", "settings_with_acc_gain_pp_positive", "settings_with_pp_loss_ge_5",
          "table3_vs_table6_deepswe_same_config", "effort_sweep_nonmonotone", "ab_rows_dominated_count", "ab_rows_dominated_by_single_model_table3", "ab_to_luna_vs_luna_alone",
          "capable_minus_luna_alone_pp", "pareto_single_only", "pareto_with_agentbaton", "table1_E->C_summary", "table1_C->E_summary", "transfer_E->C", "transfer_C->E", "table4_vs_mtrouter", "table4_vs_fixed"):
    print(k, json.dumps(out[k], ensure_ascii=False))
