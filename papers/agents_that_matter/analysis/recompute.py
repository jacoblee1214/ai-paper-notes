"""AI Agents That Matter (arXiv:2407.01502) 공개 결과 재계산.

입력(실행 없음, 저장된 결과만 읽음)
  - analysis/source_data/plot_data_0409.csv, plot_data_0613.csv : 저자 저장소의 실행별 집계(MIT)
  - analysis/source_data/pertask_humaneval.json : extract_pertask.py 가 저장소 output_data에서 뽑은 과제별 통과 여부
  - analysis/source_data/hotpotqa_compact.json : HotPotQA 과제별 결과(저장소 eval_results.json에서 추출)
출력: analysis/recompute_results.json, 표준출력 요약

사용: python3 -I analysis/recompute.py
"""
import csv, glob, json, math, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "analysis/source_data")
RNG = np.random.default_rng(0)
OUT = {}


# ---------------------------------------------------------------- 공용
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def tci(x, level=0.95):
    x = np.asarray(x, float)
    m, s, n = x.mean(), x.std(ddof=1), len(x)
    h = stats.t.ppf(0.5 + level / 2, n - 1) * s / math.sqrt(n)
    return m - h, m + h


def paired_boot(d, B=20000, rng=RNG):
    """d: 과제별 차이(길이 n). 과제를 복원추출해 평균 차이의 분포를 만든다."""
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
        "ci90": [float(np.percentile(m, 5)), float(np.percentile(m, 95))],
    }


def load_csv(name):
    rows = list(csv.DictReader(open(os.path.join(SRC, name))))
    g = {}
    for r in rows:
        g.setdefault((r["strategy"], r["model"]), []).append(r)
    return g


# ---------------------------------------------------------------- A. 표 A1·A2 재계산
# 논문 Table A1(p.20): 정확도 평균(최소-최대) %, 총비용 평균(최소-최대) USD
PAPER_A1 = {
    "LATS (GPT-4)": ("lats", "gpt-4-turbo-2024-04-09", 88.0, 82.3, 91.5, 134.50, 123.98, 147.13),
    "LATS (GPT-3.5)": ("lats", "gpt-3.5-turbo-0125", 80.4, 78.7, 83.5, 9.49, 8.94, 9.89),
    "LDB (GPT-4, GPT-3.5)": ("ldb", "gpt-3.5-turbo-0125+gpt-4-turbo-2024-04-09", 91.0, 89.0, 92.1, 2.19, 2.14, 2.25),
    "LDB (Reflexion, GPT-4)": ("ldb", "gpt-4-turbo-2024-04-09+reflexion", 92.9, 91.5, 95.1, 7.26, 6.19, 9.63),
    "LDB (Reflexion, GPT-3.5)": ("ldb", "gpt-3.5-turbo-0125+reflexion", 88.9, 86.6, 91.5, 4.19, 3.98, 4.37),
    "LDB (GPT-4)": ("ldb", "gpt-4-turbo-2024-04-09+gpt-4-turbo-2024-04-09", 93.3, 92.1, 94.5, 6.36, 5.11, 7.08),
    "LDB (GPT-3.5)": ("ldb", "gpt-3.5-turbo-0125+gpt-3.5-turbo-0125", 80.2, 78.7, 82.9, 0.63, 0.56, 0.77),
    "GPT-4 (baseline)": ("simple", "gpt-4-turbo-2024-04-09", 89.6, 87.8, 90.9, 1.93, 1.91, 1.95),
    "GPT-3.5 (baseline)": ("simple", "gpt-3.5-turbo-0125", 73.9, 72.0, 76.2, 0.05, 0.05, 0.05),
    "Reflexion (GPT-4)": ("reflexion", "gpt-4-turbo-2024-04-09", 87.8, 86.0, 90.9, 3.90, 3.76, 4.13),
    "Warming (GPT-4)": ("simple_incr_temp", "gpt-4-turbo-0409-5step", 93.2, 92.1, 93.9, 2.45, 2.36, 2.54),
    "Retry (GPT-4)": ("simple_repeat", "gpt-4-turbo-0409-5repeats-no-temp-incr", 92.0, 91.4, 92.7, 2.51, 2.46, 2.56),
    "Escalation": ("simple_boosting", "l3_8b-gpt35-l3_70b-gpt4-0409", 85.0, 84.1, 85.4, 0.27, 0.25, 0.28),
}
g0409 = load_csv("plot_data_0409.csv")
table_a1 = {}
max_acc_gap = max_cost_gap = 0.0
for name, (st, md, pa, pamin, pamax, pc, pcmin, pcmax) in PAPER_A1.items():
    rows = g0409[(st, md)]
    acc = np.array([float(r["accuracy"]) for r in rows]) * 100
    cost = np.array([float(r["total_cost"]) for r in rows])
    table_a1[name] = {
        "n_runs": len(rows),
        "acc_mean": round(acc.mean(), 2), "acc_min": round(acc.min(), 2), "acc_max": round(acc.max(), 2),
        "acc_sd_runs": round(acc.std(ddof=1), 2),
        "acc_t95": [round(x, 2) for x in tci(acc)],
        "cost_mean": round(cost.mean(), 3), "cost_min": round(cost.min(), 3), "cost_max": round(cost.max(), 3),
        "paper": {"acc": [pa, pamin, pamax], "cost": [pc, pcmin, pcmax]},
        "acc_gap": round(acc.mean() - pa, 2), "cost_gap": round(cost.mean() - pc, 3),
    }
    max_acc_gap = max(max_acc_gap, abs(acc.mean() - pa), abs(acc.min() - pamin), abs(acc.max() - pamax))
    max_cost_gap = max(max_cost_gap, abs(cost.mean() - pc))
OUT["table_a1"] = table_a1
OUT["table_a1_max_abs_gap"] = {"acc_pp": round(max_acc_gap, 3), "cost_usd": round(max_cost_gap, 3)}

# 표 A7 재현 대 보고 (논문 p.30)
PAPER_A7 = {"LATS (GPT-4)": 94.4, "LATS (GPT-3.5)": 83.8, "LDB (GPT-4, GPT-3.5)": 89.6,
            "LDB (Reflexion, GPT-3.5)": 95.1, "LDB (GPT-3.5)": 82.9, "GPT-4 (baseline)": 75.0,
            "Reflexion (GPT-4)": 91.0}
a7 = {}
for k, rep in PAPER_A7.items():
    t = table_a1[k]
    pos = "above_max" if rep > t["acc_max"] + 1e-9 else ("below_min" if rep < t["acc_min"] - 1e-9 else "within")
    if abs(rep - t["acc_max"]) < 0.06:
        pos = "at_max"
    a7[k] = {"reported": rep, "run_min": t["acc_min"], "run_max": t["acc_max"], "position": pos}
OUT["table_a7_reported_vs_runs"] = a7

# ---------------------------------------------------------------- B. 비용 산술
PRICE_0409 = {"gpt4": (10, 30), "gpt35": (0.5, 1.5), "l3_70b": (0.9, 0.9), "l3_8b": (0.2, 0.2)}  # USD/1M, 논문 A.1 + Together 가정(8B)
PRICE_0613 = {"gpt4": (30, 60), "gpt35": (0.5, 1.5), "l3_70b": (0.9, 0.9), "l3_8b": (0.2, 0.2)}


def token_cost(r, price):
    c = {}
    for key, (pi, po) in price.items():
        c[key] = (float(r[f"total_{'gpt_4' if key=='gpt4' else 'gpt_35' if key=='gpt35' else key}_prompt_tokens"]) * pi
                  + float(r[f"total_{'gpt_4' if key=='gpt4' else 'gpt_35' if key=='gpt35' else key}_completion_tokens"]) * po) / 1e6
    return c


cost_check = {}
for (st, md), rows in g0409.items():
    res = []
    for r in rows:
        c = token_cost(r, PRICE_0409)
        res.append(float(r["total_cost"]) - sum(c.values()))
    cost_check[f"{st}|{md}"] = {"residual_mean": round(float(np.mean(res)), 4), "residual_sd": round(float(np.std(res, ddof=1)), 5),
                                "total_mean": round(float(np.mean([float(r['total_cost']) for r in rows])), 4)}
OUT["cost_residual_0409"] = cost_check
# LDB·Reflexion 계열 잔차 = 시드(초기 코드)를 만든 에이전트의 비용인지 확인 (실행 번호 대응)
seed_match = {}
for ldb_key, seed_key in [(("ldb", "gpt-4-turbo-2024-04-09+gpt-4-turbo-2024-04-09"), ("simple", "gpt-4-turbo-2024-04-09")),
                          (("ldb", "gpt-3.5-turbo-0125+gpt-4-turbo-2024-04-09"), ("simple", "gpt-4-turbo-2024-04-09")),
                          (("ldb", "gpt-3.5-turbo-0125+gpt-3.5-turbo-0125"), ("simple", "gpt-3.5-turbo-0125")),
                          (("ldb", "gpt-3.5-turbo-0125+reflexion"), ("reflexion", "gpt-4-turbo-2024-04-09")),
                          (("ldb", "gpt-4-turbo-2024-04-09+reflexion"), ("reflexion", "gpt-4-turbo-2024-04-09"))]:
    diffs = []
    for r, s in zip(g0409[ldb_key], g0409[seed_key]):
        res = float(r["total_cost"]) - sum(token_cost(r, PRICE_0409).values())
        diffs.append(res - float(s["total_cost"]))
    seed_match["|".join(ldb_key)] = {"max_abs(residual - seed_run_cost)": round(float(np.max(np.abs(diffs))), 4)}
OUT["ldb_residual_equals_seed_cost"] = seed_match

# 0613: 가격이 논문 본문 A.1에 모호하게 적혀 있어 gpt-3.5-turbo-0613 단가를 자료에서 역산
g0613 = load_csv("plot_data_0613.csv")
r = g0613[("simple", "gpt-3.5-turbo-0613")][0]
pi_, po_ = float(r["total_gpt_35_prompt_tokens"]), float(r["total_gpt_35_completion_tokens"])
OUT["gpt35_0613_price_check"] = {"cost_if_0.5_1.5": round((pi_ * 0.5 + po_ * 1.5) / 1e6, 4), "cost_in_data": round(float(r["total_cost"]), 4),
                                  "cost_if_10_30": round((pi_ * 10 + po_ * 30) / 1e6, 4)}
r = g0613[("simple", "gpt-4-0613")][0]
OUT["gpt4_0613_price_check"] = {"cost_if_30_60": round((float(r["total_gpt_4_prompt_tokens"]) * 30 + float(r["total_gpt_4_completion_tokens"]) * 60) / 1e6, 4),
                                "cost_in_data": round(float(r["total_cost"]), 4)}

# ---------------------------------------------------------------- C. 파레토 전선과 가격 변화
def hull(points):
    """points: [(cost, acc, name)]. 비용을 줄이고 정확도를 높이는 방향의 볼록 전선(상부 볼록 껍질)."""
    pts = sorted(set((c, a, n) for c, a, n in points))
    # 지배되는 점 제거(비용 오름차순에서 정확도가 최대 갱신일 때만 유지)
    nd, best = [], -1
    for c, a, n in pts:
        if a > best + 1e-12:
            nd.append((c, a, n)); best = a
    h = []
    for p in nd:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= 0:
            h.pop()
        h.append(p)
    return h


def nondom(points):
    out = []
    for c, a, n in points:
        if not any((c2 <= c and a2 >= a and (c2 < c or a2 > a)) for c2, a2, _ in points):
            out.append(n)
    return out


def comp_costs(rows, price, residual_bucket):
    """실행 평균 비용을 모델별 구성요소로 쪼갠다: {'gpt4':..,'gpt35':..,'other':..}"""
    out = {"gpt4": 0.0, "gpt35": 0.0, "other": 0.0}
    for r in rows:
        c = token_cost(r, price)
        res = float(r["total_cost"]) - sum(c.values())
        out["gpt4"] += c["gpt4"]; out["gpt35"] += c["gpt35"]; out["other"] += c["l3_70b"] + c["l3_8b"]
        out[residual_bucket] += res
    return {k: v / len(rows) for k, v in out.items()}


MAIN = {  # 이름: (strategy, model, 잔차 귀속)
    "GPT-3.5": ("simple", "gpt-3.5-turbo-0125", "gpt4"), "GPT-4": ("simple", "gpt-4-turbo-2024-04-09", "gpt4"),
    "LDB (GPT-3.5)": ("ldb", "gpt-3.5-turbo-0125+gpt-3.5-turbo-0125", "gpt35"),
    "LDB (GPT-4, GPT-3.5)": ("ldb", "gpt-3.5-turbo-0125+gpt-4-turbo-2024-04-09", "gpt4"),
    "LDB (Reflexion, GPT-3.5)": ("ldb", "gpt-3.5-turbo-0125+reflexion", "gpt4"),
    "LDB (GPT-4)": ("ldb", "gpt-4-turbo-2024-04-09+gpt-4-turbo-2024-04-09", "gpt4"),
    "LDB (Reflexion, GPT-4)": ("ldb", "gpt-4-turbo-2024-04-09+reflexion", "gpt4"),
    "Reflexion (GPT-4)": ("reflexion", "gpt-4-turbo-2024-04-09", "gpt4"),
    "LATS (GPT-3.5)": ("lats", "gpt-3.5-turbo-0125", "gpt4"), "LATS (GPT-4)": ("lats", "gpt-4-turbo-2024-04-09", "gpt4"),
    "Retry (GPT-4)": ("simple_repeat", "gpt-4-turbo-0409-5repeats-no-temp-incr", "gpt4"),
    "Warming (GPT-4)": ("simple_incr_temp", "gpt-4-turbo-0409-5step", "gpt4"),
    "Escalation": ("simple_boosting", "l3_8b-gpt35-l3_70b-gpt4-0409", "gpt4"),
}
acc_mean = {k: float(np.mean([float(r["accuracy"]) for r in g0409[(a, b)]])) * 100 for k, (a, b, _) in MAIN.items()}
comp = {k: comp_costs(g0409[(a, b)], PRICE_0409, bucket) for k, (a, b, bucket) in MAIN.items()}
base_pts = [(sum(comp[k].values()), acc_mean[k], k) for k in MAIN]
OUT["pareto_0409"] = {
    "nondominated": nondom(base_pts),
    "convex_frontier": [n for _, _, n in hull(base_pts)],
    "points": {n: {"cost": round(c, 3), "acc": round(a, 2)} for c, a, n in base_pts},
}
# GPT-4 단가를 s배 (입력·출력 모두)
sweep = []
prev = None
for s in [round(x, 3) for x in np.concatenate([np.linspace(0.02, 0.2, 19), np.linspace(0.2, 3, 57)])]:
    pts = [(comp[k]["gpt4"] * s + comp[k]["gpt35"] + comp[k]["other"], acc_mean[k], k) for k in MAIN]
    fr = [n for _, _, n in hull(pts)]
    if fr != prev:
        sweep.append({"gpt4_price_scale": s, "convex_frontier": fr})
        prev = fr
OUT["pareto_gpt4_price_sweep"] = sweep
# 0613 자료 직접 비교
acc0613 = {}
MAIN0613 = {
    "GPT-3.5": ("simple", "gpt-3.5-turbo-0613"), "GPT-4": ("simple", "gpt-4-0613"),
    "LDB (GPT-3.5)": ("ldb", "gpt-3.5-turbo-0613+gpt-3.5-turbo-0613"), "LDB (GPT-4, GPT-3.5)": ("ldb", "gpt-3.5-turbo-0613+gpt-4-0613"),
    "LDB (Reflexion, GPT-3.5)": ("ldb", "gpt-3.5-turbo-0613+reflexion"), "LDB (GPT-4)": ("ldb", "gpt-4-0613+gpt-4-0613"),
    "LDB (Reflexion, GPT-4)": ("ldb", "gpt-4-0613+reflexion"), "Reflexion (GPT-4)": ("reflexion", "gpt-4-0613"),
    "LATS (GPT-3.5)": ("lats", "gpt-3.5-turbo-0613"), "LATS (GPT-4)": ("lats", "gpt-4-0613"),
    "Retry (GPT-4)": ("simple_repeat", "gpt-4-0613-5repeats-no-temp-incr"), "Warming (GPT-4)": ("simple_incr_temp", "gpt-4-0613-5step"),
    "Escalation": ("simple_boosting", "l3_8b-gpt-3.5-turbo-0613-l3_70b-gpt4-0613"),
}
pts13 = []
for k, key in MAIN0613.items():
    rows = g0613[key]
    a = float(np.mean([float(r["accuracy"]) for r in rows])) * 100
    c = float(np.mean([float(r["total_cost"]) for r in rows]))
    pts13.append((c, a, k))
OUT["pareto_0613"] = {"nondominated": nondom(pts13), "convex_frontier": [n for _, _, n in hull(pts13)],
                      "points": {n: {"cost": round(c, 3), "acc": round(a, 2)} for c, a, n in pts13}}

# ---------------------------------------------------------------- D. 정확도 차이의 불확실성 (과제별 쌍비교)
PT = json.load(open(os.path.join(SRC, "pertask_humaneval.json")))
ids0 = None
M = {}
for name, v in PT.items():
    rr = []
    for rk in sorted(v["runs"]):
        ids = v["runs"][rk]["ids"]
        if ids0 is None:
            ids0 = ids
        order = {t: i for i, t in enumerate(ids)}
        rr.append([v["runs"][rk]["solved"][order[t]] for t in ids0])
    M[name] = np.array(rr, float)  # (5, 164)
N = len(ids0)
OUT["n_tasks"] = N

single = {}
for name, m in M.items():
    k_runs = m.sum(axis=1)
    lo, hi = zip(*[wilson(int(k), N) for k in k_runs])
    p = m.mean()
    single[name] = {
        "runs_correct": [int(k) for k in k_runs], "mean_acc": round(float(p) * 100, 2),
        "run_sd_pp": round(float(np.std(m.mean(axis=1), ddof=1)) * 100, 2),
        "binomial_sd_pp(n=164)": round(math.sqrt(p * (1 - p) / N) * 100, 2),
        "wilson95_single_run_mean": [round(float(np.mean(lo)) * 100, 1), round(float(np.mean(hi)) * 100, 1)],
    }
OUT["single_run_uncertainty"] = single

PAIRS = [
    ("Warming (GPT-4)", "LDB (GPT-4)"), ("Warming (GPT-4)", "LDB (Reflexion, GPT-4)"),
    ("Warming (GPT-4)", "Reflexion (GPT-4)"), ("Warming (GPT-4)", "LDB (GPT-4, GPT-3.5)"),
    ("Warming (GPT-4)", "Retry (GPT-4)"), ("Warming (GPT-4)", "GPT-4 zero-shot"),
    ("Retry (GPT-4)", "GPT-4 zero-shot"), ("Retry (GPT-4)", "LDB (GPT-4)"),
    ("Escalation", "LDB (GPT-3.5)"), ("Escalation", "GPT-3.5 zero-shot"), ("LDB (GPT-3.5)", "GPT-3.5 zero-shot"),
    ("LDB (GPT-4, GPT-3.5)", "GPT-4 zero-shot"), ("LDB (GPT-4)", "GPT-4 zero-shot"),
    ("Reflexion (GPT-4)", "GPT-4 zero-shot"), ("LDB (Reflexion, GPT-4)", "GPT-4 zero-shot"),
]
pairs = []
for a, b in PAIRS:
    ra, rb = M[a].mean(axis=1) * 100, M[b].mean(axis=1) * 100  # 실행별 정확도
    d = M[a].mean(axis=0) - M[b].mean(axis=0)  # 과제별 통과율 차이(5회 평균)
    bt = paired_boot(d * 100)
    tt = stats.ttest_ind(ra, rb, equal_var=False)
    sd_d = float(np.std(d * 100, ddof=1))
    se = sd_d / math.sqrt(N)
    pairs.append({
        "A": a, "B": b, "diff_pp": round(bt["mean"], 2),
        "boot_ci95_pp": [round(x, 2) for x in bt["ci95"]], "boot_ci90_pp": [round(x, 2) for x in bt["ci90"]],
        "min_equiv_margin_pp(TOST90)": round(max(abs(bt["ci90"][0]), abs(bt["ci90"][1])), 2),
        "welch_p_runs(n=5 vs 5)": round(float(tt.pvalue), 4),
        "sd_of_task_diff_pp": round(sd_d, 1), "se_pp": round(se, 2), "mde80_pp": round(2.8 * se, 2),
        "tasks_a_better/worse/tie(rate)": [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())],
        "wilcoxon_p": round(float(stats.wilcoxon(d[d != 0]).pvalue), 4) if (d != 0).sum() > 0 else None,
    })
OUT["humaneval_pairs"] = pairs

# LATS는 jsonl 정확도가 중단·재시작 때문에 부정확(저장소 README) → 실행별 CSV 값으로 실행 수준 비교만
lats = np.array([float(r["accuracy"]) for r in g0409[("lats", "gpt-4-turbo-2024-04-09")]]) * 100
warm = M["Warming (GPT-4)"].mean(axis=1) * 100
OUT["lats_vs_warming_runs"] = {"lats_runs": [round(x, 2) for x in lats], "warming_runs": [round(x, 2) for x in warm],
                                "welch_p": round(float(stats.ttest_ind(lats, warm, equal_var=False).pvalue), 4),
                                "diff_pp": round(float(warm.mean() - lats.mean()), 2), "cost_ratio": round(float(
                                    np.mean([float(r["total_cost"]) for r in g0409[("lats", "gpt-4-turbo-2024-04-09")]]) /
                                    np.mean([float(r["total_cost"]) for r in g0409[("simple_incr_temp", "gpt-4-turbo-0409-5step")]])), 1)}

# 비용 배수 (본문 2.3 주장)
w = comp["Warming (GPT-4)"]; wc = sum(w.values())
OUT["cost_ratio_vs_warming"] = {k: round(sum(v.values()) / wc, 2) for k, v in comp.items()}

# 포화: 과제 난이도 분포
allm = np.stack([M[k].mean(axis=0) for k in M])  # (agents, tasks)
never = int((allm.max(axis=0) == 0).sum())
always = int((allm.min(axis=0) == 1).sum())
top = ["Warming (GPT-4)", "Retry (GPT-4)", "LDB (GPT-4)", "LDB (Reflexion, GPT-4)", "LDB (GPT-4, GPT-3.5)", "Reflexion (GPT-4)", "GPT-4 zero-shot"]
topm = np.stack([M[k].mean(axis=0) for k in top])
OUT["task_difficulty"] = {
    "tasks_never_solved_by_any_agent_run": never, "tasks_always_solved_by_all_agents": always,
    "top_gpt4_agents_tasks_never_solved": int((topm.max(axis=0) == 0).sum()),
    "top_gpt4_agents_tasks_always_solved": int((topm.min(axis=0) == 1).sum()),
    "top_gpt4_agents_discriminating_tasks": int(N - (topm.max(axis=0) == 0).sum() - (topm.min(axis=0) == 1).sum()),
}

# 디버깅의 순수 이득: LDB(GPT-4,GPT-3.5)의 시드는 GPT-4 zero-shot → 단순 재시도·온도 올리기와 비교
OUT["seed_vs_debug_note"] = "LDB (GPT-4, GPT-3.5) 시드 = GPT-4 zero-shot 프로그램, 디버거 = GPT-3.5 (저장소 파일명·토큰 열로 확인)"

# 예시 테스트가 없는 3과제(HumanEval/32, 38, 50)가 Reflexion 결과에 주는 영향
sel = [i for i, t in enumerate(ids0) if t in ("HumanEval/32", "HumanEval/38", "HumanEval/50")]
rest = [i for i in range(N) if i not in sel]
OUT["three_tasks_without_example_tests"] = {
    n: {"acc_3tasks_per_task": [round(float(x), 2) for x in M[n][:, sel].mean(axis=0)], "acc_161tasks": round(float(M[n][:, rest].mean() * 100), 2),
        "acc_164": round(float(M[n].mean() * 100), 2)}
    for n in ["GPT-4 zero-shot", "Reflexion (GPT-4)", "Warming (GPT-4)", "LDB (GPT-4)"]}

# ---------------------------------------------------------------- E. HotPotQA (표 A3, 그림 A6)
hp = {}
HPC = json.load(open(os.path.join(SRC, "hotpotqa_compact.json")))
# 논문 Table A3 값 (정확도 = 정답 문서 검색 성공률, 변동비 = 100건당 USD, 고정비 USD)
PA3 = {"GPT-3.5": {"random_search": (0.495, 0.376, 2.696), "few_shot": (0.47, 0.384, 0.029), "joint": (0.509, 0.174, 2.714),
                   "format": (0.377, 0.071, None), "uncompiled": (0.436, 0.095, None)},
       "Llama-3-70B": {"random_search": (0.617, 0.643, 4.820), "few_shot": (0.622, 0.661, 0.028), "joint": (0.601, 0.374, 3.844),
                       "format": (0.527, 0.111, None), "uncompiled": (0.448, 0.194, None)}}
for model, designs in HPC.items():
    res, pertask = {}, {}
    for dname, E in designs.items():
        if dname == "joint":
            dev = [e["dev_acc"] for e in E]
            chosen = int(np.argmax(dev))
            cand = []
            for i, e in enumerate(E):
                g = [np.mean(r["gold"]) for r in e["runs"].values()]
                c = [r["cost_200"] / 2 for r in e["runs"].values()]
                cand.append({"i": i, "dev_acc": e["dev_acc"], "test_gold_mean": round(float(np.mean(g)), 4), "cost_per100": round(float(np.mean(c)), 4)})
            res["joint_candidates"] = cand
            res["joint_chosen_by_dev"] = chosen
            use = [E[chosen]]
        else:
            use = E  # uncompiled·format은 항목 5개가 각각 1회 실행
        gold = [np.mean(r["gold"]) for e in use for r in e["runs"].values()]
        costs = [r["cost_200"] / 2 for e in use for r in e["runs"].values()]
        ems = [r["em"] for e in use for r in e["runs"].values()]
        pt = [np.array(r["gold"], float) for e in use for r in e["runs"].values()]
        res[dname] = {"n_runs": len(gold), "acc_mean": round(float(np.mean(gold)), 4), "acc_min": float(min(gold)), "acc_max": float(max(gold)),
                      "answer_exact_match_mean": round(float(np.mean(ems)), 4),
                      "var_cost_per100": round(float(np.mean(costs)), 4), "fixed_cost": round(float(use[0]["compile_cost"]), 3),
                      "paper": PA3[model][dname]}
        pertask[dname] = np.mean(pt, axis=0)
    j, fs, rs = res["joint"], res["few_shot"], res["random_search"]
    res["variable_cost_saving_vs_random_search"] = round(1 - j["var_cost_per100"] / rs["var_cost_per100"], 3)
    res["variable_cost_saving_vs_few_shot"] = round(1 - j["var_cost_per100"] / fs["var_cost_per100"], 3)
    res["breakeven_tasks_vs_few_shot"] = round((j["fixed_cost"] - fs["fixed_cost"]) / ((fs["var_cost_per100"] - j["var_cost_per100"]) / 100), 0)
    res["breakeven_tasks_vs_random_search"] = round((j["fixed_cost"] - rs["fixed_cost"]) / ((rs["var_cost_per100"] - j["var_cost_per100"]) / 100), 0)
    tests = {}
    for other in ["few_shot", "random_search", "uncompiled", "format"]:
        d = (pertask["joint"] - pertask[other]) * 100
        bt = paired_boot(d)
        tests[f"joint - {other}"] = {"diff_pp": round(bt["mean"], 2), "ci95": [round(x, 2) for x in bt["ci95"]],
                                       "min_equiv_margin_pp(TOST90)": round(max(abs(bt["ci90"][0]), abs(bt["ci90"][1])), 2)}
    res["paired_tests_200_questions"] = tests
    k = round(j["acc_mean"] * 200)
    res["wilson95_single_run_joint"] = [round(x, 3) for x in wilson(k, 200)]
    hp[model] = res
OUT["hotpotqa"] = hp

# ---------------------------------------------------------------- F. NovelQA (표 A5)
nq = {}
ratio_check = 1.115 / 0.051
nq["table_a5"] = {
    "long_context_cost_total": 98.09 + 1.709, "paper_total": 99.80,
    "tokens_total_check": round(88 * 690.807 + 110094, 1), "paper_tokens_total": 170885.016,
    "question_cost_check_usd(10/1M)": round(170885.016 * 10 / 1e6, 3), "per_novel_long_context": round(98.09 / 88, 4),
    "rag_cost_per_qa_new_novel": round(0.0285 + 0.0222, 4), "paper_rag_cost_per_qa_new_novel": 0.051,
    "cost_ratio": round(ratio_check, 2), "paper_ratio": 21.86,
    "novelqa_setting_ratio": round(99.8 / 52.8, 2),
    "overestimate_factor": round(21.86 / (99.8 / 52.8), 1),
    "rag_total_check": round(88 * 0.0285 + 0.0222 * 2305, 2), "paper_rag_total": 52.80,
}
# 정답 수 환산: 어떤 N에서 67.89와 67.81이 모두 반올림으로 성립하나
cands = []
for n in range(2200, 2450):
    for k1 in range(int(0.6785 * n), int(0.6795 * n) + 2):
        for k2 in range(int(0.6775 * n), int(0.6785 * n) + 2):
            if round(100 * k1 / n, 2) == 67.89 and round(100 * k2 / n, 2) == 67.81:
                cands.append((n, k1, k2))
nq["consistent_N_k1_k2(first5)"] = cands[:5]
n_ = 2305
se_unpaired = math.sqrt(2 * 0.6785 * 0.3215 / n_) * 100
nq["acc_diff_pp"] = round(67.89 - 67.81, 2)
nq["unpaired_se_pp(N~2305)"] = round(se_unpaired, 2)
nq["unpaired_95ci_halfwidth_pp"] = round(1.96 * se_unpaired, 2)
nq["note_single_run_each"] = "표 A5: 각 1회 실행, 문항별 짝 자료 없음"
OUT["novelqa"] = nq

# ---------------------------------------------------------------- G. 홀드아웃 표 (부록 표 A4, 본문 표 1)
# (수준, 홀드아웃 유무와 수준 적합 여부) — 표 A4 전사. 적합 = 열 "◦"가 붙고 보유 홀드아웃이 해당 수준에서 요구하는 종류
A4 = [
    ("MLAgentBench", "task", "none", False), ("SWE-bench", "task", "in-dist", False), ("WebArena", "domain", "none", False),
    ("WebShop", "task", "ood", True), ("Mind2Web", "domain", "tasks", True), ("PPNL", "task", "ood", True),
    ("TravelPlanner", "task", "in-dist", False), ("agbenchmark", "full", "none", False), ("LLF-Bench", "domain", "none", False),
    ("TaskBench", "domain", "none", False), ("AlfWorld", "domain", "ood", False), ("ScienceWorld", "task", "ood", True),
    ("BabyAI", "domain", "in-dist", False), ("Jericho", "dist", "in-dist", True), ("GAIA", "full", "in-dist", False),
    ("MINT", "domain", "none_plan", False), ("tau-bench", "domain", "none_plan", False),
]
lvl = {}
for n, l, h, ok in A4:
    d = lvl.setdefault(l, [0, 0]); d[1] += 1; d[0] += int(ok)
OUT["holdout_table"] = {
    "by_level_appropriate/total": lvl, "total_appropriate": sum(ok for *_, ok in A4), "total": len(A4),
    "no_holdout(any)": sum(h.startswith("none") for _, _, h, _ in A4),
    "no_holdout_and_no_plan": sum(h == "none" for _, _, h, _ in A4),
    "no_holdout_but_plan": sum(h == "none_plan" for _, _, h, _ in A4),
    "paper_claims": {"table1": {"dist": "1/1", "task": "3/6", "domain": "1/8", "full": "0/2"}, "no_holdout_and_no_plan": 7, "has_holdout": 10, "appropriate_of_those": 5},
}

# ---------------------------------------------------------------- H. WebArena 상한
OUT["webarena_bounds"] = {
    "reddit_tasks": 129, "total_tasks": 812, "step_rate_limit_affected": 30, "step_removed_tasks": 8,
    "max_pp_if_all_affected_flipped": round(30 / 812 * 100, 2), "step_leaderboard_pct": 35.8,
    "step_repro_in_repo_notebook_pct": round(201 / 812 * 100, 2),
    "note": "저장소 webarena_step_eval.ipynb 출력: STeP 전체 24.75%(812과제 중 201), Reddit 21.7% (논문 본문은 이 수치를 쓰지 않음)",
}

json.dump(OUT, open(os.path.join(ROOT, "analysis/recompute_results.json"), "w"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 요약 출력
print("표 A1 최대 차이:", OUT["table_a1_max_abs_gap"])
print("A7:", {k: v["position"] for k, v in a7.items()})
print("비용 잔차=시드 비용 최대오차:", seed_match)
print("0613 단가 확인:", OUT["gpt35_0613_price_check"], OUT["gpt4_0613_price_check"])
print("파레토 0409 볼록 전선:", OUT["pareto_0409"]["convex_frontier"], "비지배:", OUT["pareto_0409"]["nondominated"])
print("파레토 0613 볼록 전선:", OUT["pareto_0613"]["convex_frontier"], "비지배:", OUT["pareto_0613"]["nondominated"])
for s in sweep: print("  GPT-4 가격 x%.3f ->" % s["gpt4_price_scale"], s["convex_frontier"])
print("비용 배수(대 Warming):", OUT["cost_ratio_vs_warming"])
for p in pairs:
    print(f"{p['A']} - {p['B']}: {p['diff_pp']:+.2f}pp CI95 {p['boot_ci95_pp']} equivδ {p['min_equiv_margin_pp(TOST90)']} welch_p {p['welch_p_runs(n=5 vs 5)']} MDE80 {p['mde80_pp']}")
print("LATS vs warming:", OUT["lats_vs_warming_runs"])
print("난이도:", OUT["task_difficulty"])
print("단일실행 변동:", {k: (v['run_sd_pp'], v['binomial_sd_pp(n=164)']) for k, v in single.items()})
print("홀드아웃:", OUT["holdout_table"])
print("NovelQA:", nq)
for m, r in hp.items():
    print(m, {k: (v if not isinstance(v, dict) else None) for k, v in r.items() if not isinstance(v, (dict, list))})
    for d in ["uncompiled", "format", "few_shot", "random_search", "joint"]:
        print("   ", d, r[d])
    print("   candidates", r["joint_candidates"])
    print("   tests", r["paired_tests_200_questions"], r["wilson95_single_run_joint"])
