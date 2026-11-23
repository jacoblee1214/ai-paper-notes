"""AgentDebug(arXiv:2509.25370) 공개 표·그림 값과 비공식 HF 미러(200 궤적) 재계산.
입력: data/*.parquet (davide221/agenterrorbench, 비공식 미러; 없으면 데이터 부분 건너뜀), 논문에서 옮겨 적은 값.
실행: ../../research_roadmap/.venv/bin/python -I analysis/recompute_tables.py  (pandas+pyarrow 필요)
출력: analysis/recompute_tables_results.json
"""
import json, glob, math, os, collections
from pathlib import Path
from scipy.stats import fisher_exact, binomtest, norm

HERE = Path(__file__).resolve().parent
OUT = {}

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

def mcnemar_p(b, c):
    return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue

def mcnemar_bounds(a, b, n, floor=0):
    """a, b: 두 방법의 성공 수(n 중). floor: 둘 다 성공한 것이 최소 이 수(첫 시도 성공 집합 포함)."""
    best = None; worst = None
    lo_o = max(floor, a + b - n); hi_o = min(a, b)
    for o in range(lo_o, hi_o + 1):
        p = mcnemar_p(a - o, b - o)
        best = p if best is None else min(best, p)
        worst = p if worst is None else max(worst, p)
    return {"p_min": best, "p_max": worst}

# ---- 1. Table 1 (논문 표 1), 환경별 n
N = {"ALFWorld": 100, "WebShop": 50, "GAIA": 50}
T1 = {  # 퍼센트 (S, S+M, ALL)
    "Direct Prompting": {"ALFWorld": (28, 14, 1), "WebShop": (30, 6, 0), "GAIA": (26, 10, 0)},
    "Brute Force": {"ALFWorld": (10, 5, 0), "WebShop": (8, 0, 0), "GAIA": (18, 8, 0)},
    "Binary Search": {"ALFWorld": (20, 6, 1), "WebShop": (14, 8, 0), "GAIA": (22, 10, 0)},
    "AgentDebug": {"ALFWorld": (35, 28, 21), "WebShop": (42, 22, 14), "GAIA": (58, 44, 38)},
}
T1_AVG_REPORTED = {"Direct Prompting": (28.0, 10.0, 0.3), "Brute Force": (12.0, 4.3, 0.0),
                   "Binary Search": (18.7, 8.0, 0.3), "AgentDebug": (45.0, 31.3, 24.3)}
t1 = {}
for m, d in T1.items():
    cnt = {}
    grid_ok = True
    for env, vals in d.items():
        ks = []
        for v in vals:
            k = v * N[env] / 100
            if abs(k - round(k)) > 1e-9: grid_ok = False
            ks.append(round(k))
        cnt[env] = ks
    macro = [round(sum(d[e][i] for e in d) / 3, 1) for i in range(3)]
    pooled = [round(100 * sum(cnt[e][i] for e in cnt) / 200, 1) for i in range(3)]
    t1[m] = {"counts": cnt, "grid_ok": grid_ok, "macro_avg": macro, "pooled_200": pooled,
             "reported_avg": T1_AVG_REPORTED[m]}
OUT["table1"] = t1
# Wilson, AgentDebug vs Direct Prompting (짝지은 가능 범위)
cmp = {}
for i, name in enumerate(["S", "S+M", "ALL"]):
    row = {}
    for env in N:
        a = t1["AgentDebug"]["counts"][env][i]; b = t1["Direct Prompting"]["counts"][env][i]
        n = N[env]
        row[env] = {"AD": f"{a}/{n}", "AD_wilson": wilson(a, n), "DP": f"{b}/{n}", "DP_wilson": wilson(b, n),
                    "mcnemar_bounds": mcnemar_bounds(a, b, n)}
    a = sum(t1["AgentDebug"]["counts"][e][i] for e in N); b = sum(t1["Direct Prompting"]["counts"][e][i] for e in N)
    row["pooled"] = {"AD": f"{a}/200", "AD_wilson": wilson(a, 200), "DP": f"{b}/200", "DP_wilson": wilson(b, 200),
                     "mcnemar_bounds": mcnemar_bounds(a, b, 200)}
    cmp[name] = row
OUT["table1_AD_vs_DirectPrompting"] = cmp

# ---- 2. 하위 문장 불일치 (논문 본문)
OUT["text_vs_table1"] = {
    "findings_text": "50.0% step, 42.5% all-correct (Sec 4.2 Findings)",
    "table1_macro": "45.0% step, 24.3% all-correct",
    "pooled_step_AD": t1["AgentDebug"]["pooled_200"][0],
    "pooled_all_AD": t1["AgentDebug"]["pooled_200"][2],
    "note": "42.5는 풀링 단계 정확도(85/200)와 같고 all-correct가 아니다. 50.0은 어느 값과도 맞지 않는다.",
    "step_rel_gain_pct": round(100 * (45.0 - 28.0) / 28.0, 1),
    "step_abs_gain_pp": 17.0,
}

# ---- 3. 그림 5·6·7 (그림에서 읽은 값)
F5 = {  # ALFWorld 성공률(%), 순서: First, ToT, Vanilla-Debug, Best-of-N(5), Self-Refine, AgentDebug
    "GPT-4o-mini": [21, 21, 29, 31, 41, 55],
    "Qwen3-8B": [48, 48, 65, 68, 74, 74],
    "Qwen3-Next-80B": [60, 60, 76, 78, 78, 84],
}
names = ["First", "ToT", "Vanilla-Debug", "Best-of-N(5)", "Self-Refine", "AgentDebug"]
F7A = {"GPT-4o-mini": [21, 30, 41, 48, 55], "Qwen3-8B": [48, 58, 63, 70, 74], "Qwen3-Next-80B": [60, 76, 80, 80, 84]}
F7A_DELTA = {"GPT-4o-mini": [9, 11, 7, 7], "Qwen3-8B": [10, 5, 7, 4], "Qwen3-Next-80B": [16, 4, 0, 4]}
n_down = 100  # 가정: 논문은 ALFWorld 하위 평가의 과제 수를 적지 않음
f5 = {"assumed_n": n_down, "models": {}}
for mdl, v in F5.items():
    first = v[0]; failed = n_down - first
    ad = v[5]
    r = {"gain_pp": {names[i]: v[i] - first for i in range(1, 6)},
         "gain_relative_pct_AD": round(100 * (ad - first) / first, 1),
         "recovery_of_failed_pct": {names[i]: round(100 * (v[i] - first) / failed, 1) for i in range(1, 6)},
         "share_of_AD_gain_by_BestOfN_pct": round(100 * (v[3] - first) / (ad - first), 1),
         "AD_minus_BestOfN_pp": ad - v[3], "AD_minus_SelfRefine_pp": ad - v[4], "AD_minus_Vanilla_pp": ad - v[2],
         "wilson_AD": wilson(ad, n_down), "wilson_BoN": wilson(v[3], n_down), "wilson_SR": wilson(v[4], n_down)}
    tests = {}
    for j, nm in [(3, "BestOfN"), (4, "SelfRefine"), (2, "Vanilla")]:
        tests[nm] = {"fisher_unpaired_p": fisher_exact([[ad, n_down - ad], [v[j], n_down - v[j]]])[1],
                     "mcnemar_bounds_supersets_of_first": mcnemar_bounds(ad, v[j], n_down, floor=first)}
    r["tests_vs_AD"] = tests
    # 그림 7(a) 누적 값이 그림 5와 일치하는지, 증분 합이 맞는지
    r["fig7a_final_equals_fig5_AD"] = (F7A[mdl][-1] == ad)
    r["fig7a_increments_ok"] = all(F7A[mdl][i + 1] - F7A[mdl][i] == F7A_DELTA[mdl][i] for i in range(4))
    f5["models"][mdl] = r
OUT["fig5_alfworld"] = f5
OUT["fig6_check"] = {"ALFWorld_matches_Qwen3-8B_fig5_pp": {"AD": 74 - 48, "BoN": 68 - 48, "SR": 74 - 48},
                     "reported": {"AD": 26, "BoN": 20, "SR": 26},
                     "avg_check": {"AD": round((26 + 14 + 16) / 3, 2), "BoN": round((20 + 14 + 10) / 3, 2), "SR": round((26 + 12 + 4) / 3, 2)},
                     "qwen3_8b_relative_gain_pct": round(100 * 26 / 48, 1),
                     "gpt4o_mini_abs_gain_pp": 34}
# 그림 7(c): 롤아웃 전략 (n 미기재, 100 가정)
f7c = {"Modular": 38, "Memory+ReAct": 34, "Reflection": 32, "ReAct": 26, "Act Only": 10}
OUT["fig7c"] = {"assumed_n": 100, "fisher_Modular_vs_MemoryReAct_p": fisher_exact([[38, 62], [34, 66]])[1],
                "fisher_Modular_vs_Reflection_p": fisher_exact([[38, 62], [32, 68]])[1],
                "fisher_Modular_vs_ReAct_p": fisher_exact([[38, 62], [26, 74]])[1]}
# 그림 7(b): n=50 (값이 모두 2의 배수)
f7b = {"Llama-3.3-70B": (16, 16, 6, 2), "GPT-4o-mini": (14, 10, 4, 2), "Qwen3-Next-80B": (4, 14, 2, 2), "GPT-4.1": (42, 44, 32, 32)}
OUT["fig7b"] = {"assumed_n": 50, "all_multiples_of_2": all(x % 2 == 0 for v in f7b.values() for x in v),
                "GPT4.1_all_wilson": wilson(16, 50), "GPT4.1_vs_Llama_all_fisher_p": fisher_exact([[16, 34], [1, 49]])[1],
                "GPT4.1_step_wilson": wilson(21, 50), "Llama_step_wilson": wilson(8, 50)}

# ---- 4. 주석자 일치도 κ=0.55 의 의미
mod = {"memory": 38, "reflection": 39, "plan": 78, "action": 22, "system": 22}
tot = sum(mod.values()); pe = sum((v / tot) ** 2 for v in mod.values())
kappa = 0.55
OUT["kappa"] = {"reported": 0.55, "n_double_annotated": None, "ci": None,
                "landis_koch_band": "0.41-0.60 moderate (substantial은 0.61-0.80)",
                "chance_agreement_pe_from_module_marginals": round(pe, 3),
                "implied_raw_agreement_if_module_level": round(kappa * (1 - pe) + pe, 3),
                "mast_reference": "MAST 최종 14모드 κ 0.73 (4 trace), 중간판 0.88"}

# ---- 5. 공개 미러 데이터 재계산
files = sorted(glob.glob(str(HERE.parent / "data" / "*.parquet")))
if files:
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["mod"] = df.critical_failure_module.replace({"planning": "plan"})
    df["ft"] = df.failure_types.map(lambda x: x[0].strip().lower())
    df["ft"] = df.ft.replace({"plan_inefficient": "inefficient_plan"})
    PAPER_MOD = {  # 그림 11-13 (ALFWorld, WebShop, GAIA)
        "alfworld": {"memory": 18, "reflection": 22, "plan": 33, "action": 14, "system": 13},
        "webshop": {"memory": 15, "reflection": 8, "plan": 19, "action": 2, "system": 5},
        "gaia": {"memory": 5, "reflection": 9, "plan": 26, "action": 6, "system": 4}}
    PAPER_STEP = {  # 구간별 수 (1,2,3,4,5,6-10,11-15,16-20,21+)
        "alfworld": [4, 6, 11, 15, 7, 24, 9, 2, 22], "webshop": [5, 9, 0, 6, 2, 17, 7, 0, 3], "gaia": [9, 13, 6, 10, 5, 7, 0, 0, 0]}
    def bins(s):
        edges = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 15), (16, 20), (21, 10**6)]
        return [int(((s >= a) & (s <= b)).sum()) for a, b in edges]
    d = {"n": int(len(df)), "by_env": df.task_type.value_counts().to_dict(),
         "agent_model_by_env": {e: g.llm_model.value_counts().to_dict() for e, g in df.groupby("task_type")},
         "module_counts": {}, "module_diff_vs_paper": {}, "step_bins": {}, "step_bins_diff_vs_paper": {}}
    for e, g in df.groupby("task_type"):
        mc = g["mod"].value_counts().to_dict()
        d["module_counts"][e] = mc
        d["module_diff_vs_paper"][e] = {k: mc.get(k, 0) - PAPER_MOD[e][k] for k in PAPER_MOD[e]}
        b = bins(g.critical_failure_step); d["step_bins"][e] = b
        d["step_bins_diff_vs_paper"][e] = [x - y for x, y in zip(b, PAPER_STEP[e])]
    allb = bins(df.critical_failure_step)
    d["step_bins"]["all"] = allb
    d["share_steps_1_5_pct"] = round(100 * sum(allb[:5]) / 200, 1)
    d["share_steps_6_15_pct"] = round(100 * (allb[5] + allb[6]) / 200, 1)
    d["share_steps_21plus_pct"] = round(100 * allb[8] / 200, 1)
    d["plan_share_pct"] = round(100 * (df["mod"] == "plan").mean(), 1)
    d["memory_plus_reflection_share_pct"] = round(100 * df["mod"].isin(["memory", "reflection"]).mean(), 1)
    # 오류 유형 라벨 위생
    d["error_type_blank"] = int((df.ft == "").sum())
    d["error_type_blank_by_module"] = df[df.ft == ""]["mod"].value_counts().to_dict()
    d["error_type_raw_variants"] = sorted({x[0] for x in df.failure_types if x[0] != x[0].strip().lower() or x[0] in ("plan_inefficient",)})
    d["module_label_variants"] = sorted(df.critical_failure_module.unique().tolist())
    d["types_not_in_paper_table2"] = sorted(set(df.ft) - {"", "over_simplification", "hallucination", "memory_retrieval_failure",
        "progress_misjudge", "outcome_misinterpretation", "causal_misattribution", "constraint_ignorance", "impossible_action",
        "inefficient_plan", "misalignment", "format_error", "parameter_error", "step_limit", "tool_execution_error", "llm_limit", "environment_error"})
    # 상수 추측 기준선 (표본 내 최적 상수; 상한)
    const = {}
    for e, g in list(df.groupby("task_type")) + [("pooled", df)]:
        vc = g.critical_failure_step.value_counts()
        const[e] = {"best_step": int(vc.index[0]), "acc_pct": round(100 * vc.iloc[0] / len(g), 1), "k": int(vc.iloc[0]), "n": int(len(g))}
    d["constant_step_baseline_in_sample"] = const
    # 상수 모듈 기준선 (가장 흔한 모듈)
    d["constant_module_baseline_pct"] = {e: round(100 * g["mod"].value_counts().iloc[0] / len(g), 1) for e, g in list(df.groupby("task_type")) + [("pooled", df)]}
    # 단계+모듈 상수 기준선
    d["constant_step_module_baseline_pct"] = {e: round(100 * g.groupby(["critical_failure_step", "mod"]).size().max() / len(g), 1) for e, g in list(df.groupby("task_type")) + [("pooled", df)]}
    # AD 대 상수 기준선: 환경별 단계 정확도 (AD 35/42/58)
    ad = {"alfworld": 35, "webshop": 21, "gaia": 29}
    d["AD_step_vs_constant_fisher_p"] = {e: fisher_exact([[ad[e], N[{"alfworld": "ALFWorld", "webshop": "WebShop", "gaia": "GAIA"}[e]] - ad[e]],
                                       [const[e]["k"], const[e]["n"] - const[e]["k"]]])[1] for e in ad}
    # ALFWorld 궤적은 모두 30단계(한도)까지 돈 실패인가
    def nasst(s): return sum(m["role"] == "assistant" for m in json.loads(s)["messages"])
    df["nasst"] = df.full_trajectory.map(nasst)
    d["assistant_turns_by_env"] = {e: {"min": int(g.nasst.min()), "max": int(g.nasst.max()), "share_at_30_pct": round(100 * (g.nasst == 30).mean(), 1)} for e, g in df.groupby("task_type")}
    d["critical_step_gt_turns"] = int((df.critical_failure_step > df.nasst).sum())
    d["step_annotations_entries_per_traj"] = sorted({len(json.loads(s)) for s in df.step_annotations})
    d["critical_step_rel_pos_median"] = round(float((df.critical_failure_step / df.nasst).median()), 3)
    OUT["mirror_data"] = d

(HERE / "recompute_tables_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("ok")
