"""논문 표 재계산과 통계 검정 (공개 표 + 공개 데이터의 기준선만 사용, 모델 호출 없음).

1. Table 1 셀이 k/N으로 설명되는가(분모 추정)
2. Table 1 'Random' 행을 공개 데이터에서 재계산해 대조 (12.00 / 19.06 위치 문제)
3. 초록의 53.5%, 14.2% 가 어떤 평균에서 나오는가
4. 방법 간·기준선 대비 정확도 검정 (Wilson 구간, Fisher, 포아송-이항 정확 검정)
5. Table 2(허용 오차), Table 3, Table 4 내부 일치
사용: python3 -I analysis/recompute_tables.py   (analysis/data_stats_results.json 이 먼저 있어야 함)
"""
import json, math, os, itertools
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data_stats_results.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "recompute_results.json")
res = {}

# ---------------------------------------------------------------- 논문 표 값 (v3, Table 1)
T1 = {  # (system, setting, method, metric): value
    ("alg", "GT", "random", "agent"): 29.10, ("alg", "GT", "random", "step"): 19.06,
    ("hand", "GT", "random", "agent"): 12.00, ("hand", "GT", "random", "step"): 4.16,
    ("alg", "GT", "AAO", "agent"): 54.33, ("alg", "GT", "AAO", "step"): 12.50,
    ("hand", "GT", "AAO", "agent"): 55.17, ("hand", "GT", "AAO", "step"): 5.26,
    ("alg", "noGT", "AAO", "agent"): 51.12, ("alg", "noGT", "AAO", "step"): 13.53,
    ("hand", "noGT", "AAO", "agent"): 53.44, ("hand", "noGT", "AAO", "step"): 3.51,
    ("alg", "GT", "SBS", "agent"): 35.20, ("alg", "GT", "SBS", "step"): 25.51,
    ("hand", "GT", "SBS", "agent"): 34.48, ("hand", "GT", "SBS", "step"): 7.02,
    ("alg", "noGT", "SBS", "agent"): 26.02, ("alg", "noGT", "SBS", "step"): 15.31,
    ("hand", "noGT", "SBS", "agent"): 32.75, ("hand", "noGT", "SBS", "step"): 8.77,
    ("alg", "GT", "BS", "agent"): 44.13, ("alg", "GT", "BS", "step"): 23.98,
    ("hand", "GT", "BS", "agent"): 51.72, ("hand", "GT", "BS", "step"): 6.90,
    ("alg", "noGT", "BS", "agent"): 30.11, ("alg", "noGT", "BS", "step"): 16.59,
    ("hand", "noGT", "BS", "agent"): 36.21, ("hand", "noGT", "BS", "step"): 6.90,
}


def fits(v, N):
    """k/N 이 표 값 v 와 소수 둘째 자리(반올림 또는 버림)로 같은 k 를 반환."""
    for k in range(N + 1):
        x = 100 * k / N
        if round(x, 2) == v or math.floor(x * 100) / 100 == v:
            return k
    return None


# ---------------------------------------------------------------- 1. 분모 추정
den = {}
for key, v in T1.items():
    if key[2] == "random":
        continue
    sysn = key[0]
    cands = [58, 57] if sysn == "hand" else [126, 98, 28, 127]
    den["/".join(key)] = {str(N): fits(v, N) for N in cands if fits(v, N) is not None}
res["denominator_fit"] = den
res["denominator_summary"] = {
    "hand_agent_level_all_k_over_58": all("58" in den[k] for k in den if k.startswith("hand") and k.endswith("agent")),
    "hand_step_cells_k_over_57": [k for k in den if k.startswith("hand") and k.endswith("step") and "57" in den[k]],
    "hand_step_cells_k_over_58": [k for k in den if k.startswith("hand") and k.endswith("step") and "58" in den[k]],
    "alg_cells_k_over_126": [k for k in den if k.startswith("alg") and "126" in den[k]],
    "alg_cells_with_no_fit_126_98_28_127": [k for k in den if k.startswith("alg") and not den[k]],
}

# ---------------------------------------------------------------- 2. Random 행 재계산
H, A = D["Hand-Crafted"], D["Algorithm-Generated"]
recomp = {
    "alg_agent": A["baselines_pct"]["agent_random_uniform_over_distinct_agents"],
    "alg_step": A["baselines_pct"]["step_random_uniform"],
    "hand_agent": H["baselines_pct"]["agent_random_uniform_over_distinct_agents"],
    "hand_step": H["baselines_pct"]["step_random_uniform"],
}
paper = {"alg_agent": 29.10, "alg_step": 19.06, "hand_agent": 12.00, "hand_step": 4.16}
res["random_row"] = {"paper_table1": paper, "recomputed": recomp,
    "note": "재계산 hand_agent=19.06, alg_step=12.00 → 표의 두 값(12.00, 19.06)이 자리가 바뀐 것으로 보임"}

# ---------------------------------------------------------------- 3. 초록 숫자
avg = lambda s, m, t: (T1[(s, "GT", m, t)] + T1[(s, "noGT", m, t)]) / 2
w_alg, w_hand = 126, 58
ab = {}
for m in ("AAO", "SBS", "BS"):
    for t in ("agent", "step"):
        a, h = avg("alg", m, t), avg("hand", m, t)
        ab[f"{m}_{t}"] = dict(alg_avg=round(a, 2), hand_avg=round(h, 2), unweighted_mean=round((a + h) / 2, 2),
                              weighted_by_n=round((w_alg * a + w_hand * h) / (w_alg + w_hand), 2))
res["abstract_numbers"] = {"paper": {"best_agent": 53.5, "best_step": 14.2}, "by_method": ab}

# ---------------------------------------------------------------- 4. 검정 도구
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def poisson_binomial_sf(ps, k):
    """P(X >= k), X = sum of independent Bernoulli(p_i)."""
    dist = [1.0]
    for p in ps:
        nd = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            nd[i] += q * (1 - p)
            nd[i + 1] += q * p
        dist = nd
    return sum(dist[k:])


def fisher(k1, n1, k2, n2):
    return stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def cell(sysn, setting, m, t):
    v = T1[(sysn, setting, m, t)]
    if sysn == "hand":
        N = 57 if (t == "step" and m in ("AAO", "SBS")) else 58
    else:
        N = 126
    k = fits(v, N) if fits(v, N) is not None else round(v * N / 100)
    return k, N


# per-trajectory 확률 벡터를 만들기 위해 data_stats 의 기준선을 그대로 쓰되, 정확 검정은 균등 무작위에 대해서만 한다.
# (균등 무작위 기대값만 json 에 있으므로 정규 근사 대신 평균 확률로 이항 근사: 보수적 표시)
def binom_vs(k, n, p):
    return stats.binomtest(k, n, p, alternative="greater").pvalue

tests = {}
# 4a. 손수 만든 58건, 방법 vs 기준선 (agent)
rows = []
base_hand_agent = {
    "paper_random(19.06, role-string)": H["baselines_pct"]["agent_random_uniform_over_distinct_agents"] / 100,
    "random over base agents (42.21)": H["baselines_base_pct"]["agent_random_uniform_over_base_agents"] / 100,
    "constant WebSurfer (56.90)": H["baselines_base_pct"]["agent_constant_WebSurfer"] / 100,
    "last non-human speaker (60.34)": H["baselines_base_pct"]["agent_last_speaker_base"] / 100,
}
for setting in ("GT", "noGT"):
    for m in ("AAO", "SBS", "BS"):
        k, N = cell("hand", setting, m, "agent")
        row = dict(setting=setting, method=m, k=k, N=N, pct=round(100 * k / N, 2), wilson95=wilson(k, N))
        for name, p in base_hand_agent.items():
            row[f"p_greater_than[{name}]"] = round(binom_vs(k, N, p), 4)
        rows.append(row)
tests["hand_agent_vs_baselines"] = rows

# 4b. 손수 만든 step vs 기준선
base_hand_step = {
    "random 1/n (4.16)": H["baselines_pct"]["step_random_uniform"] / 100,
    "always last step (12.07)": H["baselines_pct"]["step_always_last"] / 100,
}
rows = []
for setting in ("GT", "noGT"):
    for m in ("AAO", "SBS", "BS"):
        k, N = cell("hand", setting, m, "step")
        row = dict(setting=setting, method=m, k=k, N=N, pct=round(100 * k / N, 2), wilson95=wilson(k, N))
        for name, p in base_hand_step.items():
            row[f"p_greater_than[{name}]"] = round(binom_vs(k, N, p), 4)
        rows.append(row)
tests["hand_step_vs_baselines"] = rows

# 4c. 알고리즘 생성 126건 (분모는 근사: 표 값이 k/126 이 아님)
base_alg_agent = {
    "random (29.10)": A["baselines_pct"]["agent_random_uniform_over_distinct_agents"] / 100,
    "first speaker (49.21)": A["baselines_pct"]["agent_first_speaker"] / 100,
    "most talkative (42.86)": A["baselines_pct"]["agent_most_talkative"] / 100,
}
base_alg_step = {
    "paper_random(19.06)": 0.1906,
    "recomputed random 1/n (12.00)": A["baselines_pct"]["step_random_uniform"] / 100,
    "always step 0 (15.87)": A["baselines_pct"]["step_always_first(0)"] / 100,
}
for t, bases in (("agent", base_alg_agent), ("step", base_alg_step)):
    rows = []
    for setting in ("GT", "noGT"):
        for m in ("AAO", "SBS", "BS"):
            k, N = cell("alg", setting, m, t)
            row = dict(setting=setting, method=m, k_approx=k, N=N, pct=T1[("alg", setting, m, t)], wilson95_approx=wilson(k, N))
            for name, p in bases.items():
                row[f"p_greater_than[{name}]"] = round(binom_vs(k, N, p), 4)
            rows.append(row)
    tests[f"alg_{t}_vs_baselines"] = rows

# 4d. 방법 간 (비짝 Fisher: 공개된 개별 예측이 없어 짝지은 검정(McNemar)은 불가 → 보수적)
pair = []
for sysn in ("hand", "alg"):
    for setting in ("GT", "noGT"):
        for t in ("agent", "step"):
            for m1, m2 in itertools.combinations(("AAO", "SBS", "BS"), 2):
                k1, n1 = cell(sysn, setting, m1, t); k2, n2 = cell(sysn, setting, m2, t)
                pair.append(dict(system=sysn, setting=setting, metric=t, a=m1, b=m2,
                                 pct_a=round(100 * k1 / n1, 2), pct_b=round(100 * k2 / n2, 2), fisher_p=round(fisher(k1, n1, k2, n2), 4)))
tests["method_pairs_fisher"] = pair
# 4e. GT vs noGT (같은 궤적이라 비짝 Fisher 는 보수적)
gt = []
for sysn in ("hand", "alg"):
    for m in ("AAO", "SBS", "BS"):
        for t in ("agent", "step"):
            k1, n1 = cell(sysn, "GT", m, t); k2, n2 = cell(sysn, "noGT", m, t)
            gt.append(dict(system=sysn, method=m, metric=t, gt=round(100 * k1 / n1, 2), nogt=round(100 * k2 / n2, 2), fisher_p=round(fisher(k1, n1, k2, n2), 4)))
tests["gt_vs_nogt_fisher"] = gt
res["tests"] = tests
res["tests_summary"] = {
    "hand_agent_pairs_p_lt_0.05": [p for p in pair if p["system"] == "hand" and p["metric"] == "agent" and p["fisher_p"] < 0.05],
    "hand_step_pairs_p_lt_0.05": [p for p in pair if p["system"] == "hand" and p["metric"] == "step" and p["fisher_p"] < 0.05],
    "alg_agent_pairs_p_lt_0.05": [p for p in pair if p["system"] == "alg" and p["metric"] == "agent" and p["fisher_p"] < 0.05],
    "alg_step_pairs_p_lt_0.05": [p for p in pair if p["system"] == "alg" and p["metric"] == "step" and p["fisher_p"] < 0.05],
    "gt_vs_nogt_p_lt_0.05": [g for g in gt if g["fisher_p"] < 0.05],
}

# ---------------------------------------------------------------- 5a. Table 2 허용 오차 vs 균등 무작위 / 마지막 단계
T2 = {1: (12.07, 14.66, 13.79), 2: (19.83, 16.38, 18.97), 3: (30.17, 18.10, 22.41), 4: (37.07, 31.90, 31.89), 5: (43.10, 33.62, 36.21)}
t2 = []
for tol, (aao, sbs, bs) in T2.items():
    rnd = H["baselines_pct"][f"step_random_tol{tol}"]
    last = H["baselines_pct"][f"step_last_tol{tol}"]
    row = dict(tol=tol, AAO=aao, SBS=sbs, BS=bs, random_uniform=rnd, always_last=last,
               methods_above_random=[n for n, v in (("AAO", aao), ("SBS", sbs), ("BS", bs)) if v > rnd],
               k_over_116_AAO=fits(aao, 116), k_over_116_SBS=fits(sbs, 116), k_over_116_BS=fits(bs, 116))
    # N=58 궤적 기준 단측 이항 검정 (설정 2개가 같은 궤적이므로 116 을 쓰지 않음)
    for n_, v in (("AAO", aao), ("SBS", sbs), ("BS", bs)):
        k = round(v * 58 / 100)
        row[f"p_{n_}_gt_random"] = round(binom_vs(k, 58, rnd / 100), 3)
    t2.append(row)
res["table2_tolerance"] = t2

# ---------------------------------------------------------------- 5b. Table 3 vs Table 1/4 일치
T4_gpt4o = dict(AAO_agent=54.31, AAO_step=4.39, SBS_agent=33.62, SBS_step=7.90)
from_T1 = {  # 설정 2개를 합쳐 116 분모로 풀링
    "AAO_agent": (fits(55.17, 58) + fits(53.44, 58), 116), "AAO_step": (fits(5.26, 57) + fits(3.51, 57), 114),
    "SBS_agent": (fits(34.48, 58) + fits(32.75, 58), 116), "SBS_step": (fits(7.02, 57) + fits(8.77, 57), 114),
    "BS_agent": (fits(51.72, 58) + fits(36.21, 58), 116), "BS_step": (fits(6.90, 58) * 2, 116),
}
chk = {k: dict(pooled_k=v[0], pooled_N=v[1], pooled_pct=round(100 * v[0] / v[1], 2)) for k, v in from_T1.items()}
T3 = dict(BS_agent=43.97, AAO_agent=57.02, SBS_agent=35.96, hybrid_agent=57.02, BS_step=6.90, AAO_step=4.39, SBS_step=7.90, hybrid_step=12.28)
res["table3_vs_table1"] = dict(
    table1_pooled=chk, table4_gpt4o_paper=T4_gpt4o, table3_paper=T3,
    mismatch={"AAO_agent": dict(table1_pooled=chk["AAO_agent"]["pooled_pct"], table3=57.02, k_over_114_table3=fits(57.02, 114)),
              "SBS_agent": dict(table1_pooled=chk["SBS_agent"]["pooled_pct"], table3=35.96, k_over_114_table3=fits(35.96, 114))},
    hybrid_step=dict(pct=12.28, k_over_57=fits(12.28, 57), k_over_58=fits(12.28, 58), k_over_114=fits(12.28, 114)),
    note="Table 3 의 AAO·SBS 에이전트 정확도가 Table 1·4 의 풀링 값(54.31, 33.62)과 다름. 단계 정확도는 일치.")

# hybrid vs 최선 단일(SBS step 7.90) : 7/57 대 9/114 는 N=57 vs 114 동일 궤적 – 보수적으로 (7/57 vs 4.5/57)
res["table3_hybrid_step_tests"] = dict(
    hybrid_k=7, N=57, vs_SBS_single_setting_k=round(7.90 * 57 / 100), fisher_p=round(fisher(7, 57, round(7.90 * 57 / 100), 57), 3),
    vs_always_last_12_07=round(binom_vs(7, 57, H["baselines_pct"]["step_always_last"] / 100), 3),
    wilson95_hybrid=wilson(7, 57))

# ---------------------------------------------------------------- 5c. Table 4 (o1, R1) k/58
T4 = {"o1": dict(AAO_agent=41.38, AAO_step=10.34, SBS_agent=36.21, SBS_step=13.79),
      "R1": dict(AAO_agent=56.90, AAO_step=3.45, SBS_agent=32.76, SBS_step=6.90)}
g4 = dict(AAO_agent=(63, 116), AAO_step=(5, 114), SBS_agent=(39, 116), SBS_step=(9, 114))
t4 = {}
for mdl, vals in T4.items():
    t4[mdl] = {}
    for key, v in vals.items():
        k = fits(v, 58)
        kg, ng = g4[key]
        t4[mdl][key] = dict(pct=v, k_over_58=k, wilson95=wilson(k, 58), fisher_p_vs_gpt4o=round(fisher(k, 58, round(kg * 58 / ng), 58), 3))
res["table4_reasoning_models"] = t4

# ---------------------------------------------------------------- 5d. 부록 B.1 추론 프롬프트 ablation (Figure 7)
f7 = [("alg agent AAO", 52.7, 45.3, 126), ("alg step SBS", 20.4, 16.0, 126), ("hand agent AAO", 54.3, 47.3, 58), ("hand step SBS", 7.9, 6.9, 58)]
res["fig7_reasoning_ablation_fisher"] = [dict(cell=n, with_r=a, without=b, diff_pp=round(a - b, 1),
    fisher_p=round(fisher(round(a * N / 100), N, round(b * N / 100), N), 3), N=N) for n, a, b, N in f7]


# ---------------------------------------------------------------- 6. AgenTracer(arXiv:2509.03312) 표 1의 Who&When 열과 대조
# 값은 ../agentracer/analysis/recompute_tables.py 의 T (AgenTracer 논문 표 1)에서 옮긴 것. (w/ G, w/o G)
AT = {  # 모델: (hand agent, hand step, auto agent, auto step) 의 w/G 값
    "AgenTracer-8B": (69.10, 20.68, 69.62, 42.86), "Claude-Sonnet-4": (56.90, 17.24, 57.93, 40.65),
    "DeepSeek-R1": (56.90, 13.29, 66.67, 31.32), "GPT-4.1": (43.10, 3.44, 55.55, 29.52),
    "Gemini-2.5-Pro": (51.72, 9.72, 61.11, 29.52), "Qwen3-8B": (42.10, 1.72, 58.73, 3.97)}
# 분모: 손수 58, 자동 126 (이 폴더의 데이터로 확인됨). 셀이 k/N 이 아니면 반올림 k 사용.
KN = lambda v, N: fits(v, N) if fits(v, N) is not None else round(v * N / 100)
base = {"hand_agent": [("last non-human speaker", 35, 58), ("constant WebSurfer", 33, 58)],
        "hand_step": [("always last step", 7, 58)],
        "auto_agent": [("first speaker", 62, 126), ("most talkative", 54, 126)],
        "auto_step": [("always step 0", 20, 126)]}
atx = []
for mdl, vals in AT.items():
    for key, v, N in zip(("hand_agent", "hand_step", "auto_agent", "auto_step"), vals, (58, 58, 126, 126)):
        k = KN(v, N)
        row = dict(model=mdl, cell=key, pct=v, k=k, N=N, k_is_exact=fits(v, N) is not None)
        for nm, kb, nb_ in base[key]:
            row[f"fisher_p_vs[{nm} {round(100*kb/nb_,2)}]"] = round(fisher(k, N, kb, nb_), 4)
        atx.append(row)
res["agentracer_crosscheck"] = dict(
    note="AgenTracer 표 1 Who&When 열의 w/G 값과 이 데이터의 단순 규칙 기준선 비교. 분모는 58/126 로 확인됨(데이터 개수). 기준선은 데이터 라벨을 보고 고르는 규칙이 아니라 고정 규칙(마지막 단계 등)이거나 라벨 최빈값(WebSurfer 상수)임.",
    rows=atx)
for r in atx:
    print(r)

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ("denominator_summary", "random_row", "abstract_numbers", "tests_summary", "table3_vs_table1", "table3_hybrid_step_tests", "table4_reasoning_models", "fig7_reasoning_ablation_fisher")}, ensure_ascii=False, indent=1))
for r in res["table2_tolerance"]:
    print(r)
for k in ("hand_agent_vs_baselines", "hand_step_vs_baselines", "alg_agent_vs_baselines", "alg_step_vs_baselines"):
    print(k)
    for r in tests[k]:
        print(" ", r)
