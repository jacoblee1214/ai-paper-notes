"""서베이(Kamoi et al., arXiv:2406.01297 v3)가 근거로 든 원 논문들의 표를 다시 계산한다.

입력은 원 논문 표에서 손으로 옮긴 값이다(출처를 각 항목에 적음). 하는 일:
  1) 값이 정수 개수/표본 수로 떨어지는지 확인해 표본 수 n을 추정한다(표본 수가 본문에 없을 때).
  2) 오라클(정답 신호)로 멈춘 조건과 그렇지 않은 조건의 이득을 한 표로 모은다.
  3) 이득의 Wilson/Newcombe 95% 구간과 두 비율 z 검정(대응 자료가 공개되지 않아 독립 표본 근사, 보수적)을 낸다.
  4) 논문 내부 불일치를 찾는다(RCI 버전 간 기준선 변경, CRITIC 표 2 대 표 10, Self-Refine 신뢰구간 설명).
결과는 recheck_cases_results.json. 외부 호출 없음, 수 초.
실행: python3 -I analysis/recheck_cases.py
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recheck_cases_results.json"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(p1, n1, p2, n2, z=1.96):
    """p2 - p1 의 Newcombe(Wilson 기반) 95% 구간. p는 0-1 비율."""
    l1, u1 = wilson(round(p1 * n1), n1, z)
    l2, u2 = wilson(round(p2 * n2), n2, z)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return lo, hi


def z_p(p1, n1, p2, n2):
    pp = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    return math.erfc(abs(z) / math.sqrt(2))


def fit_n(values_pct, lo=20, hi=2000, tol=0.0101):
    """모든 값이 k/n(소수 둘째 자리 한 칸 이내: 반올림 또는 내림)으로 설명되는 n 목록."""
    ok = []
    for n in range(lo, hi + 1):
        if all(abs(round(v * n / 100) / n * 100 - v) <= tol for v in values_pct):
            ok.append(n)
    return ok


def cmp(a_pct, na, b_pct, nb):
    """a -> b 변화(퍼센트포인트), 구간, p."""
    p1, p2 = a_pct / 100, b_pct / 100
    lo, hi = newcombe(p1, na, p2, nb)
    return {"delta_pp": round(b_pct - a_pct, 2), "ci95_pp": [round(lo * 100, 1), round(hi * 100, 1)],
            "p_unpaired": round(z_p(p1, na, p2, nb), 4)}


res = {}

# ---------------------------------------------------------------- A. RCI (arXiv:2303.17491)
rci_names = ["GSM8K", "MultiArith", "AddSub", "SVAMP", "SingleEq", "AQuA", "CommonSenseQA", "StrategyQA"]
rci = {
    # v3 표 1: 정답 레이블로 멈추는 RCI (본문 §3.1 및 부록 C.2)
    "v3_table1_zero": [77.95, 94.48, 88.58, 80.70, 86.61, 60.23, 64.56, 48.81],
    "v3_table1_rci_oracle_stop": [85.43, 97.64, 89.76, 84.65, 94.49, 67.32, 68.11, 61.81],
    # v3 부록 표 17: 외부 피드백 없음
    "v3_table17_zero": [77.95, 94.48, 88.58, 80.70, 86.61, 60.23, 64.56, 48.42],
    "v3_table17_rci_no_feedback": [72.83, 86.61, 83.07, 79.13, 83.07, 59.84, 46.06, 48.81],
    # v1 표 1 (2023-03): 기준선 행이 v3와 다름
    "v1_table1_zero": [78.35, 96.06, 85.83, 78.35, 91.34, 55.91, 53.15, 51.57],
}
n_fit = {k: fit_n(v, 100, 400) for k, v in rci.items()}
res["A_rci_n_fit_100_400"] = n_fit
# 모든 열에서 공통으로 맞는 n
common = set(n_fit["v3_table1_zero"])
for k in rci:
    common &= set(n_fit[k])
res["A_rci_common_n"] = sorted(common)[:10]
n = 254 if 254 in common else (sorted(common)[0] if common else 254)
rows = []
for i, nm in enumerate(rci_names):
    rows.append({
        "bench": nm,
        "oracle_stop_gain": cmp(rci["v3_table1_zero"][i], n, rci["v3_table1_rci_oracle_stop"][i], n),
        "no_feedback_gain": cmp(rci["v3_table17_zero"][i], n, rci["v3_table17_rci_no_feedback"][i], n),
        "v1_baseline_gain": cmp(rci["v1_table1_zero"][i], n, rci["v3_table1_rci_oracle_stop"][i], n),
        "baseline_shift_v1_to_v3_pp": round(rci["v3_table1_zero"][i] - rci["v1_table1_zero"][i], 2),
    })
res["A_rci_rows_n_used"] = n
res["A_rci_rows"] = rows

# ---------------------------------------------------------------- B. Reflexion (arXiv:2303.11366 v4), 표 5, n=100 (HotPotQA 100문항, §4.2)
refl_t5 = {
    "CoT(GT ctx)+text-davinci-003": (60, 77), "CoT(GT ctx)+gpt-3.5-turbo": (57, 71), "CoT(GT ctx)+gpt-4": (68, 80),
    "ReAct+text-davinci-003": (30, 55), "ReAct+gpt-3.5-turbo": (26, 38), "ReAct+gpt-4": (39, 51),
}
res["B_reflexion_table5_n100"] = {k: cmp(a, 100, b, 100) for k, (a, b) in refl_t5.items()}
# 표 2·3: 자기 생성 단위 테스트(정답 없음)를 쓰는 코드 조건. HumanEval-Py n=164(공식 크기), Rust 50문항(표 3 설명)
res["B_reflexion_code"] = {
    "HumanEval-Py base 0.80 -> 0.91 (n=164)": cmp(80, 164, 91, 164),
    "HumanEval-Rust(hard 50) base 0.60 -> 0.68": cmp(60, 50, 68, 50),
    "HumanEval-Rust(hard 50) base 0.60 -> reflexion w/o test generation 0.52 (표 3)": cmp(60, 50, 52, 50),
    "starchat-beta HumanEval-Py 0.26 -> 0.26 (표 4)": cmp(26, 164, 26, 164),
}

# ---------------------------------------------------------------- C. Self-Refine (arXiv:2303.17651 v2)
# 표 1 (GPT-3.5, ChatGPT, GPT-4)  (base, +SELF-REFINE)
sr_t1 = {
    "Math Reasoning": [(64.1, 64.1), (74.8, 75.0), (92.9, 93.1)],
    "Sentiment Reversal": [(8.8, 30.4), (11.4, 43.2), (3.8, 36.2)],
    "Dialogue Response": [(36.4, 63.6), (40.1, 59.9), (25.4, 74.6)],
    "Constrained Generation": [(28.0, 37.0), (44.0, 67.0), (15.0, 45.0)],
}
# 표 9: 수학, 오라클 피드백(틀린 답만 고침): (base, +SR oracle)
sr_t9 = [(64.06, 68.9), (74.8, 76.2), (92.9, 93.8)]
res["C_selfrefine_math_table1_vs_table9"] = [
    {"model": m, "no_oracle_gain_pp": round(a[1] - a[0], 2), "oracle_gain_pp": round(o[1] - o[0], 2)}
    for m, a, o in zip(["GPT-3.5", "ChatGPT", "GPT-4"], sr_t1["Math Reasoning"], sr_t9)]
# 표 13의 Wilson 구간 설명: 캡션 95%, 본문 99%. 수학(n=1319 가정) 반폭으로 z 추정
for z, lab in [(1.96, "z=1.96 (95%)"), (2.576, "z=2.576 (99%)")]:
    lo, hi = wilson(round(0.641 * 1319), 1319, z)
    res.setdefault("C_selfrefine_table13_halfwidth_math_gpt35_n1319", {})[lab] = round((hi - lo) / 2 * 100, 2)
res["C_selfrefine_table13_reported_halfwidth_pp"] = 3.47
# 수학 이득 0.0~0.2: 구간
res["C_selfrefine_math_gain_ci"] = {m: cmp(a[0], 1319, a[1], 1319) for m, a in zip(["GPT-3.5", "ChatGPT", "GPT-4"], sr_t1["Math Reasoning"])}

# ---------------------------------------------------------------- D. CRITIC (arXiv:2305.11738 v4)
# 표 1 EM, n=500(§4.1). 열: AmbigNQ, TriviaQA, HotpotQA
cr_qa = {
    "text-davinci-003": {"CoT": [44.2, 67.4, 33.7], "CRITIC_wo_tool": [42.0, 67.3, 34.9], "CRITIC": [50.0, 72.7, 38.7], "CRITIC_oracle": [59.8, 77.0, 43.1]},
    "gpt-3.5-turbo": {"CoT": [51.8, 72.9, 32.7], "CRITIC_wo_tool": [55.2, 73.5, 33.1], "CRITIC": [62.0, 75.1, 40.3], "CRITIC_oracle": [69.6, 80.9, 44.3]},
}
qa_names = ["AmbigNQ", "TriviaQA", "HotpotQA"]
res["D_critic_qa_n500"] = {}
for m, d in cr_qa.items():
    for i, ds in enumerate(qa_names):
        res["D_critic_qa_n500"][f"{m}/{ds}"] = {
            k: cmp(d["CoT"][i], 500, d[k][i], 500) for k in ["CRITIC_wo_tool", "CRITIC", "CRITIC_oracle"]}
# CRITIC 대 자기 일관성(표 1, 10 샘플, 같은 EM): RQ3 관점의 강한 기준선 비교
cr_sc = {"text-davinci-003": [44.6, 67.3, 34.9], "gpt-3.5-turbo": [52.6, 75.4, 35.8]}
res["D_critic_vs_selfconsistency_n500"] = {
    f"{m}/{ds}": cmp(cr_sc[m][i], 500, cr_qa[m]["CRITIC"][i], 500) for m in cr_sc for i, ds in enumerate(qa_names)}
# 표 2 (수학 프로그램). 열: GSM8k(n=1319, 논문 부록), SVAMP(공식 test 1000), TabMWP(n 미기재: 1000 가정, 보수적)
cr_math = {
    "text-davinci-003": {"PoT": [70.1, 84.0, 64.6], "CRITIC_wo_tool": [68.3, 80.7, 84.9], "CRITIC": [72.2, 80.7, 87.6], "CRITIC_oracle": [77.4, 91.0, 95.0]},
    "gpt-3.5-turbo": {"PoT": [72.5, 82.0, 75.0], "CRITIC_wo_tool": [77.0, 82.0, 87.0], "CRITIC": [78.2, 83.3, 89.0], "CRITIC_oracle": [83.9, 89.0, 94.0]},
}
math_n = {"GSM8k": 1319, "SVAMP": 1000, "TabMWP(n=1000 가정)": 1000}
res["D_critic_math"] = {}
for m, d in cr_math.items():
    for i, (ds, nn) in enumerate(math_n.items()):
        res["D_critic_math"][f"{m}/{ds}"] = {
            k: cmp(d["PoT"][i], nn, d[k][i], nn) for k in ["CRITIC_wo_tool", "CRITIC", "CRITIC_oracle"]}
# 내부 불일치: 표 2 davinci GSM8k CRITIC 72.2 (+2.1) 대 표 10 +CRITIC 71.2 (+1.1)
res["D_critic_internal_check"] = {"table2_davinci_gsm8k_critic": 72.2, "table10_davinci_gsm8k_critic": 71.2,
                                   "pot_baseline": 70.1, "delta_table2": round(72.2 - 70.1, 2), "delta_table10": round(71.2 - 70.1, 2)}

# ---------------------------------------------------------------- E. Huang et al. (arXiv:2310.01798 v2)
# 표 2(오라클) 표 3(내재). GPT-3.5: 전체 평가셋 GSM8K 1319, CSQA 1221, HotpotQA 100. GPT-4: 각 200문항(HotpotQA 100)
hu = {
    "GPT-3.5": {"n": [1319, 1221, 100], "std": [75.9, 75.8, 26.0], "oracle": [84.3, 89.7, 29.0], "intrinsic_r1": [75.1, 38.1, 25.0], "intrinsic_r2": [74.7, 41.8, 25.0]},
    "GPT-4": {"n": [200, 200, 100], "std": [95.5, 82.0, 49.0], "oracle": [97.5, 85.5, 59.0], "intrinsic_r1": [91.5, 79.5, 49.0], "intrinsic_r2": [89.0, 80.0, 43.0]},
}
hn = ["GSM8K", "CommonSenseQA", "HotpotQA"]
res["E_huang"] = {}
for m, d in hu.items():
    for i, ds in enumerate(hn):
        res["E_huang"][f"{m}/{ds}"] = {k: cmp(d["std"][i], d["n"][i], d[k][i], d["n"][i]) for k in ["oracle", "intrinsic_r1", "intrinsic_r2"]}
res["E_huang_n_fit_check"] = {f"{m}/{hn[i]}": fit_n([d["std"][i], d["oracle"][i], d["intrinsic_r1"][i], d["intrinsic_r2"][i]], 100, 1400)[:1] + [d["n"][i]]
                              for m, d in hu.items() for i in range(3)}
# 표 7: 다중 에이전트 토론 대 자기 일관성(GSM8K 전체 1319)
res["E_huang_debate_vs_sc_gsm8k_n1319"] = {
    "debate_r1(6 responses) vs SC6": cmp(85.3, 1319, 83.2, 1319),
    "debate_r2(9 responses) vs SC9": cmp(88.2, 1319, 83.0, 1319),
    "debate_r1(6 responses) vs SC3(3 responses)": cmp(82.5, 1319, 83.2, 1319),
}
# 표 8: CommonGen-Hard. Self-Refine ChatGPT 44.0 -> 67.0 (Self-Refine 표 1), Huang 개선 초기 프롬프트 81.8 -> 75.1
res["E_huang_commongen"] = {"selfrefine_reported": [44.0, 67.0], "huang_stronger_init_prompt": [81.8, 75.1],
                            "init_prompt_gain_pp": round(81.8 - 44.0, 1), "selfrefine_gain_pp": 23.0,
                            "note": "표본 수는 두 논문 모두 이 표에서 확인하지 못함(Self-Refine 표 1 CommonGen 열은 퍼센트 단위가 정수라 n=100 추정)"}

# ---------------------------------------------------------------- F. Welleck et al. (arXiv:2211.00053 v1) 표 1
we = {"Multiarith": [60.00, 98.33, 99.17], "Multitask": [49.02, 73.53, 78.24], "GSM": [8.57, 21.26, 24.22]}
res["F_welleck_n_fit"] = {k: fit_n(v, 20, 1400)[:6] for k, v in we.items()}

# ---------------------------------------------------------------- G. Self-Debug (arXiv:2304.05128 v2) 표 2 대 표 3
sd = {
    "MBPP(n=500)": {"n": 500, "base": {"Codex": 61.4, "GPT-3.5": 67.6, "GPT-4": 72.8},
                    "with_exec": {"Codex": {"Simple": 68.2, "UT": 69.4, "+Expl": 69.8, "+Trace": 70.8}, "GPT-3.5": {"Simple": 70.8, "UT": 72.2, "+Expl": 74.2, "+Trace": 72.8}, "GPT-4": {"Simple": 78.8, "UT": 80.6, "+Expl": 80.4, "+Trace": 80.2}},
                    "no_exec": {"Codex": {"Simple": 57.6, "+Expl": 64.4, "+Trace": 66.2}, "GPT-3.5": {"Simple": 68.2, "+Expl": 68.2, "+Trace": 69.2}, "GPT-4": {"Simple": 76.0, "+Expl": 76.0, "+Trace": 76.4}}},
    "TransCoder(n=560)": {"n": 560, "base": {"Codex": 80.4, "GPT-3.5": 89.1, "GPT-4": 77.3},
                          "with_exec": {"Codex": {"Simple": 89.3, "UT": 91.6, "+Expl": 92.5, "+Trace": 87.9}, "GPT-3.5": {"Simple": 91.6, "UT": 92.7, "+Expl": 92.7, "+Trace": 92.3}, "GPT-4": {"Simple": 80.9, "UT": 88.8, "+Expl": 90.4, "+Trace": 89.5}},
                          "no_exec": {"Codex": {"Simple": 83.4, "+Expl": 83.9, "+Trace": 83.9}, "GPT-3.5": {"Simple": 89.1, "+Expl": 89.1, "+Trace": 89.1}, "GPT-4": {"Simple": 78.2, "+Expl": 78.0, "+Trace": 78.4}}},
}
res["G_selfdebug_n_fit"] = {
    "MBPP": fit_n([61.4, 67.6, 72.8, 68.2, 70.8, 78.8, 57.6, 64.4, 66.2], 100, 1200)[:3],
    "TransCoder": fit_n([80.4, 89.1, 77.3, 89.3, 91.6, 80.9, 83.4, 78.2], 100, 1200)[:3],
}
g = {}
for ds, d in sd.items():
    for m in ["Codex", "GPT-3.5", "GPT-4"]:
        b = d["base"][m]
        g[f"{ds}/{m}"] = {
            "with_exec_UT": cmp(b, d["n"], d["with_exec"][m]["UT"], d["n"]),
            "no_exec_trace": cmp(b, d["n"], d["no_exec"][m]["+Trace"], d["n"]),
            "no_exec_simple": cmp(b, d["n"], d["no_exec"][m]["Simple"], d["n"]),
        }
res["G_selfdebug"] = g
spider = {"Codex": (81.3, 81.3, 84.1), "GPT-3.5": (71.1, 72.2, 72.2), "GPT-4": (73.2, 73.4, 73.6), "StarCoder": (64.7, 64.9, 64.9)}
res["G_selfdebug_spider_n1034_no_unit_test"] = {m: {"simple": cmp(b, 1034, s, 1034), "+Expl": cmp(b, 1034, e, 1034)} for m, (b, s, e) in spider.items()}
res["G_selfdebug_spider_n_fit"] = fit_n([81.3, 71.1, 73.2, 64.7, 72.2, 73.4, 84.1, 73.6], 500, 1500)[:3]

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
