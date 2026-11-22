"""MAST-Data(HF mcemri/MAD, MAD_full_dataset.json, 1642 trace)로 논문 수치를 재계산한다.
 - 표 1 trace 수, 그림 1·2 실패 모드 비율, 표 6(오픈소스 모델 400 trace), 그림 8·9 비교, 표 8, 그림 6·7 상관
 - LLM 판정자 라벨(o1) 기준이다. 사람 라벨이 아니다.
실행: python3 -I analysis/mad_recompute.py
"""
import json, os, collections, itertools
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
F = json.load(open(os.path.join(ROOT, "data/hf/MAD_full_dataset.json")))
MODES = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3.1", "3.2", "3.3"]
FC = {"FC1": MODES[:5], "FC2": MODES[5:11], "FC3": MODES[11:]}
res = {}


def vec(x):
    a = x["mast_annotation"]
    return None if any(a.get(m) is None for m in MODES) else np.array([a[m] for m in MODES], dtype=int)


def bench(x):
    """HF 파일은 ProgramDev(30)와 ProgramDev-v2(100)를 모두 'ProgramDev'로 표기한다. trace.key로 구분한다."""
    k = x["trace"]["key"]
    return "ProgramDev-v2" if "ProgramDev2" in k else x["benchmark_name"]


for x in F:
    x["benchmark_name_raw"] = x["benchmark_name"]
    x["benchmark_name"] = bench(x)

rows = [(x, vec(x)) for x in F]
valid = [(x, v) for x, v in rows if v is not None]
res["n_total"] = len(F)
res["n_with_null_annotation"] = len(F) - len(valid)
res["null_annotation_trace"] = [(x["mas_name"], x["llm_name"], x["benchmark_name"], x["trace_id"]) for x, v in rows if v is None]

# 표 1: 구성별 trace 수
cfg = collections.Counter((x["mas_name"], x["benchmark_name"], x["llm_name"]) for x in F)
res["table1_counts"] = {"|".join(k): v for k, v in sorted(cfg.items())}
print("구성 수", len(cfg), "합계", sum(cfg.values()))

# 모드별 trace 비율과 총 발생 대비 비율
V = np.array([v for _, v in valid])
tot = V.sum()
res["mode_trace_rate"] = {m: float(V[:, i].mean()) for i, m in enumerate(MODES)}
res["mode_share_of_all_flags"] = {m: float(V[:, i].sum() / tot) for i, m in enumerate(MODES)}
res["total_flags"] = int(tot)
res["mean_flags_per_trace"] = float(V.sum(1).mean())
res["category_share_of_all_flags"] = {k: float(V[:, [MODES.index(m) for m in ms]].sum() / tot) for k, ms in FC.items()}
res["category_trace_any_rate"] = {k: float((V[:, [MODES.index(m) for m in ms]].sum(1) > 0).mean()) for k, ms in FC.items()}
res["traces_with_zero_flags"] = int((V.sum(1) == 0).sum())
print("총 flag", tot, "trace당", res["mean_flags_per_trace"])
print("모드별 trace 비율", {m: round(r, 3) for m, r in res["mode_trace_rate"].items()})
print("모드별 flag 점유율", {m: round(r, 3) for m, r in res["mode_share_of_all_flags"].items()})
print("범주 점유율", res["category_share_of_all_flags"])

# 그림 1 값(모드별)과 그림 2 값
fig1 = {"1.1": 11.8, "1.2": 1.5, "1.3": 15.7, "1.4": 2.8, "1.5": 12.4, "2.1": 2.2, "2.2": 6.8, "2.3": 7.4, "2.4": 0.8, "2.5": 1.9, "2.6": 13.2, "3.1": 6.2, "3.2": 8.2, "3.3": 9.1}
fig2 = {"1.1": 15.2, "1.2": 1.57, "1.3": 11.5, "1.4": 2.36, "1.5": 6.54, "2.1": 5.50, "2.2": 2.09, "2.3": 5.50, "2.4": 6.02, "2.5": 4.71, "2.6": 7.59, "3.1": 8.64, "3.2": 9.16, "3.3": 13.61}
res["fig1_stated"] = fig1
res["fig2_stated"] = fig2
res["fig1_sum"] = sum(fig1.values())
res["fig2_sum"] = sum(fig2.values())
res["fig1_cat_sum"] = {"FC1": sum(fig1[m] for m in FC["FC1"]), "FC2": sum(fig1[m] for m in FC["FC2"]), "FC3": sum(fig1[m] for m in FC["FC3"])}
res["fig2_cat_sum"] = {"FC1": sum(fig2[m] for m in FC["FC1"]), "FC2": sum(fig2[m] for m in FC["FC2"]), "FC3": sum(fig2[m] for m in FC["FC3"])}
# 그림 1 값을 '총 flag 대비 점유율'과 비교
cmp1 = {m: (fig1[m], round(100 * res["mode_share_of_all_flags"][m], 1), round(100 * res["mode_trace_rate"][m], 1)) for m in MODES}
res["fig1_vs_data(stated, share_of_flags, trace_rate)"] = cmp1
print("그림 1 합", res["fig1_sum"], "그림 2 합", res["fig2_sum"], res["fig1_cat_sum"], res["fig2_cat_sum"])
print("그림 1 대 데이터", cmp1)

# 그림 1/2와 가장 가까운 부분집합 찾기(MAS 단위 또는 MAS x LLM 단위 합)
def share_for(sel):
    S = np.array([v for x, v in valid if sel(x)])
    return {m: 100 * S[:, i].sum() / S.sum() for i, m in enumerate(MODES)}, len(S)


def dist(a, b):
    return float(np.sqrt(sum((a[m] - b[m]) ** 2 for m in MODES)))


# (a) 첫 30 trace씩(그림 4: 7개 시스템 x 30 = 210). trace_id 순서가 아닌 trace.index 기준이 없으므로 trace_id 오름차순
by_key = collections.defaultdict(list)
for x, v in valid:
    by_key[(x["mas_name"], x["llm_name"], x["benchmark_name"])].append((int(x["trace_id"]), v))
first30 = []
for k, L in by_key.items():
    if k in [("AG2", "GPT-4o", "GSM"), ("AppWorld", "GPT-4o", "Test-C"), ("HyperAgent", "Claude", "SWE-Bench-Lite"), ("OpenManus", "GPT-4o", "ProgramDev"),
             ("ChatDev", "GPT-4o", "ProgramDev"), ("MetaGPT", "GPT-4o", "ProgramDev"), ("Magentic", "GPT-4o", "GAIA")]:
        first30 += [v for _, v in sorted(L, key=lambda t: t[0])[:30]]
S30 = np.array(first30)
res["first30_per_system_n"] = len(S30)
share30 = {m: 100 * S30[:, i].sum() / S30.sum() for i, m in enumerate(MODES)}
res["first30_share"] = share30
res["first30_category_share"] = {k: float(100 * S30[:, [MODES.index(m) for m in ms]].sum() / S30.sum()) for k, ms in FC.items()}
print("첫30/시스템 n=", len(S30), "범주 점유율", res["first30_category_share"], "(그림 4 머리말: 41.8/36.9/21.3)")
allshare = {m: 100 * res["mode_share_of_all_flags"][m] for m in MODES}
res["dist_fig1_to_all"] = dist(fig1, allshare)
res["dist_fig2_to_all"] = dist(fig2, allshare)
res["dist_fig1_to_first30"] = dist(fig1, share30)
res["dist_fig2_to_first30"] = dist(fig2, share30)
print("거리(그림1,전체)", res["dist_fig1_to_all"], "(그림2,전체)", res["dist_fig2_to_all"], "(그림1,첫30)", res["dist_fig1_to_first30"], "(그림2,첫30)", res["dist_fig2_to_first30"])

# 표 6: 오픈소스 모델 400 trace
t6 = {"Qwen|ChatDev": [35, 4, 96, 1, 94, 2, 1, 9, 0, 2, 20, 1, 16, 12], "Qwen|MetaGPT": [12, 1, 35, 0, 3, 0, 4, 0, 0, 12, 16, 47, 51, 32],
      "CodeLlama|ChatDev": [76, 45, 97, 46, 97, 50, 16, 76, 2, 42, 93, 25, 67, 69], "CodeLlama|MetaGPT": [94, 12, 99, 23, 76, 9, 15, 57, 0, 40, 18, 26, 55, 56]}
res["table6"] = {}
for key, paper in t6.items():
    llm, mas = key.split("|")
    S = np.array([v for x, v in valid if x["llm_name"] == llm and x["mas_name"] == mas and x["benchmark_name"] == "ProgramDev-v2"])
    mine = S.sum(0).tolist()
    res["table6"][key] = {"n": len(S), "paper": paper, "recomputed": mine, "equal": paper == mine, "diff": [a - b for a, b in zip(mine, paper)]}
    print("표 6", key, len(S), "일치" if paper == mine else ("차이", res["table6"][key]["diff"]))

# 그림 8·9 / 본문 "39%", "60–68%", "1.56x"
def fc_v2(mas, llm):
    S = np.array([v for x, v in valid if x["mas_name"] == mas and x["llm_name"] == llm and x["benchmark_name"] == "ProgramDev-v2"])
    return {k: int(S[:, [MODES.index(m) for m in ms]].sum()) for k, ms in FC.items()}, S.sum(0).tolist(), len(S)


res["fig8_9_v2"] = {}
for name, (mas, llm) in {"MetaGPT_GPT4o": ("MetaGPT", "GPT-4o"), "MetaGPT_Claude": ("MetaGPT", "Claude"), "ChatDev_GPT4o": ("ChatDev", "GPT-4o")}.items():
    res["fig8_9_v2"][name] = fc_v2(mas, llm)
    print("ProgramDev-v2", name, res["fig8_9_v2"][name])
# 논문 그림 8·9의 범주 막대 값(그림에서 읽음): MetaGPT-GPT4o 38/47/77, MetaGPT-Claude 63/58/83, ChatDev-GPT4o 118/116/30
paper_fig = {"MetaGPT_GPT4o": [38, 47, 77], "MetaGPT_Claude": [63, 58, 83], "ChatDev_GPT4o": [118, 116, 30]}
res["fig8_9_paper_category_bars"] = paper_fig
a = res["fig8_9_v2"]["MetaGPT_GPT4o"][0]; b = res["fig8_9_v2"]["MetaGPT_Claude"][0]; c = res["fig8_9_v2"]["ChatDev_GPT4o"][0]
pa, pb, pc = paper_fig["MetaGPT_GPT4o"], paper_fig["MetaGPT_Claude"], paper_fig["ChatDev_GPT4o"]
res["claims_recomputed"] = {
    "paper_bars: GPT4o vs Claude FC1 reduction % (text 39)": 100 * (1 - pa[0] / pb[0]),
    "paper_bars: MetaGPT vs ChatDev FC1 reduction % (text 60-68)": 100 * (1 - pa[0] / pc[0]),
    "paper_bars: MetaGPT vs ChatDev FC2 reduction % (text 60-68)": 100 * (1 - pa[1] / pc[1]),
    "paper_bars: MetaGPT FC3 / ChatDev FC3 (text 1.56x more)": pa[2] / pc[2],
    "data: GPT4o vs Claude FC1 reduction %": 100 * (1 - a["FC1"] / b["FC1"]),
    "data: MetaGPT vs ChatDev FC1 reduction %": 100 * (1 - a["FC1"] / c["FC1"]),
    "data: MetaGPT vs ChatDev FC2 reduction %": 100 * (1 - a["FC2"] / c["FC2"]),
    "data: MetaGPT FC3 / ChatDev FC3": a["FC3"] / c["FC3"],
}
for k, v in res["claims_recomputed"].items():
    print(k, round(v, 2))
# 모드 단위로 그림 8·9 값과 공개 라벨을 나란히(그림 막대에서 읽은 값, 읽기 어려운 칸은 None)
res["mode_gpt4o_metagpt_data"] = res["fig8_9_v2"]["MetaGPT_GPT4o"][1]
res["mode_chatdev_data"] = res["fig8_9_v2"]["ChatDev_GPT4o"][1]
res["mode_claude_data"] = res["fig8_9_v2"]["MetaGPT_Claude"][1]

# 표 8: AG2 벤치마크별 범주 비율 (trace당 flag 수)
res["table8"] = {}
for bench in ["GSM", "MMLU", "Olympiad"]:
    S = np.array([v for x, v in valid if x["mas_name"] == "AG2" and x["benchmark_name"] == bench and x["llm_name"] in ("GPT-4o", "GPT-4o-mini")])
    llms = sorted(set(x["llm_name"] for x, v in valid if x["mas_name"] == "AG2" and x["benchmark_name"] == bench and x["llm_name"] in ("GPT-4o", "GPT-4o-mini")))
    res["table8"][bench] = {"n": len(S), "llm": llms, **{k: float(S[:, [MODES.index(m) for m in ms]].sum() / len(S)) for k, ms in FC.items()}}
    print("표 8", bench, res["table8"][bench], "(논문: GSM .53/1.33/.37, MMLU 1.06/1.01/.60, Olympiad 1.19/1.21/.67)")

# 그림 7: 14x14 phi 상관, 그림 6: 범주 상관
phi = np.corrcoef(V.T)
off = phi[np.triu_indices(14, 1)]
i, j = np.unravel_index(np.argmax(np.where(np.eye(14, dtype=bool), -9, phi)), phi.shape)
res["phi_max_offdiag"] = [MODES[i], MODES[j], float(phi[i, j])]
res["phi_min_offdiag"] = float(off.min())
res["phi_pairs_paper_check"] = {"1.4-2.1 (paper 0.63)": float(phi[3, 5]), "1.3-1.5 (paper 0.55)": float(phi[2, 4]), "2.2-2.3 (paper 0.52)": float(phi[6, 7])}
C = np.array([V[:, [MODES.index(m) for m in ms]].sum(1) for ms in FC.values()]).T
cc = np.corrcoef(C.T)
C2 = (C > 0).astype(int)
cc2 = np.corrcoef(C2.T)
res["category_corr_counts"] = cc.round(3).tolist()
res["category_corr_any"] = cc2.round(3).tolist()
print("phi max", res["phi_max_offdiag"], res["phi_pairs_paper_check"], "범주 상관(건수)", cc.round(2).tolist(), "범주 상관(유무)", cc2.round(2).tolist(), "(논문 0.32/0.17/0.28)")

# 시스템별 모드 비율과 비율 차이 검정 예: AppWorld의 3.1 (논문 5.1절)
def rate(mas, llm=None, bench=None, mode="3.1"):
    S = [v for x, v in valid if x["mas_name"] == mas and (llm is None or x["llm_name"] == llm) and (bench is None or x["benchmark_name"] == bench)]
    return int(sum(s[MODES.index(mode)] for s in S)), len(S)

res["system_rates_first30"] = {}
for mas, llm in [("AppWorld", "GPT-4o"), ("OpenManus", "GPT-4o"), ("HyperAgent", "Claude")]:
    res["system_rates_first30"][mas] = {m: rate(mas, llm, mode=m) for m in MODES}
# 실패 모드 유병률의 Wilson 구간
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]

res["mode_trace_rate_wilson"] = {m: wilson(int(V[:, i].sum()), len(V)) for i, m in enumerate(MODES)}

# 구성 단위 rate(시스템별 분포) 저장
per_cfg = {}
for k in sorted(cfg):
    S = np.array([v for x, v in valid if (x["mas_name"], x["benchmark_name"], x["llm_name"]) == k])
    per_cfg["|".join(k)] = {"n": len(S), "mean_flags": float(S.sum(1).mean()), "rate": {m: float(S[:, i].mean()) for i, m in enumerate(MODES)}}
res["per_config"] = per_cfg
json.dump(res, open(os.path.join(HERE, "mad_results.json"), "w"), ensure_ascii=False, indent=1)
