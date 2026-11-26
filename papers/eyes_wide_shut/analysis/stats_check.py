"""MMVP·MMVP-VLM·MoF 수치의 통계 검산 (numpy, scipy만 쓴다, CPU 수 초).

입력: paper_tables.py(표를 옮겨 적은 값), figure_values.json(figure_extract.py가 PDF 벡터에서 뽑은 그림 6 값)
출력: stats_results.json

S1 MMVP 점수의 정수 격자 확인: 보고값이 k/150(쌍)에 맞는지, 안 맞는 값은 어느 것인지.
S2 그림 4: Wilson 95% 구간(150쌍), 우연 수준(25%)과의 정확 이항 검정, 모델 쌍별 Fisher 검정, 짝지은 MDE.
S3 표 1(MMVP-VLM): 평균 재계산, Wilson(15쌍·135쌍), 해상도·규모 비교, IN-1k와의 상관, "7 of 9 패턴" 주장 점검.
S4 그림 6·표 5: 패턴별 상관 재계산(피어슨 r, Fisher z 구간, 순열 p, 스피어만), 표 5와 대조, CLIP 정의 복원,
   MLLM끼리의 상관(패턴 난이도 공통 요인), 패턴별 칸 크기 추정.
S5 표 2·3·6·7 MoF: 괄호 증분 산술, MMVP 차이의 짝지은 검정(불일치율 가정), 단조성, LLaVA 벤치마크 하락의 표본 오차 가정,
   부록 E.2 보조 벤치마크 차이.
S6 사람 성능(95.7%)과 쌍 단위 환산.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_tables as T  # noqa: E402

FIG = json.loads((HERE / "figure_values.json").read_text(encoding="utf-8"))
res = {}
N = T.N_PAIRS


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mde(n, psi, z_a=1.96, z_b=0.84):
    """짝지은 비교의 최소 검출 차이(비율). 불일치율 psi. (z_a+z_b)=2.80."""
    return (z_a + z_b) * math.sqrt(psi / n)


def mcnemar_exact(b, c):
    """불일치 b(앞이 맞고 뒤가 틀림), c. 양측 정확 이항."""
    m = b + c
    return 1.0 if m == 0 else float(stats.binomtest(min(b, c), m, 0.5).pvalue)


def delta_p(delta_pct, psi, n=N):
    """MMVP 점수 차이(%p)를 n쌍에서 불일치율 psi(비율)로 짝지은 검정할 때의 정확 p.
    순증가 쌍 = round(delta*n), 불일치 쌍 = round(psi*n) (순증가보다 작을 수 없다)."""
    net = round(abs(delta_pct) / 100 * n)
    m = max(round(psi * n), net)
    if (m - net) % 2:
        m += 1
    c = (m - net) // 2
    b = c + net
    return mcnemar_exact(b, c), m


# ------------------------------------------------------------------ S1
grid = {}
all_scores = dict(T.FIG4)
grid_rows = []
vals = [("Fig4 " + k, v) for k, v in T.FIG4.items() if k not in ("Human", "Random Guess")]
vals += [(f"Table2 ratio={r}", m) for r, m, _ in T.TABLE2]
vals += [(f"Table3 {a} {b}px", m) for a, b, _, m, *_ in T.TABLE3]
vals += [(f"Table6 {n}", m) for n, m, _ in T.TABLE6]
vals += [(f"Table7 {a} {b}px", row[0]) for a, b, _, row in T.TABLE7]
vals += [("B3 swap_options", T.B3["swap_options"]), ("B3 notation_1_2", T.B3["notation_1_2"])]
for name, v in vals:
    k = round(v / 100 * N)
    on = abs(round(100 * k / N, 1) - v) < 1e-9
    grid_rows.append({"cell": name, "value": v, "nearest_pairs": k, "k_over_150_rounded": round(100 * k / N, 1), "on_grid": on,
                      "pairs_if_exact": round(v / 100 * N, 2)})
res["S1_grid"] = {"rows": grid_rows, "off_grid": [r["cell"] + f" {r['value']} (k/150: {r['k_over_150_rounded']})" for r in grid_rows if not r["on_grid"]]}
dedupe = {}
for r in grid_rows:
    dedupe[(r["cell"].split(" ")[0] if False else r["cell"])] = r

# ------------------------------------------------------------------ S2
s2 = []
for name, v in T.FIG4.items():
    if name in ("Human", "Random Guess"):
        continue
    k = round(v / 100 * N)
    lo, hi = wilson(k, N)
    pv = float(stats.binomtest(k, N, 0.25).pvalue)
    s2.append({"model": name, "pairs": k, "score": v, "wilson95": [round(lo, 1), round(hi, 1)], "p_vs_25pct_exact": round(pv, 4)})
res["S2_fig4_intervals"] = s2
cmp = []
for a, b in [("Gemini", "GPT-4V"), ("GPT-4V", "LLaVA-1.5"), ("Gemini", "LLaVA-1.5"), ("GPT-4V", "Bard"), ("LLaVA-1.5", "InstructBLIP"), ("InstructBLIP", "mini-GPT4"), ("Bing Chat", "InstructBLIP")]:
    ka, kb = round(T.FIG4[a] / 100 * N), round(T.FIG4[b] / 100 * N)
    p = float(stats.fisher_exact([[ka, N - ka], [kb, N - kb]])[1])
    cmp.append({"a": a, "b": b, "diff_pts": round(T.FIG4[a] - T.FIG4[b], 1), "fisher_p_unpaired": round(p, 4)})
# PaliGemma(paligemma/ 03 §3.4: 71/150, 68/150)와 Gemini·GPT-4V
for pg, kp in (("PaliGemma 448", 71), ("PaliGemma 224", 68)):
    for other in ("Gemini", "GPT-4V"):
        ko = round(T.FIG4[other] / 100 * N)
        cmp.append({"a": pg, "b": other, "diff_pts": round(100 * (kp - ko) / N, 1), "fisher_p_unpaired": round(float(stats.fisher_exact([[kp, N - kp], [ko, N - ko]])[1]), 4)})
res["S2_pairwise_unpaired"] = cmp
res["S2_mde_paired_pts"] = {f"psi={psi}": round(100 * mde(N, psi), 1) for psi in (0.1, 0.2, 0.3, 0.5)}
res["S2_mde_unpaired_pts_at_p0.4"] = round(100 * 2.80 * math.sqrt(2 * 0.4 * 0.6 / N), 1)
res["S2_chance_sd_pts"] = round(100 * math.sqrt(0.25 * 0.75 / N), 2)

# ------------------------------------------------------------------ S3
t1 = []
for name, res_, par, in1k, vals9, avg in T.TABLE1:
    mean = sum(vals9) / 9
    k = round(sum(v / 100 * 15 for v in vals9))
    lo, hi = wilson(k, 135)
    t1.append({"model": f"{name} {res_}", "avg_table": avg, "avg_recomputed": round(mean, 2), "pairs_of_135": k, "wilson95_135": [round(lo, 1), round(hi, 1)],
               "in1k": in1k, "grid_ok": all(abs(round(v / 100 * 15) * 100 / 15 - v) < 0.06 for v in vals9)})
res["S3_table1_rows"] = t1
res["S3_per_cell_wilson_15"] = {f"{k}/15": [round(x, 1) for x in wilson(k, 15)] for k in (0, 1, 2, 3, 4, 5, 8, 10, 12)}

rows = {f"{r[0]} {r[1]}": r for r in T.TABLE1}
pairs_res = []
for a, b in [("OpenAI ViT-L-14 224", "OpenAI ViT-L-14 336"), ("SigLIP ViT-SO-14 224", "SigLIP ViT-SO-14 384"), ("DFN ViT-H-14 224", "DFN ViT-H-14 378"),
             ("MetaCLIP ViT-L-14 224", "MetaCLIP ViT-H-14 224"), ("EVA01 ViT-g-14 224", "EVA02 ViT-bigE-14+ 224"),
             ("OpenAI ViT-L-14 224", "SigLIP ViT-SO-14 224"), ("SigLIP ViT-SO-14 224", "DFN ViT-H-14 224")]:
    ka = round(sum(rows[a][4]) * 15 / 100); kb = round(sum(rows[b][4]) * 15 / 100)
    p = float(stats.fisher_exact([[ka, 135 - ka], [kb, 135 - kb]])[1])
    pairs_res.append({"a": a, "b": b, "pairs_a": ka, "pairs_b": kb, "diff_pts": round(100 * (kb - ka) / 135, 1), "fisher_p_unpaired": round(p, 4)})
res["S3_pairwise"] = pairs_res

# 패턴별 최고 점수와 우연(독립 무작위 25%) 대비
best = []
for j, pat in enumerate(T.PATTERNS):
    col = [(r[4][j], f"{r[0]} {r[1]}") for r in T.TABLE1]
    top = max(col)
    k = round(top[0] * 15 / 100)
    # 10개 모델 중 최댓값이 k 이상일 확률 (독립 무작위, 25%): 1-(P(X<k))^10
    pk = float(stats.binom.sf(k - 1, 15, 0.25))
    best.append({"pattern": pat, "best_pct": top[0], "best_model": top[1], "pairs": k, "p_single_ge": round(pk, 4), "p_max_of_10_ge": round(1 - (1 - pk) ** 10, 3),
                 "n_models_ge_50": sum(1 for v, _ in col if v >= 50), "median": float(np.median([v for v, _ in col]))})
res["S3_best_per_pattern"] = best

in1k = np.array([r[3] for r in T.TABLE1]); avg = np.array([sum(r[4]) / 9 for r in T.TABLE1])
r_all, p_all = stats.pearsonr(in1k, avg)
sp_all = stats.spearmanr(in1k, avg)
mask = in1k >= 80
r_hi, p_hi = stats.pearsonr(in1k[mask], avg[mask]) if mask.sum() > 2 else (None, None)
mask2 = in1k < 80
r_lo, p_lo = stats.pearsonr(in1k[mask2], avg[mask2])
z = np.arctanh(r_all); se = 1 / math.sqrt(len(in1k) - 3)
res["S3_in1k_vs_mmvpvlm_avg"] = {
    "n_rows": len(in1k), "pearson_all": round(float(r_all), 3), "p_all": float(f"{p_all:.2g}"),
    "ci95_all": [round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)], "spearman_all": round(float(sp_all.statistic), 3),
    "n_ge80": int(mask.sum()), "pearson_ge80": round(float(r_hi), 3), "p_ge80": round(float(p_hi), 3),
    "n_lt80": int(mask2.sum()), "pearson_lt80": round(float(r_lo), 3), "p_lt80": round(float(p_lo), 3),
    "note": "10행은 독립이 아니다(같은 체크포인트의 해상도 변형 3쌍). 구간은 행을 독립으로 본 근사."}

# ------------------------------------------------------------------ S4
order = FIG["order"]
clip = np.array(FIG["clip"])
# CLIP 정의 복원: Table 1의 OpenAI 224, OpenAI 336, EVA01 평균이 그림 6 CLIP 막대와 일치하는가
idx = [T.PATTERNS.index(p) for p in ["Color", "State", "Structural", "Viewpoint", "Presence", "Quantity", "Positional", "Text", "Orientation"]]
assert [T.PATTERNS[i] for i in idx] == ["Color", "State", "Structural", "Viewpoint", "Presence", "Quantity", "Positional", "Text", "Orientation"]
cand = {"OpenAI224+OpenAI336+EVA01": ["OpenAI ViT-L-14 224", "OpenAI ViT-L-14 336", "EVA01 ViT-g-14 224"],
        "all 10 rows": list(rows.keys()),
        "OpenAI224 only": ["OpenAI ViT-L-14 224"], "OpenAI224+OpenAI336": ["OpenAI ViT-L-14 224", "OpenAI ViT-L-14 336"]}
recon = {}
for name, ms in cand.items():
    m = np.array([np.mean([rows[x][4][j] for x in ms]) for j in idx])
    recon[name] = {"max_abs_diff_vs_fig6_clip": round(float(np.max(np.abs(m - clip))), 2)}
res["S4_clip_definition"] = recon


def corr_row(x, y):
    r, p = stats.pearsonr(x, y)
    n = len(x)
    z = np.arctanh(r); se = 1 / math.sqrt(n - 3)
    # 정확 순열 p (9! = 362,880)
    from itertools import permutations
    cnt = 0; tot = 0
    xc = x - x.mean()
    ys = y - y.mean()
    denom = math.sqrt((xc ** 2).sum() * (ys ** 2).sum())
    for perm in permutations(range(n)):
        rr = (xc * ys[list(perm)]).sum() / denom
        tot += 1
        if abs(rr) >= abs(r) - 1e-12:
            cnt += 1
    sp = stats.spearmanr(x, y)
    return {"r": round(float(r), 3), "ci95_fisher": [round(float(np.tanh(z - 1.96 * se)), 2), round(float(np.tanh(z + 1.96 * se)), 2)],
            "p_t": round(float(p), 4), "p_perm_exact": round(cnt / tot, 4), "spearman": round(float(sp.statistic), 3), "spearman_p": round(float(sp.pvalue), 3)}


s4 = {}
for name in ["InstructBLIP", "LLaVA-1.5", "Gemini", "GPT-4"]:
    y = np.array(FIG[name])
    d = corr_row(clip, y)
    d["table5_printed"] = T.TABLE5.get(name)
    s4[name] = d
res["S4_clip_vs_mllm"] = s4
# 패턴 하나 빼기 (leave-one-out) 범위
loo = {}
for name in ["InstructBLIP", "LLaVA-1.5", "Gemini", "GPT-4"]:
    y = np.array(FIG[name])
    rs = [stats.pearsonr(np.delete(clip, i), np.delete(y, i))[0] for i in range(9)]
    loo[name] = [round(float(min(rs)), 2), round(float(max(rs)), 2)]
res["S4_leave_one_pattern_out_r_range"] = loo
# 글자·방향 두 패턴(모든 모델이 낮음)을 뺀 상관
no_low = {}
keep = [i for i, p in enumerate(order) if p not in ("Text", "Orientation")]
for name in ["InstructBLIP", "LLaVA-1.5", "Gemini", "GPT-4"]:
    y = np.array(FIG[name])
    no_low[name] = round(float(stats.pearsonr(clip[keep], y[keep])[0]), 2)
res["S4_without_text_orientation_r"] = no_low
# MLLM끼리의 상관: CLIP 기반이 아닌 모델도 CLIP과 비슷하게 상관 -> 패턴 난이도 공통 요인
mm = {}
names = ["InstructBLIP", "LLaVA-1.5", "Gemini", "GPT-4"]
for i, a in enumerate(names):
    for b in names[i + 1:]:
        mm[f"{a} ~ {b}"] = round(float(stats.pearsonr(FIG[a], FIG[b])[0]), 2)
res["S4_mllm_vs_mllm_r"] = mm
# 표 1의 한 모델 행 대 MLLM (CLIP 평균 대신): 선별에 쓰인 OpenAI 224 대 LLaVA-1.5, 선별과 독립인 EVA01 대 InstructBLIP
def row_by_pattern(key):
    return np.array([rows[key][4][j] for j in idx])
one = {}
for mdl, row in [("InstructBLIP", "EVA01 ViT-g-14 224"), ("LLaVA-1.5", "OpenAI ViT-L-14 336"), ("LLaVA-1.5", "OpenAI ViT-L-14 224"), ("Gemini", "SigLIP ViT-SO-14 224"), ("GPT-4", "SigLIP ViT-SO-14 224"), ("Gemini", "EVA02 ViT-bigE-14+ 224")]:
    x = row_by_pattern(row); y = np.array(FIG[mdl])
    one[f"{mdl} ~ {row}"] = round(float(stats.pearsonr(x, y)[0]), 2)
res["S4_single_clip_row_r"] = one
# 칸 크기 추정: 패턴별로 네 MLLM 값이 모두 k/n(±0.1)로 설명되는 최소 n
cell = {}
for j, p in enumerate(order):
    vs = [FIG[m][j] for m in names]
    found = None
    for n in range(2, 41):
        if all(min(abs(100 * k / n - v) for k in range(n + 1)) < 0.12 for v in vs):
            found = n
            break
    cell[p] = {"values": vs, "smallest_common_n_le40": found}
res["S4_cell_size_inference"] = cell

# ------------------------------------------------------------------ S5
t2 = []
base = T.TABLE2[0]
for ratio, mm_, llv in T.TABLE2:
    dm = round(mm_ - base[1], 1); dl = round(llv - base[2], 1)
    row = {"ratio": ratio, "mmvp": mm_, "d_mmvp": dm, "llava": llv, "d_llava": dl, "printed": T.TABLE2_DELTA_PRINTED.get(ratio)}
    if ratio:
        row["delta_matches_print"] = (T.TABLE2_DELTA_PRINTED[ratio] == (dm, dl))
        for psi in (None, 0.2, 0.3, 0.5):
            if psi is None:
                net = round(abs(dm) / 100 * N)
                p, m = delta_p(dm, abs(dm) / 100)  # 불일치가 순증가뿐인 가장 유리한 경우
                row["p_best_case"] = round(p, 4)
            else:
                p, m = delta_p(dm, psi)
                row[f"p_psi{psi}"] = round(p, 4)
    t2.append(row)
res["S5_table2"] = t2
# 인접 비율 비교: 0.75 대 0.875 대 1.0
adj = []
for (r1, m1, _), (r2, m2, _) in [(T.TABLE2[4], T.TABLE2[5]), (T.TABLE2[4], T.TABLE2[6]), (T.TABLE2[3], T.TABLE2[4]), (T.TABLE2[2], T.TABLE2[4])]:
    p3, _ = delta_p(m2 - m1, 0.3)
    adj.append({"from_ratio": r1, "to_ratio": r2, "d_mmvp": round(m2 - m1, 1), "p_psi0.3": round(p3, 3)})
res["S5_table2_adjacent"] = adj
# 단조성 (7점): MMVP는 0.75에서 꺾임, LLaVA 벤치는 단조 감소
r_ = [r[0] for r in T.TABLE2]; mmv = [r[1] for r in T.TABLE2]; llv = [r[2] for r in T.TABLE2]
res["S5_table2_monotonic"] = {
    "spearman_ratio_vs_llava": float(stats.spearmanr(r_, llv).statistic), "exact_p_monotone_7pts": 2 / math.factorial(7),
    "spearman_ratio_vs_mmvp": round(float(stats.spearmanr(r_, mmv).statistic), 2), "spearman_p_mmvp": round(float(stats.spearmanr(r_, mmv).pvalue), 3),
    "mmvp_peak_ratio": r_[int(np.argmax(mmv))],
    "pearson_mmvp_vs_llava_across_ratios": round(float(stats.pearsonr(mmv, llv)[0]), 2)}
# LLaVA 벤치마크(지시 따르기) 하락의 표본 오차 가정: n=90 문항(통상 크기, 논문엔 없음), 문항별 상대 점수 SD s, 두 모델 문항별 점수 상관 rho
sens = {}
for s in (15, 25, 35):
    for rho in (0.3, 0.6):
        se_ = s * math.sqrt(2 * (1 - rho)) / math.sqrt(90)
        sens[f"s={s},rho={rho}"] = {"se_diff": round(se_, 2), "mde_80pct": round(2.8 * se_, 1)}
res["S5_llava_bench_assumption"] = {"n_assumed": 90, "note": "LLaVA-Bench(COCO) 통상 90문항(미확인, 논문에 없음). GPT-4 판정 상대 점수라 이항이 아니다. SD·상관은 가정.",
                                      "grid": sens, "drops": {str(r): round(llv_ - base[2], 1) for r, _, llv_ in T.TABLE2 if r}}
# 표 3
t3 = []
for name, resn, tok, mm_, lv, pope, d in T.TABLE3:
    t3.append({"method": f"{name} {resn}", "tokens": tok, "mmvp": mm_, "printed_delta": d})
# 괄호 증분이 기준으로 삼은 행
res["S5_table3_delta_base"] = {
    "LLaVA+I-MoF 224 printed +10.7": {"vs_LLaVA_336 (6.0)": round(16.7 - 6.0, 1), "vs_LLaVA_224 (5.5)": round(16.7 - 5.5, 1)},
    "LLaVA1.5+I-MoF 224 printed +3.3": {"vs_LLaVA1.5_336 (24.7)": round(28.0 - 24.7, 1), "note": "224 해상도 CLIP 단독 LLaVA-1.5 기준선은 없음"},
}
t3p = {}
for lab, d_ in [("LLaVA 224 -> +I-MoF (+11.2)", 11.2), ("LLaVA 336 -> +I-MoF 224 (+10.7)", 10.7), ("LLaVA1.5 336 -> +I-MoF 224 (+3.3)", 3.3),
                ("LLaVA1.5 336 -> +I-MoF 336 (+6.6)", 6.6), ("I-MoF 224 -> 336 (+3.3)", 3.3), ("MoCov3 vs None (+2.0)", 2.0), ("MAE vs None (+2.6)", 2.6),
                ("DINOv2 vs MAE (+0.7)", 0.7)]:
    t3p[lab] = {f"psi={psi}": round(delta_p(d_, psi)[0], 3) for psi in (0.1, 0.2, 0.3, 0.5)}
    t3p[lab]["p_best_case"] = round(delta_p(d_, d_ / 100)[0], 3)
res["S5_table3_6_7_mmvp_tests"] = t3p
# 표 7 보조 벤치마크 (표본 크기는 통상 크기, 미확인)
SIZES = {"LLVB": 90, "LLVW": 60, "MMB": 4377, "VQAT": 5000, "POPE": 9000, "VQAV2": 107394, "MM-V": 218}
t7 = []
base7 = T.TABLE7[0][3]
for name, resn, tok, vals_ in T.TABLE7[1:]:
    row = {"model": f"{name} {resn}"}
    for col, b, v in zip(T.TABLE7_COLS, base7, vals_):
        if col == "MMVP":
            continue
        n_ = SIZES[col]
        if col in ("MMB", "VQAT", "POPE", "VQAV2"):
            p0 = b / 100
            se_ = math.sqrt(2 * p0 * (1 - p0) / n_) * 100 * 0.7  # 짝지은 효과를 반영해 비짝지은 SE의 0.7배
            row[col] = {"delta": round(v - b, 1), "n_assumed": n_, "se_paired_approx": round(se_, 2), "z": round((v - b) / se_, 1)}
        else:
            row[col] = {"delta": round(v - b, 1), "n_assumed": n_, "note": "GPT-4 판정 상대 점수, 이항 아님"}
    t7.append(row)
res["S5_table7_secondary"] = {"sizes_note": "통상 크기(미확인): MMBench dev 4,377, TextVQA val 5,000, POPE 9,000, VQAv2 test-dev 107,394, MM-Vet 218, LLaVA-Bench 90/60. 논문에는 없음", "rows": t7}

# ------------------------------------------------------------------ S6
h = T.FIG4["Human"] / 100
res["S6_human"] = {
    "participants": T.HUMAN_PARTICIPANTS, "answers_if_4x300": 4 * 300,
    "implied_correct_answers": round(h * 1200, 1),
    "if_question_level_pair_level_under_independence_pct": round(100 * h * h, 1),
    "if_pair_level_question_level_at_least_pct": round(100 * math.sqrt(h), 1),
    "note": "논문은 사람 점수를 '질문의 평균'(§2.3)이라 쓰면서 쌍 단위 정의도 같은 문단에 쓴다. 두 해석의 값이 다르고 4명(자원자)의 표본이다.",
    "wilson_if_1200_answers_iid": [round(x, 1) for x in wilson(round(h * 1200), 1200)]}
res["S6_b3_ablation"] = {"swap_options_diff_pts": round(T.B3["swap_options"] - T.B3["original"], 1), "pairs_swap": round(T.B3["swap_options"] / 100 * N), "pairs_orig": round(T.B3["original"] / 100 * N),
                          "notation_diff_pts": round(T.B3["notation_1_2"] - T.B3["original"], 1), "mde_paired_psi0.2_pts": round(100 * mde(N, 0.2), 1),
                          "swap_p_psi0.2": round(delta_p(1.6, 0.2)[0], 2)}

(HERE / "stats_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("S1_grid", "S2_fig4_intervals", "S2_pairwise_unpaired", "S2_mde_paired_pts", "S3_in1k_vs_mmvpvlm_avg", "S4_clip_definition", "S4_clip_vs_mllm")}, ensure_ascii=False, indent=1))
