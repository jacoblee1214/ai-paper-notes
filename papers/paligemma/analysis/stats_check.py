"""PaliGemma 표 1·2·4와 절제 그림의 통계 검산 (numpy, scipy만 쓴다, CPU 수 초).

입력: paper_tables.py(표를 옮겨 적은 값), figure_values.json(figure_extract.py가 PDF 벡터에서 뽑은 값)
출력: stats_results.json

S1 표 1: 224 대 448 차이를 '재실행 표준편차'와 비교한다 (같은 최적 설정으로 5번 다시 전이한 값의 표준편차).
S2 평가 세트 크기로 본 표본 오차: 크기를 아는 과제(논문 또는 우리가 확인한 것 + 외부 지식 근사)에 Wilson 구간,
   짝지은 비교의 MDE(≈2.80·√(불일치율/n)), 관측 차이의 z 범위. 크기는 논문에 거의 없다 → 아래 TEST_N 구분 참고.
S3 MMVP 정수 격자 확인과 GPT-4V·Gemini·우연 수준과의 비교.
S4 표 2·4 재계산 (평균 행, 후회 일치).
S5 그림 17(마스크 절제 과제별)에서 그림 5 평균을 다시 만들고, 과제 수준 부트스트랩과 캡션 과제 기여를 본다.
S6 그림 11의 표준편차 분포로 '36개 과제 평균'의 재실행 잡음을 어림한다.
S7 그림 9의 해상도 대 시퀀스 길이 분해, 그림 10·7·8 값 정리.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import paper_tables as T

HERE = Path(__file__).resolve().parent
FIG = json.loads((HERE / "figure_values.json").read_text(encoding="utf-8"))
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- S1
rows = {r[1]: r for r in T.TABLE1}
s1 = []
FLOOR = 0.05  # 표시 자리(소수 첫째 자리) 반올림 절반. 표준편차가 0.0으로 적힌 칸의 하한
for sec, name, zs, a, b, c in T.TABLE1:
    if a is None or b is None:
        continue
    d = b[0] - a[0]
    if a[1] is None or b[1] is None:
        s1.append({"task": name, "delta_448_224": round(d, 2), "z_rerun": None, "note": "표준편차 없음(MMVP/POPE/Objaverse)"})
        continue
    sd = math.sqrt(max(a[1], FLOOR) ** 2 + max(b[1], FLOOR) ** 2)
    s1.append({"task": name, "zero_shot": zs, "v224": a[0], "sd224": a[1], "v448": b[0], "sd448": b[1],
               "delta_448_224": round(d, 2), "sd_diff_single_run": round(sd, 3), "z_rerun": round(d / sd, 1),
               "sensitive": name in T.RES_SENSITIVE})
with_sd = [r for r in s1 if r["z_rerun"] is not None]
cnt = lambda f: sum(1 for r in with_sd if f(r))
res["S1_table1_res_delta"] = {
    "rows": s1,
    "n_tasks_with_both_resolutions": len(s1),
    "n_with_sd": len(with_sd),
    "n_cells_with_sd_224": sum(1 for r in T.TABLE1 if r[3] and r[3][1] is not None),
    "n_cells_with_sd_448": sum(1 for r in T.TABLE1 if r[4] and r[4][1] is not None),
    "n_gain_gt3sd": cnt(lambda r: r["z_rerun"] > 3),
    "n_loss_gt3sd": cnt(lambda r: r["z_rerun"] < -3),
    "n_within_2sd": cnt(lambda r: abs(r["z_rerun"]) <= 2),
    "tasks_within_2sd": [r["task"] for r in with_sd if abs(r["z_rerun"]) <= 2],
    "tasks_loss_gt3sd": [(r["task"], r["delta_448_224"]) for r in with_sd if r["z_rerun"] < -3],
    "mean_gain_sensitive9": round(float(np.mean([r["delta_448_224"] for r in with_sd if r["sensitive"]])), 2),
    "mean_gain_other": round(float(np.mean([r["delta_448_224"] for r in with_sd if not r["sensitive"]])), 2),
    "sd_quantiles_all_cells": [round(float(q), 2) for q in np.quantile(
        [x[1] for r in T.TABLE1 for x in (r[3], r[4]) if x and x[1] is not None], [0.5, 0.75, 0.9, 1.0])],
    "note": "z_rerun = Δ/√(sd224²+sd448²), 표준편차는 '단일 실행'의 변동. 시험 세트 표본 오차는 포함하지 않음(S2).",
}

# 896px (표준편차 없음): 448 → 896 차이
g896 = [(r[1], r[5][0] - r[4][0]) for r in T.TABLE1 if r[5] and r[4]]
res["S1_table1_448_to_896"] = {"deltas": g896, "min": round(min(d for _, d in g896), 2), "max": round(max(d for _, d in g896), 2),
                               "note": "896px 칸은 표준편차가 없다(재실행 1번도 보고 안 함). 차이를 448 열의 재실행 SD(0.1–0.5)와만 견줄 수 있다."}

# ---------------------------------------------------------------- S2
TEST_N = {**{k: (v, "논문/확인") for k, v in T.TEST_N_IN_PAPER_OR_CHECKED.items()},
          **{k: (v, "외부 지식(미확인)") for k, v in T.TEST_N_RECALLED_UNVERIFIED.items()}}
s2 = []
for task, (n, src) in TEST_N.items():
    if task == "MMVP (pairs)":
        continue
    key = {"POPE": "POPE"}.get(task, task)
    r = rows.get(key)
    if not r:
        continue
    p1, p2 = r[3][0] / 100, r[4][0] / 100
    k1, k2 = round(p1 * n), round(p2 * n)
    lo1, hi1 = wilson(k1, n)
    lo2, hi2 = wilson(k2, n)
    dlt = p2 - p1
    # 짝지은 비교: 불일치율 d는 |Δ| 이상 (p1(1-p2)+p2(1-p1)) 이하
    dmin = abs(dlt)
    dmax = min(1.0, p1 * (1 - p2) + p2 * (1 - p1))
    z_hi = abs(dlt) / math.sqrt(dmin / n) if dmin > 0 else None
    z_lo = abs(dlt) / math.sqrt(dmax / n)
    s2.append({"task": task, "n": n, "n_source": src, "p224": p1, "p448": p2,
               "wilson224": [round(lo1 * 100, 1), round(hi1 * 100, 1)], "wilson448": [round(lo2 * 100, 1), round(hi2 * 100, 1)],
               "delta_pts": round(dlt * 100, 2),
               "z_paired_range": [round(z_lo, 1), round(z_hi, 1)],
               "mde_pts_at_discord_0.1": round(2.80 * math.sqrt(0.1 / n) * 100, 2),
               "mde_pts_at_discord_0.2": round(2.80 * math.sqrt(0.2 / n) * 100, 2),
               "mde_pts_at_discord_0.3": round(2.80 * math.sqrt(0.3 / n) * 100, 2),
               "rerun_sd_pts_224_448": [rows[key][3][1], rows[key][4][1]]})
res["S2_sampling_error"] = {"rows": s2,
                            "note": "ANLS·VQA 소프트 정확도는 문항 점수가 [0,1] 연속값이라 이항 분산이 상한이다. 크기 열의 '외부 지식(미확인)'은 공식 페이지로 확인하지 않은 근사값."}

# CountBenchQA 정수 격자 확인 (n=540이면 81.9, 83.1이 가능한가)
def grid_ok(pct, n):
    return [k for k in range(n + 1) if round(k / n * 100 + 1e-9, 1) == pct or abs(k / n * 100 - pct) < 0.05 - 1e-9]
res["S2_countbench_grid"] = {f"n={n}": {"81.9": grid_ok(81.9, n), "83.1": grid_ok(83.1, n)} for n in (490, 540)}

# ---------------------------------------------------------------- S3 MMVP
n = 150
mm = {"PaliGemma224": 47.3, "PaliGemma448": 45.3, "GPT-4V": 38.7, "Gemini": 40.7}
grid = {k: [kk for kk in range(n + 1) if abs(kk / n * 100 - v) < 0.05] for k, v in mm.items()}
pal, gpt, gem = 71, 58, 61
res["S3_mmvp"] = {
    "pairs": n, "chance_pct": 25.0, "grid_pairs": grid,
    "paligemma_vs_chance_binom_p": float(stats.binomtest(pal, n, 0.25, alternative="greater").pvalue),
    "wilson_paligemma224": [round(x * 100, 1) for x in wilson(pal, n)],
    "wilson_paligemma448": [round(x * 100, 1) for x in wilson(68, n)],
    "fisher_vs_gemini_p_two_sided": float(stats.fisher_exact([[pal, n - pal], [gem, n - gem]])[1]),
    "fisher_vs_gpt4v_p_two_sided": float(stats.fisher_exact([[pal, n - pal], [gpt, n - gpt]])[1]),
    "delta_224_448_pairs": 3,
    "mde_pts_discord_0.3": round(2.80 * math.sqrt(0.3 / n) * 100, 1),
    "note": "Fisher는 쌍 단위 비짝지은 검정이라 같은 문항 짝을 쓰는 McNemar보다 보수적(문항별 결과가 없어 못 함). 모델마다 전이 방식이 달라 비교 조건도 같지 않다(PaliGemma는 VQAv2 미세조정 후 제로샷).",
}

# ---------------------------------------------------------------- S4
cols = np.array([r[1:] for r in T.TABLE4], dtype=object)
names = [r[0] for r in T.TABLE4]
vals = np.array([r[1:] for r in T.TABLE4], dtype=float)
means = vals.mean(axis=0)
regret = lambda a, b: (vals[:, b] - vals[:, a]) / vals[:, a] * 100  # b 열이 a 열 대비 몇 %
res["S4_table4"] = {
    "mean_columns_recomputed": [round(float(x), 1) for x in means],
    "mean_columns_paper": list(T.TABLE4_AVG),
    "relative_of_means_single_to_prefix_pct": round(float((means[2] - means[1]) / means[1] * 100), 2),
    "relative_of_means_prefix_to_noprefix_pct": round(float((means[3] - means[2]) / means[2] * 100), 2),
    "mean_of_rowwise_regret_prefix_vs_single_pct": round(float(regret(1, 2).mean()), 2),
    "mean_of_rowwise_regret_noprefix_vs_prefix_pct": round(float(regret(2, 3).mean()), 2),
    "mean_of_rowwise_regret_noprefix_vs_single_pct": round(float(regret(1, 3).mean()), 2),
    "paper_printed_regrets_pct": [-2.0, -3.5],
    "table1_vs_single_simple_regret_pct": {nm: round(float((vals[i, 0] - vals[i, 1]) / vals[i, 0] * 100), 1)
                                           for i, nm in enumerate(names) if nm in ("SciCap", "RefCOCO (val)", "RefCOCO+ (val)", "RefCOCOg (val)")},
    "table2_printed_regret_pct": T.TABLE2_TASKS,
    "avg_without_scicap_columns": [round(float(np.delete(vals, names.index("SciCap"), axis=0).mean(axis=0)[i]), 1) for i in range(4)],
    "n_rows": len(names),
}
res["S4_table2"] = {"counts": T.TABLE2, "total_tasks": sum(T.TABLE2.values()),
                    "n_tasks_over_2.5pct": sum(v for k, v in T.TABLE2.items() if not k.startswith("[None")),
                    "note": "합 43. 그림 5는 36개, 그림 7은 38개, 그림 8은 37개 과제 평균, 표 1은 48행이다. 그림마다 과제 집합이 다르다."}

# ---------------------------------------------------------------- S5 그림 17
f17 = FIG["fig17_per_task"]
keys = ["AR_suf_loss_suf", "AR_pref_loss_suf", "AR_pref_loss_pref", "AR_all_loss_suf", "AR_all_loss_pref"]
M = np.array([[t[k] for k in keys] for t in f17])
avg = M.mean(axis=0)
base = M[:, 0]
caption_tasks = {"COCOcap", "NoCaps", "Screen2Words", "TextCaps", "SciCap", "WidgetCap", "COCO-35L (en)", "COCO-35L (avg34)",
                 "XM3600 (en)", "XM3600 (avg35)", "ActivityNet-CAP", "MSRVTT-CAP", "VATEX"}
is_cap = np.array([t["task"] in caption_tasks for t in f17])
rng = np.random.default_rng(0)
boot = {}
for j in range(1, 5):
    d = M[:, j] - base
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    boot[keys[j]] = {
        "mean_delta": round(float(d.mean()), 2),
        "boot95": [round(float(np.quantile(bs, 0.025)), 2), round(float(np.quantile(bs, 0.975)), 2)],
        "n_tasks_worse_gt1": int((d < -1).sum()), "n_tasks_better_gt1": int((d > 1).sum()), "n_tasks_within_1": int((abs(d) <= 1).sum()),
        "sign_test_p": float(stats.binomtest(int((d < 0).sum()), int((d != 0).sum()), 0.5).pvalue),
        "mean_delta_caption_tasks": round(float(d[is_cap].mean()), 2), "mean_delta_noncaption_tasks": round(float(d[~is_cap].mean()), 2),
        "mean_delta_without_caption_excl": round(float(d[~is_cap].mean()), 2),
        "relative_mean_delta_pct": round(float(np.mean(d / base) * 100), 2),
        "worst5": sorted([(f17[i]["task"], round(float(d[i]), 1)) for i in range(len(d))], key=lambda t: t[1])[:5],
    }
res["S5_fig17"] = {
    "n_tasks": len(f17), "mean_per_variant_recomputed": [round(float(x), 2) for x in avg],
    "fig5_paper_bars": [FIG["fig5_avg36"][k] for k in ("AR_suffix_only(prefix-LM)__loss_suffix", "AR_on_prefix__loss_suffix",
                        "AR_on_prefix__loss_suffix+prefix", "AR_on_image+prefix__loss_suffix", "AR_on_image+prefix__loss_suffix+prefix")],
    "diff_recomputed_minus_fig5": [round(float(a - b), 2) for a, b in zip(avg, [FIG["fig5_avg36"][k] for k in (
        "AR_suffix_only(prefix-LM)__loss_suffix", "AR_on_prefix__loss_suffix", "AR_on_prefix__loss_suffix+prefix",
        "AR_on_image+prefix__loss_suffix", "AR_on_image+prefix__loss_suffix+prefix")])],
    "bootstrap_over_tasks": boot, "n_caption_tasks": int(is_cap.sum()),
    "note": "과제는 서로 다른 지표(CIDEr 0–180, 정확도 0–100)를 섞어 평균한다. 캡션 과제 기여를 따로 적었다.",
}

# ---------------------------------------------------------------- S6 그림 11
w = 2.0 / 24
def hist_noise(h, n_tasks=36):
    counts = {int(k): v for k, v in h["counts_by_bin"].items()}
    tot = sum(counts.values())
    sds = np.array([ (i + 1) * w for i, c in counts.items() for _ in range(c)])      # 칸의 위쪽 끝(보수적)
    mids = np.array([ (i + 0.5) * w for i, c in counts.items() for _ in range(c)])
    below = sum(c for i, c in counts.items() if (i + 1) * w <= 0.5 + 1e-9)
    return {"n_entries": tot, "share_sd_le_0.5": round(below / tot, 2), "median_sd_mid": round(float(np.median(mids)), 2),
            "rms_sd_mid": round(float(np.sqrt((mids ** 2).mean())), 2),
            "sd_of_mean36_independent_upper": round(float(np.sqrt((sds ** 2).mean()) / math.sqrt(n_tasks)), 3),
            "sd_of_mean36_independent_mid": round(float(np.sqrt((mids ** 2).mean()) / math.sqrt(n_tasks)), 3)}
res["S6_fig11"] = {k: hist_noise(v) for k, v in FIG["fig11_sd_hist"].items()}
res["S6_fig11"]["note"] = ("그림 11 막대 합이 표 1의 SD 있는 과제 수(45)와 다르다. 집계 단위 불명. 평균의 SD는 과제 간 독립 가정이고 "
                          "사전학습 반복 3번의 SD이므로, 절제 한 칸(사전학습 1번)의 잡음에 대한 근사일 뿐이다.")

# ---------------------------------------------------------------- S7
f9 = FIG["fig9_task_score_avg"]["resolution_sensitive_9"]
tot = f9["448px"] - f9["224px"]
cap = f9["448px_resized_to_224_first"] - f9["224px"]
res["S7_fig9"] = {"sensitive9": f9, "insensitive11": FIG["fig9_task_score_avg"]["resolution_insensitive_11"],
                  "total_gain": round(tot, 1), "sequence_length_part": round(cap, 1), "information_part": round(tot - cap, 1),
                  "sequence_length_share": round(cap / tot, 2),
                  "table1_mean_224_sensitive9": round(float(np.mean([rows[n][3][0] for n in T.RES_SENSITIVE])), 1),
                  "table1_mean_448_sensitive9": round(float(np.mean([rows[n][4][0] for n in T.RES_SENSITIVE])), 1),
                  "note": "민감 9과제는 표 1의 224→448 이득이 큰 과제에서 고른 것이라 총 이득은 선택 효과를 받는다. 분해(길이 대 정보)는 별도 실행이라 선택 영향이 작다."}
res["S7_fig10"] = FIG["fig10_9task_avg_score_vs_224"]
res["S7_fig7"] = FIG["fig7_avg38_diff_to_TT"]
res["S7_fig8"] = FIG["fig8_avg37"]
res["S7_fig4"] = FIG["fig4_median_regret_pct"]
res["S7_fig5"] = FIG["fig5_avg36"]

(HERE / "stats_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

# ---- 요약 출력
s1r = res["S1_table1_res_delta"]
print("S1 표 1: 두 해상도 모두 있는 과제", s1r["n_tasks_with_both_resolutions"], "중 SD 있음", s1r["n_with_sd"],
      "| 이득>3SD", s1r["n_gain_gt3sd"], "손실>3SD", s1r["n_loss_gt3sd"], "|z|<=2", s1r["n_within_2sd"])
print("   |z|<=2 과제:", s1r["tasks_within_2sd"])
print("   손실>3SD:", s1r["tasks_loss_gt3sd"], "| SD 셀 수 224/448:", s1r["n_cells_with_sd_224"], s1r["n_cells_with_sd_448"])
print("   평균 이득 민감9 / 나머지:", s1r["mean_gain_sensitive9"], s1r["mean_gain_other"], "| SD 분위수(.5,.75,.9,max):", s1r["sd_quantiles_all_cells"])
for r in res["S2_sampling_error"]["rows"]:
    print("S2", r["task"], "n", r["n"], "Δ", r["delta_pts"], "z범위", r["z_paired_range"], "MDE(d=.2)", r["mde_pts_at_discord_0.2"], "재실행SD", r["rerun_sd_pts_224_448"])
print("S2 CountBench 격자:", res["S2_countbench_grid"])
print("S3 MMVP:", {k: v for k, v in res["S3_mmvp"].items() if k != "note"})
print("S4 표4 평균 재계산:", res["S4_table4"]["mean_columns_recomputed"], "논문", res["S4_table4"]["mean_columns_paper"])
print("   행별 후회 평균:", res["S4_table4"]["mean_of_rowwise_regret_prefix_vs_single_pct"], res["S4_table4"]["mean_of_rowwise_regret_noprefix_vs_prefix_pct"],
      res["S4_table4"]["mean_of_rowwise_regret_noprefix_vs_single_pct"], "| 표1 대 단일 단순:", res["S4_table4"]["table1_vs_single_simple_regret_pct"])
print("   SciCap 뺀 평균:", res["S4_table4"]["avg_without_scicap_columns"])
print("S5 그림17 평균 재계산:", res["S5_fig17"]["mean_per_variant_recomputed"], "그림5:", res["S5_fig17"]["fig5_paper_bars"])
for k, v in res["S5_fig17"]["bootstrap_over_tasks"].items():
    print("  ", k, v)
print("S6", res["S6_fig11"])
print("S7 그림9:", res["S7_fig9"])
