"""Cambrian-1 부록 표(12-16, 3, 19-20, 9, 21), 그림 3·15 값에서 03_analysis.md의 수치를 재계산한다.

입력: analysis/tables.json (parse_tables.py), analysis/fig3_values.json (fig3_extract.py)
출력: analysis/recompute_results.json (+ 표준 출력 요약)
의존: numpy, scipy (표준 python3). 수 초.
MDE = 2.80*sqrt(q/n) (쌍체 가정, q=불일치율, 기본 0.2, 점 단위 x100). 80% 검정력, 양측 5%.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
T = json.loads((ROOT / "analysis" / "tables.json").read_text())
F3 = json.loads((ROOT / "analysis" / "fig3_values.json").read_text())
CV = json.loads((ROOT / "analysis" / "cvbench_results.json").read_text())
COLS = T["cols"][1:]  # 16개 벤치마크
GROUPS = T["groups"]
RNG = np.random.default_rng(0)
Q = 0.2
OUT = {}

# 문항 수: 표 21(# images 열이 사실상 문항 수), GQA는 공식 testdev-balanced 12,578(prismatic_vlms 03과 같은 값),
# CV-Bench 2D/3D는 HF 주석에서 계산(1438/1200). MMVP는 쌍 정확도라 유효 표본이 150쌍.
N_Q = {"MMEP": 2374, "MMB": 4377, "SEEDI": 17990, "GQA": 12578, "SQAI": 2017, "MMMU": 900, "MathVista": 1000,
       "AI2D": 3088, "ChartQA": 2500, "OCRBench": 1000, "TextVQA": 5000, "DocVQA": 5188, "MMVP": 150,
       "RWQA": 765, "CV2D": 1438, "CV3D": 1200}
OUT["n_questions"] = N_Q


def to100(r, c):
    return r[c] / 20 if c == "MMEP" else r[c]


def wilson(p, n, z=1.96):
    p = p / 100
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def mde(n, q=Q):
    return 100 * 2.80 * np.sqrt(q / n)


def cat_mde(cols, q=Q):
    """4개 벤치마크 평균의 차이에 대한 MDE (벤치마크 표본은 독립, 쌍체)."""
    s = sum(q / N_Q[c] for c in cols)
    return 100 * 2.80 * np.sqrt(s) / len(cols)


def avg16_mde(q=Q):
    return cat_mde(COLS, q)


OUT["mde_per_benchmark_q0.2"] = {c: round(mde(N_Q[c]), 2) for c in COLS}
OUT["mde_category_q0.2"] = {g: round(cat_mde(cs), 2) for g, cs in GROUPS.items()}
OUT["mde_avg16_q0.1_0.2_0.3"] = [round(avg16_mde(q), 2) for q in (0.1, 0.2, 0.3)]


def mat(rows):
    return np.array([[to100(r, c) for c in COLS] for r in rows])


def catmean(rows_or_row):
    r = rows_or_row
    return {g: float(np.mean([to100(r, c) for c in cs])) for g, cs in GROUPS.items()}


# ---------------------------------------------------------------- 0. 표 전사 점검
chk = {}
for k in ["t3_ensemble", "t12_0M", "t13_0.5M", "t14_1.2M", "t15_1.2M_unfrozen", "t16_5M", "t19_balance", "t20_ratio"]:
    d = [abs(np.mean([to100(r, c) for c in COLS]) - r["avg"]) for r in T[k]]
    chk[k] = {"rows": len(T[k]), "max_abs_diff_avg_col": round(max(d), 3)}
t9 = {}
for r in T["t9_final"]:
    v = r["vals"]
    # 열: tokens, avgG, MMEP, MMB, SEED, GQA, avgK, SQA, MMMU, MathVista, AI2D, avgO, ChartQA, OCRBench, TextVQA, DocVQA,
    #     avgV, MMVP, RWQA, CV2D, CV3D  (표 9 열 순서)
    if r["name"].startswith("Cambrian") or True:
        v = v[1:] if len(v) == 21 else v
        avgG, mmep, mmb, seed, gqa, avgK, sqa, mmmu, mv, ai2d, avgO, chq, ocr, tvqa, doc, avgV, mmvp, rwqa, c2d, c3d = v
        t9[r["name"]] = {
            "avgG": (avgG, round(np.mean([mmep / 20, mmb, seed, gqa]), 2)),
            "avgK": (avgK, round(np.mean([sqa, mmmu, mv, ai2d]), 2)),
            "avgO": (avgO, round(np.mean([chq, ocr, tvqa, doc]), 2)),
            "avgV": (avgV, round(np.mean([mmvp, rwqa, c2d, c3d]), 2)),
        }
OUT["transcription_check"] = chk
OUT["t9_category_avg_check_(paper,recomputed)"] = t9
f24 = T["t24_fuzzy"]
OUT["t24_13B_fuzzy_equals_8B_fuzzy_cells"] = int(sum(a == b for a, b in zip(f24["8B_fuzzy"], f24["13B_fuzzy"])))
t9d = {r["name"]: r["vals"] for r in T["t9_final"]}
OUT["t24_13B_fuzzy_vs_t9_13B"] = {
    "t9_13B": {"SEEDI": 74.4, "GQA": 64.3, "SQAI": 79.3, "MMMU": 40.0, "AI2D": 73.6, "ChartQA": 73.8, "OCRBench": 61.9,
               "TextVQA": 72.8, "MMVP": 41.3, "RWQA": 63.0},
    "t24_13B_fuzzy": dict(zip(f24["cols"], f24["13B_fuzzy"])),
}

# ---------------------------------------------------------------- 1. 이미지 없음 비교 (그림 3 왼쪽)
fv = F3["values"]
N_FIG3 = {"SQA-I": N_Q["SQAI"], "MMMU": 900, "MathVista": 1000, "AI2D": 3088, "RealWorldQA": 765, "TextVQA": 5000,
          "ChartQA": 2500, "OCRBench": 1000, "DocVQA": 5188, "GQA": 12578, "SEED-I": 17990, "MMVP": 150,
          "MME": 2374, "MMB": 4377}
rows = []
for i, b in enumerate(F3["bench"]):
    E, D, R = fv["vision_enabled"][i], fv["vision_disabled"][i], fv["random"][i]
    R_corr = 37.5 if b == "MME" else R  # MME-P = (acc + acc+)/2 척도에서 무작위는 (50+25)/2
    n = N_FIG3[b]
    lo, hi = wilson(max(D, 0.01), n)
    gap = E - D
    rows.append({
        "bench": b, "n": n, "random_fig": R, "random_metric": R_corr, "enabled": E, "disabled": D, "gap_E_D": round(gap, 1),
        "disabled_wilson95": [round(lo, 1), round(hi, 1)],
        "disabled_minus_random_fig": round(D - R, 1), "disabled_minus_random_metric": round(D - R_corr, 1),
        "mde_single_model_q0.2": round(mde(n), 2),
        "gap_over_mde": round(gap / mde(n), 1),
        "vision_share_of_above_chance": (round(gap / (E - R_corr), 2) if E - R_corr > 0 else None),
    })
OUT["fig3"] = rows

# 모델별 상한: 이미지 없음 점수를 23개 평균(D_avg)으로 두고 인코더별 점수와 비교 (표 14 행)
mp = {"SQA-I": "SQAI", "MMMU": "MMMU", "MathVista": "MathVista", "AI2D": "AI2D", "RealWorldQA": "RWQA", "TextVQA": "TextVQA",
      "ChartQA": "ChartQA", "OCRBench": "OCRBench", "DocVQA": "DocVQA", "GQA": "GQA", "SEED-I": "SEEDI", "MMVP": "MMVP",
      "MME": "MMEP", "MMB": "MMB"}
t14 = T["t14_1.2M"]
byname14 = {r["name"]: r for r in t14}
best, worst = byname14["SigLIP ViT-SO400M/14@384"], byname14["SAM ViT-H/16@1024"]
per_model_gap = {}
for i, b in enumerate(F3["bench"]):
    D = fv["vision_disabled"][i]
    per_model_gap[b] = {"best_SigLIP_SO400M_minus_Davg": round(to100(best, mp[b]) - D, 1),
                        "worst_SAM_minus_Davg": round(to100(worst, mp[b]) - D, 1)}
OUT["fig3_per_encoder_gap_using_pooled_disabled"] = per_model_gap
# 그림 3의 '이미지 있음' 평균이 어느 표의 평균과 같은가
def tab_mean(tab):
    return {b: float(np.mean([to100(r, mp[b]) for r in T[tab]])) for b in F3["bench"]}
e_fig = np.array(fv["vision_enabled"])
OUT["fig3_enabled_vs_table_means"] = {}
for tab in ["t12_0M", "t13_0.5M", "t14_1.2M", "t15_1.2M_unfrozen"]:
    m = tab_mean(tab)
    d = np.array([m[b] for b in F3["bench"]])
    OUT["fig3_enabled_vs_table_means"][tab] = {"rows": len(T[tab]), "mae": round(float(np.abs(d - e_fig).mean()), 2),
                                               "max": round(float(np.abs(d - e_fig).max()), 2)}

# ---------------------------------------------------------------- 2. 벤치마크 군집 안정성
BENCH15 = T["fig15_names"]
LAB = {"MME": "G", "MMB": "G", "SEED-I": "G", "GQA": "G", "SQA-I": "K", "MMMU": "K", "MathVista": "K", "AI2D": "K",
       "ChartQA": "O", "OCRBench": "O", "TextVQA": "O", "DocVQA": "O", "MMVP": "V", "RW QA": "V", "CV-CB": "V"}
labels = np.array([LAB[b] for b in BENCH15])
C15 = np.array(T["fig15_corr"])


def ari(a, b):
    from itertools import combinations
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    ct = {}
    for x, y in zip(a, b):
        ct[(x, y)] = ct.get((x, y), 0) + 1
    comb2 = lambda v: v * (v - 1) / 2
    sij = sum(comb2(v) for v in ct.values())
    sa = sum(comb2(v) for v in np.unique(a, return_counts=True)[1])
    sb = sum(comb2(v) for v in np.unique(b, return_counts=True)[1])
    exp = sa * sb / comb2(n)
    mx = (sa + sb) / 2
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def cluster(C, k=4, method="average"):
    Cc = np.clip((C + C.T) / 2, -1, 1)
    D = 1 - Cc
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method=method)
    return fcluster(Z, k, criterion="maxclust")


def within_between(C):
    res = {}
    for i, b in enumerate(BENCH15):
        own = [C[i, j] for j in range(15) if j != i and labels[j] == labels[i]]
        oth = {g: np.mean([C[i, j] for j in range(15) if labels[j] == g]) for g in "GKOV" if g != labels[i]}
        res[b] = {"label": labels[i], "mean_corr_own_cluster": round(float(np.mean(own)), 2),
                  "best_other": max(oth, key=oth.get), "mean_corr_best_other": round(float(max(oth.values())), 2)}
    return res


cl15 = cluster(C15)
OUT["cluster_fig15"] = {
    "assign": dict(zip(BENCH15, map(int, cl15))),
    "ari_vs_paper_labels": round(ari(cl15, labels), 3),
    "within_between": within_between(C15),
    "offdiag_corr_min_mean": [round(float(C15[np.triu_indices(15, 1)].min()), 2),
                              round(float(C15[np.triu_indices(15, 1)].mean()), 2)],
}
bench_keys = {  # 표 컬럼 -> 그림 15 이름
    "MMEP": "MME", "MMB": "MMB", "SEEDI": "SEED-I", "GQA": "GQA", "SQAI": "SQA-I", "MMMU": "MMMU", "MathVista": "MathVista",
    "AI2D": "AI2D", "ChartQA": "ChartQA", "OCRBench": "OCRBench", "TextVQA": "TextVQA", "DocVQA": "DocVQA", "MMVP": "MMVP",
    "RWQA": "RW QA"}


def score_matrix(rows):
    X = np.array([[to100(r, c) for c in bench_keys] + [(r["CV2D"] + r["CV3D"]) / 2] for r in rows])
    return X  # 열 순서 = BENCH15 (MME,MMB,SEED,GQA,SQA,MMMU,MathVista,AI2D,ChartQA,OCRBench,TextVQA,DocVQA,MMVP,RWQA,CV)


def corr_of(X):
    Xs = X - X.mean(0)
    sd = Xs.std(0)
    sd[sd == 0] = np.nan
    Cm = (Xs / sd).T @ (Xs / sd) / len(X)
    return np.nan_to_num(Cm, nan=0.0)


sets = {
    "T14_18rows": T["t14_1.2M"],
    "T12-15_union_72rows": T["t12_0M"] + T["t13_0.5M"] + T["t14_1.2M"] + T["t15_1.2M_unfrozen"],
    "T14_without_SAM_MiDaS_SD_SupViT": [r for r in T["t14_1.2M"] if not any(s in r["name"] for s in ["SAM", "MiDaS", "Diffusion", "Supervised"])],
    "T14_language_supervised_only": [r for r in T["t14_1.2M"][:9]],
}
OUT["cluster_data"] = {}
for nm, rs in sets.items():
    X = score_matrix(rs)
    Cm = corr_of(X)
    cl = cluster(Cm)
    OUT["cluster_data"][nm] = {
        "n_models": len(rs),
        "mean_abs_diff_vs_fig15": round(float(np.abs(Cm - C15)[np.triu_indices(15, 1)].mean()), 3),
        "corr_with_fig15_offdiag": round(float(np.corrcoef(Cm[np.triu_indices(15, 1)], C15[np.triu_indices(15, 1)])[0, 1]), 3),
        "ari_vs_paper_labels": round(ari(cl, labels), 3),
        "assign": dict(zip(BENCH15, map(int, cl))),
    }
    # PCA(표준화 점수, 행=모델)
    Xz = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    s = np.linalg.svd(Xz, compute_uv=False) ** 2
    OUT["cluster_data"][nm]["pca_var_ratio_top3"] = [round(float(v), 3) for v in (s / s.sum())[:3]]


def boot(rows, B=2000):
    X = score_matrix(rows)
    n = len(X)
    aris, co = [], np.zeros((15, 15))
    same = 0
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        if len(np.unique(idx)) < 4:
            continue
        cl = cluster(corr_of(X[idx]))
        aris.append(ari(cl, labels))
        co += (cl[:, None] == cl[None, :])
        same += (ari(cl, labels) == 1.0)
    co /= len(aris)
    return aris, co, same / len(aris)


for nm in ["T14_18rows", "T12-15_union_72rows", "T14_without_SAM_MiDaS_SD_SupViT"]:
    aris, co, pexact = boot(sets[nm])
    # 라벨 안 쌍의 공동 군집 빈도와 라벨 밖 쌍의 빈도
    within = [co[i, j] for i in range(15) for j in range(i + 1, 15) if labels[i] == labels[j]]
    between = [co[i, j] for i in range(15) for j in range(i + 1, 15) if labels[i] != labels[j]]
    # 벤치마크별: 자기 라벨 군집 동료와 함께 묶이는 평균 빈도
    per = {BENCH15[i]: round(float(np.mean([co[i, j] for j in range(15) if j != i and labels[j] == labels[i]])), 2) for i in range(15)}
    OUT["cluster_data"][nm]["bootstrap"] = {
        "B": 2000, "ari_median": round(float(np.median(aris)), 3), "ari_5_95": [round(float(np.percentile(aris, 5)), 3), round(float(np.percentile(aris, 95)), 3)],
        "p_exactly_paper_partition": round(pexact, 3),
        "mean_cocluster_within_label_pairs": round(float(np.mean(within)), 2),
        "mean_cocluster_between_label_pairs": round(float(np.mean(between)), 2),
        "per_benchmark_cocluster_with_own_label": per,
    }
    # 점 하나 빼기
    X = score_matrix(sets[nm]); loo = []
    for i in range(len(X)):
        cl = cluster(corr_of(np.delete(X, i, 0)))
        loo.append(ari(cl, labels))
    OUT["cluster_data"][nm]["leave_one_model_out"] = {"ari_min": round(min(loo), 3), "ari_median": round(float(np.median(loo)), 3),
                                                      "n_exact": int(sum(v == 1.0 for v in loo)), "n": len(loo)}

# ---------------------------------------------------------------- 3. 인코더 비교 (표 14, 동결, 1.2M 정렬)
g_lang = t14[:9]
g_ssl = [r for r in t14 if any(s in r["name"] for s in ["DINOv2", "MoCo", "MAE", "I-JEPA"])]
g_oth = [r for r in t14 if any(s in r["name"] for s in ["SAM", "MiDaS", "Diffusion", "Supervised"])]
assert len(g_lang) == 9 and len(g_ssl) == 5 and len(g_oth) == 4
enc = {"n_lang": 9, "n_ssl": 5, "n_other": 4, "per_benchmark": {}, "per_category": {}}
for c in COLS:
    a = np.array([to100(r, c) for r in g_lang]); b = np.array([to100(r, c) for r in g_ssl])
    enc["per_benchmark"][c] = {"lang_mean": round(a.mean(), 2), "ssl_mean": round(b.mean(), 2), "diff": round(a.mean() - b.mean(), 2),
                               "lang_min": round(a.min(), 1), "ssl_max": round(b.max(), 1), "mde_q0.2": round(mde(N_Q[c]), 2)}
enc["lang_gt_ssl_benchmarks"] = int(sum(v["diff"] > 0 for v in enc["per_benchmark"].values()))
enc["lang_gt_ssl_beyond_mde"] = int(sum(v["diff"] > v["mde_q0.2"] for v in enc["per_benchmark"].values()))
enc["lang_min_gt_ssl_max_benchmarks"] = int(sum(v["lang_min"] > v["ssl_max"] for v in enc["per_benchmark"].values()))


def perm_p(a, b, B=None):
    pool = np.concatenate([a, b]); n1 = len(a); obs = a.mean() - b.mean()
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pool)), n1):
        m = np.zeros(len(pool), bool); m[list(idx)] = True
        d = pool[m].mean() - pool[~m].mean()
        tot += 1; cnt += (d >= obs - 1e-12)
    return cnt / tot


for g, cs in GROUPS.items():
    a = np.array([np.mean([to100(r, c) for c in cs]) for r in g_lang]); b = np.array([np.mean([to100(r, c) for c in cs]) for r in g_ssl])
    enc["per_category"][g] = {"lang_mean": round(a.mean(), 2), "ssl_mean": round(b.mean(), 2), "diff": round(a.mean() - b.mean(), 2),
                              "exact_perm_p_one_sided": round(perm_p(a, b), 4), "category_mde_q0.2": round(cat_mde(cs), 2),
                              "lang_range": [round(a.min(), 1), round(a.max(), 1)], "ssl_range": [round(b.min(), 1), round(b.max(), 1)]}
# DINOv2@336이 언어 지도 인코더보다 높은 비전 중심 점수
d336 = byname14["DINOv2 ViT-L/14@336"]
enc["dinov2_336_vision_centric_vs_lang"] = {
    "dinov2": round(catmean(d336)["Vision"], 2),
    "lang": {r["name"]: round(catmean(r)["Vision"], 2) for r in g_lang},
    "n_lang_below_dinov2": int(sum(catmean(r)["Vision"] < catmean(d336)["Vision"] for r in g_lang)),
}
# 표 2 순위 재계산 (표 14 행 기준)
def ranks(rows, key):
    vals = [key(r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -vals[i])
    rk = [0] * len(rows)
    for pos, i in enumerate(order):
        rk[i] = pos + 1
    return rk


lang_names = {x[0]: x[1:] for x in T["t2_ranks"]["lang"]}
ssl_names = {x[0]: x[1:] for x in T["t2_ranks"]["ssl"]}
rk_lang = [r for r in g_lang]
t2_cmp = {"lang": {}, "ssl": {}}
keys = {"All": lambda r: np.mean([to100(r, c) for c in COLS])}
for gname, cs in GROUPS.items():
    keys[gname[0]] = (lambda cs: (lambda r: np.mean([to100(r, c) for c in cs])))(cs)
order = ["All", "General", "Knowledge", "OCR", "Vision"]
keymap = {"All": keys["All"], "G": keys["G"], "K": keys["K"], "O": keys["O"], "V": keys["V"]}
tn = {"OpenAI CLIP ViT-L/14@336": "OpenAI CLIP ViT-L/14@336", "DFN-CLIP ViT-L/14@224": "DFN-CLIP ViT-L/14@224"}
paper_lang = {r[0]: r[1:] for r in T["t2_ranks"]["lang"]}
# 언어 지도 9개의 순위(DINOv2* 제외)를 표 14 행 9개로 재계산하고 표 2의 상대 순서와 비교
sub_names_14 = [r["name"] for r in g_lang]
paper_ranks_for_lang9 = {}
for r in T["t2_ranks"]["lang"]:
    if r[0].startswith("DINOv2"):
        continue
    paper_ranks_for_lang9[r[0].replace("OpenCLIP ConvNeXt-L@1024", "OpenCLIP ConvNeXt-L@1024")] = r[1:]
name_map14 = {"SigLIP ViT-SO400M/14@384": "SigLIP ViT-SO400M/14@384", "OpenCLIP ConvNeXt-XXL@1024": "OpenCLIP ConvNeXt-XXL@1024"}
mism = 0; tot = 0; detail = []
rk14 = {k: ranks(g_lang + [byname14["DINOv2 ViT-L/14@336"]], keymap[k]) for k in ["All", "G", "K", "O", "V"]}
# 표 2는 DINOv2* ViT-L/14@518를 10번째로 넣은 순위라 DINOv2@336으로 대체한 재계산과 직접 비교하지 않고,
# 언어 지도 9개의 '상대 순서'(DINOv2 제외 순위)만 비교한다.
rel14 = {k: ranks(g_lang, keymap[k]) for k in ["All", "G", "K", "O", "V"]}
for i, r in enumerate(g_lang):
    pr = paper_ranks_for_lang9.get(r["name"])
    if pr is None:
        continue
    ours = [rel14[k][i] for k in ["All", "G", "K", "O", "V"]]
    # 논문 순위에서 DINOv2*를 뺀 상대 순위
    detail.append({"model": r["name"], "ours_rel": ours, "paper_with_dino": pr})
OUT["t2_rank_check_lang9"] = detail
# 논문 순위에서 DINOv2*를 제거한 상대 순위와의 일치
def strip_dino(col):
    ps = {r[0]: r[1 + col] for r in T["t2_ranks"]["lang"]}
    names = [n for n in ps if not n.startswith("DINOv2")]
    order_ = sorted(names, key=lambda n: ps[n])
    return {n: i + 1 for i, n in enumerate(order_)}
match = 0; cells = 0
for ci, k in enumerate(["All", "G", "K", "O", "V"]):
    pr = strip_dino(ci)
    for i, r in enumerate(g_lang):
        nm = r["name"]
        key = {"OpenCLIP ConvNeXt-L@1024": "OpenCLIP ConvNeXt-L@1024"}.get(nm, nm)
        if key in pr:
            cells += 1; match += (pr[key] == rel14[k][i])
OUT["t2_rank_match_lang9_cells"] = [match, cells]
ssl_rows = [r for r in t14 if any(s in r["name"] for s in ["DINOv2", "MAE", "I-JEPA", "Diffusion", "MiDaS", "Supervised", "MoCo", "SAM"])]
paper_ssl = {r[0]: r[1:] for r in T["t2_ranks"]["ssl"]}
ssl_pairs = {"MAE ViT-L/16@224": "MAE ViT-L/16@224", "I-JEPA ViT-H/14@224": "I-JEPA ViT-H/14@224",
             "Diffusion SD2.1/16@512": "SD2.1 VAE+UNet/16@512", "MiDaS 3.0 ViT-L/16@384": "MiDaS 3.0 ViT-L/16@384",
             "Supervised ViT-L/16@224": "SupViT ViT-L/16@224", "MoCo v3 ViT-B/16@224": "MoCo v3 ViT-B/16@224",
             "MoCo v3 ViT-L/16@224": "MoCo v3 ViT-L/16@224", "SAM ViT-H/16@1024": "SAM ViT-H/16@1024"}
rows_ssl14 = [r for r in t14 if r["name"] in ssl_pairs]
rel_ssl = {k: ranks(rows_ssl14, keymap[k]) for k in ["All", "G", "K", "O", "V"]}
def strip_ssl(col):
    ps = {n: v[col] for n, v in paper_ssl.items() if n in ssl_pairs.values()}
    order_ = sorted(ps, key=lambda n: ps[n])
    return {n: i + 1 for i, n in enumerate(order_)}
m2 = c2 = 0
for ci, k in enumerate(["All", "G", "K", "O", "V"]):
    pr = strip_ssl(ci)
    for i, r in enumerate(rows_ssl14):
        c2 += 1; m2 += (pr[ssl_pairs[r["name"]]] == rel_ssl[k][i])
OUT["t2_rank_match_ssl8_cells"] = [m2, c2]
OUT["encoder_lang_vs_ssl"] = enc

# 해상도: OpenCLIP ConvNeXt-L 512 -> 1024 (같은 계열, 토큰 576으로 보간)
def get(tab, name):
    for r in T[tab]:
        if r["name"] == name:
            return r
    return None


conv = {}
for tab in ["t12_0M", "t13_0.5M", "t14_1.2M"]:
    a, b = get(tab, "OpenCLIP ConvNeXt-L@512"), get(tab, "OpenCLIP ConvNeXt-L@1024")
    conv[tab] = {"avg_512": a["avg"], "avg_1024": b["avg"], "delta": round(b["avg"] - a["avg"], 2),
                 "MMB_512_1024": [a["MMB"], b["MMB"]], "OCR_cat_delta": round(catmean(b)["OCR"] - catmean(a)["OCR"], 2),
                 "Vision_cat_delta": round(catmean(b)["Vision"] - catmean(a)["Vision"], 2)}
OUT["resolution_convnext_L_512_to_1024"] = conv
# SigLIP 384 ViT-L vs SO400M (크기 교란)와 DFN L224 vs H378는 해상도와 크기가 함께 바뀜 -> 표에서는 단일 요인 쌍이 ConvNeXt-L 하나뿐
dfn = {}
for tab in ["t12_0M", "t13_0.5M", "t14_1.2M"]:
    a, b = get(tab, "DFN-CLIP ViT-L/14@224"), get(tab, "DFN-CLIP ViT-H/14@378")
    dfn[tab] = [a["avg"], b["avg"], round(b["avg"] - a["avg"], 2)]
OUT["dfn_clip_L224_to_H378_(size+res)"] = dfn

# 해상도와 점수의 순위 상관 (표 14, 언어 지도 9개; 아키텍처·크기·데이터가 같이 바뀌므로 교란이 있다)
RES = {"OpenAI CLIP ViT-L/14@336": 336, "DFN-CLIP ViT-L/14@224": 224, "DFN-CLIP ViT-H/14@378": 378, "EVA-CLIP-02 ViT-L/14@336": 336,
       "SigLIP ViT-L/16@384": 384, "SigLIP ViT-SO400M/14@384": 384, "OpenCLIP ConvNeXt-L@512": 512,
       "OpenCLIP ConvNeXt-L@1024": 1024, "OpenCLIP ConvNeXt-XXL@1024": 1024}
res_corr = {}
for label, rs in [("lang9_all", g_lang), ("vit_only6", [r for r in g_lang if "ConvNeXt" not in r["name"]])]:
    xs = [RES[r["name"]] for r in rs]
    res_corr[label] = {g: {"spearman": round(float(stats.spearmanr(xs, [catmean(r)[g] for r in rs])[0]), 2),
                           "p": round(float(stats.spearmanr(xs, [catmean(r)[g] for r in rs])[1]), 3)} for g in GROUPS}
    res_corr[label]["n"] = len(rs)
OUT["resolution_rank_corr_T14_lang"] = res_corr

# 정렬 데이터 효과: 세 표 모두에 있는 이름
names3 = [r["name"] for r in T["t12_0M"] if get("t13_0.5M", r["name"]) and get("t14_1.2M", r["name"])]
ad = {"models": names3, "avg_0M_0.5M_1.2M": {}, "deltas": {}}
for nm in names3:
    ad["avg_0M_0.5M_1.2M"][nm] = [get(t, nm)["avg"] for t in ["t12_0M", "t13_0.5M", "t14_1.2M"]]
d01 = np.array([get("t13_0.5M", n)["avg"] - get("t12_0M", n)["avg"] for n in names3])
d12 = np.array([get("t14_1.2M", n)["avg"] - get("t13_0.5M", n)["avg"] for n in names3])
ad["deltas"] = {"0->0.5M": {"mean": round(d01.mean(), 2), "median": round(float(np.median(d01)), 2), "n_pos": int((d01 > 0).sum()), "n": len(d01),
                            "sd": round(d01.std(ddof=1), 2)},
                "0.5->1.2M": {"mean": round(d12.mean(), 2), "median": round(float(np.median(d12)), 2), "n_pos": int((d12 > 0).sum()), "n": len(d12),
                              "sd": round(d12.std(ddof=1), 2)}}
# CLIP 336 (Prismatic의 기준 인코더)에서 0M -> 0.5M(LLaVA 558K)
clip_cmp = {}
for nm in ["OpenAI CLIP ViT-L/14@336", "SigLIP ViT-SO400M/14@384"]:
    a, b, c = get("t12_0M", nm), get("t13_0.5M", nm), get("t14_1.2M", nm)
    per = {col: round(to100(b, col) - to100(a, col), 2) for col in COLS}
    clip_cmp[nm] = {"avg": [a["avg"], b["avg"], c["avg"]], "delta_0_to_0.5": round(b["avg"] - a["avg"], 2),
                    "n_pos_16": int(sum(v > 0 for v in per.values())),
                    "n_beyond_mde_16": int(sum(abs(per[col]) > mde(N_Q[col]) for col in COLS)),
                    "avg16_mde_q0.2": round(avg16_mde(), 2), "per_benchmark_0_to_0.5": per}
ad["clip336_siglip_0_to_0.5"] = clip_cmp
OUT["adapter_data_effect"] = ad

# 동결 해제: 표 14 -> 표 15 (13개 모델)
unf = {"per_model": {}, "per_benchmark": {}}
common = [r["name"] for r in T["t15_1.2M_unfrozen"]]
dmat = []
for nm in common:
    a, b = get("t14_1.2M", nm), get("t15_1.2M_unfrozen", nm)
    unf["per_model"][nm] = round(b["avg"] - a["avg"], 2)
    dmat.append([to100(b, c) - to100(a, c) for c in COLS])
dmat = np.array(dmat)
for j, c in enumerate(COLS):
    d = dmat[:, j]
    unf["per_benchmark"][c] = {"mean_delta": round(d.mean(), 2), "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                               "mde_q0.2": round(mde(N_Q[c]), 2), "n_beyond_mde_pos": int((d > mde(N_Q[c])).sum()),
                               "n_beyond_mde_neg": int((d < -mde(N_Q[c])).sum())}
unf["cells_pos"] = int((dmat > 0).sum()); unf["cells_neg"] = int((dmat < 0).sum()); unf["cells_total"] = int(dmat.size)
unf["models_avg_up"] = int(sum(v > 0 for v in unf["per_model"].values())); unf["models"] = len(common)
unf["mean_avg_delta"] = round(float(np.mean(list(unf["per_model"].values()))), 2)
unf["avg16_mde_q0.2"] = round(avg16_mde(), 2)
for g, cs in GROUPS.items():
    idx = [COLS.index(c) for c in cs]
    d = dmat[:, idx].mean(1)
    unf.setdefault("per_category", {})[g] = {"mean_delta": round(d.mean(), 2), "n_pos_models": int((d > 0).sum()), "category_mde_q0.2": round(cat_mde(cs), 2)}
# 언어 지도 6개 vs 자기지도 5개별
lang_names15 = [n for n in common if any(s in n for s in ["CLIP", "SigLIP", "OpenCLIP"])]
ssl_names15 = [n for n in common if any(s in n for s in ["DINOv2", "MoCo", "MAE", "I-JEPA"])]
unf["lang_models_avg_delta"] = round(float(np.mean([unf["per_model"][n] for n in lang_names15])), 2)
unf["ssl_models_avg_delta"] = round(float(np.mean([unf["per_model"][n] for n in ssl_names15])), 2)
for gname, names in [("lang", lang_names15), ("ssl", ssl_names15)]:
    idxs = [common.index(n) for n in names]
    unf.setdefault("per_group_category", {})[gname] = {
        g: round(float(np.mean(dmat[idxs][:, [COLS.index(c) for c in cs]])), 2) for g, cs in GROUPS.items()}
    unf["per_group_category"][gname]["n_models"] = len(names)
OUT["unfreeze_effect"] = unf
cp = {}
a, b = get("t14_1.2M", "OpenAI CLIP ViT-L/14@336"), get("t15_1.2M_unfrozen", "OpenAI CLIP ViT-L/14@336")
cp["OpenAI_CLIP_336_unfreeze_deltas"] = {c: round(to100(b, c) - to100(a, c), 2) for c in COLS}
OUT["unfreeze_clip336"] = cp

# DINOv2 격차 줄이기: 0.7M 동결(표 14), 0.7M 해제(표 15), 5M 동결/해제(표 16)
gap = {}
clip0 = {"0.7M_frozen": get("t14_1.2M", "OpenAI CLIP ViT-L/14@336"), "0.7M_unfrozen": get("t15_1.2M_unfrozen", "OpenAI CLIP ViT-L/14@336")}
dino0 = {"0.7M_frozen": get("t14_1.2M", "DINOv2 ViT-L/14@336"), "0.7M_unfrozen": get("t15_1.2M_unfrozen", "DINOv2 ViT-L/14@336")}
t16 = {r["name"]: r for r in T["t16_5M"]}
clip0["5M_frozen"] = t16["OpenAI CLIP ViT-L/14@336 ×"]; clip0["5M_unfrozen"] = t16["OpenAI CLIP ViT-L/14@336 ✓"]
dino0["5M_frozen"] = t16["DINOv2 ViT-L/14@336 ×"]; dino0["5M_unfrozen"] = t16["DINOv2 ViT-L/14@336 ✓"]
for k in clip0:
    cm, dm = catmean(clip0[k]), catmean(dino0[k])
    gap[k] = {g: round(cm[g] - dm[g], 2) for g in GROUPS}
    gap[k]["avg"] = round(clip0[k]["avg"] - dino0[k]["avg"], 2)
OUT["dinov2_gap_to_clip"] = gap
d5 = dino0["5M_unfrozen"]; c07 = clip0["0.7M_frozen"]
OUT["dinov2_5M_unfrozen_vs_clip_0.7M_frozen"] = {c: [round(to100(d5, c), 2), round(to100(c07, c), 2), round(to100(d5, c) - to100(c07, c), 2)] for c in COLS}
OUT["dinov2_5M_unfrozen_vs_clip_0.7M_frozen_cats"] = {g: [round(catmean(d5)[g], 2), round(catmean(c07)[g], 2)] for g in GROUPS}
OUT["dinov2_5M_unfrozen_wins"] = int(sum(to100(d5, c) > to100(c07, c) for c in COLS))

# ---------------------------------------------------------------- 4. 인코더 결합 (표 3)
t3 = {r["name"]: r for r in T["t3_ensemble"]}
cp_pairs = [
    ("SigLIP+ConvNext -> +DINOv2 (DINOv2 추가)", "SigLIP+ConvNext", "SigLIP+DINOv2+ConvNext"),
    ("SigLIP+DINOv2 -> +ConvNext (ConvNeXt 추가)", "SigLIP+DINOv2", "SigLIP+DINOv2+ConvNext"),
    ("SigLIP+DINOv2+ConvNext -> +CLIP", "SigLIP+DINOv2+ConvNext", "SigLIP+DINOv2+ConvNext+CLIP"),
    ("SigLIP+ConvNext -> CLIP+ConvNext (SigLIP 대신 CLIP)", "SigLIP+ConvNext", "CLIP+ConvNext"),
    ("SigLIP+DINOv2+ConvNext-L -> SigLIP+CLIP+ConvNext-L (DINOv2 대신 CLIP)", "SigLIP+DINOv2+ConvNext-L", "SigLIP+CLIP+ConvNext-L"),
    ("SigLIP+DINOv2 -> 4개 모두", "SigLIP+DINOv2", "SigLIP+DINOv2+ConvNext+CLIP"),
]
ens = []
for label, a, b in cp_pairs:
    ra, rb = t3[a], t3[b]
    per = {c: round(to100(rb, c) - to100(ra, c), 2) for c in COLS}
    cat = {g: round(catmean(rb)[g] - catmean(ra)[g], 2) for g in GROUPS}
    ens.append({"pair": label, "avg_delta": round(rb["avg"] - ra["avg"], 2), "avg16_mde_q0.2": round(avg16_mde(), 2),
                "n_pos": int(sum(v > 0 for v in per.values())), "n_neg": int(sum(v < 0 for v in per.values())),
                "n_beyond_mde": int(sum(abs(per[c]) > mde(N_Q[c]) for c in COLS)),
                "cat_delta": cat, "cat_mde": {g: round(cat_mde(cs), 2) for g, cs in GROUPS.items()},
                "MMVP_RWQA_CV2D_CV3D": [per["MMVP"], per["RWQA"], per["CV2D"], per["CV3D"]],
                "OCR_cols": [per["ChartQA"], per["OCRBench"], per["TextVQA"], per["DocVQA"]]})
OUT["ensemble_pairs"] = ens
singles = {}
for nm in ["SigLIP ViT-SO400M/14@384", "OpenCLIP ConvNeXt-XXL@1024", "OpenAI CLIP ViT-L/14@336"]:
    singles[nm] = {"avg_T14": byname14[nm]["avg"]}
best4 = t3["SigLIP+DINOv2+ConvNext+CLIP"]
singles["four_encoder_minus_best_single_avg"] = round(best4["avg"] - byname14["SigLIP ViT-SO400M/14@384"]["avg"], 2)
singles["siglip+dino2_minus_siglip_single_avg"] = round(t3["SigLIP+DINOv2"]["avg"] - byname14["SigLIP ViT-SO400M/14@384"]["avg"], 2)
singles["siglip+convnext_minus_siglip_single"] = round(t3["SigLIP+ConvNext"]["avg"] - byname14["SigLIP ViT-SO400M/14@384"]["avg"], 2)
singles["note"] = "표 3은 모두 576토큰으로 보간한 융합, 표 14의 단일 인코더는 원래 토큰 수(SigLIP 729)라 토큰 수가 다름"
OUT["ensemble_vs_single"] = singles

# ---------------------------------------------------------------- 5. SVA와 연결기 절제 (표 4-8, 19)
G4 = GROUPS
cat_cols = {"General": G4["General"], "Knowledge": G4["Knowledge"], "OCR": G4["OCR"], "Vision": G4["Vision"]}
cm = {g: round(cat_mde(cs), 2) for g, cs in cat_cols.items()}
cm["Average(16)"] = round(avg16_mde(), 2)


def cmp_tab(tab, a, b, cols):
    d = {c: round(tab[b][i] - tab[a][i], 2) for i, c in enumerate(cols)}
    return d


abl = {"category_mde_q0.2": cm}
t4 = T["t4_connector"]
abl["t4"] = {"SVA_minus_Concat": cmp_tab(t4, "Concat", "SVA", t4["cols"]), "SVA_minus_NoMultiAgg": cmp_tab(t4, "SVA-no-multi-agg", "SVA", t4["cols"]),
             "NoMultiAgg_minus_Concat": cmp_tab(t4, "Concat", "SVA-no-multi-agg", t4["cols"]), "SVA_minus_Resampler": cmp_tab(t4, "Resampler", "SVA", t4["cols"])}
t5 = T["t5_compress"]
abl["t5"] = {"SVA_minus_Interp": cmp_tab(t5, "Interpolate+MLP", "SVA", t5["cols"]), "SVA_minus_CAbstractor": cmp_tab(t5, "C-Abstractor", "SVA", t5["cols"]),
             "SVA_minus_LDPv2": cmp_tab(t5, "LDPv2", "SVA", t5["cols"])}
abl["t6_OCR_diffs"] = {"D3-D2": round(52.4 - 52.1, 2), "D4-D3": round(52.8 - 52.4, 2), "D4-D2": round(52.8 - 52.1, 2),
                       "G2-G1": round(52.6 - 52.4, 2), "G3-G2": round(53.1 - 52.6, 2), "G3-G1": round(53.1 - 52.4, 2), "multiagg": round(53.3 - 52.4, 2),
                       "ocr_category_mde": cm["OCR"]}
# 표 19(데이터 균형): 16개 벤치마크 평균과 차이
t19 = {r["name"].split("=")[-1].strip().split()[0].replace("𝑘", "k"): r for r in T["t19_balance"]}
keys19 = list(t19.keys())
abl["t19"] = {"avg": {k: t19[k]["avg"] for k in keys19}, "avg16_mde_q0.2": round(avg16_mde(), 2)}
for a, b in [("150𝑘", "250𝑘"), ("250𝑘", "350𝑘"), ("350𝑘", "450𝑘")]:
    pass
ks = list(t19)
pairs = [(ks[0], ks[1]), (ks[1], ks[2]), (ks[2], ks[3]), (ks[0], ks[3])]
abl["t19"]["pairs"] = []
for a, b in pairs:
    per = {c: round(to100(t19[b], c) - to100(t19[a], c), 2) for c in COLS}
    abl["t19"]["pairs"].append({"from": a, "to": b, "avg_delta": round(t19[b]["avg"] - t19[a]["avg"], 2), "n_pos": int(sum(v > 0 for v in per.values())),
                                "n_neg": int(sum(v < 0 for v in per.values())), "n_beyond_mde": int(sum(abs(per[c]) > mde(N_Q[c]) for c in COLS))})
t8 = T["t8_data"]
abl["t8"] = {"10M_to_7M": cmp_tab(t8, "Cambrian-10M", "Cambrian-7M", t8["cols"]), "665K_to_7M": cmp_tab(t8, "LLaVA-665K", "Cambrian-7M", t8["cols"])}
t20 = {r["name"]: r for r in T["t20_ratio"]}
abl["t20_avg"] = {k: v["avg"] for k, v in t20.items()}
abl["t20_exp5_minus_exp3"] = round(t20["exp5"]["avg"] - t20["exp3"]["avg"], 2)
OUT["ablation_noise"] = abl

# ---------------------------------------------------------------- 6. 최종 모델 비교 (표 9) 와 CV-Bench 우연 수준
w = {}
vals = {r["name"]: r["vals"] for r in T["t9_final"]}
for nm in ["Cambrian-1-8B", "Cambrian-1-13B", "Cambrian-1-34B"]:
    base = nm.replace("Cambrian-1", "LLaVA-NeXT")
    a, b = vals[base], vals[nm]
    # 열 순서: tokens(0) avgG mmep mmb seed gqa avgK sqa mmmu mv ai2d avgO chq ocr tvqa doc avgV mmvp rwqa c2d c3d
    names = ["tokens", "avgG", "MMEP", "MMB", "SEEDI", "GQA", "avgK", "SQAI", "MMMU", "MathVista", "AI2D", "avgO", "ChartQA", "OCRBench",
             "TextVQA", "DocVQA", "avgV", "MMVP", "RWQA", "CV2D", "CV3D"]
    d = {names[i]: round(b[i] - a[i], 1) for i in range(1, 21)}
    bench_d = {k: v for k, v in d.items() if k in N_Q or k == "MMEP"}
    w[nm + "_minus_" + base] = {"delta": d, "n_benchmarks_better": int(sum(v > 0 for k, v in bench_d.items())),
                               "n_beyond_mde": int(sum(abs(v if k != "MMEP" else v / 20) > mde(N_Q[k]) for k, v in bench_d.items()))}
OUT["t9_cambrian_vs_llavanext"] = w
OUT["cvbench_chance_levels_from_data"] = CV["chance"]

# 부호 검정 (모델 수준: 모델이 독립이라고 보는 참고용)
bt = lambda k, n: round(float(stats.binomtest(k, n, 0.5).pvalue), 4)
OUT["sign_tests_two_sided"] = {
    "adapter_0_to_0.5M_models_up_14_of_16": bt(14, 16), "adapter_0.5_to_1.2M_models_up_13_of_16": bt(13, 16),
    "unfreeze_models_up_11_of_13": bt(11, 13),
    "lang_gt_ssl_benchmarks_16_of_16": bt(16, 16),
    "table24_abs_delta_gt2_cells_of_10": [int(sum(abs(a - b) > 2 for a, b in zip(f24["8B_fuzzy"], f24["8B_gpt35"]))), 10],
    "table24_max_abs_delta": round(max(abs(a - b) for a, b in zip(f24["8B_fuzzy"], f24["8B_gpt35"])), 1),
}
(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))
print(json.dumps({k: OUT[k] for k in ["transcription_check", "mde_category_q0.2", "mde_avg16_q0.1_0.2_0.3", "fig3_enabled_vs_table_means"]}, ensure_ascii=False, indent=1))
