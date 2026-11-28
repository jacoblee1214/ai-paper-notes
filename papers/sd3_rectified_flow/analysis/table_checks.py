"""Recomputation of the paper's tables (arXiv:2403.03206): Table 1 rank consistency and noise, Table 2 Pareto
sensitivity, Tables 4-5 arithmetic, MM-DiT parameter counts.  No training, numpy/scipy only.
Seeds: numpy default_rng(0).  Writes table_checks_results.json.
"""
import json
import pathlib
from itertools import combinations

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(0)
out = {}

# ---------------------------------------------------------------- Table 1 (transcribed from the paper)
T1 = {
    "rf/lognorm(0.00, 1.00)": (1.54, 1.25, 1.50), "rf/lognorm(1.00, 0.60)": (2.08, 3.50, 2.00),
    "rf/lognorm(0.50, 0.60)": (2.71, 8.50, 1.00), "rf/mode(1.29)": (2.75, 3.25, 3.00),
    "rf/lognorm(0.50, 1.00)": (2.83, 1.50, 2.50), "eps/linear": (2.88, 4.25, 2.75),
    "rf/mode(1.75)": (3.33, 2.75, 2.75), "rf/cosmap": (4.13, 3.75, 4.00),
    "edm(0.00, 0.60)": (5.63, 13.25, 3.25), "rf": (5.67, 6.50, 5.75), "v/linear": (6.83, 5.75, 7.75),
    "edm(0.60, 1.20)": (9.00, 13.00, 9.00), "v/cos": (9.17, 12.25, 8.75), "edm/cos": (11.04, 14.25, 11.25),
    "edm/rf": (13.04, 15.25, 13.25), "edm(-1.20, 1.20)": (15.58, 20.25, 15.00)}
names = list(T1)
A = np.array([T1[k] for k in names])     # all(24 settings), 5 steps (4 settings), 50 steps (12 settings)
# settings: 2 EMA x 2 datasets x {50 steps cfg 1, 2.5, 5; 5, 10, 25 steps cfg 5} = 24.
# 5-step column averages 4 settings, 50-step 12, so 10/25 steps are the remaining 8.
n_all, n5, n50, n_rest = 24, 4, 12, 8
rest = (n_all * A[:, 0] - n5 * A[:, 1] - n50 * A[:, 2]) / n_rest
t1 = {"grid_check": {"24*all_max_dist_to_integer": float(np.abs(24 * A[:, 0] - np.round(24 * A[:, 0])).max()),
                     "4*r5_max_dist_to_integer": float(np.abs(4 * A[:, 1] - np.round(4 * A[:, 1])).max()),
                     "12*r50_max_dist_to_integer": float(np.abs(12 * A[:, 2] - np.round(12 * A[:, 2])).max())},
      "implied_rank_10_25_steps": {k: float(v) for k, v in zip(names, rest)},
      "implied_min_max": [float(rest.min()), float(rest.max())]}
# rank reversal between 5 and 50 steps
rho, _ = stats.spearmanr(A[:, 1], A[:, 2])
perm = [stats.spearmanr(A[:, 1], rng.permutation(A[:, 2]))[0] for _ in range(20000)]
t1["spearman_5_vs_50"] = {"rho": float(rho), "perm_p_two_sided": float((np.sum(np.abs(perm) >= abs(rho)) + 1) / 20001), "n_variants": len(names)}
t1["mean_abs_rank_gap_5_vs_50"] = float(np.abs(A[:, 1] - A[:, 2]).mean())
# top variants: paired comparison across the three sampler regimes (5, 10/25, 50 steps).  n=3 groups: exploratory only.
top = names[0]
G = np.stack([A[:, 1], rest, A[:, 2]], 1)
cmp = {}
pvals = []
for j in range(1, 6):
    d = G[0] - G[j]
    tt = stats.ttest_1samp(d, 0.0)
    cmp[names[j]] = {"mean_diff_all24_rank": float(A[0, 0] - A[j, 0]), "group_diffs": [float(x) for x in d],
                     "t": float(tt.statistic), "p_raw_df2": float(tt.pvalue)}
    pvals.append(tt.pvalue)
order = np.argsort(pvals); m = len(pvals); holm = np.empty(m); run = 0
for i, idx in enumerate(order):
    run = max(run, (m - i) * pvals[idx]); holm[idx] = min(1, run)
for j, h in zip(range(1, 6), holm):
    cmp[names[j]]["p_holm"] = float(h)
t1["top_vs_next5_regime_paired"] = cmp
out["table1"] = t1

# ---------------------------------------------------------------- Table 2: Pareto ranks and noise sensitivity
T2 = {  # variant: (IN_CLIP, IN_FID, CC_CLIP, CC_FID)
    "rf": (0.247, 49.70, 0.217, 94.90), "edm(-1.20, 1.20)": (0.236, 63.12, 0.200, 116.60),
    "eps/linear": (0.245, 48.42, 0.222, 90.34), "v/cos": (0.244, 50.74, 0.209, 97.87), "v/linear": (0.246, 51.68, 0.217, 100.76),
    "rf/lognorm(0.50, 0.60)": (0.256, 80.41, 0.233, 120.84), "rf/mode(1.75)": (0.253, 44.39, 0.218, 94.06),
    "rf/lognorm(1.00, 0.60)": (0.254, 114.26, 0.234, 147.69), "rf/lognorm(-0.50, 1.00)": (0.248, 45.64, 0.219, 89.70),
    "rf/lognorm(0.00, 1.00)": (0.250, 45.78, 0.224, 89.91)}
v = list(T2)
M = np.array([T2[k] for k in v])

def pareto_ranks(clip, fid):
    n = len(clip); rank = np.zeros(n, int); left = np.ones(n, bool); r = 0
    while left.any():
        r += 1
        idx = np.where(left)[0]
        front = [i for i in idx if not any((clip[j] >= clip[i] and fid[j] <= fid[i] and (clip[j] > clip[i] or fid[j] < fid[i])) for j in idx if j != i)]
        rank[front] = r; left[front] = False
    return rank

obs = {}
for ds, (ci, fi) in {"ImageNet": (0, 1), "CC12M": (2, 3)}.items():
    obs[ds] = {k: int(r) for k, r in zip(v, pareto_ranks(M[:, ci], M[:, fi]))}
t2 = {"observed_pareto_rank_among_10_listed": obs}
target = v.index("rf/lognorm(0.00, 1.00)")
sens = {}
for s_fid in [0.0, 0.02, 0.05, 0.10]:
    for s_clip in [0.0, 0.01, 0.02]:
        res = {}
        for ds, (ci, fi) in {"ImageNet": (0, 1), "CC12M": (2, 3)}.items():
            ranks = []
            for _ in range(2000):
                clip = M[:, ci] * (1 + s_clip * rng.normal(size=len(v)))
                fid = M[:, fi] * (1 + s_fid * rng.normal(size=len(v)))
                ranks.append(pareto_ranks(clip, fid)[target])
            ranks = np.array(ranks)
            res[ds] = {"P_front1": float((ranks == 1).mean()), "P_rank_le2": float((ranks <= 2).mean()), "mean_rank": float(ranks.mean())}
        sens[f"fid_rel_sd={s_fid},clip_rel_sd={s_clip}"] = res
t2["lognorm(0,1)_rank_sensitivity_2000draws"] = sens
# does the paper's text hold: lognorm(0,1) 3rd best in 2 of 4 columns, 2nd best once (among the 10 listed)
col_rank = {}
for j, nm in enumerate(["IN_CLIP", "IN_FID", "CC_CLIP", "CC_FID"]):
    key = -M[:, j] if "CLIP" in nm else M[:, j]
    col_rank[nm] = int(np.argsort(np.argsort(key))[target] + 1)
t2["lognorm(0,1)_column_rank_among_10"] = col_rank
out["table2"] = t2

# ---------------------------------------------------------------- Table 4 and Table 5 arithmetic
t4 = {"original": [11.75, 71.54, 6.50, 33.44, 95.00, 41.41], "mix": [24.75, 68.09, 18.00, 41.56, 93.75, 52.53]}
out["table4_mean_of_six"] = {"original": float(np.mean(t4["original"])), "mix": float(np.mean(t4["mix"])), "paper": [43.27, 49.78]}
T5 = {  # overall, single, two, counting, colors, position, attribution
    "minDALL-E": (0.23, 0.73, 0.11, 0.12, 0.37, 0.02, 0.01), "SD v1.5": (0.43, 0.97, 0.38, 0.35, 0.76, 0.04, 0.06),
    "PixArt-alpha": (0.48, 0.98, 0.50, 0.44, 0.80, 0.08, 0.07), "SD v2.1": (0.50, 0.98, 0.51, 0.44, 0.85, 0.07, 0.17),
    "DALL-E 2": (0.52, 0.94, 0.66, 0.49, 0.77, 0.10, 0.19), "SDXL": (0.55, 0.98, 0.74, 0.39, 0.85, 0.15, 0.23),
    "SDXL Turbo": (0.55, 1.00, 0.72, 0.49, 0.80, 0.10, 0.18), "IF-XL": (0.61, 0.97, 0.74, 0.66, 0.81, 0.13, 0.35),
    "DALL-E 3": (0.67, 0.96, 0.87, 0.47, 0.83, 0.43, 0.45), "Ours d18 512": (0.58, 0.97, 0.72, 0.52, 0.78, 0.16, 0.34),
    "Ours d24 512": (0.62, 0.98, 0.74, 0.63, 0.67, 0.34, 0.36), "Ours d30 512": (0.64, 0.96, 0.80, 0.65, 0.73, 0.33, 0.37),
    "Ours d38 512": (0.68, 0.98, 0.84, 0.66, 0.74, 0.40, 0.43), "Ours d38 512 DPO": (0.71, 0.98, 0.89, 0.73, 0.83, 0.34, 0.47),
    "Ours d38 1024 DPO": (0.74, 0.99, 0.94, 0.72, 0.89, 0.33, 0.60)}
t5 = {}
for k, row in T5.items():
    mean6 = float(np.mean(row[1:]))
    t5[k] = {"paper_overall": row[0], "mean_of_six": round(mean6, 4), "abs_diff": round(abs(mean6 - row[0]), 4)}
out["table5_overall_vs_mean_of_six"] = t5
out["table5_max_abs_diff"] = max(v_["abs_diff"] for v_ in t5.values())
# rounding bound: each entry has +-0.005 rounding error -> mean of six within +-0.005 (+0.005 for overall)
# sample-size note: GenEval overall SE from the binomial is not recoverable (prompt count per task not in paper)
# ---------------------------------------------------------------- GenEval gaps: Ours d38 1024 vs DALL-E 3
out["table5_ours_vs_dalle3"] = {"diff_overall": round(0.74 - 0.67, 2), "note": "single run per model, no CI in the paper; GenEval has 553 prompts (Ghosh et al. 2023) -> binomial SE of a mean score near 0.7 is ~0.02 under independence"}
# ---------------------------------------------------------------- parameter counts from the reference code structure
def mmdit_params(d, text_dim=4096, vec_dim=2048, in_ch=16, patch=2, qkv_bias=True, qk_norm=True):
    h = 64 * d
    heads_dim = 64
    def stream(pre_only):
        adaln = (2 if pre_only else 6) * h * h + (2 if pre_only else 6) * h
        qkv = 3 * h * h + (3 * h if qkv_bias else 0)
        proj = 0 if pre_only else h * h + h
        mlp = 0 if pre_only else (h * 4 * h + 4 * h) + (4 * h * h + h)
        qkn = 2 * heads_dim if qk_norm else 0
        return adaln + qkv + proj + mlp + qkn
    blocks = sum(2 * stream(False) if i < d - 1 else stream(False) + stream(True) for i in range(d))
    glob = (in_ch * patch * patch * h + h) + (256 * h + h + h * h + h) + (vec_dim * h + h + h * h + h) + (text_dim * h + h) \
        + (2 * h * h + 2 * h) + (h * patch * patch * in_ch + patch * patch * in_ch)
    return blocks, glob, blocks + glob

pc = {}
for d in [15, 18, 21, 24, 30, 38]:
    b, g, tot = mmdit_params(d)
    pc[str(d)] = {"hidden": 64 * d, "blocks": b, "global": g, "total": tot, "approx_36*64^2*d^3": 36 * 64 ** 2 * d ** 3,
                  "total_B": round(tot / 1e9, 3)}
out["param_counts"] = pc
(HERE / "table_checks_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"t1_grid": t1["grid_check"], "rest": t1["implied_rank_10_25_steps"], "min_max": t1["implied_min_max"],
                  "spearman": t1["spearman_5_vs_50"], "gap": t1["mean_abs_rank_gap_5_vs_50"], "cmp": t1["top_vs_next5_regime_paired"]}, indent=1))
print(json.dumps(t2["observed_pareto_rank_among_10_listed"], indent=0)); print(col_rank)
for k in ["fid_rel_sd=0.0,clip_rel_sd=0.0", "fid_rel_sd=0.02,clip_rel_sd=0.01", "fid_rel_sd=0.05,clip_rel_sd=0.01", "fid_rel_sd=0.1,clip_rel_sd=0.02"]:
    print(k, sens[k])
print(out["table4_mean_of_six"], out["table5_max_abs_diff"], {k: v_ for k, v_ in t5.items() if v_["abs_diff"] > 0.006})
print({d: (v_["total_B"], v_["approx_36*64^2*d^3"] / 1e9) for d, v_ in pc.items()})
