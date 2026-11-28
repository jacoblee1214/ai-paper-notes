"""Refit of the Figure 8 scaling claims from digitised points (fig8_points.json; run fig8_digitize.py first).

 A. validation loss vs training FLOPs: power law (with and without an offset), bootstrap CIs, extrapolation
 B. FLOPs / parameter ratio (consistency of the digitised x axis with the parameter formula, C ~ 6 N D)
 C. correlation of validation loss with GenEval / human ELO / T2I-CompBench / video ELO: Pearson r recomputed, bootstrap CI,
    within-depth (depth fixed effects removed) r, leave-one-depth-out r, jitter from +-1 px digitisation error
No training, numpy/scipy only; seeds: numpy default_rng(0).  Writes scaling_refit_results.json.
"""
import json
import pathlib

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

HERE = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(0)
P = json.loads((HERE / "fig8_points.json").read_text())
tc = json.loads((HERE / "table_checks_results.json").read_text())["param_counts"]
out = {}

def flat(panel):
    X, Y, D = [], [], []
    for d, pts in P[panel]["points"].items():
        for p in pts:
            X.append(p["x"]); Y.append(p["y"]); D.append(int(d))
    return np.array(X), np.array(Y), np.array(D)

# ---------------------------------------------------------------- A / B. FLOPs panel
C, L, D = flat("fig8_top2_loss_vs_flops")
out["A_n_points"] = {"n": int(len(C)), "per_depth": P["fig8_top2_loss_vs_flops"]["n_per_depth"]}
last = {d: (C[D == d].max(), L[D == d][np.argmax(C[D == d])]) for d in sorted(set(D))}
ratio = {d: last[d][0] / tc[str(d)]["total"] for d in last}
out["B_final_point_flops_over_params"] = {str(d): {"C_final": float(last[d][0]), "loss_final": float(last[d][1]), "C_over_N": float(ratio[d]),
                                                   "N_total": tc[str(d)]["total"]} for d in last}
r_arr = np.array(list(ratio.values()))
out["B_ratio_spread"] = {"mean": float(r_arr.mean()), "rel_range": float((r_arr.max() - r_arr.min()) / r_arr.mean()),
                         "implied_tokens_per_step_per_sample_if_6ND": float(r_arr.mean() / 6 / (500e3 * 4096))}
x = np.log10(C / 1e21)

def pl(x, a, b):          # L = a * (C/1e21)^-b
    return a * 10 ** (-b * x)
def pl_off(x, e, a, b):   # L = e + a * (C/1e21)^-b
    return e + a * 10 ** (-b * x)

def fit_pl(x, y):
    p, _ = curve_fit(pl, x, y, p0=[0.41, 0.01], maxfev=20000)
    return p
def fit_off(x, y):
    p, _ = curve_fit(pl_off, x, y, p0=[0.37, 0.05, 0.1], bounds=([0.0, 0.0, 0.0], [0.388, 5.0, 3.0]), maxfev=40000)
    return p
def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())

res = {}
p1 = fit_pl(x, L)
res["power_all_points"] = {"a": float(p1[0]), "b": float(p1[1]), "r2": r2(L, pl(x, *p1)), "n": int(len(x))}
xl = np.log10(np.array([last[d][0] for d in last]) / 1e21); yl = np.array([last[d][1] for d in last])
sl = stats.linregress(xl, np.log(yl))
res["loglog_final_points_by_depth"] = {"slope_dlnL_dlog10C": float(sl.slope), "r2": float(sl.rvalue ** 2), "n": 5,
                                        "b_equiv": float(-sl.slope / np.log(10))}
# model-size: final loss vs parameter count
Ns = np.array([tc[str(d)]["total"] for d in last]); sn = stats.linregress(np.log10(Ns), np.log(yl))
res["loglog_final_loss_vs_params"] = {"slope_dlnL_dlog10N": float(sn.slope), "r2": float(sn.rvalue ** 2)}
p2 = fit_off(x, L)
res["power_with_offset"] = {"E": float(p2[0]), "a": float(p2[1]), "b": float(p2[2]), "r2": r2(L, pl_off(x, *p2))}
# bootstrap (resample points; and cluster bootstrap by depth)
def boot(fn, B=1500, cluster=False):
    outv = []
    depths = np.array(sorted(set(D)))
    for _ in range(B):
        if cluster:
            ch = rng.choice(depths, len(depths)); idx = np.concatenate([np.where(D == d)[0] for d in ch])
        else:
            idx = rng.integers(0, len(x), len(x))
        try:
            outv.append(fn(idx))
        except Exception:
            pass
    return np.array(outv)
x_ext = np.log10(1e23 / 1e21)
for name, fn in {"pl_b": lambda i: fit_pl(x[i], L[i])[1],
                 "pl_pred_1e23": lambda i: pl(x_ext, *fit_pl(x[i], L[i])),
                 "off_E": lambda i: fit_off(x[i], L[i])[0], "off_b": lambda i: fit_off(x[i], L[i])[2],
                 "off_pred_1e23": lambda i: pl_off(x_ext, *fit_off(x[i], L[i]))}.items():
    for cl in (False, True):
        v = boot(fn, cluster=cl)
        res[f"boot_{name}_{'cluster' if cl else 'point'}"] = {"median": float(np.median(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "n_ok": int(len(v))}
res["pred_1e23"] = {"power": float(pl(x_ext, *p1)), "offset": float(pl_off(x_ext, *p2)),
                    "last_observed_loss": float(L[np.argmax(C)]), "last_observed_C": float(C.max())}
res["digitisation"] = {"y_per_px": abs(P["fig8_top2_loss_vs_flops"]["px_to_data"]["y_per_px"]), "x_decades_per_px": 1 / 511.2}
out["A_fit"] = res

# ---------------------------------------------------------------- C. correlations
reported = {"fig8_bot1_geneval": -0.920, "fig8_bot2_human_elo": -0.982, "fig8_bot3_compbench": -0.898, "fig8_bot4_video_elo": -0.908}
cor = {}
for panel, rep in reported.items():
    X, Y, Dd = flat(panel)
    r_, p_ = stats.pearsonr(X, Y)
    rho, _ = stats.spearmanr(X, Y)
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(X), len(X))
        if X[i].std() > 0 and Y[i].std() > 0:
            bs.append(stats.pearsonr(X[i], Y[i])[0])
    # depth fixed effects removed
    Xd, Yd = X.copy(), Y.copy()
    for d in set(Dd):
        Xd[Dd == d] -= X[Dd == d].mean(); Yd[Dd == d] -= Y[Dd == d].mean()
    n_df = len(X) - len(set(Dd)) - 1
    rw = float(np.corrcoef(Xd, Yd)[0, 1])
    t_w = rw * np.sqrt(n_df / (1 - rw ** 2)); p_w = float(2 * stats.t.sf(abs(t_w), n_df))
    perm = [np.corrcoef(Xd, rng.permutation(Yd))[0, 1] for _ in range(4000)]
    p_perm = float((np.sum(np.abs(perm) >= abs(rw)) + 1) / 4001)
    loo = {}
    for d in sorted(set(Dd)):
        keep = Dd != d
        if keep.sum() > 3:
            loo[str(d)] = float(np.corrcoef(X[keep], Y[keep])[0, 1])
    # +-1 px jitter
    ax_, ay_ = P[panel]["px_to_data"]["x_per_px"], P[panel]["px_to_data"]["y_per_px"]
    jit = [np.corrcoef(X + rng.normal(0, 1, len(X)) * abs(ax_), Y + rng.normal(0, 1, len(Y)) * abs(ay_))[0, 1] for _ in range(1000)]
    # within-depth r for each depth with >=4 points
    wd = {str(d): float(np.corrcoef(X[Dd == d], Y[Dd == d])[0, 1]) for d in sorted(set(Dd)) if (Dd == d).sum() >= 4}
    cor[panel] = {"n": int(len(X)), "n_per_depth": P[panel]["n_per_depth"], "paper_r": rep, "r": float(r_), "spearman": float(rho),
                  "ci95_boot_points": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "r_within_depth": rw, "df_within": int(n_df), "p_within_t": p_w, "p_within_perm": p_perm,
                  "within_depth_each": wd, "leave_one_depth_out": loo,
                  "jitter_1px_sd_of_r": float(np.std(jit))}
out["C_correlations"] = cor
# cross-check: GenEval and CompBench panels show the same checkpoints -> identical loss values per depth
Y1 = {d: np.sort([p["y"] for p in P["fig8_bot1_geneval"]["points"][d]]) for d in P["fig8_bot1_geneval"]["points"]}
Y3 = {d: np.sort([p["y"] for p in P["fig8_bot3_compbench"]["points"][d]]) for d in P["fig8_bot3_compbench"]["points"]}
mm = {}
for d in Y1:
    a_, b_ = Y1[d], Y3[d]
    matched = sum(1 for v in a_ if np.abs(b_ - v).min() < 3 * 6.3e-5)
    mm[d] = {"n_geneval": int(len(a_)), "n_compbench": int(len(b_)), "geneval_losses_found_in_compbench_within_3px": int(matched)}
out["C_crosscheck_same_checkpoints"] = mm
(HERE / "scaling_refit_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ["A_n_points", "B_final_point_flops_over_params", "B_ratio_spread"]}, indent=1))
print(json.dumps(res, indent=1))
for k, v in cor.items():
    print(k, {kk: v[kk] for kk in ["n", "paper_r", "r", "spearman", "ci95_boot_points", "r_within_depth", "p_within_perm", "jitter_1px_sd_of_r"]}, v["within_depth_each"], v["leave_one_depth_out"])
print(mm)
