"""SD3 (arXiv:2403.03206) section 2-3.1 loss-weighting algebra, checked numerically.

No training. numpy/scipy only. Writes loss_weighting_results.json next to this file.
  A. eq (11)-(13): ||v - u||^2 == (lambda' b / 2)^2 ||eps_hat - eps||^2 (random tensors)
  B. stated w_t for RF, cosine-eps, cosine-v, EDM vs the exact weight implied by eq (13); LDM-linear w_t
  C. timestep densities: logit-normal, mode sampling (valid range of s), CosMap -- normalisation,
     mode location, mass in t-bins, sampling check with KS tests
  D. every formulation re-expressed as an equivalent RF timestep density pi_eff(t)
"""
import json
import pathlib

import numpy as np
from scipy import integrate, stats

_trap = getattr(np, "trapezoid", None) or np.trapz
HERE = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(0)
out = {}

# ---------------------------------------------------------------- schedules (a_t, b_t, derivatives)
def sched_rf(t):
    return 1 - t, t, -np.ones_like(t), np.ones_like(t)

def sched_cos(t):
    th = np.pi * t / 2
    return np.cos(th), np.sin(th), -np.pi / 2 * np.sin(th), np.pi / 2 * np.cos(th)

def lam_and_dlam(a, b, da, db):
    return 2 * np.log(a / b), 2 * (da / a - db / b)

# ---------------------------------------------------------------- A. eq (10)-(13) identity
def check_identity(sched, n=200):
    t = rng.uniform(0.02, 0.98, n)
    a, b, da, db = sched(t)
    lam, dlam = lam_and_dlam(a, b, da, db)
    d = 7
    x0, eps, vhat = (rng.normal(size=(n, d)) for _ in range(3))
    z = a[:, None] * x0 + b[:, None] * eps
    u = da[:, None] * x0 + db[:, None] * eps                      # eq (5) conditional velocity
    u10 = (da / a)[:, None] * z - (b / 2 * dlam)[:, None] * eps   # eq (10)
    eps_hat = (-2 / (dlam * b))[:, None] * (vhat - (da / a)[:, None] * z)
    lhs = ((vhat - u) ** 2).sum(1)
    rhs = ((b * dlam / 2) ** 2) * ((eps_hat - eps) ** 2).sum(1)  # eq (12)
    w = -0.5 * dlam * b ** 2                                      # w_t for L_CFM
    rhs13 = -0.5 * w * dlam * ((eps_hat - eps) ** 2).sum(1)      # eq (13)
    return float(np.abs(u - u10).max()), float(np.abs(lhs - rhs).max() / lhs.mean()), float(np.abs(lhs - rhs13).max() / lhs.mean())

out["A_identity"] = {name: dict(zip(["eq10_max_abs_err", "eq12_rel_err", "eq13_rel_err"], check_identity(s)))
                     for name, s in [("rf", sched_rf), ("cosine", sched_cos)]}

# ---------------------------------------------------------------- B. weights
# Convention (eq 13): L_w = -1/2 E_{t~U} [ w_t lambda'_t ||eps_hat - eps||^2 ]  => per-d(lambda) weight is w/2.
# An objective that puts weight c(t) on ||eps_hat - eps||^2 per unit t therefore has w = 2 c(t)/|lambda'_t|.
t = np.linspace(0.002, 0.998, 4000)
res_B = {}

# RF: velocity MSE; v - u = (eps_hat - eps)/(1-t)  => c = 1/(1-t)^2
a, b, da, db = sched_rf(t); lam, dl = lam_and_dlam(a, b, da, db)
w_ex = 2 / np.abs(dl) / (1 - t) ** 2
res_B["rf"] = {"stated": "t/(1-t)", "ratio_exact_over_stated_min": float((w_ex / (t / (1 - t))).min()),
               "ratio_exact_over_stated_max": float((w_ex / (t / (1 - t))).max())}

# cosine, eps-prediction: c = 1 ; v-prediction: v_hat - v = (eps_hat - eps)/a  => c = 1/a^2
a, b, da, db = sched_cos(t); lam, dl = lam_and_dlam(a, b, da, db)
for name, c, stated in [("cos_eps", np.ones_like(t), 1 / np.cosh(lam / 2)), ("cos_v", 1 / a ** 2, np.exp(-lam / 2))]:
    w_ex = 2 / np.abs(dl) * c
    r = w_ex / stated
    res_B[name] = {"stated": "sech(lam/2)" if name == "cos_eps" else "exp(-lam/2)",
                   "ratio_exact_over_stated_mean": float(r.mean()), "ratio_rel_spread": float(r.std() / r.mean())}
res_B["cos_eps"]["1/pi"] = 1 / np.pi
res_B["cos_v"]["2/pi"] = 2 / np.pi
res_B["v_over_eps_ratio_exact_vs_stated"] = float((1 / (a ** 2)).mean() * 0 + np.mean((1 / a ** 2) / ((np.exp(-lam / 2)) / (1 / np.cosh(lam / 2)))))

# EDM: lognormal sigma, F-pred => loss lambda(sigma)||D - x||^2, x-err = sigma*eps-err
# => c = (sigma^2+sd^2)/sd^2 ; t uniform maps lam ~ N(-2Pm, (2Ps)^2) so |lam'| = 1/N(lam)
for Pm, Ps in [(-1.2, 1.2), (0.0, 0.6), (0.6, 1.2)]:
    mu, sd = -2 * Pm, 2 * Ps
    lam = stats.norm.ppf(t, mu, sd)
    dl = 1 / stats.norm.pdf(lam, mu, sd)
    sig2 = np.exp(-lam)
    w_ex = 2 / dl * (sig2 + 0.25) / 0.25
    stated = stats.norm.pdf(lam, mu, sd) * (np.exp(-lam) + 0.5 ** 2)
    r = w_ex / stated
    res_B[f"edm({Pm},{Ps})"] = {"ratio_exact_over_stated_mean": float(r.mean()), "ratio_rel_spread": float(r.std() / r.mean())}

# LDM-linear: paper gives no closed form. LDM 'scaled linear' beta(sqrt-linear), T=1000, beta0=0.00085, betaT=0.012
# (the values of the public SD 1.x config; an assumption, the paper does not state them)
T = 1000
beta = np.linspace(np.sqrt(0.00085), np.sqrt(0.012), T) ** 2
abar = np.cumprod(1 - beta)
a_l, b_l = np.sqrt(abar), np.sqrt(1 - abar)
lam_l = 2 * np.log(a_l / b_l)
tt = np.arange(T) / (T - 1)
dl_l = np.gradient(lam_l, tt)
w_l = 2 / np.abs(dl_l)           # eps-prediction, unit weight
res_B["ldm_linear_eps"] = {"lambda_range": [float(lam_l.min()), float(lam_l.max())],
                           "note": "SNR range of the discrete schedule; endpoints are not exactly 0/1 noise"}
out["B_weights"] = res_B

# ---------------------------------------------------------------- C. densities
def pi_ln(t, m, s):
    return np.exp(-(np.log(t / (1 - t)) - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi) * t * (1 - t))

def f_mode(u, s):
    return 1 - u - s * (np.cos(np.pi / 2 * u) ** 2 - 1 + u)

def df_mode(u, s):
    return -1 - s * (1 - np.pi / 2 * np.sin(np.pi * u))

def pi_mode_of_u(u, s):  # density value at t=f(u)
    return 1 / np.maximum(np.abs(df_mode(u, s)), 1e-9)

def mode_mass(s, bins_):
    """exact mass of t=f(u), u~U(0,1), in t-bins (f is monotone decreasing, so the mass is a u-length)."""
    u = np.linspace(0, 1, 2000001)
    f = f_mode(u, s)
    inv = lambda x: np.interp(x, f[::-1], u[::-1])
    return [float(abs(inv(hi) - inv(lo))) for lo, hi in bins_]

def pi_cos(t):
    return 2 / (np.pi - 2 * np.pi * t + 2 * np.pi * t ** 2)

def pi_mode_on_grid(tgrid, s, n=400001):
    u = np.linspace(0, 1, n)
    f = f_mode(u, s)
    d = pi_mode_of_u(u, s)
    order = np.argsort(f)
    return np.interp(tgrid, f[order], d[order])

C = {}
quad = lambda f, lo=0, hi=1: integrate.quad(f, lo, hi, limit=400, points=[0.5])[0]
tg = np.linspace(1e-6, 1 - 1e-6, 2000001)
bins = [(0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1)]
def mass_bins(pdf_grid):
    return [float(_trap(pdf_grid[(tg >= lo) & (tg <= hi)], tg[(tg >= lo) & (tg <= hi)])) for lo, hi in bins]

for m, s in [(0, 1), (0.5, 1), (-0.5, 1), (0.5, 0.6), (1, 0.6)]:
    pdf = pi_ln(tg, m, s)
    C[f"lognorm({m},{s})"] = {"integral": float(_trap(pdf, tg)), "mode_t": float(tg[np.argmax(pdf)]),
                              "mean_t": float(_trap(tg * pdf, tg)), "mass_bins": mass_bins(pdf),
                              "ks_p": float(stats.kstest(1 / (1 + np.exp(-rng.normal(m, s, 20000))),
                                                          lambda x: stats.norm.cdf(np.log(x / (1 - x)), m, s)).pvalue)}
# mode sampling: valid range of s
uu = np.linspace(0, 1, 200001)
s_lim = 2 / (np.pi - 2)
mono = {}
for s in [-1.2, -1.0, -0.5, 0.0, 1.0, 1.29, 1.75, s_lim, 1.76, 2.0]:
    d = df_mode(uu, s)
    mono[str(round(s, 4))] = {"monotone_decreasing": bool(np.all(d <= 1e-12)), "max_df": float(d.max())}
C["mode_s_limit_formula_2/(pi-2)"] = s_lim
C["mode_monotone_check"] = mono
for s in [-1.0, 0.0, 1.0, 1.29, 1.75]:
    pdf = pi_mode_on_grid(tg, s)
    C[f"mode({s})"] = {"integral": float(_trap(pdf, tg)), "pi_at_0": (float(1 / (1 + s)) if s != -1 else float("inf")),
                       "pi_at_0.5": float(pi_mode_of_u(0.5, s)), "peak_pi": float(pdf.max()),
                       "mass_bins": mode_mass(s, bins), "mass_within_0.05_of_0.5": mode_mass(s, [(.45, .55)])[0]}
    # sampling check: t=f(u), u~U vs the analytic density (KS on CDF built from the grid)
    cdf = np.cumsum(pdf) * (tg[1] - tg[0])
    t_s = f_mode(rng.uniform(size=20000), s)
    C[f"mode({s})"]["ks_p"] = float(stats.kstest(t_s, lambda x: np.interp(x, tg, cdf / cdf[-1])).pvalue)
# CosMap
pdf = pi_cos(tg)
u_s = rng.uniform(size=20000)
t_s = 1 - 1 / (np.tan(np.pi / 2 * u_s) + 1)
lam_c = 2 * np.log(np.cos(np.pi / 2 * u_s) / np.sin(np.pi / 2 * u_s))
lam_r = 2 * np.log((1 - t_s) / t_s)
cdf = np.cumsum(pdf) * (tg[1] - tg[0])
C["cosmap"] = {"integral": float(_trap(pdf, tg)), "mode_t": float(tg[np.argmax(pdf)]), "pi_at_0": float(pi_cos(0.0)),
               "pi_at_0.5": float(pi_cos(0.5)), "mass_bins": mass_bins(pdf),
               "max_abs_logsnr_mismatch": float(np.abs(lam_c - lam_r).max()),
               "ks_p": float(stats.kstest(t_s, lambda x: np.interp(x, tg, cdf)).pvalue)}
C["uniform"] = {"mass_bins": [hi - lo for lo, hi in bins]}
out["C_densities"] = C

# ---------------------------------------------------------------- D. equivalent RF density of every formulation
tt = np.linspace(0.001, 0.999, 999)
a, b, da, db = sched_rf(tt)
lam_rf = 2 * np.log((1 - tt) / tt)
def to_pi_eff(w_of_lam_on_lamgrid):
    """per-d(lambda) weight w(lam) -> RF timestep density: w = t/(1-t) * pi."""
    w = w_of_lam_on_lamgrid
    pi = w * (1 - tt) / tt
    pi = pi / _trap(pi, tt)
    return pi

D = {}
forms = {}
# eps/linear: interpolate the LDM schedule's weight on the RF lambda grid (lam_l decreasing in index)
w_lin = np.interp(lam_rf, lam_l[::-1], w_l[::-1], left=np.nan, right=np.nan)
forms["eps/linear"] = to_pi_eff(np.nan_to_num(w_lin, nan=0.0))
ac, bc, dac, dbc = sched_cos(tt)
lam_c = 2 * np.log(ac / bc)
forms["eps/cos"] = to_pi_eff(1 / np.cosh(lam_rf / 2))
forms["v/cos"] = to_pi_eff(np.exp(-lam_rf / 2))
for Pm, Ps in [(-1.2, 1.2), (0.0, 0.6)]:
    mu, sd = -2 * Pm, 2 * Ps
    forms[f"edm({Pm},{Ps})"] = to_pi_eff(stats.norm.pdf(lam_rf, mu, sd) * (np.exp(-lam_rf) + 0.25))
forms["rf (uniform)"] = np.ones_like(tt)
forms["rf/lognorm(0,1)"] = pi_ln(tt, 0, 1)
forms["rf/lognorm(0.5,0.6)"] = pi_ln(tt, 0.5, 0.6)
forms["rf/mode(1.29)"] = pi_mode_on_grid(tt, 1.29)
forms["rf/cosmap"] = pi_cos(tt)
for k, v in forms.items():
    v = v / _trap(v, tt)
    forms[k] = v
    cum = np.cumsum(v) * (tt[1] - tt[0])
    D[k] = {"mass_t_0.3_0.7": float(_trap(v[(tt >= .3) & (tt <= .7)], tt[(tt >= .3) & (tt <= .7)])),
            "median_t": float(tt[np.searchsorted(cum, 0.5)]), "mode_t": float(tt[np.argmax(v)])}
out["D_equivalent_rf_density"] = D
(HERE.parent / ".scratch").mkdir(exist_ok=True)
np.savez(HERE.parent / ".scratch" / "pi_eff.npz", t=tt, **{k.replace("/", "_"): v for k, v in forms.items()})

(HERE / "loss_weighting_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({"A": out["A_identity"], "B": out["B_weights"], "D": D}, indent=1)[:6000])
print({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("integral", "mode_t", "ks_p", "pi_at_0.5", "peak_pi")}) for k, v in C.items() if k != "mode_monotone_check"})
print(mono)
