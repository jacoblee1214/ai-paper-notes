"""Exact-velocity toy for rectified flow (arXiv:2403.03206 sec 3.1 and 5.3.2). NO training.

Data: 2D Gaussian mixture c (3 isotropic components, rescaled to unit per-coordinate variance), replicated r times
to n = 2r coordinates ("pixel-like": every pixel of a block carries the same value).  z_t = (1-t) x0 + t eps.
Sufficient statistic y = block mean: y = (1-t) c + (t s) eta, s = 1/sqrt(r).  All of the velocity is predictable
orthogonally to the block mean (z_perp = t eps_perp), so everything non-trivial lives in y.
Closed forms used:
   E[x0 | z]   from GMM posterior,  u*(z,t) = E[eps - x0 | z] = (z - E[x0|z]) / t
   irreducible velocity loss  E||v - u*||^2 = MMSE(x0|z_t) / t^2
Outputs toy_results.json and .scratch/toy_curves.npz (for plot_figures.py).
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(0)
_trap = getattr(np, "trapezoid", None) or np.trapz

K = 3
ang = 2 * np.pi * np.arange(K) / K + 0.3
MU = np.stack([np.cos(ang), np.sin(ang)], 1) * 1.0
S_SIGMA = 0.25
raw_var = (MU ** 2).mean() + S_SIGMA ** 2                 # per-coordinate variance (mixture mean is 0)
SC = 1 / np.sqrt(raw_var)
MU = MU * SC
S2 = (S_SIGMA * SC) ** 2                                  # component covariance S2 * I
W = np.ones(K) / K

def sample_c(n):
    k = rng.integers(0, K, n)
    return MU[k] + np.sqrt(S2) * rng.normal(size=(n, 2))

def posterior(y, t, s):
    """responsibilities, E[c|y], E[s*eta|y] for y=(1-t)c + t s eta."""
    var = (1 - t) ** 2 * S2 + (t * s) ** 2                # scalar (isotropic)
    d = y[:, None, :] - (1 - t) * MU[None]                # n,K,2
    logp = -0.5 * (d ** 2).sum(-1) / var - np.log(var) + np.log(W)[None]
    logp -= logp.max(1, keepdims=True)
    r = np.exp(logp); r /= r.sum(1, keepdims=True)
    ec_k = MU[None] + ((1 - t) * S2 / var) * d
    ee_k = (t * s ** 2 / var) * d
    return r, (r[..., None] * ec_k).sum(1), (r[..., None] * ee_k).sum(1)

def u_star(y, t, s):
    _, ec, ee = posterior(y, t, s)
    return ee - ec

# ------------------------------------------------------------------ 1. irreducible floor and local complexity
N_MC = 60000
tgrid = np.linspace(0.01, 0.99, 99)
curves = {"t": tgrid}
summary = {"unit_var_check": float((sample_c(200000) ** 2).mean()), "mixture": {"K": K, "sigma_comp": float(np.sqrt(S2)),
           "radius": float(np.linalg.norm(MU[0]))}}
c0 = sample_c(N_MC); eta0 = rng.normal(size=(N_MC, 2))
for r in [1, 4, 16, 64]:
    s = 1 / np.sqrt(r)
    mmse, floor_per_pixel, jac, mmse_ratio = [], [], [], []
    for t in tgrid:
        y = (1 - t) * c0 + t * s * eta0
        _, ec, _ = posterior(y, t, s)
        m = ((c0 - ec) ** 2).sum(1).mean()                 # MMSE of c (2 coords)
        mmse.append(m)
        floor_per_pixel.append(m / (2 * t ** 2))           # E||v-u*||^2 per pixel (r copies of each coordinate cancel)
        h = 1e-4
        J = np.stack([(u_star(y + h * np.eye(2)[j], t, s) - u_star(y - h * np.eye(2)[j], t, s)) / (2 * h) for j in range(2)], -1)
        jac.append(np.sqrt((J ** 2).sum((1, 2)).mean()))
    mmse = np.array(mmse)
    # MMSE at t=1 equals Var(c) (2 coords); at t->0 it vanishes
    curves[f"mmse_r{r}"] = mmse / (2.0)                     # per coordinate, in units of Var(c)=1
    curves[f"floor_r{r}"] = np.array(floor_per_pixel)
    curves[f"jac_r{r}"] = np.array(jac)
    R = mmse / 2.0
    # transition interval: posterior variance fraction in [0.1, 0.9]
    def cross(level):
        i = np.where(R >= level)[0][0]
        return float(np.interp(level, [R[i - 1], R[i]], [tgrid[i - 1], tgrid[i]]))
    t10, t50, t90 = cross(0.1), cross(0.5), cross(0.9)
    peak_floor = float(tgrid[np.argmax(floor_per_pixel)])
    summary[f"r{r}"] = {"t_mmse_0.1": t10, "t_mmse_0.5": t50, "t_mmse_0.9": t90,
                        "t50_over_(1-t50)": t50 / (1 - t50), "floor_peak_t": peak_floor,
                        "floor_peak_value": float(np.max(floor_per_pixel)),
                        "jac_peak_t": float(tgrid[np.argmax(jac)])}
# exact shift prediction: t50 / (1-t50) scales as sqrt(r) because MMSE depends on sqrt(r)(1-t)/t only
base = summary["r1"]["t50_over_(1-t50)"]
summary["shift_check"] = {f"r{r}": {"measured_ratio": summary[f"r{r}"]["t50_over_(1-t50)"] / base, "sqrt_r": float(np.sqrt(r))} for r in [4, 16, 64]}
# eq (23) applied to t50(r=1) gives t50(r)?
def shift(t, a):
    return a * t / (1 + (a - 1) * t)
summary["eq23_prediction_of_t50"] = {f"r{r}": {"pred": float(shift(summary["r1"]["t_mmse_0.5"], np.sqrt(r))),
                                                  "measured": summary[f"r{r}"]["t_mmse_0.5"]} for r in [4, 16, 64]}

# ------------------------------------------------------------------ 2. mass of timestep densities in the transition region
def pi_ln_cdf(t, m=0.0, s_=1.0):
    from scipy.stats import norm
    return norm.cdf((np.log(t / (1 - t)) - m) / s_)
def mass_interval(cdf, lo, hi):
    return float(cdf(hi) - cdf(lo))
def shifted_cdf(cdf, a):
    # t_m = shift(t_n, a): P(t_m <= x) = P(t_n <= x/(a - (a-1) x))
    return lambda x: cdf(x / (a - (a - 1) * x))
uni = lambda x: x
ln = lambda x: pi_ln_cdf(x)
mass = {}
for r in [1, 4, 16, 64]:
    lo, hi = summary[f"r{r}"]["t_mmse_0.1"], summary[f"r{r}"]["t_mmse_0.9"]
    a = np.sqrt(r)
    mass[f"r{r}"] = {"interval": [lo, hi],
                     "uniform": mass_interval(uni, lo, hi), "lognorm(0,1)": mass_interval(ln, lo, hi),
                     "uniform_shifted_sqrt_r": mass_interval(shifted_cdf(uni, a), lo, hi),
                     "lognorm(0,1)_shifted_sqrt_r": mass_interval(shifted_cdf(ln, a), lo, hi),
                     "lognorm(0,1)_shifted_3": mass_interval(shifted_cdf(ln, 3.0), lo, hi),
                     "uniform_shifted_3": mass_interval(shifted_cdf(uni, 3.0), lo, hi)}
summary["mass_in_transition_interval"] = mass

# ------------------------------------------------------------------ 3. Euler discretisation error with the exact velocity
def integrate_euler(y1, s, ts):
    y = y1.copy()
    for i in range(len(ts) - 1):
        y = y + (ts[i + 1] - ts[i]) * u_star(y, ts[i], s)
    return y

def integrate_rk4(y1, s, n=400):
    y = y1.copy(); ts = np.linspace(1, 0, n + 1)
    for i in range(n):
        h = ts[i + 1] - ts[i]; t0 = ts[i]
        k1 = u_star(y, t0, s); k2 = u_star(y + .5 * h * k1, t0 + .5 * h, s)
        k3 = u_star(y + .5 * h * k2, t0 + .5 * h, s); k4 = u_star(y + h * k3, t0 + h, s)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

def sliced_w1(a, b, n_dir=16):
    d = rng.normal(size=(n_dir, 2)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.mean([np.abs(np.sort(a @ v) - np.sort(b @ v)).mean() for v in d]))

NS = 3000
eta = rng.normal(size=(NS, 2))
data_ref = sample_c(NS)
steps_list = [2, 3, 4, 6, 8, 12, 16, 24]
alphas = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
euler = {}
for r in [1, 4, 16, 64]:
    s = 1 / np.sqrt(r)
    y1 = s * eta
    ref = integrate_rk4(y1, s, 400)
    ref_chk = integrate_rk4(y1, s, 200)
    rec = {"rk4_400_vs_200_rms": float(np.sqrt(((ref - ref_chk) ** 2).sum(1).mean()) / np.sqrt(2)),
           "sliced_w1_ref_vs_data": sliced_w1(ref, data_ref), "table": {}}
    best = {}
    for N in steps_list:
        row = {}
        for a in alphas:
            ts = shift(np.linspace(1, 0, N + 1), a)
            yN = integrate_euler(y1, s, ts)
            row[str(a)] = {"rms": float(np.sqrt(((yN - ref) ** 2).sum(1).mean() / 2)), "sw1": sliced_w1(yN, data_ref)}
        rec["table"][str(N)] = row
        best[str(N)] = {"best_alpha_rms": min(row, key=lambda k: row[k]["rms"]), "best_alpha_sw1": min(row, key=lambda k: row[k]["sw1"])}
    rec["best"] = best
    euler[f"r{r}"] = rec
summary["euler"] = euler

(HERE / "toy_results.json").write_text(json.dumps(summary, indent=1))
(HERE.parent / ".scratch").mkdir(exist_ok=True)
np.savez(HERE.parent / ".scratch" / "toy_curves.npz", **curves)
print(json.dumps({k: summary[k] for k in ["unit_var_check", "r1", "r4", "r16", "r64", "shift_check", "eq23_prediction_of_t50", "mass_in_transition_interval"]}, indent=1))
for r in euler:
    print(r, "rk4 self-check", euler[r]["rk4_400_vs_200_rms"], "ref-vs-data sw1", euler[r]["sliced_w1_ref_vs_data"])
    for N in ["2", "4", "8", "16"]:
        print(" N", N, {a: round(v["rms"], 4) for a, v in euler[r]["table"][N].items()}, euler[r]["best"][N])
