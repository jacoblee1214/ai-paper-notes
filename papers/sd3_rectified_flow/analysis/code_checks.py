"""Arithmetic replica (numpy, no model, no repo import) of the SD3.5 reference sampler's sigma schedule, compared with eq (23).

The reference code (Stability-AI/sd3.5, sd3_impls.py ModelSamplingDiscreteFlow and sd3_infer.py get_sigmas) was read statically;
this script only re-does its few lines of scalar arithmetic.  Writes code_checks_results.json.
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

def sigma_fn(ts, shift):                       # ModelSamplingDiscreteFlow.sigma
    t = np.asarray(ts, dtype=np.float64) / 1000.0
    return t if shift == 1.0 else shift * t / (1 + (shift - 1) * t)

def get_sigmas(steps, shift):                  # SD3.get_sigmas: linspace in 'timestep' space, then sigma(), then 0
    table = sigma_fn(np.arange(1, 1001), shift)          # self.sigmas
    start, end = table[-1] * 1000, table[0] * 1000       # timestep(sigma_max), timestep(sigma_min)
    ts = np.linspace(start, end, steps)
    return np.concatenate([sigma_fn(ts, shift), [0.0]]), table

def eq23(t_n, ratio):                          # ratio = sqrt(m/n)
    return ratio * t_n / (1 + (ratio - 1) * t_n)

out = {}
for shift in (1.0, 3.0):
    for steps in (4, 28, 40, 50):
        s, table = get_sigmas(steps, shift)
        u = np.linspace(1.0, table[0] * 1.0, steps)      # the uniform grid the sigmas came from (t = ts/1000)
        out[f"shift={shift},steps={steps}"] = {
            "sigma_first": float(s[0]), "sigma_last_nonzero": float(s[-2]), "table_sigma_min": float(table[0]),
            "last_nonzero_over_table_min": float(s[-2] / table[0]),
            "max_abs_diff_to_eq23_on_same_grid": float(np.abs(s[:-1] - eq23(np.linspace(1000, table[0] * 1000, steps) / 1000, shift)).max()),
            "n_sigmas_in_[0.5,1]": int((s[:-1] >= 0.5).sum()), "first_4": [float(x) for x in s[:4]]}
# shift needed by eq (23) when only the resolution changes (m/n pixel ratio)
out["sqrt_m_over_n"] = {"256->512": 2.0, "256->1024": 4.0, "512->1024": 2.0, "341->1024 (alpha=3)": 3.0,
                        "pixels_n_for_alpha3_at_1024": float(1024 ** 2 / 9), "side_for_alpha3": float(1024 / 3)}
# log-SNR shift of the schedule: lambda(t_m) = lambda(t_n) - log(m/n)  (eq 25), for alpha = sqrt(m/n)
t = np.linspace(0.05, 0.95, 10)
for a in (2.0, 3.0):
    lam = lambda x: 2 * np.log((1 - x) / x)
    out[f"eq25_alpha={a}"] = {"max_abs_err": float(np.abs(lam(eq23(t, a)) - (lam(t) - 2 * np.log(a))).max())}
(HERE / "code_checks_results.json").write_text(json.dumps(out, indent=1))
for k in ["shift=3.0,steps=40", "shift=3.0,steps=4", "shift=1.0,steps=40"]:
    print(k, out[k])
print(out["eq25_alpha=3.0"], out["sqrt_m_over_n"])
