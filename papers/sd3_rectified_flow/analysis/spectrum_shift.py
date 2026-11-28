"""Does the sqrt(m/n) shift of eq (23) survive outside the 'constant image' idealisation?  No training.

Stationary Gaussian random field on an N x N latent grid with a power-law spectrum P(k) ~ (k0^2+|k|^2)^(-beta/2),
the SAME continuous spectrum at every resolution (a mode's eigenvalue of the pixel covariance is N^2 * c * P(k)).
For a Gaussian x0, z_t = (1-t)x0 + t eps decouples per Fourier mode, MMSE_k = lam t^2 / ((1-t)^2 lam + t^2).
We find t where the pixel-averaged MMSE falls to a given fraction of its t=1 value and compare t/(1-t) across N
with eq (23)'s prediction (ratio = N / N0 = sqrt(m/n)).
"""
import json
import pathlib

import numpy as np
from scipy.optimize import brentq

HERE = pathlib.Path(__file__).resolve().parent
N0 = 32
Ns = [32, 64, 128, 256]
out = {}

def eig(N, beta, k0=1.0, c=None):
    k = np.fft.fftfreq(N, 1 / N)
    kk = k[:, None] ** 2 + k[None, :] ** 2
    P = (k0 ** 2 + kk) ** (-beta / 2)
    return P

for beta in [1.0, 2.0, 2.5, 3.0, 4.0]:
    P0 = eig(N0, beta)
    c = 1.0 / P0.sum()                       # per-pixel variance 1 at the base resolution
    rec = {}
    t50 = {}
    for N in Ns:
        lam = N ** 2 * c * eig(N, beta).ravel()
        tot = lam.sum() / N ** 2
        def frac(t):
            return (lam * t ** 2 / ((1 - t) ** 2 * lam + t ** 2)).sum() / N ** 2 / tot
        ts = {lvl: brentq(lambda t: frac(t) - lvl, 1e-6, 1 - 1e-9) for lvl in (0.1, 0.5, 0.9)}
        rec[str(N)] = {"pixel_var": float(tot), "t": ts, "odds_t50": ts[0.5] / (1 - ts[0.5])}
        t50[N] = ts
    for N in Ns:
        rec[str(N)]["odds_ratio_vs_N0"] = rec[str(N)]["odds_t50"] / rec[str(N0)]["odds_t50"]
        rec[str(N)]["eq23_ratio_sqrt_m_over_n"] = N / N0
        rec[str(N)]["odds_ratio_t10"] = (t50[N][0.1] / (1 - t50[N][0.1])) / (t50[N0][0.1] / (1 - t50[N0][0.1]))
        rec[str(N)]["odds_ratio_t90"] = (t50[N][0.9] / (1 - t50[N][0.9])) / (t50[N0][0.9] / (1 - t50[N0][0.9]))
    out[f"beta={beta}"] = rec
(HERE / "spectrum_shift_results.json").write_text(json.dumps(out, indent=1))
for b, rec in out.items():
    print(b, {N: (round(v["odds_ratio_vs_N0"], 3), round(v["odds_ratio_t10"], 3), round(v["odds_ratio_t90"], 3), v["eq23_ratio_sqrt_m_over_n"], round(v["pixel_var"], 3)) for N, v in rec.items()})
