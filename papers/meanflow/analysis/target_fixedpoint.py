"""MeanFlow 목표(식 10·11)의 고정점, 섭동에 대한 반응, (r,t) 표집 분포와 반경사(semi-gradient) 학습의 안정성.
학습은 하지 않는다. 섭동·특징은 무작위로 고정한 함수다. (이 파일의 안정성 분석은 이 저장소의 자체 유도이며 논문에는 없다.)"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.func import jvp
from mf_exact import GMM, u_rk4, full

OUT = Path(__file__).with_name("target_fixedpoint_results.json")
gen = torch.Generator().manual_seed(3)
ang = torch.arange(3) * 2 * torch.pi / 3
gm = GMM(3.0 * torch.stack([ang.cos(), ang.sin()], 1), s=0.5)
rms = lambda a: float(a.pow(2).mean().sqrt())
res = {}

# 고정 무작위 섭동 g(z,r,t) (학습하지 않은 임의의 매끈한 함수)
W1 = torch.randn(16, 4, generator=gen) * 1.0
b1 = torch.randn(16, generator=gen)
W2 = torch.randn(2, 16, generator=gen) / 4.0


def g(z, r, t):
    return torch.tanh(torch.cat([z, r[:, None], t[:, None]], 1) @ W1.T + b1) @ W2.T


def make_u(delta, N=64):
    return lambda z, r, t: u_rk4(gm.v, z, r, t, N=N) + delta * g(z, r, t)


def target_marginal(u_fn, z, r, t):
    """식 10: u_tgt = v - (t-r) JVP(u_fn; (v,0,1)), v 는 주변 속도"""
    v = gm.v(z, t)
    _, jv = jvp(u_fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    return v - (t - r)[:, None] * jv


def target_cond(u_fn, z, r, t, vt):
    """식 11: 접선과 첫 항 모두 표본 조건부 속도 vt"""
    _, jv = jvp(u_fn, (z, r, t), (vt, torch.zeros_like(r), torch.ones_like(t)))
    return vt - (t - r)[:, None] * jv


# --- 1. 정확한 u 는 고정점 ---------------------------------------------
B = 512
r0, t0 = 0.1, 0.9
zt, *_ = gm.sample_zt(B, torch.tensor(t0), gen)
rr, tt = full(B, r0), full(B, t0)
u_star = make_u(0.0, N=128)
u0 = u_star(zt, rr, tt)
res["fixed_point_exact_u"] = {"rms_|T[u*]-u*|": rms(target_marginal(u_star, zt, rr, tt) - u0), "rms_u": rms(u0)}

# --- 2. 섭동: T[u*+e] - u* = -(t-r) D e (정확히 선형), 반응 비율 ---------
rows = {}
for delta in [0.01, 0.1, 1.0]:
    u_th = make_u(delta, N=128)
    e = delta * g(zt, rr, tt)
    err_tgt = target_marginal(u_th, zt, rr, tt) - u0
    # -(t-r) D e 를 직접 계산
    _, De = jvp(lambda z, r, t: delta * g(z, r, t), (zt, rr, tt), (gm.v(zt, tt), torch.zeros_like(rr), torch.ones_like(tt)))
    pred = -(t0 - r0) * De
    rows[str(delta)] = {"rms_e": rms(e), "rms_target_error": rms(err_tgt),
                        "ratio_target_err_over_e": rms(err_tgt) / rms(e),
                        "max|T-u* - (-(t-r)De)|": float((err_tgt - pred).abs().max())}
res["perturbation_response_random_g"] = rows

# --- 3. 표본 조건부 목표(식 11)의 조건부 기댓값 = 주변 목표(식 10) -----
def posterior_sample(z, t, M, gen):
    """p(x, eps | z_t = z): 성분 책임도 -> x|z,k 가우시안 -> eps = (z - (1-t)x)/t. z: (n,d) 고정, 각 z마다 M개"""
    n, d = z.shape
    tt_ = torch.full((n,), t)
    vk, logw = gm.comp_vel(z, tt_)
    w = torch.softmax(logw, 1)
    k = torch.multinomial(w, M, replacement=True, generator=gen)       # (n, M)
    sig2 = gm.sig2(torch.tensor(t))
    a = (1 - t) * gm.s ** 2 / sig2
    mk = gm.m[k]                                                         # (n, M, d)
    mean = mk + a * (z[:, None, :] - (1 - t) * mk)
    var = gm.s ** 2 * t ** 2 / sig2
    x = mean + var ** 0.5 * torch.randn(n, M, d, generator=gen)
    eps = (z[:, None, :] - (1 - t) * x) / t
    return x, eps


n, M = 6, 20000
t1, r1 = 0.6, 0.1
z, *_ = gm.sample_zt(n, torch.tensor(t1), gen)
x, eps = posterior_sample(z, t1, M, gen)
vt = (eps - x).reshape(-1, 2)
zz = z[:, None, :].expand(n, M, 2).reshape(-1, 2)
rr1, tt1 = full(n * M, r1), full(n * M, t1)
v_marg = gm.v(z, full(n, t1))
mc_v = vt.reshape(n, M, 2).mean(1)
u_th = make_u(0.5, N=32)
tc = target_cond(u_th, zz, rr1, tt1, vt).reshape(n, M, 2)
tm = target_marginal(u_th, z, full(n, r1), full(n, t1))
se = tc.std(1) / M ** 0.5
res["conditional_vs_marginal_target"] = {
    "n_z": n, "M_posterior_samples_per_z": M, "seed": 3,
    "max|E[vt|z]-v|": float((mc_v - v_marg).abs().max()),
    "max|E[target_cond|z]-target_marg|": float((tc.mean(1) - tm).abs().max()),
    "max_z_score": float(((tc.mean(1) - tm).abs() / se).max()),
    "mean_per_sample_target_std": float(tc.std(1).mean()),
}

# --- 4. 주파수별 반응: 점별 비율 rho=|A e|/|e|, 반경사 이차형식 q=<e,(I-A)e>/<e,e> ------
# A e = -(t-r) D e.  I-A = I + (t-r) D.  분포는 훈련 분포(주변 z_t ~ p_t, (r,t)~표집기).
def lognorm(n, mu, sig, gen):
    return torch.sigmoid(torch.randn(n, generator=gen) * sig + mu)


def sample_rt(n, mu, sig, gen, ratio_neq=0.25, uniform=False):
    if uniform:
        a, b = torch.rand(n, generator=gen), torch.rand(n, generator=gen)
    else:
        a, b = lognorm(n, mu, sig, gen), lognorm(n, mu, sig, gen)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    eq = torch.rand(n, generator=gen) >= ratio_neq
    r = torch.where(eq, t, r)
    return r, t


def draw(n, sampler, gen):
    r, t = sampler(n, gen)
    comp = torch.multinomial(gm.pi, n, replacement=True, generator=gen)
    x, _ = gm.sample_data(n, gen, comp)
    eps = torch.randn(n, 2, generator=gen)
    z = (1 - t[:, None]) * x + t[:, None] * eps
    return z, r, t


samplers = {
    "lognorm(-0.4,1.0) r!=t 25% (논문 기본)": lambda n, g_: sample_rt(n, -0.4, 1.0, g_, 0.25),
    "uniform r!=t 25%": lambda n, g_: sample_rt(n, 0, 0, g_, 0.25, uniform=True),
    "lognorm(-2,2) r!=t 75% (CIFAR)": lambda n, g_: sample_rt(n, -2.0, 2.0, g_, 0.75),
    "lognorm(-0.4,1.0) r!=t 100%": lambda n, g_: sample_rt(n, -0.4, 1.0, g_, 1.0),
}
freq = {}
Nn = 200000
for sname, smp in samplers.items():
    z, r, t = draw(Nn, smp, gen)
    v = gm.v(z, t)
    tau = t - r
    out = {}
    for om in [0.5, 1, 2, 4, 8, 16, 32]:
        # (a) tau 방향 파동 e = sin(om (t-r))
        e = torch.sin(om * tau)
        De = om * torch.cos(om * tau)                        # d/dt (r 고정, z 무관)
        # (b) z 방향 파동 e = sin(om z_1): D e = om v_1 cos(om z_1)
        e2 = torch.sin(om * z[:, 0])
        De2 = om * v[:, 0] * torch.cos(om * z[:, 0])
        neq = tau > 0
        out[str(om)] = {
            "tau_wave": {"rho_rms": float(((tau * De) ** 2)[neq].mean().sqrt() / (e[neq] ** 2).mean().sqrt()),
                         "q": float((e * (e + tau * De))[neq].mean() / (e[neq] ** 2).mean())},
            "z_wave": {"rho_rms": float(((tau * De2) ** 2)[neq].mean().sqrt() / (e2[neq] ** 2).mean().sqrt()),
                       "q": float((e2 * (e2 + tau * De2))[neq].mean() / (e2[neq] ** 2).mean())},
        }
    freq[sname] = out
res["frequency_response_(rho=pointwise contraction, q=semi-gradient quadratic form)"] = freq

# --- 5. 표집 밀도 조건: w(t|r) - (t-r) dw/dt > 0 인가 ------------------
def logit_normal_logpdf_and_grad(t, mu, sig):
    lg = torch.log(t) - torch.log1p(-t)
    # log f = -(lg-mu)^2/(2 sig^2) - log t - log(1-t) + const ; d/dt
    dlg = 1 / t + 1 / (1 - t)
    dlogf = -(lg - mu) / sig ** 2 * dlg - 1 / t + 1 / (1 - t)
    return dlogf


cond = {}
n = 400000
for name, (mu, sig) in {"lognorm(-0.4,1.0)": (-0.4, 1.0), "lognorm(-0.2,1.0)": (-0.2, 1.0),
                        "lognorm(-0.4,1.2)": (-0.4, 1.2), "lognorm(-2,2)": (-2.0, 2.0), "uniform": (None, None)}.items():
    if mu is None:
        a, b = torch.rand(n, generator=gen), torch.rand(n, generator=gen)
        dlogf = lambda t: torch.zeros_like(t)
    else:
        a, b = lognorm(n, mu, sig, gen), lognorm(n, mu, sig, gen)
        dlogf = lambda t: logit_normal_logpdf_and_grad(t, mu, sig)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    keep = (t - r) > 1e-6
    t, r = t[keep], r[keep]
    coef = 1 - (t - r) * dlogf(t)        # >0 이면 이 (r,t)에서 이차형식 기여가 양
    cond[name] = {"P[coef<=0] (r!=t 쌍 기준)": float((coef <= 0).float().mean()),
                  "min_coef": float(coef.min()),
                  "mean_t_of_pair": float(t.mean()),
                  "mean_t_of_max_of_two(r=t 표본)": float(torch.maximum(lognorm(n, mu, sig, gen), lognorm(n, mu, sig, gen)).mean()) if mu is not None else 2 / 3,
                  "mean_t_single_draw": float(lognorm(n, mu, sig, gen).mean()) if mu is not None else 0.5}
res["sampler_density_condition"] = cond

# --- 6. 무작위 특징 계의 반경사 동역학 행렬 M = E<f_i, f_j + (t-r) D f_j> -------------
def features(F, gen):
    w = torch.randn(F, 4, generator=gen) * 1.5
    b = torch.randn(F, generator=gen)
    c = torch.arange(F) % 2
    return w, b, c


def eval_feats(w, b, c, z, r, t, v):
    q = torch.cat([z, r[:, None], t[:, None]], 1) @ w.T + b          # (N, F)
    f = torch.tanh(q)
    df = (1 - f ** 2) * (v @ w[:, :2].T + w[:, 3][None])            # D f (r 고정): w_z.v + w_t
    N, F = f.shape
    fv = torch.zeros(N, F, 2); fv[:, torch.arange(F), c] = f          # 특징 j 의 출력 벡터 (N,F,2)
    dv = torch.zeros(N, F, 2); dv[:, torch.arange(F), c] = df
    return fv, dv


mat = {}
F = 48
for sname, smp in samplers.items():
    rows = []
    for rep in range(3):
        g_ = torch.Generator().manual_seed(100 + rep)
        w, b, c = features(F, g_)
        z, r, t = draw(60000, smp, g_)
        v = gm.v(z, t)
        fv, dv = eval_feats(w, b, c, z, r, t, v)
        tau = (t - r)[:, None, None]
        G = torch.einsum("nic,njc->ij", fv, fv) / len(z)
        Msg = torch.einsum("nic,njc->ij", fv, fv + tau * dv) / len(z)    # 반경사
        Mfull = torch.einsum("nic,njc->ij", fv + tau * dv, fv + tau * dv) / len(z)  # 이중 역전파(참고)
        ev = torch.linalg.eigvals(torch.linalg.solve(G + 1e-9 * torch.eye(F), Msg))
        sym = (Msg + Msg.T) / 2
        evs = torch.linalg.eigvalsh(torch.linalg.solve(torch.linalg.cholesky(G + 1e-9 * torch.eye(F)), sym) @ torch.linalg.inv(torch.linalg.cholesky(G + 1e-9 * torch.eye(F)).T))
        rows.append({"min_Re_eig(G^-1 M)": float(ev.real.min()), "max_|Im|": float(ev.imag.abs().max()),
                     "min_eig_sym_part": float(evs.min()), "max_eig": float(ev.real.max()),
                     "cond_number_Re": float(ev.real.max() / ev.real.min()) if ev.real.min() > 0 else None})
    mat[sname] = rows
res["random_feature_semigradient_matrix(F=48,N=60000,3 seeds)"] = mat

OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ["fixed_point_exact_u", "perturbation_response_random_g", "conditional_vs_marginal_target", "sampler_density_condition"]}, indent=1))
print(json.dumps(res["random_feature_semigradient_matrix(F=48,N=60000,3 seeds)"], indent=1))
for s, o in freq.items():
    print(s, {k: (round(v["tau_wave"]["rho_rms"], 2), round(v["tau_wave"]["q"], 2), round(v["z_wave"]["rho_rms"], 2), round(v["z_wave"]["q"], 2)) for k, v in o.items()})
