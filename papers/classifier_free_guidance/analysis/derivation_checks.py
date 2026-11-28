"""CFG 식 유도 검산(학습 없음). 결과는 derivation_checks_results.json. CPU 수 초."""
import json
import math

import numpy as np
import torch

import minimal_impl as mi

M = mi.default_toy()
R = {}
g = torch.Generator().manual_seed(0)
Z = 2.0 * torch.randn(400, 2, generator=g)

# D1. lambda 매개변수화: alpha^2 = sigmoid(lambda), d alpha / d lambda = alpha sigma^2 / 2, d sigma / d lambda = -alpha^2 sigma / 2
err1 = 0.0
for lam in (-10.0, -2.0, 0.0, 3.0, 12.0):
    l = torch.tensor(lam, requires_grad=True)
    a, s = mi.alpha_sigma(l)
    da = torch.autograd.grad(a, l, retain_graph=True)[0]
    ds = torch.autograd.grad(s, l)[0]
    err1 = max(err1, abs(da - a * s ** 2 / 2).item(), abs(ds + a ** 2 * s / 2).item(),
               abs(torch.log(a ** 2 / s ** 2) - lam).item())
R["D1_lambda_param_max_err"] = err1

# D2. eps* = -sigma grad log p (닫힌 식 대 자동미분)
err2 = 0.0
for lam in (-6.0, 0.0, 6.0):
    a, s = mi.alpha_sigma(lam)
    for c in (None, 0, 1, 2):
        z = Z.clone().requires_grad_(True)
        auto = torch.autograd.grad(M.log_p(z, a, s, c).sum(), z)[0]
        ex = (-s * auto - (M.eps_c(Z, lam, c) if c is not None else M.eps_u(Z, lam))).abs().max().item()
        err2 = max(err2, ex)
R["D2_eps_equals_minus_sigma_score_max_err"] = err2

# D3. 무조건 항: p_uncond 드롭아웃의 최소제곱 해 E[eps|z]는 eps_u이고 eps_u = sum_c p(c|z) eps_c
err3 = 0.0
for lam in (-3.0, 0.0, 4.0):
    a, s = mi.alpha_sigma(lam)
    post = torch.exp(M.log_post(Z, a, s))
    mix = sum(post[:, c:c + 1] * M.eps_c(Z, lam, c) for c in range(M.C))
    err3 = max(err3, (mix - M.eps_u(Z, lam)).abs().max().item())
R["D3_eps_u_equals_posterior_average_max_err"] = err3
# E[eps | z0] 의 중요도 표집 추정(자기정규화)과 eps_u 비교. 학습 목적 E||eps_theta(z,null) - eps||^2의 최소해가 eps_u임을 확인
mc = []
gen = torch.Generator().manual_seed(11)
cls = torch.multinomial(M.pi, 2_000_000, replacement=True, generator=gen)
comp = torch.multinomial(M.wm[cls], 1, generator=gen).squeeze(1)
X = M.mu[cls, comp] + math.sqrt(M.s2) * torch.randn(2_000_000, 2, generator=gen)
for lam, z0 in ((0.0, [1.0, 0.5]), (2.0, [-1.0, 1.5])):
    a, s = mi.alpha_sigma(lam)
    z0 = torch.tensor(z0)
    logw = -((z0[None] - a * X) ** 2).sum(1) / (2 * s ** 2)
    w_ = torch.softmax(logw, 0)
    eps_post = ((z0[None] - a * X) / s * w_[:, None]).sum(0)    # E[eps | z0] = E[(z0 - a x)/s | z0]
    ex = M.eps_u(z0[None], lam)[0]
    mc.append({"lam": lam, "mc": eps_post.tolist(), "exact": ex.tolist(), "ess": float(1 / (w_ ** 2).sum())})
R["D3_importance_check"] = mc

# D4. 암시적 분류기 항등식과 식 (6) = 분류기 안내(암시적 분류기)
err4, err5, err6 = 0.0, 0.0, 0.0
for lam in (-4.0, 0.0, 5.0):
    a, s = mi.alpha_sigma(lam)
    z = Z.clone().requires_grad_(True)
    lp = M.log_post(z, a, s)
    for c in range(M.C):
        gp = torch.autograd.grad(lp[:, c].sum(), z, retain_graph=True)[0]
        err4 = max(err4, (gp + (M.eps_c(Z, lam, c) - M.eps_u(Z, lam)) / s).abs().max().item())
        for w in (0.5, 2.0, 7.0):
            cg = M.eps_c(Z, lam, c) - w * s * gp
            err5 = max(err5, (cg - mi.eps_guided(M, Z, lam, c, w)).abs().max().item())
            gam = 1 + w
            err6 = max(err6, (M.eps_u(Z, lam) + gam * (M.eps_c(Z, lam, c) - M.eps_u(Z, lam))
                              - mi.eps_guided(M, Z, lam, c, w)).abs().max().item())
R["D4_implicit_classifier_identity_max_err"] = err4
R["D5_cfg_equals_classifier_guidance_max_err"] = err5
R["D6_gamma_convention_max_err"] = err6

# D7. DDIM의 한 걸음은 dz/dlam = (sigma^2/2) z - (sigma/2) eps 의 오일러(작은 dlam에서 2차 오차)
errs = []
for dl in (1e-1, 1e-2, 1e-3):
    lam = 0.5
    a, s = mi.alpha_sigma(lam)
    eps = M.eps_c(Z, lam, 0)
    a2, s2 = mi.alpha_sigma(lam + dl)
    ddim = a2 * ((Z - s * eps) / a) + s2 * eps
    euler = Z + dl * (s ** 2 / 2 * Z - s / 2 * eps)
    errs.append({"dlam": dl, "max_abs_diff": (ddim - euler).abs().max().item()})
R["D7_ddim_vs_ode_euler"] = errs

# D8. 가우시안 쌍: ODE 끝점 대 정확히 기울인 목표
rows = []
for name, (vc, vu) in {"equal_var": (0.4, 0.4), "uncond_wider": (0.4, 1.3), "uncond_narrower": (0.4, 0.25)}.items():
    P = mi.GaussPair([1.0, -0.5], vc, [0.0, 0.2], vu)
    for w in (0.0, 0.5, 1.0, 2.0, 4.0):
        m, v = P.ode_moments(w)
        t = P.tilted_moments(w)
        rows.append({"case": name, "w": w, "ode_mean": m.tolist(), "ode_var": v,
                     "target_mean": None if t is None else t[0].tolist(), "target_var": None if t is None else t[1],
                     "var_ratio_ode_over_target": None if t is None else v / t[1]})
R["D8_gaussian_pair"] = rows

# D9. 안내 점수 대 확산된 기울임의 정확한 점수(가우시안, uncond_wider): lambda별 최대 차
P = mi.GaussPair([1.0, -0.5], 0.4, [0.0, 0.2], 1.3)
w = 2.0
tm, tv = P.tilted_moments(w)
gaps = []
zz = 1.5 * torch.randn(2000, 2, generator=g)
for lam in (-12.0, -6.0, -3.0, -1.0, 0.0, 2.0, 4.0, 8.0, 14.0):
    a, s = mi.alpha_sigma(lam)
    true_eps = s * (zz - a * tm) / (a ** 2 * tv + s ** 2)
    cfg = (1 + w) * P.eps_c(zz, lam) - w * P.eps_u(zz, lam)
    gaps.append({"lam": lam, "rms_gap": ((cfg - true_eps) ** 2).mean().sqrt().item(),
                 "rms_true": (true_eps ** 2).mean().sqrt().item()})
R["D9_cfg_vs_diffused_tilted_eps"] = gaps

# D10. 안내 장의 포커-플랑크 잔차(toy GMM, 클래스 0): lambda, w별
zf = 1.5 * torch.randn(400, 2, generator=g)
tab = {}
for lam in (-3.0, 0.0, 3.0):
    tab[str(lam)] = {str(w): mi.fp_residual(M, zf, lam, 0, w)[0] for w in (0.0, 0.5, 1.0, 2.0, 4.0)}
R["D10_fokker_planck_residual_std"] = tab

# D11. 정확한 기울임 p(x|c)p(c|x)^w 의 모멘트(t=0, 격자): w 대 평균·분산·클래스 확신도
mom = []
for w in (0.0, 1.0, 2.0, 4.0, 8.0):
    tg = mi.TiltedGrid(M, 1, w)
    m, cov = tg.moments()
    one, zero = torch.tensor(1.0), torch.tensor(0.0)
    pc = (tg.p * torch.exp(M.log_post(tg.pts, one, zero)[:, 1])).sum().item()
    mom.append({"w": w, "mean": m.tolist(), "var_per_dim": cov.diagonal().mean().item(), "E_p_c_given_x": pc})
R["D11_exact_tilted_class1"] = mom

# D12. 포커-플랑크 잔차의 닫힌 식: R(z) = const(lambda) + (sigma^2/2) w (1+w) |grad_z log p_lambda(c|z)|^2
d12 = []
for lam in (-3.0, 0.0, 3.0):
    a, s = mi.alpha_sigma(lam)
    for w in (0.5, 2.0, 4.0):
        _, Rr = mi.fp_residual(M, zf, lam, 0, w)
        gn = ((M.eps_c(zf, lam, 0) - M.eps_u(zf, lam)) ** 2).sum(1) / s ** 2     # |grad log p(c|z)|^2
        pred = (s ** 2 / 2) * w * (1 + w) * gn
        d12.append({"lam": lam, "w": w, "max_abs_err": ((Rr - Rr.mean()) - (pred - pred.mean())).abs().max().item(),
                    "std_R": Rr.std().item()})
R["D12_fp_residual_closed_form"] = d12

with open("derivation_checks_results.json", "w") as f:
    json.dump(R, f, indent=1)
for k, v in R.items():
    if not isinstance(v, (list, dict)):
        print(k, v)
print("D12", [(x["lam"], x["w"], x["max_abs_err"]) for x in R["D12_fp_residual_closed_form"]])
print("D3", R["D3_importance_check"])
for r in R["D8_gaussian_pair"]:
    print(r)
for r in R["D9_cfg_vs_diffused_tilted_eps"]:
    print(r)
print(R["D10_fokker_planck_residual_std"])
print(R["D7_ddim_vs_ode_euler"])
for r in mom:
    print(r)
