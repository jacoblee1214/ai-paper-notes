"""DDPM 논문의 식 (4)-(14)와 §3.2, §4의 산술을 수치로 검산한다. 학습 없음, CPU, float64.

A. 스케줄 산술: abar_T, SNR, L_T (선형 스케줄 beta 1e-4 -> 0.02, T=1000)
B. 식 (4): 순방향 사슬의 분산 점화식과 닫힌 식, 사슬 시뮬레이션
C. 식 (6)(7): 가우시안 조건부 분포를 직접 계산(brute force)한 값과 비교
D. 식 (3) = 식 (5): 같은 경로에서 두 변분 하한 표현을 MC로 비교 (T=3, 임의의 reverse 과정)
E. 식 (8)-(12): mu_theta의 eps 매개변수화와 VLB 가중
F. L_simple과 진짜 VLB의 t별 가중
G. eps* = -sqrt(1-abar) * score (autograd)
H. sigma_t^2 = beta~_t 일 때 L_{t-1} = (SNR_{t-1} - SNR_t)/2 * E||x_0 - x0_hat||^2
I. sigma_t^2 최적성(§3.2): 정확한 사후 분산, 최적 sigma^2의 위치
결과는 derivation_checks_results.json.
"""
import json
import math
from pathlib import Path

import torch

from minimal_impl import (DT, GMM, LOG2PI, Schedule, forward_marginal_variance, gaussian_sampler_variances, kl_iso,
                          posterior_mean_var, q_sample, toy_mixture_1d, toy_mixture_2d)

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)
R = {}


def sched():
    return Schedule.linear(1000, 1e-4, 0.02)


# ---------- A. 스케줄 산술 ----------
def check_A():
    s = sched()
    abT = float(s.abar[-1])
    # CIFAR10 화소의 널리 알려진 평균·표준편차([0,1] 스케일, 이 저장소에서 재계산하지 않은 상수)
    mean01 = torch.tensor([0.4914, 0.4822, 0.4465], dtype=DT)
    std01 = torch.tensor([0.2470, 0.2435, 0.2616], dtype=DT)
    Ex2 = float(((2 * std01) ** 2 + (2 * mean01 - 1) ** 2).mean())  # [-1,1] 스케일의 E[x0^2]
    LT_nat = 0.5 * (-abT - math.log(1 - abT) + abT * Ex2)  # 좌표당 KL(N(sqrt(abar) x0,(1-abar)) || N(0,1)) 기댓값
    pts = {}
    for t in [1, 2, 10, 100, 250, 500, 750, 800, 900, 1000]:
        ab = float(s.abar[t - 1])
        pts[t] = {"abar": ab, "snr": ab / (1 - ab), "snr_db": 10 * math.log10(ab / (1 - ab)),
                  "info_nat_unit_var": 0.5 * math.log(1 + ab / (1 - ab))}
    t_half = int((s.abar < 0.5).nonzero()[0]) + 1
    R["A_schedule"] = {
        "sum_beta": float(s.betas.sum()), "sum_beta2": float((s.betas ** 2).sum()),
        "abar_T": abT, "sqrt_abar_T": math.sqrt(abT), "one_minus_abar_T": 1 - abT,
        "snr_T": abT / (1 - abT), "snr_T_db": 10 * math.log10(abT / (1 - abT)),
        "abar_T_approx_exp(-sum beta - sum beta^2/2)": math.exp(-float(s.betas.sum()) - 0.5 * float((s.betas ** 2).sum())),
        "cifar_Ex2": Ex2, "L_T_nat_per_dim": LT_nat, "L_T_bits_per_dim": LT_nat / LN2,
        "Ex2_implied_by_paper_1e-5_bits": 1e-5 * LN2 / (0.5 * abT),
        "t_abar_below_half": t_half, "points": pts,
        "frac_steps_snr_below_1e-2": float(((s.abar / (1 - s.abar)) < 1e-2).double().mean()),
    }


# ---------- B. 식 (4) ----------
def check_B():
    s = sched()
    # 분산 점화식 Var_t = alpha_t Var_{t-1} + beta_t (x0 ~ 분산 v0) 와 닫힌 식 abar v0 + 1 - abar
    v0 = 0.37
    v = v0
    err = 0.0
    for i in range(s.T):
        v = float(s.alphas[i]) * v + float(s.betas[i])
        err = max(err, abs(v - (float(s.abar[i]) * v0 + 1 - float(s.abar[i]))))
    # 사슬 시뮬레이션
    g = torch.Generator().manual_seed(1)
    n = 200000
    data = toy_mixture_1d()
    x0 = data.sample(n, g)
    x = x0.clone()
    zs = {}
    checkpoints = {1, 10, 100, 500, 1000}
    for i in range(s.T):
        x = s.alphas[i].sqrt() * x + s.betas[i].sqrt() * torch.randn(n, 1, generator=g, dtype=DT)
        if i + 1 in checkpoints:
            mean_th = s.abar[i].sqrt() * x0
            res = (x - mean_th)
            zs[i + 1] = {"resid_mean": float(res.mean()), "resid_var": float(res.var()),
                         "theory_var": float(1 - s.abar[i]),
                         "z_var": float((res.var() / (1 - s.abar[i]) - 1) * math.sqrt((n - 1) / 2))}
    R["B_eq4"] = {"max_abs_err_variance_recursion": err, "chain_sim_n": n, "seed": 1, "checkpoints": zs}


# ---------- C. 식 (6)(7) ----------
def check_C():
    s = sched()
    x0 = 0.7
    worst_m, worst_v = 0.0, 0.0
    for i in range(1, s.T):
        ab, abp, a = float(s.abar[i]), float(s.abar[i - 1]), float(s.alphas[i])
        # (x_{t-1}, x_t | x_0)의 결합 가우시안 (식 2의 사슬에서 직접)
        mu = torch.tensor([math.sqrt(abp) * x0, math.sqrt(ab) * x0], dtype=DT)
        v1, v2 = 1 - abp, 1 - ab
        cov12 = math.sqrt(a) * v1  # Cov(x_{t-1}, x_t) = sqrt(alpha_t) Var(x_{t-1})
        xt = 0.3
        cm = mu[0] + cov12 / v2 * (xt - mu[1])
        cv = v1 - cov12 ** 2 / v2
        m, bt = posterior_mean_var(s, i, torch.tensor([[x0]], dtype=DT), torch.tensor([[xt]], dtype=DT))
        worst_m = max(worst_m, abs(float(m) - float(cm)))
        worst_v = max(worst_v, abs(float(bt) - float(cv)))
    R["C_eq6_7"] = {"max_abs_err_mean": worst_m, "max_abs_err_var": worst_v, "t_range": "2..1000", "x0": x0, "x_t": 0.3}


# ---------- D. 식 (3) = 식 (5) ----------
def check_D():
    betas = torch.tensor([0.10, 0.20, 0.30], dtype=DT)
    s = Schedule(betas)
    data = toy_mixture_1d()
    # 임의의 reverse 과정: mu_theta(x_t, t) = a_t x_t + c_t, 분산 sig2_t
    a_t = [0.9, 0.8, 0.7]
    c_t = [0.05, -0.1, 0.2]
    sig2 = [0.07, 0.10, 0.15]
    g = torch.Generator().manual_seed(2)
    chunk, n_chunks = 250000, 8
    n = chunk * n_chunks

    def logN(x, m, v):
        return -0.5 * (LOG2PI + math.log(v)) - 0.5 * (x - m) ** 2 / v

    def mu_th(t, xt):
        return a_t[t - 1] * xt + c_t[t - 1]

    acc = torch.zeros(6, dtype=DT)  # sum L3, sum L3^2, sum L5, sum L5^2, sum (L3-L5)^2, sum (L3-L5)
    for _ in range(n_chunks):
        x0 = data.sample(chunk, g)
        xs = [x0]
        for i in range(3):
            xs.append(s.alphas[i].sqrt() * xs[-1] + s.betas[i].sqrt() * torch.randn(chunk, 1, generator=g, dtype=DT))
        # 식 (3): E_q[-log p(x_T) - sum_t log p_theta(x_{t-1}|x_t)/q(x_t|x_{t-1})]
        L3 = -logN(xs[3], 0.0, 1.0)
        for t in (1, 2, 3):
            L3 = L3 - logN(xs[t - 1], mu_th(t, xs[t]), sig2[t - 1]) + logN(xs[t], s.alphas[t - 1].sqrt() * xs[t - 1], float(s.betas[t - 1]))
        # 식 (5): KL들은 닫힌 식(사후는 식 6-7)
        abT = float(s.abar[2])
        L5 = kl_iso(math.sqrt(abT) * x0, 1 - abT, torch.zeros_like(x0), 1.0)
        for t in (2, 3):
            mt, bt = posterior_mean_var(s, t - 1, xs[0], xs[t])
            L5 = L5 + kl_iso(mt, float(bt), mu_th(t, xs[t]), sig2[t - 1])
        L5 = L5 - logN(xs[0], mu_th(1, xs[1]), sig2[0]).squeeze(-1)
        L3 = L3.squeeze(-1)
        acc += torch.stack([L3.sum(), (L3 ** 2).sum(), L5.sum(), (L5 ** 2).sum(), ((L3 - L5) ** 2).sum(), (L3 - L5).sum()])
    m3, m5 = acc[0] / n, acc[2] / n
    sd3, sd5 = (acc[1] / n - m3 ** 2).sqrt(), (acc[3] / n - m5 ** 2).sqrt()
    dd = acc[5] / n
    sdd = (acc[4] / n - dd ** 2).sqrt()
    R["D_eq3_vs_eq5"] = {"T": 3, "n": n, "seed": 2,
                         "eq3_mean": float(m3), "eq3_se": float(sd3 / math.sqrt(n)),
                         "eq5_mean": float(m5), "eq5_se": float(sd5 / math.sqrt(n)),
                         "eq5_std_over_eq3_std": float(sd5 / sd3),
                         "diff": float(dd), "diff_se_paired": float(sdd / math.sqrt(n))}


# ---------- E. 식 (8)-(12), 알고리즘 2의 한 단계 ----------
def check_E():
    s = sched()
    g = torch.Generator().manual_seed(3)
    n = 100000
    idx = torch.randint(1, s.T, (n,), generator=g)
    x0 = torch.randn(n, 2, generator=g, dtype=DT)
    eps = torch.randn(n, 2, generator=g, dtype=DT)
    ab, abp, a, b = s.abar[idx], s.abar_prev[idx], s.alphas[idx], s.betas[idx]
    xt = q_sample(x0, ab, eps)
    mu_tilde = abp.sqrt()[:, None] * b[:, None] / (1 - ab)[:, None] * x0 + a.sqrt()[:, None] * (1 - abp)[:, None] / (1 - ab)[:, None] * xt
    # 식 (10): mu~ = (x_t - beta/sqrt(1-abar) eps)/sqrt(alpha)
    mu_eps = (xt - (b / (1 - ab).sqrt())[:, None] * eps) / a.sqrt()[:, None]
    e10 = float((mu_tilde - mu_eps).abs().max())
    # 식 (12): ||mu~ - mu_theta||^2 / (2 sigma^2) = beta^2 / (2 sigma^2 alpha (1-abar)) ||eps - eps_theta||^2
    eps_th = torch.randn(n, 2, generator=g, dtype=DT)  # 임의의 예측
    mu_th = (xt - (b / (1 - ab).sqrt())[:, None] * eps_th) / a.sqrt()[:, None]
    sig2 = b
    lhs = ((mu_tilde - mu_th) ** 2).sum(-1) / (2 * sig2)
    rhs = b ** 2 / (2 * sig2 * a * (1 - ab)) * ((eps - eps_th) ** 2).sum(-1)
    e12 = float(((lhs - rhs).abs() / rhs.abs().clamp_min(1e-300)).max())
    # 알고리즘 2의 갱신식 = 식 (11) = "x0_hat을 만들어 사후 평균 식 (7)에 넣는" 공식 코드의 형태
    x0_hat = (xt - (1 - ab).sqrt()[:, None] * eps_th) / ab.sqrt()[:, None]
    mu_code = abp.sqrt()[:, None] * b[:, None] / (1 - ab)[:, None] * x0_hat + a.sqrt()[:, None] * (1 - abp)[:, None] / (1 - ab)[:, None] * xt
    e11 = float((mu_th - mu_code).abs().max())
    R["E_eps_param"] = {"n": n, "seed": 3, "max_abs_err_eq10": e10, "max_rel_err_eq12": e12, "max_abs_err_alg2_vs_posterior_form": e11}


# ---------- F. L_simple과 진짜 VLB의 t별 가중 ----------
def check_F():
    s = sched()
    t = torch.arange(1, s.T + 1, dtype=DT)
    w_beta = s.betas ** 2 / (2 * s.betas * s.alphas * (1 - s.abar))  # sigma^2 = beta_t
    w_bt = s.betas ** 2 / (2 * s.beta_tilde.clamp_min(1e-300) * s.alphas * (1 - s.abar))  # sigma^2 = beta~_t (t>=2)
    w_bt[0] = float("nan")
    # L_simple는 모든 t에 가중 1(식 14의 1/T 제외). 상대 가중 = w_T / w_t
    rel = w_beta[-1] / w_beta
    tot = w_beta.sum()
    rows = {}
    for tt in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        rows[tt] = {"w_beta": float(w_beta[tt - 1]), "w_beta_tilde": float(w_bt[tt - 1]),
                    "simple_over_vlb_rel_to_T": float(rel[tt - 1]), "approx_1_over_2t": 1 / (2 * tt)}
    cum = w_beta.cumsum(0) / tot
    R["F_weights"] = {
        "points": rows,
        "frac_vlb_weight_t_le_10": float(cum[9]), "frac_vlb_weight_t_le_100": float(cum[99]),
        "frac_vlb_weight_t_le_250": float(cum[249]), "uniform_frac_t_le_100": 0.1,
        "max_w": float(w_beta.max()), "min_w": float(w_beta.min()), "w_ratio_max_over_min": float(w_beta.max() / w_beta.min()),
        "t_where_w_within_10pct_of_w_T": int((w_beta < 1.1 * w_beta[-1]).nonzero()[0]) + 1,
        "w_beta_tilde_over_w_beta_min": float((w_bt[1:] / w_beta[1:]).min()), "w_beta_tilde_over_w_beta_max": float((w_bt[1:] / w_beta[1:]).max()),
    }


# ---------- G. eps* = -sqrt(1-abar) * score ----------
def check_G():
    out = {}
    for name, data in (("1d", toy_mixture_1d()), ("2d", toy_mixture_2d())):
        g = torch.Generator().manual_seed(4)
        worst = 0.0
        for ab in (0.9999, 0.9, 0.5, 0.05, 4e-5):
            qt = data.at(ab)
            x = data.sample(200, g)
            xt = (q_sample(x, ab, torch.randn(200, data.D, generator=g, dtype=DT))).requires_grad_(True)
            score = torch.autograd.grad(qt.log_prob(xt).sum(), xt)[0]
            e = data.eps_star(xt.detach(), ab)
            worst = max(worst, float((e + math.sqrt(1 - ab) * score).abs().max()))
        out[name] = worst
    R["G_eps_is_score"] = {"max_abs_err": out, "seed": 4}


# ---------- H. SNR 형태 ----------
def check_H():
    s = sched()
    snr = s.abar / (1 - s.abar)
    snr_prev = s.abar_prev / (1 - s.abar_prev)
    # eps 손실의 가중 w_t 곱하기 abar/(1-abar) = x0 손실의 가중
    w_beta = s.betas / (2 * s.alphas * (1 - s.abar))
    w_bt = s.betas[1:] ** 2 / (2 * s.beta_tilde[1:] * s.alphas[1:] * (1 - s.abar[1:]))
    coef_beta = w_beta * snr
    coef_bt = w_bt * snr[1:]
    half_dsnr = 0.5 * (snr_prev - snr)
    e_bt = float(((coef_bt - half_dsnr[1:]).abs() / half_dsnr[1:]).max())
    ratio = coef_beta[1:] / half_dsnr[1:]
    ratio_theory = (1 - s.abar_prev[1:]) / (1 - s.abar[1:])
    R["H_snr_form"] = {"max_rel_err_beta_tilde_vs_half_dSNR": e_bt,
                       "beta_ratio_min": float(ratio.min()), "beta_ratio_max": float(ratio.max()),
                       "max_abs_err_ratio_vs_(1-abar_prev)/(1-abar)": float((ratio - ratio_theory).abs().max())}


# ---------- I. sigma^2 최적성 ----------
def check_I():
    s = sched()
    out = {}
    # (1) x0 ~ N(0, 1): Var(x_{t-1} | x_t) 를 결합 가우시안에서 직접 계산 -> beta_t ?
    worst1 = 0.0
    for i in range(1, s.T):
        # x_{t-1} ~ N(0,1), x_t = sqrt(alpha) x_{t-1} + sqrt(beta) z => Var(x_t)=1, Cov = sqrt(alpha)
        cv = 1.0 - float(s.alphas[i]) / 1.0
        worst1 = max(worst1, abs(cv - float(s.betas[i])))
    out["gauss_unit_Var(x_{t-1}|x_t)_minus_beta"] = worst1
    # (2) x0 고정: Var(x_{t-1} | x_t, x_0) = beta~_t (식 7)는 C에서 확인. 분산 비율 확인
    out["beta_tilde_over_beta_t_2"] = float(s.beta_tilde[1] / s.betas[1])
    out["beta_tilde_over_beta_t_1000"] = float(s.beta_tilde[-1] / s.betas[-1])
    out["beta_tilde_over_beta_t_500"] = float(s.beta_tilde[499] / s.betas[499])
    # (3) 표본기의 정확한 최종 분산(1차원 가우시안, 정확한 eps*): 모든 항이 닫힌 식
    rows = {}
    for s2d in (1.0, 0.25, 1e-4, 1e-12):
        rows[str(s2d)] = {m: gaussian_final_var(s, s2d, m) for m in ("beta", "beta_tilde")}
    out["gaussian_sampler_final_variance_T1000"] = rows
    track = {}
    for s2d in (1.0, 0.25, 1e-4, 1e-8, 1e-12):
        v = torch.tensor(forward_marginal_variance(s2d, s), dtype=DT)
        track[str(s2d)] = {}
        for m in ("beta", "beta_tilde"):
            r = torch.tensor(gaussian_sampler_variances(s2d, s, m), dtype=DT) / v
            track[str(s2d)][m] = {"max_abs_log_ratio_t1..T": float(r[1:].log().abs().max()), "ratio_t500": float(r[500]),
                                  "ratio_t100": float(r[100]), "ratio_t10": float(r[10]), "ratio_t0": float(r[0])}
    out["gaussian_marginal_tracking_V_t_over_v_t"] = track
    # (4) 최적 sigma^2 = beta~ + c0^2 m_t / D, 위치 f_t = m_t / (D (1-abar_t)) in [0,1]
    g = torch.Generator().manual_seed(5)
    n = 100000
    res = {}
    for name, data in (("1d", toy_mixture_1d()), ("2d", toy_mixture_2d())):
        res[name] = {}
        for t in [2, 5, 10, 20, 50, 100, 200, 500, 1000]:
            i = t - 1
            ab = s.abar[i]
            x0 = data.sample(n, g)
            xt = q_sample(x0, ab, torch.randn(n, data.D, generator=g, dtype=DT))
            xh = data.x0_hat(xt, ab)
            m_t_mc = float(((x0 - xh) ** 2).sum(-1).mean())
            m_t = float(data.cond_var_trace(xt, ab).mean())  # E tr Var(x0|x_t): 조건부 분산의 닫힌 식(Rao-Blackwell)
            c0 = float(s.abar_prev[i].sqrt() * s.betas[i] / (1 - ab))
            bt = float(s.beta_tilde[i])
            sig_opt = bt + c0 ** 2 * m_t / data.D
            # 직접 최소화: sigma^2 격자에서 E KL(q(x_{t-1}|x_t,x_0) || N(mu_theta(eps*), sigma^2 I))
            e = data.eps_star(xt, ab)
            mu_th = (xt - s.betas[i] / (1 - ab).sqrt() * e) / s.alphas[i].sqrt()
            mu_t, _ = posterior_mean_var(s, i, x0, xt)
            grid = torch.logspace(math.log10(bt) - 0.5, math.log10(float(s.betas[i])) + 0.5, 4001, dtype=DT)
            klv = torch.stack([kl_iso(mu_t, bt, mu_th, float(v)).mean() for v in grid[::8]])
            j = int(klv.argmin()) * 8
            sig_grid = float(grid[j])
            res[name][t] = {"beta_tilde": bt, "beta": float(s.betas[i]), "sigma2_opt_formula": sig_opt,
                            "sigma2_opt_grid": sig_grid, "f_t": (sig_opt - bt) / (float(s.betas[i]) - bt),
                            "m_t_over_D(1-abar)": m_t / (data.D * float(1 - ab)),
                            "m_t_mc_over_m_t_rb": m_t_mc / m_t}
    out["mixture_optimal_sigma2"] = res
    out["mixture_n"] = n
    out["mixture_seed"] = 5
    R["I_sigma2"] = out


def gaussian_final_var(s, s2_data, mode):
    from minimal_impl import gaussian_sampler_variance
    return gaussian_sampler_variance(s2_data, s, mode)


def main():
    for f in (check_A, check_B, check_C, check_D, check_E, check_F, check_G, check_H, check_I):
        f()
        print(f.__name__, "ok")
    (HERE / "derivation_checks_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
