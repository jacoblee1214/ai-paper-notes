"""DDPM 최소 구현의 핵심 성질 검사 (pytest). 학습 없음, CPU, 전체 1분 안팎.

식 (4), (6)-(7), (10)-(12), 알고리즘 1·2, eps*의 최소성, sigma_t^2 최적성(§3.2), L_T, 식 (3) = (5), 공식 U-Net 모수 수.
확률적 검사는 고정 시드를 쓰고 허용 오차는 몬테카를로 표준오차의 몇 배로 둔다.
"""
import math

import pytest
import torch

from minimal_impl import (DT, GMM, Schedule, forward_marginal_variance, gaussian_sampler_variances, kl_iso, l_simple_samples,
                          marginal_kl_terms, posterior_mean_var, q_sample, sample, toy_mixture_1d, toy_mixture_2d)

SCH = Schedule.linear(1000, 1e-4, 0.02)


def gen(seed):
    return torch.Generator().manual_seed(seed)


def test_forward_variance_recursion_equals_eq4():
    v0 = 0.37
    v = v0
    for i in range(SCH.T):
        v = float(SCH.alphas[i]) * v + float(SCH.betas[i])
        assert v == pytest.approx(float(SCH.abar[i]) * v0 + 1 - float(SCH.abar[i]), abs=1e-12)


def test_forward_chain_simulation_matches_eq4():
    g = gen(0)
    n = 100000
    x0 = toy_mixture_1d().sample(n, g)
    x = x0.clone()
    for i in range(SCH.T):
        x = SCH.alphas[i].sqrt() * x + SCH.betas[i].sqrt() * torch.randn(n, 1, generator=g, dtype=DT)
        if i + 1 in (10, 100, 500, 1000):
            res = x - SCH.abar[i].sqrt() * x0
            z = (res.var() / (1 - SCH.abar[i]) - 1) * math.sqrt((n - 1) / 2)
            assert abs(float(z)) < 4.0
            assert abs(float(res.mean())) < 4 * math.sqrt(float(1 - SCH.abar[i]) / n)


def test_posterior_equals_brute_force_gaussian_conditioning():
    x0, xt = 0.7, 0.3
    for i in range(1, SCH.T):
        ab, abp, a = float(SCH.abar[i]), float(SCH.abar[i - 1]), float(SCH.alphas[i])
        v1, v2, c12 = 1 - abp, 1 - ab, math.sqrt(a) * (1 - abp)  # (x_{t-1}, x_t | x_0)의 결합 가우시안
        cm = math.sqrt(abp) * x0 + c12 / v2 * (xt - math.sqrt(ab) * x0)
        cv = v1 - c12 ** 2 / v2
        m, b = posterior_mean_var(SCH, i, torch.tensor([[x0]], dtype=DT), torch.tensor([[xt]], dtype=DT))
        assert float(m) == pytest.approx(cm, abs=1e-10)
        assert float(b) == pytest.approx(cv, abs=1e-12)


def test_eps_parameterization_eq10_eq11_eq12():
    g = gen(1)
    n = 5000
    idx = torch.randint(1, SCH.T, (n,), generator=g)
    x0 = torch.randn(n, 2, generator=g, dtype=DT)
    eps = torch.randn(n, 2, generator=g, dtype=DT)
    eps_th = torch.randn(n, 2, generator=g, dtype=DT)
    ab, abp, a, b = SCH.abar[idx], SCH.abar_prev[idx], SCH.alphas[idx], SCH.betas[idx]
    xt = q_sample(x0, ab, eps)
    mu_tilde = abp.sqrt()[:, None] * b[:, None] / (1 - ab)[:, None] * x0 + a.sqrt()[:, None] * (1 - abp)[:, None] / (1 - ab)[:, None] * xt
    mu_eps = (xt - (b / (1 - ab).sqrt())[:, None] * eps) / a.sqrt()[:, None]
    assert torch.allclose(mu_tilde, mu_eps, atol=1e-10)  # 식 (10)
    mu_th = (xt - (b / (1 - ab).sqrt())[:, None] * eps_th) / a.sqrt()[:, None]  # 식 (11)
    for sig2 in (b, SCH.beta_tilde[idx]):  # 식 (12): 두 분산 선택 모두에서 성립
        lhs = ((mu_tilde - mu_th) ** 2).sum(-1) / (2 * sig2)
        rhs = b ** 2 / (2 * sig2 * a * (1 - ab)) * ((eps - eps_th) ** 2).sum(-1)
        assert torch.allclose(lhs, rhs, rtol=1e-8)


def test_eps_star_is_scaled_score():
    for data in (toy_mixture_1d(), toy_mixture_2d()):
        g = gen(2)
        for ab in (0.9999, 0.5, 4e-5):
            x = data.sample(50, g)
            xt = q_sample(x, ab, torch.randn(50, data.D, generator=g, dtype=DT)).requires_grad_(True)
            score = torch.autograd.grad(data.at(ab).log_prob(xt).sum(), xt)[0]
            assert torch.allclose(data.eps_star(xt.detach(), ab), -math.sqrt(1 - ab) * score, atol=1e-8)


def test_eps_star_is_the_minimizer_of_l_simple():
    data = toy_mixture_1d()
    g = gen(3)
    n = 300000
    x0 = data.sample(n, g)
    ii = torch.randint(0, SCH.T, (n,), generator=g)
    eps = torch.randn(n, 1, generator=g, dtype=DT)
    ab = SCH.abar[ii]
    xt = q_sample(x0, ab, eps)
    es = data.eps_star(xt, ab)
    base = ((eps - es) ** 2).sum(-1)
    e2 = float((es ** 2).sum(-1).mean())
    for c in (-0.2, -0.05, 0.05, 0.2):  # 같은 난수에서 손실 증가 = c^2 E||eps*||^2 (직교성)
        d = ((eps - (1 + c) * es) ** 2).sum(-1) - base
        se = float(d.std() / math.sqrt(n))
        assert float(d.mean()) > 0
        assert abs(float(d.mean()) - c * c * e2) < 4 * se
    # 임의의 다른 함수로 바꿔도 손실이 늘어난다
    other = ((eps - (es + 0.1 * torch.tanh(xt))) ** 2).sum(-1)
    assert float((other - base).mean()) > 0


def test_l_simple_at_eps_star_equals_irreducible_floor_for_unit_gaussian():
    gauss = GMM([1.0], [[0.0]], [[[1.0]]])  # 바닥 E_t[abar_t]
    zs = []
    for sd in range(6):
        loss, _ = l_simple_samples(gauss.eps_star, gauss, SCH, 300000, gen(10 + sd))
        zs.append(float((loss.mean() - SCH.abar.mean()) / (loss.std() / math.sqrt(300000))))
    assert abs(sum(zs) / len(zs)) < 3 / math.sqrt(len(zs)) + 0.5  # 평균 z가 0 근처
    assert max(abs(z) for z in zs) < 4.0


def test_sampler_moments_match_gaussian_data():
    m = torch.tensor([0.5, -0.3], dtype=DT)
    S = torch.tensor([[0.8, 0.3], [0.3, 0.5]], dtype=DT)
    data = GMM([1.0], m[None], S[None])
    n = 20000
    x = sample(data.eps_star, SCH, n, 2, "beta", gen(4))
    se_mean = (S.diagonal() / n).sqrt()
    assert torch.all((x.mean(0) - m).abs() < 5 * se_mean + 0.02)
    assert (torch.cov(x.T) - S).abs().max() < 0.05  # 몬테카를로 표준오차(약 0.008)의 몇 배


def test_gaussian_sampler_variance_exact_recursion():
    # N(0, s2) 데이터에서 T=1000, sigma^2 = beta_t: 최종 분산이 s2에 거의 같다(상대 오차 < 1%)
    for s2 in (1.0, 0.5, 0.25):
        v = gaussian_sampler_variances(s2, SCH, "beta")
        assert abs(v[0] / s2 - 1) < 1e-2


def test_sigma_optimality_claims_of_section_3_2():
    # x0 ~ N(0, I): sigma^2 = beta_t 이면 표본기의 모든 단계 분산이 순방향 주변 분산과 정확히 같다
    v = torch.tensor(forward_marginal_variance(1.0, SCH), dtype=DT)
    vb = torch.tensor(gaussian_sampler_variances(1.0, SCH, "beta"), dtype=DT)
    assert float((vb[1:] / v[1:] - 1).abs().max()) < 1e-12
    vt = torch.tensor(gaussian_sampler_variances(1.0, SCH, "beta_tilde"), dtype=DT)
    assert float((vt[1:] / v[1:] - 1).abs().max()) > 5e-3  # beta~는 이 경우 최적이 아니다
    # x0 = 한 점(분산 -> 0): sigma^2 = beta~_t 이면 분산이 1 - abar_t를 거의 정확히 따라간다. beta는 최대 10% 넘친다
    s2 = 1e-12
    v = torch.tensor(forward_marginal_variance(s2, SCH), dtype=DT)
    vt = torch.tensor(gaussian_sampler_variances(s2, SCH, "beta_tilde"), dtype=DT)
    vb = torch.tensor(gaussian_sampler_variances(s2, SCH, "beta"), dtype=DT)
    assert float((vt[1:] / v[1:] - 1).abs().max()) < 1e-3
    assert float((vb[1:] / v[1:] - 1).abs().max()) > 0.05


def test_optimal_sigma2_lies_between_beta_tilde_and_beta_for_unit_variance_data():
    # 최적 sigma^2 = beta~ + c0^2 E tr Var(x0|x_t) / D 이고 위치 f_t = E tr Var / (D (1-abar)) <= 1 (LMMSE 상계)
    data = toy_mixture_2d()
    g = gen(5)
    for t in (5, 50, 100, 500):
        i = t - 1
        ab = SCH.abar[i]
        x0 = data.sample(20000, g)
        xt = q_sample(x0, ab, torch.randn(20000, 2, generator=g, dtype=DT))
        f = float(data.cond_var_trace(xt, ab).mean() / (2 * (1 - ab)))
        assert 0.0 <= f <= 1.0 + 1e-2


def test_L_T_is_negligible_for_linear_schedule():
    ab_T = float(SCH.abar[-1])
    assert 4.0e-5 < ab_T < 4.1e-5  # sqrt = 6.35e-3, SNR = -43.9 dB
    ex2 = 0.256  # [-1,1]로 스케일한 CIFAR10 화소의 E[x0^2](널리 알려진 평균·표준편차에서 계산)
    lt_bits = 0.5 * (-ab_T - math.log(1 - ab_T) + ab_T * ex2) / math.log(2)
    assert lt_bits < 1e-5  # 논문 §4: 약 1e-5 bits/dim


def test_eq3_equals_eq5_on_the_same_paths():
    # T=3, 임의의 reverse 과정. 식 (3)(경로의 로그 비율)과 식 (5)(닫힌 식 KL)가 같은 기댓값을 준다
    s = Schedule(torch.tensor([0.1, 0.2, 0.3], dtype=DT))
    data = toy_mixture_1d()
    g = gen(6)
    n = 400000
    a_t, c_t, sig2 = [0.9, 0.8, 0.7], [0.05, -0.1, 0.2], [0.07, 0.10, 0.15]
    mu = lambda t, x: a_t[t - 1] * x + c_t[t - 1]
    logN = lambda x, m, v: -0.5 * (math.log(2 * math.pi * v)) - 0.5 * (x - m) ** 2 / v
    x0 = data.sample(n, g)
    xs = [x0]
    for i in range(3):
        xs.append(s.alphas[i].sqrt() * xs[-1] + s.betas[i].sqrt() * torch.randn(n, 1, generator=g, dtype=DT))
    L3 = -logN(xs[3], 0.0, 1.0)
    for t in (1, 2, 3):
        L3 = L3 - logN(xs[t - 1], mu(t, xs[t]), sig2[t - 1]) + logN(xs[t], s.alphas[t - 1].sqrt() * xs[t - 1], float(s.betas[t - 1]))
    abT = float(s.abar[2])
    L5 = kl_iso(math.sqrt(abT) * x0, 1 - abT, torch.zeros_like(x0), 1.0)
    for t in (2, 3):
        mt, bt = posterior_mean_var(s, t - 1, xs[0], xs[t])
        L5 = L5 + kl_iso(mt, float(bt), mu(t, xs[t]), sig2[t - 1])
    L5 = L5 - logN(xs[0], mu(1, xs[1]), sig2[0]).squeeze(-1)
    d = L3.squeeze(-1) - L5
    assert abs(float(d.mean())) < 4 * float(d.std() / math.sqrt(n))


def test_vlb_gap_is_positive_and_shrinks_with_more_steps():
    data = toy_mixture_1d()
    gaps = []
    for S in (10, 40):
        sch = SCH.respace(sorted({int(round(v)) for v in torch.linspace(1, 1000, S).tolist()}))
        r = marginal_kl_terms(data, sch, "beta", 6000, gen(7))
        gaps.append(r["KL_T"] + float(r["terms"].sum()))
        se = math.sqrt(float((r["se"] ** 2).sum()))
        assert gaps[-1] > 3 * se  # 간격 = 한계 KL의 합 > 0
    assert gaps[1] < gaps[0]


def test_respace_keeps_abar():
    steps = [1, 11, 101, 501, 1000]
    r = SCH.respace(steps)
    assert torch.allclose(r.abar, SCH.abar[torch.tensor(steps) - 1], rtol=1e-12)


def test_official_unet_parameter_count_matches_paper():
    from paper_arithmetic import count_unet
    assert count_unet(128, (1, 2, 2, 2), 2, (16,), 32) == pytest.approx(35.7e6, rel=0.005)
    assert count_unet(128, (1, 1, 2, 2, 4, 4), 2, (16,), 256) == pytest.approx(114e6, rel=0.005)
