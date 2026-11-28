"""minimal_impl.py 의 핵심 성질 검사. 실행: python3 -m pytest -q test_minimal_impl.py"""
import math

import numpy as np
import pytest
import torch

import minimal_impl as m

DT = m.DT
AB = m.make_alpha_bar()
T = 1000


@pytest.fixture(scope="module")
def gmm():
    return m.default_gmm()


def randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=DT)


# ---------------------------------------------------------------- 스케줄과 격자
def test_schedule():
    assert AB.shape == (T + 1,) and float(AB[0]) == 1.0
    assert abs(float(AB[1]) - (1 - 1e-4)) < 1e-15
    assert float(AB[T]) < 1e-4 and bool((AB[1:] < AB[:-1]).all())


def test_taus():
    for S in (10, 20, 50, 100, 1000):
        t = m.make_taus(T, S, "linear")
        assert len(t) == S and t[-1] == T and t == sorted(set(t))
    q = m.make_taus(T, 50, "quad")
    assert q == sorted(set(q)) and q[0] >= 1 and q[-1] == T and len(q) < 50   # 앞쪽이 겹쳐 지워진다
    # 공식 코드 격자: uniform S=10 은 (0 기준) 0..900 -> 논문 색인 1..901, quad 는 마지막이 0.8T
    assert m.make_taus_official(T, 10, "uniform")[-1] == 901
    assert m.make_taus_official(T, 10, "quad")[-1] == 801
    assert len(set(m.make_taus_official(T, 100, "quad"))) < 100


# ---------------------------------------------------------------- 정확한 eps*
def test_eps_star_is_scaled_score(gmm):
    """eps* = -sqrt(1-a) grad log p_t(x). autograd 로 검산."""
    for a in (0.999, 0.5, 0.01):
        x = randn(50, 2, seed=1).requires_grad_(True)
        g, = torch.autograd.grad(gmm.log_pdf_t(x, a).sum(), x)
        assert torch.allclose(gmm.eps(x.detach(), a), -math.sqrt(1 - a) * g, atol=1e-9)


# ---------------------------------------------------------------- Lemma 1: 임의의 sigma 에서 주변분포 보존
def _random_sigmas(seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros(T + 1)
    for t in range(2, T + 1):
        out[t] = rng.uniform(0.05, 1.0) * math.sqrt(1 - float(AB[t - 1]))   # sigma_t^2 <= 1-ab_{t-1} 이어야 함
    return out


def test_lemma1_analytic():
    """x_t|x0 ~ N(sqrt(ab_t) x0, (1-ab_t)) 이면 q_sigma 한 스텝 뒤 평균 계수와 분산이 정확히 t-1의 값."""
    sig = _random_sigmas()
    for t in range(2, T + 1):
        A, B = m.q_sigma_coefs(AB, t, sig[t])
        # 평균: A x0 + B sqrt(ab_t) x0 = sqrt(ab_{t-1}) x0
        assert abs(A + B * math.sqrt(float(AB[t])) - math.sqrt(float(AB[t - 1]))) < 1e-12
        # 분산: B^2 (1-ab_t) + sigma^2 = 1-ab_{t-1}
        assert abs(B ** 2 * (1 - float(AB[t])) + sig[t] ** 2 - (1 - float(AB[t - 1]))) < 1e-12


def test_lemma1_monte_carlo():
    """x_T 에서 t=500 까지 q_sigma 로 내려 보내(한 차원, x0 고정) 평균·분산이 q(x_500|x0)와 같은지."""
    sig = _random_sigmas(3)
    n, x0v = 200000, 0.7
    g = torch.Generator().manual_seed(5)
    x = math.sqrt(float(AB[T])) * x0v + math.sqrt(1 - float(AB[T])) * torch.randn(n, generator=g, dtype=DT)
    for t in range(T, 500, -1):
        A, B = m.q_sigma_coefs(AB, t, sig[t])
        x = A * x0v + B * x + sig[t] * torch.randn(n, generator=g, dtype=DT)
    a500 = float(AB[500])
    se_mean = math.sqrt((1 - a500) / n)
    assert abs(float(x.mean()) - math.sqrt(a500) * x0v) < 5 * se_mean
    assert abs(float(x.var()) / (1 - a500) - 1) < 5 * math.sqrt(2 / n)


# ---------------------------------------------------------------- eta=1 은 DDPM 조상 표집
def test_eta1_is_ddpm_posterior():
    """sigma_t(eta=1)^2 = beta_tilde_t, 그리고 평균 계수가 DDPM 사후 평균(Ho 식 7)과 같다 (T=1000 전체 사슬)."""
    for t in range(2, T + 1):
        a, ap = float(AB[t]), float(AB[t - 1])
        beta = 1 - a / ap
        beta_tilde = beta * (1 - ap) / (1 - a)
        sig = m.sigma_eta(a, ap, 1.0)
        assert abs(sig ** 2 - beta_tilde) < 1e-15
        # x_{t-1} 평균 = A f + B x_t  (f = 예측 x0).  DDPM: sqrt(ap) beta/(1-a) f + sqrt(alpha_t)(1-ap)/(1-a) x_t
        A, B = m.q_sigma_coefs(AB, t, sig)
        assert abs(A - math.sqrt(ap) * beta / (1 - a)) < 1e-12
        assert abs(B - math.sqrt(a / ap) * (1 - ap) / (1 - a)) < 1e-12


def test_sigma_hat_larger():
    for t in (2, 100, 500, 1000):
        a, ap = float(AB[t]), float(AB[t - 1])
        assert m.sigma_hat(a, ap) ** 2 == pytest.approx(1 - a / ap)          # = beta_t
        assert m.sigma_hat(a, ap) > m.sigma_eta(a, ap, 1.0)


def test_eta0_zero_noise_and_last_step_eta():
    assert m.sigma_eta(float(AB[100]), float(AB[0]), 1.0) == 0.0            # 마지막 스텝(a_prev=1)은 eta=1 도 잡음 0
    assert m.sigma_eta(float(AB[100]), float(AB[50]), 0.0) == 0.0


def test_hat_final_step_noise(gmm):
    """sigma_hat 은 논문 식대로면 마지막 스텝에도 잡음이 있고, 공식 코드의 mask 는 그것을 지운다."""
    taus = m.make_taus(T, 10)
    xT = randn(500, 2, seed=2)
    a = m.sample(gmm.eps, xT, taus, AB, gen=torch.Generator().manual_seed(1), hat=True)
    b = m.sample(gmm.eps, xT, taus, AB, gen=torch.Generator().manual_seed(1), hat=True, mask_last_hat=True)
    assert (a - b).abs().max() > 0.05
    # 마지막 스텝 직전까지 같은 난수라 차이는 마지막 잡음 크기 sqrt(1-ab_{tau_1}) 정도
    assert float((a - b).std()) == pytest.approx(math.sqrt(1 - float(AB[taus[0]])), rel=0.15)


# ---------------------------------------------------------------- 정리 1: 가중치 항등식
def test_theorem1_kl_identity():
    """KL(q_sigma(x_{t-1}|x_t,x0) || q_sigma(x_{t-1}|x_t,f)) = gamma_t ||eps - eps_theta||^2 (임의 sigma)."""
    rng = np.random.default_rng(1)
    for t in (2, 10, 300, 999):
        sig = rng.uniform(0.1, 0.9) * math.sqrt(1 - float(AB[t - 1]))
        x0, f = randn(1, 5, seed=t), randn(1, 5, seed=t + 1)
        eps = randn(1, 5, seed=t + 2)
        xt = math.sqrt(float(AB[t])) * x0 + math.sqrt(1 - float(AB[t])) * eps
        eps_th = (xt - math.sqrt(float(AB[t])) * f) / math.sqrt(1 - float(AB[t]))
        A, B = m.q_sigma_coefs(AB, t, sig)
        kl = float(((A * x0 + B * xt - (A * f + B * xt)) ** 2).sum() / (2 * sig ** 2))
        assert kl == pytest.approx(m.theorem1_gamma(AB, t, sig) * float(((eps - eps_th) ** 2).sum()), rel=1e-9)


def test_theorem1_gamma_matches_ddpm_but_not_paper():
    """sigma 가 DDPM 사후 분산이면 gamma_t 가 Ho(2020) 식 12의 가중치와 일치. 논문 부록 식 35의 1/(2 d sigma^2 abar)는 다르다."""
    for t in (2, 50, 500, 1000):
        a, ap = float(AB[t]), float(AB[t - 1])
        sig = m.sigma_eta(a, ap, 1.0)
        assert m.theorem1_gamma(AB, t, sig) == pytest.approx(m.ho_gamma(AB, t), rel=1e-9)
        assert m.paper_gamma(AB, t, sig) != pytest.approx(m.ho_gamma(AB, t), rel=0.05)


def test_theorem1_gamma_positive_and_diverges_at_sigma0():
    """모든 sigma>0 에서 gamma>0 (정리 1). sigma->0 에서 1/sigma^2 로 커진다(정리가 sigma=0을 포함하지 않는 이유)."""
    t = 200
    g = [m.theorem1_gamma(AB, t, s) for s in (1e-1, 1e-2, 1e-3)]
    assert all(v > 0 for v in g)
    assert g[2] / g[1] == pytest.approx(100.0, rel=0.05)


# ---------------------------------------------------------------- DDIM = 재매개화 ODE의 오일러
@pytest.mark.parametrize("S", [10, 50])
def test_ddim_equals_euler_on_reparam_ode(gmm, S):
    xT = randn(400, 2, seed=11)
    taus = m.make_taus(T, S)
    a = m.sample(gmm.eps, xT, taus, AB, eta=0.0)
    b = m.sample_euler_bar(gmm.eps, xT, taus, AB, "ddim")
    assert (a - b).abs().max() < 1e-9


def test_ddim_equals_euler_stepwise_random_grid(gmm):
    taus = sorted(np.random.default_rng(0).choice(np.arange(1, T + 1), 12, replace=False).tolist())
    if taus[-1] != T:
        taus[-1] = T
    xT = randn(100, 2, seed=12)
    a, traj = m.sample(gmm.eps, xT, taus, AB, eta=0.0, record=True)
    sb = torch.sqrt((1 - AB) / AB)
    prev = [0] + taus[:-1]
    for k, (t, tp) in enumerate(zip(reversed(taus), reversed(prev))):
        x = traj[k]
        xb = x / math.sqrt(float(AB[t]))
        xb2 = xb + float(sb[tp] - sb[t]) * gmm.eps(x, float(AB[t]))
        assert (traj[k + 1] / math.sqrt(float(AB[tp])) - xb2).abs().max() < 1e-10


def test_pf_ode_euler_differs_but_converges(gmm):
    """식 15(PF-ODE 오일러)는 DDIM과 O(스텝^2)로만 다르다: S가 커지면 격차가 줄고 S=10 에서는 눈에 띄게 다르다."""
    xT = randn(300, 2, seed=13)
    gaps = []
    for S in (10, 100, 1000):
        taus = m.make_taus(T, S)
        a = m.sample_euler_bar(gmm.eps, xT, taus, AB, "ddim")
        b = m.sample_euler_bar(gmm.eps, xT, taus, AB, "pf")
        gaps.append(float((a - b).norm(dim=1).median()))   # 모드 경계 근처 몇 점은 다른 봉우리로 가므로 중앙값
    assert gaps[0] > 10 * gaps[2] and gaps[1] > gaps[2]


# ---------------------------------------------------------------- eta=0: 결정적, 역변환 가능, 오차는 S와 함께 감소
def test_eta0_deterministic(gmm):
    xT = randn(200, 2, seed=21)
    taus = m.make_taus(T, 20)
    a = m.sample(gmm.eps, xT, taus, AB, eta=0.0, gen=torch.Generator().manual_seed(1))
    b = m.sample(gmm.eps, xT, taus, AB, eta=0.0, gen=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)
    c = m.sample(gmm.eps, xT, taus, AB, eta=1.0, gen=torch.Generator().manual_seed(1))
    d = m.sample(gmm.eps, xT, taus, AB, eta=1.0, gen=torch.Generator().manual_seed(2))
    assert not torch.equal(c, d)


def test_encode_decode_error_shrinks(gmm):
    x0 = gmm.sample(1500, torch.Generator().manual_seed(31))
    errs = []
    for S in (10, 50, 250, 1000):
        taus = m.make_taus(T, S)
        rec = m.sample(gmm.eps, m.encode(gmm.eps, x0, taus, AB), taus, AB, eta=0.0)
        errs.append(float(((rec - x0) ** 2).mean()))
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert errs[3] < 1e-3 * errs[0]


def test_ddim_error_to_ode_shrinks_first_order(gmm):
    """연속 ODE 기준 해(RK4)와의 RMS 거리가 S에 따라 줄고, 로그-로그 기울기가 -1 근방(1차 오일러)."""
    xT = randn(400, 2, seed=41)
    ref = m.ode_reference(gmm, xT, AB, n_steps=800)
    Ss = [20, 50, 200, 500]
    e = [float(((m.sample(gmm.eps, xT, m.make_taus(T, S), AB, eta=0.0) - ref) ** 2).sum(1).mean().sqrt()) for S in Ss]
    assert all(e[i] > e[i + 1] for i in range(3))
    slope = np.polyfit(np.log(Ss), np.log(e), 1)[0]
    assert -1.6 < slope < -0.5


def test_ode_reference_consistent_with_marginal(gmm):
    """x_T ~ N(0,I) 를 정확한 ODE 로 보내면 데이터 분포(표본 잡음 바닥 안)."""
    n = 3000
    xT = torch.randn(n, 2, generator=torch.Generator().manual_seed(51), dtype=DT)
    x0 = m.ode_reference(gmm, xT, AB, n_steps=600)
    ref = gmm.sample(n, torch.Generator().manual_seed(52))
    floor = m.sliced_w1(gmm.sample(n, torch.Generator().manual_seed(53)), ref)
    assert m.sliced_w1(x0, ref) < 1.6 * floor + 0.005


# ---------------------------------------------------------------- 표집 품질과 지표
def test_full_chain_eta1_matches_data(gmm):
    n = 3000
    xT = torch.randn(n, 2, generator=torch.Generator().manual_seed(61), dtype=DT)
    x0 = m.sample(gmm.eps, xT, m.make_taus(T, 1000), AB, eta=1.0, gen=torch.Generator().manual_seed(62))
    ref = gmm.sample(n, torch.Generator().manual_seed(63))
    floor = m.sliced_w1(gmm.sample(n, torch.Generator().manual_seed(64)), ref)
    assert m.sliced_w1(x0, ref) < 1.6 * floor + 0.005


def test_metrics_sanity():
    g = torch.Generator().manual_seed(7)
    a, b = torch.randn(2000, 2, generator=g, dtype=DT), torch.randn(2000, 2, generator=g, dtype=DT)
    assert m.sliced_w1(a, b) < 0.1 and m.sliced_w1(a, b + 0.5) > 0.25
    assert abs(m.mmd2(a, b)) < 0.02 and m.mmd2(a, b + 1.0) > 0.1
    lo, hi = m.bootstrap_ci(np.arange(100.0), B=2000)[1:]
    assert 40 < lo < 49.5 < hi < 60
    adj, rej = m.holm([0.001, 0.02, 0.04, 0.5])
    assert rej == [True, False, False, False] and adj[0] == pytest.approx(0.004)


def test_slerp_endpoints():
    z0, z1 = randn(5, 2, seed=1), randn(5, 2, seed=2)
    assert torch.allclose(m.slerp(z0, z1, 0.0), z0, atol=1e-9) and torch.allclose(m.slerp(z0, z1, 1.0), z1, atol=1e-9)


def test_ddim_exact_for_point_mass():
    """데이터가 (거의) 한 점이면 eps*가 궤적을 따라 상수라서 sigma_bar에 대한 오일러가 정확하다: 1스텝도 거의 맞다."""
    pm = m.GMM([1.0], [[0.3, -0.2]], [1e-3])
    xT = randn(500, 2, seed=71)
    for S in (1, 2, 1000):
        x0 = m.sample(pm.eps, xT, m.make_taus(T, S), AB, eta=0.0)
        assert float((x0 - pm.mu[0]).norm(dim=1).max()) < 2e-2
