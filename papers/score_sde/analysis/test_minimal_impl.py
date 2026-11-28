"""minimal_impl.py의 핵심 성질 검사 (pytest). 학습 없이 정확한 점수만 쓴다."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as m  # noqa: E402

SDES = [m.VE, m.VP, m.SubVP]


@pytest.fixture(scope="module")
def gmm():
    return m.default_gmm()


@pytest.fixture(scope="module")
def sw(gmm):
    return m.SlicedW1(gmm)


def _forward_em(sde, x0, t_end, N, n, g):
    """순방향 SDE dx = f x dt + g dw 를 오일러-마루야마로 시뮬레이션 (스칼라)."""
    x = torch.full((n,), x0)
    h = t_end / N
    for i in range(N):
        t = i * h
        x = x + sde.f_coef(t) * x * h + math.sqrt(sde.g2(t) * h) * torch.randn(n, generator=g)
    return x


@pytest.mark.parametrize("cls", SDES)
def test_f_and_g_from_kernel_match_paper_eqs(cls):
    """섭동 핵에서 유도한 f, g^2가 식 9(VE), 11(VP), 12(sub-VP)와 같다."""
    sde = cls()
    for t in (0.05, 0.3, 0.7, 1.0):
        assert sde.g2(t) == pytest.approx(sde.g2_from_kernel(t), rel=1e-9)   # 핵에서 유도한 값과 닫힌 식이 같다
        if cls is m.VE:
            assert sde.f_coef(t) == 0.0
            assert sde.g2(t) == pytest.approx(2 * sde.lr * sde.sig(t) ** 2, rel=1e-12)
            # 식 9: g^2 = d sigma^2 / dt
            h = 1e-6
            d = (sde.sig(t + h) ** 2 - sde.sig(t - h) ** 2) / (2 * h)
            assert sde.g2(t) == pytest.approx(d, rel=1e-6)
        elif cls is m.VP:
            assert sde.f_coef(t) == pytest.approx(-0.5 * sde.beta(t), rel=1e-12)
            assert sde.g2(t) == pytest.approx(sde.beta(t), rel=1e-12)
        else:
            assert sde.f_coef(t) == pytest.approx(-0.5 * sde.beta(t), rel=1e-12)
            assert sde.g2(t) == pytest.approx(sde.beta(t) * (1 - math.exp(-2 * sde.B(t))), rel=1e-12)


@pytest.mark.parametrize("cls", SDES)
def test_perturbation_kernel_moments_vs_monte_carlo(cls):
    """식 29: 평균 s x0, 분산 sigma^2 (VE는 식 29 문자 그대로 sigma(t)^2 - sigma(0)^2)를 시뮬레이션과 대조."""
    sde = cls()
    n, x0 = 60000, 1.3
    for t_end in (0.3, 1.0):
        g = torch.Generator().manual_seed(11)
        x = _forward_em(sde, x0, t_end, 2000, n, g)
        mean_true = sde.s(t_end) * x0
        if cls is m.VE:
            var_true = sde.sig(t_end) ** 2 - sde.sig(0.0) ** 2
        else:
            var_true = sde.sig(t_end) ** 2
        se_mean = math.sqrt(var_true / n)
        se_var = var_true * math.sqrt(2 / (n - 1))
        assert abs(float(x.mean()) - mean_true) < 4 * se_mean + 2e-3 * abs(mean_true)
        assert abs(float(x.var()) - var_true) < 4 * se_var + 2e-3 * var_true


def test_ve_kernel_eq29_vs_eq31_differ_by_sigma_min_sq():
    """논문 식 29(VE)는 sigma^2(t) - sigma^2(0), 식 31은 sigma(t)^2이다. 차이는 sigma_min^2 = 1e-4."""
    sde = m.VE()
    t = 0.5
    assert sde.sig(t) ** 2 - (sde.sig(t) ** 2 - sde.sig(0.0) ** 2) == pytest.approx(sde.smin ** 2, rel=1e-9)


@pytest.mark.parametrize("cls", SDES)
def test_exact_score_matches_finite_difference(gmm, cls):
    pg = m.PerturbedGMM(gmm, cls())
    g = torch.Generator().manual_seed(0)
    for t in (0.05, 0.4, 0.9):
        x = pg.sample_t(30, t, g)
        S = pg.score(x, t)
        h = 1e-5
        fd = torch.zeros_like(x)
        for j in range(2):
            e = torch.zeros(2)
            e[j] = h
            fd[:, j] = (pg.logp(x + e, t) - pg.logp(x - e, t)) / (2 * h)
        assert (S - fd).abs().max() < 1e-6 * (1 + S.abs().max())


@pytest.mark.parametrize("cls", SDES)
def test_analytic_hessian_matches_autograd(gmm, cls):
    pg = m.PerturbedGMM(gmm, cls())
    g = torch.Generator().manual_seed(1)
    t = 0.35
    x = pg.sample_t(20, t, g)
    _, H = pg.score_and_hess(x, t)
    ad = m.div_autograd_exact(type("O", (), {"pf_velocity": lambda self, xx, tt: pg.score(xx, tt)})(), x, t)
    assert (torch.diagonal(H, dim1=1, dim2=2).sum(-1) - ad).abs().max() < 1e-9
    assert (H - H.transpose(1, 2)).abs().max() < 1e-12


def _spatial_div(fun, x):
    x = x.clone().requires_grad_(True)
    v = fun(x)
    tr = 0.0
    for i in range(x.shape[1]):
        tr = tr + torch.autograd.grad(v[:, i].sum(), x, retain_graph=True)[0][:, i]
    return tr.detach()


@pytest.mark.parametrize("cls", SDES)
def test_pf_ode_satisfies_fokker_planck_continuity_eq(gmm, cls):
    """식 13의 PF-ODE는 연속 방정식 dp/dt = -div(p v)를, 순방향 SDE는 포커-플랑크 dp/dt = -div(f x p) + g^2/2 Lap p를 만족한다."""
    sde = cls()
    pg = m.PerturbedGMM(gmm, sde)
    g = torch.Generator().manual_seed(2)
    t = 0.45
    x = pg.sample_t(15, t, g)
    h = 1e-5
    dpdt = (pg.logp(x, t + h).exp() - pg.logp(x, t - h).exp()) / (2 * h)
    p = lambda xx: pg.logp(xx, t).exp()
    div_pv = _spatial_div(lambda xx: p(xx)[:, None] * pg.pf_velocity(xx, t), x)
    assert (dpdt + div_pv).abs().max() < 1e-6 * p(x).max() + 1e-8
    div_fp = _spatial_div(lambda xx: p(xx)[:, None] * sde.f_coef(t) * xx, x)
    lap = _spatial_div(lambda xx: p(xx)[:, None] * pg.score(xx, t), x)   # div(p grad log p) = Lap p
    assert (dpdt + div_fp - 0.5 * sde.g2(t) * lap).abs().max() < 1e-6 * p(x).max() + 1e-8


def test_pf_ode_and_reverse_sde_share_marginals(gmm):
    """같은 p_T에서 출발한 PF-ODE(RK4)와 역방향 SDE(오일러)의 t=0.4 평균·공분산이 참 p_t와 표준오차 안에서 같다."""
    pg = m.PerturbedGMM(gmm, m.VP())
    sde = pg.sde
    g = torch.Generator().manual_seed(3)
    n, t_mid = 12000, 0.4
    truth = pg.sample_t(n, t_mid, g)
    z = pg.sample_t(n, sde.T, g)
    pf = m.rk4_integrate(pg.pf_velocity, z, sde.T, t_mid, 200)
    x, t, N = z.clone(), sde.T, 600
    h = (sde.T - t_mid) / N
    for _ in range(N):
        x = m.predictor_step(pg, x, t, h, "euler", g)
        t -= h
    for a in (pf, x):
        st = m._mom_stats(a, truth)
        assert st["max_mean_z"] < 4.5
        assert st["max_norm_cov_err"] < 0.06


def test_pf_encoding_is_deterministic_and_invertible(gmm):
    pg = m.PerturbedGMM(gmm, m.VP())
    x0 = gmm.sample(100, torch.Generator().manual_seed(4))
    c1, c2 = m.encode(pg, x0, N=300), m.encode(pg, x0, N=300)
    assert torch.equal(c1, c2)
    back = m.rk4_pf_sample(pg, c1, N=300)
    assert (back - x0).abs().max() < 1e-6
    # 부호: 코드는 대략 N(0,I)
    big = m.encode(pg, gmm.sample(20000, torch.Generator().manual_seed(5)), N=200)
    assert big.mean(0).abs().max() < 0.05
    assert (torch.cov(big.T) - torch.eye(2)).abs().max() < 0.06


@pytest.mark.parametrize("cls", SDES)
def test_likelihood_exact_trace_matches_true_logp(gmm, cls):
    """식 39: 정확한 p_T를 prior로 쓰면 log p_eps와 RK4 오차 안(<1e-5)에서 같고, 자동미분 대각합도 같다."""
    sde = cls()
    pg = m.PerturbedGMM(gmm, sde)
    x0 = gmm.sample(40, torch.Generator().manual_seed(6))
    eps = m.default_eps(sde)
    lp_true, _, _ = m.log_density(pg, x0, "analytic", N=200, prior="true")
    assert (lp_true - pg.logp(x0, eps)).abs().max() < 1e-5
    lp_ag, _, _ = m.log_density(pg, x0, "autograd", N=200)
    lp_an, _, _ = m.log_density(pg, x0, "analytic", N=200)
    assert (lp_ag - lp_an).abs().max() < 1e-9
    # 가우시안 prior를 쓰면 prior 불일치 때문에 VP/sub-VP는 0.02 안, VE(sigma_max=10)는 더 크다
    tol = 0.02 if cls is not m.VE else 0.5
    assert (lp_an - pg.logp(x0, eps)).abs().max() < tol


def test_hutchinson_unbiased_and_variance_formula(gmm):
    pg = m.PerturbedGMM(gmm, m.VP())
    g = torch.Generator().manual_seed(7)
    P, R = 6, 800
    x0 = gmm.sample(P, g)
    lp_an, _, M = m.log_density(pg, x0, "analytic", N=100)
    xr = x0.repeat_interleave(R, 0)
    e = torch.randint(0, 2, xr.shape, generator=g).double() * 2 - 1
    lp_h, _, _ = m.log_density(pg, xr, "hutchinson", probes=e, N=100)
    err = (lp_h.reshape(P, R) - lp_an[:, None])
    se = err.std(1) / math.sqrt(R)
    assert (err.mean(1).abs() < 4.5 * se).all()                       # 불편추정
    pred = 4 * (0.5 * (M + M.transpose(1, 2)))[:, 0, 1] ** 2           # 2차원 라데마허의 분산
    assert torch.allclose(err.var(1), pred, rtol=0.2, atol=1e-6)


@pytest.mark.parametrize("cls", SDES)
def test_velocity_score_conversion_for_pf_ode(gmm, cls):
    """PF-ODE 속도 f x - g^2/2 score = s' E[x0|x] + sigma' E[eps|x] (x0 사후 평균은 점수와 독립인 닫힌 식)."""
    pg = m.PerturbedGMM(gmm, cls())
    g = torch.Generator().manual_seed(8)
    for t in (0.1, 0.5, 0.95):
        x = pg.sample_t(25, t, g)
        assert (pg.pf_velocity(x, t) - pg.fm_velocity(x, t)).abs().max() < 1e-8 * (1 + pg.pf_velocity(x, t).abs().max())


def test_linear_path_velocity_formula_and_ode_reaches_data(gmm):
    """직선 경로: v = E[eps - x0|x] = -(x + t*score)/(1-t). 이 속도로 N(0,I)을 데이터로 보낸다."""
    sde = m.LinearPath()
    pg = m.PerturbedGMM(gmm, sde)
    g = torch.Generator().manual_seed(9)
    for t in (0.2, 0.6, 0.9):
        x = pg.sample_t(25, t, g)
        v = pg.fm_velocity(x, t)
        S = pg.score(x, t)
        assert (v + (x + t * S) / (1 - t)).abs().max() < 1e-8 * (1 + v.abs().max())
        # E[eps|x]을 독립 식으로: eps = (x - s x0)/sigma
        e_eps = (x - (1 - t) * pg.x0_mean(x, t)) / t
        assert (v - (e_eps - pg.x0_mean(x, t))).abs().max() < 1e-8 * (1 + v.abs().max())
    z = torch.randn(20000, 2, generator=g)
    xs = m.rk4_integrate(pg.fm_velocity, z, 1.0, 1e-3, 100)
    data = gmm.sample(20000, g)
    assert (xs.mean(0) - data.mean(0)).abs().max() < 0.08
    assert (torch.cov(xs.T) - torch.cov(data.T)).abs().max() < 0.15


def test_subvp_variance_bounded_by_vp_eq26_eq28():
    """식 26, 28: 같은 beta와 같은 초기 분산에서 Sigma_subVP(t) <= Sigma_VP(t), 차이는 e^{-B} - e^{-2B} (VP가 크다)."""
    vp, sub = m.VP(), m.SubVP()
    rng = np.random.default_rng(0)
    for v0 in rng.uniform(0.0, 50.0, 8):
        # 분산 ODE dSigma/dt = 2 f Sigma + g^2 를 RK4로 적분해 식 26, 28과 비교
        for sde, closed in ((vp, lambda B: 1 + math.exp(-B) * (v0 - 1)),
                            (sub, lambda B: 1 + math.exp(-2 * B) + math.exp(-B) * (v0 - 2))):
            def rhs(v, t):
                return 2 * sde.f_coef(t) * v + sde.g2(t)
            v, t, N = v0, 0.0, 4000
            h = 1.0 / N
            for _ in range(N):
                k1 = rhs(v, t); k2 = rhs(v + h / 2 * k1, t + h / 2); k3 = rhs(v + h / 2 * k2, t + h / 2); k4 = rhs(v + h * k3, t + h)
                v += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            assert v == pytest.approx(closed(vp.B(1.0)), rel=1e-7, abs=1e-9)
        for t in np.linspace(0.01, 1, 20):
            B = vp.B(t)
            gap = (1 + math.exp(-B) * (v0 - 1)) - (1 + math.exp(-2 * B) + math.exp(-B) * (v0 - 2))
            assert gap == pytest.approx(math.exp(-B) - math.exp(-2 * B), abs=1e-12)
            assert gap >= 0


def test_ddpm_discrete_kernel_converges_to_vp_kernel():
    """식 22의 이산 DDPM 핵이 연속 VP 핵(식 33)으로 1/N 속도로 수렴한다 (그림 5의 수치판)."""
    errs = []
    for N in (100, 1000):
        b = np.linspace(0.1 / N, 20.0 / N, N)
        a = np.cumprod(1 - b)
        tt = np.arange(1, N + 1) / N
        Bt = 0.1 * tt + 0.5 * 19.9 * tt ** 2
        errs.append(np.abs((1 - a) - (1 - np.exp(-Bt))).max())
    assert 8 < errs[0] / errs[1] < 12


def test_ancestral_matches_reverse_diffusion_to_second_order(gmm):
    """식 4의 조상 표집은 역확산(식 46)과 평균이 O(beta^2)로 다르다: h를 절반으로 하면 차이가 약 1/4."""
    pg = m.PerturbedGMM(gmm, m.VP())
    x = pg.sample_t(200, 0.5, torch.Generator().manual_seed(10))
    diffs = []
    for h in (4e-3, 2e-3):
        g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
        a = m.predictor_step(pg, x, 0.5, h, "ancestral", g1)
        r = m.predictor_step(pg, x, 0.5, h, "reverse_diffusion", g2)
        diffs.append(float((a - r).abs().max()))
    assert 3.0 < diffs[0] / diffs[1] < 5.0


@pytest.mark.parametrize("cls", [m.VE, m.VP])
def test_corrector_snr_step_size_rule(gmm, cls):
    """알고리즘 4-5: eps = 2 alpha (r |z| / |score|)^2, x <- x + eps score + sqrt(2 eps) z (배치 평균 노름)."""
    pg = m.PerturbedGMM(gmm, cls())
    t, r, h = 0.5, 0.16, 1e-3
    x = pg.sample_t(500, t, torch.Generator().manual_seed(12))
    out = m.corrector_step(pg, x, t, r, torch.Generator().manual_seed(77), h)
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(77))
    S = pg.score(x, t)
    alpha = 1.0 if cls is m.VE else 1.0 - pg.sde.beta(t) * h
    eps = 2 * alpha * (r * z.norm(dim=-1).mean() / S.norm(dim=-1).mean()) ** 2
    assert (out - (x + eps * S + torch.sqrt(2 * eps) * z)).abs().max() < 1e-12


def test_corrector_keeps_p_t_invariant_in_small_step_limit(gmm):
    """p_t의 표본에 작은 랑주뱅 스텝을 반복해도 평균·공분산이 유지된다 (r 작게)."""
    pg = m.PerturbedGMM(gmm, m.VP())
    g = torch.Generator().manual_seed(13)
    t = 0.2
    x = pg.sample_t(20000, t, g)
    ref = x.clone()
    for _ in range(10):
        x = m.corrector_step(pg, x, t, 0.02, g, 1e-3)
    st = m._mom_stats(x, ref)
    assert st["max_norm_cov_err"] < 0.03 and st["max_abs_mean_err"] < 0.05


def test_controllable_eq49_identity_and_class_conditional_sampling(gmm, sw):
    """식 49: grad log p_t(x|y=k) = grad log p_t(x) + grad log p_t(y=k|x) (분류기는 베이즈 규칙으로 닫힌 식)."""
    pg = m.PerturbedGMM(gmm, m.VP())
    g = torch.Generator().manual_seed(14)
    t = 0.3
    x = pg.sample_t(40, t, g)
    for k in range(gmm.K):
        xx = x.clone().requires_grad_(True)
        lj = pg._parts(xx, t)[0]
        gc = torch.autograd.grad((lj[:, k] - torch.logsumexp(lj, 1)).sum(), xx)[0]
        xx2 = x.clone().requires_grad_(True)
        s_k = torch.autograd.grad((pg._parts(xx2, t)[0][:, k] - math.log(float(gmm.w[k]))).sum(), xx2)[0]
        assert (pg.score(x, t) + gc - s_k).abs().max() < 1e-9
    k = 2
    xs = m._cond_class_sample(pg, k, 3000, 400, g)
    ref = m._component_samples(gmm, k, 3000, g)
    xu = m.run_sampler(pg, 3000, 400, g, predictor="euler")
    assert m._sw_two(sw, xs, ref) < 0.05
    assert m._sw_two(sw, xu, ref) > 1.0


def test_exact_conditional_inpainting_control_is_correct(gmm):
    """대조군: 정확한 조건부 점수로 만든 x_2 표본은 닫힌 식 조건부 분포와 맞는다 (W1 < 0.03)."""
    pg = m.PerturbedGMM(gmm, m.VP())
    c = 0.0
    xe = m._cond_exact_sample(pg, gmm, c, 4000, 400, torch.Generator().manual_seed(15))
    q = m._exact_cond_x2_quantiles(gmm, c, 4000)
    assert float((torch.sort(xe)[0] - q).abs().mean()) < 0.03


def test_pc_beats_corrector_only_on_mode_weights(gmm):
    """정확한 점수에서도 C-only(어닐링 랑주뱅)는 모드 비율을 못 맞추고 PC는 맞춘다 (같은 NFE=400)."""
    pg = m.PerturbedGMM(gmm, m.VP())
    tv_c, tv_pc = [], []
    for s in range(4):
        g = torch.Generator().manual_seed(s)
        tv_c.append(m.mode_tv(gmm, m.run_sampler(pg, 3000, 200, g, n_corr=2, c_only=True, snr=0.16)))
        g = torch.Generator().manual_seed(s)
        tv_pc.append(m.mode_tv(gmm, m.run_sampler(pg, 3000, 200, g, n_corr=1, snr=0.16)))
    assert np.mean(tv_pc) < np.mean(tv_c)
    assert np.mean(tv_pc) < 0.05


def test_corrector_stationary_variance_bias_follows_fixed_point():
    """배치 노름 규칙에서 정상 분산은 v(1 + r^2) (순진한 1/(1-r^2)가 아님). r=0.4에서 16% 대 19%."""
    out = m.exp_corrector_bias(n=40000, steps=120, rs=(0.4,))
    assert out["0.4"]["var_ratio"] == pytest.approx(1.16, abs=0.015)


@pytest.mark.parametrize("cls", [m.VE, m.VP, m.SubVP, m.LinearPath])
def test_flow_matching_loss_is_score_matching_with_weight_g4_over_4(cls):
    """조건부 속도 회귀 ||v_theta - (s' x0 + sigma' eps)||^2 = (g^2/2)^2 ||s_theta - grad log p_0t||^2 (v_theta를 점수 모델에서 유도)."""
    sde = cls()
    g = torch.Generator().manual_seed(16)
    t = 0.4
    s, sg, ds, dsg = sde.s(t), sde.sig(t), sde.ds(t), sde.dsig(t)
    x0, eps = torch.randn(50, 2, generator=g), torch.randn(50, 2, generator=g)
    x = s * x0 + sg * eps
    score_theta = torch.randn(50, 2, generator=g)                 # 임의의 모델 출력
    target_score = -eps / sg                                      # grad log p_0t(x|x0)
    # v_theta = s' x0_hat + sigma' eps_hat, x0_hat = (x + sigma^2 s_theta)/s, eps_hat = -sigma s_theta
    v_theta = ds * (x + sg ** 2 * score_theta) / s + dsg * (-sg * score_theta)
    v_target = ds * x0 + dsg * eps
    lhs = ((v_theta - v_target) ** 2).sum(-1)
    g2 = sde.g2_from_kernel(t)
    rhs = (g2 / 2) ** 2 * ((score_theta - target_score) ** 2).sum(-1)
    assert torch.allclose(lhs, rhs, rtol=1e-9, atol=1e-12)
