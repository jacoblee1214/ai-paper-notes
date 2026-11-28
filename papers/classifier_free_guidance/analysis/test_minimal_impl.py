"""CFG 최소 구현의 성질 검사(학습 없음, 정확한 점수). 실행: python -m pytest -q test_minimal_impl.py"""
import math

import pytest
import torch

import minimal_impl as mi

M = mi.default_toy()


def test_w0_is_conditional_exactly():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(50, 2, generator=g)
    for lam in (-8.0, 0.0, 6.0):
        assert torch.equal(mi.eps_guided(M, z, lam, 1, 0), M.eps_c(z, lam, 1))
    # 같은 시드의 표본기 출력이 비트 단위로 같다
    for samp in (mi.sample_ddim, mi.sample_ancestral):
        x0 = samp(mi.guided_fn(M, 2, 0), 500, 32, torch.Generator().manual_seed(5))
        x1 = samp(lambda z, lam: M.eps_c(z, lam, 2), 500, 32, torch.Generator().manual_seed(5))
        assert torch.equal(x0, x1)


def test_conditional_sampler_matches_data():
    g = torch.Generator().manual_seed(1)
    x = mi.sample_ddim(mi.guided_fn(M, 0, 0), 6000, 256, g)
    ref = M.sample(0, 6000, g)
    ref2 = M.sample(0, 6000, g)
    floor = mi.sliced_w1(ref, ref2)
    assert mi.sliced_w1(x, ref) < 2.5 * floor + 0.01


def test_closed_form_score_matches_autograd():
    g = torch.Generator().manual_seed(2)
    z = torch.randn(40, 2, generator=g).requires_grad_(True)
    for lam in (-3.0, 1.0, 9.0):
        a, s = mi.alpha_sigma(lam)
        for c in (None, 0, 2):
            lp = M.log_p(z, a, s, c)
            auto = torch.autograd.grad(lp.sum(), z)[0]
            assert torch.allclose(auto, M.score(z.detach(), a, s, c), atol=1e-9, rtol=1e-9)


def test_implicit_classifier_identity():
    """grad log p^i(c|z) = grad log p(z|c) - grad log p(z) = -(eps_c - eps_u)/sigma, 그리고 p^i = p(c|z)/pi_c."""
    g = torch.Generator().manual_seed(3)
    z = (2 * torch.randn(60, 2, generator=g)).requires_grad_(True)
    for lam in (-4.0, 0.0, 5.0):
        a, s = mi.alpha_sigma(lam)
        for c in range(M.C):
            lpost = M.log_post(z, a, s)[:, c]
            auto = torch.autograd.grad(lpost.sum(), z)[0]          # 베이즈 사후를 따로 계산해 미분
            from_eps = -(M.eps_c(z.detach(), lam, c) - M.eps_u(z.detach(), lam)) / s
            assert torch.allclose(auto, from_eps, atol=1e-9, rtol=1e-8)


def test_cfg_equals_classifier_guidance_with_implicit_classifier():
    """식 (6) = eps_c - w sigma grad log p^i(c|z)  (논문 3.2절)."""
    g = torch.Generator().manual_seed(4)
    z = torch.randn(30, 2, generator=g).requires_grad_(True)
    lam, w, c = 1.5, 2.7, 1
    a, s = mi.alpha_sigma(lam)
    gradpi = torch.autograd.grad(M.log_post(z, a, s)[:, c].sum(), z)[0]
    cg = M.eps_c(z.detach(), lam, c) - w * s * gradpi
    assert torch.allclose(cg, mi.eps_guided(M, z.detach(), lam, c, w), atol=1e-10)
    # gamma 규약: eps_u + gamma (eps_c - eps_u), gamma = 1 + w
    gam = 1 + w
    alt = M.eps_u(z.detach(), lam) + gam * (M.eps_c(z.detach(), lam, c) - M.eps_u(z.detach(), lam))
    assert torch.allclose(alt, mi.eps_guided(M, z.detach(), lam, c, w), atol=1e-12)


@pytest.mark.parametrize("w", [0.0, 0.5, 2.0, 5.0])
def test_gaussian_ddim_matches_closed_form_ode(w):
    """가우시안 쌍(분산이 다름)에서 DDIM의 아핀 사상(잡음 없이 z=0, 1에서 평가)이 ODE의 닫힌 식과 같다."""
    P = mi.GaussPair([1.0, -0.5], 0.4, [0.0, 0.2], 1.3)
    fn = lambda z, lam: (1 + w) * P.eps_c(z, lam) - w * P.eps_u(z, lam)
    T = 2048
    B = mi.sample_ddim(fn, 1, T, None, z0=torch.zeros(1, 2))[0]
    A = mi.sample_ddim(fn, 1, T, None, z0=torch.tensor([[1.0, 0.0]]))[0] - B
    m, v = P.ode_moments(w)
    assert torch.allclose(B, m, atol=3e-3)
    assert abs(A[0].item() ** 2 - v) < 3e-3 * max(1.0, v)


def test_equal_variance_gaussian_cfg_is_exact_extrapolation():
    """분산이 같으면 안내된 장이 N(m_c + w (m_c - m_u), v)를 확산한 분포의 점수와 같아 ODE 끝점이 목표와 일치한다."""
    P = mi.GaussPair([1.0, -0.5], 0.4, [0.0, 0.2], 0.4)
    for w in (0.5, 3.0):
        m, v = P.ode_moments(w)
        tm, tv = P.tilted_moments(w)
        # 시작 분포가 N(0,I)라서(lambda_min=-20의 실제 주변 평균은 ~1e-4) 평균은 1e-3 안
        assert torch.allclose(m, tm, atol=1e-3) and abs(v - tv) < 1e-6
        assert torch.allclose(tm, P.mc + w * (P.mc - P.mu), atol=1e-12)


def test_unequal_variance_gaussian_cfg_differs_from_tilted_target():
    P = mi.GaussPair([1.0, -0.5], 0.4, [0.0, 0.2], 1.3)
    m, v = P.ode_moments(2.0)
    tm, tv = P.tilted_moments(2.0)
    assert abs(v - tv) / tv > 0.05 or (m - tm).abs().max() > 0.05


def test_guided_field_is_not_a_diffused_score():
    """w=0이면 포커-플랑크 잔차가 0이고, w>0이면 0이 아니다."""
    g = torch.Generator().manual_seed(6)
    z = 1.5 * torch.randn(80, 2, generator=g)
    for lam in (-1.0, 1.0, 3.0):
        r0, _ = mi.fp_residual(M, z, lam, 0, 0.0)
        r1, _ = mi.fp_residual(M, z, lam, 0, 1.0)
        assert r0 < 1e-8
        assert r1 > 1e-4


def test_variance_shrinks_monotonically_in_w():
    z0 = torch.randn(20000, 2, generator=torch.Generator().manual_seed(7))
    for c in range(M.C):
        v = []
        for w in (0.0, 0.5, 1.0, 2.0, 4.0):
            x = mi.sample_ddim(mi.guided_fn(M, c, w), 20000, 128, None, z0=z0)
            v.append(x.var(0).sum().item())
        assert all(a > b for a, b in zip(v, v[1:])), v


def test_tilted_grid_w0_is_conditional():
    tg = mi.TiltedGrid(M, 0, 0.0)
    m, cov = tg.moments()
    exact_mean = (M.wm[0][:, None] * M.mu[0]).sum(0)
    assert torch.allclose(m, exact_mean, atol=5e-3)
    assert abs(tg.p.sum().item() - 1) < 1e-12


def test_fp_residual_closed_form():
    """R(z) = const + (sigma^2/2) w (1+w) |grad log p(c|z)|^2 : 안내 장의 비일관성은 분류기 기울기 노름의 z 변동이다."""
    g = torch.Generator().manual_seed(8)
    z = 1.5 * torch.randn(120, 2, generator=g)
    for lam in (-3.0, 0.0, 3.0):
        _, s = mi.alpha_sigma(lam)
        for w in (0.5, 2.0):
            _, R = mi.fp_residual(M, z, lam, 0, w)
            gn = ((M.eps_c(z, lam, 0) - M.eps_u(z, lam)) ** 2).sum(1) / s ** 2
            pred = (s ** 2 / 2) * w * (1 + w) * gn
            assert torch.allclose(R - R.mean(), pred - pred.mean(), atol=1e-6)
