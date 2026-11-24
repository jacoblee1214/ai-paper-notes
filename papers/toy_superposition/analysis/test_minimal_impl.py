"""minimal_impl.py 성질 시험. 실행: python -m pytest analysis/test_minimal_impl.py -q  (CPU, 약 1-2분)"""
import math

import numpy as np
import pytest
import torch

import minimal_impl as mi

torch.set_num_threads(4)


# ---- 학습으로 확인하는 성질 ----------------------------------------------------


@pytest.mark.parametrize("density", [1.0, 0.1])
def test_linear_model_keeps_top_m_features_regardless_of_sparsity(density):
    n, m = 8, 3
    imp = mi.importance_curve(n, 0.7)
    r = mi.train(n, m, [density], imp, relu=False, steps=5000, lr=3e-3, seed=0)
    norm2 = (mi.feature_norms(r["W"]) ** 2)[0]
    assert (norm2[:m] > 0.8).all(), norm2
    assert (norm2[m:] < 0.2).all(), norm2
    assert int(mi.count_represented(r["W"])[0]) == m
    # 이론 최적 손실 sigma^2 * sum_{i>=m} I_i 와 가깝다(희소도와 무관하게 같은 특징을 고르는 증거)
    theory = mi.linear_optimal_loss(imp.numpy(), m, density)
    assert r["loss"][0].item() == pytest.approx(theory, rel=0.15)


def test_relu_dense_has_no_superposition():
    n, m = 20, 5
    imp = mi.importance_curve(n, 0.7)
    r = mi.train(n, m, [1.0], imp, steps=6000, seed=0)
    assert int(mi.count_represented(r["W"])[0]) <= m


def test_relu_sparse_represents_more_than_m_features():
    n, m = 20, 5
    imp = mi.importance_curve(n, 0.7)
    r = mi.train(n, m, [0.02], imp, steps=8000, seed=0)
    assert int(mi.count_represented(r["W"])[0]) > m


# ---- 차원수 함수 -------------------------------------------------------------


def test_dimensionality_orthogonal_is_one():
    assert torch.allclose(mi.dimensionality(torch.eye(4)), torch.ones(4))


def test_dimensionality_antipodal_pair_is_half():
    W = torch.tensor([[1.0, -1.0]])
    assert torch.allclose(mi.dimensionality(W), torch.full((2,), 0.5))


@pytest.mark.parametrize("k", [3, 5])
def test_dimensionality_regular_polygon_is_2_over_k(k):
    ang = 2 * math.pi * torch.arange(k) / k
    W = torch.stack([ang.cos(), ang.sin()])
    assert torch.allclose(mi.dimensionality(W), torch.full((k,), 2 / k), atol=1e-5)


def test_dimensionality_tetrahedron_three_quarters_and_scale_invariant():
    v = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32).T / math.sqrt(3)
    assert torch.allclose(mi.dimensionality(v), torch.full((4,), 0.75), atol=1e-5)
    assert torch.allclose(mi.dimensionality(7.0 * v), torch.full((4,), 0.75), atol=1e-5)


def test_dimensionality_unlearned_feature_is_zero_and_tegum_factor_independent():
    # 평면 1: 대척점 쌍, 평면 2(직교): 특징 하나 전담, 마지막 열: 안 배운 특징
    W = torch.tensor([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    D = mi.dimensionality(W)
    assert torch.allclose(D, torch.tensor([0.5, 0.5, 1.0, 0.0]))
    # 직교한 평면은 서로 간섭하지 않는다: 열마다 D 합 = 차원 수(효율적으로 쌓였을 때)
    assert D.sum().item() == pytest.approx(2.0)


# ---- 닫힌 형식과 구적법·몬테카를로의 일치 ----------------------------------------


@pytest.mark.parametrize("c,b", [(0.8, 0.0), (0.3, -0.1), (0.3, -0.5), (-0.6, 0.2), (1.2, -0.4), (0.0, 0.3)])
def test_relu_integral_matches_numeric(c, b):
    xs = (np.arange(200000) + 0.5) / 200000
    num = np.mean(np.maximum(c * xs + b, 0.0) ** 2)
    assert mi.relu_integral_sq(c, b) == pytest.approx(num, abs=1e-8)


def test_negative_interference_is_free_and_negative_bias_filters_small_positive():
    assert mi.relu_integral_sq(-0.5, 0.0) == 0.0          # b=0, 음의 간섭: 손실 0
    assert mi.relu_integral_sq(0.2, 0.0) == pytest.approx(0.04 / 3)
    assert mi.relu_integral_sq(0.2, -0.2) == 0.0          # c <= -b 이면 양의 간섭도 완전히 걸러짐


def test_one_sparse_loss_matches_monte_carlo():
    torch.manual_seed(0)
    n, m = 6, 2
    W = torch.randn(m, n) * 0.8
    b = torch.randn(n) * 0.1 - 0.05
    imp = 0.8 ** torch.arange(n).float()
    exact = mi.one_sparse_loss(W.numpy().astype(float), b.numpy().astype(float), imp.numpy().astype(float))
    N = 400000
    total = 0.0
    for i in range(n):
        x = torch.zeros(N, n)
        x[:, i] = torch.rand(N)
        out = torch.relu(x @ W.T @ W + b)
        total += (imp * (x - out) ** 2).sum(-1).mean().item()
    assert exact == pytest.approx(total, rel=0.01)


def test_two_feature_closed_forms_match_quadrature():
    z = lambda v: torch.tensor([v], dtype=torch.float64)
    r, d = 0.6, 0.35
    sig2 = d / 3 - d * d / 4
    # 대척점, b=0: d^2 (1+r)/6
    C = mi.population_loss_2feat(z(1.0), z(-1.0), z(0.0), z(0.0), z(r), z(d)).item()
    assert C == pytest.approx(d * d * (1 + r) / 6, rel=2e-3)
    # 특징 1 전담, b2 = E[x2]: r sigma^2
    A = mi.population_loss_2feat(z(1.0), z(0.0), z(0.0), z(d / 2), z(r), z(d)).item()
    assert A == pytest.approx(r * sig2, rel=2e-3)


def test_b0_boundary_is_where_antipodal_ties_best_single_feature():
    for r in (0.3, 0.7, 1.0, 1.5, 3.0):
        d = mi.b0_boundary_density(r)
        t = mi.three_config_losses(torch.tensor([r], dtype=torch.float64), torch.tensor([d], dtype=torch.float64), optimize_bias=False)
        lone = min(t["A"].item(), t["B"].item())
        assert t["C"].item() == pytest.approx(lone, rel=1e-9)


def test_dimensionality_sums_to_m_for_tight_frames_and_equals_m_over_k():
    # W W^T = c I (tight frame)이면 D_i = ||W_i||^2 / c, 합은 정확히 m (03 §4). 단위 벡터 k개면 D_i = m/k.
    k = 7
    ang = 2 * math.pi * torch.arange(k) / k
    W = torch.stack([ang.cos(), ang.sin()])
    assert mi.dimensionality(W).sum().item() == pytest.approx(2.0, abs=1e-5)
    # 직교 인자 둘의 합(텍검 곱): 삼각형(평면 1) + 대척점 쌍(축 3)
    tri = torch.tensor([[1.0, -0.5, -0.5], [0.0, math.sqrt(3) / 2, -math.sqrt(3) / 2]])
    W = torch.zeros(3, 5)
    W[:2, :3] = tri
    W[2, 3], W[2, 4] = 1.0, -1.0
    D = mi.dimensionality(W)
    assert torch.allclose(D, torch.tensor([2 / 3] * 3 + [0.5, 0.5]), atol=1e-5)
    assert mi.dims_per_feature(W).item() == pytest.approx(3 / 5)  # D* = m/||W||_F^2 = 3/5 = D_i 평균 (03 §4)
