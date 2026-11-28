"""minimal_impl.py 시험(CPU, float64, 수 초). 실행: python3 -m pytest test_minimal_impl.py -q"""
import pytest
import torch

from minimal_impl import (dampened, gptq, grid_params, hessian, layer_error, obq_fixed_order,
                          obq_greedy, quantize, rtn)

DT = torch.float64


def make(d_row, d_col, n, seed, corr=True):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_row, d_col, generator=g, dtype=DT)
    X = torch.randn(n, d_col, generator=g, dtype=DT)
    if corr:  # 열마다 크기가 다르고 서로 상관이 있는 활성
        A = torch.randn(d_col, d_col, generator=g, dtype=DT) / d_col ** 0.5 + torch.eye(d_col, dtype=DT)
        X = (X @ A) * torch.logspace(-0.5, 0.5, d_col, dtype=DT)
    return W, X


# ---- 식 2: OBS 한 단계가 정말 '나머지 가중치의 최적 보상'인가 ----
def test_obs_update_is_exact_constrained_minimizer():
    W, X = make(1, 10, 200, 0)
    H = hessian(X)
    w = W[0]
    q, target = 3, 0.7  # w_q를 target으로 고정
    Hinv = torch.linalg.inv(H)
    delta = -(w[q] - target) / Hinv[q, q] * Hinv[:, q]
    w_obs = w + delta
    assert abs(w_obs[q] - target) < 1e-12
    # 직접 풀기: w_q = target으로 고정한 채 ||(w'-w)X||² 최소화 -> 나머지 열에 대한 최소제곱
    rest = [i for i in range(10) if i != q]
    Xr, xq = X[:, rest], X[:, q]
    # 최소화할 것: ||Xr (w'_r - w_r) + xq (target - w_q)||²
    sol = torch.linalg.lstsq(Xr, -xq * (target - w[q])).solution
    w_ref = w.clone()
    w_ref[rest] = w[rest] + sol
    w_ref[q] = target
    assert torch.allclose(w_obs, w_ref, atol=1e-9)


def test_eq3_rank_one_update_equals_submatrix_inverse():
    _, X = make(1, 9, 100, 1)
    H = hessian(X)
    Hinv = torch.linalg.inv(H)
    q = 4
    upd = Hinv - torch.outer(Hinv[:, q], Hinv[q, :]) / Hinv[q, q]
    keep = [i for i in range(9) if i != q]
    assert torch.allclose(upd[keep][:, keep], torch.linalg.inv(H[keep][:, keep]), atol=1e-10)


def test_cholesky_rows_are_scaled_inverse_rows():
    """U = chol(H⁻¹, upper)의 j행 = H_F⁻¹의 j행 / √[H_F⁻¹]_jj, F = {j, ..., d-1} (Step 3의 핵심)."""
    _, X = make(1, 8, 100, 2)
    H = dampened(hessian(X), 0.01)
    U = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)
    for j in range(8):
        HF = torch.linalg.inv(H[j:, j:])
        assert torch.allclose(U[j, j:], HF[0] / HF[0, 0].sqrt(), atol=1e-10)
        assert torch.allclose(U[j, :j], torch.zeros(j, dtype=DT))


# ---- GPTQ == 순차 OBQ(같은 열 순서) ----
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("block", [1, 3, 5, 100])
def test_gptq_equals_sequential_obq(bits, block):
    W, X = make(4, 12, 80, 3)
    H = hessian(X)
    Q, _ = gptq(W, H, bits, block=block, damp=0.01)
    Qn = obq_fixed_order(W, H, bits, range(12), damp=0.01)
    assert torch.allclose(Q, Qn, atol=1e-8)


def test_gptq_equals_sequential_obq_with_permutation():
    W, X = make(4, 12, 80, 4)
    H = hessian(X)
    perm = torch.argsort(torch.diag(H), descending=True).tolist()
    Q, _ = gptq(W, H, 3, block=4, damp=0.01, act_order=True)
    Qn = obq_fixed_order(W, H, 3, perm, damp=0.01)
    assert torch.allclose(Q, Qn, atol=1e-8)


@pytest.mark.parametrize("group,block", [(4, 4), (4, 8), (2, 8), (8, 8)])
def test_gptq_equals_sequential_obq_with_groups(group, block):
    W, X = make(4, 16, 80, 5)
    H = hessian(X)
    Q, _ = gptq(W, H, 3, block=block, damp=0.01, group=group)
    Qn = obq_fixed_order(W, H, 3, range(16), damp=0.01, group=group)
    assert torch.allclose(Q, Qn, atol=1e-8)


def test_stale_group_params_differs_only_when_group_smaller_than_block():
    W, X = make(6, 16, 80, 6)
    H = hessian(X)
    cur, _ = gptq(W, H, 3, block=8, group=4)
    stale, _ = gptq(W, H, 3, block=8, group=4, stale_group_params=True)
    assert not torch.allclose(cur, stale)
    cur, _ = gptq(W, H, 3, block=8, group=8)
    stale, _ = gptq(W, H, 3, block=8, group=8, stale_group_params=True)
    assert torch.allclose(cur, stale)


# ---- 예측 손실, 격자, 보조 성질 ----
def test_predicted_loss_equals_dampened_objective():
    """Σ (w-q)²/U_jj²/2 = ½ tr(ΔW (H+λI) ΔW ᵀ) (식 2의 손실이 단계마다 정확히 더해짐)."""
    W, X = make(5, 20, 120, 7)
    H = hessian(X)
    Q, pred = gptq(W, H, 3, block=6, damp=0.01)
    D = W - Q
    true = 0.5 * torch.trace(D @ dampened(H, 0.01) @ D.T).item()
    assert pred == pytest.approx(true, rel=1e-9)
    # dampening을 빼면 H = 2/n XᵀX 이므로 ½ΔW H ΔWᵀ = mean_n ||ΔW x||²
    assert 0.5 * torch.trace(D @ H @ D.T).item() == pytest.approx(layer_error(W, Q, X), rel=1e-9)


def test_output_on_grid():
    W, X = make(6, 32, 100, 8)
    Q, _ = gptq(W, hessian(X), 3, block=8)
    for r in range(6):
        assert len(torch.unique(Q[r])) <= 8
    s, z = grid_params(W, 3)
    assert torch.allclose(quantize(Q, s, z, 3), Q, atol=1e-10)  # 같은 격자 위의 점이다


def test_high_bits_recovers_weights():
    W, X = make(4, 16, 100, 9)
    Q, _ = gptq(W, hessian(X), 16)
    assert (W - Q).abs().max() < 1e-3


def test_dead_columns_zeroed_and_finite():
    W, X = make(4, 16, 100, 10)
    X[:, 5] = 0
    Q, _ = gptq(W, hessian(X), 4, block=4)
    assert torch.isfinite(Q).all() and (Q[:, 5] == 0).all()


def test_block_size_does_not_change_result():
    W, X = make(8, 40, 200, 11)
    H = hessian(X)
    a, _ = gptq(W, H, 4, block=1)
    b, _ = gptq(W, H, 4, block=16)
    c, _ = gptq(W, H, 4, block=40)
    assert torch.allclose(a, b, atol=1e-8) and torch.allclose(a, c, atol=1e-8)


def test_rtn_matches_manual_rounding():
    W = torch.tensor([[-1.0, 0.2, 0.9, 1.0]], dtype=DT)
    Q = rtn(W, 2)  # min=-1, max=1, scale=2/3, zero=round(1.5)=2 (파이토치는 짝수로 반올림)
    s, z = grid_params(W, 2)
    assert s.item() == pytest.approx(2 / 3) and z.item() == 2
    assert torch.allclose(Q, s * (torch.clamp(torch.round(W / s) + z, 0, 3) - z))


# ---- 목적함수 비교 ----
def test_gptq_beats_rtn_on_correlated_activations():
    wins = 0
    for seed in range(10):
        W, X = make(32, 64, 400, 100 + seed)
        H = hessian(X)
        Q, _ = gptq(W, H, 3, block=16)
        wins += layer_error(W, Q, X) < layer_error(W, rtn(W, 3), X)
    assert wins == 10


def test_gptq_close_to_greedy_obq_and_both_beat_rtn():
    W, X = make(6, 24, 300, 12)
    H = hessian(X)
    e_rtn = layer_error(W, rtn(W, 3), X)
    e_gptq = layer_error(W, gptq(W, H, 3, block=8)[0], X)
    e_obq = layer_error(W, obq_greedy(W, H, 3), X)
    assert e_gptq < e_rtn and e_obq < e_rtn
    assert e_gptq < 1.5 * e_obq  # 임의 순서가 탐욕 순서에 크게 뒤지지 않는다(작은 층, 여유 있는 경계)


def test_group_grid_improves_rtn_error():
    W, X = make(16, 64, 300, 13)
    assert layer_error(W, rtn(W, 3, group=16), X) < layer_error(W, rtn(W, 3), X)


def test_greedy_obq_with_eq3_matches_replay_of_its_order_with_explicit_inverse():
    """식 3(H⁻¹ 한 단계 소거)로 굴린 탐욕 OBQ == 같은 순서를 H_F를 매번 새로 뒤집어 재생한 순차 OBQ."""
    W, X = make(1, 10, 200, 14)
    H = hessian(X)
    orders = []
    Qg = obq_greedy(W, H, 3, orders=orders)
    Qr = obq_fixed_order(W, H, 3, orders[0])
    assert torch.allclose(Qg, Qr, atol=1e-8)
