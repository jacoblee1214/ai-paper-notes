"""minimal_impl.py 성질 시험. 실행: python -m pytest analysis/test_minimal_impl.py -q  (CPU, 약 1분)"""
import numpy as np
import pytest
import torch

import minimal_impl as mi

torch.set_num_threads(4)


# ---- 순수 선형대수: 식 (2), 부록 A ---------------------------------------------------------


def _memory(d_out=6, d_in=10, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(d_in, n, generator=g, dtype=torch.float64) * torch.rand(d_in, 1, generator=g, dtype=torch.float64).add(0.3)
    V = torch.randn(d_out, n, generator=g, dtype=torch.float64)
    W = V @ torch.linalg.pinv(K)  # 최소제곱 해: W K ~ V (부록 A식 (6)의 가정)
    return W, K, V, K @ K.T


def test_rank_one_update_hits_target_exactly():
    W, K, V, C = _memory()
    g = torch.Generator().manual_seed(1)
    k, v = torch.randn(10, generator=g, dtype=torch.float64), torch.randn(6, generator=g, dtype=torch.float64)
    dW = mi.rank_one_update(W, None, k, v, C)
    assert torch.linalg.matrix_rank(dW) == 1
    assert torch.allclose((W + dW) @ k, v, atol=1e-10)


def test_bias_is_accounted_for():
    W, K, V, C = _memory()
    b = torch.randn(6, dtype=torch.float64)
    k, v = torch.randn(10, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    dW = mi.rank_one_update(W, b, k, v, C)
    assert torch.allclose((W + dW) @ k + b, v, atol=1e-10)


def test_identity_C_gives_minimum_norm_update():
    W, K, V, C = _memory()
    k, v = torch.randn(10, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    dW = mi.rank_one_update(W, None, k, v, None)
    # Delta k = r 를 만족하는 최소 프로베니우스 노름 해는 r k^+ = r k^T/(k^T k)
    r = v - W @ k
    expected = r[:, None] @ torch.linalg.pinv(k[:, None])
    assert torch.allclose(dW, expected, atol=1e-10)
    # C = I 를 명시해도 같다
    assert torch.allclose(mi.rank_one_update(W, None, k, v, torch.eye(10, dtype=torch.float64)), dW, atol=1e-10)


def test_update_is_optimal_among_feasible_updates():
    W, K, V, C = _memory()
    g = torch.Generator().manual_seed(2)
    k, v = torch.randn(10, generator=g, dtype=torch.float64), torch.randn(6, generator=g, dtype=torch.float64)
    dW = mi.rank_one_update(W, None, k, v, C)
    cost = lambda D: torch.linalg.norm(D @ K) ** 2  # ||(W^ - W) K||_F^2
    P = torch.eye(10, dtype=torch.float64) - torch.outer(k, k) / (k @ k)
    for _ in range(20):
        # Delta' k = r 을 유지하는 임의의 다른 갱신 (Delta + E (I - k k^T/|k|^2))
        E = torch.randn(6, 10, generator=g, dtype=torch.float64) * 0.5
        assert cost(dW + E @ P) >= cost(dW) - 1e-9


def test_matches_constrained_least_squares_via_kkt():
    """||W^ K - V||_F^2 최소화, W^ k* = v* 를 라그랑주 연립(KKT)으로 직접 풀어 ROME 식과 비교한다."""
    W, K, V, C = _memory()
    g = torch.Generator().manual_seed(3)
    k, v = torch.randn(10, generator=g, dtype=torch.float64), torch.randn(6, generator=g, dtype=torch.float64)
    # 정상 방정식: W^ C + Lam k^T = V K^T,  W^ k = v  ->  [[C, k],[k^T, 0]] [W^^T; Lam^T] = [K V^T; v^T]
    d = 10
    A = torch.zeros(d + 1, d + 1, dtype=torch.float64)
    A[:d, :d], A[:d, d], A[d, :d] = C, k, k
    rhs = torch.cat([K @ V.T, v[None]], dim=0)
    sol = torch.linalg.solve(A, rhs)
    W_kkt = sol[:d].T
    W_rome = W + mi.rank_one_update(W, None, k, v, C)
    assert torch.allclose(W_kkt, W_rome, atol=1e-8)


def test_scale_of_C_does_not_matter():
    W, K, V, C = _memory()
    k, v = torch.randn(10, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    assert torch.allclose(mi.rank_one_update(W, None, k, v, C), mi.rank_one_update(W, None, k, v, 7.3 * C), atol=1e-10)


def test_keys_C_orthogonal_to_new_key_are_untouched():
    """(C^-1 k*)^T k_i = 0 이면 Delta k_i = 0: 기존 키의 출력이 그대로다."""
    W, K, V, C = _memory()
    k, v = torch.randn(10, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    u = torch.linalg.solve(C, k)
    ki = torch.randn(10, dtype=torch.float64)
    ki = ki - (u @ ki) / (u @ u) * u
    dW = mi.rank_one_update(W, None, k, v, C)
    assert torch.allclose(dW @ ki, torch.zeros(6, dtype=torch.float64), atol=1e-10)


# ---- 작은 모델에서 인과 추적과 편집 -----------------------------------------------------------


@pytest.fixture(scope="module")
def toy():
    w = mi.World(seed=0, n_subj=60, n_first=12, n_last=12)
    m = mi.train_model(w, seed=0, d=64, n_layer=4, n_head=4, steps=800, bs=256, lr=3e-3)
    assert mi.fact_accuracy(m, w) > 0.97, "토이가 사실을 외우지 못했다"
    return w, m


def _facts(w, n=40, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, w.n_subj, n), rng.randint(0, w.n_rel, n)


def test_prompt_layout():
    w = mi.World(seed=0)
    tok, last, pf, pl = w.build([5], [2], [1], [2], np.array([[w.fill0, w.fill0 + 1, w.fill0 + 2]]))
    t = tok[0].tolist()
    assert t[:2] == [w.fill0, w.fill0 + 1]
    assert t[2:4] == w.subj[5].tolist() and int(pf[0]) == 2 and int(pl[0]) == 3
    assert t[4:6] == [w.relB0 + 4, w.relB0 + 5] and int(last[0]) == 5 and t[6] == w.PAD


def test_zero_noise_gives_zero_indirect_effect(toy):
    w, m = toy
    s, r = _facts(w, 10)
    res = mi.causal_trace(m, w, s, r, noise_mult=0.0, n_draws=2, windows=(1,), sever=False)
    assert np.allclose(res["p_corrupt"], res["p_clean"], atol=1e-6)
    for k, v in res["ie"].items():
        assert np.abs(v).max() < 1e-5, k


def test_restoring_final_state_recovers_clean_probability(toy):
    """마지막 층, 마지막 토큰의 잔차 상태를 되돌리면 출력이 깨끗한 실행과 같아 IE = TE (최대)."""
    w, m = toy
    s, r = _facts(w, 20)
    res = mi.causal_trace(m, w, s, r, n_draws=4, windows=(1,), sever=False)
    L, T = res["L"], res["T"]
    ie = res["ie"]["resid"]
    te = res["p_clean"] - res["p_corrupt"]
    assert te.mean() > 0.3, "잡음이 정답 확률을 충분히 낮추지 못했다"
    assert np.allclose(ie[:, L - 1, T - 1], te, atol=1e-5)
    assert ie.mean(0).max() <= te.mean() + 1e-4  # 평균 격자의 최댓값이 ATE(이 토이에서 경험적으로)


def test_restoring_all_positions_at_layer0_recovers_clean(toy):
    w, m = toy
    s, r = _facts(w, 10)
    rng = np.random.RandomState(0)
    fills = w.random_fills(rng, len(s))
    tok, last, pf, pl = w.build(s, r, np.zeros(len(s), int), np.full(len(s), 2), fills)
    clean = {}
    hooks = {"resid0": lambda x: clean.__setitem__("h", x.clone()) or x}
    p_clean = mi.probs_for(m, tok, last, hooks)
    sigma = mi.embedding_std(m, w)
    noise = torch.randn(len(s), 2, m.d) * 3 * sigma
    ar = torch.arange(len(s))

    def emb(x):
        x = x.clone()
        x[ar, pf] += noise[:, 0]
        x[ar, pl] += noise[:, 1]
        return x

    p_corrupt = mi.probs_for(m, tok, last, {"embed": emb})
    p_restored = mi.probs_for(m, tok, last, {"embed": emb, "resid0": lambda x: clean["h"]})
    assert torch.allclose(p_restored, p_clean, atol=1e-5)
    assert (p_corrupt - p_clean).abs().max() > 0.1


def test_edit_improves_efficacy_and_keeps_other_facts(toy):
    w, m = toy
    Cs, _ = mi.key_second_moment(m, w, (3000,))
    C = Cs[3000]
    rng = np.random.RandomState(5)
    before, after, nb, w0 = [], [], [], m.blocks[0].proj.weight.clone()
    for s, r in zip(*_facts(w, 6, seed=9)):
        o_old = int(w.obj[s, r])
        lo, hi = w.obj_range(r)
        o_new = int(rng.choice([x for x in range(lo, hi) if x != o_old]))
        ev0 = mi.evaluate_edit(m, w, s, r, o_old, o_new, seed=1)
        dW, _ = mi.rome_edit(m, w, 0, s, r, o_new, C[0], rng, pos="final", steps=100, lr=0.3, clamp=10.0)
        with mi.Edited(m, 0, dW):
            ev = mi.evaluate_edit(m, w, s, r, o_old, o_new, seed=1)
        before.append(ev0["es"])
        after.append(ev["es"])
        nb.append(ev["nb_near"])
    assert torch.equal(m.blocks[0].proj.weight, w0), "Edited 블록이 가중치를 되돌리지 않았다"
    assert np.mean(before) == 0.0
    assert np.mean(after) > 0.5
    assert np.mean(nb) > 0.9
