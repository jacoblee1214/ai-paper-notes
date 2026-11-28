"""minimal_impl.py 검증. 합성 범주형 분포에서 정확한 분포와 표본 분포를 비교한다(카이제곱), 논문 식·표를 확인한다.
실행: python3 -m pytest analysis/test_minimal_impl.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as M

torch.set_default_dtype(torch.float64)
P_MIN = 1e-3   # 검정 하나의 기각 수준. 시드를 고정하므로 결과는 결정적이다.


def dirichlet(V, conc, gen):
    x = torch._standard_gamma(torch.full((V,), conc), generator=gen)
    return x / x.sum()


def chi2_p(counts, probs):
    """기대 빈도 5 미만인 칸은 하나로 묶어 카이제곱 p값."""
    n = counts.sum()
    exp = probs * n
    big = exp >= 5
    c = np.append(counts[big], counts[~big].sum())
    e = np.append(exp[big], exp[~big].sum())
    if e[-1] < 5:
        c, e = c[:-1], e[:-1]
    stat = ((c - e) ** 2 / e).sum()
    return stats.chi2.sf(stat, len(c) - 1)


def sample_counts(p, q, N, seed, **kw):
    gen = torch.Generator().manual_seed(seed)
    x, acc = M.speculative_sampling(p.expand(N, -1), q.expand(N, -1), gen, **kw)
    return torch.bincount(x, minlength=p.numel()).numpy(), acc.double().mean().item()


# ------------------------------------------------------------ 식
def test_alpha_is_one_minus_tv_and_lemma_3_3():
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        p, q = dirichlet(10, 0.4, g), dirichlet(10, 0.4, g)
        a = M.alpha_from_dists(p, q)
        assert abs(a - (1 - M.d_lk(p, q))) < 1e-12                           # Lemma 3.3
        assert abs(a - (1 - 0.5 * (p - q).abs().sum())) < 1e-12              # D_LK = TV
        assert abs(M.d_lk(p, q) - M.d_lk(q, p)) < 1e-12                      # 대칭 (Cor 3.4)
    assert abs(M.alpha_from_dists(p, p) - 1) < 1e-12
    e1, e2 = torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0])
    assert M.alpha_from_dists(e1, e2) == 0 and M.d_lk(e1, e2) == 1           # 서로소 지지집합


def test_expected_tokens_matches_capped_geometric_sum():
    for alpha in (0.0, 0.3, 0.6, 0.9, 0.999):
        for gamma in (1, 3, 7, 12):
            pmf = M.capped_geometric_pmf(alpha, gamma)
            assert abs(sum(pmf) - 1) < 1e-12
            brute = sum((k + 1) * pk for k, pk in enumerate(pmf))            # n + 1 개 생성
            assert abs(brute - M.expected_tokens(alpha, gamma)) < 1e-9
    assert M.expected_tokens(1.0, 5) == 6 and M.expected_tokens(0.0, 5) == 1


def test_table1_values():
    # 논문 Table 1 (c = c_hat = 0): (alpha, gamma, operations, speed)
    rows = [(0.6, 2, 1.53, 1.96), (0.7, 3, 1.58, 2.53), (0.8, 2, 1.23, 2.44),
            (0.8, 5, 1.63, 3.69), (0.9, 2, 1.11, 2.71), (0.9, 10, 1.60, 6.86)]
    for a, g, ops, speed in rows:
        assert abs(M.ops_factor(a, g, 0) - ops) < 0.005 + 1e-9
        assert abs(M.walltime_factor(a, g, 0) - speed) < 0.005 + 1e-9


def test_walltime_corollary_3_9_and_gamma_one():
    # Cor 3.9: gamma = 1에서 (1+alpha)/(1+c); 개선이 생기는 조건은 alpha > c
    for a, c in [(0.5, 0.1), (0.5, 0.5), (0.3, 0.5), (0.9, 0.05)]:
        assert abs(M.walltime_factor(a, 1, c) - (1 + a) / (1 + c)) < 1e-12
        assert (M.walltime_factor(a, 1, c) > 1) == (a > c)
        best_g, best = M.optimal_gamma(a, c)
        if a > c:
            assert best >= (1 + a) / (1 + c) - 1e-12
    assert M.walltime_factor(0.7, 0, 0.3) == 1.0                             # gamma = 0은 기준선


def test_optimal_gamma_monotone_in_alpha_and_c():
    g_a = [M.optimal_gamma(a, 0.05)[0] for a in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert g_a == sorted(g_a)
    g_c = [M.optimal_gamma(0.8, c)[0] for c in (0.01, 0.05, 0.1, 0.3)]
    assert g_c == sorted(g_c, reverse=True)


def test_optimal_gamma_marginal_condition():
    # f(g) = E(g)/(1 + g c)에서 f(g+1) > f(g) <=> alpha^(g+1) (1 + g c) > c E(g). 이 조건이 처음 깨지는 g가 최적 gamma.
    for a in (0.3, 0.5, 0.7, 0.9, 0.95):
        for c in (0.005, 0.02, 0.1, 0.3):
            g = 1
            while a ** (g + 1) * (1 + g * c) > c * M.expected_tokens(a, g):
                g += 1
            assert g == M.optimal_gamma(a, c, gamma_max=500)[0]


# ------------------------------------------------------------ 토큰 하나: 정확성 (부록 A.1)
@pytest.mark.parametrize("name", ["random", "close", "q_uniform", "sharp_p", "q_zero_gap", "disjoint"])
def test_single_token_exact(name):
    g = torch.Generator().manual_seed(1)
    V = 8
    p, q = dirichlet(V, 0.5, g), dirichlet(V, 0.5, g)
    if name == "close":
        q = (p + 0.05 * dirichlet(V, 1.0, g)); q = q / q.sum()
    elif name == "q_uniform":
        q = torch.full((V,), 1 / V)
    elif name == "sharp_p":
        p = torch.softmax(torch.log(p) / 0.2, 0)
    elif name == "q_zero_gap":
        q = q.clone(); q[:3] = 0; q = q / q.sum()
    elif name == "disjoint":
        p = torch.tensor([.5, .3, .2, 0, 0, 0, 0, 0]); q = torch.tensor([0, 0, 0, .1, .2, .3, .2, .2])
    N = 400_000
    counts, acc_rate = sample_counts(p, q, N, seed=2)
    assert chi2_p(counts, p.numpy()) > P_MIN
    alpha = M.alpha_from_dists(p, q).item()
    assert abs(acc_rate - alpha) < 4 * math.sqrt(alpha * (1 - alpha) / N) + 1e-9


def test_residual_is_needed_wrong_variants_fail():
    g = torch.Generator().manual_seed(3)
    p, q = dirichlet(8, 0.5, g), dirichlet(8, 0.5, g)
    N = 400_000
    # 거절 시 p에서 다시 뽑기: P(x) = min(p,q) + (1-alpha) p != p
    counts, _ = sample_counts(p, q, N, 4, residual_fn=lambda p_, q_: p_)
    assert chi2_p(counts, p.numpy()) < 1e-6
    # 거절 시 q에서 다시 뽑기도 틀림
    counts, _ = sample_counts(p, q, N, 5, residual_fn=lambda p_, q_: q_)
    assert chi2_p(counts, p.numpy()) < 1e-6
    # 항상 받기 = q 그대로
    counts, _ = sample_counts(p, q, N, 6, lenience=0.0)
    assert chi2_p(counts, p.numpy()) < 1e-6 and chi2_p(counts, q.numpy()) > P_MIN


def test_edge_cases():
    N = 20_000
    g = torch.Generator().manual_seed(7)
    p = dirichlet(6, 0.5, g)
    gen = torch.Generator().manual_seed(8)
    x, acc = M.speculative_sampling(p.expand(N, -1), p.expand(N, -1), gen)      # q = p: 항상 받음
    assert acc.all()
    onehot = torch.zeros(6); onehot[2] = 1                                       # T = 0 목표: 항상 argmax
    q = dirichlet(6, 0.5, g)
    x, acc = M.speculative_sampling(onehot.expand(N, -1), q.expand(N, -1), gen)
    assert (x == 2).all() and abs(acc.double().mean().item() - q[2].item()) < 0.02
    x, acc = M.speculative_sampling(onehot.expand(N, -1), onehot.expand(N, -1), gen)
    assert acc.all() and (x == 2).all()
    wrong = torch.zeros(6); wrong[4] = 1                                         # 초안이 틀린 argmax: 항상 거절, 그래도 정답
    x, acc = M.speculative_sampling(onehot.expand(N, -1), wrong.expand(N, -1), gen)
    assert (~acc).all() and (x == 2).all()


def test_nonlenient_variant_a2_is_exact_but_less_efficient():
    # 부록 A.2: M = max p/q, 받을 확률 p/(M q), 거절하면 p에서 뽑기. 분포는 맞지만 받음률 1/M <= alpha.
    g = torch.Generator().manual_seed(9)
    for _ in range(20):
        p, q = dirichlet(8, 0.7, g), dirichlet(8, 0.7, g)
        Mx = (p / q).max()
        assert (1 / Mx) <= M.alpha_from_dists(p, q) + 1e-12


def test_lenience_bound_a5():
    # 부록 A.5: l <= 1, 받을 확률 min(1, p/(l q)), 거절 시 norm(max(0, p - l q)). 어떤 토큰도 p/l을 넘지 않는다.
    g = torch.Generator().manual_seed(10)
    for _ in range(100):
        p, q = dirichlet(10, 0.4, g), dirichlet(10, 0.4, g)
        for l in (1.0, 0.5, 0.3, 0.1):
            a_l = torch.minimum(q, p / l).sum()
            Pout = torch.minimum(q, p / l) + (1 - a_l) * M.residual(p, q, l)
            assert abs(Pout.sum() - 1) < 1e-9
            assert (Pout <= p / l + 1e-12).all()
            if l == 1.0:
                assert torch.allclose(Pout, p, atol=1e-12)


# ------------------------------------------------------------ Algorithm 1: 여러 토큰
def markov_fns(P, Q):
    """토큰 하나를 조건으로 하는 합성 마르코프 모델. ids[:, -1]이 직전 토큰."""
    draft = lambda ids: Q[ids[:, -1]]
    target = lambda ids, k: P[ids[:, -k:]]
    return draft, target


def random_markov(V, seed, conc=0.5):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([dirichlet(V, conc, g) for _ in range(V)])


@pytest.mark.parametrize("gamma", [0, 1, 2, 4])
def test_algorithm1_joint_distribution_exact(gamma):
    V, K, N = 4, 3, 300_000
    P, Q = random_markov(V, 11), random_markov(V, 12)
    draft, target = markov_fns(P, Q)
    gen = torch.Generator().manual_seed(100 + gamma)
    prefix = torch.full((N, 1), 1, dtype=torch.long)
    out = M.sample_first_k(draft, target, prefix, K, gamma, gen)
    cell = out[:, 0] * V * V + out[:, 1] * V + out[:, 2]
    counts = torch.bincount(cell, minlength=V ** 3).numpy()
    exact = torch.zeros(V ** 3)
    for a in range(V):
        for b in range(V):
            for c in range(V):
                exact[a * V * V + b * V + c] = P[1, a] * P[a, b] * P[b, c]
    assert chi2_p(counts, exact.numpy()) > P_MIN
    # 초안이 쓸모없는 경우(q 무작위)에도 분포는 같다는 것이 위 검정의 요점. 기준선(gamma=0)과도 같은 분포.


def test_iid_beta_gives_capped_geometric_and_eq1():
    # 문맥과 무관한 p, q면 beta가 모든 위치에서 같다 -> n은 정확히 capped geometric (식 1의 가정이 정확히 성립)
    V, gamma, N = 6, 5, 400_000
    g = torch.Generator().manual_seed(13)
    p, q = dirichlet(V, 0.6, g), dirichlet(V, 0.6, g)
    alpha = M.alpha_from_dists(p, q).item()
    draft = lambda ids: q.expand(ids.shape[0], -1)
    target = lambda ids, k: p.expand(ids.shape[0], k, -1)
    gen = torch.Generator().manual_seed(14)
    res = M.speculative_step(draft, target, torch.zeros(N, 1, dtype=torch.long), gamma, gen)
    counts = torch.bincount(res["n"], minlength=gamma + 1).numpy()
    assert chi2_p(counts, torch.tensor(M.capped_geometric_pmf(alpha, gamma)).numpy()) > P_MIN
    mean_tokens = (res["n"] + 1).double().mean().item()
    assert abs(mean_tokens - M.expected_tokens(alpha, gamma)) < 4 * (res["n"].double().std().item() / math.sqrt(N))
    assert torch.allclose(res["beta"], torch.full_like(res["beta"], alpha))


def test_greedy_matches_target_exactly_for_any_draft():
    # T = 0 (one-hot 분포): 초안이 무엇이든 출력이 목표의 argmax 경로와 같다
    V = 7
    P, Q = random_markov(V, 21), random_markov(V, 22)
    onehot = lambda T: torch.nn.functional.one_hot(T.argmax(-1), V).double()
    Pg, Qg = onehot(P), onehot(Q)
    draft, target = markov_fns(Pg, Qg)
    gen = torch.Generator().manual_seed(23)
    for start in range(V):
        seq, _ = M.speculative_decode(draft, target, torch.tensor([[start]]), 25, 4, gen)
        ref, cur = [], start
        for _ in range(25):
            cur = int(P[cur].argmax()); ref.append(cur)
        assert seq == ref


def test_step_shapes_and_bounds():
    V, B, gamma = 5, 64, 4
    P, Q = random_markov(V, 31), random_markov(V, 32)
    draft, target = markov_fns(P, Q)
    gen = torch.Generator().manual_seed(33)
    res = M.speculative_step(draft, target, torch.zeros(B, 3, dtype=torch.long), gamma, gen)
    assert res["n"].min() >= 0 and res["n"].max() <= gamma
    assert all(1 <= len(t) <= gamma + 1 for t in res["tokens"])                 # 한 번에 최소 1, 최대 gamma+1 토큰
    assert res["beta"].shape == (B, gamma)
    assert ((res["beta"] >= 0) & (res["beta"] <= 1 + 1e-12)).all()


def test_standardize():
    z = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    assert torch.allclose(M.standardize(z, 0), torch.tensor([[0, 0, 1.0, 0]]))
    assert torch.allclose(M.standardize(z, 1.0).sum(-1), torch.ones(1))
    pk = M.standardize(z, 1.0, top_k=2)
    assert (pk > 0).sum() == 2 and abs(pk.sum() - 1) < 1e-12 and pk[0, 2] > pk[0, 1]
    assert M.standardize(z, 0.1)[0, 2] > 0.99
