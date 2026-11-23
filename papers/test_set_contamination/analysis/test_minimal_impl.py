"""minimal_impl.py 검증. python3 -m pytest analysis/test_minimal_impl.py -q  (수십 초)

A. 검정 자체의 성질 (합성 로그우도 함수: 문항이 i.i.d.이고 로그우도가 위치에 의존)
B. 토이 n-gram LM의 정규화
C. 토이 LM에서의 1종 오류율, 오염 횟수별 검정력, 정준 순서 없음, 같은 계산량 비교
"""
import numpy as np
import pytest
from scipy import stats

from minimal_impl import (NgramLM, build_stream, make_background, make_benchmark, permutation_test,
                          permutation_z_test, shard_sizes, sharded_test, split_shards)

V = 256


# ---------------------------------------------------------------- A
def synthetic_loglik(weights):
    """문항(스칼라)이 i.i.d.일 때 위치 가중합. 순서에 의존하지만 H0(정준 순서가 무작위)에서는 교환 가능."""
    def ll(items):
        return float(np.dot(weights[: len(items)], np.asarray(items, dtype=float)))
    return ll


def test_shard_sizes_follow_algorithm_1():
    assert shard_sizes(1000, 50) == [20] * 50
    s = shard_sizes(1003, 50)
    assert sum(s) == 1003 and s[:3] == [21, 21, 21] and s[3:] == [20] * 47
    parts = split_shards(list(range(10)), 3)
    assert [len(p) for p in parts] == [4, 3, 3] and sum(parts, []) == list(range(10))


def test_permutation_p_is_exact_on_its_grid_under_null():
    """H0: 정준 순서도 균등 무작위. p는 {1/(m+1),...,1} 위 균등이라 P(p <= j/(m+1)) = j/(m+1) (동점 없음)."""
    rng = np.random.default_rng(0)
    m, T = 9, 2000
    ll = synthetic_loglik(rng.normal(size=20))
    ps = np.array([permutation_test(ll, list(rng.normal(size=20)), m, rng)["p"] for _ in range(T)])
    for j in (1, 2, 5):
        rate = (ps <= j / (m + 1) + 1e-12).mean()
        assert abs(rate - j / (m + 1)) < 4 * np.sqrt(j / 10 * (1 - j / 10) / T)
    assert ps.min() >= 1 / (m + 1) - 1e-12  # 하한 1/(m+1)


def test_permutation_ties_are_counted_conservatively():
    rng = np.random.default_rng(1)
    p = permutation_test(lambda items: 0.0, list(range(8)), 19, rng)["p"]
    assert p == 1.0  # 전부 동점이면 p=1 (엄격한 부등호였다면 1/20)


def test_sharded_type1_error_near_nominal_on_synthetic():
    rng = np.random.default_rng(2)
    ll = synthetic_loglik(rng.normal(size=200))
    T = 1500
    ps = np.array([sharded_test(ll, list(rng.normal(size=200)), 10, 25, rng)["p"] for _ in range(T)])
    assert abs((ps <= 0.05).mean() - 0.05) < 0.02
    assert abs((ps <= 0.10).mean() - 0.10) < 0.025
    assert stats.kstest(ps, "uniform").pvalue > 0.001


def test_ddof0_statistic_is_inflated_by_sqrt_r_over_r_minus_1():
    rng = np.random.default_rng(3)
    ll = synthetic_loglik(rng.normal(size=100))
    items = list(rng.normal(size=100))
    r = 10
    a = sharded_test(ll, items, r, 5, np.random.default_rng(7), ddof=1)
    b = sharded_test(ll, items, r, 5, np.random.default_rng(7), ddof=0)
    assert b["t"] == pytest.approx(a["t"] * np.sqrt(r / (r - 1)))
    assert b["p"] <= a["p"] if a["t"] > 0 else b["p"] >= a["p"]


def test_sharded_p_can_go_below_permutation_floor_at_equal_compute():
    """로그우도가 정준 순서에서 위치 의존 이득을 얻는 합성 모델: 순열 검정은 1/(m+1) 아래로 못 내려가고 샤드 검정은 내려간다."""
    rng = np.random.default_rng(4)
    n, m = 200, 25
    order_items = list(range(n))

    def ll2(seq):
        a = np.array(seq)
        return float(np.sum(a[1:] == a[:-1] + 1) * 1.0) + 0.1 * rng.normal()
    p_perm = permutation_test(ll2, order_items, m, rng)["p"]
    p_shard = sharded_test(ll2, order_items, 10, m, rng)["p"]
    assert p_perm == pytest.approx(1 / (m + 1))
    assert p_shard < 1e-6


# ---------------------------------------------------------------- B
def test_ngram_is_normalized():
    rng = np.random.default_rng(5)
    stream = build_stream(rng, make_background(rng, 5000, 32), make_benchmark(rng, 20, 32), 3)
    lm = NgramLM(32, 4, 1.0).fit(stream)
    ctx = np.array([3, 7, 0])
    tot = sum(np.exp(lm.logprob_tokens(np.append(ctx, w))[-1]) for w in range(32))
    assert tot == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- C
def toy_p(seed, k, canonical=True, n=200, r=10, m=25, extras=False):
    rng = np.random.default_rng(seed)
    bench = make_benchmark(rng, n, V)
    stream = build_stream(rng, make_background(rng, 100_000, V), bench, k, canonical)
    lm = NgramLM(V, 4, 0.01, seed).fit(stream)
    items = bench if canonical else [bench[i] for i in rng.permutation(n)]
    out = {"sharded": sharded_test(lm.loglik_items, items, r, m, rng)["p"]}
    if extras:
        out["perm"] = permutation_test(lm.loglik_items, items, m, rng)["p"]
    return out


@pytest.fixture(scope="module")
def toy():
    T = 30
    res = {k: [toy_p(1000 * k + s, k, extras=True) for s in range(T)] for k in (0, 4, 16)}
    return res


def test_toy_power_increases_with_duplication(toy):
    med = {k: np.median([x["sharded"] for x in v]) for k, v in toy.items()}
    assert med[16] < med[4] < med[0]
    assert np.mean([x["sharded"] < 0.05 for x in toy[16]]) >= 0.9
    assert np.mean([x["sharded"] < 0.05 for x in toy[4]]) >= 0.5
    assert np.mean([x["sharded"] < 0.05 for x in toy[0]]) <= 0.2


def test_toy_equal_compute_sharded_beats_permutation_floor(toy):
    perm = np.array([x["perm"] for x in toy[16]])
    sh = np.array([x["sharded"] for x in toy[16]])
    assert perm.min() >= 1 / 26 - 1e-12          # m=25 순열 검정의 하한
    assert np.median(sh) < 1 / 26 / 10            # 샤드 검정은 하한보다 훨씬 작음


def test_toy_type1_error_without_contamination():
    ps = np.array([toy_p(5000 + s, 0)["sharded"] for s in range(300)])
    assert (ps <= 0.05).mean() <= 0.09          # 보수적이어도 됨, 넘치면 안 됨
    assert (ps <= 0.01).mean() <= 0.03


def test_toy_no_detection_when_training_order_was_shuffled():
    """16번 넣었지만 복사본마다 다른 순서: 문항은 외웠어도 정준 순서가 없다. 검정은 검출하지 않아야 한다."""
    ps = np.array([toy_p(9000 + s, 16, canonical=False)["sharded"] for s in range(300)])
    assert (ps <= 0.05).mean() <= 0.09


def test_permutation_z_test_runs():
    rng = np.random.default_rng(6)
    ll = synthetic_loglik(rng.normal(size=50))
    out = permutation_z_test(ll, list(rng.normal(size=50)), 30, rng)
    assert 0 <= out["p"] <= 1
