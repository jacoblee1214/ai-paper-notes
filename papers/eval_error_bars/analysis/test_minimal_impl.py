"""minimal_impl.py의 성질을 산술과 몬테카를로로 확인한다(전체 수십 초). 실행: python3 -m pytest -q analysis"""
import numpy as np
import pytest
from scipy import stats

import minimal_impl as m


# ------------------------------------------------------------ 식 (1)-(3)
def test_bernoulli_is_clt_scaled():
    rng = np.random.default_rng(0)
    s = (rng.random(200) < 0.3).astype(float)
    n = len(s)
    assert m.se_clt(s) == pytest.approx(m.se_bernoulli(s) * np.sqrt(n / (n - 1)))


def test_bernoulli_se_wider_than_clt_for_fractional_scores():
    """2.1절: F1처럼 분수 점수에 식 (2)를 쓰면 보수적(과대)이다."""
    rng = np.random.default_rng(1)
    s = rng.beta(5, 2, size=2000)  # 0~1 연속 점수
    assert m.se_bernoulli(s) > 1.3 * m.se_clt(s)


def test_ci95_coverage_iid():
    rng = np.random.default_rng(2)
    reps, n, p = 4000, 300, 0.6
    s = rng.random((reps, n)) < p
    mu = s.mean(1)
    se = np.sqrt(s.var(1, ddof=1) / n)
    cov = np.mean(np.abs(mu - p) < m.Z975 * se)
    assert 0.935 < cov < 0.965


# ------------------------------------------------------------ 식 (4): 군집
def _clustered_scores(rng, reps, g, size, icc):
    """군집 효과 + 개별 잡음. 문항 점수 분산 1, 군집 내 상관 icc, 참 평균 0."""
    c = rng.normal(0, np.sqrt(icc), (reps, g, 1))
    e = rng.normal(0, np.sqrt(1 - icc), (reps, g, size))
    return (c + e).reshape(reps, g * size)


def test_clustered_se_restores_coverage():
    rng = np.random.default_rng(3)
    reps, g, size, icc = 3000, 80, 10, 0.5      # 설계 효과 1+9*0.5 = 5.5
    x = _clustered_scores(rng, reps, g, size, icc)
    cl = np.repeat(np.arange(g), size)
    naive_ok = clu_ok = 0
    for r in range(reps):
        mu = x[r].mean()
        naive_ok += abs(mu) < m.Z975 * m.se_clt(x[r])
        clu_ok += abs(mu) < m.Z975 * m.se_clustered(x[r], cl)
    assert naive_ok / reps < 0.75               # 과소 포함
    assert 0.925 < clu_ok / reps < 0.965        # 약 0.95 (군집 80개)


def test_design_effect_matches_theory():
    rng = np.random.default_rng(4)
    g, size, icc = 400, 10, 0.5
    x = _clustered_scores(rng, 1, g, size, icc)[0]
    cl = np.repeat(np.arange(g), size)
    assert m.design_effect(x, cl) == pytest.approx(1 + (size - 1) * icc, rel=0.15)


def test_clustered_equals_naive_when_each_question_is_its_own_cluster():
    rng = np.random.default_rng(5)
    s = rng.normal(size=100)
    assert m.se_clustered(s, np.arange(100)) == pytest.approx(m.se_clt(s))


def test_clustered_cross_term_matches_explicit_triple_sum():
    rng = np.random.default_rng(6)
    s = rng.normal(size=12)
    cl = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    u = s - s.mean()
    cross = sum(u[i] * u[j] for i in range(12) for j in range(12) if i != j and cl[i] == cl[j])
    expect = np.sqrt(m.se_clt(s) ** 2 + cross / 144)
    assert m.se_clustered(s, cl) == pytest.approx(expect)


# ------------------------------------------------------------ 4절: 짝지은 차이
def _two_models(rng, reps, n, rho):
    """난이도 x는 U[0,1] (가우시안 코퓰라로 두 모델의 Pearson 상관 rho), 점수는 Bernoulli(x)."""
    r = 2 * np.sin(np.pi * rho / 6)             # Pearson rho를 주는 코퓰라 상관
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=(reps, n))
    xa, xb = stats.norm.cdf(z[..., 0]), stats.norm.cdf(z[..., 1])
    return (rng.random((reps, n)) < xa).astype(float), (rng.random((reps, n)) < xb).astype(float), xa, xb


def test_paired_formula_equals_correlation_form():
    rng = np.random.default_rng(7)
    a, b, _, _ = _two_models(rng, 1, 500, 0.5)
    assert m.se_paired(a[0], b[0]) == pytest.approx(m.se_paired_from_corr(a[0], b[0]))


def test_paired_se_smaller_and_matches_empirical_sd():
    rng = np.random.default_rng(8)
    reps, n = 4000, 400
    a, b, _, _ = _two_models(rng, reps, n, 0.5)
    emp_sd = (a.mean(1) - b.mean(1)).std(ddof=1)
    se_p = np.mean([m.se_paired(a[r], b[r]) for r in range(reps)])
    se_u = np.mean([m.se_unpaired(a[r], b[r]) for r in range(reps)])
    assert se_p < se_u
    assert se_p == pytest.approx(emp_sd, rel=0.04)      # 짝지은 SE가 실제 변동과 일치
    # 비짝 SE는 과대: 이진 점수의 상관은 1/6이라 분산비 (1/2)/(5/12) = 1.2
    assert se_u == pytest.approx(emp_sd * np.sqrt(1.2), rel=0.04)


def test_paired_variance_reduction_is_rho_not_one_third():
    """4.2절 예: U[0,1] 두 점수, 상관 0.5, 조건부 분산 0. 논문은 1/6 -> 1/9(1/3 감소)라 했지만
    Var(sA)+Var(sB)-2Cov = 1/12+1/12-2/24 = 1/12 (1/2 감소 = rho)이다."""
    rng = np.random.default_rng(9)
    _, _, xa, xb = _two_models(rng, 1, 400000, 0.5)
    xa, xb = xa[0], xb[0]
    assert np.var(xa - xb) == pytest.approx(1 / 12, rel=0.02)
    assert np.var(xa - xb) / (np.var(xa) + np.var(xb)) == pytest.approx(0.5, rel=0.03)


def test_paired_clustered_matches_manual_and_coverage():
    rng = np.random.default_rng(10)
    s = rng.normal(size=60)
    t = rng.normal(size=60)
    cl = np.repeat(np.arange(12), 5)
    d = s - t
    u = d - d.mean()
    manual = np.sqrt(sum(u[cl == c].sum() ** 2 for c in range(12))) / 60
    assert m.se_paired_clustered(s, t, cl) == pytest.approx(manual)
    # 군집 내 상관이 있는 짝지은 차이에서 포함률 (군집 100개)
    reps, g, size = 2000, 100, 8
    dd = _clustered_scores(rng, reps, g, size, 0.4)
    c = np.repeat(np.arange(g), size)
    ok = np.mean([abs(x.mean()) < m.Z975 * m.se_paired_clustered(x, np.zeros_like(x), c) for x in dd])
    assert 0.92 < ok < 0.965


# ------------------------------------------------------------ 3절: K번 샘플링
def _binary_uniform_k(rng, reps, n, k):
    x = rng.random((reps, n, 1))
    return (rng.random((reps, n, k)) < x).astype(float)


def test_resampling_reduces_only_within_question_variance():
    rng = np.random.default_rng(11)
    reps, n = 6000, 200
    sc = {k: _binary_uniform_k(rng, reps, n, k) for k in (1, 2, 4, 6)}
    v1 = sc[1].mean((1, 2)).var()
    for k in (2, 4, 6):
        ratio = sc[k].mean((1, 2)).var() / v1
        assert ratio == pytest.approx((1 + 2 / k) / 3, rel=0.08)      # 3.1절 식
    # 하한: 문항 간 분산 1/12 / (1/4) = 1/3 (감소 상한 2/3)
    assert m.variance_ratio_vs_k1(1 / 12, 1 / 6, 10 ** 6) == pytest.approx(1 / 3, abs=1e-5)


def test_variance_components_recovered():
    rng = np.random.default_rng(12)
    sc = _binary_uniform_k(rng, 1, 20000, 8)[0]
    vx, es = m.variance_components(sc)
    assert vx == pytest.approx(1 / 12, rel=0.05)
    assert es == pytest.approx(1 / 6, rel=0.03)
    # 순진한 Var(xhat)은 vx를 es/K만큼 과대평가한다
    assert sc.mean(1).var(ddof=1) == pytest.approx(vx + es / 8, rel=1e-6)


def test_pooled_se_over_all_samples_is_too_small():
    """3.1절 경고: KN개 답을 한꺼번에 풀링하면 독립 가정 위반. 문항 평균 위의 SE가 맞다."""
    rng = np.random.default_rng(13)
    reps, n, k = 3000, 100, 5
    sc = _binary_uniform_k(rng, reps, n, k)
    emp = sc.mean((1, 2)).std(ddof=1)
    pooled = np.mean([m.se_clt(s.ravel()) for s in sc[:300]])
    qlevel = np.mean([m.se_clt(s.mean(1)) for s in sc[:300]])
    assert pooled < 0.8 * emp
    assert qlevel == pytest.approx(emp, rel=0.05)


# ------------------------------------------------------------ 5절: 검정력
def test_sample_size_arithmetic_from_paper():
    assert m.n_required(0.03, 1 / 9) == pytest.approx(969, abs=0.5)      # 5절 예 (논문 값)
    # 논문의 ω²=1/9는 4.2절 산술 오류에서 온 값이다. 같은 조건의 올바른 ω²=1/12이면 727.
    assert m.n_required(0.03, 1 / 12) == pytest.approx(727, abs=0.5)
    assert m.mde(198, 1 / 9, 1 / 6, 1 / 6, 1, 1) == pytest.approx(0.132, abs=0.001)    # 논문 13.2%
    assert m.mde(198, 1 / 9, 1 / 6, 1 / 6, 10, 10) == pytest.approx(0.075, abs=0.002)  # 논문 7.5%


def test_simulated_power_near_target_at_required_n():
    """식 (9)로 정한 n에서 z 검정의 시뮬레이션 검정력은 0.8 근처(t/z 차이로 약간 낮음)."""
    rng = np.random.default_rng(14)
    delta, omega2, sig2, k = 0.03, 1 / 12, 1 / 6, 3
    n = int(np.ceil(m.n_required(delta, omega2, sig2, sig2, k, k)))
    reps = 4000
    # 차이 = 난이도 차이(분산 omega2) + 두 모델의 문항 안 잡음(각 sig2/k)
    d = delta + rng.normal(0, np.sqrt(omega2 + 2 * sig2 / k), (reps, n))
    z = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
    power = np.mean(np.abs(z) > m.Z975)
    assert 0.77 < power < 0.83
    assert m.power_z(n, delta, omega2, sig2, sig2, k, k) == pytest.approx(0.8, abs=0.01)


def test_clustered_sample_size_power():
    """부록 C: omega²_clustered를 식 (9)에 넣어 정한 문항 수에서 군집 SE 검정의 검정력이 0.8 근처."""
    rng = np.random.default_rng(15)
    size, icc, delta = 10, 0.5, 0.15
    omega2_cl = 1 + (size - 1) * icc                      # Var(d)=1, 설계 효과 5.5
    n = int(np.ceil(m.n_required(delta, omega2_cl)))
    n = size * int(np.ceil(n / size))
    g = n // size
    reps = 1500
    d = _clustered_scores(rng, reps, g, size, icc) + delta
    cl = np.repeat(np.arange(g), size)
    zs = [x.mean() / m.se_paired_clustered(x, np.zeros_like(x), cl) for x in d]
    power = np.mean(np.abs(zs) > m.Z975)
    assert 0.74 < power < 0.86


def test_omega2_estimators_unbiased_with_noise_correction():
    rng = np.random.default_rng(16)
    n, k = 20000, 4
    a = _binary_uniform_k(rng, 1, n, k)[0]
    b = _binary_uniform_k(rng, 1, n, k)[0]             # 독립 -> omega² = 1/12 + 1/12 = 1/6
    raw = (a.mean(1) - b.mean(1)).var(ddof=1)
    assert m.omega2_paired(a, b) == pytest.approx(1 / 6, rel=0.04)
    assert raw > m.omega2_paired(a, b) + 0.03          # 보정 없이 쓰면 sigma²/K 만큼 과대
    cl = np.arange(n)
    assert m.omega2_clustered(a, b, cl) == pytest.approx(1 / 6, rel=0.06)


def test_stratified_se_removes_between_strata_variance():
    """층 평균이 크게 다르면 식 (1)은 층 간 분산까지 세므로 과대, 층화 SE가 층별 고정 표집의 실제 변동과 맞다."""
    rng = np.random.default_rng(17)
    sizes, probs = [100, 300, 600], [0.05, 0.3, 0.7]
    strata = np.repeat(np.arange(3), sizes)
    reps = 3000
    draws = np.concatenate([rng.random((reps, n)) < p for n, p in zip(sizes, probs)], axis=1).astype(float)
    emp = draws.mean(1).std(ddof=1)
    se_s = np.mean([m.se_stratified(d, strata) for d in draws[:500]])
    se_n = np.mean([m.se_clt(d) for d in draws[:500]])
    assert se_s == pytest.approx(emp, rel=0.05)
    assert se_n > 1.1 * se_s
