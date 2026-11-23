"""minimal_impl.py의 핵심 성질 검사. 실행: python3 -m pytest -q analysis/  (CPU, 1분 안팎)

몬테카를로 검사는 시드를 고정했다. 허용 범위는 관측 표본 오차(포함률 SE 약 1.3%p)보다 넉넉히 잡았다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_impl import (ToyWorld, average_score_profile, bootstrap_ci, iqm, mannwhitney_check,  # noqa: E402
                          mean_of_task_means, median_of_task_means, optimality_gap, performance_profile,
                          prob_improvement, prob_improvement_ci, stratified_resample)


# ------------------------------------------------------------- 정의와 식 검산
def test_iqm_matches_scipy_trim_mean_and_hand_example():
    x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 100.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])  # n = 12
    srt = np.sort(x.ravel())
    assert iqm(x) == pytest.approx(srt[3:9].mean())  # 12개에서 양끝 3개씩 -> 6개 평균
    assert iqm(x) == pytest.approx(scipy.stats.trim_mean(x, 0.25, axis=None))
    rng = np.random.default_rng(0)
    y = rng.normal(size=(7, 5))  # n=35, 4의 배수가 아님: 양끝 floor(8.75)=8개씩 -> 19개 남음(논문 floor(NM/2)=17과 다름)
    assert iqm(y) == pytest.approx(np.sort(y.ravel())[8:35 - 8].mean())
    assert 35 - 2 * int(0.25 * 35) == 19 and 35 // 2 == 17


def test_iqm_equals_clipped_area_of_profile():
    """§4.3 그림 8: X>=0이면 IQM = 2 * ∫ clip(F(τ) - 0.25, 0, 0.5) dτ  (n이 4의 배수일 때 경험분포에서 정확)."""
    rng = np.random.default_rng(1)
    x = rng.lognormal(size=(8, 5))  # n = 40
    s = np.sort(x.ravel())
    n = s.size
    edges = np.concatenate([[0.0], s])  # τ가 [edges[k], edges[k+1])에서 F = (n-k)/n
    f = (n - np.arange(n)) / n
    area = np.sum(np.clip(f - 0.25, 0, 0.5) * np.diff(edges))
    assert iqm(x) == pytest.approx(2 * area)


def test_mean_and_optimality_gap_are_profile_areas():
    rng = np.random.default_rng(2)
    x = rng.lognormal(sigma=0.8, size=(6, 7))
    s = np.sort(x.ravel())
    n = s.size
    edges = np.concatenate([[0.0], s])
    f = (n - np.arange(n)) / n
    assert x.mean() == pytest.approx(np.sum(f * np.diff(edges)))  # 평균 = 프로필 아래 면적
    gamma = 1.0
    top = np.minimum(edges, gamma)
    area_to_gamma = np.sum(f * np.diff(top))  # ∫_0^γ F
    assert optimality_gap(x, gamma) == pytest.approx(gamma - area_to_gamma)  # OG = γ - ∫_0^γ F = 프로필 위쪽 면적


def test_median_of_task_means_zero_scores():
    """절반 가까운 과제가 0점이어도 중앙값은 그대로, IQM은 크게 변한다(§4.3 서술)."""
    rng = np.random.default_rng(3)
    base = rng.uniform(1, 2, size=(5, 10))
    worse2 = base.copy()
    worse2[:, :2] = 0.0
    # 순위 5,6번째가 영향받지 않는 범위(2과제 0점)에서 중앙값 변화는 작고 IQM 변화는 더 크다
    d_med = abs(median_of_task_means(worse2) - median_of_task_means(base))
    d_iqm = abs(iqm(worse2) - iqm(base))
    assert d_iqm > d_med


# ------------------------------------------------------------- 견고성
def test_iqm_insensitive_to_single_outlier_while_mean_is_not():
    rng = np.random.default_rng(4)
    x = rng.lognormal(size=(10, 10))
    y = x.copy()
    y[0, 0] = x.max() * 1e4
    assert abs(iqm(y) - iqm(x)) < 0.2 * x.std()  # 한 점이 IQM을 크게 못 움직임
    assert abs(mean_of_task_means(y) - mean_of_task_means(x)) > 100 * abs(iqm(y) - iqm(x))


def test_profile_outlier_changes_at_most_one_over_MN():
    rng = np.random.default_rng(5)
    x = rng.lognormal(size=(4, 6))
    y = x.copy()
    y[2, 3] = 1e9
    taus = np.linspace(0, 5, 60)
    diff = np.abs(performance_profile(y, taus) - performance_profile(x, taus)).max()
    assert diff <= 1 / x.size + 1e-12


# ------------------------------------------------------------- 층화 부트스트랩
def test_stratified_resample_keeps_per_task_runs_and_never_mixes_tasks():
    n, m = 5, 4
    x = np.arange(m)[None, :] * 100.0 + np.arange(n)[:, None] * 0.1  # 과제 m의 값은 [100m, 100m+0.4]
    bs = stratified_resample(x, 300, np.random.default_rng(0))
    assert bs.shape == (300, n, m)  # 과제별 실행 수 유지
    for j in range(m):
        assert np.all(bs[:, :, j] >= 100 * j - 1e-9) and np.all(bs[:, :, j] <= 100 * j + 0.4 + 1e-9)  # 다른 과제 값이 안 섞임
        assert set(np.unique(bs[:, :, j])) <= set(x[:, j])
    # 과제별로 독립: 같은 반복에서 서로 다른 과제의 인덱스가 같지 않은 경우가 있다
    idx0 = np.rint((bs[:, :, 0] - 0) / 0.1).astype(int)
    idx1 = np.rint((bs[:, :, 1] - 100) / 0.1).astype(int)
    assert (idx0 != idx1).any()


def test_bootstrap_ci_reproducible_contains_estimate_and_shrinks_with_runs():
    w = ToyWorld()
    rng = np.random.default_rng(6)
    x = w.sample(5, rng)
    a = bootstrap_ci(x, iqm, reps=500, rng=np.random.default_rng(1))
    b = bootstrap_ci(x, iqm, reps=500, rng=np.random.default_rng(1))
    assert a[0] == b[0] and a[1][0] == b[1][0] and a[1][1] == b[1][1]
    assert a[1][0] <= a[0] <= a[1][1]
    wide = np.mean([np.subtract(*bootstrap_ci(w.sample(3, rng), iqm, 300, rng=rng)[1][::-1]) for _ in range(20)])
    narrow = np.mean([np.subtract(*bootstrap_ci(w.sample(30, rng), iqm, 300, rng=rng)[1][::-1]) for _ in range(20)])
    assert narrow < wide / 2


# ------------------------------------------------------------- 개선 확률
def test_prob_improvement_equals_mannwhitney_and_handles_ties():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(6, 9)), rng.normal(size=(4, 9)) + 0.3
    assert prob_improvement(x, y) == pytest.approx(mannwhitney_check(x, y))
    assert prob_improvement(x, y) + prob_improvement(y, x) == pytest.approx(1.0)  # 동점 0.5 처리면 합이 정확히 1
    assert prob_improvement(x, x) == pytest.approx(0.5)  # 같은 배열 (rliable의 특수 처리와 같은 값)
    xi = np.round(x)  # 동점이 많은 정수 점수
    yi = np.round(y)
    assert prob_improvement(xi, yi) == pytest.approx(mannwhitney_check(xi, yi))
    assert prob_improvement(x + 100, y) == 1.0 and prob_improvement(x - 100, y) == 0.0


def test_prob_improvement_same_distribution_is_half():
    w = ToyWorld()
    rng = np.random.default_rng(8)
    p = [prob_improvement(w.sample(5, rng), w.sample(5, rng)) for _ in range(1500)]
    assert abs(np.mean(p) - 0.5) < 0.005  # SE 약 0.001 (표준편차 0.04 / sqrt(1500))


def test_prob_improvement_ci_width_and_center():
    w = ToyWorld()
    rng = np.random.default_rng(9)
    p, (lo, hi) = prob_improvement_ci(w.sample(10, rng, lift=0.1), w.sample(10, rng), reps=400, rng=rng)
    assert lo < p < hi and 0 <= lo and hi <= 1


# ------------------------------------------------------------- 이진 점수의 한계
def test_iqm_of_binary_scores_is_degenerate():
    """성공률 r의 이진 점수에서 IQM = clip((r-0.25)/0.5, 0, 1): r<=0.25면 항상 0, r>=0.75면 항상 1."""
    for r, want in [(0.1, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 1.0), (0.9, 1.0), (0.3, 0.1)]:
        x = np.zeros((400, 1))
        x[: int(r * 400)] = 1
        assert iqm(x) == pytest.approx(want)


# ------------------------------------------------------------- 몬테카를로
def test_profile_variance_formula_and_average_profile_bias():
    w = ToyWorld()
    rng = np.random.default_rng(10)
    tau, n = 0.3, 5
    x = w.sample(n, rng, reps=20000)
    run_prof = (x > tau).mean(axis=(1, 2))
    avg_prof = (x.mean(axis=1) > tau).mean(axis=1)
    f_m = 1 - scipy.stats.norm.cdf((np.log(tau) - np.log(w.a) + w.s ** 2 / 2) / w.s)
    assert run_prof.mean() == pytest.approx(f_m.mean(), abs=0.002)  # 불편
    assert run_prof.var() == pytest.approx((f_m * (1 - f_m)).sum() / (w.m ** 2 * n), rel=0.05)  # 논문 σ_X^2
    assert avg_prof.mean() - f_m.mean() > 0.03  # 평균 점수 분포는 편향(이 토이에서 +0.05)


def test_estimator_variance_n3_vs_n100_and_median_bias():
    w = ToyWorld()
    rng = np.random.default_rng(11)
    truth = w.truth()
    res = {}
    for n in (3, 100):
        x = w.sample(n, rng, reps=1000)
        res[n] = {k: f(x) for k, f in [("mean", mean_of_task_means), ("median", median_of_task_means), ("iqm", iqm)]}
    for k in ("mean", "median", "iqm"):
        assert res[100][k].std() < res[3][k].std() / 2
    assert res[3]["iqm"].std() < res[3]["median"].std()  # IQM이 중앙값보다 효율적
    assert res[3]["iqm"].std() < res[3]["mean"].std()  # 꼬리가 두꺼운 이 토이에서는 평균보다도
    bias_med = res[3]["median"].mean() - truth["median"]
    se = res[3]["median"].std() / np.sqrt(1000)
    assert bias_med < -5 * se  # 표본 중앙값은 N이 작을 때 눈에 띄게 낮게 편향(논문 그림 3의 현상)
    assert abs(res[3]["iqm"].mean() - truth["iqm"]) < 0.05 * truth["iqm"]


def test_95_percent_stratified_bootstrap_coverage():
    """IQM의 95% 백분위 구간은 N=10에서 대략 맞고, 중앙값은 N=3에서 크게 모자란다(논문 그림 6의 주의와 같은 방향)."""
    w = ToyWorld()
    truth = w.truth()
    rng = np.random.default_rng(12)

    def coverage(n, stat, key, r=300, boot=500):
        hit = 0
        for _ in range(r):
            bs = stratified_resample(w.sample(n, rng), boot, rng)
            lo, hi = np.percentile(stat(bs), [2.5, 97.5])
            hit += lo <= truth[key] <= hi
        return hit / r

    cov_iqm10 = coverage(10, iqm, "iqm")
    cov_med3 = coverage(3, median_of_task_means, "median")
    assert 0.86 <= cov_iqm10 <= 0.99
    assert cov_med3 < 0.85
