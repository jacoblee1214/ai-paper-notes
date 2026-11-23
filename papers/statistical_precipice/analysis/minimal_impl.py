"""Agarwal et al. (NeurIPS 2021) 평가 도구의 최소 구현. numpy + scipy만 쓴다(신경망이 없는 통계 방법이라 PyTorch가 필요 없음).

입력은 모두 scores[run, task] (N x M) 이다. 집계 함수는 마지막 두 축이 (run, task)이면
앞쪽 축(부트스트랩 반복 등)을 그대로 두고 계산한다. 즉 (..., N, M) -> (...).

논문 대응
  iqm                    §4.3, 25% 절사 평균 (모든 실행·과제를 한데 모아 절사)
  optimality_gap         §4.3, γ - E[min(X, γ)]
  mean_of_task_means / median_of_task_means   §2, 과제별 평균 뒤 평균·중앙값
  stratified_resample    §4.1, 과제별로 실행을 독립적으로 복원추출
  bootstrap_ci           §4.1, 백분위(percentile) 구간
  performance_profile    §4.2 식 (1), 실행 점수 분포(= 꼬리 분포의 경험적 추정)
  prob_improvement       §4 + 식 A.2, 과제별 Mann–Whitney U / (N K) 의 과제 평균
"""
from __future__ import annotations

import numpy as np
import scipy.stats


# ---------------------------------------------------------------- 집계 지표
def iqm(scores: np.ndarray) -> np.ndarray:
    """상하 25%를 버린 평균. 모든 (run, task) 점수를 한데 모아 자른다.

    양끝에서 floor(0.25 * n)개씩 버린다(scipy.stats.trim_mean, rliable과 같음).
    n = N*M이 4의 배수가 아니면 남는 개수가 논문 본문의 floor(NM/2)보다 많을 수 있다.
    """
    flat = np.sort(scores.reshape(*scores.shape[:-2], -1), axis=-1)
    n = flat.shape[-1]
    cut = int(0.25 * n)
    return flat[..., cut:n - cut].mean(axis=-1)


def mean_of_task_means(scores: np.ndarray) -> np.ndarray:
    return scores.mean(axis=-2).mean(axis=-1)


def median_of_task_means(scores: np.ndarray) -> np.ndarray:
    return np.median(scores.mean(axis=-2), axis=-1)


def optimality_gap(scores: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """γ에 못 미치는 양의 평균: γ - mean(min(x, γ)). γ를 넘는 점수는 γ로 자른다."""
    return gamma - np.minimum(scores, gamma).mean(axis=(-2, -1))


# ---------------------------------------------------------------- 층화 부트스트랩
def stratified_resample(scores: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    """과제(열)마다 N개 실행을 독립적으로 복원추출. 반환 shape (reps, N, M).

    과제별 실행 수 N이 그대로 유지되고, 다른 과제의 실행과 섞이지 않는다.
    """
    n, m = scores.shape
    idx = rng.integers(0, n, size=(reps, n, m))
    return scores[idx, np.arange(m)[None, None, :]]


def bootstrap_ci(scores: np.ndarray, stat=iqm, reps: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None):
    """(점추정, (하한, 상한)). 층화 부트스트랩 분포의 [α/2, 1-α/2] 백분위 구간.

    rliable 기본 reps는 집계 지표 50,000, 프로필·개선 확률 2,000. 여기 기본값은 토이용으로 작게 잡았다.
    """
    rng = rng or np.random.default_rng(0)
    boot = stat(stratified_resample(scores, reps, rng))
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return stat(scores), (lo, hi)


# ---------------------------------------------------------------- 성능 프로필
def performance_profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """실행 점수 분포 F̂(τ) = (1/MN) Σ 1[x > τ]  (논문 식 (1)). 반환 (len(taus),)."""
    return (scores[..., None] > np.asarray(taus)).mean(axis=(-3, -2))


def average_score_profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """과제 평균 점수의 분포(편향 있음). 비교용."""
    return (scores.mean(axis=-2)[..., None] > np.asarray(taus)).mean(axis=-2)


def profile_ci(scores: np.ndarray, taus: np.ndarray, reps: int = 2000, alpha: float = 0.05,
               rng: np.random.Generator | None = None):
    """점별(pointwise) 95% 밴드. 층화 부트스트랩 백분위."""
    rng = rng or np.random.default_rng(0)
    boot = performance_profile(stratified_resample(scores, reps, rng), taus)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return performance_profile(scores, taus), (lo, hi)


# ---------------------------------------------------------------- 개선 확률
def prob_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) = (1/M) Σ_m U_m / (N K).  U_m = Σ_ij [1(x_i > y_j) + 0.5 * 1(x_i == y_j)]  (식 A.2).

    x는 (N, M), y는 (K, M). 실행 수가 달라도 된다.
    """
    gt = (x[:, None, :] > y[None, :, :]).mean(axis=(0, 1))
    eq = (x[:, None, :] == y[None, :, :]).mean(axis=(0, 1))
    return float((gt + 0.5 * eq).mean())


def prob_improvement_ci(x: np.ndarray, y: np.ndarray, reps: int = 2000, alpha: float = 0.05,
                        rng: np.random.Generator | None = None):
    """x, y를 각각 독립적으로 층화 재추출(rliable StratifiedIndependentBootstrap과 같은 방식)."""
    rng = rng or np.random.default_rng(0)
    bx = stratified_resample(x, reps, rng)
    by = stratified_resample(y, reps, rng)
    gt = (bx[:, :, None, :] > by[:, None, :, :]).mean(axis=(1, 2))
    eq = (bx[:, :, None, :] == by[:, None, :, :]).mean(axis=(1, 2))
    boot = (gt + 0.5 * eq).mean(axis=-1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return prob_improvement(x, y), (float(lo), float(hi))


def mannwhitney_check(x: np.ndarray, y: np.ndarray) -> float:
    """scipy의 Mann–Whitney U1/(NK)로 계산한 같은 양(검산용)."""
    n, k = x.shape[0], y.shape[0]
    return float(np.mean([scipy.stats.mannwhitneyu(x[:, m], y[:, m], alternative="greater").statistic / (n * k)
                          for m in range(x.shape[1])]))


# ---------------------------------------------------------------- 토이 세계 (분석·테스트 공용)
class ToyWorld:
    """과제별 로그정규 점수. 과제 난이도 a_m은 치우침이 크다(대부분 낮고 몇 과제만 큼).

    X_{m,n} = a_m * (1 + lift) * exp(s_m * Z - s_m^2 / 2) 로 E[X_m] = a_m (1 + lift).
    """

    def __init__(self, m: int = 26, seed: int = 0):
        r = np.random.default_rng(seed)
        self.m = m
        self.a = np.exp(r.normal(-1.2, 1.3, m))
        self.s = r.uniform(0.3, 1.0, m)

    def sample(self, n: int, rng: np.random.Generator, lift: float = 0.0, reps: int | None = None) -> np.ndarray:
        shape = (n, self.m) if reps is None else (reps, n, self.m)
        z = rng.standard_normal(shape)
        return self.a * (1 + lift) * np.exp(self.s * z - self.s ** 2 / 2)

    def truth(self, lift: float = 0.0, big: int = 20000, seed: int = 123, gamma: float = 1.0) -> dict:
        """'실행 무한' 값. 평균·중앙값은 정확(E[X_m]=a_m(1+lift)), IQM·최적성 격차는 큰 표본으로 근사."""
        pop = self.sample(big, np.random.default_rng(seed), lift)
        mu = self.a * (1 + lift)
        return {"mean": float(mu.mean()), "median": float(np.median(mu)),
                "iqm": float(iqm(pop)), "optgap": float(optimality_gap(pop, gamma))}
