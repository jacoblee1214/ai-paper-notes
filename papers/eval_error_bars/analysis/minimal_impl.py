"""Adding Error Bars to Evals (Miller, arXiv:2411.00640)의 식을 numpy·scipy로 구현한 최소 구현.

신경망이 없는 통계 방법이라 PyTorch 대신 numpy를 쓴다.

입력 규약
  scores   : (n,) 문항당 점수 하나, 또는 (n, K) 문항당 K번 샘플링한 점수. scores[question, sample]
  clusters : (n,) 군집 id 배열(문항이 같은 지문·문서·원문을 공유할 때). 없으면 문항마다 독립.
식 번호는 논문의 번호다. 논문 표기와 다른 점은 각 함수의 docstring에 적었다.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

Z975 = stats.norm.ppf(0.975)


def question_means(scores) -> np.ndarray:
    """scores[question, sample] -> 문항별 평균 s_i (K=1이면 그대로). 3.1절: 문항 평균 위에서 SE를 계산한다."""
    s = np.asarray(scores, dtype=float)
    return s if s.ndim == 1 else s.mean(axis=1)


# ---------------------------------------------------------------- 식 (1), (2), (3)
def se_clt(s) -> float:
    """식 (1): sqrt(Var(s)/n), Var는 1/(n-1) 표본 분산."""
    s = np.asarray(s, dtype=float)
    return float(np.sqrt(s.var(ddof=1) / len(s)))


def se_bernoulli(s) -> float:
    """식 (2): sqrt(sbar(1-sbar)/n). 이진 점수에서 식 (1)에 sqrt((n-1)/n)를 곱한 값과 같다."""
    s = np.asarray(s, dtype=float)
    m = s.mean()
    return float(np.sqrt(m * (1 - m) / len(s)))


def ci95(estimate: float, se: float) -> tuple[float, float]:
    """식 (3): estimate +- 1.96 SE."""
    return estimate - Z975 * se, estimate + Z975 * se


# ---------------------------------------------------------------- 식 (4): 군집 SE
def _cluster_sums(u: np.ndarray, clusters) -> np.ndarray:
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return np.bincount(inv, weights=u)


def se_clustered(s, clusters, small_sample: bool = False) -> float:
    """식 (4): SE^2 = SE_CLT^2 + (1/n^2) sum_c sum_i sum_{j!=i} u_i u_j,  u = s - sbar.

    교차항 합 = sum_c (sum_i u_i)^2 - sum_i u_i^2 이므로 O(n)에 계산한다.
    small_sample=True: 부록 A의 순수 sandwich 식(1/n^2) sum_c (sum_i u_i)^2에 G/(G-1)을 곱한 CR1.
    (논문 식 (4)는 SE_CLT에 1/(n-1)을, 부록 A는 1/n을 쓴다. 차이는 n/(n-1)배로 작다.)
    """
    s = np.asarray(s, dtype=float)
    n = len(s)
    u = s - s.mean()
    cs = _cluster_sums(u, clusters)
    if small_sample:
        g = len(cs)
        return float(np.sqrt(g / (g - 1) * (cs ** 2).sum()) / n)
    cross = (cs ** 2).sum() - (u ** 2).sum()
    return float(np.sqrt(se_clt(s) ** 2 + cross / n ** 2))


def design_effect(s, clusters) -> float:
    """(SE_clustered / SE_CLT)^2. 균등 군집 크기 m에서 약 1 + (m-1) rho (군집 내 상관 rho)."""
    return (se_clustered(s, clusters) / se_clt(s)) ** 2


def se_stratified(s, strata) -> float:
    """논문에 없는 대조용. 층(저장소처럼 고정된 하위 집합)이 모집단에서 뽑힌 것이 아니라 고정일 때의 SE:
    sqrt(sum_h W_h^2 s_h^2 / n_h), W_h = n_h/n. 층 간 평균 차이가 분산에서 빠지므로 식 (1)보다 작아진다.
    군집 SE(식 4)는 군집이 무작위 표본일 때, 층화 SE는 층이 고정일 때의 가정이다. n_h=1인 층은 분산 0으로 둔다."""
    s = np.asarray(s, dtype=float)
    strata = np.asarray(strata)
    n = len(s)
    v = 0.0
    for h in np.unique(strata):
        x = s[strata == h]
        if len(x) > 1:
            v += (len(x) / n) ** 2 * x.var(ddof=1) / len(x)
    return float(np.sqrt(v))


# ---------------------------------------------------------------- 4절: 두 모델 비교
def se_unpaired(sA, sB) -> float:
    """4.1절: SE_{A-B} = sqrt(SE_A^2 + SE_B^2)."""
    return float(np.hypot(se_clt(sA), se_clt(sB)))


def se_paired(sA, sB) -> float:
    """식 (7): 문항별 차이 s_A,i - s_B,i의 SE."""
    return se_clt(np.asarray(sA, float) - np.asarray(sB, float))


def se_paired_from_corr(sA, sB) -> float:
    """4.2절 끝 식: sqrt(SE_A^2 + SE_B^2 - 2 SE_A SE_B Corr). 식 (7)과 정확히 같다(같은 ddof)."""
    a, b = se_clt(sA), se_clt(sB)
    r = np.corrcoef(sA, sB)[0, 1]
    return float(np.sqrt(a * a + b * b - 2 * a * b * r))


def se_paired_clustered(sA, sB, clusters) -> float:
    """식 (8): (1/n) sqrt(sum_c sum_i sum_j d_ic d_jc), d = 차이 - 평균 차이 (i=j 포함, CR0)."""
    d = np.asarray(sA, float) - np.asarray(sB, float)
    u = d - d.mean()
    return float(np.sqrt((_cluster_sums(u, clusters) ** 2).sum()) / len(d))


def compare(sA, sB, clusters=None) -> dict:
    """모델 A-B 비교 한 줄: 차이, 비짝/짝/군집 짝 SE, z, 95% 구간, 점수 상관."""
    sA, sB = np.asarray(sA, float), np.asarray(sB, float)
    diff = float(sA.mean() - sB.mean())
    out = {"diff": diff, "se_unpaired": se_unpaired(sA, sB), "se_paired": se_paired(sA, sB),
           "corr": float(np.corrcoef(sA, sB)[0, 1])}
    if clusters is not None:
        out["se_paired_clustered"] = se_paired_clustered(sA, sB, clusters)
    best = out.get("se_paired_clustered", out["se_paired"])
    out["z"] = diff / best
    out["ci95"] = ci95(diff, best)
    return out


# ---------------------------------------------------------------- 3절: 분산 분해와 K번 샘플링
def variance_components(scores) -> tuple[float, float]:
    """scores[n, K], K>=2 -> (Var(x), E[sigma_i^2]) 추정.

    E[sigma^2]: 문항 안 표본 분산(ddof=1, 부록 C가 K-1로 나누라는 바로 그 보정)의 평균.
    Var(x): Var(xhat) - E[sigma^2]/K. xhat 분산에는 sigma^2/K가 섞여 있어서 빼 준다
    (논문은 xhat를 그대로 쓰라고 한다. 03 §5 참고).
    """
    sc = np.asarray(scores, dtype=float)
    n, k = sc.shape
    e_sig2 = sc.var(axis=1, ddof=1).mean()
    var_x = sc.mean(axis=1).var(ddof=1) - e_sig2 / k
    return float(var_x), float(e_sig2)


def var_mean(var_x: float, e_sig2: float, n: int, k: int = 1) -> float:
    """3절: Var(mu_hat) = (Var(x) + E[sigma^2]/K) / n."""
    return (var_x + e_sig2 / k) / n


def variance_ratio_vs_k1(var_x: float, e_sig2: float, k: int) -> float:
    """3.1절: Var(mu_hat | K) / Var(mu_hat | K=1). 균등 난이도 이진 예에서 (1 + 2/K)/3."""
    return (var_x + e_sig2 / k) / (var_x + e_sig2)


def omega2_paired(scoresA, scoresB) -> float:
    """5절 omega^2 = Var(xA) + Var(xB) - 2Cov(xA, xB) = Var(xA - xB) 추정.
    xhat의 잡음(sigma^2/K)을 빼서 편향을 없앤다. K=1이면 추정할 수 없다(문항 안 분산을 모름)."""
    a, b = np.asarray(scoresA, float), np.asarray(scoresB, float)
    ka = a.shape[1]
    kb = b.shape[1]
    dx = a.mean(1) - b.mean(1)
    sa = a.var(1, ddof=1).mean()
    sb = b.var(1, ddof=1).mean()
    return float(dx.var(ddof=1) - sa / ka - sb / kb)


# ---------------------------------------------------------------- 5절: 검정력
def n_required(delta, omega2, sig2A=0.0, sig2B=0.0, kA=1, kB=1, alpha=0.05, power=0.8) -> float:
    """식 (9): n = (z_{a/2} + z_b)^2 (omega^2 + sig2A/KA + sig2B/KB) / delta^2 (올림 안 한 값)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z ** 2 * (omega2 + sig2A / kA + sig2B / kB) / delta ** 2


def mde(n, omega2, sig2A=0.0, sig2B=0.0, kA=1, kB=1, alpha=0.05, power=0.8) -> float:
    """식 (10): delta = (z_{a/2} + z_b) sqrt((omega^2 + sig2A/KA + sig2B/KB)/n)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt((omega2 + sig2A / kA + sig2B / kB) / n))


def power_z(n, delta, omega2, sig2A=0.0, sig2B=0.0, kA=1, kB=1, alpha=0.05) -> float:
    """식 (9)의 역: 양측 z 검정의 검정력 Phi(delta/se - z) + Phi(-delta/se - z)."""
    se = np.sqrt((omega2 + sig2A / kA + sig2B / kB) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(delta / se - z) + stats.norm.cdf(-delta / se - z))


def omega2_clustered(scoresA, scoresB, clusters) -> float:
    """부록 C: omega^2_clustered = (1/n) sum_c (sum_i (dxhat_ic - mean))^2, dxhat = xhatA - xhatB.
    n * Var_clustered(mu_hat_{A-B}) 이므로 식 (9)·(10)에 그대로 넣는다. 잡음 보정은 omega2_paired와 같은 이유로 뺀다."""
    a, b = np.asarray(scoresA, float), np.asarray(scoresB, float)
    dx = a.mean(1) - b.mean(1)
    u = dx - dx.mean()
    raw = (_cluster_sums(u, clusters) ** 2).sum() / len(dx)
    noise = a.var(1, ddof=1).mean() / a.shape[1] + b.var(1, ddof=1).mean() / b.shape[1]
    return float(raw - noise)
