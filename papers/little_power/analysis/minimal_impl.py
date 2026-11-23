"""With Little Power Comes Great Responsibility (arXiv:2010.06595) 최소 구현.

논문의 시뮬레이션 기반 검정력 분석(그림 2)을 numpy·scipy로 짠 토이입니다.
통계 방법이라 PyTorch가 필요 없어서 numpy를 씁니다. CPU에서 수 초에서 수십 초면 끝납니다.

  (1) 짝지은 이진 결과 시뮬레이터 + McNemar 검정 + 시뮬레이션 검정력·Type-M·Type-S·MDE
  (2) 해석식(정규 근사) 검정력, 표본 크기, MDE의 닫힌 형태
  (3) 사람 평가(Likert) 단순화 시뮬레이션: 평가자·문항 무작위 효과(완전 교차 설계)
  (4) 번역(BLEU) Delta-Laplace 혼합 시뮬레이션과 정규 근사 (저자 노트북 BLEU.ipynb와 같은 절차)

기호: n 문항 수, delta 두 모델의 정확도 차이(모델 2 - 모델 1), pa 두 모델이 같은 결과를 내는 비율
(불일치율 d = 1 - pa). McNemar 검정은 불일치 쌍의 개수 (b, c)만 봅니다.
  b = 모델 1만 맞은 문항 수, c = 모델 2만 맞은 문항 수, P(b쪽) = (d - delta)/2, P(c쪽) = (d + delta)/2.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# (1) 짝지은 이진 결과 시뮬레이터와 McNemar 검정
# ----------------------------------------------------------------------------


def discordant_probs(delta: float, pa: float) -> tuple[float, float]:
    """(P(모델 1만 맞음), P(모델 2만 맞음)). 두 확률의 합이 불일치율 d = 1 - pa, 차가 delta."""
    d = 1.0 - pa
    if not (abs(delta) <= d + 1e-12):
        raise ValueError(f"|delta|={abs(delta)} 가 불일치율 d={d} 보다 클 수 없습니다")
    return (d - delta) / 2.0, (d + delta) / 2.0


def mcnemar_pvalue(b, c, method: str = "chi2") -> np.ndarray:
    """McNemar p값(벡터화). b, c는 같은 모양의 정수 배열.

    chi2     : 논문 §3.1의 식 (b - c)^2 / (b + c) 를 자유도 1 카이제곱과 비교(연속성 보정 없음)
    chi2_cc  : 연속성 보정 (|b - c| - 1)^2 / (b + c)
    exact    : 정확 이항 검정(양쪽). 저자 노트북의 statsmodels mcnemar 기본값이 이것입니다.
    """
    b = np.asarray(b)
    c = np.asarray(c)
    m = b + c
    safe_m = np.where(m == 0, 1, m)
    if method == "chi2":
        p = stats.chi2.sf((b - c) ** 2 / safe_m, 1)
    elif method == "chi2_cc":
        p = stats.chi2.sf(np.maximum(np.abs(b - c) - 1, 0) ** 2 / safe_m, 1)
    elif method == "exact":
        p = np.minimum(1.0, 2.0 * stats.binom.cdf(np.minimum(b, c), m, 0.5))
    else:
        raise ValueError(method)
    return np.where(m == 0, 1.0, p)


def simulate_power(n: int, delta: float, pa: float, alpha: float = 0.05, r: int = 20000,
                   method: str = "chi2", seed: int = 0) -> dict:
    """그림 2의 알고리즘. r번 n문항 데이터를 뽑아 McNemar를 돌리고 검정력·Type-M·Type-S를 센다.

    검정력은 '유의하고 부호가 참값과 같은' 비율입니다(논문 그림 2의 식). delta = 0이면 이 값은
    1종 오류율의 절반이므로 1종 오류율은 simulate_type1에서 따로 구합니다.
    """
    rng = np.random.default_rng(seed)
    p10, p01 = discordant_probs(delta, pa)
    counts = rng.multinomial(n, [p10, p01, 1.0 - p10 - p01], size=r)
    b, c = counts[:, 0], counts[:, 1]
    p = mcnemar_pvalue(b, c, method)
    obs = (c - b) / n  # 관측된 정확도 차이(모델 2 - 모델 1)
    sig = p <= alpha
    right = sig & (np.sign(obs) == np.sign(delta))
    n_sig = sig.sum()
    return {
        "power": float(right.mean()),
        "type_m": float(np.mean(np.abs(obs[sig])) / abs(delta)) if n_sig else float("nan"),
        "type_s": float(np.mean(np.sign(obs[sig]) != np.sign(delta))) if n_sig else float("nan"),
        "mean_obs": float(obs.mean()),
    }


def simulate_type1(n: int, pa: float, alpha: float = 0.05, r: int = 20000,
                   method: str = "chi2", seed: int = 0) -> float:
    """귀무가설(delta = 0)에서 유의로 판정되는 비율 = 1종 오류율."""
    rng = np.random.default_rng(seed)
    d = 1.0 - pa
    counts = rng.multinomial(n, [d / 2, d / 2, pa], size=r)
    return float(np.mean(mcnemar_pvalue(counts[:, 0], counts[:, 1], method) <= alpha))


def simulate_mde(n: int, pa: float, alpha: float = 0.05, power: float = 0.8, r: int = 20000,
                 method: str = "chi2", seed: int = 0, tol: float = 1e-4) -> float:
    """시뮬레이션 검정력이 power가 되는 최소 delta(이분법). 같은 시드를 쓰므로 delta에 대해 거의 단조입니다."""
    d = 1.0 - pa
    f = lambda dl: simulate_power(n, dl, pa, alpha, r, method, seed)["power"] - power
    lo, hi = 1e-6, d
    if f(hi) < 0:
        return float("nan")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def feasible_disagreement(acc1: float, acc2: float) -> tuple[float, float]:
    """두 모델의 정확도(주변 합)만 알 때 가능한 불일치율의 범위 [d_min, d_max].

    d_min = |acc2 - acc1| (틀린 문항이 한쪽의 부분집합), d_max = min(acc1 + acc2, 2 - acc1 - acc2)
    (맞은 문항 또는 틀린 문항이 서로 겹치지 않음). 이 저장소의 15·16·18번 분석이 '주변 합만으로
    McNemar 가능 범위'를 낼 때 쓴 것과 같은 제약입니다.
    """
    return abs(acc2 - acc1), min(acc1 + acc2, 2.0 - acc1 - acc2)


# ----------------------------------------------------------------------------
# (2) 해석식(정규 근사)
# ----------------------------------------------------------------------------
# 유도 (README와 03_analysis.md §1에 상세):
#   D = c - b, 불일치 쌍 수 M = b + c. n문항에서 E[D] = n*delta, Var[D] = n*(d - delta^2).
#   검정: |D| > z_{1-a/2} * sqrt(M), M ~ n*d 로 근사. 따라서
#   power = Phi( (sqrt(n)*delta - z_a*sqrt(d)) / sqrt(d - delta^2) )   (맞는 부호 쪽만)
#   이 식은 저자 노트북(R, Connett·Duffy 식)의 'normal' 방법과 대수적으로 같습니다.


def analytic_power(n: float, delta: float, pa: float, alpha: float = 0.05, both_tails: bool = False) -> float:
    d = 1.0 - pa
    za = stats.norm.ppf(1 - alpha / 2)
    s = np.sqrt(max(d - delta ** 2, 1e-18))
    pw = stats.norm.cdf((np.sqrt(n) * abs(delta) - za * np.sqrt(d)) / s)
    if both_tails:  # 반대 부호로 유의할 확률(Type-S 몫)
        pw += stats.norm.cdf((-np.sqrt(n) * abs(delta) - za * np.sqrt(d)) / s)
    return float(pw)


def analytic_n_required(delta: float, pa: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """검정력 power에 필요한 문항 수. 위 식을 n에 대해 풀면 닫힌 형태입니다."""
    d = 1.0 - pa
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((za * np.sqrt(d) + zb * np.sqrt(d - delta ** 2)) ** 2 / delta ** 2)


def analytic_mde(n: float, pa: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """검정력 power를 내는 최소 검출 효과(MDE)의 닫힌 형태.

    (sqrt(n)*D - za*sqrt(d))^2 = zb^2 (d - D^2) 을 D에 대해 풀면
      MDE = sqrt(d) * ( za*sqrt(n) + zb*sqrt(n - za^2 + zb^2) ) / (n + zb^2)
    n이 크면 MDE ~ (za + zb) * sqrt(d / n) = 2.80 * sqrt(d/n) (alpha=0.05, power=0.8).
    """
    d = 1.0 - pa
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float(np.sqrt(d) * (za * np.sqrt(n) + zb * np.sqrt(n - za ** 2 + zb ** 2)) / (n + zb ** 2))


def analytic_mde_from_accuracies(n: float, acc1: float, alpha: float = 0.05, power: float = 0.8,
                                 pa_of_delta=None) -> float:
    """불일치율이 효과 크기의 함수 pa(delta)일 때(논문의 회귀 방식) MDE를 수치로 푼다."""
    f = lambda dl: analytic_power(n, dl, pa_of_delta(acc1, dl), alpha) - power
    hi = 1.0 - acc1 - 1e-6
    if f(hi) < 0:  # 정확도 상한(1.0)까지 올려도 검정력이 모자란 경우
        return float("nan")
    return float(brentq(f, 1e-6, hi))


def mde_binomial_unpaired(n: float, p1: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """짝짓지 않은 두 비율 검정(R power.prop.test, 문항 수 n씩)의 MDE. 논문 표 9의 'MDE Binomial'."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)

    def need_n(dl):
        p2 = p1 + dl
        pbar = (p1 + p2) / 2
        return (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / dl ** 2 - n

    return float(brentq(need_n, 1e-6, 1.0 - p1 - 1e-9))


# ----------------------------------------------------------------------------
# (3) 사람 평가(Likert) 단순화 시뮬레이션
# ----------------------------------------------------------------------------
# 모델 (논문 부록 H.3 식 5): Y_wi,x = b0 + W0w + I0i + (b1 + W1w + I1i) * x + e,  x = -1/+1 (모델 A/B)
# 두 모델 평균 차이 delta = 2*b1. 모든 평가자가 모든 문항을 두 모델 모두에 대해 평가(완전 교차, 저자 R 코드와 같음).
# 저자는 lme4로 적합해 t > 1.96 이면 검출로 셉니다. 여기서는 lme4 없이 같은 효과를 바로 뽑을 수 있는
# 셀별 차이 D_wi = Y_wi,+ - Y_wi,- 에 두 방향 ANOVA 분산 추정을 써서 같은 t 통계량을 만듭니다.
#   D̄ 의 분산 = (MS_W + MS_I - MS_E) / (W*I)  (평가자, 문항, 잔차 평균제곱)
# 해석식: Var(b1-hat) = sd_w1^2/W + sd_i1^2/I + sd_e^2/(2*W*I). 절편 분산은 차이를 취할 때 사라집니다.

LIKERT_HIGH_VAR = dict(sd_w1=0.11, sd_i1=0.14, sd_e=0.26)  # 논문 표 16 high variance (절편 분산은 영향 없음)
LIKERT_LOW_VAR = dict(sd_w1=0.04, sd_i1=0.13, sd_e=0.16)   # 논문 표 16 low variance


def simulate_likert_power(n_workers: int, n_items: int, delta: float, sd_w1: float, sd_i1: float, sd_e: float,
                          r: int = 2000, crit: float = 1.96, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    W, I = n_workers, n_items
    b1 = delta / 2.0
    w1 = rng.normal(0, sd_w1, size=(r, W, 1))
    i1 = rng.normal(0, sd_i1, size=(r, 1, I))
    e = rng.normal(0, sd_e, size=(r, W, I, 2))
    # D = Y(+1) - Y(-1) = 2*(b1 + W1w + I1i) + e(+1) - e(-1); 절편 W0, I0 은 상쇄
    D = 2 * (b1 + w1 + i1) + e[..., 0] - e[..., 1]
    grand = D.mean(axis=(1, 2))
    row = D.mean(axis=2)  # 평가자 평균 (r, W)
    col = D.mean(axis=1)  # 문항 평균 (r, I)
    ms_w = I * row.var(axis=1, ddof=1) if W > 1 else np.zeros(r)
    ms_i = W * col.var(axis=1, ddof=1)
    resid = D - row[:, :, None] - col[:, None, :] + grand[:, None, None]
    ms_e = (resid ** 2).sum(axis=(1, 2)) / max((W - 1) * (I - 1), 1)
    var = np.maximum((ms_w + ms_i - ms_e) / (W * I), 1e-12)
    t = grand / np.sqrt(var)
    return float(np.mean(t > crit))


def analytic_likert_power(n_workers: int, n_items: int, delta: float, sd_w1: float, sd_i1: float, sd_e: float,
                          crit: float = 1.96) -> float:
    """분산을 안다고 보고(추정 잡음 무시) 정규 근사한 검정력. 평가자 수를 늘려도 sd_i1^2/I 항은 줄지 않는다."""
    se = np.sqrt(sd_w1 ** 2 / n_workers + sd_i1 ** 2 / n_items + sd_e ** 2 / (2 * n_workers * n_items))
    return float(stats.norm.cdf((delta / 2) / se - crit))


# ----------------------------------------------------------------------------
# (4) 번역(BLEU): Delta-Laplace 혼합 + 무작위화 검정 (저자 BLEU.ipynb와 같은 절차)
# ----------------------------------------------------------------------------


def simulate_bleu_power(n: int, delta_b: float, p0: float, b0: float, n_trials: int = 300, n_rand: int = 1000,
                        seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    mu = -2.0 * delta_b / (n * (1 - p0))
    b = b0 / n
    hits = 0
    for _ in range(n_trials):
        nonzero = rng.random(n) >= p0
        delta_i = np.where(nonzero, rng.laplace(mu, b, size=n), 0.0)
        obs = -delta_i.sum() / 2.0
        sel = rng.random((n_rand, n)) > 0.5
        null = sel @ delta_i + obs
        lo, hi = np.percentile(null, [2.5, 97.5])
        if (obs >= hi or obs <= lo) and np.sign(obs) == np.sign(delta_b):
            hits += 1
    return hits / n_trials


def analytic_bleu_power(n: int, delta_b: float, p0: float, b0: float, alpha: float = 0.05) -> float:
    """정규 근사. 관측 차이 obs = -sum(delta_i)/2 의 분산과, 무작위화 귀무분포의 표준편차 sqrt(sum delta_i^2)/2 를 쓴다."""
    mu = -2.0 * delta_b / (n * (1 - p0))
    b = b0 / n
    e2 = 2 * b ** 2 + mu ** 2                      # E[Laplace^2]
    var_di = (1 - p0) * e2 - ((1 - p0) * mu) ** 2  # 0을 섞은 delta_i 의 분산
    sd_obs = np.sqrt(n * var_di) / 2.0
    sd_null = np.sqrt(n * (1 - p0) * e2) / 2.0
    za = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf((delta_b - za * sd_null) / sd_obs))


if __name__ == "__main__":
    # 논문 부록 C의 예: n=500, delta=0.02, pa=0.9 -> 검정력 약 0.25, Type-M 약 1.9
    for n in (500, 2000):
        s = simulate_power(n, 0.02, 0.9, r=40000)
        print(f"n={n}: sim power={s['power']:.3f} typeM={s['type_m']:.2f} typeS={s['type_s']:.4f} "
              f"analytic={analytic_power(n, 0.02, 0.9):.3f}")
    print("MDE(n=1725, pa=0.94) analytic:", round(analytic_mde(1725, 0.94), 4))
