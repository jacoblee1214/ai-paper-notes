"""minimal_impl.py의 핵심 성질 검사. 실행: python3 -m pytest analysis/test_minimal_impl.py -q (수십 초, CPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as mi  # noqa: E402


# ---- McNemar 검정 자체
def test_exact_pvalue_matches_scipy_binomtest():
    for b, c in [(3, 10), (0, 7), (12, 12), (25, 40)]:
        want = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        assert mi.mcnemar_pvalue(np.array(b), np.array(c), "exact") == pytest.approx(want, abs=1e-12)


def test_chi2_pvalue_formula():
    b, c = 10, 30
    want = stats.chi2.sf((b - c) ** 2 / (b + c), 1)
    assert mi.mcnemar_pvalue(np.array(b), np.array(c), "chi2") == pytest.approx(want)
    assert mi.mcnemar_pvalue(np.array(0), np.array(0), "chi2") == 1.0  # 불일치가 없으면 판단 불가


# ---- 1종 오류율
def test_type1_error_near_alpha():
    # 표본이 충분히 크면 카이제곱형 McNemar의 1종 오류율 ~ 0.05, 정확 검정은 보수적(<= 0.05 + 오차)
    t_chi2 = mi.simulate_type1(2000, 0.9, r=20000, method="chi2", seed=1)
    t_exact = mi.simulate_type1(2000, 0.9, r=20000, method="exact", seed=1)
    assert t_chi2 == pytest.approx(0.05, abs=0.01)
    assert t_exact <= 0.055
    assert t_exact <= t_chi2 + 0.005


# ---- 단조성
def test_power_increases_with_n():
    p = [mi.simulate_power(n, 0.02, 0.9, r=8000, seed=2)["power"] for n in (200, 500, 1000, 2000, 5000)]
    assert all(b > a for a, b in zip(p, p[1:]))


def test_power_increases_with_effect_size():
    p = [mi.simulate_power(1000, dl, 0.9, r=8000, seed=3)["power"] for dl in (0.005, 0.01, 0.02, 0.04, 0.08)]
    assert all(b > a for a, b in zip(p, p[1:]))


def test_power_decreases_with_disagreement():
    # 같은 n, 같은 정확도 차이에서 두 모델이 더 자주 엇갈리면(불일치율 d 커짐) 검정력이 떨어진다
    p = [mi.simulate_power(1000, 0.02, pa, r=8000, seed=4)["power"] for pa in (0.98, 0.95, 0.9, 0.8, 0.6)]
    assert all(b < a for a, b in zip(p, p[1:]))
    a = [mi.analytic_power(1000, 0.02, pa) for pa in (0.98, 0.95, 0.9, 0.8, 0.6)]
    assert all(b < a_ for a_, b in zip(a, a[1:]))


def test_type_m_exaggeration_shrinks_with_power():
    low = mi.simulate_power(500, 0.02, 0.9, r=40000, method="exact", seed=5)
    high = mi.simulate_power(5000, 0.02, 0.9, r=40000, method="exact", seed=5)
    assert low["type_m"] > 1.5 > high["type_m"] >= 1.0


# ---- 논문의 부록 C 예 (n=500, delta=0.02, pa=0.9: 검정력 0.25, Type-M 1.9)
def test_appendix_c_example_with_exact_test():
    s = mi.simulate_power(500, 0.02, 0.9, r=60000, method="exact", seed=6)
    assert s["power"] == pytest.approx(0.25, abs=0.02)
    assert s["type_m"] == pytest.approx(1.9, abs=0.1)


# ---- 해석식 대 시뮬레이션
@pytest.mark.parametrize("n,delta,pa", [(1000, 0.02, 0.9), (2000, 0.02, 0.9), (2000, 0.01, 0.95), (500, 0.04, 0.8),
                                        (5000, 0.01, 0.9)])
def test_analytic_power_matches_simulation(n, delta, pa):
    sim = mi.simulate_power(n, delta, pa, r=40000, method="chi2", seed=7)["power"]
    assert mi.analytic_power(n, delta, pa) == pytest.approx(sim, abs=0.03)


def test_analytic_formula_equals_connett_duffy_form():
    # 저자 노트북(R)의 normal 방법: paid=작은 불일치 확률, psi=큰/작은
    n, delta, pa, alpha = 1725, 0.0162, 0.942, 0.05
    d = 1 - pa
    paid, psi = (d - delta) / 2, (d + delta) / (d - delta)
    za = stats.norm.ppf(1 - alpha / 2)
    r_form = stats.norm.cdf((np.sqrt(n * paid) * (psi - 1) - za * np.sqrt(psi + 1)) / np.sqrt((psi + 1) - paid * (psi - 1) ** 2))
    assert mi.analytic_power(n, delta, pa) == pytest.approx(r_form, abs=1e-9)


def test_mde_closed_form_is_root_of_power_equation():
    for n, d in [(100, 0.2), (500, 0.1), (2000, 0.1), (1725, 0.058), (10000, 0.3)]:
        mde = mi.analytic_mde(n, 1 - d)
        assert mi.analytic_power(n, mde, 1 - d) == pytest.approx(0.8, abs=1e-9)
        root = brentq(lambda x: mi.analytic_power(n, x, 1 - d) - 0.8, 1e-8, d - 1e-9)
        assert mde == pytest.approx(root, rel=1e-6)


def test_n_required_inverts_power():
    n = mi.analytic_n_required(0.02, 0.9)
    assert mi.analytic_power(n, 0.02, 0.9) == pytest.approx(0.8, abs=1e-9)


def test_mde_scaling_sqrt_d_over_n():
    # MDE는 불일치율 d의 제곱근에 비례하고 문항 수 n의 제곱근에 반비례한다 (큰 n 근사)
    m1 = mi.analytic_mde(4000, 0.9)
    assert mi.analytic_mde(4000, 0.6) / m1 == pytest.approx(np.sqrt(0.4 / 0.1), rel=0.02)
    assert mi.analytic_mde(16000, 0.9) / m1 == pytest.approx(0.5, rel=0.02)
    assert m1 == pytest.approx(2.8016 * np.sqrt(0.1 / 4000), rel=0.01)


def test_simulated_mde_matches_analytic():
    sim = mi.simulate_mde(1000, 0.9, method="chi2", r=20000, seed=8)
    assert sim == pytest.approx(mi.analytic_mde(1000, 0.9), abs=0.0025)


def test_feasible_disagreement_bounds():
    lo, hi = mi.feasible_disagreement(0.11, 0.18)  # swe_agent 54 대 33 (n=300)
    assert lo == pytest.approx(0.07) and hi == pytest.approx(0.29)
    # 불일치율이 범위 끝일 때 정확 McNemar p: 이 저장소 swe_agent 분석의 p_max = 0.031
    n, a, c = 300, 33, 54
    b = a  # 불일치 최대: 모델 1이 맞춘 33개가 모두 모델 2와 다른 문항
    assert float(mi.mcnemar_pvalue(np.array(b), np.array(b + c - a), "exact")) == pytest.approx(0.0314, abs=5e-4)


# ---- Likert (평가자·문항 무작위 효과)
def test_likert_power_monotone_in_effect_workers_and_items():
    v = mi.LIKERT_HIGH_VAR
    by_delta = [mi.simulate_likert_power(10, 100, dl, **v, r=1500, seed=9) for dl in (0.05, 0.1, 0.2, 0.4)]
    assert all(b > a for a, b in zip(by_delta, by_delta[1:]))
    by_w = [mi.analytic_likert_power(w, 100, 0.2, **v) for w in (3, 10, 30, 100)]
    by_i = [mi.analytic_likert_power(10, i, 0.2, **v) for i in (20, 50, 100, 500)]
    assert all(b > a for a, b in zip(by_w, by_w[1:])) and all(b > a for a, b in zip(by_i, by_i[1:]))


def test_likert_analytic_matches_simulation_when_variance_is_well_estimated():
    v = mi.LIKERT_HIGH_VAR
    sim = mi.simulate_likert_power(30, 100, 0.1, **v, r=3000, seed=10)
    assert mi.analytic_likert_power(30, 100, 0.1, **v) == pytest.approx(sim, abs=0.05)


def test_likert_items_term_is_a_floor_for_adding_workers():
    # 평가자를 무한히 늘려도 표준오차는 sd_i1/sqrt(I) 아래로 내려가지 않는다
    v = mi.LIKERT_LOW_VAR
    se_inf = v["sd_i1"] / np.sqrt(50)
    delta = 2 * 2.5 * se_inf  # beta1 = 2.5*SE 바닥 -> 검정력 상한이 Phi(2.5 - 1.96)
    cap = stats.norm.cdf(2.5 - 1.96)
    assert mi.analytic_likert_power(10 ** 6, 50, delta, **v) == pytest.approx(cap, abs=1e-3)
    assert mi.analytic_likert_power(10, 50, delta, **v) < cap


def test_likert_small_worker_count_inflates_false_positive_rate():
    # t > 1.96 기준은 평가자 3명이면 분산 추정이 불안정해 거짓 양성이 공칭(0.025 단측)보다 훨씬 많다
    v = mi.LIKERT_HIGH_VAR
    fp3 = mi.simulate_likert_power(3, 100, 0.0, **v, r=4000, seed=11)
    fp30 = mi.simulate_likert_power(30, 100, 0.0, **v, r=4000, seed=11)
    assert fp3 > 0.05 and fp3 > 1.5 * fp30


# ---- BLEU
def test_bleu_simulation_matches_normal_approximation_and_is_monotone():
    p1 = mi.simulate_bleu_power(1000, 1.0, 0.13, 25.8, n_trials=250, n_rand=500, seed=12)
    p2 = mi.simulate_bleu_power(2000, 1.0, 0.13, 25.8, n_trials=250, n_rand=500, seed=12)
    assert p2 > p1
    assert p1 == pytest.approx(mi.analytic_bleu_power(1000, 1.0, 0.13, 25.8), abs=0.08)
    assert p2 == pytest.approx(mi.analytic_bleu_power(2000, 1.0, 0.13, 25.8), abs=0.08)
    assert mi.analytic_bleu_power(2000, 1.0, 0.13, 25.8) == pytest.approx(0.75, abs=0.03)  # 논문 그림 4
