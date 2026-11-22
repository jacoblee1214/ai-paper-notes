"""Kadavath et al. (arXiv:2207.05221v4) 재계산과 토이.

입력은 논문 표(Table 1-3)와 그림에서 눈으로 읽은 값(오차 약 +-0.01)입니다. 외부 데이터를 받지 않습니다.
결과는 recompute_calibration_results.json. numpy, scipy만 씁니다. 실행: python3 -I recompute_calibration.py
"""
import json
import math
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. 지표 정의 검산
def ece_equal_mass(p, y, bins=10, rms=False):
    """식 A.1(MAD)과 A.2(RMS). 같은 개수씩 나눈 구간이라 구간 가중치는 같다."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    d = np.array([y[c].mean() - p[c].mean() for c in chunks])
    return float(np.sqrt((d ** 2).mean()) if rms else np.abs(d).mean())

def ece_equal_width(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)

def brier(p, y):
    return float(((p - y) ** 2).mean())

def auroc(score, y):
    r = stats.rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# (a) 완벽히 보정된 예측기의 ECE 잡음 바닥: 표본 수와 구간 수에 따라
floor = {}
for n in (200, 1000, 5000, 20000, 100000):
    for bins in (5, 10, 20):
        vals = []
        for _ in range(20):
            p = rng.uniform(0, 1, n)
            y = (rng.uniform(0, 1, n) < p).astype(float)
            vals.append(ece_equal_mass(p, y, bins))
        floor[f"n={n},bins={bins}"] = round(float(np.mean(vals)), 4)
out["ece_noise_floor_perfect_calibration"] = floor

# (b) 같은 예측기, 구간 수 민감도 (실제로 과신하는 예측기: 참 확률 = 0.8 * p + 0.1)
n = 5000
p = rng.uniform(0, 1, n)
y = (rng.uniform(0, 1, n) < (0.8 * p + 0.1)).astype(float)
out["bin_sensitivity_overconfident"] = {
    f"bins={b}": {"equal_mass": round(ece_equal_mass(p, y, b), 4), "equal_width": round(ece_equal_width(p, y, b), 4)}
    for b in (3, 5, 10, 20, 50, 100)
}
out["bin_sensitivity_overconfident"]["rms_equal_mass_10"] = round(ece_equal_mass(p, y, 10, rms=True), 4)
out["bin_sensitivity_overconfident"]["true_expected_gap"] = round(float(np.abs(0.8 * np.linspace(0, 1, 10001) + 0.1 - np.linspace(0, 1, 10001)).mean()), 4)

# (c) 상수(기저율) 예측기의 Brier는 pi(1-pi). "chance 0.25"는 pi=0.5일 때뿐
out["brier_constant_base_rate"] = {f"pi={pi}": round(pi * (1 - pi), 4) for pi in (0.05, 0.07, 0.14, 0.2, 0.46, 0.5, 0.715)}

# ---------------------------------------------------------------- 2. Table 1 Brier 대 기저율
# P(IK) Brier가 상수 기저율 예측기보다 나으려면 pi(1-pi) > Brier 여야 한다 -> pi 범위
table1 = {  # (AUROC, Brier) TriviaQA-only / all-4
    "TriviaQA": ((0.864, 0.151), (0.873, 0.145)),
    "Mixed-Arithmetic": ((0.928, 0.194), (0.987, 0.042)),
    "LAMBADA": ((0.606, 0.431), (0.853, 0.108)),
    "PyFuncSynth": ((0.687, 0.164), (0.881, 0.109)),
    "GSM8K": ((0.624, 0.200), (0.752, 0.121)),
}
t1 = {}
for k, (a, b) in table1.items():
    row = {}
    for name, (au, br) in (("trivia_only", a), ("all4", b)):
        disc = 1 - 4 * br
        if disc < 0:
            row[name] = {"brier": br, "pi_range_where_beats_constant": "none (최대 pi(1-pi)=0.25 < Brier)"}
        else:
            lo = (1 - math.sqrt(disc)) / 2; hi = (1 + math.sqrt(disc)) / 2
            row[name] = {"brier": br, "pi_range_where_beats_constant": [round(lo, 3), round(hi, 3)]}
    t1[k] = row
out["table1_brier_vs_constant"] = t1

# ---------------------------------------------------------------- 3. Hanley-McNeil AUROC 표준오차
def hm_se(auc, n_pos, n_neg):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(v)

# 표본 수는 논문에 없음. 시나리오로 계산 (n, 양성 비율).
se_rows = {}
for task, auc, scen in (
    ("LAMBADA(trivia-only, 0.606)", 0.606, [(1000, .5), (5000, .5), (5000, .3)]),
    ("GSM8K(trivia-only, 0.624)", 0.624, [(1319, .10), (1319, .2)]),
    ("GSM8K(all-4, 0.752)", 0.752, [(1319, .10), (1319, .2)]),
    ("Table3 B-test(0.8460 vs 0.8717)", 0.8460, [(2000, .5), (5000, .5), (10000, .5)]),
):
    se_rows[task] = {f"n={n},pos={f}": round(hm_se(auc, int(n * f), n - int(n * f)), 4) for n, f in scen}
out["hanley_mcneil_se_scenarios"] = se_rows
# 독립 가정의 차이 검정 (짝지은 DeLong은 상관 때문에 더 작은 SE -> 이 z는 보수적)
def z_indep(a1, a2, n, pos=.5):
    s = math.hypot(hm_se(a1, int(n * pos), n - int(n * pos)), hm_se(a2, int(n * pos), n - int(n * pos)))
    return round((a2 - a1) / s, 2)
out["table3_B_test_z_independent_conservative"] = {f"n={n}": z_indep(0.8460, 0.8717, n) for n in (500, 1000, 2000, 5000)}
out["table3_A_test_diff"] = round(0.8633 - 0.8631, 4)

# ---------------------------------------------------------------- 4. Table 2 교차 실험
# A: 650개(A만 정답) vs B: 826개(B만 정답)로 해석. 평균 P(IK)의 표준오차는 sd 가정 필요 (그림 20 산포에서 대략 0.15-0.30)
t2 = {}
for sd in (0.15, 0.2, 0.3):
    # 모델 A: 자기 정답 부분집합(650) 평균 0.463 대 상대가 맞힌 부분집합(826) 평균 0.408
    se_a = sd * math.sqrt(1 / 650 + 1 / 826)
    z_a = (0.463 - 0.408) / se_a
    # 모델 B: 자기 정답(826) 0.477 대 상대 정답(650) 0.409
    se_b = sd * math.sqrt(1 / 650 + 1 / 826)
    z_b = (0.477 - 0.409) / se_b
    t2[f"sd={sd}"] = {"A_diff": 0.055, "z_A": round(z_a, 2), "B_diff": 0.068, "z_B": round(z_b, 2)}
out["table2_cross_diag_vs_offdiag_z_independent"] = t2
# 대각 - 비대각 합산 효과: (0.463+0.477)-(0.408+0.409)
out["table2_interaction"] = round((0.463 + 0.477) - (0.408 + 0.409), 3)
out["table2_note"] = "A 행 0.463/0.408, B 행 0.409/0.477. 650=A만 정답, 826=B만 정답(본문 5.5 순서). 부분집합 단위 표준편차는 논문에 없어 시나리오"

# ---------------------------------------------------------------- 5. 그림에서 읽은 값 산술
# Fig 14 (800M, 3B, 12B, 52B), 읽은 값
fig14 = {
    "TriviaQA": [0.813, 0.884, 0.877, 0.864],
    "Mixed-Arithmetic": [0.372, 0.611, 0.773, 0.928],
    "LAMBADA": [0.536, 0.460, 0.590, 0.611],
    "PyFuncSynth": [0.598, 0.502, 0.677, 0.690],
}
out["fig14_auroc_by_size_read"] = fig14
out["fig14_monotone_nondecreasing"] = {k: bool(all(b >= a for a, b in zip(v, v[1:]))) for k, v in fig14.items()}
out["fig14_52B_matches_table1"] = {"TriviaQA": (0.864, 0.864), "Mixed-Arithmetic": (0.928, 0.928), "LAMBADA": (0.611, 0.606), "PyFuncSynth": (0.690, 0.687)}

# Fig 11 / Fig 1: 52B(와 작은 모델) 기저 정확도와 P(True)>0.5 조건부 정확도
sizes = ["3B", "12B", "52B"]
fig11 = {
    "Lambada": {"base": [0.46, 0.58, 0.715], "cond20": [0.46, 0.685, 0.82], "brier20": [0.262, 0.185, 0.126]},
    "Codex": {"base": [0.065, 0.094, 0.138], "cond20": [0.07, 0.207, 0.292], "brier20": [0.209, 0.155, 0.19]},
}
res11 = {}
for t, d in fig11.items():
    rows = []
    for i, s in enumerate(sizes):
        b, c, br = d["base"][i], d["cond20"][i], d["brier20"][i]
        const = b * (1 - b)
        rows.append({
            "size": s, "base_acc": b, "cond_acc": c, "abs_gain": round(c - b, 3),
            "error_ratio_cond_over_base": round((1 - c) / (1 - b), 3),
            "coverage_upper_bound": round(b / c, 3),
            "brier_model": br, "brier_constant_base_rate": round(const, 3),
            "model_beats_constant": bool(br < const),
        })
    res11[t] = rows
out["fig11_derived"] = res11

# Fig 1 오른쪽, 52B 다섯 과제
fig1 = {"GSM8k": (0.18, 0.38), "Arithmetic": (0.56, 0.715), "TriviaQA": (0.45, 0.69), "Lambada": (0.715, 0.82), "Codex": (0.14, 0.29)}
out["fig1_52B_coverage_upper_bound"] = {k: round(b / c, 3) for k, (b, c) in fig1.items()}
out["fig1_52B_mean_base_cond"] = [round(np.mean([v[0] for v in fig1.values()]), 3), round(np.mean([v[1] for v in fig1.values()]), 3)]

# Fig 4 우측 ECE (읽은 값, 800M/3B/12B/52B)
fig4 = {"lettered_5shot": [0.13, 0.078, 0.028, 0.019], "lettered_0shot": [0.28, 0.115, 0.048, 0.055],
        "true_false_5shot": [0.11, 0.09, 0.09, 0.011], "bb_default_5shot": [0.29, 0.29, 0.23, 0.20]}
out["fig4_ece_read"] = fig4
# 잡음 바닥: BIG-bench 총 예측 수를 모르므로 N 시나리오
out["fig4_noise_floor_by_N"] = {f"N={N}": floor.get(f"n={N},bins=10") for N in (5000, 20000, 100000)}

# Fig 15 GSM8K 보정 (all-4 학습, 52B, 읽은 10점)
f15 = [(0.09, 0.083), (0.20, 0.107), (0.23, 0.13), (0.255, 0.14), (0.28, 0.14), (0.31, 0.165), (0.345, 0.195), (0.38, 0.175), (0.44, 0.233), (0.74, 0.248)]
out["fig15_all4_gsm8k"] = {
    "ece_mad": round(float(np.mean([abs(x - y) for x, y in f15])), 3),
    "mean_pred": round(float(np.mean([x for x, _ in f15])), 3),
    "mean_gt": round(float(np.mean([y for _, y in f15])), 3),
    "top_bin_pred_minus_gt": round(0.74 - 0.248, 3),
    "fig1_gsm8k_base_acc_52B": 0.18,
}

# ---------------------------------------------------------------- 6. 토이: P(True)에 자기 표본을 k개 보여 줄 때
def simulate(k, n_q=4000, correct_diverse=False, n_wrong=20, noise=0.0):
    """문항마다 정답 확률 q~Beta(0.8,0.8)로 표본을 뽑는다. 틀린 답은 n_wrong개 중 균등.
    correct_diverse=True이면 정답도 4가지 표현으로 갈라진다(HumanEval 식).
    평가자는 k-1개의 다른 표본 중 제안 답과 같은 것의 비율(평활)로 P(True)를 낸다.
    k=1이면 정보가 없어 전체 평균. noise>0이면 단일 표본 신호(정답 여부를 노이즈로 보는 약한 직관)를 섞는다."""
    q = rng.beta(0.8, 0.8, n_q)
    ys, ps = [], []
    for qi in q:
        def draw():
            if rng.random() < qi:
                return ("c", int(rng.integers(0, 4)) if correct_diverse else 0)
            return ("w", int(rng.integers(0, n_wrong)))
        prop = draw()
        others = [draw() for _ in range(k - 1)]
        ys.append(1.0 if prop[0] == "c" else 0.0)
        agree = sum(o == prop for o in others)
        # 단일 표본 직관: 정답이면 약간 더 그럴듯하다고 느낌
        # 약한 단일 표본 신호: 40% 확률로 틀린 방향(AUROC 약 0.6)
        sig = (prop[0] == "c") ^ (rng.random() < 0.4)
        intuit = 0.5 + noise * (1 if sig else -1)
        ps.append((agree + intuit * 1.0) / (k - 1 + 1.0) if k > 1 else intuit)
    return np.array(ps), np.array(ys)

toy = {}
for diverse in (False, True):
    for k in (1, 2, 5, 20):
        p, y = simulate(k, correct_diverse=diverse, noise=0.1)
        # 재보정(등압 대신 단순 구간 평균)으로 보정과 변별을 분리: 변별=AUROC, 보정=원 점수 ECE
        h = len(p) // 2
        edges = np.quantile(p[:h], np.linspace(0, 1, 11)[1:-1])
        bi = np.searchsorted(edges, p)
        tab = np.array([y[:h][bi[:h] == b].mean() if (bi[:h] == b).any() else y[:h].mean() for b in range(10)])
        pr = tab[bi[h:]]
        toy[f"diverse_correct={diverse},k={k}"] = {
            "auroc": round(auroc(p + 1e-9 * rng.random(len(p)), y), 3),
            "brier_recalibrated_heldout": round(brier(pr, y[h:]), 3),
            "ece_recalibrated_heldout": round(ece_equal_mass(pr + 1e-9 * rng.random(len(pr)), y[h:], 10), 3),
            "ece_raw": round(ece_equal_mass(np.clip(p, 0, 1), y, 10), 3),
            "brier_raw": round(brier(np.clip(p, 0, 1), y), 3),
            "brier_const": round(float(y.mean() * (1 - y.mean())), 3),
        }
out["toy_k_samples"] = toy

with open(os.path.join(HERE, "recompute_calibration_results.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
