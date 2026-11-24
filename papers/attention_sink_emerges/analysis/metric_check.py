"""Sink_k^eps 지표(논문 3.2절)와 부록 C.1 명제 1–4의 검산. numpy만 사용.

1. 균일 attention의 alpha_1 = H_T / T (기준선), 그리고 T에 따른 변화(논문 그림 3의 하락 경향의 원인)
2. "첫 토큰에 평균 w를 주고 나머지는 균일"인 헤드의 alpha_1 = (1 + (T-1) w) / T, 임계값 eps에 대응하는 w*
3. alpha_1 의 범위 1/T <= alpha_1 <= 1 (논문 본문 텍스트는 "T <= alpha" 로 추출됨, 원문 PDF 기호 확인용)
4. 명제 1(NoPE 균일), 명제 4(Rotary 상한 e^{2xi}/(e^{2xi}+t-1))의 수치 확인과 상한이 의미를 갖는 조건
실행: python3 -I analysis/metric_check.py
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = {}


def alpha1(A):
    """A: (T,T) 하삼각 attention. 논문 식: alpha_1 = (1/T) sum_i A[i,0] (k=1 이므로 분모 T-1+1 = T)."""
    return A[:, 0].sum() / A.shape[0]


# 1. 균일 attention 기준선
rows = []
for T in [8, 9, 10, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    A = np.tril(np.ones((T, T)))
    A /= A.sum(1, keepdims=True)
    H = sum(1.0 / i for i in range(1, T + 1))
    assert abs(alpha1(A) - H / T) < 1e-12
    rows.append({"T": T, "uniform_alpha1": H / T})
out["uniform_baseline"] = rows
# 균일 헤드가 eps=0.3 을 넘는 가장 큰 T
out["max_T_where_uniform_exceeds_0.3"] = max(r["T"] for r in rows if r["uniform_alpha1"] > 0.3)

# 2. w* : alpha_1 = eps 가 되는 평균 첫 토큰 질량 (i>=2 에서 일정한 w)
w_rows = []
for T in [16, 64, 256, 1024, 2048]:
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w = (eps * T - 1) / (T - 1)
        # 직접 구성해 검산
        A = np.zeros((T, T))
        A[0, 0] = 1
        for i in range(1, T):
            A[i, 0] = w
            A[i, 1:i + 1] = (1 - w) / i
        assert abs(A.sum(1) - 1).max() < 1e-12 and abs(alpha1(A) - eps) < 1e-12
        w_rows.append({"T": T, "eps": eps, "w_star": w})
out["w_star"] = w_rows

# 3. 범위: 첫 토큰 질량 0 이면 1/T, 모든 질의가 첫 토큰만 보면 1
T = 64
A0 = np.tril(np.ones((T, T)))
A0[1:, 0] = 0
A0 /= A0.sum(1, keepdims=True)
A1 = np.zeros((T, T))
A1[:, 0] = 1
out["range_T64"] = {"min": alpha1(A0), "one_over_T": 1 / T, "max": alpha1(A1)}

# 4. 명제 1: NoPE + 반복 토큰 -> 균일. 명제 4: Rotary + 반복 토큰 -> 상한
rng = np.random.RandomState(0)
dh = 32
q = rng.randn(dh)
k = rng.randn(dh)
T = 64
S = np.full((T, T), q @ k)  # NoPE: 모든 위치의 q, k 가 같다
S = np.where(np.tril(np.ones((T, T))) > 0, S, -np.inf)
A = np.exp(S - S.max(1, keepdims=True))
A /= A.sum(1, keepdims=True)
out["prop1_nope_uniform_max_abs_err"] = float(np.abs(A - np.tril(np.ones((T, T))) / np.arange(1, T + 1)[:, None]).max())
out["prop1_nope_alpha1"] = alpha1(A)


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((dh, dh))
    for j in range(dh // 2):
        R[2 * j:2 * j + 2, 2 * j:2 * j + 2] = [[c[j], -s[j]], [s[j], c[j]]]
    return R


theta = 1.0 / (10000 ** (np.arange(0, dh, 2) / dh))
prop4 = []
for xi_scale in [0.5, 1.0, 2.0, 4.0, 8.0]:
    qq = q / np.linalg.norm(q) * math.sqrt(xi_scale)
    kk = k / np.linalg.norm(k) * math.sqrt(xi_scale)
    xi = float(np.linalg.norm(qq) * np.linalg.norm(kk))
    S = np.full((T, T), -np.inf)
    for t in range(T):
        for i in range(t + 1):
            # <q_t, k_i> = q R_{Theta, i-t} k  (논문 식 23); 같은 q, k 이므로 오프셋만 중요
            S[t, i] = qq @ rot((i - t) * theta) @ kk
    A = np.exp(S - S.max(1, keepdims=True))
    A /= A.sum(1, keepdims=True)
    # 상한 e^{2xi}/(e^{2xi}+t-1), t 는 1-indexed 질의 위치
    ts = np.arange(1, T + 1)
    bound = np.exp(2 * xi) / (np.exp(2 * xi) + ts - 1)
    holds = bool((A.max(1) <= bound + 1e-12).all())
    # 상한만으로 alpha_1 < 0.3 을 보증할 수 있나
    ub_alpha1 = float(bound.sum() / T)
    prop4.append({"xi": xi, "bound_holds": holds, "alpha1_actual": float(alpha1(A)),
                  "alpha1_upper_bound_from_prop4": ub_alpha1,
                  "bound_vacuous_for_sink_decision": ub_alpha1 > 0.3})
out["prop4_rotary"] = prop4
# 상한이 alpha_1<0.3 을 보증하는 최대 xi (T=64)
lo, hi = 0.0, 20.0
for _ in range(80):
    mid = (lo + hi) / 2
    ub = (np.exp(2 * mid) / (np.exp(2 * mid) + np.arange(1, 65) - 1)).sum() / 64
    lo, hi = (mid, hi) if ub < 0.3 else (lo, mid)
out["prop4_max_xi_for_guarantee_T64_eps0.3"] = lo

(ROOT / "analysis" / "metric_check_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ["max_T_where_uniform_exceeds_0.3", "range_T64", "prop1_nope_uniform_max_abs_err",
                                       "prop4_max_xi_for_guarantee_T64_eps0.3"]}, indent=1))
print([(r["T"], round(r["uniform_alpha1"], 4)) for r in out["uniform_baseline"]])
print([(r["T"], r["eps"], round(r["w_star"], 4)) for r in out["w_star"] if r["eps"] == 0.3])
for r in prop4:
    print(r)
