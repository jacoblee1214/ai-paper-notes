"""FID의 표본 수 편향 토이 (numpy, 1분 이내).

같은 분포에서 뽑은 두 표본 집합의 FID는 참값 0이어야 하지만 유한 표본에서는 양수다(공분산 추정 오차).
논문은 본 비교(표 3, 9)에 30k, 절제(표 4–8)에 5k 프롬프트를 쓴다(각주 11). 이 편향이 5k에서 얼마나 커지는지 가늠한다.
설정: 차원 d의 가우시안 특성(공분산 = 고유값이 거듭제곱으로 줄어드는 대각), 생성 표본 N개 vs 기준 표본 N_ref=30,000개. 같은 분포이므로 참 FID = 0.
FID = |μ1-μ2|² + tr(Σ1+Σ2-2(Σ1Σ2)^{1/2}).  실제 FID는 d=2048 Inception 특성이고 분포가 가우시안이 아니므로 절대값이 아니라 N 의존성 비율만 본다.
출력: fid_bias_results.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def fid(x, y):
    m1, m2 = x.mean(0), y.mean(0)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    w, U = np.linalg.eigh(s1)
    r1 = (U * np.sqrt(np.clip(w, 0, None))) @ U.T
    ev = np.linalg.eigvalsh(r1 @ s2 @ r1)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


out = {}
for d in (64, 128, 256):
    scale = np.sqrt(np.arange(1, d + 1) ** -0.7)      # 고유값 ~ k^-0.7: 완만하게 줄어드는 스펙트럼
    ref = rng.standard_normal((30000, d)) * scale
    out[str(d)] = {}
    for N in (2000, 5000, 10000, 30000):
        v = [fid(rng.standard_normal((N, d)) * scale, ref) for _ in range(6)]
        out[str(d)][str(N)] = {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)}
    print(d, out[str(d)], flush=True)
# 편향이 N에 어떻게 줄어드는지: log-log 기울기, 그리고 d에 대한 의존
sl = {}
for d in out:
    xs = np.log([2000, 5000, 10000, 30000]); ys = np.log([out[d][str(n)]["mean"] for n in (2000, 5000, 10000, 30000)])
    sl[d] = {"slope_vs_N": round(float(np.polyfit(xs, ys, 1)[0]), 3), "bias_5k_over_30k": round(out[d]["5000"]["mean"] / out[d]["30000"]["mean"], 2)}
out["summary"] = sl
out["note"] = "참 FID=0인 설정에서의 편향. 실제 d=2048·비가우시안 특성에서는 절대 크기가 다르다. 논문의 표 3(30k)과 표 7(5k)의 FID 차이 16.8 대 18.6은 이 편향과 CFG 5 대 3의 합이다(분리 불가)."
(HERE / "fid_bias_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(sl)
