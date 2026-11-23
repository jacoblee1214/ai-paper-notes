"""몬테카를로 두 가지(수 초). minimal_impl의 성질을 표로 남긴다.

  E1. 군집 수 G에 따른 95% 구간 포함률: 단순 SE, 식 (4) 군집 SE, CR1(G/(G-1)) 보정.
      점수 분산 1, 군집 크기 10, 군집 내 상관 0.5(설계 효과 5.5), 참 평균 0.
  E2. 두 모델 점수(U[0,1] 난이도, 상관 rho)에서 짝지은 분산 감소율과 4.2절 산술(논문 1/3 대 정답 rho).
  E4. 문항 수 N이 작을 때 식 (1)의 95% 구간 포함률(이진 점수, 난이도 θ~U[0,1]을 새로 뽑아 평균), Wilson 구간과 비교.
  E3. 필요 표본 수 식 (9)로 정한 n에서 문항 수 n(군집 크기 1)와 K에 따른 시뮬레이션 검정력.
결과: analysis/mc_results.json.  실행: python3 -I analysis/mc_experiments.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as m  # noqa: E402

rng = np.random.default_rng(0)
out = {}

# E1
size, icc, reps = 10, 0.5, 4000
rows = []
for g in (5, 10, 20, 50, 100, 400):
    x = (rng.normal(0, np.sqrt(icc), (reps, g, 1)) + rng.normal(0, np.sqrt(1 - icc), (reps, g, size))).reshape(reps, -1)
    cl = np.repeat(np.arange(g), size)
    cov = {"naive": 0, "cluster_eq4": 0, "cluster_cr1": 0, "cluster_cr1_t": 0}
    tcrit = stats.t.ppf(0.975, g - 1)
    for r in range(reps):
        mu = abs(x[r].mean())
        cov["naive"] += mu < m.Z975 * m.se_clt(x[r])
        cov["cluster_eq4"] += mu < m.Z975 * m.se_clustered(x[r], cl)
        s1 = m.se_clustered(x[r], cl, small_sample=True)
        cov["cluster_cr1"] += mu < m.Z975 * s1
        cov["cluster_cr1_t"] += mu < tcrit * s1
    rows.append({"clusters": g, **{k: v / reps for k, v in cov.items()}})
out["E1_coverage_by_clusters"] = rows

# E2
rows = []
for rho in (0.0, 0.25, 0.5, 0.75, 0.9):
    r = 2 * np.sin(np.pi * rho / 6)
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=400000)
    xa, xb = stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])
    unp = np.var(xa) + np.var(xb)
    pai = np.var(xa - xb)
    rows.append({"rho": rho, "var_unpaired": unp, "var_paired": pai, "reduction": 1 - pai / unp})
out["E2_paired_reduction_uniform"] = rows

# E3
rows = []
delta, alpha = 0.03, 0.05
for k in (1, 4, 16):
    sig2, omega2 = 1 / 6, 1 / 12
    n = int(np.ceil(m.n_required(delta, omega2, sig2, sig2, k, k)))
    reps = 3000
    # 이진 점수 모델: 난이도 xa, xb(상관 0.5 U[0,1]) 에 delta를 더하는 대신 A의 난이도를 위로 이동(clip)해 평균차 delta 유지
    r = 2 * np.sin(np.pi * 0.5 / 6)
    z = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=(reps, n))
    xb = stats.norm.cdf(z[..., 1])
    xa = np.clip(stats.norm.cdf(z[..., 0]) + delta, 0, 1)       # 경계에서 약간 잘려 평균차는 delta보다 조금 작음
    true_delta = float((xa - xb).mean())
    sa = rng.binomial(k, xa) / k
    sb = rng.binomial(k, xb) / k
    d = sa - sb
    zs = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(n))
    om2 = float(np.var(xa - xb))
    pred = m.power_z(n, true_delta, om2, sig2, sig2, k, k)       # 식 (9)의 역 (실제 omega²과 sigma²는 이진이라 xa(1-xa) 평균)
    rows.append({"K": k, "n_eq9": n, "true_delta": true_delta, "sim_power": float(np.mean(np.abs(zs) > m.Z975)),
                 "omega2_actual": om2, "power_from_formula_actual_params": pred})
out["E3_power_binary_sim"] = rows

# E4
rows = []
for n in (5, 10, 30, 100, 300, 1000):
    reps = 20000
    th = rng.random(reps)
    y = rng.random((reps, n)) < th[:, None]
    p = y.mean(1)
    se = np.sqrt(y.var(1, ddof=1) / n)
    clt = np.mean(np.abs(p - th) < m.Z975 * se)
    # Wilson
    z = m.Z975
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    wil = np.mean(np.abs(centre - th) < half)
    rows.append({"N": n, "clt_eq1": float(clt), "wilson": float(wil)})
out["E4_iid_coverage_small_n"] = rows

Path(__file__).with_name("mc_results.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k)
    for row in v:
        print(" ", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()})
