"""4.1절의 베이즈 로지스틱 회귀(특징 23개, 라플라스 사전분포 b=0.01, 15K 쌍)가 합성 데이터에서
(1) 보고된 신용 구간 폭을 내려면 특징이 얼마나 자주 나타나야 하는지,
(2) 사전분포가 효과 크기를 얼마나 줄이는지(축소 편향)를 확인하는 numpy 토이입니다.
논문의 hh-rlhf 특징 라벨은 공개되지 않아 실제 데이터 재분석은 못 합니다. 특징 밀도는 가정입니다.
CPU, 수 분 이내. 결과: toy_bayes_logistic_results.json
"""
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
FV = json.load(open(os.path.join(HERE, "figure_values.json")))["fig5_feature_effects"]
B = 0.01
N = 15000
P = len(FV) - 1  # 첫 행(믿음 일치)은 explicit+implicit 두 특징의 합이라 하나로 둔 22+1 = 23개 전부 사용
P = len(FV)
sig = lambda z: 1 / (1 + np.exp(-z))
alpha_true = np.array([math.log(r["median"] / (100 - r["median"])) for r in FV])
hw_prob = np.array([(r["ci95"][1] - r["ci95"][0]) / 2 for r in FV])


def implied_density():
    """보고된 95% 구간 반폭(확률 %p)에서 효과 α의 표준오차를 구하고, 특징 i가 0이 아닐 확률 d_i를 거꾸로 추정.
    정보량 ≈ N*d*σ'(0)=N*d/4 (다른 특징과의 상관은 무시)."""
    se_a = (hw_prob / 100) / 1.96 / 0.25  # dp/dα = p(1-p) ≈ 0.25
    return 4 / (N * se_a ** 2)


def simulate(density, seed, n_samp=1500, warm=500):
    rng = np.random.default_rng(seed)
    d = np.broadcast_to(density, (P,))
    X = np.zeros((N, P))
    for j in range(P):
        nz = rng.random(N) < d[j]
        X[:, j] = nz * rng.choice([-1.0, 1.0], N)
    y = (rng.random(N) < sig(X @ alpha_true)).astype(float)

    def nlp(a):
        z = X @ a
        return np.sum(np.logaddexp(0, z) - y * z) + np.sum(np.abs(a)) / B

    def grad(a):
        return X.T @ (sig(X @ a) - y) + np.sign(a) / B

    a0 = minimize(nlp, np.zeros(P), jac=grad, method="L-BFGS-B").x
    p0 = sig(X @ a0)
    H = (X * (p0 * (1 - p0))[:, None]).T @ X
    L = np.linalg.cholesky(np.linalg.inv(H))  # 질량 행렬 = H: 운동량 p ~ N(0, H) -> 위치 속도 = H^-1 p
    Minv = np.linalg.inv(H)
    Mchol = np.linalg.cholesky(H)
    a = a0.copy(); eps = 0.5; samples = []; acc = 0
    for it in range(warm + n_samp):
        p = Mchol @ rng.standard_normal(P)
        a_new, p_new = a.copy(), p.copy()
        p_new -= 0.5 * eps * grad(a_new)
        for _ in range(8):
            a_new += eps * (Minv @ p_new)
            p_new -= eps * grad(a_new)
        p_new += 0.5 * eps * grad(a_new)  # 마지막 반 걸음 보정
        dH = (nlp(a_new) + 0.5 * p_new @ Minv @ p_new) - (nlp(a) + 0.5 * p @ Minv @ p)
        ok = math.log(rng.random()) < -dH
        if ok:
            a = a_new
        if it < warm:
            eps *= 1.05 if ok else 0.9
        else:
            acc += ok; samples.append(a.copy())
    S = np.array(samples)
    prob = sig(S)
    med = np.median(prob, 0) * 100
    lo, hi = np.percentile(prob, [2.5, 97.5], 0) * 100
    mle_se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {"accept_rate": round(acc / n_samp, 2), "eps": round(eps, 3), "median": med, "lo": lo, "hi": hi, "mle_se_alpha": mle_se, "map_alpha": a0}


def main():
    dens = implied_density()
    out = {"implied_feature_density_from_reported_ci": {r["feature"]: round(float(d), 2) for r, d in zip(FV, dens)},
           "implied_density_median": round(float(np.median(dens)), 2), "N": N, "prior_b": B}
    true_p = sig(alpha_true) * 100
    out["runs"] = {}
    for name, d in [("implied_per_feature_density_clipped_to_1", np.clip(dens, 0.02, 1.0)), ("density_0.15_all", 0.15), ("density_0.5_all", 0.5)]:
        res = simulate(d, seed=0)
        shrink = true_p - res["median"]
        hw = (res["hi"] - res["lo"]) / 2
        out["runs"][name] = {"accept_rate": res["accept_rate"], "mean_shrinkage_pp_median_vs_true": round(float(np.mean(shrink)), 3),
                             "max_shrinkage_pp": round(float(np.max(shrink)), 3), "top_feature_true_median_hat": [round(float(true_p[0]), 2), round(float(res["median"][0]), 2)],
                             "mean_halfwidth_pp_sim": round(float(np.mean(hw)), 2), "mean_halfwidth_pp_reported_fig5": round(float(np.mean(hw_prob)), 2),
                             "n_features_zeroed_by_MAP_of_23": int(np.sum(np.abs(res["map_alpha"]) < 1e-6)),
                             "ci_excludes_50_sim": int(np.sum((res["lo"] > 50) | (res["hi"] < 50))), "ci_excludes_50_reported": int(sum((r["ci95"][0] > 50) or (r["ci95"][1] < 50) for r in FV))}
    json.dump(out, open(os.path.join(HERE, "toy_bayes_logistic_results.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
