"""논문 표 2·식 11(필요 실행 수)을 검산하고, 문항 이항 모형 시뮬레이션과 문항 모집단 관점(Error Bars)으로 비교한다.

1. 식 11 n = 2 ((z_{a/2} + z_b) / (Delta/sigma))^2 로 표 2 전체 재계산
2. 같은 문제를 비중심 t로 정확히 풀었을 때 필요한 n (n이 작을 때 z 근사의 편향)
3. data/의 문항별 성공 확률로 만든 이항 모형에서 실제 검정력 시뮬레이션 (n=표 2의 값)
4. 같은 문항 500개에서 두 에이전트를 비교할 때 실행 수를 무한히 늘려도 남는 문항 표본 오차 하한

실행: python3 -I analysis/power_check.py  (논문 폴더에서; data/가 필요)
출력: power_check_results.json
"""
import json
import os
import numpy as np
from scipy import stats
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N = 500

PAPER_T2 = {  # (improvement %, sigma %) -> (n05, n01, n001)
    (1, 0.7): (8, 12, 17), (1, 1.5): (36, 53, 77), (1, 1.8): (51, 76, 111),
    (2, 0.7): (2, 3, 5), (2, 1.5): (9, 14, 20), (2, 1.8): (13, 19, 28),
    (5, 0.7): (1, 1, 1), (5, 1.5): (2, 3, 4), (5, 1.8): (3, 4, 5),
    (10, 0.7): (1, 1, 1), (10, 1.5): (1, 1, 1), (10, 1.8): (1, 1, 2),
}


def n_z(delta, sigma, alpha, power=0.8):
    za, zb = stats.norm.isf(alpha / 2), stats.norm.ppf(power)
    return 2 * ((za + zb) / (delta / sigma)) ** 2


def power_t(n, delta, sigma, alpha):
    """n runs per arm, 같은 sigma, 두 표본 t 검정의 정확한 검정력(비중심 t)."""
    if n < 2:
        return float("nan")
    df = 2 * n - 2
    nc = delta / (sigma * np.sqrt(2 / n))
    tc = stats.t.isf(alpha / 2, df)
    return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def n_t(delta, sigma, alpha, power=0.8):
    n = 2
    while power_t(n, delta, sigma, alpha) < power:
        n += 1
        if n > 5000:
            return None
    return n


def main():
    res = {}
    # 1, 2. 표 2 재계산
    rows = []
    for (d, s), paper in PAPER_T2.items():
        zc = [int(np.ceil(n_z(d, s, a))) for a in (0.05, 0.01, 0.001)]
        zr = [float(n_z(d, s, a)) for a in (0.05, 0.01, 0.001)]
        tc = [n_t(d, s, a) for a in (0.05, 0.01, 0.001)]
        # 표 값은 올림이 아니라 반올림일 수도 있으므로 둘 다 대조
        rr = [int(round(x)) for x in zr]
        rows.append({"improve": d, "sigma": s, "paper": paper, "z_ceil": zc, "z_round": rr, "z_raw": zr, "t_exact": tc})
    res["table2"] = rows
    res["table2_match_ceil_with_floor1"] = all([max(1, a) for a in r["z_ceil"]] == list(r["paper"]) for r in rows)
    res["table2_mismatches_ceil"] = [(r["improve"], r["sigma"], r["paper"], r["z_ceil"]) for r in rows if [max(1, a) for a in r["z_ceil"]] != list(r["paper"])]
    # 그림 5 문장: 2% 효과, sigma 1.5, 검정력 95% -> 15
    res["power95_2pct_sigma1.5_z"] = float(n_z(2, 1.5, 0.05, 0.95))
    res["power95_2pct_sigma1.5_t"] = n_t(2, 1.5, 0.05, 0.95)
    # 논문 결론: 10회 실행 시 2%p 탐지 검정력
    res["power_n10_2pp_sigma1.5_exact_t"] = power_t(10, 2, 1.5, 0.05)
    res["power_n9_2pp_sigma1.5_exact_t"] = power_t(9, 2, 1.5, 0.05)
    res["power_n10_1pp_sigma1.5_exact_t"] = power_t(10, 1, 1.5, 0.05)

    # 3. 이항 모형 시뮬레이션: r2e-gym/Devstral-2/0.0 의 문항별 성공 확률(p_i = c_i/10)
    from recompute_runs import load_all  # 같은 폴더의 스크립트
    mats, _ = load_all()
    M = mats[("r2e-gym", "Devstral-2", "0.0")]
    p = M.mean(0)
    pbar = p.mean()
    rng = np.random.default_rng(1)

    def sim_power(shift_pp, n, alpha=0.05, sims=6000):
        lam = shift_pp / 100 / (1 - pbar)
        pB = p + lam * (1 - p)  # 모든 문항에서 실패 확률의 일부를 성공으로 바꿈
        rej = 0
        sd_a = []
        for _ in range(sims):
            ra = (rng.random((n, N)) < p).mean(1) * 100
            rb = (rng.random((n, N)) < pB).mean(1) * 100
            sd_a.append(ra.std(ddof=1) if n > 1 else 0)
            if n >= 2:
                if stats.ttest_ind(ra, rb).pvalue < alpha:
                    rej += 1
        return rej / sims, float(np.mean(sd_a))

    sim = {}
    for shift in (1, 2):
        for n in (2, 5, 9, 10, 20, 36):
            pw, sd = sim_power(shift, n)
            sim[f"shift{shift}_n{n}"] = {"power": pw, "mean_run_sd": sd}
    res["sim_item_model_r2e_devstral_T0"] = sim
    res["sim_sigma_pred_pp"] = float(np.sqrt((p * (1 - p)).sum() / N**2) * 100)

    # 거짓 양성: 두 에이전트가 같을 때 n회 평균 비교의 기각률 (= 5%여야 함), 단 두 에이전트가 문항 모집단에서 다를 때는?
    # 4. 문항 모집단 관점: 두 설정 A,B 비교에서 K->무한대여도 남는 문항 표본 SE
    cfg = json_cfg(mats)
    res["population_floor"] = cfg

    json.dump(res, open(os.path.join(HERE, "power_check_results.json"), "w"), indent=1, default=float)

    print("표 2 (논문 vs 식 11 올림 vs 정확한 t)")
    for r in rows:
        print(r["improve"], r["sigma"], r["paper"], r["z_ceil"], [round(x, 2) for x in r["z_raw"]], r["t_exact"])
    print("올림 불일치:", res["table2_mismatches_ceil"])
    print("검정력 95%, 2pp, sigma1.5: z", round(res["power95_2pct_sigma1.5_z"], 2), "t", res["power95_2pct_sigma1.5_t"])
    print("정확한 t 검정력: n=10,2pp:", round(res["power_n10_2pp_sigma1.5_exact_t"], 3), "n=9:", round(res["power_n9_2pp_sigma1.5_exact_t"], 3), "n=10,1pp:", round(res["power_n10_1pp_sigma1.5_exact_t"], 3))
    print("시뮬레이션 sigma_pred", round(res["sim_sigma_pred_pp"], 2))
    for k, v in sim.items():
        print(k, v)
    print("문항 모집단 하한")
    for c in cfg:
        print(c)


def json_cfg(mats):
    """각 (스캐폴드, 모델)의 기본 온도 대 0도 비교에서, 실행 수 K에 따른 쌍체 SE와 K->무한대 하한."""
    out = []
    for sc in ["nano-agent", "r2e-gym"]:
        for mdl, t in [("Qwen3-32B", "0.6"), ("DeepSWE-preview", "1.0"), ("Devstral-2", "0.2")]:
            A, B = mats[(sc, mdl, t)], mats[(sc, mdl, "0.0")]
            m = A.shape[0]
            pa, pb = A.mean(0), B.mean(0)
            d = pa - pb
            se10_sq = d.var(ddof=1) / N  # K=10 평균 사용 시 관측된 쌍체 SE^2 (비율^2)
            # 실행 잡음의 K=10 몫: 문항별 불편 조건분산의 평균 / (K N)
            sa = (A.sum(0) * (m - A.sum(0)) / (m * (m - 1))).mean()
            sb = (B.sum(0) * (m - B.sum(0)) / (m * (m - 1))).mean()
            run_part10 = (sa + sb) / (m * N)
            pop = max(0.0, se10_sq - run_part10)  # K->무한대 때 남는 SE^2
            floor_se = np.sqrt(pop) * 100
            mde_floor = 2.8016 * floor_se
            # 같은 문항 500개에서 "이 문항들에 한정한" 효과를 탐지하는 데 필요한 K (문항 모집단 무시)
            rec = {
                "pair": f"{sc}/{mdl} {t} vs 0.0",
                "diff_pp": float(d.mean() * 100),
                "se_paired_K10_pp": float(np.sqrt(se10_sq) * 100),
                "se_run_part_K10_pp": float(np.sqrt(run_part10) * 100),
                "se_floor_Kinf_pp": float(floor_se),
                "mde_80pct_a05_floor_pp": float(mde_floor),
            }
            # K=1,3,5,10,30에서의 SE와 80% 검정력 MDE
            for K in (1, 3, 5, 10, 30):
                se = np.sqrt(pop + (sa + sb) / (K * N)) * 100
                rec[f"mde_K{K}_pp"] = float(2.8016 * se)
            out.append(rec)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    main()
