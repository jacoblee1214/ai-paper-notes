"""공개 결과 파일(실행별 resolved_ids)에서 논문 표 1·3, 그림 1·6·7 수치와 문항 단위 분산 분해를 재계산한다.

입력: data/<설정>/<run>/*.json (HF ASSERT-KTH/agentic-evals-artifacts의 SWE-bench 평가 결과, 120개, 약 6 MB)
출력: recompute_runs_results.json

실행: python3 -I analysis/recompute_runs.py   (논문 폴더에서)
"""
import glob
import json
import os
import itertools
import numpy as np
from math import comb
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
N = 500

# 설정 이름 -> (scaffold, model, temp). 논문 표 1의 행.
CONFIGS = {
    "nano-agent-Qwen_Qwen3-32B": ("nano-agent", "Qwen3-32B", "0.6"),
    "nano-agent-agentica-org_DeepSWE-Preview": ("nano-agent", "DeepSWE-preview", "1.0"),
    "nano-agent-mistral_devstral-2512": ("nano-agent", "Devstral-2", "0.2"),
    "nano-agent-Qwen_Qwen3-32B-temp0": ("nano-agent", "Qwen3-32B", "0.0"),
    "nano-agent-agentica-org_DeepSWE-Preview__temp0": ("nano-agent", "DeepSWE-preview", "0.0"),
    "nano-agent-mistral_devstral-2512__temp0": ("nano-agent", "Devstral-2", "0.0"),
    "r2e-gym-Qwen_Qwen3-32B": ("r2e-gym", "Qwen3-32B", "0.6"),
    "r2e-gym-agentica-org__DeepSWE-preview": ("r2e-gym", "DeepSWE-preview", "1.0"),
    "r2e-gym-mistral_devstral-2512": ("r2e-gym", "Devstral-2", "0.2"),
    "r2e-gym-Qwen_Qwen3-32B__temp0": ("r2e-gym", "Qwen3-32B", "0.0"),
    "r2e-gym-agentica-org__DeepSWE-preview__temp0": ("r2e-gym", "DeepSWE-preview", "0.0"),
    "r2e-gym-mistral_devstral-2512__temp0": ("r2e-gym", "Devstral-2", "0.0"),
}

# 논문 표 1 (r, std, min, max) 와 표 3 (W, p)
PAPER_T1 = {
    ("nano-agent", "Qwen3-32B", "0.6"): (16.4, 0.7, 15.0, 17.2),
    ("nano-agent", "DeepSWE-preview", "1.0"): (31.4, 1.0, 28.8, 32.4),
    ("nano-agent", "Devstral-2", "0.2"): (63.5, 1.1, 61.8, 65.0),
    ("nano-agent", "Qwen3-32B", "0.0"): (16.4, 1.2, 14.4, 18.0),
    ("nano-agent", "DeepSWE-preview", "0.0"): (20.4, 1.0, 18.2, 21.4),
    ("nano-agent", "Devstral-2", "0.0"): (63.8, 1.6, 60.6, 66.6),
    ("r2e-gym", "Qwen3-32B", "0.6"): (23.9, 1.4, 21.4, 26.4),
    ("r2e-gym", "DeepSWE-preview", "1.0"): (34.4, 1.5, 31.6, 37.0),
    ("r2e-gym", "Devstral-2", "0.2"): (34.9, 1.5, 32.2, 37.0),
    ("r2e-gym", "Qwen3-32B", "0.0"): (22.3, 1.8, 19.8, 25.2),
    ("r2e-gym", "DeepSWE-preview", "0.0"): (19.2, 1.5, 17.0, 21.6),
    ("r2e-gym", "Devstral-2", "0.0"): (35.4, 1.7, 32.0, 37.8),
}
PAPER_T3 = {
    ("nano-agent", "DeepSWE-preview", "1.0"): (0.803, 0.016),
    ("r2e-gym", "DeepSWE-preview", "1.0"): (0.989, 0.996),
    ("nano-agent", "DeepSWE-preview", "0.0"): (0.881, 0.134),
    ("r2e-gym", "DeepSWE-preview", "0.0"): (0.923, 0.383),
    ("nano-agent", "Devstral-2", "0.2"): (0.894, 0.190),
    ("r2e-gym", "Devstral-2", "0.2"): (0.956, 0.733),
    ("nano-agent", "Devstral-2", "0.0"): (0.970, 0.893),
    ("r2e-gym", "Devstral-2", "0.0"): (0.952, 0.695),
    ("nano-agent", "Qwen3-32B", "0.6"): (0.885, 0.149),
    ("r2e-gym", "Qwen3-32B", "0.6"): (0.964, 0.834),
    ("nano-agent", "Qwen3-32B", "0.0"): (0.962, 0.809),
    ("r2e-gym", "Qwen3-32B", "0.0"): (0.964, 0.830),
}
# 그림 1·6·7에 적힌 pass@5, pass^5 (%) : (pass@5, pass^5)
PAPER_FIG = {
    ("nano-agent", "DeepSWE-preview", "1.0"): None,  # 그림 1 왼쪽은 r2e-gym
    ("r2e-gym", "DeepSWE-preview", "1.0"): (52.9, 15.5),
    ("nano-agent", "Devstral-2", "0.2"): (76.2, 49.1),
    ("nano-agent", "Qwen3-32B", "0.6"): (29.1, 6.3),
    ("r2e-gym", "Qwen3-32B", "0.6"): (44.1, 7.3),
    ("nano-agent", "DeepSWE-preview", "0.0"): (36.2, 7.3),
    ("r2e-gym", "DeepSWE-preview", "0.0"): (34.4, 8.1),
    ("nano-agent", "Qwen3-32B", "0.0"): (28.8, 6.8),
    ("r2e-gym", "Qwen3-32B", "0.0"): (42.4, 6.7),
    ("r2e-gym", "Devstral-2", "0.2"): (59.3, 11.6),
    ("nano-agent", "Devstral-2", "0.0"): (76.8, 48.4),
    ("r2e-gym", "Devstral-2", "0.0"): (60.3, 11.6),
}


def load_all():
    ids_all = None
    mats = {}
    for d, key in CONFIGS.items():
        runs = []
        for f in sorted(glob.glob(os.path.join(DATA, d, "*", "*.json"))):
            j = json.load(open(f))
            assert j["total_instances"] == N, f
            runs.append(set(j["resolved_ids"]))
        assert len(runs) == 10, (d, len(runs))
        # 전체 문항 id: completed + incomplete 모두 resolved_ids 밖. 문항 id는 모든 실행의 submitted 합집합이 필요하나
        # 여기서는 resolved 합집합에 의존하지 않도록 첫 파일의 submitted_ids 개수만 확인하고, 행렬은 resolved 합집합 기준으로 만든다.
        mats[key] = runs
    universe = sorted(set().union(*[set().union(*r) for r in mats.values()]))
    # 한 번도 풀리지 않은 문항은 행렬에 없지만 N=500 분모에 들어간다 (c_i=0)
    out = {}
    for key, runs in mats.items():
        M = np.zeros((10, len(universe)), dtype=int)
        idx = {u: i for i, u in enumerate(universe)}
        for j, r in enumerate(runs):
            for u in r:
                M[j, idx[u]] = 1
        pad = np.zeros((10, N - len(universe)), dtype=int)
        out[key] = np.concatenate([M, pad], axis=1)  # (10 runs, 500 tasks)
    return out, len(universe)


def pass_at_k(c, m, k):
    return float(np.mean([1 - comb(m - ci, k) / comb(m, k) for ci in c]))


def pass_hat_k(c, m, k):
    return float(np.mean([comb(ci, k) / comb(m, k) for ci in c]))


def mcnemar_p(a, b):
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b01, b10), n, 0.5).pvalue)


def main():
    mats, n_universe = load_all()
    res = {"n_tasks": N, "tasks_ever_resolved": n_universe, "configs": {}}
    rng = np.random.default_rng(0)

    table1_rows = []
    for key, M in mats.items():
        r = M.sum(axis=1) / N * 100  # 실행별 해결률 %
        c = M.sum(axis=0)  # 문항별 성공 횟수
        m = M.shape[0]
        p = c / m
        mean = r.mean()
        sd1, sd0 = r.std(ddof=1), r.std(ddof=0)
        # 문항 단위 분해 (Error Bars 식 Var(mu_hat)=(Var(x)+E[sigma^2])/n 의 n=500)
        pbar = p.mean()
        e_sigma2 = float((c * (m - c) / (m * (m - 1))).mean())  # E[p_i(1-p_i)] 불편추정
        var_x = float(pbar * (1 - pbar) - e_sigma2)  # Var(x) = total - E[sigma^2]
        total_var_single = pbar * (1 - pbar)
        pred_run_sd = float(np.sqrt(e_sigma2 / N) * 100)  # 문항이 독립일 때 한 실행의 표준편차(%p)
        se_total = float(np.sqrt(total_var_single / N) * 100)  # 문항 모집단까지 포함한 한 번 실행의 표준오차
        # 쌍별 두 실행 차이 >= 2%p 비율, McNemar 유의 비율
        pairs = list(itertools.combinations(range(m), 2))
        d2 = np.mean([abs(r[a] - r[b]) >= 2.0 for a, b in pairs])
        d3 = np.mean([abs(r[a] - r[b]) >= 3.0 for a, b in pairs])
        mc = np.mean([mcnemar_p(M[a], M[b]) < 0.05 for a, b in pairs])
        # 실행 간 문항 성공 지시자의 평균 쌍별 상관 (phi)
        cors = [np.corrcoef(M[a], M[b])[0, 1] for a, b in pairs]
        sw = stats.shapiro(r)
        # 관측 실행 분산이 문항 독립 모형 예측과 같은가: (m-1)s^2/sigma_pred^2 ~ chi2(m-1) (양측)
        chi = (m - 1) * sd1 ** 2 / pred_run_sd ** 2
        chi_p = float(2 * min(stats.chi2.cdf(chi, m - 1), stats.chi2.sf(chi, m - 1)))
        rec = {
            "key": "/".join(key),
            "mean": mean, "sd_ddof1": sd1, "sd_ddof0": sd0,
            "min": r.min(), "max": r.max(), "range": r.max() - r.min(),
            "paper": PAPER_T1[key],
            "shapiro_W": float(sw.statistic), "shapiro_p": float(sw.pvalue), "paper_T3": PAPER_T3[key],
            "pass@1": pass_at_k(c, m, 1),
            "pass@5": pass_at_k(c, m, 5) * 100, "pass^5": pass_hat_k(c, m, 5) * 100,
            "pass@10": pass_at_k(c, m, 10) * 100, "pass^10": pass_hat_k(c, m, 10) * 100,
            "paper_fig": PAPER_FIG.get(key),
            "tasks_always": int((c == m).sum()), "tasks_never": int((c == 0).sum()),
            "tasks_mixed": int(((c > 0) & (c < m)).sum()),
            "E_sigma2": e_sigma2, "Var_x": var_x,
            "within_share_of_total_var": e_sigma2 / total_var_single,
            "pred_run_sd_indep_pp": pred_run_sd, "observed_run_sd_pp": sd1,
            "obs_over_pred_sd": sd1 / pred_run_sd, "chi2_dispersion_p": chi_p,
            "single_run_total_se_pp": se_total,
            "frac_pairs_range_ge2pp": float(d2), "frac_pairs_range_ge3pp": float(d3),
            "mcnemar_reject_rate_same_config": float(mc),
            "mean_pairwise_phi": float(np.mean(cors)),
        }
        # pass@1 -> pass@5 사이 격차, 논문 초록/서론 "24.9 pp"가 무엇의 차이인가
        rec["gap_pass5_minus_pass1"] = rec["pass@5"] - mean
        rec["gap_pass1_minus_passhat5"] = mean - rec["pass^5"]
        rec["gap_pass5_minus_passhat5"] = rec["pass@5"] - rec["pass^5"]
        res["configs"][rec["key"]] = rec
        table1_rows.append(rec)

    # 범위 2.2-6.0 확인, "temp 0에서도 표준편차 1.5%p 초과" 확인
    ranges = [x["range"] for x in table1_rows]
    res["range_min_max"] = [min(ranges), max(ranges)]
    t0 = [x for x in table1_rows if x["key"].endswith("/0.0")]
    res["temp0_sd_ddof1"] = {x["key"]: x["sd_ddof1"] for x in t0}
    res["temp0_n_sd_gt_1p5"] = int(sum(round(x["sd_ddof1"], 1) > 1.5 for x in t0))
    res["max_gap_pass5_minus_pass1"] = max((x["gap_pass5_minus_pass1"], x["key"]) for x in table1_rows)
    res["max_gap_pass5_minus_passhat5"] = max((x["gap_pass5_minus_passhat5"], x["key"]) for x in table1_rows)

    # 온도 효과: 같은 모델·스캐폴드의 기본 온도 대 0, 실행 수준 Welch t, 문항 쌍체(평균 실행) 와 단일 실행 McNemar
    temp_pairs = []
    for sc in ["nano-agent", "r2e-gym"]:
        for mdl, t in [("Qwen3-32B", "0.6"), ("DeepSWE-preview", "1.0"), ("Devstral-2", "0.2")]:
            A, B = mats[(sc, mdl, t)], mats[(sc, mdl, "0.0")]
            ra, rb = A.sum(1) / N * 100, B.sum(1) / N * 100
            w = stats.ttest_ind(ra, rb, equal_var=False)
            diff = ra.mean() - rb.mean()
            # 문항 모집단 쌍체 SE: 문항별 평균 성공률 차 d_i 의 표준오차 (Error Bars 식 7, K=10 재표집)
            d = A.mean(0) - B.mean(0)
            se_pop = float(d.std(ddof=1) / np.sqrt(N) * 100)
            # 실행 잡음만 (문항 고정) 의 SE
            se_run = float(np.sqrt(ra.var(ddof=1) / 10 + rb.var(ddof=1) / 10))
            # 단일 실행 쌍 (j,j) 100쌍 중 McNemar 유의 비율과 부호
            mcs = [mcnemar_p(A[a], B[b]) for a in range(10) for b in range(10)]
            sg = [(A[a].sum() - B[b].sum()) for a in range(10) for b in range(10)]
            temp_pairs.append({
                "pair": f"{sc}/{mdl} {t} vs 0.0", "diff_pp": float(diff),
                "welch_p_run_level": float(w.pvalue), "se_run_noise_only_pp": se_run,
                "se_task_population_paired_pp": se_pop, "z_task_population": float(diff / se_pop),
                "single_run_pairs_mcnemar_sig_frac": float(np.mean(np.array(mcs) < 0.05)),
                "single_run_pairs_sign_matches_mean_frac": float(np.mean(np.sign(sg) == np.sign(diff))),
            })
    res["temperature_effects"] = temp_pairs

    # "temperature 0에서 분산이 줄지 않는다"(§2.4.1): 기본 온도 대 0의 실행 간 분산비 F 검정
    vr = []
    for sc in ["nano-agent", "r2e-gym"]:
        for mdl, t in [("Qwen3-32B", "0.6"), ("DeepSWE-preview", "1.0"), ("Devstral-2", "0.2")]:
            ra = mats[(sc, mdl, t)].sum(1) / N * 100
            rb = mats[(sc, mdl, "0.0")].sum(1) / N * 100
            F = ra.var(ddof=1) / rb.var(ddof=1)
            pF = float(2 * min(stats.f.cdf(F, 9, 9), stats.f.sf(F, 9, 9)))
            vr.append({"pair": f"{sc}/{mdl}", "sd_default": float(ra.std(ddof=1)), "sd_T0": float(rb.std(ddof=1)), "F": float(F), "p": pF})
    res["variance_ratio_default_vs_T0"] = vr

    # 범위의 기대값: 정규 10개 표본의 범위는 약 3.08 sigma. m이 커지면 범위가 커진다.
    exp = {}
    for m in [2, 5, 10, 20, 50, 100]:
        x = rng.standard_normal((200000, m))
        exp[m] = float((x.max(1) - x.min(1)).mean())
    res["expected_range_in_sigma"] = exp
    res["range_implied_by_sigma_0.7_to_1.8_m10"] = [exp[10] * 0.7, exp[10] * 1.8]

    json.dump(res, open(os.path.join(HERE, "recompute_runs_results.json"), "w"), indent=1, default=float)

    # 요약 출력
    print(f"한 번이라도 풀린 문항 수 {n_universe} / {N}")
    print(f"{'config':32s} {'mean':>6s} {'sd1':>5s} {'sd0':>5s} {'min':>5s} {'max':>5s} | paper | SW p (paper)")
    for x in table1_rows:
        print(f"{x['key']:32s} {x['mean']:6.2f} {x['sd_ddof1']:5.2f} {x['sd_ddof0']:5.2f} {x['min']:5.1f} {x['max']:5.1f} | {x['paper']} | {x['shapiro_p']:.3f} ({x['paper_T3'][1]})")
    print("범위 최소/최대", res["range_min_max"], "temp0 sd>1.5 개수", res["temp0_n_sd_gt_1p5"], res["temp0_sd_ddof1"])
    print("\npass@5 / pass^5 (재계산 vs 그림)")
    for x in table1_rows:
        print(f"{x['key']:32s} {x['pass@5']:6.1f} {x['pass^5']:6.1f}  fig={x['paper_fig']}  gap5-1={x['gap_pass5_minus_pass1']:.1f} 1-hat5={x['gap_pass1_minus_passhat5']:.1f}")
    print("max gap", res["max_gap_pass5_minus_pass1"], res["max_gap_pass5_minus_passhat5"])
    print("\n분산 분해")
    for x in table1_rows:
        print(f"{x['key']:32s} 항상{x['tasks_always']:4d} 전혀{x['tasks_never']:4d} 섞임{x['tasks_mixed']:4d} 실행내/전체={x['within_share_of_total_var']:.2f} 예측sd={x['pred_run_sd_indep_pp']:.2f} (p={x['chi2_dispersion_p']:.2f}) 관측sd={x['observed_run_sd_pp']:.2f} 단일실행총SE={x['single_run_total_se_pp']:.2f} 쌍차>=2pp={x['frac_pairs_range_ge2pp']:.2f} McNemar유의={x['mcnemar_reject_rate_same_config']:.2f} phi={x['mean_pairwise_phi']:.2f}")
    print("\n온도 효과")
    for t in temp_pairs:
        print(t)
    print("\n분산비", *[(v["pair"], round(v["sd_default"],2), round(v["sd_T0"],2), round(v["p"],2)) for v in vr], sep="\n")
    print("\n기대 범위(σ 배수)", exp, res["range_implied_by_sigma_0.7_to_1.8_m10"])


if __name__ == "__main__":
    main()
