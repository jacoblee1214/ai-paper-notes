"""논문 수치 재계산: 부록 C 예, 표 2·9의 MDE, 표 10(GLUE 주장 검정력), 그림 10(SQuAD 2.0 리더보드 근사),
번역(그림 4), Likert(그림 6)을 minimal_impl.py로 다시 만들어 논문 값·저자 공개 결과 파일과 비교한다.

입력: ../repo/data/ (저자 저장소 dallascard/NLP-power-analysis, 정적으로 읽기만 함. repo/는 gitignore).
출력: recompute_results.json
실행: python3 analysis/recompute.py   (numpy, scipy 필요, 약 1분)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimal_impl as mi  # noqa: E402

DATA = HERE.parent / "repo" / "data"
OUT: dict = {}

# 저자가 GLUE 시험 예측 10개 모델로 적합한 겹침 회귀 (논문 표 5, 저자 노트북에 하드코딩된 값)
PA_COEF = (0.41422307, 0.58187475, -0.46624237)  # 절편, 기준 정확도, 효과 크기


SQUAD_PA_COEF = (0.4339, 0.5932, -1.2849)  # 논문 표 6 (SQuAD 2.0 리더보드, 모델 14개)


def pa_regression(base_acc: float, delta: float, coef=PA_COEF) -> float:
    """겹침(두 모델이 같은 결과를 내는 비율) 회귀. 저자 노트북의 non.overlapping = 1 - (절편 + 기준정확도 + 효과)."""
    non_overlap = 1.0 - (coef[0] + coef[1] * base_acc + coef[2] * delta)
    return 1.0 - non_overlap


# ---------------------------------------------------------------- A. 부록 C 예
def appendix_c():
    res = {}
    for method in ("chi2", "chi2_cc", "exact"):
        for n in (500, 2000):
            s = mi.simulate_power(n, 0.02, 0.9, r=200000, method=method, seed=1)
            res[f"{method}_n{n}"] = {k: round(v, 4) for k, v in s.items()}
    res["analytic_n500"] = round(mi.analytic_power(500, 0.02, 0.9), 4)
    res["analytic_n2000"] = round(mi.analytic_power(2000, 0.02, 0.9), 4)
    res["paper"] = {"n500": {"power": 0.25, "type_m": 1.9}, "n2000": {"power": "nearly 0.80", "type_m": 1.1}}
    OUT["appendix_c"] = res


# ---------------------------------------------------------------- B. 표 2·9 MDE
TABLE9 = {  # 데이터셋: (n, SOTA, MDE binomial, Lach (min, mid, max), MDE regression)  논문 표 9, 단위 %p
    "WNLI": (147, 0.945, 5.38, (5.36, 5.42, 5.45), 5.26),
    "MRPC": (1725, 0.920, 2.40, (0.45, 1.91, 2.48), 1.62),
    "SST-2": (1821, 0.972, 1.34, (0.43, 1.10, 1.35), 1.02),
    "RTE": (3000, 0.917, 1.89, (0.26, 1.48, 1.96), 1.23),
    "QNLI": (5463, 0.975, 0.77, (0.14, 0.60, 0.78), 0.55),
    "MNLI-m": (9796, 0.916, 1.08, (0.08, 0.82, 1.12), 0.67),
    "MNLI-mm": (9847, 0.913, 1.09, (0.08, 0.84, 1.14), 0.68),
    "QQP": (390965, 0.910, 0.18, (0.0000845, 0.13, 0.19), 0.11),
    "SQuAD 2.0": (8862, 0.90724, 1.18, (0.09, 0.91, 1.23), 0.556),
}
TABLE2_AVG = {"WNLI": 1.72, "MRPC": 0.63, "SST-2": 0.57, "RTE": 3.89, "QNLI": 1.31, "MNLI-m": 0.97,
              "MNLI-mm": 1.29, "QQP": 0.36, "SQuAD 2.0": 2.23}


def mde_bounds(n, acc1, alpha=0.05, power=0.8):
    """주변 합(정확도)만 알 때: 불일치율이 가능한 최솟값/최댓값이면 MDE가 얼마인가. 모델 2의 정확도는 acc1 + delta."""
    def mde_at(which):
        def f(dl):
            dmin, dmax = mi.feasible_disagreement(acc1, acc1 + dl)
            d = dmin if which == "min" else dmax
            return mi.analytic_power(n, dl, 1.0 - d, alpha) - power
        try:
            return brentq(f, 1e-7, 1.0 - acc1 - 1e-7) if f(1.0 - acc1 - 1e-7) > 0 else float("nan")
        except ValueError:
            return float("nan")
    return mde_at("min"), mde_at("max")


def table2_9():
    rows = []
    for name, (n, sota, bino, lach, reg) in TABLE9.items():
        coef = SQUAD_PA_COEF if name == "SQuAD 2.0" else PA_COEF
        pa_fn = lambda a, dl, coef=coef: pa_regression(a, dl, coef)
        ours_reg = mi.analytic_mde_from_accuracies(n, sota, pa_of_delta=pa_fn) * 100
        ours_bin = mi.mde_binomial_unpaired(n, sota) * 100
        lo, hi = mde_bounds(n, sota)
        dl = ours_reg / 100
        rows.append({
            "dataset": name, "n": n, "sota": sota * 100,
            "mde_regression": {"ours": round(ours_reg, 3), "paper": reg},
            "mde_binomial_unpaired": {"ours": round(ours_bin, 3), "paper": bino},
            "mde_marginal_bounds_ours": [round(lo * 100, 3), round(hi * 100, 3)], "lachenbruch_paper_min_mid_max": lach,
            "implied_pa_at_mde": round(pa_regression(sota, dl, coef), 4) if dl == dl else None,
            "avg_reported_improvement": TABLE2_AVG[name],
            "avg_improvement_below_mde": bool(TABLE2_AVG[name] < (ours_reg if ours_reg == ours_reg else reg)),
        })
    OUT["table2_table9_mde"] = rows
    # WNLI(n=147): 불일치 쌍이 한 자릿수라 정규 근사가 맞지 않는 영역. 논문 MDE 5.26%p에서의 검정력을 방법별로 비교
    pa = pa_regression(0.945, 0.0526)
    OUT["wnli_small_count"] = {
        "delta": 0.0526, "pa": round(pa, 4), "expected_discordant_pairs": round(147 * (1 - pa), 1),
        "analytic_normal": round(mi.analytic_power(147, 0.0526, pa), 3),
        **{f"sim_{m}": round(mi.simulate_power(147, 0.0526, pa, r=100000, method=m, seed=2)["power"], 3) for m in ("chi2", "chi2_cc", "exact")}}


# ---------------------------------------------------------------- C. 표 10: GLUE 주장 검정력
def table10():
    with open(DATA / "GLUE" / "Glue-Paper-Annotations.csv") as f:
        rows = list(csv.reader(f))[1:]
    sizes = {"SST-2": 1821, "MRPC": 1725, "QNLI": 5463, "RTE": 3000, "MNLI-m": 9796, "MNLI-mm": 9847, "QQP": 390965,
             "WNLI": 147}
    tasks = sorted(sizes)
    recs = [dict(task=r[0], prev=float(r[2]), new=float(r[3]), sota=r[4] == "Y", claim=r[5] == "Y") for r in rows]
    fit_pool = [i for i, r in enumerate(recs) if r["claim"] and r["sota"]]  # 회귀에 쓰는 행(주장 + SOTA 비교)

    def design(rs):
        X = np.array([[1.0, r["prev"]] + [1.0 if r["task"] == t else 0.0 for t in tasks[1:]] for r in rs])
        y = np.array([r["new"] - r["prev"] for r in rs])
        return X, y

    def predict(train_idx, r):
        X, y = design([recs[i] for i in train_idx])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        x = np.array([1.0, r["prev"]] + [1.0 if r["task"] == t else 0.0 for t in tasks[1:]])
        return float(x @ beta) / 100.0

    targets = [i for i in fit_pool if recs[i]["new"] - recs[i]["prev"] > 0]  # 양의 주장만

    def run(exclude_rule):
        n_pow = n_sig = n_both = 0
        used = 0
        pws, pvs = [], []
        for k, i in enumerate(targets):
            r = recs[i]
            train = [j for j in fit_pool if j != exclude_rule(i, k)]
            eff = predict(train, r)
            if eff <= 0:
                continue  # 저자 코드는 이 행도 NA로 처리해 na.omit에서 빠진다
            used += 1
            n, base = sizes[r["task"]], r["prev"] / 100.0
            power = mi.analytic_power(n, eff, pa_regression(base, eff))
            tru = (r["new"] - r["prev"]) / 100.0
            non = 1.0 - pa_regression(base, tru)
            paid = (non - tru) / 2
            p21 = (non + tru) / 2
            p11 = base - p21
            p00 = 1.0 - (p11 + p21 + paid)
            b, c = int(paid * n), int(p21 * n)  # 저자 코드의 as.integer: 버림
            p = float(mi.mcnemar_pvalue(np.array(b), np.array(c), "chi2_cc"))  # R mcnemar.test 기본값(연속성 보정)
            pws.append(power)
            pvs.append(p)
            n_pow += power >= 0.8
            n_sig += p < 0.05
            n_both += (p < 0.05) and (power >= 0.8)
        return {"n_claims": used, "pct_powered": round(100 * n_pow / used, 1), "pct_significant": round(100 * n_sig / used, 1),
                "pct_both": round(100 * n_both / used, 1), "n_powered": int(n_pow), "n_sig": int(n_sig), "n_both": int(n_both),
                "pct_power_below_0.8": round(100 * (1 - n_pow / used), 1),
                "power_mean_sd_q25_q75": [round(float(np.mean(pws)), 3), round(float(np.std(pws, ddof=1)), 3),
                                          round(float(np.percentile(pws, 25)), 3), round(float(np.percentile(pws, 75)), 3)],
                "p_mean": round(float(np.mean(pvs)), 3)}

    # 올바른 leave-one-out: 대상 주장 i를 학습에서 뺀다
    proper = run(lambda i, k: i)
    # 저자 노트북의 인덱싱: newdata[-i,] 의 i가 '주장만 거른 표의 행 번호(1부터)'인데 전체 표 newdata에 적용된다.
    # 전체 표에서 i번째(1부터) 행을 뺀 것을 흉내 낸다(그 행이 학습 풀에 없으면 아무것도 빠지지 않음).
    pool_pos = {i: k + 1 for k, i in enumerate(fit_pool)}  # 거른 표에서의 행 번호
    as_coded = run(lambda i, k: pool_pos[i] - 1)  # 전체 표의 (번호)번째 행(0부터 -1)
    # 그림 14: 60개 주장 중 '관측한 개선폭이 MDE 이상'인 비율 (이항 비짝 검정, Lachenbruch 상·하한 = 불일치율의 양 끝)
    fig14 = {"binomial": 0, "marginal_max_disagreement": 0, "marginal_min_disagreement": 0}
    for i in targets:
        r = recs[i]
        n, base, eff = sizes[r["task"]], r["prev"] / 100.0, (r["new"] - r["prev"]) / 100.0
        p1, p2 = base, base + eff
        pbar = (p1 + p2) / 2
        za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
        need = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / eff ** 2
        fig14["binomial"] += n >= need
        dmin, dmax = mi.feasible_disagreement(p1, p2)
        fig14["marginal_max_disagreement"] += mi.analytic_power(n, eff, 1 - dmax) >= 0.8
        fig14["marginal_min_disagreement"] += mi.analytic_power(n, eff, 1 - dmin) >= 0.8
    fig14 = {k: [int(v), round(100 * v / len(targets), 1)] for k, v in fig14.items()}
    fig14["paper_pct"] = {"binomial": 26.7, "lachenbruch_max": 26.7, "lachenbruch_min_most_generous": 78.3, "lachenbruch_mid": 30.0}
    OUT["table10_glue"] = {"fig14_share_observed_effect_meets_mde": fig14, 
        "n_rows": len(recs), "n_claim_and_sota": len(fit_pool), "n_positive_claims": len(targets),
        "paper": {"n": 57, "pct_powered": 45.6, "pct_significant": 50.9, "pct_both": 36.8,
                  "stargazer_power_mean_sd_q25_q75": [0.698, 0.352, 0.407, 1.0], "stargazer_p_mean": 0.22},
        "ours_proper_loo": proper, "ours_as_coded_index": as_coded,
    }


# ---------------------------------------------------------------- D. 그림 10: SQuAD 2.0 리더보드 근사
def squad_leaderboard():
    with open(DATA / "SQuAD2" / "models.tsv") as f:
        models = list(csv.DictReader(f, delimiter="\t"))
    with open(DATA / "SQuAD2" / "pairs_dev.tsv") as f:
        pairs = {(int(r["i"]), int(r["j"])): (int(r["dev_delta"]), int(r["dev_disagree"])) for r in csv.DictReader(f, delimiter="\t")}
    # 개발 세트 문항 수: 쌍 (0,1)에서 dev_delta / (EM 차) 로 복원
    dev_n = round(pairs[(0, 1)][0] / ((float(models[1]["dev_em"]) - float(models[0]["dev_em"])) / 100))
    test_n = 8862
    em = [(float(m["dev_em"]), float(m["test_em"])) for m in models]
    # 쌍 (i, j) 에서 dev_delta 의 부호를 확인: 모델 1 쪽이 더 맞으면 양수라고 가정하고 EM 차와 맞춰 본다
    sign_ok = np.mean([np.sign(pairs[(i, j)][0]) == np.sign(em[j][0] - em[i][0]) for (i, j) in pairs if em[j][0] != em[i][0]])

    # 개발 대 시험 정확도 차이의 상관 (논문 그림 9: r = 0.99). 모든 모델 쌍에 대해 EM 차이로 계산
    ii, jj = zip(*pairs)
    dd = np.array([em[j][0] - em[i][0] for i, j in zip(ii, jj)])
    tt = np.array([em[j][1] - em[i][1] for i, j in zip(ii, jj)])
    corr_delta = float(np.corrcoef(dd, tt)[0, 1])
    sota = 0
    cats = {"powered_sig_better": 0, "powered_not_sig": 0, "other_sig_better": 0, "other_not_sig": 0, "sig_worse": 0}
    n_cmp = 0
    for j in range(1, len(models)):
        key = (min(sota, j), max(sota, j))
        dd, dis = pairs[key]
        sgn = 1 if j > sota else -1
        dev_delta = sgn * dd / dev_n  # 모델 j - 현재 SOTA (개발 세트)
        pa = 1.0 - dis / dev_n
        test_delta = (em[j][1] - em[sota][1]) / 100.0
        n_cmp += 1
        powered = dev_delta > 0 and mi.analytic_power(test_n, dev_delta, pa) >= 0.8
        # 시험 세트 유의성 근사: 시험 EM 차와 개발 세트 불일치율로 불일치 쌍 수를 만들어 정확 McNemar
        tot = dis / dev_n * test_n
        c = int(round((tot + test_delta * test_n) / 2))
        b = int(round((tot - test_delta * test_n) / 2))
        p = float(mi.mcnemar_pvalue(np.array(b), np.array(c), "exact")) if b >= 0 and c >= 0 else 1.0
        sig = p < 0.05
        if sig and test_delta < 0:
            cats["sig_worse"] += 1
        elif sig:
            cats["powered_sig_better" if powered else "other_sig_better"] += 1
            sota = j  # 유의하게 나아진 제출이 새 SOTA
        else:
            cats["powered_not_sig" if powered else "other_not_sig"] += 1
    OUT["squad_fig10_proxy"] = {
        "dev_n": dev_n, "test_n": test_n, "sign_convention_agreement": float(sign_ok), "dev_test_em_diff_corr_all_pairs": round(corr_delta, 4), "paper_corr": 0.99, "n_comparisons": n_cmp,
        "ours": cats,
        "paper_fig10": {"powered_sig_better": 14, "powered_not_sig": 5, "other_sig_better": 3, "other_not_sig": 16, "sig_worse": 105},
        "note": "시험 세트 쌍별 예측은 비공개라 시험 EM 차와 개발 불일치율로 근사. SOTA 갱신 규칙도 근사.",
    }


# ---------------------------------------------------------------- E. 번역 (그림 4, 표 3)
def mt():
    res = {}
    p0, b0 = 0.13, 25.8
    for n in (500, 1000, 2000, 5000):
        res[f"n{n}_dB1"] = {"sim": round(mi.simulate_bleu_power(n, 1.0, p0, b0, n_trials=400, seed=n), 3),
                            "normal": round(mi.analytic_bleu_power(n, 1.0, p0, b0), 3)}
    res["n2000_dB0.5"] = {"sim": round(mi.simulate_bleu_power(2000, 0.5, p0, b0, n_trials=400, seed=7), 3),
                          "normal": round(mi.analytic_bleu_power(2000, 0.5, p0, b0), 3)}
    res["paper"] = {"n2000_dB1": "approximately 75%", "p0_b0_avg_of_table3": [np.mean([0.19, 0.09, 0.12, 0.10]), np.mean([23.7, 29.4, 22.5, 27.6])]}
    OUT["mt_bleu"] = res


# ---------------------------------------------------------------- F. Likert (그림 6)와 저자 공개 결과(lme4) 비교
def likert():
    path = DATA / "human_eval" / "highvar_lowmer_lmer.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        var = mi.LIKERT_HIGH_VAR if r["settings"].startswith("High") else mi.LIKERT_LOW_VAR
        beta = float(r["diff"])
        W, I = int(float(r["num.workers"])), int(float(r["ex"]))
        out.append({"setting": r["settings"], "beta1": beta, "items": I, "workers": W, "lme4_power": float(r["power"]),
                    "ours_sim_delta_2beta": mi.simulate_likert_power(W, I, 2 * beta, **var, r=1500, seed=3),
                    "ours_analytic_delta_2beta": mi.analytic_likert_power(W, I, 2 * beta, **var),
                    "ours_analytic_delta_beta": mi.analytic_likert_power(W, I, beta, **var)})
    err = lambda k: float(np.sqrt(np.mean([(o[k] - o["lme4_power"]) ** 2 for o in out])))
    OUT["likert"] = {
        "rows": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()} for o in out],
        "rmse_vs_lme4": {"sim_delta=2*beta1": round(err("ours_sim_delta_2beta"), 3),
                         "analytic_delta=2*beta1": round(err("ours_analytic_delta_2beta"), 3),
                         "analytic_delta=beta1": round(err("ours_analytic_delta_beta"), 3)},
        "type1_one_sided_x2_ours": {f"W{W}_I100": round(2 * mi.simulate_likert_power(W, 100, 0.0, **mi.LIKERT_HIGH_VAR, r=6000, seed=5), 3)
                                    for W in (3, 10, 30)},
    }
    # 평가자 수를 늘려도 문항 항이 남는다: 분산 하한
    v = mi.LIKERT_HIGH_VAR
    OUT["likert"]["se_floor_items_term_I100"] = round(v["sd_i1"] / np.sqrt(100), 4)


# ---------------------------------------------------------------- F2. EMNLP 2019 사람 평가 메타분석 (§5.1, 부록 H.1)
def meta_analysis():
    with open(DATA / "human_eval" / "num_items_vs_effect_size.csv") as f:
        rows = list(csv.DictReader(f))
    ex = np.array([float(r["Examples"]) if r["Examples"] not in ("", "NA") else np.nan for r in rows])
    diff = np.array([float(r["diff"]) for r in rows])
    ok = ~np.isnan(ex)
    keep = ok & ~((ex == 500) & (diff < -0.3))  # 논문 각주 11: N=500의 큰 음의 효과 둘 제외
    tau, p = stats.kendalltau(ex[keep], diff[keep])
    ann = [r["Ann.Per.Ex"] for r in rows if r["Ann.Per.Ex"] != "NA"]
    with open(DATA / "human_eval" / "humaneval_master.csv") as f:
        master = list(csv.DictReader(f))
    rel = [x for x in master if x["Probably not relevant to us"] == "0"]
    OUT["meta_analysis_emnlp2019"] = {
        "n_comparisons": len(rows), "n_papers": len({r["Link"] for r in rows}), "paper": {"comparisons": 117, "papers": 41},
        "n_items_known": int(ok.sum()), "pct_items_le_100": round(100 * float((ex[ok] <= 100).mean()), 1), "paper_pct_le_100": 69,
        "pct_items_gt_200": round(100 * float((ex[ok] > 200).mean()), 1), "paper_pct_gt_200": 18,
        "effect_sd": round(float(diff.std(ddof=1)), 3), "paper_effect_sd": 0.12,
        "kendall_tau_effect_vs_items": [round(float(tau), 3), round(float(p), 3)], "paper_kendall": [-0.07, 0.32],
        "pct_3_annotations_of_known": round(100 * sum(a == "3" for a in ann) / len(ann), 1), "paper_pct_3_annotations": 57,
        "pct_with_significance_test_of_relevant": round(100 * sum(x["HasTest"] == "1" for x in rel) / len(rel), 1), "paper_pct_test": 24,
        "pct_bold_of_relevant": round(100 * sum(x["Bold text in results table"] == "yes" for x in rel) / len(rel), 1), "paper_pct_bold": 63,
        "n_relevant_experiments": len(rel),
    }


# ---------------------------------------------------------------- G. 표준 테스트 크기의 MDE와 불일치율 의존성
def mde_grid():
    ns = [100, 200, 500, 1000, 2000, 5000, 10000]
    ds = [0.02, 0.05, 0.10, 0.20, 0.30]
    grid = {"n": ns, "d": ds, "mde_pp": [[round(100 * mi.analytic_mde(n, 1 - d), 2) for d in ds] for n in ns]}
    # 닫힌 형태가 수치 해와 같은지, 근사 2.8*sqrt(d/n)의 오차
    chk = []
    for n, d in [(100, 0.2), (500, 0.1), (2000, 0.1), (1725, 0.058)]:
        f = lambda dl: mi.analytic_power(n, dl, 1 - d) - 0.8
        root = float(brentq(f, 1e-7, d - 1e-9))
        chk.append({"n": n, "d": d, "closed_form": round(mi.analytic_mde(n, 1 - d), 6), "numeric_root": round(root, 6),
                    "approx_2.8sqrt(d/n)": round(2.8016 * np.sqrt(d / n), 6)})
    grid["closed_form_check"] = chk
    # 시뮬레이션 MDE 대 해석 MDE (논문 방식의 시뮬레이션으로 확인)
    sim = []
    for n, d in [(500, 0.1), (1000, 0.1), (2000, 0.1), (500, 0.3)]:
        sim.append({"n": n, "d": d, "analytic": round(mi.analytic_mde(n, 1 - d), 4),
                    "sim_chi2": round(mi.simulate_mde(n, 1 - d, method="chi2", r=20000, seed=11), 4),
                    "sim_exact": round(mi.simulate_mde(n, 1 - d, method="exact", r=20000, seed=11), 4)})
    grid["sim_vs_analytic"] = sim
    OUT["mde_grid"] = grid


# ---------------------------------------------------------------- H. 주변 합만으로 가능한 McNemar 범위 (15·16·18번 분석과 연결)
def marginal_bounds_link():
    res = []
    for label, n, a, c in [("SWE-agent 데모 있음 54 대 33 (swe_agent/03)", 300, 33, 54)]:
        pmin_b = 0                       # 불일치가 가장 적은 배치
        pmax_b = min(a, n - c)           # 가장 많은 배치
        p_at = lambda b: float(mi.mcnemar_pvalue(np.array(b), np.array(b + (c - a)), "exact"))
        res.append({"case": label, "p_min_arrangement": p_at(pmin_b), "p_max_arrangement": p_at(pmax_b),
                    "d_range": [round(x, 4) for x in mi.feasible_disagreement(a / n, c / n)],
                    "analytic_power_at_observed_delta_dmin_dmax": [round(mi.analytic_power(n, (c - a) / n, 1 - d), 3)
                                                                     for d in mi.feasible_disagreement(a / n, c / n)]})
    OUT["marginal_bounds_link"] = res


def main():
    for fn in (appendix_c, table2_9, table10, squad_leaderboard, mt, likert, meta_analysis, mde_grid, marginal_bounds_link):
        fn()
        print("done", fn.__name__, flush=True)
    path = HERE / "recompute_results.json"
    path.write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
