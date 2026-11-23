"""논문 핵심 주장을 토이 세계(ToyWorld, 26과제 로그정규)에서 직접 확인한다. 약 1-2분(CPU).

실행: python3 -I analysis/toy_experiments.py   (결과 toy_results.json, 그림 figures/own_*.png)

  E0  성능 프로필: 식 (1)의 분산 공식과 평균 점수 분포의 편향
  E1  실행 수 N별 추정량의 평균·표준편차(중앙값 편향, IQM의 효율)           -> 논문 §3, 그림 2·3
  E2  같은 알고리즘 둘 / 10% 좋은 알고리즘의 점추정 순위, 관측 향상률 범위  -> 논문 그림 4
  E3  층화 백분위 부트스트랩 95% 구간의 실제 포함률                        -> 논문 그림 6
  E4  이진 성공률(과제 수 M이 작음)에서의 포함률, 이진 데이터의 IQM        -> 논문 범위 밖, 적용 한계
  E5  개선 확률 P(X>Y)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_impl import (ToyWorld, bootstrap_ci, iqm, mean_of_task_means, median_of_task_means,  # noqa: E402
                          optimality_gap, performance_profile, average_score_profile, profile_ci,
                          prob_improvement, prob_improvement_ci, stratified_resample)
import scipy.stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATS = {"mean": mean_of_task_means, "median": median_of_task_means, "iqm": iqm, "optgap": optimality_gap}
NS = [3, 5, 10, 25, 100]


def r3(x):
    return None if x is None else round(float(x), 4)


def r6(x):
    return round(float(x), 6)


def e0(w):
    rng = np.random.default_rng(10)
    tau, n, reps = 0.3, 5, 40000
    x = w.sample(n, rng, reps=reps)
    per_rep = (x > tau).mean(axis=(1, 2))
    avg_rep = (x.mean(axis=1) > tau).mean(axis=1)
    f_m = 1 - scipy.stats.norm.cdf((np.log(tau) - np.log(w.a) + w.s ** 2 / 2) / w.s)
    f_true = f_m.mean()
    var_formula = (f_m * (1 - f_m)).sum() / (w.m ** 2 * n)
    return {"tau": tau, "N": n, "F_true": r3(f_true),
            "run_score_profile": {"mean_est": r3(per_rep.mean()), "var_emp": r6(per_rep.var()), "var_formula": r6(var_formula)},
            "avg_score_profile": {"mean_est": r3(avg_rep.mean()), "bias": r3(avg_rep.mean() - f_true), "var_emp": r6(avg_rep.var()), "var_ratio_vs_run_score": r3(avg_rep.var() / per_rep.var())},
            "note": "실행 점수 분포는 불편(식 (1)), 분산은 (1/(M^2 N)) sum F_m(1-F_m); 평균 점수 분포는 편향"}


def e1(w, reps=2000):
    rng = np.random.default_rng(1)
    truth = w.truth()
    out = {"truth": {k: r3(v) for k, v in truth.items()}, "by_N": {}}
    for n in NS:
        x = w.sample(n, rng, reps=reps)
        row = {}
        for k, f in STATS.items():
            v = f(x)
            row[k] = {"mean": r3(v.mean()), "bias": r3(v.mean() - truth[k]), "sd": r3(v.std()), "rel_sd": r3(v.std() / truth[k])}
        out["by_N"][n] = row
    return out


def e2(w, reps=4000):
    rng = np.random.default_rng(2)
    out = {}
    for n in NS:
        a = w.sample(n, rng, reps=reps)
        b0 = w.sample(n, rng, reps=reps)
        b10 = w.sample(n, rng, reps=reps, lift=0.10)
        row = {}
        for k in ("median", "iqm", "mean"):
            f = STATS[k]
            lift0 = f(b0) / f(a) - 1
            lift10 = f(b10) / f(a) - 1
            row[k] = {"lift0_ci95": [r3(np.percentile(lift0, 2.5)), r3(np.percentile(lift0, 97.5))],
                      "P(|lift|>10%|same algo)": r3((np.abs(lift0) > 0.10).mean()),
                      "P(point est ranks 10%-better algo first)": r3((f(b10) > f(a)).mean()),
                      "lift10_ci95": [r3(np.percentile(lift10, 2.5)), r3(np.percentile(lift10, 97.5))]}
        out[n] = row
    return out


def e3(w, reps_exp=300, boot=1000, ns=(3, 5, 10, 25)):
    rng = np.random.default_rng(3)
    truth = w.truth()
    out = {}
    for n in ns:
        hit = {k: 0 for k in STATS}
        width = {k: [] for k in STATS}
        for _ in range(reps_exp):
            x = w.sample(n, rng)
            bs = stratified_resample(x, boot, rng)
            for k, f in STATS.items():
                lo, hi = np.percentile(f(bs), [2.5, 97.5])
                hit[k] += lo <= truth[k] <= hi
                width[k].append(hi - lo)
        out[n] = {k: {"coverage": r3(hit[k] / reps_exp), "mean_width": r3(np.mean(width[k])),
                      "rel_width": r3(np.mean(width[k]) / truth[k])} for k in STATS}
    return {"experiments": reps_exp, "bootstrap_reps": boot, "se_of_coverage_at_0.95": r3(np.sqrt(0.95 * 0.05 / reps_exp)), "by_N": out}


def e4(reps_exp=1000, boot=1000):
    """이진 성공률. 과제(시나리오) 수 M과 과제당 반복 N이 작을 때 층화 백분위 구간의 포함률."""
    rng = np.random.default_rng(4)
    cov = {}
    for m in (3, 5, 8, 10, 20):
        p = np.random.default_rng(100 + m).beta(2, 2, m)
        truth = p.mean()
        for n in (3, 5, 10, 20):
            hit, w_ = 0, []
            for _ in range(reps_exp):
                x = (rng.random((n, m)) < p).astype(float)
                lo, hi = np.percentile(mean_of_task_means(stratified_resample(x, boot, rng)), [2.5, 97.5])
                hit += lo <= truth <= hi
                w_.append(hi - lo)
            cov[f"M={m},N={n}"] = {"coverage": r3(hit / reps_exp), "mean_width": r3(np.mean(w_)), "true": r3(truth)}
    rates = [0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.75, 0.8, 0.9]
    iqm_binary = {}
    for r in rates:
        n = 4000
        x = np.zeros((n, 1)); x[: int(r * n)] = 1
        iqm_binary[str(r)] = r3(iqm(x))
    return {"coverage_mean_success": cov, "iqm_of_binary_by_success_rate": iqm_binary,
            "note": "이진 점수의 IQM은 성공률 r에서 clip((r-0.25)/0.5, 0, 1)"}


def e5(w, reps=2000):
    rng = np.random.default_rng(5)
    out = {}
    for n in (3, 5, 10, 25):
        same = [prob_improvement(w.sample(n, rng), w.sample(n, rng)) for _ in range(reps)]
        lift = [prob_improvement(w.sample(n, rng, lift=0.10), w.sample(n, rng)) for _ in range(reps)]
        excl = 0
        for _ in range(300):
            p, (lo, hi) = prob_improvement_ci(w.sample(n, rng, lift=0.10), w.sample(n, rng), reps=500, rng=rng)
            excl += lo > 0.5
        out[n] = {"same_algo_mean": r3(np.mean(same)), "same_algo_sd": r3(np.std(same)),
                  "lift10_mean": r3(np.mean(lift)), "lift10_sd": r3(np.std(lift)),
                  "lift10_P(CI lower > 0.5)": r3(excl / 300)}
    big = prob_improvement(w.sample(20000, np.random.default_rng(9), lift=0.10), w.sample(20000, np.random.default_rng(10)))
    analytic = float(scipy.stats.norm.cdf(np.log(1.10) / (w.s * np.sqrt(2))).mean())  # ln X - ln Y = ln(1+l) + s(Z1-Z2)
    return {"by_N": out, "lift10_large_N": r3(big), "lift10_analytic": r3(analytic)}


def narrowness(w):
    """플러그인 부트스트랩 분산은 불편 분산의 (N-1)/N 배 -> 구간 폭이 sqrt((N-1)/N)배(정규 근사).
    명목 95%가 실제로는 2*Phi(1.96*sqrt((N-1)/N)) - 1 이 된다. 과제 평균들의 평균(mean) 구간의 포함률 기준 근사."""
    pop = w.sample(20000, np.random.default_rng(321))
    return {"normal_approx_coverage": {str(n): r3(2 * scipy.stats.norm.cdf(1.96 * np.sqrt((n - 1) / n)) - 1) for n in (3, 5, 10, 20, 25)},
            "pooled_median_vs_median_of_task_means": {"pooled_median_of_all_scores": r3(np.median(pop)),
                                                      "median_of_E[X_m]": r3(np.median(w.a))}}


def plots(w, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    col = {"mean": "#8a8a8a", "median": "#d95f02", "iqm": "#1b6ca8", "optgap": "#7570b3"}
    nm = {"mean": "평균", "median": "중앙값", "iqm": "IQM", "optgap": "최적성 격차"}

    # 그림 1: 추정량 변동과 점추정 순위
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for k in STATS:
        ax[0].plot(NS, [res["E1"]["by_N"][n][k]["rel_sd"] for n in NS], "o-", c=col[k], label=nm[k])
    ax[0].set_xscale("log"); ax[0].set_xlabel("과제당 실행 수 N"); ax[0].set_ylabel("표준편차 / 참값"); ax[0].set_title("추정량의 상대 변동")
    ax[0].legend(frameon=False)
    for k in ("median", "iqm", "mean"):
        ax[1].plot(NS, [res["E1"]["by_N"][n][k]["bias"] / res["E1"]["truth"][k] for n in NS], "o-", c=col[k], label=nm[k])
    ax[1].axhline(0, c="k", lw=.5); ax[1].set_xscale("log"); ax[1].set_xlabel("N"); ax[1].set_title("상대 편향 (추정 평균 / 참값 - 1)")
    for k in ("median", "iqm", "mean"):
        ax[2].plot(NS, [res["E2"][n][k]["P(point est ranks 10%-better algo first)"] for n in NS], "o-", c=col[k], label=nm[k])
    ax[2].axhline(0.5, c="k", lw=.5, ls=":"); ax[2].set_xscale("log"); ax[2].set_ylim(0.45, 1.02)
    ax[2].set_xlabel("N"); ax[2].set_title("진짜 10% 좋은 알고리즘이 점추정에서 앞설 확률")
    fig.tight_layout(); fig.savefig(fig_dir / "own_variability.png", dpi=150); plt.close(fig)

    # 그림 2: 포함률
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ns3 = sorted(res["E3"]["by_N"], key=int)
    for k in STATS:
        ax[0].plot([int(n) for n in ns3], [res["E3"]["by_N"][n][k]["coverage"] for n in ns3], "o-", c=col[k], label=nm[k])
    ax[0].axhline(0.95, c="k", lw=.6, ls="--"); ax[0].set_xscale("log"); ax[0].set_xticks((3, 5, 10, 25)); ax[0].set_xticklabels(("3", "5", "10", "25")); ax[0].set_ylim(0.55, 1.0)
    ax[0].set_xlabel("N (과제 26개)"); ax[0].set_ylabel("95% 구간의 실제 포함률"); ax[0].set_title("연속 점수: 층화 백분위 구간"); ax[0].legend(frameon=False)
    for m, c in zip((3, 5, 8, 10, 20), ("#e7298a", "#d95f02", "#7570b3", "#1b6ca8", "#1b9e77")):
        ax[1].plot((3, 5, 10, 20), [res["E4"]["coverage_mean_success"][f"M={m},N={n}"]["coverage"] for n in (3, 5, 10, 20)], "o-", c=c, label=f"과제 M={m}")
    ax[1].axhline(0.95, c="k", lw=.6, ls="--"); ax[1].set_xscale("log"); ax[1].set_xticks((3, 5, 10, 20)); ax[1].set_xticklabels(("3", "5", "10", "20")); ax[1].set_ylim(0.6, 1.0)
    ax[1].set_xlabel("과제당 반복 N"); ax[1].set_title("이진 성공률(평균): 과제 수가 적을 때"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(fig_dir / "own_coverage.png", dpi=150); plt.close(fig)

    # 그림 3: 성능 프로필 + 밴드 (A와 10% 좋은 B, N=5)
    rng = np.random.default_rng(7)
    a, b = w.sample(5, rng), w.sample(5, rng, lift=0.10)
    taus = np.linspace(0, 2.0, 120)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for x, c, lab in ((a, "#8a8a8a", "A"), (b, "#1b6ca8", "B (참값 +10%)")):
        p, (lo, hi) = profile_ci(x, taus, rng=rng)
        ax[0].plot(taus, p, c=c, label=lab); ax[0].fill_between(taus, lo, hi, color=c, alpha=.25)
        pa = average_score_profile(x, taus)
        ax[1].step(taus, pa, c=c, label=lab)
    ax[0].set_title("실행 점수 분포 + 95% 층화 밴드 (N=5)"); ax[0].set_xlabel("점수 τ"); ax[0].set_ylabel("P(X > τ)"); ax[0].legend(frameon=False)
    ax[1].set_title("평균 점수 분포 (같은 데이터, 편향·계단 큼)"); ax[1].set_xlabel("점수 τ")
    fig.tight_layout(); fig.savefig(fig_dir / "own_profile.png", dpi=150); plt.close(fig)


def main():
    w = ToyWorld()
    res = {"world": {"M": w.m, "a_sorted": [r3(v) for v in np.sort(w.a)], "note": "X = a_m (1+lift) exp(s_m Z - s_m^2/2), 로그정규"}}
    res["E0"] = e0(w); print("E0 done")
    res["E1"] = e1(w); print("E1 done")
    res["E2"] = e2(w); print("E2 done")
    res["E3"] = e3(w); print("E3 done")
    res["E4"] = e4(); print("E4 done")
    res["E5"] = e5(w); print("E5 done")
    res["narrowness"] = narrowness(w)
    (Path(__file__).resolve().parent / "toy_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    plots(w, res)
    print(json.dumps({k: res[k] for k in ("E0",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
