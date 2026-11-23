"""우리가 그린 그림 네 장 (matplotlib). 입력: agreement_results.json, mtbench_scores_results.json
출력: report/figures/own_*.png   실행: python3 -I analysis/plot_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
A = json.load(open(HERE / "agreement_results.json"))
M = json.load(open(HERE / "mtbench_scores_results.json"))
C = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

# 1. 일치도: GPT-4-사람 대 사람-사람, Wilson(가는 선)과 질문 군집 부트스트랩(굵은 선), 질문 무시 기준선
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.6), constrained_layout=True)
T = A["table5_13_recompute"]
for ax, s, R in zip(axs, ("S1", "S2"), (1 / 3, 0.5)):
    labels, ys, w, cl, cols = [], [], [], [], []
    for t in (1, 2):
        for kind, name, c in (("g4_human", "GPT-4-사람", C[0]), ("human_human", "사람-사람", C[1])):
            r = T[f"turn{t}_{s}_{kind}"]
            labels.append(f"{t}턴\n{name}"); ys.append(r["rate"]); w.append(r["wilson95"]); cl.append(r["cluster_boot95"]); cols.append(c)
    x = np.arange(len(ys))
    for i in x:
        ax.plot([i, i], cl[i], color=cols[i], lw=5, alpha=0.9, solid_capstyle="butt")
        ax.plot([i, i], w[i], color="black", lw=1.2, zorder=4)
        ax.plot([i - 0.06, i + 0.06], [w[i][0]] * 2, color="black", lw=1.2, zorder=4); ax.plot([i - 0.06, i + 0.06], [w[i][1]] * 2, color="black", lw=1.2, zorder=4)
        ax.plot(i, ys[i], "o", color="white", ms=5, zorder=3)
        ax.annotate(f"{100 * ys[i]:.1f}%", (i, ys[i]), textcoords="offset points", xytext=(14, -3), fontsize=8)
    ax.axhline(R, color="gray", ls=":", lw=1) if s == "S1" else None
    for t, xs in ((1, (-0.3, 1.3)), (2, (1.7, 3.3))):
        b = A["chance_corrected"][f"turn{t}_{s}"]["baseline_pair_majority_agree_with_human"]
        ax.plot(xs, [b, b], color=C[3], ls="--", lw=1.3)
    ax.plot([], [], color=C[3], ls="--", lw=1.3, label="질문을 보지 않는 모델쌍 다수결 기준선"); ax.legend(fontsize=7, loc="lower right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim((0.55, 0.75) if s == "S1" else (0.72, 0.92)); ax.set_title(f"{s} ({'동점 포함' if s == 'S1' else '동점 제외'})", fontsize=10)
axs[0].set_ylabel("일치율")
fig.suptitle("굵은 선: 질문 80개 군집 부트스트랩 95%, 가는 선: Wilson 95% (표를 독립으로 본 구간)", fontsize=9)
fig.savefig(FIG / "own_agreement_intervals.png", dpi=150); plt.close(fig)

# 2. 모델 쌍별: 사람 승률 격차 대 GPT-4-사람 일치(S2)와 GPT-4 쌍 비교의 비일관 비율
bp = A["by_model_pair"]; pp = A["gpt4_pair_position"]["per_pair"]
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0))
gap = np.array([r["human_winrate_gap_in_pair"] for r in bp]); ag = np.array([r["agree"] for r in bp])
lo = np.array([r["wilson95"][0] for r in bp]); hi = np.array([r["wilson95"][1] for r in bp])
axs[0].errorbar(gap, ag, yerr=[ag - lo, hi - ag], fmt="o", color=C[0], ecolor=C[0], alpha=0.8, capsize=2)
axs[0].set_xlabel("모델 쌍 안 사람 승률 격차 (비동점 표)"); axs[0].set_ylabel("GPT-4-사람 일치율 (S2, 두 턴 합침)")
axs[0].set_title(f"쌍이 갈릴수록 일치가 높다 (Spearman {A['by_model_pair_spearman_gap_vs_agree']['rho']:.2f})", fontsize=9)
inc = np.array([pp[r["pair"]]["rate"] for r in bp])
axs[1].scatter(gap, inc, color=C[3])
axs[1].set_xlabel("모델 쌍 안 사람 승률 격차"); axs[1].set_ylabel("GPT-4 쌍 비교 비일관 비율(순서를 바꾸면 결과가 달라짐)")
axs[1].set_title(f"쌍이 가까울수록 위치 편향이 드러난다 (Spearman {A['gpt4_pair_position']['spearman_gap_vs_inconsistent']['rho']:.2f})", fontsize=9)
for ax in axs: ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "own_pair_gap.png", dpi=150); plt.close(fig)

# 3. 자기 선호: GPT-4 판정 승률 - 사람 승률 (같은 표 위), 질문 군집 95%
W = A["win_rate_self_preference"]["all"]
names = list(W); d = np.array([W[m]["diff"] for m in names]); lo = np.array([W[m]["diff_cluster_boot95"][0] for m in names]); hi = np.array([W[m]["diff_cluster_boot95"][1] for m in names])
order = np.argsort(-np.array([W[m]["human_winrate"] for m in names]))
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for k, i in enumerate(order):
    col = C[3] if names[i] == "gpt-4" else C[0]
    ax.errorbar(100 * d[i], k, xerr=[[100 * (d[i] - lo[i])], [100 * (hi[i] - d[i])]], fmt="o", color=col, capsize=3)
ax.axvline(0, color="gray", lw=1)
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{names[i]} (사람 승률 {100 * W[names[i]]['human_winrate']:.0f}%)" for i in order], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("GPT-4 판정 승률 - 사람 승률 (%p, 비동점 표, 같은 표 위)")
ax.set_title("GPT-4 판정은 강한 모델 쪽 승률을 올리고 약한 모델 쪽을 내린다(자기 선호와 분리되지 않음)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "own_self_preference.png", dpi=150); plt.close(fig)

# 4. MT-Bench 80문항 점수 순위와 불확실성
R = M["ranking"]; rc = {r["model"]: r for r in M["rank_ci"]}
fig, axs = plt.subplots(1, 2, figsize=(10, 6.6), gridspec_kw={"width_ratios": [1.2, 1]})
y = np.arange(len(R))
axs[0].barh(y, [r["mean"] for r in R], xerr=[[1.96 * r["se_clt_80q"] for r in R]] * 2, color=C[0], alpha=0.85, ecolor="black", capsize=1.5)
axs[0].set_yticks(y); axs[0].set_yticklabels([r["model"] for r in R], fontsize=7); axs[0].invert_yaxis()
axs[0].set_xlabel("MT-Bench 평균 점수 ± 1.96 SE (문항 80개)")
for k, a in enumerate(M["adjacent_ranks"]):
    if not a["sig_p05"]:
        axs[0].plot([10.3], [k + 0.5], marker="_", color=C[3], ms=8)
axs[0].plot([], [], marker="_", color=C[3], ls="", ms=8, label="바로 아래 순위와 짝지은 차이가 p<0.05가 아님"); axs[0].legend(fontsize=7, loc="lower right")
axs[0].set_xlim(0, 11.5)
for k, r in enumerate(R):
    lo_, hi_ = rc[r["model"]]["rank95"]
    axs[1].plot([lo_, hi_], [k, k], color=C[1], lw=4, solid_capstyle="butt")
    axs[1].plot(k + 1, k, "o", color="black", ms=3)
axs[1].set_yticks([]); axs[1].invert_yaxis(); axs[1].set_xlabel("순위 95% 구간 (문항 재표집 4000회)")
axs[1].set_title(f"순위 구간 너비 중앙값 {M['rank_ci_width_median']:.0f}", fontsize=9)
axs[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "own_mtbench_ranking.png", dpi=150); plt.close(fig)
print("saved", [p.name for p in FIG.glob("own_*.png")])
