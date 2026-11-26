"""우리가 그린 그래프 (matplotlib). 입력: derivation_results.json, toy_results.json, gap_results.json.
실행: python3 analysis/plot_figures.py  -> figures/own_*.png"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for name in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False


def load(n):
    return json.loads((HERE / n).read_text())


def fig_derivation():
    d = load("derivation_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for N, col in ((8, "#0072b2"), (32768, "#d55e00")):
        xs = [0.5 + i * 0.25 for i in range(0, 80)]
        ax[0].plot(xs, [math.log1p((N - 1) * math.exp(-s)) for s in xs], color=col, label=f"정렬된 직교 쌍, N={N}")
        ax[0].axhline(math.log(N), color=col, ls=":", lw=1)
    ax[0].axvline(1 / 0.07, color="#555", ls="--", lw=1)
    ax[0].text(1 / 0.07 + 0.3, 6, "초기 배율 1/0.07", fontsize=8)
    ax[0].set_ylim(0, 11)
    ax[0].set_xlabel("로짓 배율 exp(t)")
    ax[0].set_ylabel("손실")
    ax[0].set_title("닫힌 식 log(1+(N-1)e^{-s})와 무작위 수준 log N (점선)")
    ax[0].legend(frameon=False, fontsize=8)
    mi = d["mi_bound_gaussian"]
    for N, col in ((16, "#0072b2"), (128, "#d55e00")):
        rows = [r for r in mi if r["N"] == N]
        ax[1].plot([r["I_true"] for r in rows], [r["bound"] for r in rows], "o-", color=col, label=f"InfoNCE 추정, N={N}")
        ax[1].axhline(math.log(N), color=col, ls=":", lw=1)
    ax[1].plot([0, 8], [0, 8], color="#555", lw=1, ls="--", label="참값 I (y = x)")
    ax[1].set_xlabel("참 상호정보 I(X;Y) (nats)")
    ax[1].set_ylabel("log N - 손실")
    ax[1].set_title("하한은 min(I, log N)을 넘지 않는다 (이변량 정규)")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_derivation.png", dpi=150)
    plt.close(fig)


def fig_toy():
    r = load("toy_results.json")
    order = [c for c in ("smallbatch", "baseline", "hardneg") if c in r]
    lab = {"smallbatch": "배치 32\n표준 손실", "baseline": "배치 128\n표준 손실", "hardneg": "배치 128\n+ 맞바꾼 캡션"}
    cols = {"smallbatch": "#8c8c8c", "baseline": "#0072b2", "hardneg": "#d55e00"}
    metrics = [("zs_held_out_photo", "제로샷, 본 적 없는 조합"), ("bind_attr_acc", "속성 결합 (모양 맞바꿈 대비)"),
               ("bind_order_acc", "왼쪽-오른쪽 순서 (뒤집은 캡션 대비)")]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for a, (k, title) in zip(ax, metrics):
        for i, c in enumerate(order):
            mean, lo, hi = r[c]["mean_ci95"][k]
            a.bar(i, mean, color=cols[c], alpha=0.55, yerr=[[mean - lo], [hi - mean]], capsize=4)
            ys = r[c]["per_seed"][k]
            a.scatter([i + (j - len(ys) / 2) * 0.05 for j in range(len(ys))], ys, color="k", s=10, zorder=3)
        a.set_xticks(range(len(order)))
        a.set_xticklabels([lab[c] for c in order], fontsize=8)
        a.set_title(title, fontsize=9)
        a.grid(axis="y", alpha=0.3)
    ax[0].axhline(1 / 64, color="#555", ls=":", lw=1)
    ax[1].axhline(0.5, color="#555", ls=":", lw=1)
    ax[2].axhline(0.5, color="#555", ls=":", lw=1)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("정확도 (막대: 5시드 평균과 95% 구간, 점: 시드)")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_derivation()
    if (HERE / "toy_results.json").exists():
        fig_toy()
