"""그림 두 장(우리가 그린 그래프). 입력: bench_results.json, toy_results.json. 출력: figures/own_bench.png, figures/own_toy.png.
실행: python3 llava/analysis/plot_figures.py  (저장소 루트에서)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

for name in ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic", "AppleGothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"blue": "#2b6cb0", "orange": "#dd6b20", "gray": "#718096", "green": "#2f855a", "red": "#c53030"}


def fig_bench():
    b = json.load(open(HERE / "bench_results.json"))
    grid = {(r["setting"], r["sd_cand"], r["corr"]): r for r in b["uncertainty_grid"]}
    rows = [("ITW 전체 (n=60)", 67.3, "ITW all (n=60)"), ("ITW 대화 (n=17)", 57.3, "ITW conv (n=17)"),
            ("ITW 상세 (n=15)", 52.5, "ITW detail (n=15)"), ("ITW 추론 (n=28)", 81.7, "ITW complex (n=28)"),
            ("COCO 전체 (n=90)", 85.1, "COCO all (n=90)"), ("COCO 범주별 (n=30)", 83.1, "COCO per category (n=30)")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(rows))[::-1]
    for yi, (label, pt, key) in zip(y, rows):
        lo = grid[(key, 1.5, 0.5)]["ci95_half_width"]
        hi = grid[(key, 2.5, 0.0)]["ci95_half_width"]
        mid = grid[(key, 2.0, 0.3)]["ci95_half_width"]
        ax.plot([pt - hi, pt + hi], [yi, yi], color=C["blue"], alpha=0.30, lw=7, solid_capstyle="butt")
        ax.plot([pt - mid, pt + mid], [yi, yi], color=C["blue"], lw=2.5)
        ax.plot([pt], [yi], "o", color=C["orange"], ms=7, zorder=3)
        ax.text(pt + hi + 1.0, yi, f"±{mid:.1f}", va="center", fontsize=9, color=C["gray"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(100, color=C["gray"], lw=0.8, ls=":")
    ax.set_xlabel("GPT-4 상대 점수 (%)")
    ax.set_title("상대 점수의 문항 표본 오차 (95%, 델타 방법)\n진한 선: 후보 SD 2.0, 상관 0.3 / 연한 띠: SD 1.5–2.5, 상관 0–0.5", fontsize=10)
    ax.set_xlim(30, 112)
    v = b["vicuna80_reviews"]
    names = list(v)
    ci = b["vicuna80_bootstrap_ci95_ratio"]
    for i, n in enumerate(names):
        pt = v[n]["ratio_pct"]
        lo, hi = ci[n]
        ax2.plot([lo, hi], [i, i], color=C["green"], lw=2.5)
        ax2.plot([pt], [i], "o", color=C["orange"], ms=7)
        ax2.text(hi + 1, i, f"[{lo:.0f}, {hi:.0f}]", va="center", fontsize=9, color=C["gray"])
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([f"{n}\n(n={v[n]['n_reviews'] - v[n]['parse_failures']})" for n in names])
    ax2.axvline(100, color=C["gray"], lw=0.8, ls=":")
    ax2.set_xlabel("상대 점수 (%)")
    ax2.set_title("실측 앵커: Vicuna-80 GPT-4 판정 4세트\n문항 부트스트랩 95% 구간", fontsize=10)
    ax2.set_xlim(85, 165)
    fig.tight_layout()
    fig.savefig(FIG / "own_bench.png", dpi=160)
    plt.close(fig)


def fig_toy():
    t = json.load(open(HERE / "toy_results.json"))
    arms = list(t["arms"])
    labels = {"base": "선형, -2, 두 단계", "one_stage": "단일 단계", "one_stage_long": "단일 단계(스텝 동일)", "mlp": "MLP, -2, 두 단계",
              "mlp_one_stage": "MLP, 단일 단계", "last_layer": "선형, -1(마지막 층)", "layer_m3": "선형, -3"}
    qtypes = ["count", "color_at", "where", "exist"]
    qn = {"count": "개수", "color_at": "칸의 색", "where": "위치", "exist": "색 유무"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    x = np.arange(len(arms))
    for ax, q in zip(axes, qtypes):
        m = np.array([t["arms"][a][q]["mean"] for a in arms])
        lo = np.clip([t["arms"][a][q]["lo"] for a in arms], 0, 1)      # t 구간을 0-1 로 자름
        hi = np.clip([t["arms"][a][q]["hi"] for a in arms], 0, 1)
        ax.bar(x, m, color=[C["blue"] if a == "base" else C["gray"] for a in arms], alpha=0.85)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(a, a) for a in arms], rotation=60, ha="right", fontsize=8)
        ax.set_title(qn[q])
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("정확도 (보류 장면 2,000문항, 시드 평균 ± t 95%)")
    fig.suptitle("합성 장면 토이: 질문 유형별 정확도 (논문 재현이 아님)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_bench()
    if (HERE / "toy_results.json").exists():
        fig_toy()
