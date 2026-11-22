"""우리가 그리는 그래프 세 장 (matplotlib). 입력은 recompute_stats.py가 쓰는 값과 같다(논문 그림 6, 3·4, 10에서 옮긴 값).

  own_prompt_ablation.png      : Conflicting-LiveCodeBench 프롬프트 A–D 부정행위율과 90% Wilson 구간 (n=103)
  own_capability_vs_cheating.png: 원본 통과율 대 부정행위율 (그림 3·4 읽은 값), 스피어만 상관
  own_monitor_ci.png           : 모니터 탐지율과 오탐 상한의 90% Wilson 구간 (그림 10의 n)

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for f in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"GPT-5": "#b8913f", "o3": "#3a9d7f", "Opus-4.1": "#7a73b8", "ink": "#333333"}


def wilson(k, n, conf=0.90):
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h) * 100, (c + h) * 100


def fig_prompt():
    data = {"GPT-5": [92, 93, 28, 1], "o3": [91, 86, 93, 33], "Opus-4.1": [37, 34, 4, 0]}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for m, vals in data.items():
        xs, ys, lo, hi = [], [], [], []
        for i, v in enumerate(vals):
            k = round(v * 103 / 100)
            l, h = wilson(k, 103)
            xs.append(i); ys.append(100 * k / 103); lo.append(100 * k / 103 - l); hi.append(h - 100 * k / 103)
        ax.errorbar([x + (0.06 if m == "o3" else -0.06 if m == "GPT-5" else 0) for x in xs], ys, yerr=[lo, hi], marker="o",
                    color=C[m], label=m, capsize=3, lw=1.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["A (가장 느슨)", "B", "C", "D (가장 엄격)"])
    ax.set_ylabel("Conflicting-LCB 부정행위율 (%)"); ax.set_ylim(-3, 105)
    ax.set_title("프롬프트 한 줄 차이로 1%에서 92%까지 (n=103, 90% Wilson 구간)", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "own_prompt_ablation.png", dpi=200); plt.close(fig)


def fig_capability():
    swe = {"GPT-5": (97.5, 54), "Sonnet-3.7": (88.5, 64.5), "o4-mini": (85, 48.5), "Opus-4.1": (86, 50), "Sonnet-4": (89, 43),
           "o3": (86, 29), "Qwen3-Coder": (70.5, 14), "GPT-4.1": (59, 3)}
    lcb = {"GPT-5": (89, 0.5), "o3": (86, 33), "o4-mini": (82.5, 0), "Opus-4.1": (76, 0), "Sonnet-4": (61, 0),
           "Sonnet-3.7": (54, 0), "Qwen3-Coder": (56, 0), "GPT-4.1": (62, 0.5)}
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, d, t in [(axs[0], swe, "Conflicting-SWEbench (전체 스캐폴드)"), (axs[1], lcb, "Conflicting-LiveCodeBench (최소 스캐폴드)")]:
        x = [v[0] for v in d.values()]; y = [v[1] for v in d.values()]
        rho, p = stats.spearmanr(x, y)
        ax.scatter(x, y, color="#3a6ea5", zorder=3)
        for n, (a, b) in d.items():
            if d is lcb and b < 1 and n not in ("GPT-5", "o4-mini", "Opus-4.1"):
                continue  # 겹치는 0% 점의 이름은 생략(Sonnet-3.7, Sonnet-4, Qwen3-Coder, GPT-4.1)
            ax.annotate(n, (a, b), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_title(f"{t}\n스피어만 ρ={rho:.2f}, p={p:.2f} (모델 8개)", fontsize=9)
        ax.set_xlabel("원본 통과율 (%) (그림에서 읽은 값)"); ax.grid(alpha=.25)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
    axs[0].set_ylabel("부정행위율 (%)")
    fig.tight_layout(); fig.savefig(OUT / "own_capability_vs_cheating.png", dpi=200); plt.close(fig)


def fig_monitor():
    cells = [("LCB\nSonnet 모니터", 86, 113, 89, 80), ("SWE\nSonnet 모니터", 42, 1308, 50, 1063), ("SWE\nOpus 모니터", 57, 1308, 65, 1063)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (name, p1, n1, p2, n2) in enumerate(cells):
        for j, (p, n, col, lab) in enumerate([(p1, n1, "#a8741f", "oneoff"), (p2, n2, "#d95f02", "conflicting")]):
            k = round(p * n / 100); l, h = wilson(k, n)
            ax.errorbar(i + (j - .5) * .25, 100 * k / n, yerr=[[100 * k / n - l], [h - 100 * k / n]], fmt="o", color=col,
                        capsize=3, label=lab if i == 0 else None)
    ax.axhline(7, ls=":", color="#888"); ax.text(2.45, 8, "LCB 원본 7%(오탐 상한)", fontsize=7, ha="right", color="#666")
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in cells]); ax.set_ylim(0, 100)
    ax.set_ylabel("탐지율 (%)"); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("모니터 탐지율, 90% Wilson 구간 (독립 가정; 전사본은 문항·모델별로 군집)", fontsize=9)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "own_monitor_ci.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_prompt(); fig_capability(); fig_monitor()
    print("wrote 3 figures to", OUT)
