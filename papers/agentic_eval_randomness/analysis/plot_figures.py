"""그래프 세 장을 report/figures/ 에 그린다 (own_*.png). 입력: recompute_runs_results.json, power_check_results.json

  own_run_spread.png   : 12개 설정의 10회 실행 해결률(점)과 범위
  own_power_runs.png   : 효과 크기별 필요 실행 수 (식 11 z 근사, 정확한 t, 시뮬레이션 지점)
  own_mde_vs_K.png     : 문항당 반복 수 K에 따른 쌍체 비교의 최소 탐지 효과(80% 검정력), 문항 모집단 하한 포함

실행: python3 -I analysis/plot_figures.py   (matplotlib)
"""
import json
import os
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(OUT, exist_ok=True)

for name in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"]:
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, BLUE, ORANGE = "#1b1b1a", "#6b6a64", "#1f5fae", "#c2570c"
R = json.load(open(os.path.join(HERE, "recompute_runs_results.json")))
P = json.load(open(os.path.join(HERE, "power_check_results.json")))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color="#eee", lw=0.6)
    ax.set_axisbelow(True)


def fig_spread():
    import glob
    sys_path = HERE
    import sys
    sys.path.insert(0, sys_path)
    from recompute_runs import load_all
    mats, _ = load_all()
    keys = sorted(mats, key=lambda k: (k[0], k[1], k[2]))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, k in enumerate(keys):
        r = mats[k].sum(1) / 500 * 100
        col = ORANGE if k[2] == "0.0" else BLUE
        ax.plot([r.min(), r.max()], [i, i], color=col, lw=2, alpha=0.5, solid_capstyle="round")
        ax.scatter(r, np.full(10, i), s=14, color=col, zorder=3, edgecolor="white", linewidth=0.5)
        ax.text(r.max() + 0.8, i, f"{r.max() - r.min():.1f}%p", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{k[0]} {k[1]} T={k[2]}" for k in keys], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("실행별 해결률 (%), 점 하나가 실행 하나 (SWE-bench Verified 500문항)", fontsize=8.5, color=MUTED)
    ax.set_xlim(12, 72)
    style(ax)
    ax.text(0.99, 0.02, "파랑: 권장 온도, 주황: 온도 0", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_run_spread.png"), dpi=200)
    plt.close(fig)


def fig_power():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    d = np.linspace(0.8, 10.5, 300)
    from scipy import stats
    za, zb = stats.norm.isf(0.025), stats.norm.ppf(0.8)
    n = 2 * ((za + zb) / (d / 1.5)) ** 2
    ax.plot(d, n, color=BLUE, lw=2)
    # 정확한 t (표 2의 sigma=1.5 행)
    rows = [r for r in P["table2"] if r["sigma"] == 1.5]
    ax.scatter([r["improve"] for r in rows], [r["t_exact"][0] for r in rows], color=ORANGE, s=30, zorder=3)
    ax.scatter([r["improve"] for r in rows], [r["paper"][0] for r in rows], color=BLUE, s=22, zorder=3, facecolor="white", linewidth=1.2)
    ax.text(1.9, 42, "식 11 (z 근사), 논문 표 2와 일치", color=BLUE, fontsize=8.5)
    ax.text(3.0, 16, "정확한 t 검정 (주황 점)", color=ORANGE, fontsize=8.5)
    ax.set_ylim(0, 60)
    ax.set_xlabel("탐지할 개선 폭 Δ (%p), σ=1.5%p, α=0.05, 검정력 80%", fontsize=8.5, color=MUTED)
    ax.set_ylabel("에이전트당 필요 실행 수", fontsize=8.5, color=MUTED)
    style(ax)
    ax.grid(axis="y", color="#eee", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_power_runs.png"), dpi=200)
    plt.close(fig)


def fig_mde():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    Ks = [1, 3, 5, 10, 30]
    for rec in P["population_floor"]:
        real = abs(rec["diff_pp"]) > 5
        col = ORANGE if real else BLUE
        ax.plot(Ks, [rec[f"mde_K{K}_pp"] for K in Ks], color=col, lw=1.4, alpha=0.8 if real else 0.45, marker="o", ms=3)
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.axhline(2, color="#999", lw=0.8, ls="--")
    ax.text(1.05, 2.1, "2%p", color=MUTED, fontsize=8)
    ax.text(1.1, 5.9, "주황: 두 설정이 실제로 다른 쌍 (효과가 문항마다 다름)", color=ORANGE, fontsize=8)
    ax.text(1.1, 5.55, "파랑: 거의 같은 쌍", color=BLUE, fontsize=8)
    ax.set_xlabel("문항당 반복 수 K (같은 500문항, 쌍체 비교)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("80% 검정력으로 탐지되는 최소 차이 (%p)", fontsize=8.5, color=MUTED)
    ax.set_ylim(0, 6.5)
    style(ax)
    ax.grid(axis="y", color="#eee", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "own_mde_vs_K.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_spread()
    fig_power()
    fig_mde()
    print("ok")
