"""우리가 그리는 그래프 두 장 (matplotlib). 입력: recompute_trace.py와 같은 값(논문 그림 10의 막대 라벨)과 토이.

  own_gain.png  : 수학 8개 설정의 TRACE 대 72B 모니터 F1과 상대 이득, 초록의 65% 기준선
  own_toy.png   : 토이 점수 분포(정직 대 해킹)와 임계값, 과잉 사고에서 오탐 비율

실행: python3 analysis/plot_figures.py  → report/figures/own_*.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
for f in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, INK = "#1f5fae", "#c8641e", "#333333"

fig10 = {"IC": {"Llama3.2-3B": (0.755, 0.964), "Qwen2.5-3B": (0.660, 0.932), "Qwen2.5-7B": (0.595, 0.898), "Qwen2.5-14B": (0.517, 0.883)},
         "RM": {"Llama3.2-3B": (0.830, 0.956), "Qwen2.5-3B": (0.528, 0.927), "Qwen2.5-7B": (0.558, 0.878), "Qwen2.5-14B": (0.394, 0.875)}}

def fig_gain():
    labels, gains = [], []
    for s, d in fig10.items():
        for k, (m, t) in d.items():
            labels.append(f"{s} {k}"); gains.append((t / m - 1) * 100)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cols = [ORANGE if g > 65 else BLUE for g in gains]
    ax.bar(range(len(gains)), gains, color=cols)
    ax.axhline(65, color=INK, ls="--", lw=1); ax.text(7.45, 67, "초록의 65%", ha="right", fontsize=8)
    for i, g in enumerate(gains): ax.text(i, g + 2, f"+{g:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(gains))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("TRACE의 F1 상대 이득 (대 72B 모니터)"); ax.set_title("그림 10 라벨로 다시 계산한 수학 이득: 8개 중 3개만 65% 초과", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT / "own_gain.png", dpi=200); plt.close(fig)

def fig_toy():
    rng = np.random.default_rng(1)
    G = np.arange(0.1, 1.01, 0.1)
    def honest(n, lo=0.6, hi=1.0):
        t = rng.uniform(lo, hi, n)[:, None]; return 0.9 / (1 + np.exp(-(G[None] - t) / 0.05))
    def score(p, m=5): return 100 * (rng.binomial(m, np.clip(p, 0, 1)) / m).mean(axis=1)
    n = 4000
    sh = score(honest(n)); sk = score(np.full((n, len(G)), 0.95))
    thr = sh.mean()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(sh, bins=40, alpha=.7, color=BLUE, label="정직"); ax[0].hist(sk, bins=40, alpha=.7, color=ORANGE, label="해킹")
    ax[0].axvline(thr, color=INK, ls="--"); ax[0].text(thr + 1, ax[0].get_ylim()[1] * .85, f"임계값=정직 평균\n정직의 {100*(sh>thr).mean():.0f}% 오탐", fontsize=8)
    ax[0].set_xlabel("TRACE 점수 (토이, m=5)"); ax[0].legend(fontsize=8); ax[0].set_title("순위는 완벽, 임계값이 F1을 결정", fontsize=10)
    shifts = [(0.6, 1.0), (0.5, 0.9), (0.4, 0.8), (0.3, 0.7)]
    fr = [(score(honest(n, a, b)) > thr).mean() * 100 for a, b in shifts]
    ax[1].bar(range(4), fr, color=BLUE)
    for i, f in enumerate(fr): ax[1].text(i, f + 1.5, f"{f:.0f}%", ha="center", fontsize=8)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["이동 0", "0.1 일찍", "0.2 일찍", "0.3 일찍"], fontsize=8)
    ax[1].set_ylabel("임계값을 넘는 정직 응답 (%)"); ax[1].set_title("과잉 사고 이동이 만드는 오탐 (해킹 없음)", fontsize=10); ax[1].set_ylim(0, 112)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "own_toy.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    fig_gain(); fig_toy(); print("ok")
