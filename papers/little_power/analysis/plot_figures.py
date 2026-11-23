"""우리가 그리는 그래프 두 장. 입력은 minimal_impl.py와 recompute_results.json. 실행: python3 analysis/plot_figures.py"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimal_impl as mi  # noqa: E402

FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
C = ["#1b6ca8", "#d1495b", "#2a9d8f", "#8a6d1f"]
R = json.loads((HERE / "recompute_results.json").read_text())

# ---- 그림 1: (a) 같은 정확도 차이 2%p에서 n과 불일치율에 따른 검정력, (b) n에 따른 MDE와 GLUE 데이터셋 크기
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ns = np.unique(np.round(np.logspace(np.log10(100), np.log10(20000), 40)).astype(int))
for col, d in zip(C, (0.05, 0.10, 0.30)):
    ax[0].plot(ns, [mi.analytic_power(n, 0.02, 1 - d) for n in ns], color=col, label=f"불일치율 d={d:.2f} (해석식)")
    for n in (200, 500, 1000, 2000, 5000):
        ax[0].plot(n, mi.simulate_power(n, 0.02, 1 - d, r=20000, method="chi2", seed=n)["power"], "o", color=col, ms=4)
ax[0].axhline(0.8, color="gray", ls="--", lw=0.8)
ax[0].set_xscale("log")
ax[0].set_xticks([100, 1000, 10000])
ax[0].set_xticklabels(["100", "1천", "1만"])
ax[0].minorticks_off()
ax[0].set_xlabel("테스트 문항 수 n")
ax[0].set_ylabel("검정력 (정확도 차이 2%p, α=0.05)")
ax[0].set_title("(a) 같은 효과라도 불일치율에 따라 달라짐 (점 = 시뮬레이션)", fontsize=10)
ax[0].legend(fontsize=8, loc="lower right")
grid = np.logspace(np.log10(100), np.log10(400000), 60)
for col, d in zip(C, (0.05, 0.10, 0.30)):
    ax[1].plot(grid, [100 * mi.analytic_mde(n, 1 - d) for n in grid], color=col, label=f"d={d:.2f}")
rows = R["table2_table9_mde"]
for r in rows:
    ax[1].plot(r["n"], r["avg_reported_improvement"], "k^", ms=5)
    ax[1].annotate(r["dataset"], (r["n"], r["avg_reported_improvement"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
ax[1].set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
ax[1].minorticks_off()
ax[1].set_xticks([100, 1000, 10000, 100000])
ax[1].set_xticklabels(["100", "1천", "1만", "10만"])
ax[1].set_xlabel("테스트 문항 수 n")
ax[1].set_ylabel("MDE (%p, 검정력 80%)")
ax[1].set_title("(b) MDE ≈ 2.8·√(d/n). 검은 삼각형 = 논문 표 2의 평균 보고 개선폭", fontsize=10)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "own_power_mde.png", dpi=150)

# ---- 그림 2: Likert. 평가자를 늘려도 문항 항이 바닥이다 (두 모델 평균 차이 0.1, 높은 분산 설정)
from scipy import stats  # noqa: E402

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ws = np.arange(1, 101)
v = mi.LIKERT_HIGH_VAR
for col, items in zip(C, (50, 100, 500)):
    ax.plot(ws, [mi.analytic_likert_power(w, items, 0.1, **v) for w in ws], color=col, label=f"문항 {items}개")
    cap = stats.norm.cdf(0.05 / (v["sd_i1"] / np.sqrt(items)) - 1.96)  # 평가자 무한대의 상한
    ax.axhline(cap, color=col, ls=":", lw=0.9)
ax.axhline(0.8, color="gray", ls="--", lw=0.8)
ax.set_xlabel("문항당 평가자 수")
ax.set_ylabel("검정력 (두 모델 평균 차이 0.1)")
ax.set_title("평가자를 늘려도 문항×모델 분산 항(점선: 상한) 때문에 한계가 있음", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "own_likert_workers.png", dpi=150)
print("saved")
