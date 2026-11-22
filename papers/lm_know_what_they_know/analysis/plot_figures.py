"""우리가 그리는 그래프 두 장. 입력은 recompute_calibration_results.json (논문 그림에서 읽은 값 포함).
실행: python3 analysis/plot_figures.py  -> report/figures/own_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
matplotlib.rcParams["axes.unicode_minus"] = False
R = json.load(open(os.path.join(HERE, "recompute_calibration_results.json")))

# 그림 1: Fig 14 읽은 값 + Brier 모델 대 상수
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sizes = [0.8, 3, 12, 52]
for name, v in R["fig14_auroc_by_size_read"].items():
    ax[0].plot(sizes, v, marker="o", label=name)
ax[0].axhline(0.5, color="gray", lw=0.8, ls="--")
ax[0].set_xscale("log"); ax[0].set_xlabel("파라미터 (B)"); ax[0].set_ylabel("AUROC")
ax[0].set_title("Fig 14 읽은 값: TriviaQA만 학습한 P(IK)")
ax[0].legend(fontsize=8)
cols = {"Lambada": "#1f5fae", "Codex": "#c05a00"}
for t, rows in R["fig11_derived"].items():
    xs = [r["size"] for r in rows]
    ax[1].plot(xs, [r["brier_model"] for r in rows], marker="o", color=cols[t], label=f"{t} 모델 (20-shot)")
    ax[1].plot(xs, [r["brier_constant_base_rate"] for r in rows], marker="x", ls="--", color=cols[t], label=f"{t} 기저율 상수 pi(1-pi)")
ax[1].set_ylabel("Brier"); ax[1].set_title("Fig 11 읽은 값: 자기 평가 Brier 대 상수 예측기")
ax[1].legend(fontsize=7)
fig.text(0.5, -0.02, "값은 논문 그림에서 눈으로 읽은 것(오차 약 0.01). 우리가 그린 그래프.", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_readings.png"), dpi=160, bbox_inches="tight")

# 그림 2: ECE 잡음 바닥
fl = R["ece_noise_floor_perfect_calibration"]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
Ns = [200, 1000, 5000, 20000, 100000]
for b in (5, 10, 20):
    ax.plot(Ns, [fl[f"n={n},bins={b}"] for n in Ns], marker="o", label=f"구간 {b}개")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("예측 수 N"); ax.set_ylabel("완벽히 보정된 예측기의 ECE")
ax.set_title("ECE 잡음 바닥 (같은 개수 구간, 토이)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_ece_floor.png"), dpi=160)
print("ok")
