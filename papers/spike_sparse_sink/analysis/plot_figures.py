"""우리가 그리는 그래프: own_decoupling.png(표 3–8의 sink–spike 산점도), own_sink_threshold.png(sink ratio 문턱의 의미),
own_toy.png(toy_spike_sink_results.json이 있으면). 실행: python3 plot_figures.py (matplotlib)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",):
    if Path(f).exists():
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
for n in ("Noto Sans CJK KR", "Noto Sans KR"):
    if any(n in x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"] = n
plt.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
R = json.load(open(HERE / "recompute_tables_results.json"))
T = R["tables_3_8_transcribed"]
COL = {"T3": "#1f5fae", "T4": "#c0392b", "T5": "#2e8b57", "T6": "#8e6bbf", "T7": "#d98e04", "T8": "#555555"}
LAB = {"T3": "표 3 최적화", "T4": "표 4 FFN", "T5": "표 5 정규화", "T6": "표 6 head", "T7": "표 7 gate", "T8": "표 8 문맥"}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
seen = set()
for g, rows in T.items():
    k = g.split(".")[0]
    for n, (ppl, sink, spk) in rows.items():
        base = "base" in n
        ax.scatter(spk, sink, s=70 if base else 38, c=COL[k], marker="*" if base else "o",
                   edgecolors="k" if base else "none", label=LAB[k] if k not in seen and not base else None, zorder=3)
        seen.add(k) if not base else None
ax.scatter([], [], marker="*", c="w", edgecolors="k", s=70, label="기준 모델 (모든 표 공통)")
ax.set_xscale("log"); ax.set_xlabel("spike 크기 (최대 activation, 로그)"); ax.set_ylabel("sink ratio (%)")
for nm, (x, y) in {"Sandwich": (520, 44.7), "Sandwich(QK)": (92, 42.0), "DynTanh": (153, 61.0), "gate(channel)": (202, 4.5), "문맥 1024/4096": (38470, 13.0)}.items():
    ax.annotate(nm, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
ax.axhline(46.0, color="#999", lw=0.7, ls=":"); ax.axvline(3818, color="#999", lw=0.7, ls=":")
ax.set_title("표 3–8의 모든 행: sink와 spike는 한 방향으로만 움직이지 않는다", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "own_decoupling.png", dpi=170); plt.close(fig)

# sink ratio 문턱: ε별로 필요한 "첫 토큰에 몰아주는 비율" (T=64)와 T에 따른 변화
import numpy as np
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
eps = np.linspace(0.05, 0.9, 50)
for T_ in (64, 256, 4096):
    ax[0].plot(eps, (eps * T_ - 1) / (T_ - 1), label=f"T={T_}")
ax[0].axvline(0.3, color="#999", ls=":"); ax[0].set_xlabel("문턱 ε"); ax[0].set_ylabel("모든 질의가 첫 토큰에 몰아줘야 하는 비율")
ax[0].set_title("ε=0.3이면 약 29%", fontsize=9); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
Ts = np.array([16, 32, 64, 128, 256, 512, 1024, 4096])
ax[1].plot(Ts, [sum(1 / t for t in range(1, T_ + 1)) / T_ for T_ in Ts], "o-")
ax[1].set_xscale("log"); ax[1].set_xlabel("길이 T"); ax[1].set_ylabel("균일 인과 attention의 첫 토큰 평균 수신량 α_1")
ax[1].set_title("균일 기준선 (H_T/T)", fontsize=9); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT / "own_sink_threshold.png", dpi=170); plt.close(fig)

# 토이
tj = HERE / "toy_spike_sink_results.json"
if tj.exists():
    import collections
    J = json.load(open(tj)); runs = J["runs"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
    for v, c in (("prenorm", "#1f5fae"), ("sandwich", "#2e8b57")):
        rr = [r for r in runs if r["variant"] == v]
        if not rr: continue
        xs = [r["base"]["layers"][1]["ratio"] for r in rr]; ys = [r["base"]["sink"]["0.1"] * 100 for r in rr]
        ax[0].scatter(xs, ys, c=c, label=v)
        ax[1].scatter([r["base"]["loss"] for r in rr], [r["ablate_top3"]["loss"] - r["base"]["loss"] for r in rr], c=c, label=v + " top3 0")
        ax[1].scatter([r["base"]["loss"] for r in rr], [r["ablate_random3"]["loss"] - r["base"]["loss"] for r in rr], c=c, marker="x", label=v + " 무작위3 0")
        ax[2].scatter([r["base"]["alpha0_mean"] for r in rr], [r["base"]["layers"][1]["ratio"] for r in rr], c=c, label=v)
    ax[0].set_xlabel("spike 비 (층 2, 위치 0 최대/나머지 중앙값)"); ax[0].set_ylabel("sink ratio ε=0.1 (%)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("기준 손실"); ax[1].set_ylabel("개입 후 손실 증가"); ax[1].legend(fontsize=6.5)
    ax[2].set_xlabel("첫 토큰 평균 수신량 α"); ax[2].set_ylabel("spike 비")
    for a in ax: a.grid(alpha=0.25)
    fig.suptitle("토이(4층 d=128 바이트 LM, 시드별 한 점)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "own_toy.png", dpi=170); plt.close(fig)
print("ok")
