"""우리가 그리는 그래프 3장 (matplotlib). 입력: recompute_results.json, toy_results.json, figure_values.json
실행: python3 -I analysis/plot_figures.py -> report/figures/own_*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(OUTD, exist_ok=True)
for f in ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
        plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8"))
T = json.load(open(os.path.join(HERE, "toy_results.json"), encoding="utf-8"))
BLUE, ORANGE, GREY, INK, GREEN = "#2a6fbb", "#d9822b", "#8a8982", "#1b1b1a", "#4a9a6b"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbb"); ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e4de", lw=0.7); ax.set_axisbelow(True)


TABLE2 = {"AQuA": (0.44, 28), "LogiQA": (0.26, 42), "MMLU": (0.12, 68), "HellaSwag": (0.12, 71), "TruthfulQA": (0.11, 59),
          "OpenBookQA": (0.07, 82), "ARC (Challenge)": (0.05, 88), "ARC (Easy)": (0.02, 96)}

# 1. AOC(early answering) 대 CoT 없는 정확도 (표 2)
fig, ax = plt.subplots(figsize=(7.2, 3.9))
xs = np.array([v[1] for v in TABLE2.values()]); ys = np.array([v[0] for v in TABLE2.values()])
lr = R["A_table2"]["linreg_AOC_EA_on_noCoT_acc"]
xx = np.linspace(25, 100, 20)
ax.plot(xx, lr["intercept"] + lr["slope"] * xx, color=GREY, ls="--", lw=1.3)
ax.scatter(xs, ys, color=BLUE, s=38, zorder=3)
for t, (a, acc) in TABLE2.items():
    off = (6, -12) if t == "HellaSwag" else (-6, 7) if t == "MMLU" else (6, 4)
    ax.annotate(t, (acc, a), textcoords="offset points", xytext=off, fontsize=8.5, color=INK, ha="right" if t == "MMLU" else "left")
sp = R["A_table2"]["spearman_AOC_EA_vs_noCoT_accuracy"]
ax.set_xlabel("CoT 없이 맞힌 정확도 (%)  (표 2)", fontsize=9); ax.set_ylabel("early answering AOC (표 2)", fontsize=9)
ax.set_title(f"AOC는 'CoT 없이 못 푸는 정도'와 거의 일직선: Spearman ρ={sp['spearman_rho']}, R²={lr['r2']} (n=8)", fontsize=9.5, loc="left")
ax.set_xlim(20, 104); style(ax); fig.tight_layout(); fig.savefig(os.path.join(OUTD, "own_aoc_vs_nocot.png"), dpi=170); plt.close(fig)

# 2. 175B에서 같은 답 비율과 정확도가 강제하는 바닥
fig, ax = plt.subplots(figsize=(7.6, 3.9))
at = R["E_accuracy_floor"]["at_175B"]
order = sorted(at, key=lambda t: at[t]["same"])
for i, t in enumerate(order):
    ax.bar(i, at[t]["same"], color="#cfdcec", width=0.62)
    ax.bar(i, at[t]["floor_from_accuracy"], color=BLUE, width=0.62)
    ax.text(i, at[t]["same"] + 1.5, f'{at[t]["same"]:.0f}', ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(order))); ax.set_xticklabels([t.replace(" (", "\n(") for t in order], fontsize=8)
ax.set_ylim(0, 108); ax.set_ylabel("CoT 유무에서 같은 답 (%), 175B (그림 7)", fontsize=9)
ax.set_title("진한 막대 = 두 정확도(그림 10)만으로 이미 강제되는 하한 a+b-100", fontsize=10, loc="left")
style(ax); fig.tight_layout(); fig.savefig(os.path.join(OUTD, "own_floor_175b.png"), dpi=170); plt.close(fig)

# 3. 토이: 절단 곡선
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
M = T["models"]
sel = [("reader(CoT 필수)", BLUE, "-"), ("capable(k=2)", GREEN, "-"), ("post-hoc 정당화(답 선결정)", ORANGE, "-"),
       ("혼합(단서 30% + reader)", "#8e5bb5", "--")]
for name, c, ls in sel:
    p = M[name]["EA_p_by_truncation"]
    ax.plot(np.arange(len(p)) / (len(p) - 1) * 100, np.array(p) * 100, color=c, ls=ls, marker="o", ms=3.5, lw=1.7,
            label=f'{name}  AOC={M[name]["AOC_EA"]:.2f}')
ax.axhline(10, color=GREY, lw=0.8, ls=":")
ax.set_xlabel("제공한 CoT 비율 (%)", fontsize=9); ax.set_ylabel("완전한 CoT와 같은 답 (%)", fontsize=9)
ax.set_title("토이 early answering 곡선 (n=6단계, 선택지 10)", fontsize=9.5, loc="left")
ax.legend(fontsize=7.4, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.66)); style(ax)
ax = axs[1]
sw = T["capability_sweep"]
ax.plot([s["k"] for s in sw], [s["AOC_EA"] for s in sw], color=BLUE, marker="o", lw=1.8)
ax.set_xlabel("모델이 암산할 수 있는 단계 수 k", fontsize=9); ax.set_ylabel("AOC", fontsize=9)
ax.set_title("능력이 오르면 AOC는 선형으로 하락\n(CoT는 계속 정확히 쓰이는데도)", fontsize=9.5, loc="left")
style(ax); fig.tight_layout(); fig.savefig(os.path.join(OUTD, "own_toy_truncation.png"), dpi=170); plt.close(fig)
print("ok")
