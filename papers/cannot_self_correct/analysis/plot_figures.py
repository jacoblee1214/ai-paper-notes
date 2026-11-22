"""우리가 그린 그래프: Figure 1 분포로 복원한 '두 라운드 자기 수정 전후 정확도 변화'(짝지은 차이)와 95% 구간.

실행: python3 analysis/plot_figures.py  -> report/figures/own_paired_change.png
입력: analysis/recheck_tables_results.json (recheck_tables.py가 만든다)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
if os.path.exists(FONT):
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

d = json.load(open(os.path.join(HERE, "recheck_tables_results.json"), encoding="utf-8"))["figure1_paired"]
names = list(d)
fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=200)
for i, k in enumerate(names):
    v = d[k]
    lo, hi = v["paired_diff_ci95"]
    sig = v["mcnemar_exact_p"] < 0.05
    c = "#b3261e" if sig else "#8c8a82"
    ax.errorbar(v["paired_diff_pts"], i, xerr=[[v["paired_diff_pts"] - lo], [hi - v["paired_diff_pts"]]],
                fmt="o", color=c, capsize=3, lw=1.3)
    ax.text(hi + 1.0, i, f"n={v['n']}, p={v['mcnemar_exact_p']:.3f}" if v["mcnemar_exact_p"] >= 0.001 else f"n={v['n']}, p<0.001",
            va="center", fontsize=7, color="#55544f")
ax.axvline(0, color="#333", lw=0.8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-42, 18)
ax.set_xlabel("자기 수정 2라운드 후 정확도 변화 (%p), 짝지은 차이와 95% 구간 (Figure 1 복원)")
ax.set_title("빨강: McNemar p<0.05 / 회색: 구분 안 됨", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(ROOT, "report", "figures", "own_paired_change.png")
fig.savefig(out)
print("wrote", out)
