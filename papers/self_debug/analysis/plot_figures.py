"""우리가 그린 그래프: Self-Debug 이득을 단계별 증분으로 분해(표 2).

기준선 -> 단순 피드백(통과/실패 판정) -> 단위 테스트 피드백(실행 결과 내용) -> 코드 설명 추가.
실행: python3 analysis/plot_figures.py  -> report/figures/own_gain_decomposition.png
입력: analysis/recompute_tables_results.json (recompute_tables.py가 만든다)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for FONT in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",):
    if os.path.exists(FONT):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

dec = json.load(open(os.path.join(HERE, "recompute_tables_results.json"), encoding="utf-8"))["decomposition"]
models = ["Codex", "GPT-3.5", "GPT-4", "StarCoder"]
steps = [("simple_minus_base", "단순 피드백 (통과·실패 판정)", "#1f5fae"),
         ("UT_minus_simple", "+ 단위 테스트 결과 내용", "#7aa6d8"),
         ("Expl_minus_UT", "+ 코드 설명 (rubber duck)", "#d9822b")]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=200, sharey=True)
for ax, bench, title in zip(axes, ["transcoder", "mbpp"], ["TransCoder (560문항)", "MBPP (500문항)"]):
    for i, m in enumerate(models):
        left = 0.0
        for key, label, c in steps:
            v = dec[bench][m][key]
            ax.barh(i, v, left=left if v >= 0 else left + v, color=c, label=label if i == 0 else None, height=0.6)
            left += v
        ax.text(left + 0.2, i, f"+{left:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("기준선 대비 정확도 증가 (%p, 표 2)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 15.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False)
fig.text(0.5, 0.005, "막대 끝 숫자는 UT+Expl 설정의 총 이득(%p). 음수 증분은 왼쪽으로 표시.", ha="center", fontsize=7, color="#55544f")
fig.tight_layout(rect=(0, 0.1, 1, 1))
os.makedirs(os.path.join(ROOT, "report", "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "report", "figures", "own_gain_decomposition.png"))
print("saved")
