"""recompute_results.json 으로 우리 그래프 세 장을 그린다 (report/figures/own_*.png).
실행: python3 -I analysis/plot_figures.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams["axes.unicode_minus"] = False
d = json.load(open(os.path.join(HERE, "recompute_results.json")))
INK, GREY, BLUE, ORANGE, LIGHT = "#1b1b1a", "#8a8981", "#1f5fae", "#c8641b", "#a9c4e8"
tools = ["CodeStory", "LearnByInteract", "OpenHands"]


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK); ax.grid(axis="x", color="#e5e3dc", lw=0.7); ax.set_axisbelow(True)

# 1. 해결률: 원래, RQ1 오답 제외, suspicious 제외
fig, ax = plt.subplots(figsize=(7.2, 3.6)); clean(ax)
w = 0.26
series = [("원래 해결률", [d["table1"][t]["resolved_pct"] for t in tools], BLUE),
          ("RQ1 오답 제외", [d["table1"][t]["updated_pct"] for t in tools], LIGHT),
          ("suspicious 제외(상한)", [d["table2"][t]["wo_susp_pct"] for t in tools], ORANGE)]
for i, (lab, vals, c) in enumerate(series):
    xs = [j + (i - 1) * w for j in range(3)]
    ax.bar(xs, vals, w * 0.92, color=c, label=lab)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(tools); ax.set_ylabel("SWE-bench Verified 해결률 (%)", color=INK)
ax.set_ylim(0, 72); ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
ax.set_title("해결률은 테스트를 늘리면 내려가지만, 내려간 몫이 곧 오답은 아니다", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_resolution.png"), dpi=170); plt.close(fig)

# 2. 부풀림 추정 구간
e = d["extrapolation"]
rows = [("직접 확인된 오답만", d["confirmed_only"]["inflation_mean_pp"], BLUE),
        ("22/77 Wilson 하한", e["wilson_low"]["inflation_mean_pp"], LIGHT),
        ("논문 점추정 22/77", e["point_22_of_77"]["inflation_mean_pp"], BLUE),
        ("22/77 Wilson 상한", e["wilson_high"]["inflation_mean_pp"], LIGHT)]
fig, ax = plt.subplots(figsize=(7.2, 3.0)); clean(ax)
for i, (lab, v, c) in enumerate(rows[::-1]):
    ax.barh(i, v, 0.55, color=c); ax.text(v + 0.15, i, f"{v:.1f}%p", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlim(0, 10); ax.set_xlabel("해결률 부풀림 평균 (%p)", color=INK)
ax.set_title("6.4%p는 5.0–8.4%p 구간 안의 값이다 (판정 불가 51건은 오답으로 세지 않음)", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_inflation.png"), dpi=170); plt.close(fig)

# 3. 탐지 LLM에 따른 suspicious 비율
fig, ax = plt.subplots(figsize=(7.2, 3.4)); clean(ax)
models = [("gpt4omini", "GPT-4o-mini", GREY), ("deepseek", "DeepSeek-V3", LIGHT), ("qwen", "Qwen3-235B", BLUE)]
for i, (k, lab, c) in enumerate(models):
    xs = [j + (i - 1) * w for j in range(3)]
    vals = [d["table4"][t][k] for t in tools]
    ax.bar(xs, vals, w * 0.92, color=c, label=lab)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, str(v), ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([t + " (n=100)" for t in tools]); ax.set_ylabel("suspicious 패치 수", color=INK)
ax.set_ylim(0, 60); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("같은 300개 패치에서 탐지 모델만 바꾸면 28.0% → 47.7%", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_llm_sensitivity.png"), dpi=170); plt.close(fig)
print("ok")
