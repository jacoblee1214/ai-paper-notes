"""우리가 그린 그래프: 오답 추적에서 오류 위치 제공 vs 무작위 위치의 정확도 향상 (표 6 재구성 + Wilson 95% 구간).

실행: python3 analysis/plot_figures.py  → report/figures/own_delta_ci.png
입력: analysis/recompute_results.json (recompute_tables.py가 만든다)
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

with open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8") as f:
    d = json.load(f)["table6_incorrect_ci"]

labels = {"word_sorting": "단어 정렬", "tracking_shuffled_objects": "섞인 물건 추적",
          "logical_deduction": "논리 추론", "multistep_arithmetic": "다단 산술", "dyck_languages": "Dyck"}
tasks = list(labels)
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
y = range(len(tasks))
for off, key, color, name in [(0.18, "oracle", "#1f5fae", "오류 위치 제공(오라클)"),
                               (-0.18, "random", "#b5b1a6", "무작위 위치")]:
    vals, lo, hi = [], [], []
    for t in tasks:
        k = d[t][f"{key}_k"]; n = 504 if t == "dyck_languages" else 255
        p = 100.0 * k / n
        l, h = d[t][f"{key}_wilson95"]
        vals.append(p); lo.append(p - l); hi.append(h - p)
    ax.barh([i + off for i in y], vals, height=0.34, color=color, label=name,
            xerr=[lo, hi], error_kw={"ecolor": "#333", "capsize": 2, "lw": 0.8})
ax.set_yticks(list(y)); ax.set_yticklabels([labels[t] for t in tasks])
ax.invert_yaxis()
ax.set_xlabel("오답 추적 중 정답으로 바뀐 비율 (%), Wilson 95% 구간")
ax.set_title("표 6 재구성: 오답 추적에서 수정 성공 비율", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
out = os.path.join(ROOT, "report", "figures", "own_delta_ci.png")
fig.savefig(out)
print("wrote", out)
