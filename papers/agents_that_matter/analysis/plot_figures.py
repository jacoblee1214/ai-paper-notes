"""우리가 그린 그래프 두 장. 입력: analysis/recompute_results.json (recompute.py가 만든다)
  report/figures/own_pareto.png          : HumanEval 정확도 대 비용(2024-04 가격), 볼록 전선, 정확도는 단일 실행 Wilson 95% 구간
  report/figures/own_hotpot_candidates.png : HotPotQA 결합 최적화 후보 4개의 시험 정확도 대 비용 (개발 정확도로 고른 후보 표시)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for FONT in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
    if os.path.exists(FONT):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open(os.path.join(HERE, "recompute_results.json"), encoding="utf-8"))
OUTDIR = os.path.join(ROOT, "report", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ---- 그림 1: 파레토
pts = R["pareto_0409"]["points"]
front = R["pareto_0409"]["convex_frontier"]
simple = {"Warming (GPT-4)", "Retry (GPT-4)", "Escalation", "GPT-4", "GPT-3.5"}
OFF = {"LDB (GPT-4)": (6, -12), "LDB (Reflexion, GPT-4)": (6, 5), "Retry (GPT-4)": (7, -13), "LDB (GPT-4, GPT-3.5)": (-6, 10), "GPT-4": (-6, -14),
       "Reflexion (GPT-4)": (-6, -14), "LDB (Reflexion, GPT-3.5)": (6, 3), "Warming (GPT-4)": (-6, 12)}
HA = {"LDB (GPT-4, GPT-3.5)": "right", "GPT-4": "right", "Reflexion (GPT-4)": "right", "Warming (GPT-4)": "right"}
fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
xs = [pts[n]["cost"] for n in front]
ys = [pts[n]["acc"] for n in front]
ax.plot(xs, ys, color="#8a8a85", lw=1.2, ls="--", zorder=1, label="볼록 전선")
for n, p in pts.items():
    c = "#1f5fae" if n in simple else "#d9822b"
    k = 164
    a = p["acc"] / 100
    z = 1.96
    center = (a + z * z / (2 * k)) / (1 + z * z / k)
    half = z * ((a * (1 - a) / k + z * z / (4 * k * k)) ** 0.5) / (1 + z * z / k)
    ax.errorbar(p["cost"], p["acc"], yerr=[[p["acc"] - 100 * (center - half)], [100 * (center + half) - p["acc"]]],
                fmt="o", color=c, ms=6, capsize=2, lw=0.9, zorder=3)
    ax.annotate(n, (p["cost"], p["acc"]), textcoords="offset points", xytext=OFF.get(n, (5, 4)), fontsize=7, ha=HA.get(n, "left"))
ax.set_xscale("log")
ax.set_xticks([0.05, 0.1, 0.3, 1, 3, 10, 30, 100])
ax.set_xticklabels(["0.05", "0.1", "0.3", "1", "3", "10", "30", "100"])
ax.minorticks_off()
ax.set_xlabel("실행 5회 평균 총비용 (USD, 164문항, 로그 축, 2024-04 가격)")
ax.set_ylabel("정확도 (%)")
ax.set_ylim(64, 98)
ax.spines[["top", "right"]].set_visible(False)
ax.plot([], [], "o", color="#1f5fae", label="단순 기준선")
ax.plot([], [], "o", color="#d9822b", label="복잡한 에이전트 (LDB, LATS, Reflexion)")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title("HumanEval: 정확도 대 비용 (오차 막대 = 단일 실행의 Wilson 95% 구간)", fontsize=10)
fig.text(0.5, 0.005, "논문 표 A1의 평균을 저자 공개 실행 집계에서 다시 계산. 오차 막대는 과제 표본(164) 불확실성을 보이며 논문 그림 A1의 실행 간 구간보다 훨씬 넓다.", ha="center", fontsize=6.5, color="#55544f")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUTDIR, "own_pareto.png"))

# ---- 그림 2: HotPotQA 후보
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=200)
for ax, model in zip(axes, ["GPT-3.5", "Llama-3-70B"]):
    h = R["hotpotqa"][model]
    ch = h["joint_chosen_by_dev"]
    for c in h["joint_candidates"]:
        sel = c["i"] == ch
        ax.scatter(c["cost_per100"], c["test_gold_mean"] * 100, s=70 if sel else 40, color="#d9822b" if sel else "#1f5fae", zorder=3,
                   edgecolor="k" if sel else "none")
        ax.annotate(f"후보{c['i']} (개발 {c['dev_acc']*100:.0f}%)", (c["cost_per100"], c["test_gold_mean"] * 100), textcoords="offset points", xytext=(5, 4), fontsize=7)
    for name, lab, col in [("few_shot", "few-shot", "#7a7a75"), ("random_search", "random search", "#a8a8a2")]:
        d = h[name]
        ax.scatter(d["var_cost_per100"], d["acc_mean"] * 100, marker="s", s=40, color=col, zorder=2)
        ax.annotate(lab, (d["var_cost_per100"], d["acc_mean"] * 100), textcoords="offset points", xytext=(5, -9), fontsize=7)
    ax.set_title(model, fontsize=10)
    ax.set_xlabel("변동비 (USD / 100건)")
    ax.set_ylabel("시험 정확도 = 정답 문서 검색 성공률 (%)")
    ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, 0.005, "주황 = 논문이 쓴(개발 정확도가 가장 높은) 후보. 후보 4개, 시험 200문항, 5회 평균.", ha="center", fontsize=7, color="#55544f")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUTDIR, "own_hotpot_candidates.png"))
print("saved")
