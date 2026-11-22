"""우리가 그리는 그래프 3장 (matplotlib). 입력: analysis/recompute_results.json, data_stats_results.json
실행: python3 -I analysis/plot_figures.py -> report/figures/own_*.png
"""
import json, os, platform
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
D = json.load(open(os.path.join(HERE, "data_stats_results.json"), encoding="utf-8"))
BLUE, ORANGE, GREY, INK = "#2a6fbb", "#d9822b", "#8a8982", "#1b1b1a"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbb"); ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e4de", lw=0.7); ax.set_axisbelow(True)


# 1. 손수 만든 58건 에이전트 수준: LLM 방법(GT 있음) 대 단순 기준선
fig, ax = plt.subplots(figsize=(7.6, 3.9))
rows = {(r["method"], r["setting"]): r for r in R["tests"]["hand_agent_vs_baselines"]}
names = ["올인원\n(GPT-4o)", "이진 탐색\n(GPT-4o)", "단계별\n(GPT-4o)"]
for i, m in enumerate(("AAO", "BS", "SBS")):
    r = rows[(m, "GT")]
    ax.bar(i, r["pct"], color=BLUE, width=0.55)
    ax.errorbar(i, r["pct"], yerr=[[r["pct"] - r["wilson95"][0]], [r["wilson95"][1] - r["pct"]]], color=INK, capsize=3, lw=1)
    ax.text(i, r["wilson95"][1] + 2, f'{r["pct"]:.1f}', ha="center", fontsize=9, color=INK)
base = D["Hand-Crafted"]
bl = [("무작위(역할 문자열 기준) 19.1", 19.06, GREY), ("기본 에이전트 무작위 42.2", base["baselines_base_pct"]["agent_random_uniform_over_base_agents"], GREY),
      ("항상 WebSurfer 56.9", base["baselines_base_pct"]["agent_constant_WebSurfer"], ORANGE), ("마지막 발화자 60.3", base["baselines_base_pct"]["agent_last_speaker_base"], ORANGE)]
for lab, v, c in bl:
    ax.axhline(v, color=c, lw=1.3, ls="--")
    ax.text(2.45, v + 0.8, lab, ha="right", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 85); ax.set_xlim(-0.5, 2.5); ax.set_ylabel("에이전트 수준 정확도 (%)", fontsize=9)
ax.set_title("손수 만든 시스템(N=58), 정답 제공: 막대는 Wilson 95% 구간", fontsize=10, loc="left")
style(ax); fig.tight_layout(); fig.savefig(os.path.join(OUTD, "own_hand_agent_baselines.png"), dpi=170); plt.close(fig)

# 2. 허용 오차 곡선(Table 2) 대 균등 무작위, 항상 마지막 단계
fig, ax = plt.subplots(figsize=(7.6, 3.9))
T2 = R["table2_tolerance"]; x = [r["tol"] for r in T2]
for key, c, lab in (("AAO", BLUE, "올인원"), ("SBS", ORANGE, "단계별"), ("BS", "#4a9a6b", "이진 탐색")):
    ax.plot(x, [r[key] for r in T2], marker="o", color=c, lw=1.8, label=lab)
ax.plot(x, [r["random_uniform"] for r in T2], color=GREY, lw=1.6, ls="--", label="균등 무작위(재계산)")
ax.plot(x, [r["always_last"] for r in T2], color="#555", lw=1.2, ls=":", label="항상 마지막 단계")
ax.set_xlabel("허용 오차 ±k 단계", fontsize=9); ax.set_ylabel("단계 정확도 (%)", fontsize=9)
ax.set_xticks(x); ax.set_title("표 2(손수 만든 58건): 모든 방법이 균등 무작위 곁에 있음", fontsize=10, loc="left")
ax.legend(fontsize=8.5, frameon=False, loc="upper left"); style(ax)
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "own_tolerance_vs_random.png"), dpi=170); plt.close(fig)

# 3. 길이 수준별 표본 수와 기준선(4.4절): 기준선 자체가 길이에 따라 떨어짐
fig, ax = plt.subplots(figsize=(7.6, 3.6))
L = D["hand_length_levels"]; xs = [l["level"] for l in L]
ax.plot(xs, [l["step_random_pct"] for l in L], marker="o", color=GREY, lw=1.8, label="무작위 단계 (1/n 평균)")
ax.plot(xs, [l["step_always_last_pct"] for l in L], marker="s", color=ORANGE, lw=1.8, label="항상 마지막 단계")
for l in L:
    ax.text(l["level"], -4.2, f'N={l["N"]}', ha="center", fontsize=8.5, color=INK)
ax.set_ylim(-7, 48); ax.set_xticks(xs); ax.set_xticklabels([f'수준 {l["level"]}\n({l["range"]}단계)' for l in L], fontsize=8.5)
ax.set_ylabel("단계 정확도 기준선 (%)", fontsize=9)
ax.set_title("길이 수준별 기준선: 길수록 맞히기 어려운 것은 원래 그렇다", fontsize=10, loc="left")
ax.legend(fontsize=8.5, frameon=False); style(ax)
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "own_length_baselines.png"), dpi=170); plt.close(fig)
print("ok")
