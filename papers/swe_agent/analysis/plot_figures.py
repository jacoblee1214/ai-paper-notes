"""recompute_results.json 으로 우리 그래프 세 장을 그린다 (report/figures/own_*.png).
실행: python3 -I analysis/plot_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)
for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
d = json.load(open(os.path.join(HERE, "recompute_results.json")))
INK, GREY, BLUE, ORANGE, LIGHT = "#1b1b1a", "#8a8981", "#1f5fae", "#c8641b", "#a9c4e8"


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(axis=grid, color="#e5e3dc", lw=0.7)
    ax.set_axisbelow(True)


# 1. 표 3 어블레이션: 기준 대비 하락(문항 수)과 McNemar p의 범위
t3 = d["table3"]
names = list(t3.keys())
order = sorted(names, key=lambda n: -t3[n]["baseline_minus_variant_instances"])
fig, ax = plt.subplots(figsize=(7.6, 4.0))
clean(ax)
for i, n in enumerate(order[::-1]):
    r = t3[n]
    diff = r["baseline_minus_variant_instances"]
    z = r["z_if_discordance_like_between_runs"]
    col = BLUE if z >= 1.96 else (LIGHT if z >= 1.0 else GREY)
    ax.barh(i, diff, 0.6, color=col)
    pv = r["p_if_discordance_like_between_runs_normal_approx"]
    ptxt = "p<0.001" if pv < 0.001 else f"p≈{pv:.3f}"
    ax.text(diff + 0.4, i, f"{diff}문항 ({diff / 3:.1f}%p)  {ptxt}",
            va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order[::-1], fontsize=8)
ax.set_xlim(0, 36)
ax.set_xlabel("기준(54/300) 대비 해결 문항 감소", color=INK)
ax.set_title("표 3 어블레이션: 같은 설정 두 번 실행 사이의 불일치 수준을 잡음으로 보면\n3%p 안팎의 차이는 구분되지 않는다", fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_ablation.png"), dpi=170)
plt.close(fig)

# 2. 6회 실행에서 문항별 성공 횟수 분포(표 10의 pass@k에서 복원)
h = d["table10"]["histogram_n_solved_c_of_6_runs"]
labels = list(h.keys())
vals = [h[k] for k in labels]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
clean(ax, "y")
cols = [GREY] + [ORANGE] * 5 + [BLUE]
ax.bar(range(7), vals, 0.7, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(7))
ax.set_xticklabels([f"{i}회" for i in range(7)])
ax.set_xlabel("6회 실행 중 해결한 횟수 (Lite 300문항)", color=INK)
ax.set_ylabel("문항 수", color=INK)
n_unst = d["table10"]["n_unstable_0<c<6"]
ax.set_title(f"한 번이라도 푼 98문항 중 약 {n_unst:.0f}문항은 실행마다 결과가 갈린다 (표 10의 pass@k에서 복원)", fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_runs.png"), dpi=170)
plt.close(fig)

# 3. 논문 표 1 대 공개 결과 파일
p = d["public_vs_paper_table1"]
rows = [("Lite GPT-4 Turbo", "lite/gpt4"), ("Lite Claude 3 Opus", "lite/claude3opus"),
        ("전체 GPT-4 Turbo", "test/gpt4"), ("전체 Claude 3 Opus", "test/claude3opus")]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
clean(ax, "y")
w = 0.36
for j, (lab, key) in enumerate(rows):
    r = p[key]
    ax.bar(j - w / 2, r["paper_table1_pct"], w * 0.95, color=BLUE, label="논문 표 1" if j == 0 else None)
    ax.bar(j + w / 2, r["public_pct"], w * 0.95, color=ORANGE, label="공개 results.json(중복 제거)" if j == 0 else None)
    ax.text(j - w / 2, r["paper_table1_pct"] + 0.3, f"{r['paper_table1_pct']:.2f}", ha="center", fontsize=8, color=INK)
    ax.text(j + w / 2, r["public_pct"] + 0.3, f"{r['public_pct']:.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(4))
ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_ylabel("해결률 (%)", color=INK)
ax.set_ylim(0, 22)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("GPT-4 Turbo 두 행은 논문과 공개 파일이 같고, Claude 3 Opus 두 행은 다르다", fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_public_vs_paper.png"), dpi=170)
plt.close(fig)
print("saved 3 figures to", OUT)
