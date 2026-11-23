"""재계산 결과로 우리 그래프 세 장을 그린다 (report/figures/own_*.png).
실행: python3 -I analysis/plot_figures.py   (먼저 recompute_scores.py, recompute_claims.py, recompute_tau_bench.py)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures"); os.makedirs(OUT, exist_ok=True)
for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, BLUE, ORANGE = "#1b1b1a", "#8a8981", "#1f5fae", "#c8641b"
sc = json.load(open(os.path.join(HERE, "recompute_scores_results.json")))
cl = json.load(open(os.path.join(HERE, "recompute_claims_results.json")))

def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK); ax.grid(axis=grid, color="#e5e3dc", lw=0.7); ax.set_axisbelow(True)

# 1. 벤치마크별 절 점수 (T/O/R) — 분모(적용 항목 수)를 함께 표시
names = list(sc["per_benchmark"].keys())
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=True)
for ax, (sec, title) in zip(axes, (("T", "과제 타당성 T (분모 10 고정)"), ("O", "결과 타당성 O (분모 1–10 가변)"), ("R", "보고 R (분모 13 고정)"))):
    clean(ax, "x")
    vals = [sc["per_benchmark"][b][sec]["avg_pct"] for b in names]
    apps = [sc["per_benchmark"][b][sec]["applicable"] for b in names]
    ys = range(len(names))
    ax.barh(list(ys), vals, 0.6, color=BLUE if sec != "O" else ORANGE)
    for y, v, a in zip(ys, vals, apps):
        ax.text(v + 2, y, f"{v:.0f}% (n={a})", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, 140); ax.set_title(title, fontsize=9, color=INK, loc="left")
axes[0].set_yticks(list(range(len(names)))); axes[0].set_yticklabels(names, fontsize=8); axes[0].invert_yaxis()
fig.suptitle("부록 D 점수표에서 다시 계산한 절별 평균. O는 벤치마크마다 해당 항목 수가 달라 막대끼리 비교하기 어렵다", fontsize=9.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, "own_scores.png"), dpi=170); plt.close(fig)

# 2. 표 15(BIRD): 점추정과 논문이 낸 구간
b = cl["bird_table15"]
import re
txt = open(os.path.join(ROOT, "paper/paper.txt")).read().split("\n")[1984:2060]
pat = re.compile(r"^\s*(.+?)\s{2,}(\d+\.\d)\s+\[(\d+\.\d),\s*(\d+\.\d)\]\s+(\d+)\s+(\d+)-(\d+)\s*$")
rows = [tuple(m.groups()) for m in map(pat.match, txt) if m]
fig, ax = plt.subplots(figsize=(7.2, 4.2)); clean(ax, "both")
for i, r in enumerate(rows):
    acc, lo, hi = float(r[1]), float(r[2]), float(r[3])
    inside = lo <= acc <= hi
    ax.plot([i, i], [lo, hi], color=GREY, lw=2.2, zorder=1)
    ax.scatter([i], [acc], s=14, color=BLUE if inside else ORANGE, zorder=2)
ax.set_xlabel("논문 표 15의 행 (원래 순위 1–61)", color=INK); ax.set_ylabel("개발 세트 정확도 (%)", color=INK)
ax.set_title(f"점(점추정)이 회색 구간 밖인 행이 {b['acc_above_ci'] + b['acc_below_ci']}/61 (주황). 구간 중심이 점추정과 다르다", fontsize=9.2, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_bird_ci.png"), dpi=170); plt.close(fig)

# 3. 과대 추정 수치를 절대 오차(%p)로 나열: 분모를 아는 것만
items = [("τ-bench 항공 do-nothing", 38.0, "19/50 정확 일치"), ("τ-bench 항공 spamming", 40.0, "20/50 정확 일치"),
         ("OSWorld chrome (24→37/46)", 28.3, "수작업 13개, 전체로는 3.5%p"), ("WebArena do-nothing(N/A 과제)", 4.4, "36/812 정확 일치")]
fig, ax = plt.subplots(figsize=(7.2, 2.9)); clean(ax, "x")
for i, (n, v, note) in enumerate(items):
    ax.barh(i, v, 0.55, color=BLUE); ax.text(v + 0.8, i, f"{v:.1f}%p  ({note})", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=8.5); ax.invert_yaxis(); ax.set_xlim(0, 70)
ax.set_xlabel("절대 오차 (%p, 해당 벤치마크 안의 비율)", color=INK)
ax.set_title("분모를 확인할 수 있는 과대·과소 추정은 고정 과제 집합의 정확한 개수이고 표본 오차 대상이 아니다", fontsize=9, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_overestimates.png"), dpi=170); plt.close(fig)
print("ok")
