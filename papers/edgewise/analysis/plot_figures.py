"""우리가 그리는 그래프 네 장 -> report/figures/own_*.png. table_check.py, power_check.py, cost_model.py를 먼저 실행할 것.
실행: python3 analysis/plot_figures.py"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, str(HERE))
from paper_tables import *

pc = json.load(open(HERE / "power_check_results.json"))
C = {"ink": "#1b1b1a", "gray": "#8a8982", "blue": "#1f5fae", "orange": "#d9731a", "red": "#b3342b", "green": "#2e8b57"}

# 1. Self-Delegation - 바닐라 LLM: 8칸의 정확도 차와 짝지은 95% 구간(불일치율 0.2 가정)
rows = pc["self_vs_vanilla"]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
labs = [f'{r["setup"]} {r["bench"]} {r["metric"]}' for r in rows]
order = list(range(len(rows)))[::-1]
for yi, i in enumerate(order):
    r = rows[i]; lo, hi = r["disc0.2"]["ci95"]
    ax.plot([lo, hi], [yi, yi], color=C["gray"], lw=1.8)
    ax.plot(r["D_pp"], yi, "o", color=C["green"] if r["D_pp"] > 0 else (C["ink"] if r["D_pp"] == 0 else C["red"]), ms=6)
ax.axvline(0, color=C["ink"], lw=0.8); ax.axvspan(-5, 5, color=C["blue"], alpha=0.07)
ax.set_yticks(range(len(order))); ax.set_yticklabels([labs[i] for i in order], fontsize=8)
ax.set_xlabel("EdgeWise(Self) - 클라우드 단독 (정확도 %p). 막대: 짝지은 95% 구간(불일치율 0.2 가정). 옅은 띠: ±5%p")
ax.set_title("'같거나 낫다'의 구간: 8칸 모두 0을 포함하고, 대부분 -5%p도 포함", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_equiv.png", dpi=170); plt.close(fig)

# 2. 그림 1 판독값: 클라우드 노력 수준별 비용-정확도 (기준선 선택 민감도)
fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.3))
for ax, b, title in zip(axs, ("TB", "SWE"), ("Terminal-Bench 2.1 (Pass@3, n=89)", "SWE-ReBench (Pass@3, n=110)")):
    d = FIG1_READ[b]
    eff = ("low", "medium", "high", "xhigh")
    ax.plot([d[f"GPT {e}"][0] for e in eff], [d[f"GPT {e}"][1] for e in eff], "o-", color=C["blue"], label="GPT-5.5 단독 (노력 수준)")
    ax.plot([d[f"Self {e}"][0] for e in eff], [d[f"Self {e}"][1] for e in eff], "*-", color=C["green"], ms=10, label="EdgeWise Self (클라우드 노력 수준)")
    for e in eff:
        ax.annotate(e, (d[f"GPT {e}"][0], d[f"GPT {e}"][1]), textcoords="offset points", xytext=(4, -10), fontsize=7.5, color=C["blue"])
        ax.annotate(e, (d[f"Self {e}"][0], d[f"Self {e}"][1]), textcoords="offset points", xytext=(-4, 7), fontsize=7.5, color=C["green"])
    ax.set_xlabel("비용 ($, 3회 합계, 그림 1 판독)"); ax.set_ylabel("Pass@3 (%)"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.25)
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("기준선 선택: GPT-5.5 xhigh는 high에 지배됨. 헤드라인 절감(35%, 53%)은 xhigh 대비", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_effort.png", dpi=170); plt.close(fig)

# 3. 그림 4: 행동 유형별 중심화 상관과 95% 구간, Holm 보정 p
cr = pc["fig4_r"]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for yi, c in enumerate(cr[::-1]):
    r, lo, hi = FIG4_R[c["type"]]
    ax.plot([lo, hi], [yi, yi], color=C["gray"], lw=1.8)
    ax.plot(r, yi, "o", color=C["green"] if c["holm_p"] < 0.05 else C["ink"], ms=6)
    ax.annotate(f'Holm p={c["holm_p"]:.3f}', (hi + 0.01, yi), fontsize=7.5, va="center")
ax.axvline(0, color=C["ink"], lw=0.8)
ax.set_yticks(range(6)); ax.set_yticklabels([c["type"] for c in cr[::-1]], fontsize=8.5)
ax.set_xlim(-0.3, 0.45); ax.set_xlabel("군집 안 중심화 상관 r (논문 그림 4의 값을 다시 그림)")
ax.set_title("핵심 단계: 6개 유형 중 Write/Edit만 Holm 후에도 유의 (r=0.20, r²=4%)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_keystep.png", dpi=170); plt.close(fig)

# 4. 무작위 배치 K개 중 최댓값이 클라우드 단독 -1%p 이내일 확률 (참 격차 3%p와 5%p)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for key, col, lab in (("TB_n89_d0.15_gap3", C["orange"], "TB n=89, 참 격차 3%p"), ("TB_n89_d0.15_gap5", C["red"], "TB n=89, 참 격차 5%p"), ("SWE_n110_d0.09_gap3", C["blue"], "SWE n=110, 참 격차 3%p")):
    by = pc["random_alloc_selection"][key]["by_K"]
    Ks = sorted(int(k) for k in by)
    ax.plot(Ks, [by[str(k)]["P(max>=-1pp)"] for k in Ks], "o-", color=col, label=lab)
ax.set_xlabel("무작위 배치 수 K (각 3회 평균)"); ax.set_ylabel("최댓값이 클라우드 단독 -1%p 이내일 확률")
ax.set_title("'일부 배치는 전부 클라우드와 비견'은 K가 크면 노이즈만으로도 흔함", fontsize=10); ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "own_random_select.png", dpi=170); plt.close(fig)
print("saved", [p.name for p in FIG.glob("own_*.png")])
