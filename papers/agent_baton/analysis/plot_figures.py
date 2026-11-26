"""우리가 그리는 그래프 네 장 -> report/figures/own_*.png. table_check.py, power_check.py, cost_model.py를 먼저 실행할 것.
실행: python3 analysis/plot_figures.py"""
import json
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
import sys
sys.path.insert(0, str(HERE))
from paper_tables import *

tc = json.load(open(HERE / "table_check_results.json"))
pc = json.load(open(HERE / "power_check_results.json"))
cm = json.load(open(HERE / "cost_model_results.json"))
C = {"ink": "#1b1b1a", "gray": "#8a8982", "blue": "#1f5fae", "orange": "#d9731a", "red": "#b3342b", "green": "#2e8b57"}

# 1. 22개 설정의 정확도 차이와 짝지은 95% 구간(불일치율 0.2 가정), 0 기준선과 MDE 띠
rows = pc["rows22"]
fig, ax = plt.subplots(figsize=(7.6, 6.2))
labels = [f'{r["pair"].replace("->", "→")} ({r["bench"]})' for r in rows]
order = np.argsort([r["D_pp"] for r in rows])
for yi, i in enumerate(order):
    r = rows[i]; lo, hi = r["disc0.2"]["ci95"]
    col = C["green"] if r["D_pp"] > 0 else C["red"]
    ax.plot([lo, hi], [yi, yi], color=C["gray"], lw=1.6)
    ax.plot(r["D_pp"], yi, "o", color=col, ms=5)
ax.axvline(0, color=C["ink"], lw=0.8)
ax.axvspan(-5, 5, color=C["blue"], alpha=0.07)
ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[i] for i in order], fontsize=7.5)
ax.set_xlabel("AgentBaton - 강한 모델 단독 (정확도 %p). 막대: 짝지은 95% 구간(불일치율 0.2 가정). 옅은 띠: ±5%p")
ax.set_title("22개 설정의 정확도 차이와 구간: 구간 폭이 ±10%p 안팎이라 대부분 0과 -5%p를 함께 포함", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_delta_ci.png", dpi=170); plt.close(fig)

# 2. Table 3 비용-정확도 (두 벤치마크) + Wilson 구간
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6))
for ax, b, title in zip(axs, ("DS", "TB"), ("DeepSWE v1.1 (n=113)", "Terminal-Bench 2.1 (n=89)")):
    n = N[b]
    def wil(k, n, z=1.96):
        p = k / n; d = 1 + z * z / n; c0 = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return 100 * (c0 - h), 100 * (c0 + h)
    for m, v in BASE.items():
        a, c = v[b]; lo, hi = wil(round(a * n / 100), n)
        ax.errorbar(max(c, 0.02), a, yerr=[[a - lo], [hi - a]], fmt="s", color=C["ink"], ms=5, capsize=2, lw=0.8)
        ax.annotate(m, (max(c, 0.02), a), textcoords="offset points", xytext=(4, 4), fontsize=7.5)
    for key, d in AB.items():
        a, c = d[b]; lo, hi = wil(round(a * n / 100), n)
        col = C["blue"] if key[1] == "Qwen" else C["orange"]
        ax.errorbar(c, a, yerr=[[a - lo], [hi - a]], fmt="o", color=col, ms=5, capsize=2, lw=0.8, alpha=0.9)
        ax.annotate(key[0] if key[0] != "Gem" else "Gem", (c, a), textcoords="offset points", xytext=(4, -9), fontsize=7, color=col)
    ax.set_xscale("log"); ax.set_xlabel("작업당 평균 비용 ($, 로그)"); ax.set_ylabel("정확도 (%) 와 Wilson 95% 구간"); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
axs[0].plot([], [], "s", color=C["ink"], label="단일 모델"); axs[0].plot([], [], "o", color=C["blue"], label="AgentBaton → Qwen"); axs[0].plot([], [], "o", color=C["orange"], label="AgentBaton → Luna")
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Table 3 재정리: 구간이 서로 크게 겹치고, 일부 인계 점은 Luna 단독에 지배됨", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_pareto.png", dpi=170); plt.close(fig)

# 3. Table 1: 방향·시점·전달 방식별 평균 정확도 차(capable 단독 대비)와 비용 변화
t1 = tc["table1_vs_capable_only"]
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.9))
groups = [("E->C", 30, "in"), ("E->C", 30, "log"), ("E->C", 75, "in"), ("E->C", 75, "log"), ("C->E", 30, "in"), ("C->E", 30, "log"), ("C->E", 75, "in"), ("C->E", 75, "log")]
dm, cmn = [], []
for g in groups:
    r = [x for x in t1 if (x["dir"], x["pct"], x["transfer"]) == g]
    dm.append(np.mean([x["d_pp"] for x in r])); cmn.append(np.mean([x["cost_change_pct"] for x in r]))
names = [f'{"효율→강" if g[0]=="E->C" else "강→효율"}\n{g[1]}% {"문맥" if g[2]=="in" else "로그"}' for g in groups]
cols = [C["orange"] if g[0] == "E->C" else C["blue"] for g in groups]
axs[0].bar(range(8), dm, color=cols); axs[0].axhline(0, color=C["ink"], lw=0.8)
axs[0].set_xticks(range(8)); axs[0].set_xticklabels(names, fontsize=6.8); axs[0].set_ylabel("정확도 차 (%p, 6칸 평균)")
axs[0].set_title("강한 모델 단독 대비 정확도", fontsize=9.5)
axs[1].bar(range(8), cmn, color=cols); axs[1].axhline(0, color=C["ink"], lw=0.8)
axs[1].set_xticks(range(8)); axs[1].set_xticklabels(names, fontsize=6.8); axs[1].set_ylabel("비용 변화 (%, 6칸 평균)")
axs[1].set_title("강한 모델 단독 대비 비용", fontsize=9.5)
fig.suptitle("Table 1 재정리: 정확도 손실은 두 방향이 비슷하고 차이는 비용에서 남", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_direction.png", dpi=170); plt.close(fig)

# 4. 비용 절감 민감도: Qwen에 값을 매겼을 때와 탐지기 비용
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.8))
cs = [0, 0.1, 0.25, 0.5, 1.0]
sens = cm["qwen_price_sensitivity"]
axs[0].plot(cs, [sens[f"c={c}"]["avg_qwen_rows_10"] for c in cs], "o-", color=C["blue"], label="Qwen 행 10칸 평균")
axs[0].plot(cs, [sens[f"c={c}"]["pooled_22_luna_src_unchanged"] for c in cs], "s-", color=C["orange"], label="22칸 평균(65.9%는 c=0)")
axs[0].set_xlabel("Qwen 비용 / Luna 인계 후 비용 (c)"); axs[0].set_ylabel("평균 비용 절감 (%)"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.25)
axs[0].set_title("로컬 모델을 공짜로 두는 가정", fontsize=9.5)
by = cm["reduction_by_row"]
vals = sorted(by.values())
axs[1].barh(range(len(vals)), vals, color=C["gray"]); axs[1].axvline(65.9, color=C["red"], lw=1.2)
axs[1].set_yticks([]); axs[1].set_xlabel("22칸별 비용 절감 (%, 붉은 선: 평균 65.9)"); axs[1].set_title("22칸의 분포: 42.9–84.5%", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "own_cost.png", dpi=170); plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
