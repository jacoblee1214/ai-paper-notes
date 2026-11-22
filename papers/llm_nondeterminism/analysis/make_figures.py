"""우리가 그린 그래프 세 장. analysis/*_results.json을 읽는다. 실행: python3 -I analysis/make_figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "report", "figures")
os.makedirs(OUT, exist_ok=True)
for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

rec = json.load(open(os.path.join(HERE, "recompute_results.json")))
od = json.load(open(os.path.join(HERE, "overdispersion_results.json")))
MODELS = ["gpt-35-turbo", "gpt-4o", "llama3-8b", "llama3-70b", "mixtral-8x7b"]
COL = dict(zip(MODELS, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]))

# 그림 1: TARr@N 곡선 (조건별 중앙값, 모델별)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for m in MODELS:
    rs = [r for r in rec if r["model"] == m and r["N"] == 10]
    xs = list(range(2, 11))
    ys = [np.median([100 * r["tacr_by_N"][str(k)] / r["n_questions"] for r in rs]) for k in xs]
    ax.plot(xs, ys, marker="o", ms=3, label=m, color=COL[m])
ax.set_xlabel("N (앞에서부터 쓴 실행 수)"); ax.set_ylabel("TARr@N 중앙값 (%)")
ax.set_title("실행 수가 늘수록 원출력 일치율은 떨어진다\n(조건 114개의 중앙값, 공개 원출력에서 재계산)", fontsize=9)
ax.legend(fontsize=7.5); ax.set_ylim(0, 100)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "own_tarr_at_n.png"), dpi=170); plt.close()

# 그림 2: 실행 간 정확도 표준편차 대 독립 이항 기준
rows = od["rows"]
fig, ax = plt.subplots(figsize=(5.4, 4.2))
for m in MODELS:
    s = [r for r in rows if r["model"] == m]
    ax.scatter([r["sd_binom_pp"] for r in s], [r["sd_pp"] for r in s], s=14, color=COL[m], label=m, alpha=.8)
mx = max(r["sd_binom_pp"] for r in rows) * 1.05
ax.plot([0, mx], [0, mx], "k--", lw=.8)
ax.set_xlabel("독립 이항 기준 표준편차 √(p(1-p)/n) (%p)"); ax.set_ylabel("실행 간 정확도 표준편차 (%p)")
ax.set_title("관측된 실행 간 변동은 같은 정확도 독립 이항의 표본 오차보다 작다\n(점선 y=x, 조건 114개)", fontsize=8.5)
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "own_dispersion.png"), dpi=170); plt.close()

# 그림 3: 같은 조건 두 배치의 TARr
pairs = od["pairs"]
fig, ax = plt.subplots(figsize=(5.2, 4.2))
for m in MODELS:
    s = [p for p in pairs if p["model"] == m]
    ax.scatter([p["tarr_a"] for p in s], [p["tarr_b"] for p in s], s=18, color=COL[m], label=m)
ax.plot([0, 100], [0, 100], "k--", lw=.8)
ax.set_xlabel("먼저 돌린 배치(top-p 0.0)의 TARr@10 (%)"); ax.set_ylabel("나중 배치(top-p 1.0)의 TARr@10 (%)")
ax.set_title("같은 모델·과제·shot을 다시 돌리면 TARr이 달라진다\n(33쌍, top-p와 날짜가 함께 다름)", fontsize=8.5)
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "own_batch_drift.png"), dpi=170); plt.close()
print("ok")
