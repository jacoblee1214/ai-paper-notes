"""toy_results.json -> figures/own_power_vs_dup.png, figures/own_sensitivity.png (우리가 그린 그래프)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "analysis" / "toy_results.json").read_text())
(ROOT / "figures").mkdir(exist_ok=True)

ks = [0, 1, 2, 4, 8, 16]
names = [("sharded", "샤드 t 검정 (r=10, m=25)", "#1f77b4", "o"), ("perm25", "순열 검정 M=25 (같은 계산량)", "#d62728", "s"),
         ("perm100", "순열 검정 M=100 (4배 계산량)", "#ff7f0e", "^"), ("perm_z", "순열 + 정규 근사 (대조, M=25)", "#2ca02c", "d")]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for key, lab, c, mk in names:
    ax[0].plot(range(len(ks)), [d["E2_power"][str(k)][key]["rej01"] for k in ks], marker=mk, color=c, label=lab)
    ax[1].plot(range(len(ks)), [d["E2_power"][str(k)][key]["median_log10p"] for k in ks], marker=mk, color=c, label=lab)
ax[0].axhline(0.01, color="gray", ls=":", lw=1)
for a in ax:
    a.set_xticks(range(len(ks)))
    a.set_xticklabels(ks)
    a.set_xlabel("정준 순서로 학습에 넣은 횟수 k")
ax[0].set_ylabel("기각률 (α = 0.01)")
ax[0].set_title("검정력 (시행 200회)")
ax[1].set_ylabel("p값 중앙값 (log10)")
ax[1].set_title("p값 크기")
ax[0].legend(fontsize=7.5, loc="lower right")
fig.text(0.5, -0.02, "토이 4-gram LM, 문항 200개, 고차 n-gram 보존율 0.01 (논문 값 아님). 순열 M=25의 p 하한은 1/26 = 0.038이라 α=0.01에서 기각할 수 없다.",
         ha="center", fontsize=7.5)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "own_power_vs_dup.png", dpi=160, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
sv, pv = d["E4_sensitivity"]["vs_shards"], d["E4_sensitivity"]["vs_perms"]
ax[0].plot(list(sv), [v["rej05"] for v in sv.values()], "o-", label="α = 0.05")
ax[0].plot(list(sv), [v["rej01"] for v in sv.values()], "s-", label="α = 0.01")
ax[0].set_xlabel("샤드 수 r (문항 200개, m = 25)")
ax[0].set_ylabel("기각률 (k=4, 시행 100회)")
ax[0].legend(fontsize=8)
ax[1].plot(list(pv), [v["rej05"] for v in pv.values()], "o-", label="α = 0.05")
ax[1].plot(list(pv), [v["rej01"] for v in pv.values()], "s-", label="α = 0.01")
ax[1].set_xlabel("샤드당 순열 수 m (r = 10)")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "own_sensitivity.png", dpi=160, bbox_inches="tight")
print("ok")
