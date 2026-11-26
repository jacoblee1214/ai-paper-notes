"""그래프 세 장(우리가 그림): own_fig4_ci.png, own_corr.png, own_mof.png -> ../report/figures/. matplotlib, 수 초."""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_tables as T
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for f in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, GRAY, RED = "#1f5fae", "#8a8984", "#b5472f"
S = json.loads((HERE / "stats_results.json").read_text())
FIG = json.loads((HERE / "figure_values.json").read_text())

# 1 그림 4 + Wilson
fig, ax = plt.subplots(figsize=(7.2, 3.8))
rows = S["S2_fig4_intervals"][::-1]
for i, r in enumerate(rows):
    lo, hi = r["wilson95"]
    ax.plot([lo, hi], [i, i], color=BLUE, lw=2)
    ax.plot(r["score"], i, "o", color=BLUE, ms=6)
    ax.text(hi + 1.2, i, f'{r["score"]}', va="center", fontsize=8.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["model"] for r in rows])
ax.axvline(25, color=RED, lw=1.2, ls="--"); ax.text(25.5, len(rows) - 0.4, "25% = 이미지를 안 보는 응답의 상한", color=RED, fontsize=8)
ax.set_xlabel("MMVP 쌍 점수 (%), 150쌍 Wilson 95% 구간"); ax.set_xlim(0, 58)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "own_fig4_ci.png", dpi=200); plt.close(fig)

# 2 상관
fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.8), sharex=True)
clip = np.array(FIG["clip"])
for ax, name in zip(axs, ["LLaVA-1.5", "InstructBLIP", "Gemini", "GPT-4"]):
    y = np.array(FIG[name]); ax.scatter(clip, y, s=18, color=BLUE)
    d = S["S4_clip_vs_mllm"][name]
    ax.set_title(f'{name}\nr={d["r"]} {d["ci95_fisher"]}\n(표 5: {d["table5_printed"]})', fontsize=8.5)
    ax.set_xlabel("CLIP 패턴 점수", fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("MLLM 패턴 점수", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "own_corr.png", dpi=200); plt.close(fig)

# 3 MoF
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
r = [x[0] for x in T.TABLE2]; m = [x[1] for x in T.TABLE2]; l = [x[2] for x in T.TABLE2]
ks = [round(v / 100 * 150) for v in m]
def wil(k, n=150, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)
lo = [m[i] - wil(ks[i])[0] for i in range(7)]; hi = [wil(ks[i])[1] - m[i] for i in range(7)]
axs[0].errorbar(r, m, yerr=[lo, hi], fmt="o-", color=BLUE, capsize=3)
axs[0].set_title("MMVP (150쌍, Wilson 95%)", fontsize=9); axs[0].set_xlabel("DINOv2 비율")
axs[1].plot(r, l, "s-", color=GRAY); axs[1].set_title("LLaVA 벤치마크 (지시 따르기, 단일 실행)", fontsize=9); axs[1].set_xlabel("DINOv2 비율")
for a in axs:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "own_mof.png", dpi=200); plt.close(fig)
print("ok")
