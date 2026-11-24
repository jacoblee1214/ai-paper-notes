"""우리가 그리는 그래프 세 장(report/figures/own_*.png). 입력은 recompute_*_results.json와 repo/ 공개 출력.
실행: python3 -I analysis/plot_figures.py   (matplotlib; 한글 폰트는 Noto CJK -> Apple SD Gothic Neo 순)"""
import json, os, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/analysis")
import recompute_bbh as R

for cand in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if os.path.exists(cand):
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
OUT = f"{ROOT}/report/figures"; os.makedirs(OUT, exist_ok=True)
C_NC, C_COT, C_REF = "#7a7a74", "#1f5fae", "#b3261e"

bbh = json.load(open(f"{ROOT}/analysis/recompute_bbh_results.json"))
bbq = json.load(open(f"{ROOT}/analysis/recompute_bbq_results.json"))

# 1) 정확도 변화(편향 맥락 - 무편향 맥락), 편향-모순 정답 부분집합, paired 95% 구간
labels = [("suggested_answer|GPT-3.5|ZS", "제안 답\nGPT-3.5 ZS"), ("suggested_answer|GPT-3.5|FS", "제안 답\nGPT-3.5 FS"),
          ("suggested_answer|Claude 1.0|ZS", "제안 답\nClaude ZS"), ("suggested_answer|Claude 1.0|FS", "제안 답\nClaude FS"),
          ("ans_always_a|GPT-3.5|FS", "항상 A\nGPT-3.5 FS"), ("ans_always_a|Claude 1.0|FS", "항상 A\nClaude FS")]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = np.arange(len(labels))
for off, mode, col, name in [(-0.15, "nc", C_NC, "No-CoT"), (0.15, "cot", C_COT, "CoT")]:
    d = [bbh["table8"][k]["contradicting"][mode]["delta"] for k, _ in labels]
    h = [bbh["table8"][k]["contradicting"][mode]["delta_ci95_half"] for k, _ in labels]
    ax.errorbar(x + off, d, yerr=h, fmt="o", color=col, capsize=3, label=name)
    for xi, di in zip(x + off, d): ax.annotate(f"{di:.1f}", (xi, di), textcoords="offset points", xytext=(6, -3) if mode == "cot" else (-6, -3), ha="left" if mode == "cot" else "right", fontsize=8, color=col)
ax.axhline(0, color="#999", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([l for _, l in labels], fontsize=8)
ax.set_ylabel("정확도 변화(%p): 편향 맥락 - 무편향 맥락"); ax.legend(frameon=False, loc="upper left"); ax.set_ylim(-42, 4)
ax.set_title("편향-모순 정답 예시의 정확도 하락 (BBH 13과제, paired 95% 구간)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/own_accuracy_drop.png", dpi=200); plt.close(fig)

# 2) 과제별 CoT 정확도 변화(제안 답, zero-shot)
ex = R.load()
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
for ax, model in zip(axes, ["GPT-3.5", "Claude 1.0"]):
    rows = []
    for e in ex:
        if e["bias"] == "suggested_answer" and e["model"] == model and not e["fs"]:
            v = [r for r in e["rows"] if R.ok(r["cot_b"]) and R.ok(r["cot_u"]) and r["y"] != r["t"]]
            d = np.array([(r["cot_b"] == r["y"]) - (r["cot_u"] == r["y"]) for r in v], float)
            rows.append((e["task"], 100 * d.mean(), 196 * d.std(ddof=1) / math.sqrt(len(d))))
    rows.sort(key=lambda t: t[1])
    ax.errorbar([r[1] for r in rows], range(len(rows)), xerr=[r[2] for r in rows], fmt="o", color=C_COT, capsize=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.axvline(0, color="#999", lw=0.8); ax.set_title(f"{model}, 제안 답, zero-shot CoT", fontsize=9.5); ax.set_xlabel("정확도 변화 (%p)")
fig.tight_layout(); fig.savefig(f"{OUT}/own_task_forest.png", dpi=200); plt.close(fig)

# 3) BBQ: 비충실 쌍 중 고정관념 방향 비율과 Wilson 95% 구간 (50% = 편향 없음)
order = [k for k in bbq]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ys = []; 
for i, k in enumerate(order):
    for off, mode, col in [(-0.17, "nocot", C_NC), (0.17, "cot", C_COT)]:
        v = bbq[k][mode]; lo, hi = v["explained_wilson"]
        ax.plot([lo, hi], [i + off] * 2, color=col, lw=2); ax.plot(v["pct_explained_by_bias"], i + off, "o", color=col)
ax.axvline(50, color=C_REF, lw=1, ls="--"); ax.set_yticks(range(len(order))); ax.set_yticklabels([k.replace("|", " ") for k in order], fontsize=8)
ax.set_xlabel("비충실 쌍 중 고정관념 방향 비율 (%), Wilson 95% 구간"); ax.invert_yaxis()
ax.plot([], [], "o-", color=C_NC, label="No-CoT"); ax.plot([], [], "o-", color=C_COT, label="CoT"); ax.legend(frameon=False, loc="lower right")
ax.set_title("BBQ: 구간이 50%를 걸치면 고정관념 편향을 보이지 못한 설정", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/own_bbq_explained.png", dpi=200); plt.close(fig)
print("ok")
