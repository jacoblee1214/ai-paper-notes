"""재계산 결과로 그래프 세 장을 그립니다 (우리가 그린 그래프. 논문 그림은 쓰지 않음).
입력: figure_values.json, recompute_results.json. 출력: ../report/figures/own_*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures")
for f in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open(os.path.join(HERE, "recompute_results.json")))
F = json.load(open(os.path.join(HERE, "figure_values.json")))
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#999999"
MODELS = ["GPT-3.5", "GPT-4", "Claude 1.3", "Claude 2", "LLaMA 2"]
DS = ["TruthfulQA MC", "AQuA", "MATH (CoT)", "MMLU (CoT)", "TruthfulQA freeform", "TriviaQA"]

# 1. "확실해요?" 뒤 정답->오답, 오답->정답 (전체 문항 대비 %)
fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
for ax, d in zip(axs.flat, DS):
    n = 0.4
    xs = range(5)
    c2i = [R["are_you_sure"]["cells"][f"{d}|{m}"]["c_to_i"] / R["are_you_sure"]["cells"][f"{d}|{m}"]["n"] * 100 for m in MODELS]
    i2c = [R["are_you_sure"]["cells"][f"{d}|{m}"]["i_to_c"] / R["are_you_sure"]["cells"][f"{d}|{m}"]["n"] * 100 for m in MODELS]
    ax.bar([x - n / 2 for x in xs], c2i, n, color=VERM, label="정답 → 오답")
    ax.bar([x + n / 2 for x in xs], i2c, n, color=BLUE, label="오답 → 정답")
    for x, m in zip(xs, MODELS):
        p = R["are_you_sure"]["cells"][f"{d}|{m}"]["mcnemar_exact_p"]
        ax.text(x, max(c2i[x], i2c[x]) + 1.5, "*" if p < 0.05 else "n.s.", ha="center", fontsize=8, color="#333")
    ax.set_title(f"{d} (n={R['are_you_sure']['cells'][d + '|GPT-4']['n']})", fontsize=10)
    ax.set_xticks(list(xs)); ax.set_xticklabels(MODELS, rotation=30, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axs[0][0].set_ylabel("문항 대비 %")
axs[1][0].set_ylabel("문항 대비 %")
axs[0][0].legend(fontsize=8, frameon=False)
fig.suptitle("'확실해요?' 뒤 답 전환: 그림 17에서 복원한 비율 (* McNemar 정확 p<0.05, 근사)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "own_flip_churn.png"), dpi=160); plt.close(fig)

# 2. 사람 선호 회귀 특징 효과 (그림 5 복원)
feats = R["preference_regression"]["features"]
fig, ax = plt.subplots(figsize=(6.4, 6.4))
for i, r in enumerate(feats):
    y = len(feats) - i
    col = VERM if r["feature"] == "Matches user's beliefs" else (BLUE if r["ci95_excludes_50"] else GREY)
    ax.plot(r["ci95"], [y, y], color=col, lw=1.6)
    ax.plot([r["median"]], [y], "o", color=col, ms=4)
ax.axvline(50, color="#333", lw=0.8, ls="--")
ax.set_yticks([len(feats) - i for i in range(len(feats))]); ax.set_yticklabels([r["feature"] for r in feats], fontsize=7.5)
ax.set_xlabel("특징이 있는 응답이 선호될 확률 (%, 다른 조건 동일)")
ax.set_title("그림 5 복원: 사후 중앙값과 95% 신용 구간 (회색: 구간이 50%를 포함)", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_feature_effects.png"), dpi=160); plt.close(fig)

# 3. 시 출처 흉내: 네 결과 (n=300 격자에서 복원한 개수)
fig, ax = plt.subplots(figsize=(7, 3.8))
mods = ["Claude 1.3", "Claude 2", "GPT-3.5", "GPT-4", "LLaMA 2"]
keys = [("only_incorrect", "틀린 출처만 언급 (흉내)", VERM), ("b_both", "둘 다 언급", "#E69F00"), ("c_only_correct", "맞는 출처만 언급", BLUE), ("a_neither", "출처 언급 없음", GREY)]
left = [0] * 5
for k, lab, col in keys:
    vals = [R["mimicry"]["counts"][m][k] / 3 for m in mods]
    ax.barh(mods, vals, left=left, color=col, label=lab)
    left = [a + b for a, b in zip(left, vals)]
ax.invert_yaxis(); ax.set_xlabel("300개 프롬프트 중 %")
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("흉내 과제: 응답 결과 구성 (그림 4·12에서 복원)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_mimicry_outcomes.png"), dpi=160); plt.close(fig)
print("ok")
