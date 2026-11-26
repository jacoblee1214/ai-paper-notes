"""우리가 그린 그래프 네 장을 report/figures/own_*.png 로 저장한다 (matplotlib).

own_mask.png        prefix-LM 마스크(논문 그림 2 배열)와 π0 3블록 마스크 (mask_results.json)
own_res_z.png       표 1 224→448 차이를 재실행 SD로 나눈 z (stats_results.json)
own_mask_ablation.png  마스크·손실 절제의 과제별 차이(그림 17 판독)와 평균 (stats_results.json, figure_values.json)
own_tokens.png      학습 토큰 보고값 대 산술 (compute_results.json)
먼저 mask_toy.py, stats_check.py, compute_check.py를 실행해 둘 것.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for cand in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GRAY, RED = "#1f5fae", "#d9822b", "#8a8a85", "#b23a48"

M = json.loads((HERE / "mask_results.json").read_text(encoding="utf-8"))
S = json.loads((HERE / "stats_results.json").read_text(encoding="utf-8"))
C = json.loads((HERE / "compute_results.json").read_text(encoding="utf-8"))
F = json.loads((HERE / "figure_values.json").read_text(encoding="utf-8"))

# ---- 1. 마스크
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [11, 11]})
m2 = np.array(M["T2_fig2_matrix"]["attend_rows_over_cols"])
names = M["T2_fig2_matrix"]["tokens"]
ax[0].imshow(m2, cmap="Blues", vmin=0, vmax=1.6)
ax[0].set_xticks(range(len(names)), names, rotation=90, fontsize=8)
ax[0].set_yticks(range(len(names)), names, fontsize=8)
ax[0].set_title("PaliGemma prefix-LM (그림 2 배열, 코드 식으로 계산)", fontsize=9)
ax[0].axhline(6.5, color=RED, lw=1); ax[0].axvline(6.5, color=RED, lw=1)
p = np.array(M["T6_pi0_mask_from_same_function"]["matrix"])
lab = ["img"] * 4 + ["lang"] * 3 + ["state"] + ["a1", "a2", "a3", "a4"]
ax[1].imshow(p, cmap="Blues", vmin=0, vmax=1.6)
ax[1].set_xticks(range(len(lab)), lab, rotation=90, fontsize=8)
ax[1].set_yticks(range(len(lab)), lab, fontsize=8)
ax[1].set_title("π0 3블록 (같은 함수, ar 배열만 다름)", fontsize=9)
for a in ax:
    a.set_xlabel("키(열)", fontsize=8); a.set_ylabel("질의(행)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "own_mask.png", dpi=170); plt.close(fig)

# ---- 2. z
rows = [r for r in S["S1_table1_res_delta"]["rows"] if r.get("z_rerun") is not None]
rows.sort(key=lambda r: r["z_rerun"])
fig, ax = plt.subplots(figsize=(7.5, 8.5))
y = np.arange(len(rows))
ax.barh(y, [r["z_rerun"] for r in rows], color=[ORANGE if r["sensitive"] else BLUE for r in rows], height=0.7)
ax.set_yticks(y, [r["task"] for r in rows], fontsize=7)
for x in (-3, 3):
    ax.axvline(x, color=GRAY, lw=0.8, ls="--")
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("z = (448-224) / √(SD224²+SD448²)  (재실행 SD 기준, symlog)")
ax.set_title("표 1: 224→448 차이는 재실행 변동의 몇 배인가 (주황=표 1 음영 9과제)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "own_res_z.png", dpi=170); plt.close(fig)

# ---- 3. 마스크 절제
f17 = F["fig17_per_task"]
keys = ["AR_pref_loss_suf", "AR_pref_loss_pref", "AR_all_loss_suf", "AR_all_loss_pref"]
labs = ["접두 AR\n접미 손실", "접두 AR\n접두·접미 손실", "이미지+접두 AR\n접미 손실", "이미지+접두 AR\n접두·접미 손실"]
base = np.array([t["AR_suf_loss_suf"] for t in f17])
fig, ax = plt.subplots(figsize=(8.5, 4.6))
rng = np.random.default_rng(1)
for i, k in enumerate(keys):
    d = np.array([t[k] for t in f17]) - base
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=14, color=GRAY, alpha=0.7, zorder=2)
    b = S["S5_fig17"]["bootstrap_over_tasks"][k]
    ax.errorbar(i, b["mean_delta"], yerr=[[b["mean_delta"] - b["boot95"][0]], [b["boot95"][1] - b["mean_delta"]]],
                fmt="o", color=RED, capsize=4, zorder=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(4), labs, fontsize=8)
ax.set_ylabel("prefix-LM 대비 점수 차 (36과제, 점=과제)")
ax.set_title("마스크·손실 절제: 빨강=과제 평균과 과제 부트스트랩 95% 구간 (그림 17 판독)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "own_mask_ablation.png", dpi=170); plt.close(fig)

# ---- 4. 토큰
T = C["tokens"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
labels = ["Stage1\n패딩 포함 384B", "Stage1\n논문 <350B", "Stage2\n패딩 포함", "Stage2\n이미지 토큰만", "Stage2\n논문 ~90B"]
vals = [T["stage1_padded_tokens"] / 1e9, T["stage1_reported"] / 1e9, T["stage2_padded_tokens"] / 1e9,
        T["stage2_image_tokens_only"] / 1e9, T["stage2_reported"] / 1e9]
cols = [BLUE, ORANGE, BLUE, BLUE, ORANGE]
ax.bar(range(5), vals, color=cols, width=0.65)
for i, v in enumerate(vals):
    ax.text(i, v + 5, f"{v:.0f}B", ha="center", fontsize=8)
ax.set_xticks(range(5), labels, fontsize=8)
ax.set_ylabel("토큰 수 (10억)")
ax.set_title("학습 토큰: 산술(파랑) 대 논문 보고값(주황). Stage2 보고값은 이미지 토큰만보다 작다", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "own_tokens.png", dpi=170); plt.close(fig)
print("saved", [p.name for p in sorted(OUT.glob("own_*.png"))])
