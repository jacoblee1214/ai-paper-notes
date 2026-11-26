"""우리가 그린 그림 세 장 (matplotlib). 입력: recompute_results.json, tables.json. 출력: report/figures/own_*.png
실행: python3 -I analysis/plot_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.load(open(HERE / "recompute_results.json"))
C = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
B12 = list(R["benchmarks"])

# 1. 설계 축별 평균 z 차이: 쌍별 점, 공식 대 재현 격차 띠
order = ["단일 단계 (그림 4)", "시각 인코더 미동결 (그림 5)", "인코더 (그림 6 왼쪽)", "전처리 letterbox->naive (그림 6 가운데)",
         "전처리 letterbox->resize&crop", "해상도 (그림 6 오른쪽)", "융합 DINOv2+SigLIP (그림 7 왼쪽)", "융합 DINOv2+CLIP (그림 7 왼쪽)",
         "LM 지시 튜닝 -> base (그림 7 오른쪽)", "더 좋은 LM (그림 12)", "언어 전용 공동 학습 제거 (그림 8)",
         "학습 시간 1->2 epoch (그림 10)", "데이터 추가 (그림 10)"]
short = ["다단계->단일 단계", "ViT 미동결(동결 대비)", "인코더(IN1K·DINOv2->CLIP·SigLIP)", "letterbox->naive(CLIP·SigLIP)", "letterbox->resize&crop",
         "해상도 224->336/384", "융합 SigLIP->DINO+SigLIP", "융합 CLIP->DINO+CLIP", "Vicuna->Llama-2(base)", "Llama-2·Vicuna->Mistral",
         "공동 학습 제거", "1->2 epoch", "데이터 +LVIS+LRV"]
fig, ax = plt.subplots(figsize=(8.4, 5.6))
nf = R["noise_floor_official_vs_repro"]
band = max(abs(nf["7B"]["mean_z_diff"]), abs(nf["13B"]["mean_z_diff"]))
ax.axvspan(-band, band, color="#cccccc", alpha=0.5, label=f"공식 대 재현 LLaVA의 평균 z 차이 (|{nf['7B']['mean_z_diff']:+.2f}|, |{nf['13B']['mean_z_diff']:+.2f}|)")
for i, (k, lab) in enumerate(zip(order, short)):
    v = R["axes"][k]
    ys = len(order) - 1 - i
    ax.scatter(v["pair_level_mean_z"], [ys] * len(v["pair_level_mean_z"]), color=C[0], s=22, zorder=3)
    ax.scatter([v["mean_z_diff"]], [ys], marker="D", color=C[3], s=34, zorder=4)
    ax.text(ax.get_xlim()[1] if False else 1.9, ys, f"+{v['pos']}/-{v['neg']}", va="center", fontsize=7.5, color="#444")
ax.set_yticks(range(len(order))); ax.set_yticklabels(short[::-1], fontsize=8.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-1.4, 2.2); ax.set_xlabel("12개 벤치마크 평균 z 차이 (대안 - 기준). 파란 점: 쌍별, 붉은 마름모: 축 평균")
ax.text(1.9, len(order) - 0.35, "벤치마크 부호", fontsize=7.5, ha="left", color="#444")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.13))
ax.set_title("설계 축별 효과의 크기: 작은 축은 재현 격차 안에 든다")
fig.tight_layout(); fig.savefig(FIG / "own_axis_effects.png", dpi=150); plt.close(fig)

# 2. 벤치마크별 차이 / MDE (q=0.2): letterbox->naive, 해상도, 융합, 지시 튜닝->base
sel = [("전처리 letterbox->naive (융합 포함, 4쌍)", "letterbox->naive"), ("해상도 (그림 6 오른쪽)", "해상도 224->336/384"),
       ("융합 DINOv2+SigLIP (그림 7 왼쪽)", "융합 SigLIP->DINO+SigLIP"), ("LM 지시 튜닝 -> base (그림 7 오른쪽)", "Vicuna->Llama-2")]
mde = np.array([R["benchmarks"][b]["mde_pts"]["q=0.2"] for b in B12])
fig, axs = plt.subplots(1, 4, figsize=(11.5, 5), sharey=True)
for ax, (k, lab) in zip(axs, sel):
    pp = R["axes"][k]["per_pair"]
    m = np.array([[p["delta_pts"][b] for b in B12] for p in pp]) / mde
    mean = np.nanmean(m, 0)
    ax.barh(range(12), mean[::-1], color=[C[2] if x > 1 else (C[3] if x < -1 else "#bbbbbb") for x in mean[::-1]])
    ax.axvline(1, color="gray", ls="--", lw=0.8); ax.axvline(-1, color="gray", ls="--", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel("차이 / MDE(q=0.2)\n(쌍 평균)", fontsize=8)
    ax.set_yticks(range(12)); ax.set_yticklabels(B12[::-1], fontsize=8)
    ax.set_xlim(-8, 10)
fig.suptitle("벤치마크별로 보면 차이가 표본 오차(MDE) 밖인 칸은 일부뿐이다 (회색: |x|<1)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_delta_over_mde.png", dpi=150); plt.close(fig)

# 3. 공유 학습 노이즈가 있을 때 논문식 검정의 거짓 양성률
mc = R["mc_false_positive_rate_alpha0.01"]
taus = [0.0, 0.05, 0.10, 0.20, 0.30]
fig, ax = plt.subplots(figsize=(6.4, 4))
for k, c in zip((1, 2, 4), C):
    ax.plot(taus, [mc[f"k={k},tau={t}"] for t in taus], "o-", color=c, label=f"쌍 {k}개 x 벤치마크 12개")
ax.axhline(0.01, color="gray", ls="--", lw=1); ax.text(0.3, 0.014, "명목 0.01", ha="right", fontsize=8, color="gray")
ax.set_xlabel("모델 쌍마다 공유되는 학습 노이즈 표준편차 (z 단위)"); ax.set_ylabel("귀무에서 p<0.01 비율")
ax.set_title("모든 쌍x벤치마크를 독립으로 치는 검정은\n공유 노이즈가 조금만 있어도 거짓 양성이 크게 늘어난다", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "own_false_positive.png", dpi=150); plt.close(fig)
print("wrote 3 figures")
