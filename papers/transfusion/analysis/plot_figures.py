"""우리가 그린 그래프 네 장을 report/figures/own_*.png 로 저장한다 (matplotlib).

own_mask.png       Transfusion 마스크(캡션 먼저 배치와 이미지 둘 배치)와 토이의 ε-MSE (mask_loss_results.json)
own_parity_ci.png  동등 FLOP 비율: 논문 표 3, 재계산, 세 가지 95% 구간 (scaling_results.json)
own_fid_refit.png  그림 5 FID 패널의 점을 읽어 다시 맞춘 선과 Chameleon 7B 수준 (figure_values.json, scaling_results.json)
own_effects.png    절제 효과를 그림 5 산포 단위로 환산 (ablation_results.json)
먼저 figure_extract.py, scaling_refit.py, ablation_check.py, mask_loss_toy.py를 실행해 둘 것.
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

R = json.loads((HERE / "scaling_results.json").read_text())
F = json.loads((HERE / "figure_values.json").read_text())
A = json.loads((HERE / "ablation_results.json").read_text())
M = json.loads((HERE / "mask_loss_results.json").read_text())


def mask_from(toks, blk):
    n = len(toks)
    is_img = [t == "p" for t in toks]
    return np.array([[(j <= i) or (is_img[i] and is_img[j] and blk[i] == blk[j]) for j in range(n)] for i in range(n)])


# ---- 1. 마스크와 토이
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.05, 1.05, 1.0]})
toksA = ["BOS", "cap", "BOI"] + ["p"] * 8 + ["EOI", "END"]
mA = mask_from(toksA, [0] * 13)
ax[0].imshow(mA, cmap="Blues", vmin=0, vmax=1.6)
ax[0].set_xticks(range(13), toksA, rotation=90, fontsize=7); ax[0].set_yticks(range(13), toksA, fontsize=7)
ax[0].set_title("캡션 먼저 배치의 마스크\n(이미지 패치 8개는 서로 양방향)", fontsize=9)
toks2 = ["t", "BOI", "p", "p", "p", "EOI", "t", "BOI", "p", "p", "p", "EOI", "t"]
blk2 = [-1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, -1, -1]
m2 = mask_from(toks2, blk2)
ax[1].imshow(m2, cmap="Blues", vmin=0, vmax=1.6)
ax[1].set_xticks(range(13), toks2, rotation=90, fontsize=7); ax[1].set_yticks(range(13), toks2, fontsize=7)
ax[1].set_title("이미지 둘: 첫 이미지는 둘째를 못 보고\n둘째는 첫째를 전부 봄", fontsize=9)
S = M["T4_mean_sd_over_3_seeds"]
names = {"bidirectional": "양방향", "causal_only": "인과만", "bidirectional_noise_limit500": "양방향+노이즈 제한"}
keys = [("eps_mse_A_t500", "캡션 먼저\nt=500"), ("eps_mse_B_t500", "이미지 먼저\nt=500"), ("eps_mse_B_t900", "이미지 먼저\nt=900")]
w = 0.26
for i, (k, c) in enumerate(zip(S, (BLUE, RED, ORANGE))):
    ys = [S[k][kk][0] for kk, _ in keys]; es = [S[k][kk][1] for kk, _ in keys]
    ax[2].bar(np.arange(3) + (i - 1) * w, ys, w, yerr=es, color=c, label=names[k], capsize=2)
ax[2].set_xticks(range(3), [l for _, l in keys], fontsize=8)
ax[2].set_ylabel("ε-MSE (3시드, 평균 ± SD)", fontsize=8); ax[2].legend(fontsize=7)
ax[2].set_title("토이: 이미지 먼저일 때 인과 마스크의 손해", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "own_mask.png", dpi=150); plt.close()

# ---- 2. 동등 FLOP 비율 구간
labels = {"c4_ppl": "C4 PPL", "wiki_ppl": "Wiki PPL", "llama_acc": "Llama 정확도", "coco_cider": "CIDEr", "coco_fid": "FID", "coco_clip": "CLIP"}
fig, ax = plt.subplots(figsize=(8.4, 4.4))
for i, (k, lab) in enumerate(labels.items()):
    v = R["panels"][k]
    y = len(labels) - i
    for off, (m, c) in zip((0.22, 0.0, -0.22), (("point", BLUE), ("resid", GRAY), ("cluster", ORANGE))):
        if m in v["ratio_95ci"]:
            lo, med, hi = v["ratio_95ci"][m]["A"]
            ax.plot([lo, hi], [y + off] * 2, color=c, lw=3, solid_capstyle="butt", label={"point": "점 부트스트랩", "resid": "잔차 부트스트랩", "cluster": "모델 크기 군집 부트스트랩"}[m] if i == 1 else None)
    ax.plot(v["paper_parity_ratio"], y, "D", color=RED, ms=7, label="논문 표 3" if i == 0 else None)
    ax.plot(v["ratio_A_nominal_X_2.1e22"], y, "o", color="black", ms=4, label="재계산(명목 7B)" if i == 0 else None)
ax.axvline(1, color="black", lw=0.8, ls=":"); ax.axvline(1 / 3, color=GRAY, lw=0.8, ls="--")
ax.text(1 / 3, 0.35, "1/3", ha="center", fontsize=8, color=GRAY)
ax.set_xscale("log"); ax.set_xlim(0.002, 8)
ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], ["0.003", "0.01", "0.03", "0.1", "0.3", "1", "3"]); ax.minorticks_off()
ax.set_yticks(range(len(labels), 0, -1), list(labels.values()))
ax.set_xlabel("동등 FLOP 비율 (Chameleon 7B와 같은 지표에 필요한 Transfusion FLOPs ÷ Chameleon FLOPs), 95% 구간", fontsize=8)
ax.legend(fontsize=7, loc="upper left", ncol=2); ax.set_title("표 3 동등 FLOP 비율의 재계산과 불확실성", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "own_parity_ci.png", dpi=150); plt.close()

# ---- 3. FID 패널 재적합
fig, ax = plt.subplots(figsize=(6.8, 4.4))
P = F["coco_fid"]["points"]
for who, col, cf in (("Transfusion", RED, R["panels"]["coco_fid"]["coef_lnln"]["Transfusion"]), ("Chameleon", BLUE, R["panels"]["coco_fid"]["coef_lnln"]["Chameleon"])):
    x = np.array([p[0] for p in P[who]]); y = np.array([p[1] for p in P[who]])
    ax.scatter(10 ** x, y, s=14, color=col, label=f"{who} (점 {len(x)}개)")
    xs = np.linspace(x.min(), 22.4, 50)
    ax.plot(10 ** xs, np.exp(cf[0] + cf[1] * xs * np.log(10)), color=col, lw=1)
ax.axhline(29.6, color=GRAY, ls="--", lw=0.8)
need = R["panels"]["coco_fid"]["ratio_A_actual_C7B"] * R["X7_flops_6ND"]
ax.axvline(need, color=GRAY, ls=":", lw=0.8)
ax.annotate(f"Transfusion 선이 FID 29.6에 닿는 FLOPs {need:.1e}\n(Chameleon 7B {R['X7_flops_6ND']:.1e}의 {R['panels']['coco_fid']['ratio_A_actual_C7B']:.3f}배)", xy=(need, 29.6), xytext=(1.5e20, 100), fontsize=7, arrowprops=dict(arrowstyle="->", color=GRAY))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("FLOPs (6ND)"); ax.set_ylabel("MS-COCO 30k FID (낮을수록 좋음)")
ax.set_yticks([16, 32, 64, 128], ["16", "32", "64", "128"]); ax.legend(fontsize=8)
ax.set_title("논문 그림 5 FID 패널: PDF 벡터에서 읽은 점과 다시 맞춘 선", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "own_fid_refit.png", dpi=150); plt.close()

# ---- 4. 절제 효과 (산포 단위)
rows = []
def add(label, d, col): rows.append((label, d[col]["units_of_scatter"], d[col]["pct"], col))
add("표5 선형 양방향 vs 인과", A["table5_bidirectional_vs_causal"]["Linear"], "FID")
add("표5 U-Net 양방향 vs 인과", A["table5_bidirectional_vs_causal"]["U-Net"], "FID")
add("표6 U-Net 4×4 vs 2×2", A["table6_unet_patch_vs_2x2"]["4x4"], "FID")
add("표6 U-Net 8×8 vs 2×2", A["table6_unet_patch_vs_2x2"]["8x8"], "FID")
add("표6 선형 8×8 vs 2×2", A["table6_linear_patch_vs_2x2"]["8x8"], "FID")
add("표7 U-Net vs 선형 (0.16B)", A["table7_unet_vs_linear_by_size"]["0.16B"], "FID")
add("표7 U-Net vs 선형 (7B)", A["table7_unet_vs_linear_by_size"]["7B"], "FID")
add("표8 노이즈 제한 (0.76B)", A["table8_noise_limit"]["0.76B"], "CIDEr")
add("표8 노이즈 제한 (7B)", A["table8_noise_limit"]["7B"], "CIDEr")
add("표7 U-Net vs 선형 (7B)", A["table7_unet_vs_linear_by_size"]["7B"], "CIDEr")
add("표3 Transfusion vs Chameleon", A["table3_transfusion_vs_chameleon"], "FID")
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ys = np.arange(len(rows))[::-1]
for y, (lab, u, pct, col) in zip(ys, rows):
    ax.barh(y, u, color=BLUE if col == "FID" else ORANGE, height=0.6)
    ax.text(u + 0.1, y, f"{pct:+.1f}%", va="center", fontsize=7)
ax.set_yticks(ys, [f"{r[0]} [{r[3]}]" for r in rows], fontsize=7)
ax.axvline(1, color=GRAY, ls="--", lw=0.8); ax.axvline(2, color=RED, ls="--", lw=0.8)
ax.set_xlabel("효과 ÷ 그림 5 맞춤선 잔차 SD (FID 14%, CIDEr 12%). 1 이하는 산포 안, 2 이상이면 구별", fontsize=8)
ax.set_title("절제 효과 크기의 판정 (모든 표가 학습 1회)", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "own_effects.png", dpi=150); plt.close()
print("ok")
