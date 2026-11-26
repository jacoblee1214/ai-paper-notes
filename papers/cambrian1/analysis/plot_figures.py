"""우리가 그린 그림 네 장 (matplotlib). 입력: recompute_results.json, cvbench_results.json, tables.json. 출력: report/figures/own_*.png
실행: python3 -I analysis/plot_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
_avail = {f.name for f in font_manager.fontManager.ttflist}
matplotlib.rcParams["font.family"] = [next((n for n in ("Noto Sans CJK KR", "Apple SD Gothic Neo") if n in _avail), "DejaVu Sans")]
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.load(open(HERE / "recompute_results.json"))
CV = json.load(open(HERE / "cvbench_results.json"))
T = json.load(open(HERE / "tables.json"))
C = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

# 1. 이미지 없음 대 있음 (그림 3 왼쪽 복원값, MME 우연 수준 보정)
rows = sorted(R["fig3"], key=lambda r: r["gap_E_D"])
fig, ax = plt.subplots(figsize=(8.6, 5.8))
y = np.arange(len(rows))
for i, r in enumerate(rows):
    ax.plot([r["disabled"], r["enabled"]], [i, i], color="#bbbbbb", lw=3, zorder=1)
    lo, hi = r["disabled_wilson95"]
    ax.plot([lo, hi], [i, i], color=C[3], lw=1.2, zorder=2)
ax.scatter([r["enabled"] for r in rows], y, color=C[0], s=36, label="이미지 있음 (23개 모델 평균)", zorder=3)
ax.scatter([r["disabled"] for r in rows], y, color=C[3], s=36, label="이미지 없음 (붉은 선: 단일 모델 가정 Wilson 95%)", zorder=3)
ax.scatter([r["random_metric"] for r in rows], y, marker="s", color=C[1], s=30, label="우연 수준 (MME는 37.5로 보정)", zorder=3)
for i, r in enumerate(rows):
    ax.text(101, i, f"{r['gap_E_D']:.1f}", va="center", fontsize=8, color="#333")
ax.text(101, len(rows) - 0.1, "있음-없음", fontsize=8, color="#333", ha="left")
ax.set_yticks(y); ax.set_yticklabels([r["bench"] for r in rows], fontsize=9)
ax.set_xlim(-3, 108); ax.set_xlabel("점수 (MME는 /20 척도)")
ax.set_title("이미지 없이 얻는 점수: SQA-I, MMMU, MathVista, AI2D는 차이가 5점 미만")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "own_no_image.png", dpi=150); plt.close(fig)

# 2. 군집 안정성: 그림 15 상관행렬(라벨 순) 대 부트스트랩 공동 군집 빈도
names = T["fig15_names"]
C15 = np.array(T["fig15_corr"])
lab = ["G"] * 4 + ["K"] * 4 + ["O"] * 4 + ["V"] * 3
paper_order = ["MME", "MMB", "SEED-I", "GQA", "SQA-I", "MMMU", "MathVista", "AI2D", "ChartQA", "OCRBench", "TextVQA", "DocVQA", "MMVP", "RW QA", "CV-CB"]
idx = [names.index(n) for n in paper_order]
M = C15[np.ix_(idx, idx)]
# 공동 군집 빈도는 recompute.py가 per-pair로 저장하지 않으므로 같은 절차를 여기서 다시 돌린다 (B=1000)
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
rng = np.random.default_rng(1)
cols = T["cols"][1:]
bk = {"MMEP": "MME", "MMB": "MMB", "SEEDI": "SEED-I", "GQA": "GQA", "SQAI": "SQA-I", "MMMU": "MMMU", "MathVista": "MathVista", "AI2D": "AI2D",
      "ChartQA": "ChartQA", "OCRBench": "OCRBench", "TextVQA": "TextVQA", "DocVQA": "DocVQA", "MMVP": "MMVP", "RWQA": "RW QA"}
rows_u = T["t12_0M"] + T["t13_0.5M"] + T["t14_1.2M"] + T["t15_1.2M_unfrozen"]
X = np.array([[r[c] / (20 if c == "MMEP" else 1) for c in bk] + [(r["CV2D"] + r["CV3D"]) / 2] for r in rows_u])
Xn = [*bk.values(), "CV-CB"]


def corr(Xs):
    Z = Xs - Xs.mean(0); sd = Z.std(0); sd[sd == 0] = np.nan
    return np.nan_to_num((Z / sd).T @ (Z / sd) / len(Xs))


def clus(Cm):
    D = 1 - Cm; np.fill_diagonal(D, 0)
    return fcluster(linkage(squareform((D + D.T) / 2, checks=False), "average"), 4, "maxclust")


co = np.zeros((15, 15)); B = 1000
for _ in range(B):
    cl = clus(corr(X[rng.integers(0, len(X), len(X))]))
    co += cl[:, None] == cl[None, :]
co /= B
order_u = [Xn.index(n) for n in paper_order]
CO = co[np.ix_(order_u, order_u)]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.4))
for ax, mat, ttl, vmin in [(axs[0], M, "논문 그림 15 상관행렬 (라벨 순)", 0.5), (axs[1], CO, "부트스트랩 공동 군집 빈도 (모델 72행, 평균연결 k=4)", 0.0)]:
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=1.0)
    ax.set_xticks(range(15)); ax.set_yticks(range(15))
    ax.set_xticklabels(paper_order, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(paper_order, fontsize=8)
    for b in (3.5, 7.5, 11.5):
        ax.axhline(b, color="white", lw=1.5); ax.axvline(b, color="white", lw=1.5)
    ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("논문의 4개 라벨(General, Knowledge, OCR & Chart, Vision-Centric) 중 General과 OCR만 데이터에서 안정적으로 묶인다", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_cluster_stability.png", dpi=150); plt.close(fig)

# 3. 표 3: 인코더 하나를 더하거나 바꿀 때 범주 평균 변화 대 MDE
pairs = [p for p in R["ensemble_pairs"] if not p["pair"].endswith("4개 모두")]
short = {"SigLIP+DINOv2 -> +ConvNext (ConvNeXt 추가)": "+ConvNeXt\n(SigLIP+DINOv2에)", "SigLIP+ConvNext -> +DINOv2 (DINOv2 추가)": "+DINOv2\n(SigLIP+ConvNeXt에)", "SigLIP+DINOv2 -> +ConvNext (ConvNeXt 추가)": "+ConvNeXt\n(SigLIP+DINOv2에)",
         "SigLIP+DINOv2+ConvNext -> +CLIP": "+CLIP\n(3개 조합에)", "SigLIP+ConvNext -> CLIP+ConvNext (SigLIP 대신 CLIP)": "SigLIP->CLIP\n(ConvNeXt와)",
         "SigLIP+DINOv2+ConvNext-L -> SigLIP+CLIP+ConvNext-L (DINOv2 대신 CLIP)": "DINOv2->CLIP\n(SigLIP+ConvNeXt-L)"}
cats = ["General", "Knowledge", "OCR", "Vision"]
fig, axs = plt.subplots(1, 4, figsize=(12, 4.4), sharey=True)
for ax, g in zip(axs, cats):
    vals = [p["cat_delta"][g] for p in pairs]
    ax.barh(range(len(pairs)), vals, color=[C[0] if v >= 0 else C[3] for v in vals])
    m = pairs[0]["cat_mde"][g]
    ax.axvspan(-m, m, color="#cccccc", alpha=0.6, zorder=0)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title({"General": "General", "Knowledge": "Knowledge", "OCR": "OCR & Chart", "Vision": "Vision-Centric"}[g] + f"\nMDE ±{m:.1f}", fontsize=9)
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels([short[p["pair"]] for p in pairs], fontsize=8)
    ax.invert_yaxis()
fig.suptitle("표 3의 단일 비교: ConvNeXt를 더하면 OCR·Vision이 오르지만 DINOv2를 더하면 범주 평균이 MDE(회색 띠) 안이다", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_ensemble_delta.png", dpi=150); plt.close(fig)

# 4. CV-Bench 2D: 우연 수준, 이미지 없는 규칙, 보고된 모델 점수
t3 = {r["name"]: r for r in T["t3_ensemble"]}
t14 = T["t14_1.2M"]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
bars = [("우연 수준 (선택지 수에서 계산)", CV["chance"]["2D_pooled"], "#999999"),
        ("이미지 없는 규칙: 선택지 값의 medoid (매개변수 없음)", CV["no_image_rule"]["2D_pooled"], C[1]),
        ("이미지 없는 규칙: 선택지 수별 순위 (3개 숫자를 표본 내에서 맞춤)", CV["no_image_rule_rank"]["2D_by_formula(ADE+COCO)/2"], C[3]),
        ("표 14 단일 인코더 18개 (범위)", None, C[0]),
        ("표 3 융합 7개 (범위)", None, C[0]),
        ("Cambrian-1 8B/13B/34B (표 9)", None, C[2])]
yy = np.arange(len(bars))[::-1]
for (lab, v, col), y_ in zip(bars, yy):
    if v is not None:
        ax.barh(y_, v, color=col); ax.text(v + 0.8, y_, f"{v:.1f}", va="center", fontsize=8)
rng14 = [r["CV2D"] for r in t14]; rng3 = [r["CV2D"] for r in T["t3_ensemble"]]; rng9 = [72.3, 72.5, 74.0]
for y_, rg, col in [(yy[3], rng14, C[0]), (yy[4], rng3, C[0]), (yy[5], rng9, C[2])]:
    ax.plot([min(rg), max(rg)], [y_, y_], color=col, lw=6, solid_capstyle="butt")
    ax.scatter(rg, [y_] * len(rg), color=col, s=14, zorder=3)
    ax.text(max(rg) + 0.8, y_, f"{min(rg):.1f}-{max(rg):.1f}", va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels([b[0] for b in bars], fontsize=8)
ax.set_xlim(0, 90); ax.set_xlabel("CV-Bench 2D 정확도 (%) = (ADE20K + COCO)/2")
ax.set_title("CV-Bench 2D: 이미지 없는 규칙이 Count를 92%까지 맞춘다")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "own_cvbench_no_image.png", dpi=150); plt.close(fig)
print("wrote", sorted(p.name for p in FIG.glob("own_*.png")))
