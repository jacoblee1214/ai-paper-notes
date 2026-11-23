"""우리가 그린 그림 세 장 (matplotlib). 입력: apply_results.json, mc_results.json. 출력: figures/own_*.png
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
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
A = json.load(open(HERE / "apply_results.json"))
MC = json.load(open(HERE / "mc_results.json"))
C = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

# 1. 군집 수에 따른 포함률
rows = MC["E1_coverage_by_clusters"]
g = [r["clusters"] for r in rows]
fig, ax = plt.subplots(figsize=(6.4, 4))
for key, lab, c in [("naive", "단순 SE (식 1)", C[3]), ("cluster_eq4", "군집 SE (식 4)", C[0]),
                    ("cluster_cr1", "군집 SE + G/(G-1)", C[1]), ("cluster_cr1_t", "군집 SE + G/(G-1) + t(G-1)", C[2])]:
    ax.plot(g, [r[key] for r in rows], "o-", label=lab, color=c)
ax.axhline(0.95, color="gray", ls="--", lw=1)
ax.set_xscale("log"); ax.set_xlabel("군집 수 G (군집당 문항 10개, 군집 내 상관 0.5)")
ax.set_ylabel("95% 구간의 실제 포함률"); ax.set_ylim(0.5, 1.0); ax.legend(fontsize=8, loc="lower right")
ax.set_title("군집 SE는 군집이 적으면 약간 좁고, t 보정이 이를 메운다")
fig.tight_layout(); fig.savefig(FIG / "own_cluster_coverage.png", dpi=150); plt.close(fig)

# 2. SWE-bench 공개 결과: 비짝 / 짝 / 군집 짝 SE
fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
for ax, split, title in zip(axs, ("test", "lite"), ("전체 2,294문항", "Lite 300문항")):
    pr = A["A_swe_bench_rag"][split]["pairs"]
    names = list(pr)
    x = np.arange(len(names)); w = 0.26
    for k, (key, lab) in enumerate([("se_unpaired", "비짝"), ("se_paired", "짝지음"), ("se_paired_clustered", "짝지음+저장소 군집(12개)")]):
        ax.bar(x + (k - 1) * w, [100 * pr[n][key] for n in names], w, label=lab, color=C[k])
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" - ", "\n−\n") for n in names], fontsize=7)
    ax.set_title(title); ax.set_ylabel("차이의 표준오차 (%p)")
axs[0].legend(fontsize=7)
fig.suptitle("SWE-bench BM25 기준선 세 모델의 쌍별 차이: 같은 문항을 쓰면 SE가 줄어든다", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_swebench_pairs.png", dpi=150); plt.close(fig)

# 3. 반복 실행 공개 결과: K와 짝지음의 효과
pr = A["B_agentic_runs"]["pairs"]
short = {"nano-agent-Qwen_Qwen3-32B - nano-agent-Qwen_Qwen3-32B-temp0": "Qwen3 nano\n온도 기본−0",
         "nano-agent-mistral_devstral-2512 - nano-agent-mistral_devstral-2512__temp0": "Devstral nano\n기본−0",
         "r2e-gym-agentica-org__DeepSWE-preview - r2e-gym-agentica-org__DeepSWE-preview__temp0": "DeepSWE r2e\n기본−0",
         "r2e-gym-Qwen_Qwen3-32B - nano-agent-Qwen_Qwen3-32B": "Qwen3\nr2e−nano",
         "r2e-gym-mistral_devstral-2512 - nano-agent-mistral_devstral-2512": "Devstral\nr2e−nano",
         "r2e-gym-mistral_devstral-2512 - r2e-gym-Qwen_Qwen3-32B": "r2e\nDevstral−Qwen3"}
names = list(pr)
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(names)); w = 0.2
for k, (key, lab) in enumerate([("se_unpaired_k1", "비짝, 1회"), ("se_paired_k1", "짝지음, 1회"),
                                ("se_unpaired_k10", "비짝, 10회 평균"), ("se_paired_k10", "짝지음, 10회 평균")]):
    ax.bar(x + (k - 1.5) * w, [100 * pr[n][key] for n in names], w, label=lab, color=C[k])
ax.set_xticks(x); ax.set_xticklabels([short[n] for n in names], fontsize=7)
ax.set_ylabel("해결률 차이의 SE (%p)"); ax.legend(fontsize=8)
ax.set_title("SWE-bench Verified 500문항, 설정당 10회: 짝지음과 10회 평균을 함께 쓸 때 SE가 가장 작다", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "own_agentic_pairs.png", dpi=150); plt.close(fig)
print("ok")
