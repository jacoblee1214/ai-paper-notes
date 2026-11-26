"""우리가 그린 그래프 세 장 -> ../figures/own_*.png (formula_results.json, toy_results.json 필요).
실행: python3 vit/analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

for name in ["Noto Sans CJK KR", "Noto Sans CJK JP", "AppleGothic", "Apple SD Gothic Neo"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

COL = {"cnn": "#1b6ca8", "cnn_shift": "#7fb7e0", "vit": "#c0392b", "vit_shift": "#e8a39b"}
LAB = {"cnn": "CNN", "cnn_shift": "CNN + 이동 증강", "vit": "ViT", "vit_shift": "ViT + 이동 증강"}


def toy():
    from scipy import stats
    import numpy as np
    r = json.loads((HERE / "toy_results.json").read_text())
    sizes = r["sizes"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for arm in ["cnn", "cnn_shift", "vit", "vit_shift"]:
        m, h = [], []
        for n in sizes:
            a = np.array(r["runs"][f"{arm}|{n}"]["acc"])
            m.append(a.mean())
            h.append(stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
        ax.errorbar(sizes, m, yerr=h, color=COL[arm], marker="o", ms=4, lw=1.6, capsize=3, label=LAB[arm],
                    ls="-" if "shift" not in arm else "--")
    ax.set_xscale("log")
    ax.axhline(1 / 6, color="gray", lw=0.8, ls=":")
    ax.text(sizes[0], 1 / 6 + 0.01, "무작위 1/6", color="gray", fontsize=8)
    ax.set_xlabel("학습 세트 크기 n (합성 도형 6종, 32x32)")
    ax.set_ylabel("시험 정확도 (3000장)")
    ax.set_title("토이: 데이터가 적을 때 CNN 대 ViT (시드 5개, 95% 구간)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_data_scaling.png")
    plt.close(fig)


def cost():
    r = json.loads((HERE / "formula_results.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
    cols = {"B/16": "#1b6ca8", "L/16": "#c0392b", "H/14": "#2e8b57"}
    for model in ["B/16", "L/16", "H/14"]:
        rows = [x for x in r["resolution"] if x["model"] == model]
        res = [x["res"] for x in rows]
        axs[0].plot(res, [x["GMACs"] for x in rows], marker="o", color=cols[model], label=model)
        axs[1].plot(res, [x["attn_share_of_layer"] * 100 for x in rows], marker="o", color=cols[model], label=model)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("입력 해상도 (px)")
    axs[0].set_ylabel("순전파 GMACs")
    axs[0].set_title("해상도별 연산량", fontsize=10)
    axs[1].set_xlabel("입력 해상도 (px)")
    axs[1].set_ylabel("층 연산 중 attention 행렬곱 비율 (%)")
    axs[1].set_title("attention(QK^T, AV)의 몫", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "own_resolution_cost.png")
    plt.close(fig)


def exaflops():
    r = json.loads((HERE / "formula_results.json").read_text())["table6_exaflops"]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    xs = [x["exaflops_table6"] for x in r]
    ys = [x["exaflops_ours"] for x in r]
    ax.loglog([40, 6000], [40, 6000], color="gray", lw=0.8, ls=":")
    ax.scatter(xs, ys, color="#c0392b", zorder=3)
    for x, y, row in zip(xs, ys, r):
        ax.annotate(f"{row['model']} {row['epochs']}ep", (x, y), textcoords="offset points", xytext=(5, -10),
                    fontsize=7)
    ax.set_xlabel("논문 표 6의 exaFLOPs")
    ax.set_ylabel("우리 재계산 (6 x MAC x 303M x 에폭)")
    ax.set_title("표 6 ViT 행 재계산", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_table6_exaflops.png")
    plt.close(fig)


if __name__ == "__main__":
    cost()
    exaflops()
    if (HERE / "toy_results.json").exists():
        toy()
