"""results_*.json -> ../figures/own_*.png (우리가 그린 그래프. 논문 그림은 쓰지 않는다)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load(name):
    return json.load(open(HERE / name))


def fig_recovery():
    r = load("results_a.json")
    alphas, ratios = r["alphas"], r["ratios"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
    for row, setting in enumerate(["main", "hard"]):
        st = r["settings"][setting]
        runs = [x for x in r["runs"] if x["setting"] == setting]
        for col, (key, title, fmt) in enumerate([("frac_rec", "정답 특징 복원율 (최대 코사인 > 0.9)", ".2f"), ("l0", "평균 L0 (표본당 켜진 특징 수)", ".0f"),
                                                  ("fvu", "FVU (설명 못 한 분산 비율)", ".2f")]):
            M = np.array([[np.mean([x[key] for x in runs if x["R"] == R and x["alpha"] == a]) for a in alphas] for R in ratios])
            im = ax[row, col].imshow(M, aspect="auto", cmap="viridis" if key != "fvu" else "magma_r", vmin=0, vmax=1 if key != "l0" else None)
            for i in range(len(ratios)):
                for j in range(len(alphas)):
                    v = M[i, j]
                    ax[row, col].text(j, i, format(v, fmt), ha="center", va="center", color="w" if (v < 0.5 * np.nanmax(M)) == (key != "fvu") else "k", fontsize=9)
            ax[row, col].set_xticks(range(len(alphas)), [str(a) for a in alphas])
            ax[row, col].set_yticks(range(len(ratios)), [f"R={R} ({R * st['d']})" for R in ratios])
            ax[row, col].set_xlabel("α")
            ax[row, col].set_title(f"{title}\n[{setting}: 정답 {st['n']}개 / 차원 {st['d']}]", fontsize=10)
    fig.suptitle("정답 있는 시험대: α와 사전 크기 R에 따른 복원 (시드 3개 평균)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "own_recovery_grid.png", dpi=140)
    plt.close(fig)


def fig_shrinkage():
    from matplotlib.ticker import FuncFormatter
    r = load("results_a.json")
    r2 = load("results_a2.json")
    alphas = [a for a in r["alphas"] if a > 0]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    a2 = sorted({x["alpha"] for x in r2["rows"] if x["alpha"] > 0})
    ys = [np.nanmean([x["slope"] for x in r2["rows"] if x["alpha"] == a]) for a in a2]
    ax.plot(a2, ys, "s-", color=C[2], label="직교 정답 n=d=16 (학습)")
    for k, setting in enumerate(["main", "hard"]):
        ys, es, xs = [], [], []
        for a in alphas:
            v = [x["slope"] for x in r["runs"] if x["setting"] == setting and x["alpha"] == a and x["R"] == 2 and x["n_match"] >= 5]
            if v:
                xs.append(a); ys.append(np.mean(v)); es.append(np.std(v))
        ax.errorbar(xs, ys, yerr=es, marker="o", color=C[k], label=f"겹친 정답 ({setting}, R=2, 복원된 특징만)", capsize=3)
    t = np.geomspace(0.02, 1.2, 50)
    ax.plot(t, [M_theory(a) for a in t], "k--", label="이론값 c = max(a - α/2, 0)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("α")
    ax.set_ylabel("a에 대한 c의 회귀 기울기 (1이면 수축 없음)")
    ax.set_title("L1 수축(shrinkage)", fontsize=11)
    ax.legend(fontsize=7.5)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(FIG / "own_shrinkage.png", dpi=140)
    plt.close(fig)


def M_theory(alpha):
    t = min(alpha / 2, 1.0)
    return 3.0 * ((1 - t ** 3) / 3 - t * (1 - t ** 2) / 2)


def fig_lm():
    r = load("results_b.json")
    sites = ["resid_post_l0", "resid_post_l1", "mlp_out_l0"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for k, site in enumerate(sites):
        for i, R in enumerate((2, 4, 8)):
            xs = [x for x in r["runs"] if x["site"] == site and x["R"] == R]
            ax[0].plot([x["l0"] for x in xs], [x["fvu"] for x in xs], "o-", color=C[k], alpha=0.4 + 0.3 * i, ms=4 + 2 * i,
                       label=f"{site} R={R}" if True else None)
            ax[1].plot([x["l0"] for x in xs], [min(x["rec_zero"], x["rec_mean"]) for x in xs], "o-", color=C[k], alpha=0.4 + 0.3 * i, ms=4 + 2 * i)
    ax[0].set_xlabel("평균 L0"); ax[0].set_ylabel("FVU"); ax[0].set_xscale("log"); ax[0].set_title("재구성: 희소도 대 FVU")
    ax[1].set_xlabel("평균 L0"); ax[1].set_ylabel("loss recovered (영·평균 대치 중 엄격한 쪽)"); ax[1].set_xscale("log"); ax[1].set_title("재구성을 모델에 끼웠을 때")
    ax[0].legend(fontsize=6, ncol=1)
    for k, site in enumerate(sites):
        d = [x for x in r["runs"] if x["site"] == site]
        ax[2].scatter([x["fvu_train"] for x in d], [x["fvu"] for x in d], color=C[k], s=14 + 6 * np.log2([x["R"] for x in d]), alpha=0.7, label=site)
    ax[2].plot([0, 0.5], [0, 0.5], "k--", lw=0.8)
    ax[2].set_xlabel("FVU (학습 집합)"); ax[2].set_ylabel("FVU (평가 집합)"); ax[2].set_title("학습-평가 간격 (죽은 특징은 모든 실행에서 0.1% 이하)")
    ax[2].legend(fontsize=7)
    fig.suptitle("29번 모델 activation의 SAE (진할수록 R이 큼: 2, 4, 8)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "own_lm_sae.png", dpi=140)
    plt.close(fig)


def fig_interp():
    r = load("results_c.json")
    sites = list(r["sites"])
    methods = [("basis", "잔차 기저"), ("random_128", "무작위 128"), ("random_512", "무작위 512"), ("pca_2x128", "PCA 2×128"),
               ("sae_R2", "SAE R=2"), ("sae_R4", "SAE R=4"), ("sae_R8", "SAE R=8")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    w = 0.8 / len(methods)
    for i, (m, lab) in enumerate(methods):
        for j, key in enumerate(["top1_share_mean", "entropy_mean"]):
            vals = [r["sites"][s][m].get(key, np.nan) for s in sites]
            ax[j].bar(np.arange(len(sites)) + i * w, vals, w, label=lab, color=plt.cm.tab10(i))
    for j, t in enumerate(["상위 20개 중 최빈 토큰 비율 (클수록 집중)", "상위 20개 토큰 분포 엔트로피, 비트 (작을수록 집중)"]):
        ax[j].set_xticks(np.arange(len(sites)) + 0.4 - w / 2, sites)
        ax[j].set_title(t, fontsize=10)
    ax[0].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("해석 가능성의 거친 대용 지표 (자동 해석 점수가 아님)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "own_interp_proxy.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["recovery", "shrinkage", "lm", "interp"]
    for w in which:
        {"recovery": fig_recovery, "shrinkage": fig_shrinkage, "lm": fig_lm, "interp": fig_interp}[w]()
        print("saved", w)
