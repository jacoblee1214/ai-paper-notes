"""results_*.json -> figures/own_*.png (우리가 그린 그래프. 논문 그림은 쓰지 않는다). 실행: python3 analysis/plot_figures.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def load(name):
    p = HERE / name
    return json.load(open(p)) if p.exists() else None


def interference(W):
    """sum_{j!=i} (W^_i . W_j)^2, W: [m, n]"""
    nr = np.linalg.norm(W, axis=0, keepdims=True)
    U = W / np.maximum(nr, 1e-6)
    sq = (U.T @ W) ** 2
    return sq.sum(1) - np.diag(sq)


def fig_sweep():
    a = load("results_a.json")
    if a is None:
        return
    dens = np.array(a["density"])
    pick = [0, 3, 5, 6, 8, 9]  # 1, 0.3, 0.1, 0.05, 0.02, 0.01
    fig, axs = plt.subplots(2, len(pick), figsize=(15, 5.2), sharey=True, gridspec_kw=dict(hspace=0.5))
    for row, key, title in ((0, "linear", "선형 모델"), (1, "relu", "ReLU 출력 모델")):
        W = np.array(a[f"{key}_W_seed0"])  # [inst, m, n]
        for c, k in enumerate(pick):
            ax = axs[row, c]
            nrm = np.linalg.norm(W[k], axis=0)
            inter = np.clip(interference(W[k]), 0, 1)
            ax.bar(np.arange(W.shape[2]), nrm, color=plt.cm.viridis(inter), width=0.85)
            ax.axvline(W.shape[1] - 0.5, color="k", lw=0.6, ls=":")
            ax.set_ylim(0, 1.3)
            ax.set_title(f"{title}\n1-S = {dens[k]:g}" if row == 0 else f"1-S = {dens[k]:g}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{title}\n||W_i||")
            if row == 1:
                ax.set_xlabel("특징 i (중요도 내림차순)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axs, label="간섭 sum_{j!=i}(W^_i·W_j)^2 (보라=직교, 노랑>=1)", shrink=0.8, pad=0.01)
    fig.suptitle("n=20, m=5, I_i=0.7^i: 희소도별 특징 노름과 간섭 (시드 0, 점선=m번째 특징 뒤)", y=1.0)
    fig.savefig(FIG / "own_sparsity_sweep.png", bbox_inches="tight")
    plt.close(fig)

    # 표현된 특징 수
    fig, ax = plt.subplots(figsize=(6.2, 4))
    x = 1 / dens
    for key, col, lab in (("relu", "C3", "ReLU 출력"), ("linear", "C0", "선형")):
        cnt = np.array(a[key]["count"])
        ax.plot(x, cnt.mean(0), "o-", color=col, label=f"{lab} (시드 {cnt.shape[0]}개 평균)")
        ax.fill_between(x, cnt.min(0), cnt.max(0), color=col, alpha=0.2)
    ax.axhline(5, color="k", ls=":", lw=0.8)
    ax.text(x[0], 5.25, "m = 5", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("1/(1-S) (오른쪽일수록 희소)")
    ax.set_ylabel("표현된 특징 수 (||W_i||^2 > 0.5)")
    ax.set_title("n=20, m=5: 희소도별 표현된 특징 수 (띠=시드 최소~최대)", fontsize=10)
    ax.legend()
    fig.savefig(FIG / "own_count_vs_sparsity.png", bbox_inches="tight")
    plt.close(fig)


def fig_phase():
    b = load("results_b.json")
    if b is None:
        return
    R, Dn = np.array(b["R"]), np.array(b["density"])
    maps = [("학습(모집단 손실 최소화)", np.array(b["cls_learn"])), ("세 해 이론(최적 b)", np.array(b["cls_theory"])),
            ("세 해 이론(b=0 닫힌 형식)", np.array(b["cls_theory_b0"]))]
    colors = ["#4c78a8", "#e45756", "#72b7b2", "#bbbbbb"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    rr = np.logspace(-1, 1, 100)
    for ax, (t, m) in zip(axs, maps):
        ax.pcolormesh(np.arange(len(Dn) + 1) - 0.5, np.arange(len(R) + 1) - 0.5, m, cmap=cmap, vmin=0, vmax=3, edgecolor="w", lw=0.5)
        ax.set_xticks(range(0, len(Dn), 2))
        ax.set_xticklabels([f"{Dn[i]:.2g}" for i in range(0, len(Dn), 2)], fontsize=7)
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels([f"{r:.2g}" for r in R], fontsize=7)
        ax.set_xlabel("특징이 켜질 확률 1-S")
        ax.set_title(t, fontsize=10)
        # b=0 근사 경계: 격자 좌표로 변환(log 보간)
        import minimal_impl as mi
        bd = np.array([mi.b0_boundary_density(r) for r in rr])
        xi = np.interp(np.log(bd), np.log(Dn[::-1]), np.arange(len(Dn))[::-1])
        yi = np.interp(np.log(rr), np.log(R), np.arange(len(R)))
        ax.plot(xi, yi, "k--", lw=1)
    axs[0].set_ylabel("특징 2의 상대 중요도 I_2 (I_1=1)")
    import matplotlib.patches as mp
    axs[2].legend(handles=[mp.Patch(color=colors[0], label="특징 1만"), mp.Patch(color=colors[1], label="둘 다 (겹침)"),
                           mp.Patch(color=colors[2], label="특징 2만"), mp.Patch(color=colors[3], label="둘 다 안 배움")],
                  fontsize=7, loc="lower left")
    fig.suptitle(f"n=2, m=1 단계도 (격자 일치율: 최적 b 이론 {b['agree_theory']:.2f}, b=0 이론 {b['agree_theory_b0']:.2f}; 점선=b=0 닫힌 형식 경계)", y=1.02, fontsize=10)
    fig.savefig(FIG / "own_phase_diagram.png", bbox_inches="tight")
    plt.close(fig)


def fig_dim(fn, out, title, m_val, lines):
    c = load(fn)
    if c is None:
        return
    dens = np.array(c["density"])
    D = np.array(c["D_seed0"])
    nrm2 = np.array(c["norm2_seed0"])
    x = 1 / dens
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    for v, lab in lines:
        ax.axhline(v, color="purple", alpha=0.25, lw=0.8)
        ax.text(x[-1] * 1.05, v, lab, fontsize=7, va="center")
    rng = np.random.default_rng(0)
    for k in range(len(dens)):
        rep = nrm2[k] > 0.5
        ax.scatter(x[k] * np.exp(rng.uniform(-0.04, 0.04, rep.sum())), D[k][rep], s=4, c="k", alpha=0.5)
    ds = np.array(c["D_star"])  # [seed, inst]
    ax.plot(x, ds.mean(0), "r-o", ms=3, label="D* = m/||W||_F^2 (시드 평균)")
    ax.set_xscale("log")
    ax.set_xlabel("1/(1-S)")
    ax.set_ylabel("특징 차원수 D_i (점: 표현된 특징, 시드 0)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=10)
    ax = axs[1]
    cnt = np.array(c["count"])
    ax.plot(x, cnt.mean(0) / m_val, "o-", label="표현된 특징 수 / m")
    ax.fill_between(x, cnt.min(0) / m_val, cnt.max(0) / m_val, alpha=0.2)
    ax.plot(x, 1 / ds.mean(0), "r--", label="1/D* = ||W||_F^2/m")
    ax.set_xscale("log")
    ax.set_xlabel("1/(1-S)")
    ax.set_ylabel("m당 특징 수")
    ax.legend(fontsize=8)
    fig.savefig(FIG / out, bbox_inches="tight")
    plt.close(fig)


def fig_geom():
    g = load("results_geom.json")
    if g is None:
        return
    dens = np.array(g["density"])
    W = np.array(g["W_seed0"])  # [inst, 2, n]
    pick = [1, 3, 5, 7, 9, 10]
    fig, axs = plt.subplots(1, len(pick), figsize=(14, 2.8))
    for ax, k in zip(axs, pick):
        w = W[k]
        nr = np.linalg.norm(w, axis=0)
        for i in range(w.shape[1]):
            if nr[i] ** 2 > 0.5:
                ax.annotate("", xy=w[:, i], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="C3", lw=0.8))
        cnt = int((nr ** 2 > 0.5).sum())
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=":", color="gray"))
        ax.set_title(f"1-S={dens[k]:g}\n표현 {cnt}개", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("m=2, n=40, 균일 중요도: 표현된 특징 방향(시드 0). k개가 고르게 퍼지면 차원수 D=2/k", y=1.08, fontsize=10)
    fig.savefig(FIG / "own_geometry_m2.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    fig_sweep()
    fig_phase()
    fig_dim("results_c.json", "own_dimensionality.png", "n=200, m=15, I=1: 특징 차원수와 D* (논문 n=400, m=30의 절반 규모)", 15,
            [(1, "1"), (3 / 4, "3/4"), (2 / 3, "2/3"), (1 / 2, "1/2"), (2 / 5, "2/5"), (3 / 8, "3/8")])
    fig_geom()
