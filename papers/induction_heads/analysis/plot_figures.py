"""우리가 그린 그래프 (matplotlib). 입력: synthetic_results.json, natural_results.json, natural_mix_results.json.
실행: python3 analysis/plot_figures.py  -> figures/own_*.png"""
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

for name in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"1L": "#8c8c8c", "2L": "#d55e00", "1L_smear": "#0072b2"}
LAB = {"1L": "1층", "2L": "2층", "1L_smear": "1층 + smeared key"}


def load(n):
    return json.loads((HERE / n).read_text())


def fig_synthetic():
    r = load("synthetic_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for cfg, runs in r["configs"].items():
        for i, (seed, e) in enumerate(runs.items()):
            h = e["history"]
            ax[0].plot(h["step"], h["rep"], color=C[cfg], alpha=0.7, lw=1, label=LAB[cfg] if i == 0 else None)
            ax[1].plot(h["step"], h["prefix_max"], color=C[cfg], alpha=0.7, lw=1, label=LAB[cfg] if i == 0 else None)
    ax[0].set_title("반복 구간 손실 (합성 데이터, 시드별 선)")
    ax[1].set_title("가장 높은 head의 prefix-matching 점수")
    for a in ax:
        a.set_xlabel("학습 스텝")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("nats")
    ax[1].set_ylim(0, 1.02)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_synthetic_phase_change.png", dpi=150)
    plt.close(fig)


def fig_natural():
    mix, nat = load("natural_mix_results.json"), load("natural_results.json")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for tag, r, col in (("자연 텍스트만", nat, "#8c8c8c"), ("자연 텍스트 + 반복 구절 50%", mix, "#d55e00")):
        for i, (seed, e) in enumerate(r["seeds"].items()):
            h = e["history"]
            ax[0].plot(h["step"], h["dup_second"], color=col, alpha=0.8, lw=1, label=tag if i == 0 else None)
            ax[1].plot(h["step"], h["prefix_max"], color=col, alpha=0.8, lw=1)
            ax[2].plot(h["step"], h["icl"], color=col, alpha=0.8, lw=1)
    ax[0].set_title("반복 구절의 둘째 사본 손실 (검증)")
    ax[1].set_title("가장 높은 head의 prefix-matching 점수")
    ax[2].set_title("자연 텍스트 ICL 점수 (위치 25 대 250)")
    for a in ax:
        a.set_xlabel("학습 스텝")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("nats")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_natural_text.png", dpi=150)
    plt.close(fig)


def fig_ablation():
    s = load("synthetic_results.json")["configs"]["2L"]
    m = load("natural_mix_results.json")["seeds"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    # 합성 2층: 시드마다 (층 1 head 하나씩 끈 최댓값 / 층 1 전체 / 층 0 head 하나씩 끈 최댓값 / 층 0 전체), 전체 대 패턴 유지
    labels = ["층 1\n최대 1개", "층 1\n전체", "층 0\n최대 1개", "층 0\n전체"]
    for k, (mode, col) in enumerate((("full", "#d55e00"), ("preserve", "#0072b2"))):
        vals = []
        for e in s.values():
            a = e["ablation_rep_loss"]
            vals.append([max(a[mode][1]), a["groups"]["layer1_all"][mode], max(a[mode][0]), a["groups"]["layer0_all"][mode]])
        vals = np.array(vals)
        x = np.arange(4) + (k - 0.5) * 0.36
        ax[0].bar(x, vals.mean(0), width=0.36, color=col, alpha=0.55, label={"full": "전체 ablation", "preserve": "패턴 유지 ablation"}[mode])
        ax[0].scatter(np.repeat(x[None], len(vals), 0), vals, s=10, color=col, edgecolor="k", linewidth=0.4, zorder=3)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(labels)
    ax[0].set_title("합성 2층: 반복 구간 손실 증가 (막대 = 5시드 평균, 점 = 시드)")
    ax[0].set_ylabel("nats")
    ax[0].legend(frameon=False)
    # 자연 텍스트 + 반복 구절: 시드마다 층별 최대 한 head
    lab2 = ["층 0 최대 1개", "층 1 최대 1개"]
    for k, (mode, col) in enumerate((("full", "#d55e00"), ("preserve", "#0072b2"))):
        for layer in range(2):
            for si, e in enumerate(m.values()):
                v = max(e["ablation"][mode][f"{layer}.{h}"]["dup_second"] - e["ablation"]["base"]["dup_second"] for h in range(4))
                x = layer + (k - 0.5) * 0.36
                ax[1].scatter([x], [v], s=28, color=col, edgecolor="k", linewidth=0.4, zorder=3, marker="osd"[si])
    ax[1].set_xticks(range(2))
    ax[1].set_xticklabels(lab2)
    ax[1].set_title("자연 텍스트 + 반복 구절: 둘째 사본 손실 증가\n(주황 = 전체, 파랑 = 패턴 유지, 모양 = 시드 0, 1, 2)")
    ax[1].set_xlim(-0.6, 1.6)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "own_ablation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_synthetic()
    fig_natural()
    fig_ablation()
    print("saved", sorted(p.name for p in FIG.glob("own_*.png")))
