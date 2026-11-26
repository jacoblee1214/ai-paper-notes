"""우리가 그린 그래프 세 장 -> ../figures/own_*.png (formula_results.json, toy_results.json 필요).
실행: python3 plot_figures.py   (이 폴더에서)
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for name in ["Noto Sans CJK KR", "Noto Sans CJK JP", "AppleGothic", "Apple SD Gothic Neo"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

LAB = {"neither": "둘 다 없음", "no_center": "날카롭게만", "no_sharpen": "중심화만", "dino": "DINO(중심화+날카롭게)",
       "sk": "SK+날카롭게", "dino_koleo": "DINO+KoLeo", "dinov2": "DINOv2(SK+KoLeo)"}
COL = {"neither": "#7f7f7f", "no_center": "#c0392b", "no_sharpen": "#e08e0b", "dino": "#1b6ca8", "sk": "#2e8b57",
       "dino_koleo": "#7b4fa3", "dinov2": "#111111"}


def ci(a):
    a = np.asarray(a, float)
    return a.mean(), (stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0)


def runs_of(r, setting, cond):
    return [x for x in r["runs"] if x["setting"] == setting and x["cond"] == cond]


def collapse(r):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8), sharex=True)
    for cond in LAB:
        rs = runs_of(r, "main", cond)
        its = [t[0] for t in rs[0]["traj"]]
        for k, a in ((1, ax[0]), (2, ax[1])):
            m = np.mean([[t[k] for t in x["traj"]] for x in rs], axis=0)
            a.plot(its, m, color=COL[cond], lw=1.8, ls="--" if cond in ("neither", "sk", "dino_koleo") else "-", label=LAB[cond])
    for a, t in zip(ax, ["교사 확률의 표본별 엔트로피", "배치 평균 확률의 엔트로피(프로토타입 사용)"]):
        a.axhline(r["log_K"], color="k", lw=0.8, ls=":")
        a.text(5, r["log_K"] + 0.05, "log K (균등)", fontsize=8)
        a.set_title(t, fontsize=10)
        a.set_xlabel("스텝")
        a.set_ylabel("nats")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=4)
    fig.suptitle("토이 붕괴 궤적 (교사 온도 0.02, 시드 5개 평균)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "own_toy_collapse.png")


def summary(r):
    items = [("main", c) for c in LAB] + [("official_temp", c) for c in ("dino", "dinov2")]
    names = [LAB[c] if s == "main" else LAB[c].split("(")[0] + "\n[공식 T]" for s, c in items]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for a, key, title in zip(ax, ["marginal_ent", "erank", "probe_acc"],
                             ["프로토타입 사용 엔트로피", "특징 유효 차원(erank)", "도형 선형 프로브 정확도"]):
        for i, (s, c) in enumerate(items):
            m, h = ci([x[key] for x in runs_of(r, s, c)])
            a.bar(i, m, yerr=h, color=COL[c], alpha=0.9 if s == "main" else 0.45, capsize=3)
        a.set_xticks(range(len(items)))
        a.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        a.set_title(title, fontsize=10)
    ax[0].axhline(r["log_K"], color="k", lw=0.8, ls=":")
    init_e = np.mean([x["init_erank"] for x in r["runs"] if x["setting"] == "main"])
    init_a = np.mean([x["init_probe_acc"] for x in r["runs"] if x["setting"] == "main"])
    ax[1].axhline(init_e, color="k", lw=0.8, ls=":")
    ax[1].text(0, init_e + 1, "무작위 초기화", fontsize=7)
    ax[2].axhline(init_a, color="k", lw=0.8, ls=":")
    ax[2].axhline(1 / 6, color="gray", lw=0.8, ls=":")
    ax[2].text(0, init_a + 0.01, "무작위 초기화", fontsize=7)
    ax[2].text(0, 1 / 6 + 0.01, "우연", fontsize=7)
    fig.suptitle("토이 요약 (평균 ± 95% 구간, 시드 5개, t 분포)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_summary.png")


def formula(f):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    tab = f["entropy_vs_spread_over_T"]["table"]
    logk = f["entropy_vs_spread_over_T"]["log_K"]
    xs, ys = [], []
    for s, d in tab.items():
        sv = float(s.split("_")[1])
        for t, e in d.items():
            xs.append(sv / float(t.split("_")[1]))
            ys.append(e)
    ax[0].scatter(xs, ys, s=14, color="#1b6ca8", label="측정(K=64)")
    g = np.linspace(0.01, 15, 300)
    ax[0].plot(g, np.clip(logk - g ** 2 / 2, 0, None), color="#c0392b", lw=1.4, label="log K - (s/τ)²/2")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("로짓 퍼짐 s / 온도 τ")
    ax[0].set_ylabel("교사 엔트로피 (nats)")
    ax[0].set_title("중심화한 로짓의 엔트로피", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    for k, v in f["sinkhorn_col_err"].items():
        its = sorted(int(i) for i in v)
        ax[1].plot(its, [max(v[str(i)], 1e-5) for i in its], marker="o", ms=3, label="로짓 표준편차 " + k.split("_")[-1])
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Sinkhorn-Knopp 반복 수")
    ax[1].set_ylabel("프로토타입 질량 최대 상대 오차")
    ax[1].axvline(3, color="k", lw=0.8, ls=":")
    ax[1].set_title("SK 반복 수와 균형 오차 (τ=0.05, B=256, K=32)", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "own_formula_checks.png")


if __name__ == "__main__":
    collapse(json.loads((HERE / "toy_results.json").read_text()))
    summary(json.loads((HERE / "toy_results.json").read_text()))
    formula(json.loads((HERE / "formula_results.json").read_text()))
