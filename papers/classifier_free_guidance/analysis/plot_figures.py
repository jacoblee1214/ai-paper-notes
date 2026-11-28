"""우리가 그린 그래프를 figures/own_*.png 로 저장한다 (matplotlib). 논문 그림은 쓰지 않는다.

own_tradeoff.png  w 대 분산·재현율·지지집합 밖 비율·평균 사후 확률 (CFG-DDIM, CFG-조상, 정확한 기울임)
own_mismatch.png  정확한 기울임 대비 불일치(sliced W1): w, 단계 수 T, 일관된 기울임(오라클)과 비교
own_gaussian.png  가우시안 쌍의 닫힌 식: ODE 끝점 분산 / 기울임 분산, lambda별 안내 점수 오차, 포커-플랑크 잔차
own_scatter.png   클래스 1의 표본 산점도: w = 0, 2, 4, 8 (CFG-DDIM 대 정확한 기울임)
own_paper.png     논문 표 1·2를 다시 그린 FID-IS 곡선 (p_uncond, T)
먼저 derivation_checks.py, toy_experiments.py, paper_arithmetic.py를 실행해 둘 것.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

import minimal_impl as mi
import paper_arithmetic as pa

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for cand in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
BLUE, ORANGE, GRAY, RED, GREEN = "#1f5fae", "#d9822b", "#8a8a85", "#b23a48", "#3a8f5c"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.grid": True, "grid.alpha": 0.25})

TOY = json.loads((HERE / "toy_results.json").read_text())
DER = json.loads((HERE / "derivation_checks_results.json").read_text())
E1 = TOY["E1"]
WS = E1["settings"]["ws"]


def curve(name, key):
    m = np.array([E1[name][str(w)]["summary"][key][0] for w in WS])
    lo = np.array([E1[name][str(w)]["summary"][key][1] for w in WS])
    hi = np.array([E1[name][str(w)]["summary"][key][2] for w in WS])
    return m, lo, hi


def target(key):
    f = {"var": lambda p: np.mean([c["var"] for c in p["per_class"]]),
         "rec": lambda p: np.mean([c["rec"] for c in p["per_class"]]),
         "oos": lambda p: np.mean([c["oos"] for c in p["per_class"]]),
         "mean_pc": lambda p: p["pooled"]["mean_pc"]}[key]
    return np.array([f(E1["target"][str(w)]) for w in WS])


def fig_tradeoff():
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for a, key, title in zip(ax, ("var", "rec", "oos", "mean_pc"),
                             ("클래스 내 분산(축 평균)", "재현율 (k=3)", "지지집합 밖 비율", "평균 사후 p(c|x)")):
        for name, col, lab in (("ddim", BLUE, "CFG + DDIM"), ("ancestral", ORANGE, "CFG + 조상 표집(v=0.3)")):
            m, lo, hi = curve(name, key)
            a.plot(WS, m, "-o", color=col, ms=3.5, label=lab)
            a.fill_between(WS, lo, hi, color=col, alpha=0.2)
        a.plot(WS, target(key), "--s", color=GREEN, ms=3.5, label="정확한 기울임 p(x|c)p(c|x)^w")
        a.set_title(title, fontsize=10)
        a.set_xlabel("안내 세기 w")
    ax[2].set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: "1e%d" % round(np.log10(v)) if v > 0 else ""))
    ax[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "own_tradeoff.png")
    plt.close(fig)


def fig_mismatch():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, col, lab in (("ddim", BLUE, "CFG + DDIM"), ("ancestral", ORANGE, "CFG + 조상 표집")):
        m, lo, hi = curve(name, "sw1")
        ax[0].plot(WS, m, "-o", color=col, ms=3.5, label=lab)
        ax[0].fill_between(WS, lo, hi, color=col, alpha=0.2)
    ax[0].axhline(np.mean([E1["target"][str(w)]["floor_sw1"] for w in WS]), color=GRAY, ls=":", label="표본 잡음 바닥")
    ax[0].set_xlabel("안내 세기 w"); ax[0].set_ylabel("sliced W1 (정확한 기울임 대비)")
    ax[0].set_title("E1: T=256", fontsize=10); ax[0].legend(fontsize=8)
    E2 = TOY["E2"]
    for name, col in (("ddim", BLUE), ("ancestral", ORANGE)):
        Ts = E2["Ts"]
        ax[1].plot(Ts, [E2[name][str(T)]["summary"]["sw1"][0] for T in Ts], "-o", color=col, ms=3.5, label=name)
    ax[1].axhline(E2["floor_sw1"], color=GRAY, ls=":")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("단계 수 T"); ax[1].set_title("E2: w=2, 불일치는 T를 늘려도 남음", fontsize=10)
    E3 = TOY["E3"]
    xs, w_lab = [], []
    for i, k in enumerate(("w2.0_T64", "w2.0_T128", "w4.0_T64", "w4.0_T128")):
        s = E3[k]["summary"]
        ax[2].bar(i - 0.2, s["cfg"][0], 0.38, color=BLUE, label="CFG 식 (6)" if i == 0 else None)
        ax[2].bar(i + 0.2, s["oracle"][0], 0.38, color=GREEN, label="일관된 기울임 점수" if i == 0 else None)
        w_lab.append(k.replace("w", "w=").replace("_T", ", T="))
    ax[2].axhline(np.mean([E3[k]["floor_sw1"] for k in ("w2.0_T64", "w4.0_T64")]), color=GRAY, ls=":")
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(w_lab, fontsize=8)
    ax[2].set_title("E3: 같은 DDIM, 점수만 바꿈", fontsize=10); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "own_mismatch.png")
    plt.close(fig)


def fig_gaussian():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    rows = DER["D8_gaussian_pair"]
    for case, col in (("equal_var", GRAY), ("uncond_wider", BLUE), ("uncond_narrower", ORANGE)):
        r = [x for x in rows if x["case"] == case and x["var_ratio_ode_over_target"] is not None]
        ax[0].plot([x["w"] for x in r], [x["var_ratio_ode_over_target"] for x in r], "-o", color=col, ms=4, label=case)
    ax[0].set_yscale("log"); ax[0].set_xlabel("w"); ax[0].set_title("ODE 끝점 분산 / 기울임 분산", fontsize=10); ax[0].legend(fontsize=8)
    g = DER["D9_cfg_vs_diffused_tilted_eps"]
    ax[1].plot([x["lam"] for x in g], [x["rms_gap"] / x["rms_true"] for x in g], "-o", color=BLUE, ms=4)
    ax[1].set_xlabel("lambda (로그 SNR)"); ax[1].set_title("CFG 점수와 확산된 기울임 점수의 상대 차 (w=2)", fontsize=9)
    fp = DER["D10_fokker_planck_residual_std"]
    for lam, col in (("-3.0", BLUE), ("0.0", ORANGE), ("3.0", GREEN)):
        ws = sorted(float(k) for k in fp[lam])
        ax[2].plot(ws, [fp[lam][str(w)] + 1e-12 for w in ws], "-o", color=col, ms=4, label=f"lambda={lam}")
    ax[2].set_yscale("log"); ax[2].set_xlabel("w"); ax[2].set_title("포커-플랑크 잔차(0이어야 확산된 분포의 점수)", fontsize=9); ax[2].legend(fontsize=8)
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: "1e%d" % round(np.log10(v)) if v > 0 else "")
    for a in (ax[0], ax[2]):
        a.yaxis.set_major_formatter(fmt)
    fig.tight_layout()
    fig.savefig(OUT / "own_gaussian.png")
    plt.close(fig)


def fig_scatter():
    M = mi.default_toy()
    c = 1
    ws = (0.0, 2.0, 4.0, 8.0)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True)
    ref_all = torch.cat([M.mu.reshape(-1, 2)])
    for j, w in enumerate(ws):
        gen = torch.Generator().manual_seed(1)
        x = mi.sample_ddim(mi.guided_fn(M, c, w), 1500, 256, gen)
        t = mi.TiltedGrid(M, c, w).sample(1500, gen)
        for i, (pts, lab, col) in enumerate(((x, "CFG + DDIM", BLUE), (t, "정확한 기울임", GREEN))):
            a = ax[i, j]
            for k in range(M.C):
                dat = M.sample(k, 400, gen)
                a.scatter(dat[:, 0], dat[:, 1], s=2, color=GRAY, alpha=0.25)
            a.scatter(pts[:, 0], pts[:, 1], s=3, color=col, alpha=0.6)
            a.set_title(f"{lab}, w={w:g}", fontsize=9)
            a.set_aspect("equal")
    fig.suptitle("클래스 1 (회색은 세 클래스의 데이터, 색은 표본)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "own_scatter.png")
    plt.close(fig)


def fig_paper():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for p, col in zip((0.1, 0.2, 0.5), (BLUE, ORANGE, GREEN)):
        ax[0].plot(pa.IS1[p], pa.FID1[p], "-o", color=col, ms=3, label=f"p_uncond={p}")
    ax[0].set_xlabel("IS"); ax[0].set_ylabel("FID"); ax[0].set_title("표 1 (64x64)", fontsize=10); ax[0].legend(fontsize=8)
    for T, col in zip((128, 256, 1024), (BLUE, ORANGE, GREEN)):
        ax[1].plot(pa.IS2[T], pa.FID2[T], "-o", color=col, ms=3, label=f"T={T}")
    ax[1].set_xlabel("IS"); ax[1].set_title("표 2 (128x128)", fontsize=10); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "own_paper.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_tradeoff, fig_mismatch, fig_gaussian, fig_scatter, fig_paper):
        f()
    print("saved", sorted(p.name for p in OUT.glob("own_*.png")))
