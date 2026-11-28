"""우리가 그린 그래프를 report/figures/own_*.png 로 저장한다 (matplotlib).

own_densities.png  모든 정식화를 같은 RF 시간 밀도 pi_eff(t)로 환산 (loss_weighting.py)
own_floor.png      가우시안 혼합 토이: 속도 손실의 줄일 수 없는 바닥, 해상도별 MMSE 전이와 logit-normal 질량 (gmm_toy.py)
own_euler.png      정확한 속도장 + 오일러: 오차 대 단계 수, 해상도 r × 이동 α 지도 (gmm_toy.py)
own_spectrum.png   멱법칙 스펙트럼에서 t/(1-t) 이동비 대 식 (23) (spectrum_shift.py)
own_scaling.png    그림 8 점 재적합과 상관 (scaling_refit.py)
먼저 loss_weighting.py, gmm_toy.py, spectrum_shift.py, fig8_digitize.py, table_checks.py, scaling_refit.py를 실행해 둘 것.
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
BLUE, ORANGE, GRAY, RED, GREEN, PURPLE = "#1f5fae", "#d9822b", "#8a8a85", "#b23a48", "#3a8f5c", "#7a4fa3"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})

pi = np.load(HERE.parent / ".scratch" / "pi_eff.npz")
toy = np.load(HERE.parent / ".scratch" / "toy_curves.npz")
T = json.loads((HERE / "toy_results.json").read_text())
S = json.loads((HERE / "spectrum_shift_results.json").read_text())
SC = json.loads((HERE / "scaling_refit_results.json").read_text())
P8 = json.loads((HERE / "fig8_points.json").read_text())

# ---------------------------------------------------------------- 1. densities
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
t = pi["t"]
for key, lab, c, ls in [("rf (uniform)", "rf (균일)", GRAY, "-"), ("rf_lognorm(0,1)", "logit-normal(0,1)", BLUE, "-"),
                        ("rf_mode(1.29)", "mode(1.29)", GREEN, "-"), ("rf_cosmap", "CosMap", PURPLE, "-"),
                        ("eps_linear", "eps/linear 의 등가 밀도", ORANGE, "--"), ("edm(-1.2,1.2)", "edm(-1.2,1.2) 의 등가 밀도", RED, "--")]:
    k = key.replace("/", "_")
    ax[0].plot(t, pi[k], color=c, ls=ls, lw=1.8, label=lab)
ax[0].set_ylim(0, 3.2); ax[0].text(0.03, 0.15, "eps/linear 곡선의 왼쪽 끝은 이산 스케줄의\nSNR 범위 한계", fontsize=6.5, color=ORANGE); ax[0].set_xlabel("t (0 = 데이터, 1 = 노이즈)"); ax[0].set_ylabel("등가 RF 시간 밀도 π_eff(t)")
ax[0].set_title("모든 정식화를 같은 RF 시간 밀도로 환산"); ax[0].legend(fontsize=8, frameon=False)
# right: floor and difficulty for r = 1
tg = toy["t"]
f1 = toy["floor_r1"]; ax[1].plot(tg, f1 / f1.max(), color=RED, lw=2, label="속도 손실 바닥 MMSE/t² (r=1, 최댓값=1)")
ax[1].plot(tg, toy["jac_r1"] / toy["jac_r1"].max(), color=ORANGE, lw=2, ls="--", label="속도장 야코비안 크기 (최댓값=1)")
ax[1].fill_between(t, pi["rf_lognorm(0,1)"] * 0.5, color=BLUE, alpha=0.25, label="logit-normal(0,1) 밀도 (×0.5)")
ax[1].plot(t, 0.5 * np.ones_like(t), color=GRAY, lw=0.8, ls=":")
ax[1].set_ylim(0, 1.45); ax[1].set_xlabel("t"); ax[1].set_ylabel("상대 크기"); ax[1].set_title("토이: 어려운 구간과 logit-normal 질량"); ax[1].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "own_densities.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. floor and MMSE transition vs resolution
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
cols = {1: BLUE, 4: GREEN, 16: ORANGE, 64: RED}
for r, c in cols.items():
    ax[0].plot(tg, toy[f"mmse_r{r}"], color=c, lw=2, label=f"r={r} (n=2r 좌표)")
    ax[0].axvline(T[f"r{r}"]["t_mmse_0.5"], color=c, ls=":", lw=1)
ax[0].set_xlabel("t"); ax[0].set_ylabel("사후 분산 비율 MMSE(c|z_t)/Var(c)"); ax[0].set_title("해상도가 늘면 전이 구간이 노이즈 쪽으로 밀린다"); ax[0].legend(fontsize=8, frameon=False)
rs = [1, 4, 16, 64]; m = T["mass_in_transition_interval"]
w = 0.2
for i, (k, lab, c) in enumerate([("uniform", "균일", GRAY), ("lognorm(0,1)", "logit-normal(0,1)", BLUE),
                                 ("lognorm(0,1)_shifted_3", "logit-normal(0,1) + 이동 α=3", ORANGE), ("lognorm(0,1)_shifted_sqrt_r", "logit-normal(0,1) + 이동 α=√r", GREEN)]):
    ax[1].bar(np.arange(4) + (i - 1.5) * w, [m[f"r{r}"][k] for r in rs], w, color=c, label=lab)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([f"r={r}" for r in rs]); ax[1].set_ylabel("전이 구간(분산 비율 0.1–0.9)에 든 시간 질량")
ax[1].set_ylim(0, 0.95); ax[1].set_title("해상도별 학습 시간 질량 (r=16 ≈ 256²→1024²)"); ax[1].legend(fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "own_floor.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. euler
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
E = T["euler"]
steps = [int(n) for n in E["r16"]["table"]]
for a, c in [("1.0", GRAY), ("3.0", ORANGE), ("4.0", GREEN)]:
    ax[0].plot(steps, [E["r16"]["table"][str(n)][a]["rms"] for n in steps], "-o", color=c, lw=1.8, ms=4, label=f"r=16, 이동 α={a}")
ax[0].plot(steps, [E["r1"]["table"][str(n)]["1.0"]["rms"] for n in steps], "--", color=BLUE, lw=1.5, label="r=1, α=1 (기준)")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_yticks([0.05, 0.1, 0.2, 0.4, 0.8]); ax[0].set_yticklabels(["0.05", "0.1", "0.2", "0.4", "0.8"]); ax[0].minorticks_off(); ax[0].set_xlabel("오일러 단계 수 N"); ax[0].set_ylabel("참 ODE 해 대비 끝점 RMS 오차")
ax[0].set_title("정확한 속도장 + 오일러 (r=16)"); ax[0].legend(fontsize=8, frameon=False)
al = [float(a) for a in E["r1"]["table"]["8"]]
Z = np.array([[E[f"r{r}"]["table"]["8"][str(a)]["rms"] for a in al] for r in rs])
im = ax[1].imshow(np.log10(Z), aspect="auto", cmap="viridis_r")
ax[1].set_xticks(range(len(al))); ax[1].set_xticklabels([str(a) for a in al]); ax[1].set_yticks(range(4)); ax[1].set_yticklabels([f"r={r}" for r in rs])
for i in range(4):
    for j in range(len(al)):
        ax[1].text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if np.log10(Z[i, j]) > -0.85 else "black")
    ax[1].add_patch(plt.Rectangle((int(np.argmin(Z[i])) - .5, i - .5), 1, 1, fill=False, ec=RED, lw=2))
ax[1].set_xlabel("이동 α"); ax[1].set_title("N=8 오차 (빨간 칸 = 해상도별 최적 α)"); ax[1].grid(False)
fig.tight_layout(); fig.savefig(OUT / "own_euler.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. spectrum
fig, ax = plt.subplots(figsize=(5.4, 3.9))
Ns = [32, 64, 128, 256]
ax.plot([n / 32 for n in Ns], [n / 32 for n in Ns], color="k", lw=1.5, ls="--", label="식 (23): √(m/n) = N/32")
for (b, rec), c in zip(S.items(), [RED, ORANGE, GREEN, BLUE, PURPLE]):
    ax.plot([n / 32 for n in Ns], [rec[str(n)]["odds_ratio_vs_N0"] for n in Ns], "-o", ms=4, color=c, label=b.replace("beta", "β"))
ax.axhline(3.0, color=GRAY, lw=0.8, ls=":"); ax.text(1.05, 3.1, "논문이 고른 α=3", fontsize=8, color=GRAY)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("N / 32 (격자 한 변)"); ax.set_ylabel("t/(1-t) 50% 지점의 이동비"); ax.set_title("멱법칙 스펙트럼 P(k)∝|k|^-β 의 이동비")
ax.legend(fontsize=7.5, frameon=False); fig.tight_layout(); fig.savefig(OUT / "own_spectrum.png"); plt.close(fig)

# ---------------------------------------------------------------- 5. scaling
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
dcol = {"15": "#440154", "18": "#3b528b", "21": "#21918c", "30": "#5ec962", "38": "#c9a800"}
for d, pts in P8["fig8_top2_loss_vs_flops"]["points"].items():
    ax[0].plot([p["x"] for p in pts], [p["y"] for p in pts], "o", color=dcol[d], ms=4, label=f"depth={d}")
a = SC["A_fit"]
xs = np.logspace(20.5, 23.2, 100)
ax[0].plot(xs, a["power_all_points"]["a"] * (xs / 1e21) ** (-a["power_all_points"]["b"]), color="k", lw=1.2, label=f"L={a['power_all_points']['a']:.3f}·(C/1e21)^-{a['power_all_points']['b']:.4f}")
ax[0].axvline(2.66e22, color=GRAY, ls=":", lw=1); ax[0].text(2.8e22, 0.416, "관측 끝", fontsize=8, color=GRAY)
ax[0].set_xscale("log"); ax[0].set_xlabel("학습 FLOPs (그림 8에서 읽음)"); ax[0].set_ylabel("검증 손실"); ax[0].set_title("그림 8 재적합 (점 36개, 모델 5개)"); ax[0].legend(fontsize=7, frameon=False)
Xg = [(p["x"], p["y"], d) for d, pts in P8["fig8_bot1_geneval"]["points"].items() for p in pts]
for d in dcol:
    pts = [(x, y) for x, y, dd in Xg if dd == d]
    ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "o", color=dcol[d], ms=4)
c1 = SC["C_correlations"]["fig8_bot1_geneval"]
ax[1].set_xlabel("GenEval (overall)"); ax[1].set_ylabel("검증 손실")
ax[1].set_title(f"r(읽은 값) = {c1['r']:.3f} (논문 {c1['paper_r']}), 깊이 내 r = {c1['r_within_depth']:.2f}")
fig.tight_layout(); fig.savefig(OUT / "own_scaling.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("own_*.png")))
