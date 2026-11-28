"""우리가 그린 그래프를 figures/own_*.png 로 저장한다 (matplotlib). 논문 그림은 쓰지 않는다.

own_sweep.png   정확한 eps*에서 슬라이스 W1 대 단계 수 S, eta별 (20 시드 평균, 95% 부트스트랩 구간, 표본 잡음 바닥)
own_ode.png     같은 x_T에서 연속 ODE 기준 해와의 거리(중앙값) 대 S: DDIM(식 13) 대 PF-ODE 오일러(식 15), 1/S 기울기 참고선
own_recon.png   x_0 -> x_T -> x_0 재구성 오차(데이터 분산 대비)와 인코딩된 잠재 변수가 N(0,I)에서 벗어난 정도 대 S
먼저 minimal_impl.py 를 실행해 minimal_impl_results.json 을 만들어 둘 것.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for cand in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(cand in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
BLUE, ORANGE, GRAY, RED, GREEN, PURPLE = "#1f5fae", "#d9822b", "#8a8a85", "#b23a48", "#3a8f5c", "#7b5ea7"


def ascii_log(ax, axis="both"):
    fmt = FuncFormatter(lambda v, _: ("%g" % v) if v > 0 else "")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


R = json.loads((HERE / "minimal_impl_results.json").read_text())


def fig_sweep():
    s = R["sweep"]
    Ss = s["S"]
    spec = [("0.0", "η=0 (DDIM)", BLUE, "o"), ("0.2", "η=0.2", GREEN, "s"), ("0.5", "η=0.5", PURPLE, "^"),
            ("1.0", "η=1 (DDPM)", ORANGE, "D"), ("hat", "sigma-hat (논문 식 그대로)", RED, "v"),
            ("hat_masked", "sigma-hat, 마지막 스텝 잡음 제거", GRAY, "x")]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for k, lab, c, mk in spec:
        m = np.array([s["metrics"]["sw"][str(S)][k] for S in Ss])
        ax.errorbar(np.array(Ss) * (1 + 0.03 * ([x[0] for x in spec].index(k) - 2.5)), m[:, 0],
                    yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], label=lab, color=c, marker=mk, ms=4, lw=1.2, capsize=2)
    f = s["floor"]["sw1"]
    ax.axhspan(f[1], f[2], color=GRAY, alpha=0.18, label="표본 잡음 바닥 (참 분포 두 표본)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("단계 수 S (선형 부분 수열, T=1000)"); ax.set_ylabel("슬라이스 W1 (낮을수록 좋음)")
    ax.set_title("정확한 ε*: 표집 품질 대 단계 수 (2차원 4봉우리 혼합)", fontsize=10)
    ascii_log(ax); ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "own_sweep.png"); plt.close(fig)


def fig_ode():
    o = R["ode_error"]
    Ss = [5, 10, 20, 50, 100, 200, 500, 1000]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(Ss, [o[str(S)]["ddim_median_err_to_ode"] for S in Ss], "o-", color=BLUE, label="DDIM (Δ(sbar) 오일러, 식 13)")
    ax.plot(Ss, [o[str(S)]["ddim_p90_err_to_ode"] for S in Ss], "o--", color=BLUE, alpha=0.5, label="DDIM 90분위")
    ref = o["10"]["ddim_median_err_to_ode"] * 10 / np.array(Ss)
    ax.plot(Ss, ref, ":", color=GRAY, label="∝ 1/S")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("단계 수 S"); ax.set_ylabel("같은 x_T의 연속 ODE 해와의 거리")
    ax.set_title("η=0 이산화 오차 (정확한 ε*): 중앙값 기울기 %.2f" % o["loglog_slope_ddim_median_S10_to_1000"], fontsize=10)
    ascii_log(ax); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "own_ode.png"); plt.close(fig)


def fig_recon():
    r = R["reconstruction"]
    Ss = [10, 20, 50, 100, 200, 500, 1000]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    e = [r[str(S)]["rel_mse"] for S in Ss]
    axs[0].plot(Ss, e, "o-", color=BLUE, label="재구성 오차 (η=0)")
    axs[0].plot(Ss, e[0] * 10 / np.array(Ss), ":", color=GRAY, label="∝ 1/S")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("단계 수 S"); axs[0].set_ylabel("평균제곱오차 / 데이터 분산")
    axs[0].set_title("인코딩-디코딩 오차 (기울기 %.2f)" % r["loglog_slope_rel_mse"], fontsize=10)
    ascii_log(axs[0]); axs[0].legend(fontsize=8, frameon=False)
    sw = [r[str(S)]["latent_sw1_to_N01"] for S in Ss]
    fl = r["latent_sw1_floor"]
    axs[1].plot(Ss, sw, "o-", color=ORANGE, label="인코딩된 x_T 대 N(0, I)")
    axs[1].axhspan(fl[1], fl[2], color=GRAY, alpha=0.18, label="표본 잡음 바닥")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("단계 수 S"); axs[1].set_ylabel("슬라이스 W1")
    axs[1].set_title("인코딩된 잠재 변수는 S가 작으면 사전분포가 아니다", fontsize=10)
    ascii_log(axs[1]); axs[1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "own_recon.png"); plt.close(fig)


if __name__ == "__main__":
    fig_sweep(); fig_ode(); fig_recon()
    print("saved", sorted(p.name for p in OUT.glob("own_*.png")))
