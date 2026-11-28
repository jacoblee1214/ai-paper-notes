"""우리가 그린 그래프를 figures/own_*.png 로 저장한다 (matplotlib). 논문 그림은 쓰지 않는다.

own_weights.png  진짜 VLB의 t별 가중 vs L_simple (식 12 vs 14), 변분 하한이 쌓이는 t 구간
own_sigma.png    sigma_t^2 = beta_t vs beta~_t: 슬라이스 W1 대 단계 수(넓은 혼합, 점질량에 가까운 혼합), 최적 sigma^2의 위치 f_t
own_floor.png    eps*의 줄일 수 없는 MSE 바닥, 섭동에 따른 손실 증가
own_gap.png      변분 하한과 참 엔트로피의 간격 대 단계 수
먼저 derivation_checks.py, toy_experiments.py를 실행해 둘 것.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from minimal_impl import Schedule, toy_mixture_1d
from toy_experiments import quad_mse_1d

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
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})

def fix_log(fig):
    """로그 축 눈금을 ASCII 지수 표기(1e-2)로 바꾼다(수학 글꼴의 음수 부호 문제 회피)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda v, _: "1e%d" % round(np.log10(v)) if v > 0 else "")
    for a in fig.axes:
        for axis, name in ((a.xaxis, "x"), (a.yaxis, "y")):
            if getattr(a, "get_%sscale" % name)() == "log":
                axis.set_major_formatter(fmt)
                axis.set_minor_formatter(NullFormatter())


T = json.loads((HERE / "toy_results.json").read_text())
D = json.loads((HERE / "derivation_checks_results.json").read_text())
sch = Schedule.linear()
t = np.arange(1, 1001)
w_beta = (sch.betas ** 2 / (2 * sch.betas * sch.alphas * (1 - sch.abar))).numpy()
w_bt = (sch.betas[1:] ** 2 / (2 * sch.beta_tilde[1:] * sch.alphas[1:] * (1 - sch.abar[1:]))).numpy()

# ---- 1. 가중 ----
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
ax[0].loglog(t, w_beta, color=BLUE, label=r"VLB 가중, $\sigma_t^2=\beta_t$")
ax[0].loglog(t[1:], w_bt, color=ORANGE, ls="--", label=r"VLB 가중, $\sigma_t^2=\tilde\beta_t$")
ax[0].loglog(t, 1 / (2 * t), color=GRAY, ls=":", label=r"$1/(2t)$ (작은 t 근사)")
ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$"); ax[0].set_title("식 (12)의 가중 (L_simple은 모든 t에서 1)"); ax[0].legend(fontsize=8)
ax[1].loglog(t, w_beta[-1] / w_beta, color=BLUE)
ax[1].axhline(1, color=GRAY, lw=0.8)
ax[1].axvline(100, color=RED, lw=0.8, ls="--")
ax[1].set_xlabel("t"); ax[1].set_ylabel("L_simple 대 VLB 상대 가중 (t=T에서 1로 정규화)")
ax[1].set_title("L_simple은 t<100을 최대 50배 낮춘다")
for name, col in (("broad", BLUE), ("tight", ORANGE)):
    lm = np.array(T["E4_vlb"][name]["T1000_by_t"]["beta"]["L_mid"])
    ax[2].semilogx(t[1:], np.cumsum(lm) / lm.sum(), color=col, label={"broad": "넓은 혼합", "tight": "점질량에 가까운 혼합"}[name])
ax[2].semilogx(t, np.cumsum(w_beta) / w_beta.sum(), color=GRAY, ls=":", label="VLB 가중 합(상수 MSE 가정)")
ax[2].semilogx(t, t / 1000, color="k", lw=0.6, ls="--", label="균일")
ax[2].set_xlabel("t"); ax[2].set_ylabel("sum L_{t'-1} (t'≤t) / 합"); ax[2].set_title("변분 하한(nat)이 쌓이는 t (정확한 eps*, T=1000)"); ax[2].legend(fontsize=8)
plt.tight_layout(); fix_log(fig); plt.savefig(OUT / "own_weights.png"); plt.close()

# ---- 2. sigma ----
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
steps = [10, 20, 50, 100, 250, 1000]
for k, (name, title) in enumerate((("broad", "넓은 혼합(좌표별 분산 1, 클러스터 폭 0.4)"), ("tight", "점질량에 가까운 혼합(클러스터 폭 0.04)"))):
    E1 = T["E1_sampler"][name]
    for mode, col, lab in (("beta", BLUE, r"$\sigma_t^2=\beta_t$"), ("beta_tilde", ORANGE, r"$\sigma_t^2=\tilde\beta_t$")):
        m = np.array([E1["steps"][str(s)][mode]["sw"]["mean"] for s in steps])
        lo = np.array([E1["steps"][str(s)][mode]["sw"]["lo"] for s in steps])
        hi = np.array([E1["steps"][str(s)][mode]["sw"]["hi"] for s in steps])
        ax[k].errorbar(steps, m, yerr=[m - lo, hi - m], color=col, marker="o", ms=4, capsize=2, label=lab)
    nl = E1["null_sw_data_vs_data"]
    ax[k].axhspan(nl["lo"], nl["hi"], color=GRAY, alpha=0.3, label="데이터 대 데이터(바닥)")
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel("표본 단계 수 S"); ax[k].set_ylabel("슬라이스 W1 (시드 8개, 95% 부트스트랩)")
    ax[k].set_title(title, fontsize=10); ax[k].legend(fontsize=8)
for name, col in (("1d", BLUE), ("2d", ORANGE)):
    d = D["I_sigma2"]["mixture_optimal_sigma2"][name]
    ts = sorted(int(x) for x in d)
    ax[2].semilogx(ts, [d[str(x)]["f_t"] for x in ts], marker="o", ms=4, color=col, label={"1d": "1차원 혼합", "2d": "2차원 넓은 혼합"}[name])
ax[2].axhline(1, color=BLUE, lw=0.8, ls=":"); ax[2].axhline(0, color=ORANGE, lw=0.8, ls=":")
ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$f_t=(\sigma^{*2}-\tilde\beta_t)/(\beta_t-\tilde\beta_t)$"); ax[2].set_title(r"최적 $\sigma^2$의 위치: 0=$\tilde\beta$, 1=$\beta$"); ax[2].legend(fontsize=8)
plt.tight_layout(); fix_log(fig); plt.savefig(OUT / "own_sigma.png"); plt.close()

# ---- 3. 바닥과 섭동 ----
data1 = toy_mixture_1d()
idx = list(range(0, 1000, 10))
qm = quad_mse_1d(data1, sch, idx)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
ax[0].semilogy(t, sch.abar.numpy(), color=GRAY, label=r"N(0,1): 정확한 바닥 $\bar\alpha_t$")
ax[0].semilogy(np.array(idx) + 1, qm, color=BLUE, label="1차원 혼합: 구적")
bins = T["E3_floor"]["mixture_1d_bins"]
ax[0].errorbar([(b["t_range"][0] + b["t_range"][1]) / 2 for b in bins], [b["mc"] for b in bins], yerr=[1.96 * b["se"] for b in bins],
               fmt="s", color=RED, ms=4, capsize=2, label="알고리즘 1 MC, t 100개 구간 평균(95% CI)")
ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$E\|\epsilon-\epsilon^*\|^2$"); ax[0].set_title("eps*에서 L_simple의 줄일 수 없는 값"); ax[0].legend(fontsize=8)
ax[0].set_ylim(1e-5, 2)
by = T["E3_floor"]["perturbation_1d"]["by_c"]
cs = sorted(by, key=float)
ax[1].errorbar([float(c) for c in cs], [by[c]["increase_mc"] for c in cs], yerr=[1.96 * by[c]["se"] for c in cs], fmt="o", color=RED, capsize=3, label="MC (같은 난수)")
xs = np.linspace(-0.22, 0.22, 100)
ax[1].plot(xs, xs ** 2 * T["E3_floor"]["perturbation_1d"]["E_eps_star_sq"], color=BLUE, label=r"$c^2\,E\|\epsilon^*\|^2$")
ax[1].set_xlabel("c  (예측기 = (1+c) eps*)"); ax[1].set_ylabel("L_simple 증가"); ax[1].set_title("eps*에서 어느 방향으로든 증가"); ax[1].legend(fontsize=8)
plt.tight_layout(); fix_log(fig); plt.savefig(OUT / "own_floor.png"); plt.close()

# ---- 4. 간격 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
for k, (name, title) in enumerate((("broad", "넓은 혼합"), ("tight", "점질량에 가까운 혼합"))):
    E4 = T["E4_vlb"][name]["by_steps"]
    for mode, col, lab in (("beta", BLUE, r"$\sigma_t^2=\beta_t$"), ("beta_tilde", ORANGE, r"$\sigma_t^2=\tilde\beta_t$")):
        g = np.array([E4[str(s)][mode]["gap_eq16_nat"] for s in steps])
        se = np.array([E4[str(s)][mode]["se_eq16_nat"] for s in steps])
        ax[k].errorbar(steps, g, yerr=1.96 * se, color=col, marker="o", ms=4, capsize=2, label=lab)
    ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel("단계 수 S"); ax[k].set_ylabel("L - H(x_0)  (nat, 2차원 전체)")
    ax[k].set_title(title); ax[k].legend(fontsize=8)
plt.tight_layout(); fix_log(fig); plt.savefig(OUT / "own_gap.png"); plt.close()
print("saved", sorted(p.name for p in OUT.glob("own_*.png")))
