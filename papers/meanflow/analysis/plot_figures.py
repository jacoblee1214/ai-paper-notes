"""분석 결과 JSON으로 그래프 네 장을 그린다 (report/figures/own_*.png)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for name in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "dejavusans"
C = ["#1f5fae", "#d9822b", "#2a9d6f", "#b5443b", "#7a5ea8", "#555555"]
from matplotlib.ticker import FuncFormatter
_fmt = FuncFormatter(lambda v, _: "1e%d" % round(np.log10(v)) if v > 0 else "")


def save(fig, name):
    for a in fig.axes:
        if a.get_xscale() == "log":
            a.xaxis.set_major_formatter(_fmt)
        if a.get_yscale() == "log":
            a.yaxis.set_major_formatter(_fmt)
    fig.tight_layout(); fig.savefig(FIG / name, dpi=160); plt.close(fig)


L = lambda n: json.loads((HERE / n).read_text())

# 1. 항등식 잔차 + 접선 변형
r = L("identity_results.json")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for (k, row), c in zip(r["identity_residual_vs_rk4_steps"].items(), C):
    Ns = [int(n) for n in row]
    ax[0].loglog(Ns, [row[n]["max_resid"] for n in row], "o-", color=c, label=k)
Ns = np.array([2, 4, 8, 16, 32, 64, 128.0])
ax[0].loglog(Ns, 3 * Ns ** -4.0 * 100, "k:", label="기울기 -4 (RK4)")
ax[0].set_xlabel("RK4 스텝 수 N (u 계산)"); ax[0].set_ylabel("max |d/dt u - (v-u)/(t-r)|")
ax[0].set_title("항등식 잔차는 적분 오차로 수렴"); ax[0].legend(fontsize=7)
tv = r["tangent_variants_target_error"]
names = list(tv)
vals = [tv[n]["rms_target_error"] for n in names]
ax[1].bar(names, vals, color=[C[2]] + [C[3]] * 3)
ax[1].set_yscale("log"); ax[1].set_ylabel("목표 u_tgt의 RMS 오차 (정확한 u 대비)")
ax[1].set_title("접선 (v,0,1)만 목표가 정확 (표 1b 대응)")
for i, v in enumerate(vals):
    ax[1].text(i, v * 1.3, f"{v:.1e}", ha="center", fontsize=8)
save(fig, "own_identity.png")

# 2. 한 스텝 정확 u 대 Euler
o = L("onestep_results.json")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
Rs = [float(k.split("=")[1]) for k in o]
for (k, row), c in zip(o.items(), C):
    Ns = [1, 2, 4, 8, 16, 64]
    ax[0].loglog(Ns, [row[f"euler{n}_sw2_vs_ode"] for n in Ns], "o-", color=c, label=f"R={row['mean_sep_R']:g}")
ax[0].axhline(0.03, color="k", ls=":", lw=1)
ax[0].text(1.1, 0.034, "점선: 표본 2000개 sliced W2의 잡음 수준(약 0.03). 정확한 u 한 스텝은 ODE 해와 일치", fontsize=7)
ax[0].set_xlabel("Euler 스텝 수 (v 사용)"); ax[0].set_ylabel("sliced W2 (대 정확 ODE 해)")
ax[0].set_title("v로 한 스텝은 분포를 못 맞춘다"); ax[0].legend(fontsize=7, title="평균 간격")
ax[1].plot(Rs, [o[k]["euler1_pointwise_rms"] for k in o], "o-", color=C[3], label="Euler 1스텝 점별 RMS 오차")
ax[1].plot(Rs, [o[k]["S_velocity_deviation"] ** 0.5 for k in o], "s--", color=C[0], label="sqrt(S) 속도 편차(직진도 지표)")
ax[1].plot(Rs, [o[k]["euler1_output_std"] for k in o], "^-", color=C[2], label="Euler 1스텝 출력의 표준편차(=0)")
ax[1].set_xlabel("혼합 성분 평균 간격 R"); ax[1].set_title("Euler 1스텝 출력은 데이터 평균 한 점"); ax[1].legend(fontsize=7)
save(fig, "own_onestep.png")

# 3. 목표 연산자의 주파수 반응
t = L("target_fixedpoint_results.json")
fr = t["frequency_response_(rho=pointwise contraction, q=semi-gradient quadratic form)"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
oms = [0.5, 1, 2, 4, 8, 16, 32]
for (k, row), c in zip(fr.items(), C):
    ax[0].loglog(oms, [row[str(o_)]["tau_wave"]["rho_rms"] for o_ in oms], "o-", color=c, label=k)
    ax[1].semilogx(oms, [row[str(o_)]["tau_wave"]["q"] for o_ in oms], "o-", color=c)
ax[0].axhline(1, color="k", ls=":")
ax[0].set_xlabel("오차의 (t-r) 방향 주파수 ω"); ax[0].set_ylabel("점별 수축비 ρ = |(t-r) D e| / |e|")
ax[0].set_title("정확한 고정점 반복은 수축이 아님"); ax[0].legend(fontsize=6)
ax[1].axhline(0, color="k", ls=":")
ax[1].set_ylim(-0.2, 2.2)
ax[1].set_xlabel("오차의 (t-r) 방향 주파수 ω"); ax[1].set_ylabel("q = <e, e + (t-r) D e> / <e, e>")
ax[1].set_title("반경사 이차형식은 모든 ω에서 양 (q > 0)")
save(fig, "own_target.png")

# 4. 절제 표 차이와 시드 잡음 가정
tb = L("tables_results.json")["ablation_vs_noise"]
fig, ax = plt.subplots(figsize=(10, 3.9))
labels, rel, flag1, flag2 = [], [], [], []
for name, v in tb["CoV=0.02"].items():
    if name.startswith(("1b", "1f", "1a")) and False:
        continue
    for row, row1 in zip(v["rows"], tb["CoV=0.01"][name]["rows"]):
        labels.append(f"{name.split()[0]} {row['item']}")
        rel.append(row["rel_diff"] * 100)
        flag2.append(row["distinguishable"]); flag1.append(row1["distinguishable"])
idx = [i for i, r_ in enumerate(rel) if r_ < 60]
x = np.arange(len(idx))
col = [C[2] if flag2[i] else (C[1] if flag1[i] else C[3]) for i in idx]
ax.bar(x, [rel[i] for i in idx], color=col)
ax.set_xticks(x); ax.set_xticklabels([labels[i] for i in idx], rotation=75, fontsize=7)
ax.set_ylabel("표 내 최저 FID 대비 상대 차이 (%)")
ax.axhspan(0, 2.0 * np.sqrt(2) * 1.96, color="#cccccc", alpha=0.4)
ax.text(0.2, 7.2, "회색: 시드 CoV 2%일 때 두 실행 차의 95% 잡음대(약 5.5%)", fontsize=7)
ax.set_title("절제 표 차이 대 학습 시드 잡음 가정 (초록: CoV 2%에서도 구별, 주황: CoV 1%에서만, 빨강: 둘 다 구별 안 됨; 상대차 60% 미만만 표시)", fontsize=8)
save(fig, "own_noise.png")
print("ok")
