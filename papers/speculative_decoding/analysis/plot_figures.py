"""json 결과로 그래프 다섯 장을 ../figures/ 에 그린다 (논문 그림이 아니라 우리가 계산해 그린 것)."""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import minimal_impl as M

here = Path(__file__).parent
fig_dir = here.parent / "figures"
fig_dir.mkdir(exist_ok=True)
S = json.load(open(here / "synthetic_results.json"))
A = json.load(open(here / "real_alpha_results.json"))
D = json.load(open(here / "real_decode_results.json"))
T = json.load(open(here / "real_timing_results.json"))
P = json.load(open(here / "speedup_results.json"))
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
C1, C2, C3, GR = "#1f6fb4", "#d9660a", "#2a9d5c", "#555555"

# 1. 식 (1), Thm 3.8, Thm 3.11 재계산 (논문 Fig. 2-4에 해당)
al = np.linspace(0.5, 0.95, 91)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for g, col in zip((1, 3, 5, 7, 10), plt.cm.viridis(np.linspace(0.1, 0.85, 5))):
    ax[0].plot(al, [M.expected_tokens(a, g) for a in al], color=col, label=f"γ={g}")
    ax[2].plot(al, [M.walltime_factor(a, g, 0) for a in al], color=col, label=f"속도 γ={g}")
    ax[2].plot(al, [M.ops_factor(a, g, 0) for a in al], color=col, ls="--")
ax[0].set_xlabel("α"); ax[0].set_ylabel("반복당 기대 토큰 수 (식 1)"); ax[0].legend(frameon=False)
for c, col in zip((0.01, 0.02, 0.05, 0.1), (C1, C2, C3, GR)):
    ax[1].step(al, [M.optimal_gamma(a, c)[0] for a in al], where="post", color=col, label=f"c={c}")
ax[1].set_xlabel("α"); ax[1].set_ylabel("최적 γ (Thm 3.8)"); ax[1].legend(frameon=False)
ax[2].set_xlabel("α"); ax[2].set_ylabel("배수 (실선 속도, 점선 연산량; c = ĉ = 0)"); ax[2].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(fig_dir / "own_formulas.png", dpi=150); plt.close(fig)

# 2. 정확성: 경험 TV 대 귀무 TV (맞는 시나리오 8개와 틀린 변형 3개)
rows = S["single_token"]; wrong = S["single_token_wrong_variants"]
fig, ax = plt.subplots(figsize=(10, 4.2))
names = [r["scenario"] for r in rows] + list(wrong)
emp = [max(r["tv_emp_vs_p"], 1e-6) for r in rows] + [max(v["tv_emp_vs_p"], 1e-6) for v in wrong.values()]
null = [max(r["tv_null_q975"], 1e-6) for r in rows] + [rows[0]["tv_null_q975"]] * len(wrong)
x = np.arange(len(names))
ax.bar(x, emp, color=[C1] * len(rows) + [C2] * len(wrong), label="경험 분포와 p의 TV (N=2,000,000)")
ax.scatter(x, null, color="k", marker="_", s=300, label="p에서 직접 뽑았을 때 TV의 97.5% 분위수")
ax.set_yscale("log"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("TV (0은 1e-6으로 표시)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(fig_dir / "own_exactness.png", dpi=150); plt.close(fig)

# 3. 실모델 beta 분포
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for key, col, lab in (("target=70m,T=1.0", C1, "목표 70M"), ("target=160m,T=1.0", C2, "목표 160M")):
    h = np.array(A[key]["hist_10bins"]); ax[0].plot(np.linspace(0.05, 0.95, 10), h / h.sum(), "o-", color=col, label=f"{lab} (α={A[key]['alpha']:.2f})")
ax[0].set_xlabel("토큰별 β = Σ min(p, q)  (T=1)"); ax[0].set_ylabel("비율"); ax[0].legend(frameon=False)
Ts = [0, 0.5, 1.0]
for size, col in (("70m", C1), ("160m", C2)):
    a = [A[f"target={size},T={t}"]["alpha"] for t in Ts]; ci = np.array([A[f"target={size},T={t}"]["alpha_ci95_prompt_bootstrap"] for t in Ts])
    ax[1].errorbar(Ts, a, yerr=[np.array(a) - ci[:, 0], ci[:, 1] - np.array(a)], fmt="o-", color=col, capsize=3, label=f"목표 {size.upper()}")
ax[1].set_xlabel("온도 T"); ax[1].set_ylabel("α (프롬프트 부트스트랩 95%)"); ax[1].legend(frameon=False)
for t, col, mk in ((0, C3, "s"), (0.5, C2, "^"), (1.0, C1, "o")):
    ax[2].plot(np.arange(6) * 16 + 8, A[f"target=70m,T={t}"]["alpha_by_position_block"], mk + "-", color=col, label=f"T={t}")
ax[2].set_xlabel("생성 위치(토큰)"); ax[2].set_ylabel("α (목표 70M, 16토큰 블록)"); ax[2].legend(frameon=False)
fig.tight_layout(); fig.savefig(fig_dir / "own_real_alpha.png", dpi=150); plt.close(fig)

# 4. 받아들인 수 히스토그램과 위치별 beta
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a_, key, ttl in ((ax[0], "target=70m,T=1.0,gamma=5", "T=1, 목표 70M, γ=5"), (ax[1], "target=70m,T=0,gamma=5", "T=0(greedy), 목표 70M, γ=5")):
    r = D[key]; x = np.arange(len(r["n_hist_emp"]))
    a_.bar(x - 0.2, r["n_hist_emp"], 0.4, color=C1, label="관측"); a_.bar(x + 0.2, r["n_hist_geometric"], 0.4, color=C2, label="capped geometric (식 1)")
    a_.set_title(f"{ttl}\nTV={r['tv_n_hist_vs_geometric']:.3f}", fontsize=9); a_.set_xlabel("받아들인 추측 수 n"); a_.legend(frameon=False, fontsize=8)
for key, col, lab in (("target=70m,T=1.0,gamma=5", C1, "T=1"), ("target=70m,T=0,gamma=5", C3, "T=0")):
    pos = D[key]["beta_by_position"]; m = np.array([p["mean_beta"] for p in pos]); ci = np.array([p["ci95"] for p in pos])
    ax[2].errorbar(np.arange(1, 6), m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o-", color=col, capsize=3, label=lab)
ax[2].set_xlabel("추측 위치 j (방문한 반복만)"); ax[2].set_ylabel("평균 β"); ax[2].legend(frameon=False); ax[2].set_title("i.i.d.이면 평평해야 함", fontsize=9)
fig.tight_layout(); fig.savefig(fig_dir / "own_accept_hist.png", dpi=150); plt.close(fig)

# 5. gamma 스윕: 논문 가정(병렬 검증 공짜) 대 실측 검증 비용 반영
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, col, lab in (("target=70m,T=1.0", C1, "70M"), ("target=160m,T=1.0", C2, "160M")):
    rows = P[key]["rows"]; g = [r["gamma"] for r in rows]
    ax[0].plot(g, [r["s_paper"] for r in rows], "-", color=col, label=f"{lab}: Thm 3.8 (c={P[key]['c']:.2f})")
    ax[0].plot(g, [r["s_meas"] for r in rows], "--", color=col, label=f"{lab}: 실측 검증 비용 v(γ+1) 반영")
ax[0].axhline(1, color=GR, lw=1); ax[0].set_xlabel("γ"); ax[0].set_ylabel("예상 속도 배수 (T=1, α는 디코딩에서 추정)"); ax[0].legend(frameon=False, fontsize=7)
sw = P["sweep_70m_T1_observed"]; g = [s["gamma"] for s in sw]
ax[1].errorbar(g, [s["obs_tokens"] for s in sw], yerr=[[s["obs_tokens"] - s["obs_ci95"][0] for s in sw], [s["obs_ci95"][1] - s["obs_tokens"] for s in sw]], fmt="o", color=C1, capsize=3, label="관측 (프롬프트 60개)")
ax[1].plot(g, [s["eq1"] for s in sw], "s", color=C2, label="식 (1), α 관측값")
ax[1].set_xlabel("γ"); ax[1].set_ylabel("반복당 토큰 수 (목표 70M, T=1)"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(fig_dir / "own_speedup.png", dpi=150); plt.close(fig)
print("ok")
