"""saliency_toy_results.json, analysis_results.json, *_calib_size.json 으로 그래프 세 장을 ../figures/ 에 그린다."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
fig_dir = here.parent / "figures"
fig_dir.mkdir(exist_ok=True)
T = json.load(open(here / "saliency_toy_results.json"))["results"]
A = json.load(open(here / "analysis_results.json"))
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "Noto Sans CJK KR", "DejaVu Sans"], "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
C1, C2, C3, GR = "#1f6fb4", "#d9660a", "#2a8a4a", "#555555"

# 1. 토이: 오차비와 출력 오차
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
for key, c, lab in (("gauss/sym/W4", C1, "정규, 대칭 W4"), ("gauss/asym/W4", C2, "정규, zero-point W4"), ("student3/asym/W3", C3, "t(3), zero-point W3")):
    rows = T[key]
    s = [r["s"] for r in rows]
    ax[0].semilogx(s, [r["p_changed"] for r in rows], "o-", color=c, ms=4, label=lab)
    ax[1].loglog(s, [r["sal_rel"] for r in rows], "o-", color=c, ms=4, label=lab)
    ax[2].loglog(s, [r["out_mse"] for r in rows], "o-", color=c, ms=4, label=lab)
ax[1].loglog(s, [1 / x for x in s], "--", color=GR, label="1/s (식 2, Δ 불변)")
ax[2].axhline(1, color=GR, lw=0.8)
ax[0].set_title("그룹 최댓값이 바뀐 비율 (Δ' ≠ Δ)"); ax[1].set_title("s배 한 채널의 상대 오차 (s=1 기준)"); ax[2].set_title("출력 MSE (그 채널 활성 10배, s=1 기준)")
for a in ax:
    a.set_xlabel("s")
ax[0].legend(fontsize=8)
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(fig_dir / "own_saliency_toy.png", dpi=140); plt.close(fig)

# 2. 모델별 perplexity (W4, W3)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, (m, title) in zip(ax, (("gpt2", "GPT-2 small"), ("pythia160m", "Pythia-160M"))):
    d = A[m]
    names = ["rtn_w3", "fp16_random_0.01_w3", "fp16_weight_0.01_w3", "awq_w3", "fp16_act_0.01_w3", "awq_clip_w3", "rtn_w4", "awq_w4", "awq_clip_w4"]
    labels = ["RTN W3", "FP16 1%\n무작위 W3", "FP16 1%\n가중치 W3", "AWQ W3", "FP16 1%\n활성 W3", "AWQ+clip\nW3", "RTN W4", "AWQ W4", "AWQ+clip\nW4"]
    v = np.array([d["point"][n] for n in names])
    lo = np.array([d["ci"][n][0] for n in names]); hi = np.array([d["ci"][n][1] for n in names])
    cols = [C1 if "rtn" in n else (C3 if "fp16" in n else C2) for n in names]
    a.bar(range(len(names)), v, color=cols, yerr=[v - lo, hi - v], capsize=2)
    a.axhline(d["point"]["fp32"], color=GR, ls="--", lw=1, label=f"fp32 {d['point']['fp32']:.1f}")
    a.set_yscale("log"); a.set_xticks(range(len(names))); a.set_xticklabels(labels, fontsize=7)
    a.set_title(f"{title}: pile-10k 보류 문서 perplexity (그룹 128)"); a.legend(fontsize=8); a.set_ylabel("perplexity (로그 눈금, 95% 문서 부트스트랩)")
fig.tight_layout(); fig.savefig(fig_dir / "own_lm_ppl.png", dpi=140); plt.close(fig)

# 3. s 스윕(Table 2형)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
for m, c, lab in (("gpt2", C1, "GPT-2"), ("pythia160m", C2, "Pythia-160M")):
    d = A[m]
    ss = ["1.0", "1.25", "1.5", "2.0", "4.0"]
    ax[0].plot([float(s) for s in ss], [d["point"][f"s_sweep_{s}_w3"] for s in ss], "o-", color=c, label=lab)
    ax[1].plot([float(s) for s in ss], [d["table2"][s]["err_ratio"] for s in ss], "o-", color=c, label=lab)
ax[1].plot([1, 1.25, 1.5, 2, 4], [1, 0.804, 0.676, 0.519, 0.303], "s--", color=GR, label="논문 Table 2 (OPT-6.7B)")
ax[0].set_title("활성 상위 1% 채널만 s배: W3 perplexity"); ax[0].set_yscale("log"); ax[1].set_title("평균 Δ'/Δ × 1/s")
for a in ax:
    a.set_xlabel("s"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(fig_dir / "own_s_sweep.png", dpi=140); plt.close(fig)
