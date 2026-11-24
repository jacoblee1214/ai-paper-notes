"""직접 그리는 그림 4장 (report/figures/own_*.png). 논문 그림은 쓰지 않는다.

입력: metric_check_results.json, tables_recompute_results.json, sink_toy_results.json
실행: python3 -I analysis/plot_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for name in ["Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic", "NanumGothic"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
INK, BLUE, RED, GREY, GREEN = "#1b1b1a", "#1f5fae", "#c0392b", "#8a8984", "#2e7d4f"

mc = json.loads((ROOT / "analysis" / "metric_check_results.json").read_text())
tr = json.loads((ROOT / "analysis" / "tables_recompute_results.json").read_text())

# 1. 균일 기준선 H_T/T 와 임계값에 대응하는 평균 첫 토큰 질량 w*
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
Ts = np.array([r["T"] for r in mc["uniform_baseline"]])
base = np.array([r["uniform_alpha1"] for r in mc["uniform_baseline"]])
ax[0].plot(Ts, base, "o-", color=BLUE)
ax[0].axhline(0.3, color=RED, ls="--", lw=1)
ax[0].text(1100, 0.31, "ε=0.3", color=RED, fontsize=9)
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("T (토큰 수)")
ax[0].set_ylabel("균일 attention 헤드의 α1 = H_T / T")
ax[0].set_title("sink가 없어도 T가 작으면 α1이 큼", fontsize=10)
for T_, b in zip(Ts, base):
    if T_ in (8, 64, 1024):
        ax[0].annotate(f"{b:.3f}", (T_, b), textcoords="offset points", xytext=(5, 6), fontsize=8)
for eps, c in zip([0.1, 0.2, 0.3, 0.4, 0.5], ["#9bb7d8", "#6e97c9", BLUE, "#154a8a", "#0c2f58"]):
    pts = [(r["T"], r["w_star"]) for r in mc["w_star"] if r["eps"] == eps]
    ax[1].plot(*zip(*pts), "o-", color=c, label=f"ε={eps}")
ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("T")
ax[1].set_ylabel("임계값에 해당하는 첫 토큰 평균 질량 w*")
ax[1].set_title("w* = (εT-1)/(T-1)", fontsize=10)
ax[1].legend(fontsize=8, frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_metric_baseline.png", dpi=170)
plt.close(fig)

# 2. 주장별 효과 대 노이즈 대용 척도
cl = tr["claim_effects_vs_null_sd"]
keys = ["wd 0.0 대 0.1 (표 2)", "wd 0.1 대 0.5 (표 2)", "lr x steps = 8 일정: 4e-4/20k 대 2e-4/40k (표 9)",
        "lr x steps = 8 일정: 4e-4/20k 대 1e-4/80k (표 9)", "lr 4e-4 대 2e-4, 20k (표 9)",
        "normalization scale 1.0 대 0.2 (표 14 왼쪽)", "softmax 18.18 대 sigmoid 6개 평균 (표 14 오른쪽)",
        "1B: softmax 45.11 대 sigmoid 2.46 (본문 7.4절)"]
labels = ["weight decay 0.0→0.1", "weight decay 0.1→0.5", "lr×스텝 고정 4e-4→2e-4", "lr×스텝 고정 4e-4→1e-4",
          "lr 4e-4→2e-4 (20k)", "softmax 정규화 척도 1→0.2", "softmax→sigmoid (60M, 6설정 평균)", "softmax→sigmoid (1B)"]
vals = [abs(cl[k]["diff_pp"]) for k in keys]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
y = np.arange(len(keys))[::-1]
ax.axvspan(0, 1.54, color="#dfe9f5", label="낮은 노이즈 SD 1.54")
ax.axvspan(1.54, 4.45, color="#f3ece0", label="높은 노이즈 SD 4.45")
ax.barh(y, vals, color=[RED if cl[k]["z_high"] < 1 else BLUE for k in keys], height=0.55)
for yi, v in zip(y, vals):
    ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("|ΔSink1| (퍼센트 포인트, 논문 표 값)")
ax.set_title("논문은 시드 변동을 보고하지 않음: 'null 변형'의 흩어짐(음영)과 비교", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_effects_vs_noise.png", dpi=170)
plt.close(fig)

# 3. 정규화 유무: 논문 표 6, 14 값
fig, ax = plt.subplots(figsize=(7.4, 3.5))
names = ["softmax\n(표 6)", "sigmoid\n정규화 없음", "sigmoid\n정규화 있음", "elu+1\n정규화 없음", "elu 커널\n정규화 있음",
         "MLP 커널\n정규화 있음"]
vals = [18.18, 0.44, 30.24, 0.80, 53.65, 0.19]
loss = [3.73, 3.70, 3.74, 3.69, 4.19, 3.85]
cols = [GREY, GREEN, BLUE, GREEN, BLUE, GREEN]
ax.bar(names, vals, color=cols)
for i, (v, l) in enumerate(zip(vals, loss)):
    ax.text(i, v + 1.2, f"{v:.2f}%\n손실 {l:.2f}", ha="center", fontsize=8)
ax.set_ylim(0, 66)
ax.set_ylabel("Sink1 (%, ε=0.3, T=64)")
ax.set_title("정규화를 없애면 사라지고 다시 넣으면 돌아온다 (논문 값, 실행 1회, * 표는 proxy 점수)", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "own_normalization.png", dpi=170)
plt.close(fig)

# 4. 토이
p = ROOT / "analysis" / "sink_toy_results.json"
if p.exists():
    toy = json.loads(p.read_text())
    kinds = ["softmax", "sigmoid_nonorm", "sigmoid_norm"]
    names = {"softmax": "softmax", "sigmoid_nonorm": "sigmoid 정규화 없음 (proxy)", "sigmoid_norm": "sigmoid 정규화 있음"}
    cols = {"softmax": GREY, "sigmoid_nonorm": GREEN, "sigmoid_norm": BLUE}
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for k in kinds:
        runs = [r for r in toy["runs"] if r["kind"] == k]
        steps = [h["step"] for h in runs[0]["hist"]]
        for key, a, ttl in [("alpha1_mean", ax[0], "헤드 평균 α1 (기준선 H_T/T = 0.074)"),
                            ("val_loss", ax[1], "검증 손실 (바이트당 nat)")]:
            M = np.array([[h[key] for h in r["hist"]] for r in runs])
            a.plot(steps, M.mean(0), color=cols[k], label=names[k])
            a.fill_between(steps, M.min(0), M.max(0), color=cols[k], alpha=0.18)
            a.set_title(ttl, fontsize=9.5)
    ax[0].axhline(0.0741, color=INK, ls=":", lw=1)
    ax[1].set_ylim(1.0, 2.5)
    eps = ["0.1", "0.2", "0.3", "0.4", "0.5"]
    w = 0.27
    for i, k in enumerate(kinds):
        runs = [r for r in toy["runs"] if r["kind"] == k]
        m = [np.mean([r["final"]["sink"][e] for r in runs]) for e in eps]
        s = [np.std([r["final"]["sink"][e] for r in runs]) for e in eps]
        ax[2].bar(np.arange(5) + (i - 1) * w, m, w, yerr=s, color=cols[k], label=names[k], capsize=2)
    ax[2].set_xticks(range(5))
    ax[2].set_xticklabels([f"ε={e}" for e in eps], fontsize=8)
    ax[2].set_title("마지막 스텝 Sink1 (%)", fontsize=9.5)
    ax[0].legend(fontsize=7.5, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_xlabel("학습 스텝")
    ax[1].set_xlabel("학습 스텝")
    fig.suptitle("토이(4층, 폭 128, 바이트 코퍼스, 시드 3개; 선 평균, 음영 최소-최대)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_curves.png", dpi=170)
    plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("own_*.png")))
