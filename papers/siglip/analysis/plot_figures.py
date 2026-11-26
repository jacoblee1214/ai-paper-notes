"""우리가 그린 그래프 두 장 (논문 그림은 쓰지 않는다).
  own_derivation.png : (a) 초기 편향 기울기와 초기 손실 대 b (B=32768), (b) 최적 로짓 분리 (직교 임베딩), (c) 쌍 수와 메모리 산술
  own_toy_results.png: 합성 도형 토이의 softmax 대 sigmoid (배치 크기별 정확도, 두 regime)
  python3 analysis/plot_figures.py  (derivation_results.json, toy_results.json 필요)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for f in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
C = {"softmax": "#1f77b4", "sigmoid": "#d62728", "sigmoid_b0": "#7f7f7f"}


def sig(x):
    return 1 / (1 + math.exp(-x))


def softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0)


def derivation():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    B = 32768
    bs = [x / 10 for x in range(-150, 31)]
    g = [-sig(-b) + (B - 1) * sig(b) for b in bs]
    ax[0].plot(bs, [math.log10(max(abs(v), 1e-3)) for v in g], color=C["sigmoid"])
    ax[0].axvline(-math.log(B - 1), ls="--", color="k", lw=0.8)
    ax[0].axvline(-10, ls=":", color="gray")
    ax[0].set_xlabel("편향 b (코드 부호)")
    ax[0].set_ylabel("log10 |dL/db| (행당, 로짓 = b로 근사, 하한 -3)")
    ax[0].set_title("(a) B=32768 초기 편향 기울기\n점선: 균형점 -ln(B-1) = -10.4, 회색: -10")
    d = json.loads((HERE / "derivation_results.json").read_text())["optimal_logits"]
    for t, mk in ((10.0, "o"), (20.0, "s"), (100.0, "^")):
        rows = [r for r in d if r["t"] == t]
        ax[1].plot([r["B"] for r in rows], [r["logit_pos"] for r in rows], mk + "-", color=C["sigmoid"], alpha=0.4 + 0.3 * (t > 10), label=f"양성 t={t:g}")
        ax[1].plot([r["B"] for r in rows], [r["logit_neg"] for r in rows], mk + "--", color=C["softmax"], alpha=0.4 + 0.3 * (t > 10), label=f"음성 t={t:g}")
    ax[1].set_xscale("log")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("배치 B")
    ax[1].set_ylabel("최적 로짓 (직교 임베딩)")
    ax[1].set_title("(b) 최적에서 양성/음성 로짓\n간격 = t, 중점 = -ln(B-1)/2")
    ax[1].set_ylim(-70, 80)
    ax[1].legend(fontsize=6, ncol=3, loc="upper center")
    Bs = [2**k for k in range(10, 21)]
    ax[2].plot(Bs, [math.log10(b * b * 4 / 2**30) for b in Bs], color="k", label="전체 |B|x|B|")
    ax[2].plot(Bs, [math.log10(b * b / 256 * 4 / 2**30) for b in Bs], color=C["softmax"], label="CLIP식 장치별 m x B (D=256)")
    ax[2].plot(Bs, [math.log10((b / 256) ** 2 * 4 / 2**30) for b in Bs], color=C["sigmoid"], label="청크 m x m (D=256)")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("배치 B")
    ax[2].set_ylabel("log10 로짓 메모리 (GiB, float32)")
    ax[2].set_title("(c) 청크 계산의 메모리 산술")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "own_derivation.png", dpi=150)


def toy():
    r = json.loads((HERE / "toy_results.json").read_text())
    cells = r["cells"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    specs = [("fixed_steps", "zs_held_out_photo", "(a) 스텝 고정 2500 (본 예제 수가 배치에 비례)"),
             ("matched_examples", "zs_held_out_photo", "(b) 본 예제 수 고정 32만"),
             ("matched_examples", "zs_seen_photo", "(c) 같은 설정, 본 조합 정확도")]
    for a, (reg, m, title) in zip(ax, specs):
        for loss in ("softmax", "sigmoid", "sigmoid_b0"):
            pts = [(int(k.split("|")[2]), v[m]) for k, v in cells.items() if k.startswith(reg + "|" + loss + "|")]
            if not pts:
                continue
            pts.sort()
            x = [p[0] for p in pts]
            a.errorbar(x, [p[1][0] for p in pts], yerr=[[p[1][0] - p[1][1] for p in pts], [p[1][2] - p[1][0] for p in pts]],
                       fmt="o-", color=C[loss], capsize=3, label={"softmax": "softmax(CLIP)", "sigmoid": "sigmoid(b=-10)", "sigmoid_b0": "sigmoid(b=0)"}[loss])
        a.set_xscale("log", base=2)
        a.set_xlabel("배치 크기")
        a.set_title(title, fontsize=9)
    ax[0].set_ylabel(f"제로샷 정확도 (평균, t 95% 구간, {r['n_seeds']}시드)")
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_results.png", dpi=150)


if __name__ == "__main__":
    derivation()
    if (HERE / "toy_results.json").exists():
        toy()
