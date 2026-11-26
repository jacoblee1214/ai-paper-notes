"""우리가 그린 그래프 (matplotlib). 입력: derivation_results.json, toy_results.json.
실행: python3 analysis/plot_figures.py  -> figures/own_*.png  (논문 그림은 쓰지 않는다)"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
for name in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, GREY, GREEN, RED = "#0072b2", "#d55e00", "#8c8c8c", "#009e73", "#cc79a7"


def load(n):
    return json.loads((HERE / n).read_text())


def fig_derivation():
    d = load("derivation_results.json")
    fig = plt.figure(figsize=(12, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])
    titles = {"itc": "ITC: 단일 모달", "itg": "ITG: 다중 모달 인과", "itm": "ITM: 양방향"}
    for i, mode in enumerate(("itc", "itg", "itm")):
        ax = fig.add_subplot(gs[0, i])
        mk = d["masks_K3_T4"][mode]
        ax.imshow(mk["rows_see_cols"], cmap="Blues", vmin=0, vmax=1.6)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(mk["labels"], fontsize=7)
        ax.set_yticklabels(mk["labels"], fontsize=7)
        ax.xaxis.tick_top()
        ax.axhline(2.5, color="k", lw=0.8)
        ax.axvline(2.5, color="k", lw=0.8)
        ax.set_title(titles[mode], fontsize=9, y=1.18)
    ax = fig.add_subplot(gs[0, 3])
    p1 = d["params_stage1_bertbase_vitg_K32"]
    labels = ["질의용 FFN 없음", "질의용 FFN 있음\n(LAVIS 구조)", "논문"]
    vals = [p1["without_query_ffn_M"], p1["with_query_ffn_M"], p1["paper_M"]]
    cols = [GREY, BLUE, ORANGE]
    ax.bar(labels, vals, color=cols, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, f"{v:.1f}M", ha="center", fontsize=8)
    ax.set_ylim(0, 215)
    ax.set_ylabel("1단계 Q-Former 파라미터 (M)")
    ax.set_title("188M은 질의용 FFN이 있어야 맞는다", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_derivation.png", dpi=150)
    plt.close(fig)


CFG_LABEL = {"qf_K1_nos1": "Q-Former K=1", "qf_K4_nos1": "K=4", "qf_K16_nos1": "K=16", "qf_K32_nos1": "K=32",
             "mlp_pool16": "MLP 풀링\n(16토큰)", "mlp_all64": "MLP 전체\n(64토큰)", "mlp_all64_wide": "MLP 전체,\n파라미터 맞춤"}
CFG_COL = {"qf_K1_nos1": "#9ecae1", "qf_K4_nos1": "#6baed6", "qf_K16_nos1": "#3182bd", "qf_K32_nos1": "#08519c",
           "mlp_pool16": "#fdae6b", "mlp_all64": ORANGE, "mlp_all64_wide": "#a63603"}
TYPE_LABEL = {"count": "개수", "exist": "색 존재", "where": "위치(행, 열)", "row_is": "행 판별"}
CHANCE = {"count": 0.25, "exist": 0.708, "where": 1 / 64, "row_is": 0.5}   # exist는 "예"가 71%라 다수 답이 기준선


def fig_toy():
    r = load("toy_results.json")
    cfgs = [c for c in CFG_LABEL if c in r["configs"]]
    types = list(TYPE_LABEL)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    w = 0.8 / len(cfgs)
    for j, c in enumerate(cfgs):
        for i, t in enumerate(types):
            e = r["configs"][c][t]
            x = i + (j - (len(cfgs) - 1) / 2) * w
            ax[0].bar(x, e["mean"], w * 0.92, color=CFG_COL[c], yerr=[[e["mean"] - e["ci95"][0]], [e["ci95"][1] - e["mean"]]],
                      capsize=2, label=CFG_LABEL[c].replace("\n", " ") if i == 0 else None)
    for i, t in enumerate(types):
        ax[0].hlines(CHANCE[t], i - 0.45, i + 0.45, colors="k", linestyles=":", lw=1)
    ax[0].set_xticks(range(len(types)))
    ax[0].set_xticklabels([TYPE_LABEL[t] for t in types])
    ax[0].set_ylabel(f"정확도 (막대: {len(r['seeds'])}시드 평균, 95% 구간)")
    ax[0].set_title("연결기별 질문 유형 정확도 (1단계 없음; 점선 = 우연 또는 다수 답)", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8, ncol=3)
    ax[0].grid(axis="y", alpha=0.3)
    # 오른쪽: 1단계 효과와 학습량 통제 (4개 유형 평균 정확도, 점 = 시드)
    groups = [("K=4", [("qf_K4_nos1", "2단계만\n2000"), ("qf_K4_s1", "1단계 2500\n+ 2단계 2000")]),
              ("K=16", [("qf_K16_nos1", "2단계만\n2000"), ("qf_K16_s1", "1단계 2500\n+ 2단계 2000"), ("qf_K16_nos1_long", "2단계만\n4500")])]
    x, ticks, labels = 0, [], []
    cols = [BLUE, ORANGE, GREEN]
    for gname, items in groups:
        for j, (c, lab) in enumerate(items):
            if c not in r["configs"]:
                continue
            ys = r["configs"][c]["mean_acc"]["per_seed"]
            ax[1].bar(x, sum(ys) / len(ys), color=cols[j], alpha=0.45, width=0.8)
            ax[1].scatter([x + (i - len(ys) / 2) * 0.1 for i in range(len(ys))], ys, color="k", s=9, zorder=3)
            ticks.append(x)
            labels.append(lab)
            x += 1
        ax[1].text(x - len(items) / 2 - 0.5, 0.03, gname, ha="center", fontsize=9)
        x += 1
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("4개 유형 평균 정확도")
    ax[1].set_title("1단계 효과와 학습량 통제 (스텝 수)", fontsize=10)
    ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "own_toy_results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_derivation()
    if (HERE / "toy_results.json").exists():
        fig_toy()
