"""재계산 결과로 우리 그래프 세 장을 그린다 (report/figures/own_*.png).
실행: python3 -I analysis/plot_figures.py   (먼저 recompute_headline.py, recompute_irr.py)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures"); os.makedirs(OUT, exist_ok=True)
for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams["axes.unicode_minus"] = False
INK, GREY, BLUE, ORANGE = "#1b1b1a", "#8a8981", "#1f5fae", "#c8641b"
hd = json.load(open(os.path.join(HERE, "recompute_headline_results.json")))
irr = json.load(open(os.path.join(HERE, "recompute_irr_results.json")))

def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK); ax.grid(axis=grid, color="#e5e3dc", lw=0.7); ax.set_axisbelow(True)

# 1. 헤드라인 수치와 더 엄격한 정의로 다시 센 값 (분모 455행, Wilson 95%)
I, R = hd["items"], hd["raw_strict"]
pairs = [
    ("정의를 제시한 벤치마크", 78.2, ("phenomenon_defined=='Yes'", R["phenomenon_defined=='Yes'"]), "논문은 348/445, 행 455개 전체로는 348/455"),
    ("타당성을 논의한 벤치마크", 53.4, ("results_author_validity=='Yes'(엄격)", R["results_author_validity=='Yes'(엄격)"]), "'어느 정도·암묵적'을 뺀 'Yes'만"),
    ("실제 환경과 비교한 벤치마크", 31.2, ("results_realism=='Yes'(실제 비교)", R["results_realism=='Yes'(실제 비교)"]), "'벤치마크 자체가 현실적'을 뺀 비교만"),
    ("통계 검정을 한 벤치마크", 16.0, ("stats(Tests only)", I["stats(Tests only)"]["all455"]), "표준편차 보고를 뺀 검정만"),
    ("기존 자료를 재사용한 벤치마크", 38.2, ("src_reuse_any(bench|exam)", I["src_reuse_any(bench|exam)"]["all455"]), "일부+전부 재사용 (그림 3: 38.2+9.9)"),
]
fig, ax = plt.subplots(figsize=(9.4, 3.9)); clean(ax, "x")
for i, (lab, paper, (_, v), note) in enumerate(pairs):
    y = i
    ax.barh(y - 0.19, paper, 0.34, color=GREY)
    ax.barh(y + 0.19, v["pct"], 0.34, color=BLUE)
    lo, hi = v["wilson95"]
    ax.plot([lo, hi], [y + 0.19, y + 0.19], color=INK, lw=1.2)
    ax.text(paper + 1.2, y - 0.19, f"논문 {paper:.1f}", va="center", fontsize=8, color=INK)
    ax.text(max(hi, v["pct"]) + 1.2, y + 0.19, f"{v['pct']:.1f}  ({note})", va="center", fontsize=8, color=BLUE)
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p[0] for p in pairs], fontsize=9); ax.invert_yaxis()
ax.set_xlim(0, 118); ax.set_xlabel("퍼센트 (회색: 논문 보고, 파랑: 공개 코딩 자료 재계산 + Wilson 95% 구간)", fontsize=8.5)
ax.set_title("같은 코딩 자료에서 정의를 엄격히 하면 달라지는 헤드라인 수치", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_headline_vs_strict.png"), dpi=170); plt.close(fig)

# 2. 코딩 신뢰도: 표 9의 BPK와 같은 자료의 Cohen kappa
F = irr["fields"]
rows = [(k, v["paper_bpk"], v["cohen_kappa"]) for k, v in F.items() if v["cohen_kappa"] is not None]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(9.4, 5.6)); clean(ax, "x")
ys = range(len(rows))
ax.scatter([r[1] for r in rows], list(ys), color=GREY, s=26, label="논문 표 9의 BPK", zorder=3)
ax.scatter([r[2] for r in rows], list(ys), color=ORANGE, s=26, label="Cohen's kappa (공개 이중 코딩 자료, 41편)", zorder=3)
for y, r in zip(ys, rows): ax.plot([r[2], r[1]], [y, y], color="#d9d6cc", lw=0.8, zorder=1)
ax.axvline(0.4, color=INK, lw=0.8, ls=":"); ax.text(0.41, len(rows) - 0.6, "0.4", fontsize=8, color=INK)
ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlim(-0.25, 1.0); ax.legend(fontsize=8, loc="lower right", frameon=False)
ax.set_title("코딩 일치도: BPK(균등 우연 기준)와 Cohen kappa(관측 분포 기준)의 간격", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_irr_bpk_vs_kappa.png"), dpi=170); plt.close(fig)

# 3. 그림 3의 다섯 항목을 몇 개 만족하는가
dist = hd["n_best_practices_met_first445"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
clean(a1)
ks = sorted(int(k) for k in dist); vs = [dist[str(k)] for k in ks]
a1.bar(ks, vs, color=BLUE, width=0.65)
for k, v in zip(ks, vs): a1.text(k, v + 3, f"{v}\n({100*v/445:.1f}%)", ha="center", fontsize=8, color=INK)
a1.set_xlabel("만족한 항목 수 (정의 합의, 편의 표집 없음, 재사용 없음, 통계, 타당성 논의)", fontsize=7.5); a1.set_ylabel("벤치마크 수 (445)", fontsize=8.5)
a1.set_ylim(0, 200); a1.set_title("다섯 항목을 모두 만족: 6편", fontsize=9.5, color=INK, loc="left")
clean(a2, "x")
items = [("정의가 합의됨(정의 있음 중)", "defined&widely"), ("편의 표집 없음", "conv_none"), ("재사용 없음", "src_not_reused"), ("통계(표준편차·검정) 보고", "stats(Std|Tests)"), ("타당성 논의", "validity_discussed")]
for i, (lab, key) in enumerate(items):
    v = I[key]["first445"]; lo, hi = v["wilson95"]
    a2.barh(i, v["pct"], 0.55, color=BLUE); a2.plot([lo, hi], [i, i], color=INK, lw=1.2)
    a2.text(hi + 1.5, i, f"{v['pct']:.1f}%", va="center", fontsize=8, color=INK)
a2.set_yticks(range(len(items))); a2.set_yticklabels([x[0] for x in items], fontsize=8.5); a2.invert_yaxis(); a2.set_xlim(0, 85)
a2.set_title("항목별 만족 비율 (445편, Wilson 95%)", fontsize=9.5, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_best_practices.png"), dpi=170); plt.close(fig)
print("saved 3 figures")
