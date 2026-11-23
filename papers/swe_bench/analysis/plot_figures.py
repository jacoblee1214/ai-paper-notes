"""recompute_results.json 으로 우리 그래프 세 장을 그린다 (report/figures/own_*.png).
실행: python3 -I analysis/plot_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)
for name in ("Noto Sans CJK KR", "Apple SD Gothic Neo", "AppleGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
d = json.load(open(os.path.join(HERE, "recompute_results.json")))
INK, GREY, BLUE, ORANGE = "#1b1b1a", "#8a8981", "#1f5fae", "#c8641b"


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(axis=grid, color="#e5e3dc", lw=0.7)
    ax.set_axisbelow(True)


# 1. 표 5 재계산: 전체 2,294문항 해결률과 Wilson 95% 구간
rows = sorted(d["table5"], key=lambda r: r["test"]["pct"])
fig, ax = plt.subplots(figsize=(7.4, 3.6))
clean(ax)
for i, r in enumerate(rows):
    t = r["test"]
    lo, hi = t["wilson95"]
    ax.barh(i, t["pct"], 0.55, color=BLUE)
    ax.plot([lo, hi], [i, i], color=INK, lw=1.4)
    ax.text(hi + 0.12, i, f"{t['resolved']}/2,294 = {t['pct']:.2f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["model"] for r in rows], fontsize=8.5)
ax.set_xlim(0, 6.2)
ax.set_xlabel("해결률 (%, BM25 검색, 공개 results.json에서 재계산)", color=INK)
ax.set_title("표 5는 Claude 2의 1.97을 빼고 모두 일치한다. 구간은 선, Opus와 Claude 2만 서로 분리된다", fontsize=9.2, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_table5.png"), dpi=170)
plt.close(fig)

# 2. Verified 500문항 대 나머지 1,794문항
v = d["verified_vs_rest"]
names = list(v.keys())
fig, ax = plt.subplots(figsize=(7.4, 3.6))
clean(ax, "y")
w = 0.38
for j, n in enumerate(names):
    ax.bar(j - w / 2, v[n]["pct_rest"], w * 0.95, color=GREY, label="나머지 1,794문항" if j == 0 else None)
    ax.bar(j + w / 2, v[n]["pct_verified"], w * 0.95, color=ORANGE, label="Verified 500문항" if j == 0 else None)
    ax.text(j + w / 2, v[n]["pct_verified"] + 0.12, f"p={v[n]['fisher_p']:.3f}" if v[n]["fisher_p"] >= 0.001 else "p<0.001",
            ha="center", fontsize=7.2, color=INK)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=7.6, rotation=12)
ax.set_ylabel("해결률 (%)", color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("같은 모델도 Verified 부분집합에서 2–4배 높게 풀린다 (Fisher 정확 검정)", fontsize=9.2, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_verified_effect.png"), dpi=170)
plt.close(fig)

# 3. 라벨 잡음 측정치 모음 (대상이 서로 달라 합산하지 않는다)
oa = d["verified_annotation_openai2024"]
items = [
    ("OpenAI 주석: 문제 기술 불명확\n(원 벤치마크 1,699문항 표본)", oa["under_pct"], oa["under_wilson95"], BLUE),
    ("OpenAI 주석: 테스트가 유효 해법을\n막을 수 있음", oa["tests_pct"], oa["tests_wilson95"], BLUE),
    ("OpenAI 주석: 위 둘 등으로 제외", oa["filtered_pct"], oa["filtered_wilson95"], BLUE),
    ("SWE-Bench+: 해법 유출\n(SWE-Agent+GPT-4 해결 패치)", 32.67, None, ORANGE),
    ("SWE-Bench+: 약한 테스트", 31.08, None, ORANGE),
    ("Wang 등 (ICSE 2026): 정답과 다르게 동작\n(Verified 해결 패치 877개, 차별 테스트)", 29.6, (26.7, 32.8), GREY),
    ("Wang 등: 개발자 테스트 전체에서 실패", 7.8, (6.2, 9.7), GREY),
]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
clean(ax)
for i, (lab, val, ci, col) in enumerate(items[::-1]):
    ax.barh(i, val, 0.6, color=col)
    if ci:
        ax.plot(ci, [i, i], color=INK, lw=1.3)
    ax.text((ci[1] if ci else val) + 1.0, i, f"{val:.1f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([x[0] for x in items[::-1]], fontsize=7.2)
ax.set_xlim(0, 80)
ax.set_xlabel("%  (선은 95% 구간, 출처마다 분모가 다르므로 더하지 않음)", color=INK)
ax.set_title("SWE-bench 판정기의 잡음을 재는 이후 연구들", fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "own_label_noise.png"), dpi=170)
plt.close(fig)
print("saved 3 figures to", OUT)
