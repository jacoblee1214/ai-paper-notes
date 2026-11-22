"""analysis/*_results.json을 그래프 두 장으로 그린다(matplotlib). 실행: python3 analysis/plot_figures.py
출력: report/figures/own_human_study.png, report/figures/own_apps_feedback.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
GREY, ACCENT, INK = "#8a8880", "#1f5fae", "#1b1b1a"
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        plt.rcParams["font.family"] = "Apple SD Gothic Neo" if "Apple" in cand else "Noto Sans CJK KR"
        break
plt.rcParams["axes.unicode_minus"] = False

h = json.loads((ROOT / "analysis" / "recompute_human_study_results.json").read_text())
groups = [("입문", "introductory"), ("interview", "interview"), ("competition", "competition"), ("전체", "overall")]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
xs = range(len(groups))
g = [100 * h[k]["gpt4_rate"] for _, k in groups]
u = [100 * h[k]["human_rate"] for _, k in groups]
w = 0.36
b1 = ax.bar([x - w / 2 for x in xs], g, w, color=GREY, label="GPT-4 자기 피드백")
b2 = ax.bar([x + w / 2 for x in xs], u, w, color=ACCENT, label="사람 피드백")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(list(xs), [n for n, _ in groups], fontsize=9)
ax.set_ylabel("수리 후보 통과율 (%)", fontsize=9)
ax.set_ylim(0, 75)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("논문 Appendix E 재계산: 피드백 출처별 수리 통과율", fontsize=10, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_human_study.png")

a = json.loads((ROOT / "analysis" / "apps_feedback_results.json").read_text())["rows"]
keys = list(a.keys())
nps = [int(k.split(",")[0].split("=")[1]) for k in keys]
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
for name, col, lab in [("A_gpt35_self_repair_norm", GREY, "GPT-3.5 자기 피드백 (A)"),
                       ("B_gpt35_plus_gpt4_feedback_norm", ACCENT, "GPT-4 피드백 + GPT-3.5 수리 (B)")]:
    key = "A_ci95" if name.startswith("A") else "B_ci95"
    m = [a[k][name] for k in keys]
    lo = [a[k][key][0] for k in keys]
    hi = [a[k][key][1] for k in keys]
    ax.plot(nps, m, marker="o", color=col, label=lab)
    ax.fill_between(nps, lo, hi, color=col, alpha=0.15, linewidth=0)
ax.axhline(1.0, color=INK, linewidth=0.8, linestyle="--")
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 25], ["1", "2", "5", "10", "25"])
ax.set_xlabel("초기 기본 프로그램 수 np (nfr=1, 예산 k = 2np)", fontsize=9)
ax.set_ylabel("기준선 대비 정규화 값", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("APPS 300과제: 피드백 출처를 바꾸면 수리 이득이 달라진다 (95% 과제 부트스트랩)", fontsize=9, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_apps_feedback.png")
print("wrote", FIG)
