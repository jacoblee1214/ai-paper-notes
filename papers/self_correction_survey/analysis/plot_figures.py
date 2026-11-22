"""recheck_cases_results.json 으로 그래프 한 장을 그린다(matplotlib). 실행: python3 analysis/plot_figures.py
출력: report/figures/own_oracle_gap.png  (같은 벤치마크에서 정답 신호 유무에 따른 이득)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
INK, ACCENT, GREY, TOOL = "#1b1b1a", "#1f5fae", "#8a8880", "#c4572a"
for cand in ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(cand).exists():
        plt.rcParams["font.family"] = "Apple SD Gothic Neo" if "Apple" in cand else "Noto Sans CJK KR"
        break
plt.rcParams["axes.unicode_minus"] = False

r = json.loads((ROOT / "analysis" / "recheck_cases_results.json").read_text(encoding="utf-8"))
rows = []  # (그룹, 라벨, oracle, tool, none)
for x in r["A_rci_rows"]:
    rows.append(("RCI (GPT-3.5, n=254)", x["bench"], x["oracle_stop_gain"]["delta_pp"], None, x["no_feedback_gain"]["delta_pp"]))
for m in ["GPT-3.5", "GPT-4"]:
    for ds in ["GSM8K", "CommonSenseQA", "HotpotQA"]:
        e = r["E_huang"][f"{m}/{ds}"]
        rows.append(("Huang et al.", f"{m} {ds}", e["oracle"]["delta_pp"], None, e["intrinsic_r1"]["delta_pp"]))
for ds in ["AmbigNQ", "TriviaQA", "HotpotQA"]:
    e = r["D_critic_qa_n500"][f"gpt-3.5-turbo/{ds}"]
    rows.append(("CRITIC (ChatGPT)", ds, e["CRITIC_oracle"]["delta_pp"], e["CRITIC"]["delta_pp"], e["CRITIC_wo_tool"]["delta_pp"]))
for ds in ["GSM8k", "SVAMP"]:
    e = r["D_critic_math"][f"gpt-3.5-turbo/{ds}"]
    rows.append(("CRITIC (ChatGPT)", ds, e["CRITIC_oracle"]["delta_pp"], e["CRITIC"]["delta_pp"], e["CRITIC_wo_tool"]["delta_pp"]))
for d in r["C_selfrefine_math_table1_vs_table9"]:
    rows.append(("Self-Refine 수학", d["model"], d["oracle_gain_pp"], None, d["no_oracle_gain_pp"]))

fig, ax = plt.subplots(figsize=(7.4, 8.6), dpi=160)
y = 1.9
ticks, labels, prev = [], [], None
for g, lab, o, t, n in rows:
    if prev is not None and g != prev:
        y += 0.7
    if g != prev:
        ax.text(-39.5, y - 0.85, g, fontsize=8.5, color=INK, fontweight="bold", va="center")
        y += 0.6
    prev = g
    xs = [v for v in (o, t, n) if v is not None]
    ax.plot([min(xs), max(xs)], [y, y], color="#cfcdc6", lw=1.4, zorder=1)
    ax.scatter([o], [y], marker="o", s=34, color=ACCENT, zorder=3)
    if t is not None:
        ax.scatter([t], [y], marker="s", s=30, color=TOOL, zorder=3)
    ax.scatter([n], [y], marker="D", s=30, facecolor="white", edgecolor=GREY, linewidth=1.6, zorder=3)
    ticks.append(y); labels.append(lab)
    y += 1
ax.set_yticks(ticks, labels, fontsize=7.5)
ax.invert_yaxis()
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlim(-40, 28)
ax.set_xlabel("기준선 대비 이득 (퍼센트포인트)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=8)
ax.scatter([], [], marker="o", color=ACCENT, label="정답 신호로 멈춤 또는 틀린 답만 고침 (오라클)")
ax.scatter([], [], marker="s", color=TOOL, label="외부 도구 피드백 (CRITIC)")
ax.scatter([], [], marker="D", facecolor="white", edgecolor=GREY, linewidth=1.6, label="자기 피드백만 (정답 신호 없음)")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.07))
ax.set_title("같은 벤치마크에서 정답 신호를 빼면 이득이 사라진다 (원 논문 표 재계산)", fontsize=10, loc="left", color=INK)
fig.tight_layout()
fig.savefig(FIG / "own_oracle_gap.png")
print("wrote", FIG / "own_oracle_gap.png")
