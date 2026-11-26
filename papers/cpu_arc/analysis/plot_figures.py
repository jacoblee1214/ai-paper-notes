"""CPU-ARC 분석의 직접 그린 그래프 두 장 (cost_check_results.json, stats_check_results.json에서 읽음).

own_token_price.png : 벤치마크별 캐시/원본 토큰 비와 토큰당 단가 비 (배수가 어디서 나오는가)
own_acc_intervals.png : 원본 대 캐시 정확도와 Wilson 95% 구간 (문항 수 표시)
실행: python3 -I plot_figures.py   (toy_multiplier.png는 toy_cache_model.py가 만든다)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "report", "figures")
os.makedirs(out, exist_ok=True)
FONT = next((f for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                         "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                         "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if os.path.exists(f)), None)
if FONT:
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
BLUE, RED, GREY = "#1b6ca8", "#b5483a", "#777777"

cost = json.load(open(os.path.join(here, "cost_check_results.json")))
st = json.load(open(os.path.join(here, "stats_check_results.json")))
names = list(cost["token_ratio"])

# 1) 토큰 비 대 단가 비
fig, ax = plt.subplots(figsize=(8.4, 3.8))
x = np.arange(len(names))
tr = [cost["token_ratio"][k]["cache_over_source_tokens"] for k in names]
sav = [cost["appendix_c"][k]["savings_x_from_tokens"] for k in names]
pr = [cost["token_ratio"][k]["price_per_token_ratio_source_over_cache"] for k in names]
w = 0.27
b1 = ax.bar(x - w, tr, w, color=RED, label="캐시 호출 토큰 / 원본 토큰 (1보다 크면 캐시가 토큰을 더 씀)")
b2 = ax.bar(x, sav, w, color=BLUE, label="논문의 비용 절감 배수 (Savings×)")
b3 = ax.bar(x + w, pr, w, color=GREY, label="토큰당 평균 단가 비 (원본 / 캐시)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("-Diamond", "") for n in names], rotation=20)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.axhline(1, color="#444", lw=0.8, ls=":")
ax.set_ylabel("배 (로그 눈금)")
ax.set_ylim(0.7, 6000)
ax.legend(fontsize=7.5, loc="upper left", frameon=False)
ax.set_title("배수는 토큰을 덜 써서가 아니라 토큰 단가 차이에서 옵니다")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(out, "own_token_price.png"), dpi=160)
plt.close(fig)

# 2) 정확도 구간
fig, ax = plt.subplots(figsize=(8.4, 4.0))
for i, k in enumerate(names):
    a = st["accuracy_intervals"][k]
    n = a["n_used"]
    for off, key, col in ((-0.12, "s_acc", GREY), (0.12, "c_acc", BLUE)):
        lo, hi = a[key]["wilson95"]
        ax.plot([lo, hi], [i + off, i + off], color=col, lw=2.2, solid_capstyle="butt")
        ax.plot([a[key]["pct"]], [i + off], "o", color=col, ms=4.5)
    ax.text(101.2, i, f"n={n}" + ("*" if k in ("MMLU-Pro", "SciBench") else ""), va="center", fontsize=8, color="#444")
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace("-Diamond", "") for n in names])
ax.invert_yaxis()
ax.set_xlim(60, 106)
ax.set_xlabel("정확도 % (점 = 표 2, 선 = Wilson 95%)")
ax.plot([], [], color=GREY, lw=2.2, label="원본 S Acc")
ax.plot([], [], color=BLUE, lw=2.2, label="캐시 C Acc")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("AIME는 30문항이라 구간이 겹칩니다 (*는 표본 수 미보고, 격자와 맞는 가장 작은 n)")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(out, "own_acc_intervals.png"), dpi=160)
print("ok")
