"""우리가 그린 그래프 두 장 (논문 그림은 쓰지 않음).
  own_gain_decomposition.png : 그림 5의 이득을 Best-of-N(재시도만)이 설명하는 몫과 그 위의 AgentDebug 몫으로 분해
  own_step_distribution.png  : (a) 임계 단계 구간별 비율(HF 미러 200궤적), (b) 단계 정확도: 상수 추측 대 Direct Prompting 대 AgentDebug
실행: python3 analysis/plot_figures.py   입력: analysis/recompute_tables_results.json
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)
for FONT in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",):
    if os.path.exists(FONT):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
R = json.load(open(os.path.join(HERE, "recompute_tables_results.json"), encoding="utf-8"))

# 1. 이득 분해
m = R["fig5_alfworld"]["models"]
names = list(m)
bon = [m[k]["gain_pp"]["Best-of-N(5)"] for k in names]
ad = [m[k]["gain_pp"]["AgentDebug"] for k in names]
sr = [m[k]["gain_pp"]["Self-Refine"] for k in names]
fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=200)
x = range(len(names))
ax.bar(x, bon, color="#1f5fae", label="Best-of-N(5)가 설명하는 이득 (재시도만)")
ax.bar(x, [a - b for a, b in zip(ad, bon)], bottom=bon, color="#d9822b", label="AgentDebug가 그 위에 더하는 이득")
ax.scatter(x, sr, marker="D", color="black", zorder=3, label="Self-Refine 이득")
for i in x:
    ax.text(i, ad[i] + 1.6, f"+{ad[i]}p", ha="center", fontsize=9)
    ax.text(i, bon[i] / 2, f"{100*bon[i]/ad[i]:.0f}%", ha="center", color="white", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylabel("첫 시도 대비 성공률 이득 (포인트)")
ax.set_title("ALFWorld 그림 5: 이득 중 단순 재시도로 설명되는 몫", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", frameon=False)
ax.set_ylim(0, 42)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "그림 5 막대 값에서 계산. 하위 과제 표본 수 미보고(검정은 n=100 가정).", fontsize=6.5, color="#55544f")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(OUT, "own_gain_decomposition.png")); plt.close(fig)

# 2. 단계 분포와 상수 기준선
d = R["mirror_data"]
labels = ["1–5단계", "6–15단계", "16–20단계", "21단계 이후"]
def grp(b): return [sum(b[:5]), b[5] + b[6], b[7], b[8]]
envs = [("alfworld", "ALFWorld"), ("webshop", "WebShop"), ("gaia", "GAIA")]
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200)
w = 0.26
cols = ["#1f5fae", "#7aa6d8", "#d9822b"]
n = {"alfworld": 100, "webshop": 50, "gaia": 50}
for i, (k, nm) in enumerate(envs):
    g = grp(d["step_bins"][k])
    a.bar([j + (i - 1) * w for j in range(4)], [100 * v / n[k] for v in g], w, label=nm, color=cols[i])
a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8)
a.set_ylabel("임계 단계 비율 (%)")
a.set_title("(a) 임계 단계 위치 (HF 미러 200궤적)", fontsize=10)
a.legend(fontsize=8, frameon=False)
const = {k: d["constant_step_baseline_in_sample"][k]["acc_pct"] for k, _ in envs}
dp = {"alfworld": 28, "webshop": 30, "gaia": 26}
adv = {"alfworld": 35, "webshop": 42, "gaia": 58}
for i, (lab, vals, c) in enumerate([("상수 추측(표본 내 최적)", const, "#999999"), ("Direct Prompting", dp, "#7aa6d8"), ("AgentDebug", adv, "#d9822b")]):
    b.bar([j + (i - 1) * w for j in range(3)], [vals[k] for k, _ in envs], w, label=lab, color=c)
    for j, (k, _) in enumerate(envs):
        b.text(j + (i - 1) * w, vals[k] + 0.8, f"{vals[k]:.0f}", ha="center", fontsize=7.5)
b.set_xticks(range(3)); b.set_xticklabels([nm for _, nm in envs])
b.set_ylabel("단계 정확도 S (%)"); b.set_title("(b) 표 1의 S와 상수 추측", fontsize=10)
b.legend(fontsize=8, frameon=False); b.set_ylim(0, 68)
for ax_ in (a, b):
    for s in ("top", "right"): ax_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "own_step_distribution.png")); plt.close(fig)
print("ok")
