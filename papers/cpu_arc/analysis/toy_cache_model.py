"""CPU-ARC 헤드라인 배수가 어떤 조건에서 성립하는지 보는 numpy 비용·정확도 토이 (CPU, 1초).

모형: K개 진입점(문제 템플릿)이 각각 r번 재등장한다(총 M = K*r 질의). 각 템플릿의 첫 질의는 원본 모델 S가 풀고(전사 확보),
구축 비용 B를 낸다. 이후 (r-1)번의 질의는 확률 c로 캐시에 적중한다(적용률). 적중하면 캐시 비용 C를 내고 정확도 a_hit로 답한다.
적중했지만 틀린 답은 확률 d로 감지되어 S로 넘어간다(d=0이면 논문처럼 조용히 틀린 답이 나간다).
적중하지 못하면 S가 푼다(+판정 비용 eps*C).
  비용 배수 = (r*S) / (S + B + (r-1)*[c*(C + d*(1-a_hit)*S) + (1-c)*(S + eps*C)])
  정확도     = (1/r)*a_S + ((r-1)/r)*[c*(a_hit + d*(1-a_hit)*a_S) + (1-c)*a_S]
  조용한 오답률(적중 질의 대비 전체) = c*(1-a_hit)*(1-d) 를 재등장 질의에 대해
논문 표 2·부록 C의 값(S, B, C, 소스 정확도, a_hit = C Acc x 복합 교란 통과율)을 넣고 c, r, d를 바꾼다.
실행: python3 -I toy_cache_model.py  (결과: toy_cache_model_results.json, ../report/figures/toy_multiplier.png)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_tables import TABLE2, T2_KEYS

here = os.path.dirname(os.path.abspath(__file__))

def model(S, B, C, a_S, a_hit, r, c, d=0.0, eps=0.1):
    per = c * (C + d * (1 - a_hit) * S) + (1 - c) * (S + eps * C)
    cost = S + B + (r - 1) * per
    mult = r * S / cost
    acc = a_S / r + (r - 1) / r * (c * (a_hit + d * (1 - a_hit) * a_S) + (1 - c) * a_S)
    silent = (r - 1) / r * c * (1 - a_hit) * (1 - d)
    return mult, acc, silent

res = {}
C_TOK = {"AIME25": 0.000413, "GSM8K": 0.000623, "GPQA-Diamond": 0.001858}  # cost_check_results.json의 토큰 재계산값
rs = [2, 5, 13, 30, 100, 1000, 100000]
cs = [1.0, 0.9, 0.7, 0.5, 0.2]
for k in ("AIME25", "GSM8K", "GPQA-Diamond"):
    v = dict(zip(T2_KEYS, TABLE2[k]))
    S, B, C = v["src"], v["bld"], C_TOK[k]
    a_S, a_hit = v["s_acc"] / 100, v["c_acc"] / 100 * v["comb"] / 100  # 원본 정답률 x 복합 교란 통과율
    grid = {}
    for c in cs:
        grid[str(c)] = {str(r): round(model(S, B, C, a_S, a_hit, r, c)[0], 2) for r in rs}
    # 점근선: r->inf 에서 c<1이면 S/(c*C+(1-c)*S)
    asym = {str(c): round(S / (c * C + (1 - c) * (S + 0.1 * C)), 1) for c in cs}
    # 정확도·조용한 오답 (r=13), 감지율 d를 바꿈
    acc = {}
    for d in (0.0, 0.5, 0.9):
        row = {}
        for c in (1.0, 0.7):
            m, a, s = model(S, B, C, a_S, a_hit, 13, c, d)
            row[str(c)] = {"mult": round(m, 1), "acc": round(a, 4), "silent_err": round(s, 4)}
        acc[str(d)] = row
    # 손익분기 r (배수 = 1): c별
    be = {}
    for c in cs:
        r = 2
        while r < 10**6 and model(S, B, C, a_S, a_hit, r, c)[0] < 1:
            r += 1
        be[str(c)] = r if r < 10**6 else None
    res[k] = {"inputs": {"S": S, "B": B, "C": C, "a_S": a_S, "a_hit(C Acc x combined pass)": a_hit},
              "multiplier_grid[c][r]": grid, "asymptote_r_inf": asym, "accuracy_by_detection": acc, "break_even_r_by_c": be}
    print(k, "asymptote", asym, "break-even r", be)
    print("  r=13:", {c: grid[c]["13"] for c in grid}, "r=1000:", {c: grid[c]["1000"] for c in grid})
    print("  acc r=13:", json.dumps(acc))

with open(os.path.join(here, "toy_cache_model_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for fp in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",):
    if os.path.exists(fp):
        from matplotlib import font_manager
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
rr = np.logspace(0.3, 4, 60)
for ax, k in zip(axs, ("AIME25", "GSM8K")):
    v = dict(zip(T2_KEYS, TABLE2[k]))
    S, B, C = v["src"], v["bld"], C_TOK[k]
    for c, col in zip((1.0, 0.9, 0.7, 0.5), ("#1b6ca8", "#3b9c6e", "#d08c1f", "#b5483a")):
        ax.plot(rr, [model(S, B, C, v["s_acc"] / 100, v["c_acc"] / 100 * v["comb"] / 100, r, c)[0] for r in rr], color=col, label=f"적용률 c={c}")
    ax.axhline(1, color="#888", lw=0.8, ls=":")
    ax.axhline(v["sav_x"], color="#1b6ca8", lw=0.8, ls="--")
    ax.text(2.1, v["sav_x"] * 1.08, f"논문 헤드라인 {v['sav_x']}배", fontsize=8, color="#1b6ca8")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("진입점당 재등장 횟수 r"); ax.set_title(k)
    ax.grid(alpha=0.25)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axs[0].set_ylabel("실효 비용 배수 (소스만 쓸 때 대비)")
axs[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(here, "..", "report", "figures", "toy_multiplier.png"), dpi=160)
