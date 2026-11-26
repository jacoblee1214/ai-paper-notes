"""우리가 그린 그림 세 장 (matplotlib). 입력: mrope_results.json, token_results.json, bench_results.json. 출력: report/figures/own_*.png
실행: python3 -I analysis/plot_figures.py"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
M = json.load(open(HERE / "mrope_results.json"))
T = json.load(open(HERE / "token_results.json"))
B = json.load(open(HERE / "bench_results.json"))
C = {"t": "#c44e52", "h": "#4c72b0", "w": "#55a868", "gray": "#999999"}

# ---- 1. M-RoPE: 채널 대응(왼쪽)과 영상 뒤 텍스트 시작 위치(오른쪽)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax = axs[0]
inv_wl = [2 * math.pi * (1e6 ** (2 * i / 128)) for i in range(64)]
section = [16, 24, 24]
axis = []
for i, n in enumerate(section):
    axis += ["t", "h", "w"][i % 3:i % 3 + 1] * n
ax.bar(range(64), inv_wl, color=[C[a] for a in axis], width=0.9)
ax.set_yscale("log")
ax.set_xlabel("채널 쌍 번호 (0이 가장 빠른 주파수)")
ax.set_ylabel("회전 파장 (위치 단위, 로그)")
ax.axhline(40, color="k", ls=":", lw=0.8)
ax.text(0.5, 50, "격자 폭 40", fontsize=8)
ax.set_title("LM 헤드 128차원의 64쌍을 [16, 24, 24]로 자른다\n시간(붉은색)이 가장 빠르고 너비(녹색)가 가장 느리다", fontsize=9.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["t"], label="시간 16쌍"), Patch(color=C["h"], label="높이 24쌍"), Patch(color=C["w"], label="너비 24쌍")], fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))

ax = axs[1]
ts = list(range(1, 121))
side = 16  # 448x448, 병합 후 16x16
v445 = [max(t, side) for t in ts]
main = [side for _ in ts]
ax.plot(ts, v445, color=C["t"], label="transformers v4.45.0 (영상 최대 ID + 1)")
ax.plot(ts, main, color=C["h"], label="transformers main 2026-11 (시작 + max(h', w'))")
ax.plot(ts, ts, color=C["gray"], ls="--", lw=1, label="영상의 마지막 시간 ID + 1 (= t')")
ax.fill_between(ts, main, ts, where=[t > side for t in ts], color=C["h"], alpha=0.12)
ax.set_xlabel("영상의 시간 격자 길이 t' (2프레임 묶음 수, 2 fps이면 초 단위와 같음)")
ax.set_ylabel("영상 뒤 첫 텍스트의 위치 번호 (영상 시작=0)")
ax.set_title("t'가 16을 넘으면 main 코드는 텍스트가 영상 시간 ID와 겹치는 번호에서 시작한다\n(448x448 영상, 코드 정적 대조와 직접 구현으로 확인)", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "own_mrope_positions.png", dpi=150)
plt.close(fig)

# ---- 2. 이미지 한 장의 LLM 토큰 수 대 한 변 길이 (정사각 입력)
def smart(h, w, mn, mx, f=28):
    hb, wb = round(h / f) * f, round(w / f) * f
    if hb * wb > mx:
        beta = math.sqrt(h * w / mx)
        hb, wb = max(f, math.floor(h / beta / f) * f), max(f, math.floor(w / beta / f) * f)
    elif hb * wb < mn:
        beta = math.sqrt(mn / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return (hb // f) * (wb // f)


sides = list(range(28, 4100, 14))
code = [smart(s, s, 3136, 28 * 28 * 1280) for s in sides]
hub = [smart(s, s, 3136, 28 * 28 * 16384) for s in sides]
cl = [math.ceil(s / 28) ** 2 for s in sides]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(sides, hub, color=C["h"], label="max_pixels = 16384 토큰 (Hub 설정)")
ax.plot(sides, code, color=C["t"], label="max_pixels = 1280 토큰 (transformers 코드 기본)")
ax.plot(sides, cl, color=C["gray"], ls=":", label="ceil(H/28)·ceil(W/28) 근사 (제한 없음)")
for s, lab, off in ((224, "224 -> 64", (-62, 10)), (256, "256 -> 81", (8, -14))):
    t = smart(s, s, 3136, 28 * 28 * 16384)
    ax.scatter([s], [t], color="k", zorder=5, s=18)
    ax.annotate(lab, (s, t), textcoords="offset points", xytext=off, fontsize=8)
ax.set_yscale("log")
ax.set_xlabel("정사각 입력의 한 변 (px)")
ax.set_ylabel("LLM에 들어가는 시각 토큰 수 (시작·끝 토큰 제외)")
ax.set_title("토큰 수는 한 변의 제곱에 비례하다가 max_pixels에서 평평해진다\nVLA 해상도(224, 256)는 64~81토큰", fontsize=9.5)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "own_token_counts.png", dpi=150)
plt.close(fig)

# ---- 3. 차이와 MDE: 표 2 (72B - GPT-4o)와 표 8 (M-RoPE - 1D-RoPE)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2))
ax = axs[0]
rows = [r for r in B["table2"] if r["gpt4o"] is not None]
rows.sort(key=lambda r: r["gpt4o"]["delta_pts"])
for i, r in enumerate(rows):
    d = r["gpt4o"]["delta_pts"]
    m = r["mde"]["d=0.2"]
    out = abs(d) >= m
    ax.barh(i, d, color=C["h"] if out else "#bbbbbb")
    ax.plot([-m, m], [i, i], color="k", lw=1.2)
    ax.plot([-m, -m], [i - 0.25, i + 0.25], color="k", lw=1.2)
    ax.plot([m, m], [i - 0.25, i + 0.25], color="k", lw=1.2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['benchmark']} (n={r['n']})" for r in rows], fontsize=7.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Qwen2-VL-72B - GPT-4o (점)")
ax.set_title("표 2: 검은 선은 ±MDE (d=0.2). 색이 칠해진 막대만 MDE 밖\n(단일 실행, 쌍 정보 없음, 값 출처가 서로 다름)", fontsize=9.5)

ax = axs[1]
rows = [r for r in B["table8"]]
for i, r in enumerate(rows):
    d = r["delta_pts"]
    m = r["mde_d0.2"]
    out = m is not None and abs(d) >= m
    ax.barh(i, d, color=C["t"] if out else "#bbbbbb")
    if m is not None:
        ax.plot([-m, m], [i, i], color="k", lw=1.2)
        ax.plot([-m, -m], [i - 0.25, i + 0.25], color="k", lw=1.2)
        ax.plot([m, m], [i - 0.25, i + 0.25], color="k", lw=1.2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['benchmark']} (n={r['n'] if r['n'] else '?'})" for r in rows], fontsize=7.5)
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("M-RoPE - 1D-RoPE (점)")
ax.set_title("표 8: 영상 세 벤치마크는 n을 확인하지 못해 선이 없다\nn=4,000~7,000이면 MDE 약 1.5~2.0점", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG / "own_delta_vs_mde.png", dpi=150)
plt.close(fig)
print("wrote 3 figures")
