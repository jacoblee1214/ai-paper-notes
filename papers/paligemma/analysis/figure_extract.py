"""논문 PDF의 벡터 그림(막대, 선)에서 값을 읽어 figure_values.json에 저장한다.

그림 4, 5, 7, 8, 9, 10, 11, 17을 대상으로 한다. 막대 사각형과 선분의 좌표를 축 눈금 글자의 위치로 보정한다.
눈금 글자의 가운데를 눈금 위치로 쓰므로 절대값은 약 ±0.3(축 단위), 같은 그림 안의 차이는 약 ±0.1이다.
(PDF 래스터 판독이 아니라 벡터 좌표 판독이라 300 dpi 눈대중보다 정확하다.)

필요: paper/paper.pdf (arXiv:2407.07726v2), PyMuPDF.  실행: python3 figure_extract.py
"""
import json
from pathlib import Path

import fitz

fitz.TOOLS.mupdf_display_errors(False)
HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "paper" / "paper.pdf"
doc = fitz.open(PDF)


def fills(page, clip, wmax=60, hmin=0.2, skip_white=True):
    out = []
    for d in page.get_drawings():
        r, f = d["rect"], d.get("fill")
        if f is None or not r.intersects(clip) or r.width > wmax or r.height < hmin:
            continue
        if skip_white and tuple(round(c, 2) for c in f) == (1.0, 1.0, 1.0):
            continue
        out.append((r, tuple(round(c, 2) for c in f)))
    return out


def tick_centers(page, clip, labels, axis):
    """눈금 글자 {문자열: 가운데 좌표}. axis='y'면 세로 가운데."""
    res = {}
    for w in page.get_text("words", clip=clip):
        if w[4] in labels:
            res[w[4]] = (w[1] + w[3]) / 2 if axis == "y" else (w[0] + w[2]) / 2
    return res


def lin_y(ticks):
    """{값(float): y좌표} 두 점으로 y -> 값 선형 변환 함수."""
    (v0, y0), (v1, y1) = sorted(ticks.items())[0], sorted(ticks.items())[-1]
    return lambda y: v0 + (y - y0) * (v1 - v0) / (y1 - y0)


out = {}

# ---- 그림 4 (p.8): 사전학습 길이별 상대 후회의 중앙값(굵은 검은 선)
p = doc[7]
clip = fitz.Rect(60, 80, 300, 215)
tk = tick_centers(p, clip, {"0", "-20%", "-40%", "-60%", "-80%"}, "y")
f = lin_y({0.0: tk["0"], -20.0: tk["-20%"]})
xs_lab = {"1M": 1e6, "3M": 3e6, "10M": 1e7, "30M": 3e7, "100M": 1e8, "300M": 3e8, "1B": 1e9}
pts = None
for d in p.get_drawings():
    r = d["rect"]
    if r.intersects(clip) and d.get("color") == (0.0, 0.0, 0.0) and d.get("width", 0) >= 1.5 and d.get("fill") is None:
        pts = [q for it in d["items"] if it[0] == "l" for q in (it[1], it[2])]
        break
line = []
seen = set()
for q in pts:
    k = (round(q.x, 1), round(q.y, 1))
    if k not in seen:
        seen.add(k)
        line.append(k)
line.sort()  # x 오름차순: 1M ... 1B
out["fig4_median_regret_pct"] = {name: round(f(y), 2) for (name, _), (x, y) in zip(xs_lab.items(), line)}

# ---- 그림 5 (p.8): 36개 과제 평균. 왼쪽 5막대(마스크·손실), 가운데 2막대(과제 접두)
clip = fitz.Rect(330, 85, 570, 215)
tk = tick_centers(p, clip, {"71", "73", "75", "77", "79"}, "y")
g = lin_y({71.0: tk["71"], 79.0: tk["79"]})
bars = sorted([r for r, c in fills(p, clip) if r.width < 14], key=lambda r: r.x0)
vals = [round(g(r.y0), 2) for r in bars]
out["fig5_avg36"] = {
    "AR_suffix_only(prefix-LM)__loss_suffix": vals[0],
    "AR_on_prefix__loss_suffix": vals[1],
    "AR_on_prefix__loss_suffix+prefix": vals[2],
    "AR_on_image+prefix__loss_suffix": vals[3],
    "AR_on_image+prefix__loss_suffix+prefix": vals[4],
    "task_prefix_yes": vals[5],
    "task_prefix_no": vals[6],
}

# ---- 그림 7 (p.10 왼쪽): 38개 과제 평균의 TT 대비 차이. 라벨 순서 = 글자 순서
p = doc[9]
clip = fitz.Rect(60, 85, 205, 200)
tk = tick_centers(p, clip, {"0", "-10", "-20", "-30", "-40"}, "y")
h = lin_y({0.0: tk["0"], -40.0: tk["-40"]})
labels = ["TT", "FT", "FF", "TF", "FR", "TR", "RT", "RR", "RF"]
cols = [(0.72, 0.88, 0.93), (0.64, 0.83, 0.9), (0.56, 0.76, 0.86), (0.48, 0.7, 0.83), (0.4, 0.62, 0.79),
        (0.34, 0.54, 0.75), (0.27, 0.45, 0.7)]
fig7 = {"TT": 0.0, "FT": 0.0}  # 두 막대는 높이 0(보이지 않음)
bar7 = sorted([r for r, c in fills(p, clip) if c in cols], key=lambda r: r.x0)
for lab, r in zip(labels[2:], bar7):
    fig7[lab] = round(h(r.y1), 1)  # 막대는 0선(위)에서 아래로 자란다
out["fig7_avg38_diff_to_TT"] = fig7

# ---- 그림 8 (p.10 오른쪽): 예제 수별 37개 과제 평균(SigLIP 인코더 대 Fuyu 식)
clip = fitz.Rect(300, 85, 545, 200)
tk = tick_centers(p, clip, {"60", "70", "80"}, "y")
k8 = lin_y({60.0: tk["60"], 80.0: tk["80"]})
series = {}
for d in p.get_drawings():
    r = d["rect"]
    if r.intersects(clip) and d.get("color") is not None and d.get("fill") is None and r.width > 50:
        pts = sorted({(round(q.x, 1), round(q.y, 1)) for it in d["items"] if it[0] == "l" for q in (it[1], it[2])})
        if len(pts) == 3:
            key = "siglip_encoder" if d["color"][2] > d["color"][0] else "fuyu_raw_patches"
            series[key] = {n: round(k8(y), 1) for n, (x, y) in zip(["100M", "300M", "1B"], pts)}
out["fig8_avg37"] = series

# ---- 그림 9와 10 (p.11)
p = doc[10]
clip = fitz.Rect(60, 85, 290, 190)
tk = tick_centers(p, clip, {"0", "25", "50", "75", "100"}, "y")
f9 = lin_y({0.0: tk["0"], 100.0: tk["100"]})
col = {(0.44, 0.66, 0.85): "224px", (0.9, 0.65, 0.49): "448px_resized_to_224_first", (0.53, 0.79, 0.68): "448px"}
b9 = sorted([(r, c) for r, c in fills(p, clip) if c in col and r.width > 15 and r.y1 > 170], key=lambda t: t[0].x0)
fig9 = {"resolution_sensitive_9": {}, "resolution_insensitive_11": {}}
for i, (r, c) in enumerate(b9):
    fig9["resolution_sensitive_9" if i < 3 else "resolution_insensitive_11"][col[c]] = round(f9(r.y0), 1)
out["fig9_task_score_avg"] = fig9

clip = fitz.Rect(300, 85, 545, 190)
tk = tick_centers(p, clip, {"0", "+10"}, "y")
f10 = lin_y({0.0: tk["0"], 10.0: tk["+10"]})
cols10 = [(0.9, 0.65, 0.49), (0.84, 0.66, 0.76), (0.53, 0.79, 0.68)]
bars10 = sorted([r for r, c in fills(p, clip, hmin=0.0) if c in cols10], key=lambda r: r.x0)
base = max(r.y1 for r in bars10)
names10 = ["pre448_tr224", "pre224_tr224(0)", "pre224_tr448", "pre224_tr448_windows",
           "pre448w224_tr448w224_windows", "pre448_tr448"]
# 0선 y: 막대 중 위로 자라는 것의 y1, 아래로 자란 막대는 y0가 0선
zero = min(r.y1 for r in bars10 if r.y1 < 160)
vals10 = {}
bars_all = sorted([r for r, c in fills(p, clip, hmin=0.0) if c in cols10], key=lambda r: r.x0)
for nm, r in zip([n for n in names10 if n != "pre224_tr224(0)"], bars_all):
    vals10[nm] = round(f10(r.y0) if r.y1 <= zero + 0.5 else -(r.y1 - zero) * 10.0 / (f10 and (tk["0"] - tk["+10"])), 1)
vals10["pre224_tr224(0)"] = 0.0
out["fig10_9task_avg_score_vs_224"] = vals10

# ---- 그림 11 (p.12): 재실행 표준편차 히스토그램 (24칸, 0–2.0)
p = doc[11]
hist = {}
for name, clip in (("stage1_3reruns_100M", fitz.Rect(300, 80, 545, 150)), ("transfer_5reruns", fitz.Rect(300, 160, 545, 225))):
    bars = sorted([r for r, c in fills(p, clip, hmin=0.3) if r.width < 9], key=lambda r: r.x0)
    base = max(r.y1 for r in bars)
    unit = (tick_centers(p, clip, {"5"}, "y")["5"])
    zero_c = tick_centers(p, clip, {"0"}, "y")["0"]
    per_task = 5.0 / (zero_c - unit)  # 과제 5개당 픽셀 → 픽셀당 과제 수
    x0 = min(r.x0 for r in bars)
    w = (max(r.x1 for r in bars) - x0)
    bins = {}
    for r in bars:
        i = round((r.x0 - x0) / 7.87)
        bins[i] = round((base - r.y0) * per_task)
    hist[name] = {"bin_width_sd": round(2.0 / 24, 4), "counts_by_bin": {str(k): v for k, v in sorted(bins.items())}}
out["fig11_sd_hist"] = hist

# ---- 그림 17 (p.51): 마스크·손실 절제, 과제별 5개 막대 (36과제)
p = doc[50]
rects = []
for d in p.get_drawings():
    r, fcol = d["rect"], d.get("fill")
    if fcol is not None and 2 < r.width < 8 and r.height > 0.1 and r.y0 < 450 and tuple(round(c, 2) for c in fcol) != (1, 1, 1):
        rects.append((r, tuple(round(c, 2) for c in fcol)))
order = [(0.65, 0.78, 0.9), (0.44, 0.66, 0.85), (0.24, 0.54, 0.79), (0.17, 0.4, 0.61), (0.11, 0.27, 0.41)]
ticks = {}
for w in p.get_text("words"):
    if w[4] in ("0", "150") and w[0] < 90:
        ticks.setdefault(w[4], []).append((w[1] + w[3]) / 2)
panels = [{"0": ticks["0"][0], "150": ticks["150"][0]}, {"0": ticks["0"][1], "150": ticks["150"][1]}]
# 세로 글자(과제 이름): 줄 방향이 (0,-1)인 span
names = []
for b in p.get_text("dict")["blocks"]:
    for ln in b.get("lines", []):
        if abs(ln["dir"][0]) < 0.1:
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if txt:
                names.append(((ln["bbox"][0] + ln["bbox"][2]) / 2, ln["bbox"][1], txt))
rects.sort(key=lambda t: (0 if t[0].y1 < 300 else 1, t[0].x0))
tasks = []
for panel in (0, 1):
    pr = [t for t in rects if (t[0].y1 < 300) == (panel == 0)]
    fy = lin_y({0.0: panels[panel]["0"], 150.0: panels[panel]["150"]})
    for i in range(0, len(pr), 5):
        grp = pr[i:i + 5]
        assert [c for _, c in grp] == order, "막대 색 순서가 예상과 다름"
        xc = sum((r.x0 + r.x1) / 2 for r, _ in grp) / 5
        base_y = grp[0][0].y1
        tasks.append((xc, panel, [round(fy(r.y0) - fy(base_y) + 0.0, 1) * -1 if False else round((base_y - r.y0) / ((panels[panel]["0"] - panels[panel]["150"]) / 150.0), 1) for r, _ in grp]))
named = []
for xc, panel, vals in tasks:
    cand = [n for n in names if (n[1] < 300) == (panel == 0) and abs(n[0] - xc) < 6]
    named.append({"task": cand[0][2] if cand else f"?{xc:.0f}", "AR_suf_loss_suf": vals[0], "AR_pref_loss_suf": vals[1],
                  "AR_pref_loss_pref": vals[2], "AR_all_loss_suf": vals[3], "AR_all_loss_pref": vals[4]})
out["fig17_per_task"] = named

(HERE / "figure_values.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in out.items():
    if k != "fig17_per_task":
        print(k, json.dumps(v, ensure_ascii=False))
print("fig17 tasks:", len(named), [t["task"] for t in named][:5], "...")
