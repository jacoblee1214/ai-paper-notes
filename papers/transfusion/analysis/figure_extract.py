"""논문 그림 5(p.9)의 벡터 도형에서 점(체크포인트)과 추세선 값을 읽어 figure_values.json에 저장한다.

각 패널: x축은 log10(FLOPs) 눈금 글자(1e+20, 1e+21, 1e+22)의 가운데 x좌표로, y축은 로그 눈금 글자의 세로 가운데로 보정한다.
빨강 = Transfusion, 파랑 = Chameleon. 점은 작은 채움 도형, 추세선은 폭 0.4의 선분이다.
벡터 좌표 판독이라 점 위치 오차는 눈금 글자 가운데 오차(약 0.5pt)에 해당한다(x로 약 0.01 decade, y로 약 1%).
필요: paper/paper.pdf (arXiv:2408.11039 v1), PyMuPDF.   실행: python3 figure_extract.py
"""
import json
import math
from pathlib import Path

import fitz

fitz.TOOLS.mupdf_display_errors(False)
HERE = Path(__file__).resolve().parent
doc = fitz.open(HERE.parent / "paper" / "paper.pdf")
page = doc[8]

# 패널: 이름 -> (x0, x1, y0, y1 = 축 프레임), y 눈금 글자
PANELS = {
    "c4_ppl":   (122.4, 300.6, 75.3, 174.9,  ["8", "16"]),
    "wiki_ppl": (324.3, 502.6, 75.3, 174.9,  ["4", "8", "16"]),
    "llama_acc": (122.4, 300.6, 217.9, 317.5, ["32", "64"]),
    "coco_cider": (324.3, 502.6, 217.9, 317.5, ["2", "4", "8", "16", "32"]),
    "coco_fid": (125.1, 300.6, 360.5, 458.6, ["16", "32", "64", "128"]),
    "coco_clip": (324.3, 502.6, 358.8, 458.4, ["12", "24"]),
}
words = page.get_text("words")
draws = page.get_drawings()


def in_box(x, y, b, pad=0):
    return b[0] - pad <= x <= b[1] + pad and b[2] - pad <= y <= b[3] + pad


out = {}
for name, (x0, x1, y0, y1, ylabs) in PANELS.items():
    box = (x0, x1, y0, y1)
    # x 눈금: 프레임 바로 아래 글자
    xt = {}
    for w in words:
        if w[4].startswith("1e+") and x0 - 5 <= (w[0] + w[2]) / 2 <= x1 + 5 and y1 < w[1] < y1 + 8:
            xt[float(w[4][3:])] = (w[0] + w[2]) / 2
    xs = sorted(xt)
    fx = lambda px: xs[0] + (px - xt[xs[0]]) * (xs[-1] - xs[0]) / (xt[xs[-1]] - xt[xs[0]])
    # y 눈금: 프레임 왼쪽 글자(숫자)
    yt = {}
    for w in words:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if w[4] in ylabs and x0 - 14 <= cx < x0 and y0 - 4 <= cy <= y1 + 4:
            yt[float(w[4])] = cy
    yv = sorted(yt)
    ln0, ln1 = math.log(yv[0]), math.log(yv[-1])
    fy = lambda py: math.exp(ln0 + (py - yt[yv[0]]) * (ln1 - ln0) / (yt[yv[-1]] - yt[yv[0]]))
    pts = {"Transfusion": [], "Chameleon": []}
    lines = {"Transfusion": [], "Chameleon": []}
    for d in draws:
        r = d["rect"]
        col = tuple(round(c, 2) for c in (d.get("color") or ()))
        fill = tuple(round(c, 2) for c in (d.get("fill") or ()))
        who = {(1.0, 0.0, 0.0): "Transfusion", (0.0, 0.0, 1.0): "Chameleon"}.get(col)
        if who is None:
            continue
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if not in_box(cx, cy, box, 1):
            continue
        if fill and r.width < 8:
            pts[who].append((fx(cx), fy(cy)))
        elif not fill and 0.3 < (d.get("width") or 0) < 0.6 and r.width > 20:
            for it in d["items"]:
                if it[0] == "l":
                    lines[who] += [(fx(it[1].x), fy(it[1].y)), (fx(it[2].x), fy(it[2].y))]
    out[name] = {
        "x_ticks_px": {str(k): round(v, 2) for k, v in xt.items()},
        "y_ticks_px": {str(k): round(v, 2) for k, v in yt.items()},
        "points": {k: [[round(a, 4), round(b, 4)] for a, b in v] for k, v in pts.items()},  # [log10 FLOPs, metric]
        "paper_trendline_ends": {k: [[round(a, 4), round(b, 4)] for a, b in (sorted(v)[0], sorted(v)[-1])] if v else None for k, v in lines.items()},
    }
    print(name, {k: len(v) for k, v in pts.items()}, {k: bool(v) for k, v in lines.items()})

Path(HERE / "figure_values.json").write_text(json.dumps(out, indent=1))
