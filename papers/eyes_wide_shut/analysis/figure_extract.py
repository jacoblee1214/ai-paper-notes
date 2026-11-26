"""그림 6(CLIP·MLLM의 패턴별 성능)을 원문 PDF 벡터에서 읽는다 (PyMuPDF, 수 초).

출력: figure_values.json
- 쪽 6의 막대(CLIP=파랑, 나머지 회색 4개 = InstructBLIP, LLaVA 1.5, Gemini, GPT-4)의 위쪽 y좌표를
  눈금선(0 = 402.82, 70 = 318.31 pt)으로 보정해 값으로 바꾼다. 마커(별·삼각형·사각형·마름모)도 같은 값을 가리켜야 한다.
- 막대가 0이면 마커만 있으므로 마커 중심 y에서 읽는다.
- 패턴 순서는 그림의 x축 아이콘 순서(CLIP 값이 큰 순)이고, 아이콘은 글자가 아니라 벡터라서
  '표 1의 세 모델 평균이 CLIP 막대와 일치하는 패턴'으로 우리가 대응시켰다(analysis/stats_check.py S3).
"""
import json
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "paper" / "paper.pdf"

d = pymupdf.open(PDF)
p = d[5]
dr = [x for x in p.get_drawings() if x["rect"].y0 > 300 and x["rect"].x0 < 300 and x["rect"].y1 < 425]

# 눈금선: 가로 회색 선
grid = sorted({round(x["rect"].y0, 2) for x in dr if x["type"] == "s" and x["rect"].y0 == x["rect"].y1 and x["rect"].x1 - x["rect"].x0 > 200})
ys0, ys70 = max(grid), min(grid)  # 0(아래)과 70(위)
assert abs((ys0 - ys70) / 70 - 1.2073) < 0.01, (ys0, ys70)
unit = (ys0 - ys70) / 70.0


def val(y):
    return (ys0 - y) / unit


bars = [x["rect"] for x in dr if x["type"] == "fs" and abs((x["rect"].x1 - x["rect"].x0) - 3.82) < 0.05]
fills = {}
for x in dr:
    r = x["rect"]
    if x["type"] == "fs" and abs((r.x1 - r.x0) - 3.82) < 0.05:
        key = tuple(round(c, 2) for c in x["fill"])
        fills.setdefault(key, []).append(r)
blue_key = [k for k in fills if k[2] > 0.8 and k[0] < 0.6][0]
clip_bars = sorted(fills[blue_key], key=lambda r: r.x0)
gray = [r for k, v in fills.items() if k[0] > 0.8 for r in v]
assert len(clip_bars) == 9 and len(gray) == 36, (len(clip_bars), len(gray))

series = ["InstructBLIP", "LLaVA-1.5", "Gemini", "GPT-4"]
out = {"clip": [], **{s: [] for s in series}}
for g, cb in enumerate(clip_bars):
    out["clip"].append(round(val(cb.y0), 2))
    gb = sorted([r for r in gray if cb.x0 + 3 < r.x0 < cb.x0 + 3.82 * 5], key=lambda r: r.x0)
    assert len(gb) == 4, (g, len(gb))
    for s, r in zip(series, gb):
        # 막대 높이 0이면 y0==y1 이므로 value 0
        out[s].append(round(val(r.y0), 2))

# 마커로 교차 확인: 마커 중심 y와 막대 값
markers = [x["rect"] for x in dr if x["type"] == "s" and x["fill"] is None and x["color"] and x["color"][2] > 0.4 and x["color"][0] == 0.0 and 2.0 < (x["rect"].x1 - x["rect"].x0) < 3.5]
out["marker_check_max_abs_diff"] = None
diffs = []
for s_i, s in enumerate(series):
    for g, cb in enumerate(clip_bars):
        x_center = cb.x0 + 3.82 * (s_i + 1) + 1.91
        cand = [m for m in markers if abs((m.x0 + m.x1) / 2 - x_center) < 1.2]
        if cand:
            m = cand[0]
            diffs.append(abs(val((m.y0 + m.y1) / 2) - out[s][g]))
out["marker_check_max_abs_diff"] = round(max(diffs), 2) if diffs else None
out["n_markers_matched"] = len(diffs)
out["order"] = ["Color", "State", "Structural", "Viewpoint", "Presence", "Quantity", "Positional", "Text", "Orientation"]
out["order_note"] = "x축 아이콘 순서(팔레트, 순환 화살표, 톱니, 카메라, 돋보기, 위아래 화살표+숫자, 핀, A, 나침반). CLIP 막대가 큰 순서"
(HERE / "figure_values.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
