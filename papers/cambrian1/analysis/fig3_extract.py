"""논문 그림 3 왼쪽(이미지 있음/없음/우연 수준)의 벡터 마커 위치에서 값을 복원한다.

PDF 6쪽의 그림은 마커가 벡터 도형이다. 우연 수준 사각형 중 값을 아는 것(0, 25, 50)으로 y축을 보정하고,
'Enabled-Disabled'(=green-red), 'Disabled-Random'(=red-random) 두 선으로 보정을 독립 검증한다.
결과: fig3_values.json
"""
import json
from pathlib import Path

import numpy as np
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent.parent
doc = fitz.open(str(ROOT / "paper" / "paper.pdf"))
page = doc[5]
region = fitz.Rect(85, 570, 335, 695)

COL = {
    "random": (0.9607843, 0.6588082, 0.2392157),
    "enabled_minus_disabled": (0.3647059, 0.5529412, 0.9921569),
    "disabled_minus_random": (0.6823377, 0.7764706, 0.9960631),
    "vision_enabled": (0.5333181, 0.6901961, 0.4274510),
    "vision_disabled": (0.9607843, 0.2627451, 0.2392157),
}
BENCH = ["SQA-I", "MMMU", "MathVista", "AI2D", "RealWorldQA", "TextVQA", "ChartQA", "OCRBench",
         "DocVQA", "GQA", "SEED-I", "MMVP", "MME", "MMB"]
XS = [115.2 + 15.3 * i for i in range(14)]


def near(a, b):
    return all(abs(x - y) < 1e-3 for x, y in zip(a, b))


pts = {k: {} for k in COL}
for d in page.get_drawings():
    if not d["rect"].intersects(region) or d["fill"] is None:
        continue
    for k, c in COL.items():
        if near(d["fill"], c):
            r = d["rect"]
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            for i, x in enumerate(XS):
                if abs(cx - x) < 0.6:  # 범례 마커(x가 다름)는 자동으로 제외
                    pts[k][i] = cy
assert all(len(v) == 14 for v in pts.values()), {k: len(v) for k, v in pts.items()}

# y 보정: 우연 수준 중 값이 정의된 벤치마크 (개방형 0, 4지선다 25, MME 50)
known = {"MathVista": 0, "TextVQA": 0, "ChartQA": 0, "OCRBench": 0, "DocVQA": 0, "GQA": 0,
         "MMMU": 25, "AI2D": 25, "SEED-I": 25, "MMVP": 25, "MME": 50}
ys = np.array([pts["random"][BENCH.index(b)] for b in known])
vs = np.array(list(known.values()), float)
A = np.vstack([ys, np.ones_like(ys)]).T
slope, icpt = np.linalg.lstsq(A, vs, rcond=None)[0]
fit_resid = np.abs(slope * ys + icpt - vs).max()


def val(y):
    return float(slope * y + icpt)


vals = {k: [round(val(pts[k][i]), 2) for i in range(14)] for k in COL}
# 독립 검증: 선 두 개는 다른 시리즈의 차이여야 한다
chk1 = np.abs(np.array(vals["enabled_minus_disabled"]) -
              (np.array(vals["vision_enabled"]) - np.array(vals["vision_disabled"]))).max()
chk2 = np.abs(np.array(vals["disabled_minus_random"]) -
              (np.array(vals["vision_disabled"]) - np.array(vals["random"]))).max()

out = {"bench": BENCH, "values": vals, "points_per_unit": slope, "calibration_max_resid": fit_resid,
       "check_enabled_minus_disabled_maxdiff": float(chk1), "check_disabled_minus_random_maxdiff": float(chk2),
       "note": "MME는 /20 척도(0-100). 값은 PDF 벡터 마커 위치에서 복원(소수 첫째 자리까지 신뢰)."}
(ROOT / "analysis" / "fig3_values.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("calibration max resid (score pts):", round(fit_resid, 3))
print("check E-D vs green-red:", round(chk1, 3), " check D-R vs red-orange:", round(chk2, 3))
print(f"{'bench':12s} {'random':>7s} {'enabled':>8s} {'disabled':>9s} {'E-D':>6s} {'D-R':>6s}")
for i, b in enumerate(BENCH):
    print(f"{b:12s} {vals['random'][i]:7.1f} {vals['vision_enabled'][i]:8.1f} {vals['vision_disabled'][i]:9.1f} "
          f"{vals['enabled_minus_disabled'][i]:6.1f} {vals['disabled_minus_random'][i]:6.1f}")
