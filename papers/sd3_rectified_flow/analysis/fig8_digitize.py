"""Read the points of Figure 8 (arXiv:2403.03206) from the raster panels embedded in the PDF.

usage: python fig8_digitize.py <paper.pdf> <work_dir> [overlay_dir]
Extracts the page-12 images with pdfimages (poppler), finds viridis-coloured dots per depth class, and converts
pixel centres to data coordinates with hand-entered tick pixel positions (see CAL).  Writes fig8_points.json.
Digitisation error: dot centroid +-1 px; one px = 6.3e-5 in validation loss (y) and ~0.0025 / 0.00045 / 0.00035
in x for panels 1 / 2 / 3 (see 'px_to_data' in the output).  Dots fully hidden behind others are not recoverable.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
pdf, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
overlay = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
work.mkdir(parents=True, exist_ok=True)
subprocess.run(["pdfimages", "-png", "-f", "12", "-l", "12", str(pdf), str(work / "im")], check=True)

VIRIDIS5 = {15: (68, 1, 84), 18: (59, 82, 139), 21: (33, 145, 140), 30: (94, 201, 98), 38: (253, 231, 37)}
VIRIDIS3 = {18: (68, 1, 84), 24: (33, 145, 140), 30: (253, 231, 37)}
# panel: image index, colours, x calibration [(pixel, value), ...], y calibration, x is log10?, legend box (x0,y0,x1,y1)
Y_LOSS = [(101.5, 0.42), (260.5, 0.41), (418.5, 0.40), (577.5, 0.39)]
CAL = {
    "fig8_top2_loss_vs_flops": dict(idx=2, cols=VIRIDIS5, markers_on_lines=True,
                                    x=[(538.5, 21.0), (1049.7, 22.0)], logx=True,
                                    y=[(101.5, 0.425), (259.5, 0.415), (417.5, 0.405), (575.5, 0.395)], legend=(845, 28, 1340, 440)),
    "fig8_bot1_geneval": dict(idx=8, cols=VIRIDIS5, x=[(309.5, 0.45), (704.0, 0.55), (1098.5, 0.65)], y=Y_LOSS, legend=(238, 311, 705, 722)),
    "fig8_bot2_human_elo": dict(idx=10, cols=VIRIDIS5, x=[(229.4, 750.0), (717.5, 1000.0), (1108.0, 1200.0)], y=Y_LOSS, legend=(246, 313, 712, 725)),
    "fig8_bot3_compbench": dict(idx=12, cols=VIRIDIS5, x=[(269.9, 0.40), (552.2, 0.44), (834.5, 0.48), (1116.8, 0.52)], y=Y_LOSS, legend=(238, 311, 705, 725)),
    "fig8_bot4_video_elo": dict(idx=14, cols=VIRIDIS3, x=[(326.5, 600.0), (570.5, 800.0), (815.5, 1000.0), (1059.5, 1200.0)],
                                y=[(41.5, 0.30), (273.5, 0.28), (505.5, 0.26), (736.0, 0.24)], legend=(238, 415, 705, 700)),
}
DOT_AREA = {"1350": 330.0, "1200": 380.0}

def fit(cal, log=False):
    p = np.array([c[0] for c in cal]); v = np.array([c[1] for c in cal])
    a, b = np.polyfit(p, v, 1)
    resid = float(np.abs(a * p + b - v).max())
    return a, b, resid

def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x ** 2 + y ** 2 <= r * r

def kmeans(pts, k, it=30):
    c = pts[np.linspace(0, len(pts) - 1, k).astype(int)].astype(float)
    for _ in range(it):
        lab = ((pts[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([pts[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c

out = {}
for name, cfg in CAL.items():
    im = np.array(Image.open(work / f"im-{cfg['idx']:03d}.png").convert("RGB")).astype(int)
    H, W, _ = im.shape
    ax, bx, rx = fit(cfg["x"]); ay, by, ry = fit(cfg["y"])
    lx0, ly0, lx1, ly1 = cfg["legend"]
    area1 = DOT_AREA["1350" if W == 1350 else "1200"]
    rad = 8 if cfg.get("markers_on_lines") else 0
    res = {}
    for depth, col in cfg["cols"].items():
        # nearest-class assignment (L1 distance) so that anti-aliased / line-overlaid dots still land in one class
        names = list(cfg["cols"])
        solid = np.stack([np.array(cfg["cols"][k]) for k in names])
        d_all = np.abs(im[:, :, None, :] - solid[None, None]).sum(-1)
        which = d_all.argmin(-1)
        mask = (which == names.index(depth)) & (d_all.min(-1) < 75)
        faded_t = 0.2 * np.array(col) + 0.8 * 255          # legend box is drawn with alpha 0.8
        fm = (np.abs(im - faded_t).sum(-1) < 25)
        box = np.zeros_like(fm); box[ly0:ly1, lx0:lx1] = True
        mask |= fm & box
        # legend marker column
        leg_marker = np.zeros_like(mask); leg_marker[ly0:ly1, lx0:lx0 + 110] = True
        if cfg["idx"] == 2:
            leg_marker[:] = False; leg_marker[28:440, 845:1340] = True
        mask &= ~leg_marker
        mask = ndi.binary_closing(mask, structure=disk(3))   # heal the thin grey regression line drawn over dots
        mask = ndi.binary_opening(mask, structure=disk(rad if rad else 5))
        lab, n = ndi.label(mask)
        pts = []
        for j in range(1, n + 1):
            yy, xx = np.nonzero(lab == j)
            a = len(yy)
            if a < 0.35 * area1:
                continue
            k = max(1, int(round(a / area1))) if a > 1.45 * area1 else 1
            if cfg.get("markers_on_lines"):
                k = 1
            P = np.stack([xx, yy], 1)
            cs = [P.mean(0)] if k == 1 else list(kmeans(P, k))
            pts += [(float(c[0]), float(c[1]), a / k) for c in cs]
        data = []
        for px, py, a in pts:
            xv = ax * px + bx
            if cfg.get("logx"):
                xv = 10 ** xv
            data.append({"x": float(xv), "y": float(ay * py + by), "px": px, "py": py})
        res[str(depth)] = sorted(data, key=lambda d: d["x"])
    out[name] = {"points": res, "calibration_resid": {"x": rx, "y": ry},
                 "px_to_data": {"x_per_px": float(ax), "y_per_px": float(ay)}, "n_per_depth": {d: len(v) for d, v in res.items()}}
    if overlay:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axp = plt.subplots(figsize=(8, 6)); axp.imshow(im.astype(np.uint8))
        for d, v in res.items():
            axp.plot([p["px"] for p in v], [p["py"] for p in v], "x", color="red", ms=7)
        fig.savefig(overlay / f"{name}.png", dpi=80); plt.close(fig)
    print(name, out[name]["n_per_depth"], "calib resid", rx, ry)
(HERE / "fig8_points.json").write_text(json.dumps(out, indent=1))
