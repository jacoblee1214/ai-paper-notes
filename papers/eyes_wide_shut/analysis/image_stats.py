"""MMVP 이미지 300장의 크기·종횡비 통계 (PIL, CPU 수 초). evaluate_mllm.py가 pad 없이 CLIPImageProcessor.preprocess를 직접 부르므로
(resize 짧은 변 + 중앙 자르기) 가로세로 비가 1이 아닌 이미지는 가장자리가 잘린다. 얼마나 잘리는지 센다.
입력: ../data/mmvp/images/*.jpg, ../data/mmvp_vlm/images/*.jpg (HF MMVP/MMVP, MMVP/MMVP_VLM, gitignore, 커밋 안 함). 출력: image_results.json
"""
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
IMG = HERE.parent / "data" / "mmvp" / "images"
sizes = {}
for k in range(1, 301):
    with Image.open(IMG / f"{k}.jpg") as im:
        sizes[k] = im.size
ratio_lost = {k: 1 - min(w, h) / max(w, h) for k, (w, h) in sizes.items()}
n_sq = sum(1 for k, (w, h) in sizes.items() if w == h)
lost = sorted(ratio_lost.values())
pair_same_size = sum(sizes[2 * p + 1] == sizes[2 * p + 2] for p in range(150))
res = {
    "n_images": len(sizes),
    "n_square": n_sq,
    "n_nonsquare": len(sizes) - n_sq,
    "median_area_lost_by_center_crop_pct": round(100 * lost[len(lost) // 2], 1),
    "mean_area_lost_pct": round(100 * sum(lost) / len(lost), 1),
    "p90_area_lost_pct": round(100 * lost[int(0.9 * len(lost))], 1),
    "n_lost_gt_20pct": sum(1 for v in lost if v > 0.2),
    "n_lost_gt_40pct": sum(1 for v in lost if v > 0.4),
    "pairs_with_same_size": pair_same_size,
    "width_range": [min(w for w, h in sizes.values()), max(w for w, h in sizes.values())],
    "height_range": [min(h for w, h in sizes.values()), max(h for w, h in sizes.values())],
    "min_side_range": [min(min(s) for s in sizes.values()), max(min(s) for s in sizes.values())],
    "n_min_side_lt_224": sum(1 for s in sizes.values() if min(s) < 224),
    "n_min_side_lt_336": sum(1 for s in sizes.values() if min(s) < 336),
}
VLM = HERE.parent / "data" / "mmvp_vlm" / "images"
vs = {}
for k in range(1, 271):
    with Image.open(VLM / f"{k}.jpg") as im:
        vs[k] = im.size
res["mmvp_vlm"] = {"n_images": len(vs), "distinct_sizes": sorted({str(v) for v in vs.values()})}
(HERE / "image_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
