"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig3.png : 그림 3 (이미지 있음/없음 비교 산점도와 벤치마크 PCA 군집, 원문 6쪽)
  paper_fig8.png : 그림 8 (Spatial Vision Aggregator 도식, 원문 13쪽)

고른 이유: 저작자가 만든 차트와 도식이다. 그림 4(CV-Bench 예시 이미지), 그림 12, 14(모델 출력 예시, 사진 포함),
그림 19(데이터 예시)는 공개본에 넣지 않는다.
라이선스: arXiv abs 페이지는 CC BY 4.0. 원문 PDF에 © 또는 All rights reserved 표시가 없다(pdftotext 확인).
잘라낸 영역은 쪽 좌표(pt)이며 캡션 글자는 포함하지 않는다.

실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig3.png": (5, (80, 553, 525, 700)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig8.png": (12, (60, 66, 540, 276)),
}


def trim(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    return img.crop(box) if box else img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for name, (page_idx, (x0, y0, x1, y1)) in JOBS.items():
        pix = doc[page_idx].get_pixmap(dpi=300, clip=pymupdf.Rect(x0, y0, x1, y1))
        tmp = OUT / (name + ".tmp.png")
        pix.save(tmp)
        img = trim(Image.open(tmp))
        img.save(OUT / name)
        tmp.unlink()
        print("wrote", OUT / name, img.size)


if __name__ == "__main__":
    main()
