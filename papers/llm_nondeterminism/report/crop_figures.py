"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig1.png : 그림 1 (조건별 최대-최소 정확도 차이 막대 그래프, 원문 2쪽)
  paper_fig3.png : 그림 3 (모델별 TARr@10 막대 그래프, 원문 6쪽)

고른 이유: 저작자가 만든 차트다. 그림 2, 4, 5와 부록 그림도 차트지만 본문에서 쓰지 않아 싣지 않는다.
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
    "paper_fig1.png": (1, (108, 82, 504, 230)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig3.png": (5, (108, 82, 504, 247)),
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
