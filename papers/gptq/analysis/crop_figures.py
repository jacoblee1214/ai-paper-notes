"""원문 PDF에서 도식 하나와 차트 하나를 잘라 figures/에 저장한다.

  paper_fig2.png : 그림 2 (GPTQ 양자화 절차 도식, 원문 4쪽)
  paper_fig1.png : 그림 1 (OPT·BLOOM perplexity 차트, 원문 2쪽)

고른 이유: 저작자가 그린 도식과 차트다. 그림 3(LAMBADA 곡선), 4(그룹 크기 곡선)는 넣지 않는다.
라이선스: arXiv abs 페이지는 CC BY 4.0이고 원문 PDF에 ©, Copyright 표시가 없다(pdftotext 확인). 캡션 글자는 포함하지 않는다.
실행: python3 crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "figures"
JOBS = {  # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig2.png": (3, (306, 440, 535, 601)),
    "paper_fig1.png": (1, (118, 140, 500, 281)),
}


def trim(img):
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    return img.crop(box) if box else img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for name, (page_idx, (x0, y0, x1, y1)) in JOBS.items():
        pix = doc[page_idx].get_pixmap(dpi=250, clip=pymupdf.Rect(x0, y0, x1, y1))
        tmp = OUT / (name + ".tmp.png")
        pix.save(tmp)
        trim(Image.open(tmp)).save(OUT / name)
        tmp.unlink()


if __name__ == "__main__":
    main()
