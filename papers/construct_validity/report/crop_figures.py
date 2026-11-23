"""논문 그림 두 장(도식과 차트)을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig1.png : 그림 1 (체계적 검토 흐름도, 원문 2쪽)
  paper_fig3.png : 그림 3 (코드북 핵심 결과 원 그래프, 원문 5쪽)

고른 이유: 저작자가 만든 도식과 차트뿐이다. 그림 2(막대 그래프)와 그림 4(흐름도)는 넣지 않는다.
라이선스: arXiv abs 페이지는 CC BY 4.0. 원문 PDF에 회사 저작권 고지가 없다(pdftotext 확인).
캡션 글자는 포함하지 않는다. 실행: python3 report/crop_figures.py  (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig1.png": (1, (100, 60, 512, 198)),
    "paper_fig3.png": (4, (100, 20, 512, 262)),
}


def trim(img):
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
