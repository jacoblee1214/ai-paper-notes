"""원문 PDF에서 도식·그래프 세 장을 잘라 figures/에 저장한다.

  paper_fig1.png  : 그림 1 (네 방식의 attention 도식과 PPL, 원문 2쪽)
  paper_fig4.png  : 그림 4 (롤링 KV 캐시 도식, 원문 5쪽)
  paper_fig10.png : 그림 10 (지연 시간·메모리 막대 그래프, 원문 9쪽)

고른 이유: 저작자가 그린 도식과 차트다. 그림 2, 7, 11-14(attention 맵 시각화)와 그림 5·9의 곡선은 넣지 않는다.
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
    "paper_fig1.png": (1, (108, 80, 513, 196)),
    "paper_fig4.png": (4, (340, 458, 510, 529)),
    "paper_fig10.png": (8, (108, 80, 513, 150)),
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
        img = trim(Image.open(tmp))
        img.save(OUT / name)
        tmp.unlink()
        print("wrote", OUT / name, img.size)


if __name__ == "__main__":
    main()
