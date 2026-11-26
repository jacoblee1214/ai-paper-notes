"""논문 그림 1(LLaVA 구조 도식)을 원문 PDF에서 잘라 figures/paper_fig1.png 로 저장한다.

고른 이유: 저작자가 그린 도식이다(사진, 데이터셋 예시, 모델 출력 예시는 넣지 않는다).
라이선스: arXiv abs 페이지는 CC BY 4.0, 원문 PDF에 © 또는 all rights reserved 표시가 없다(pdftotext grep 0건).
잘라낸 영역은 쪽 좌표(pt)이며 캡션 글자는 포함하지 않는다.
실행: python3 llava/analysis/crop_figures.py  (저장소 루트에서, pymupdf와 Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "figures"
JOBS = {"paper_fig1.png": (3, (100, 284, 510, 380))}   # 0-based 쪽 번호, 자를 영역(pt)


def trim(img):
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    return img.crop(box) if box else img


def main():
    OUT.mkdir(exist_ok=True)
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
