"""논문 그림 네 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig3.png  : 그림 3 (Impossible-SWEbench 부정행위율 대 원본 통과율 산점도, 원문 5쪽)
  paper_fig4.png  : 그림 4 (Impossible-LiveCodeBench 같은 산점도, 원문 5쪽)
  paper_fig6.png  : 그림 6 (프롬프트 A–D 절제 막대, 원문 7쪽)
  paper_fig10.png : 그림 10 (LLM 모니터 탐지율 막대, 원문 10쪽)

고른 이유: 저작자가 만든 차트(산점도·막대)만 골랐다. 사진, 데이터셋 예시, 사람 얼굴, 생성 이미지는 논문에 없다.
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
    "paper_fig3.png": (4, (66, 70, 545, 208)),    # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig4.png": (4, (66, 336, 545, 478)),
    "paper_fig6.png": (6, (66, 372, 545, 498)),
    "paper_fig10.png": (9, (66, 378, 545, 500)),
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
