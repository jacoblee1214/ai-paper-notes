"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig2.png : 그림 2 (구성 파이프라인 3단계 도식, 원문 2쪽)
  paper_fig4.png : 그림 4 (저장소별 해결률 막대 차트, 원문 6쪽)

고른 이유: 저작자가 만든 도식과 차트다. 그림 1은 이슈 제목과 테스트 이름 예시, 그림 6은 이슈·패치 본문 발췌라서 넣지 않는다.
라이선스: arXiv abs 페이지는 CC BY 4.0. 원문 PDF에 회사 저작권 고지가 없다(pdftotext로 확인, 본문의 `Type "help", "copyright"...`는 파이썬 프롬프트 예시 줄).
실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig2.png": (1, (130, 506, 480, 558)),
    "paper_fig4.png": (5, (100, 213, 512, 306)),
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
