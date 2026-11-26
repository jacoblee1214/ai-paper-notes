"""논문 그림 한 장을 원문 PDF에서 잘라 report/figures/에 저장한다 (개인용 리포트와 게시본).

  paper_fig4.png : 그림 4 (MMVP 결과 막대 차트, 원문 4쪽)
고른 이유: 차트만이다. 그림 7(MoF 도식)에는 고양이 사진이 들어 있고 그림 1, 2, 3, 5, 8, 10–12는 MMVP 사진 쌍·모델 답 예시라 넣지 않는다(CLAUDE.md).
라이선스: arXiv abs CC BY 4.0, 원문 PDF에 © 또는 All rights reserved 표시 없음(pdftotext 확인). 캡션 글자는 자르지 않는다.
실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path
import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig4.png": (3, (50, 508, 290, 672)),
}


def trim(img):
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img.convert("RGB"), bg).getbbox()
    return img.crop(box) if box else img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)
    for name, (pi, (x0, y0, x1, y1)) in JOBS.items():
        pix = doc[pi].get_pixmap(dpi=300, clip=pymupdf.Rect(x0, y0, x1, y1))
        tmp = OUT / (name + ".tmp.png"); pix.save(tmp)
        img = trim(Image.open(tmp)); img.save(OUT / name); tmp.unlink()
        print("wrote", OUT / name, img.size)


if __name__ == "__main__":
    main()
