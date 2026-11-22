"""논문 그림을 원문 PDF(v4)에서 잘라 report/figures/에 저장한다.

  paper_fig1.png  : 그림 1 오른쪽 차트 (모니터 7/14/72B 대 TRACE F1, 원문 1쪽)
  paper_fig8.png  : 그림 8 (훈련 중 TRACE 점수·응답 길이·KL, 원문 6쪽)
  paper_fig9.png  : 그림 9 (코드 F1, 원문 6쪽)
  paper_fig10.png : 그림 10 (수학 F1, 원문 6쪽)

고른 이유: 저작자가 만든 차트(막대·선)만 골랐다. 그림 1의 왼쪽 말풍선과 아이콘, 모델 출력 텍스트 발췌는 넣지 않는다.
라이선스: arXiv abs 페이지는 CC BY 4.0. 원문 PDF에 © 또는 All rights reserved 표시가 없다(pdftotext 확인).
실행: python3 report/crop_figures.py   (pymupdf, Pillow)
"""
from pathlib import Path
import pymupdf
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "paper.pdf"
OUT = ROOT / "report" / "figures"
JOBS = {
    "paper_fig1.png": (0, (251, 440, 475, 545)),
    "paper_fig8.png": (5, (105, 72, 506, 174)),
    "paper_fig9.png": (5, (268, 390, 506, 496)),
    "paper_fig10.png": (5, (105, 624, 506, 724)),
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
        img = trim(Image.open(tmp)); img.save(OUT / name); tmp.unlink(); print("wrote", name, img.size)

if __name__ == "__main__":
    main()
