"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig6.png : 그림 6 (이미지 처리와 시각 표현 레이더 차트, 원문 6쪽)
  paper_fig7.png : 그림 7 (DINOv2 융합과 base 대 instruct LM 레이더 차트, 원문 7쪽)

고른 이유: 저작자가 그린 레이더 차트(도식)다. 그림 1, 4, 5, 9도 레이더 차트이지만 같은 정보라 둘만 썼다.
그림 2(아키텍처 도식 일부에 이미지 예시), 그림 3(평가 개요), 그림 8(혐오 발언 예시), 그림 9 오른쪽과 그림 11(모델 출력 예시)은
데이터셋·모델 출력 예시를 담고 있어 넣지 않는다.
라이선스: arXiv abs 페이지는 CC BY 4.0. 원문 PDF의 저작권 문구는 ICML 표준 "Copyright 2024 by the author(s)"뿐이고
회사 저작권 고지는 없다(pdftotext로 copyright, (c), all rights reserved 검색).
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
    "paper_fig6.png": (5, (55, 57, 542, 240)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig7.png": (6, (55, 57, 542, 225)),
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
