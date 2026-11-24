"""논문 그림 두 장을 원문 PDF에서 잘라 report/figures/에 저장한다.

  paper_fig1.png : 그림 1 (층별 상위 3 채널 크기, Llama 2 7B와 Qwen3 8B, 원문 4쪽)
  paper_fig5.png : 그림 5 (step-up 전후 spike 토큰 코사인 유사도, 원문 6쪽)

고른 이유: 저작자가 그린 차트다(모델 출력 텍스트, 사진, 사람 얼굴 없음). 그림 2–4, 6–8도 차트지만 두 장이면 충분해 넣지 않았다.
라이선스: arXiv abs 페이지는 CC0 1.0. 원문 PDF에서 ©, copyright, all rights, license를 pdftotext로 찾았으나 없었다.
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
    "paper_fig1.png": (3, (55, 62, 545, 222)),   # 0-based 쪽 번호, 자를 영역(pt)
    "paper_fig5.png": (5, (307, 64, 545, 156)),
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
