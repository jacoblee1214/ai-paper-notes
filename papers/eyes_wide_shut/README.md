# Eyes Wide Shut? (arXiv:2401.06209, CVPR 2024 구두) 분석 노트

*Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs* (arXiv v2, 2024-04-25)를 표·그림 값 재계산, 공개 데이터(MMVP, MMVP-VLM) 구조 감사, 합성 토이, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2401.06209에서 받을 수 있습니다). 논문 그림은 차트인 그림 4 한 장만 넣었고 출처를 적었습니다. 모델 실행, 가중치 다운로드, API 호출은 하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 방법, 실험, 한계, v1 대 v2 |
| `analysis/02_prior_work.md` | 계보, 후속 연구, 자료 표 |
| `analysis/03_analysis.md` | 우연 수준, 150쌍 구간·MDE, 선별 효과, 패턴별 상관, MoF 교환 관계, 데이터 감사 |
| `analysis/04_code_verification.md` | 공식 코드·데이터 대조 D1-D16 |
| `analysis/paper_tables.py` | 논문 표·그림 수치를 옮겨 적은 데이터 |
| `analysis/figure_extract.py` | 그림 6을 PDF 벡터에서 읽기(논문 PDF 필요) |
| `analysis/chance_toy.py`, `image_stats.py` | MMVP 파일 구조·우연 수준·이미지 크기(HF 데이터 필요) |
| `analysis/stats_check.py`, `selection_toy.py` | 표·그림 통계 검산, 선별 효과 토이 |
| `analysis/plot_figures.py`, `report/crop_figures.py` | 그래프, 논문 그림 4 자르기 |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf pillow
curl -sL https://arxiv.org/pdf/2401.06209v2 -o paper/paper.pdf
# 데이터(합계 약 7 MB): HF MMVP/MMVP, MMVP/MMVP_VLM 의 Questions.csv 와 이미지를 data/mmvp, data/mmvp_vlm 에 받는다
python3 analysis/figure_extract.py && python3 analysis/chance_toy.py && python3 analysis/image_stats.py
python3 analysis/stats_check.py && python3 analysis/selection_toy.py && python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 원문 PDF에 회사 저작권 표시는 없습니다. 그림 4는 "© 저자, CC BY 4.0, 출처 arXiv:2401.06209"로 실었고, MMVP 사진 쌍과 모델 답 예시, 사진이 든 도식은 넣지 않았습니다.
- 공식 코드(`github.com/tsb0601/MMVP`)와 데이터는 넣지 않았습니다. 데이터 카드는 MIT이지만 이미지는 ImageNet-1k와 LAION-Aesthetics에서 온 것이라 재배포 권리는 미확인입니다.
