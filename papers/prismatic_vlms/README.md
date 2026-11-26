# Prismatic VLMs (arXiv:2402.07865, ICML 2024 포스터) 분석 노트

*Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models* (arXiv v2, 2024-05-30)를 부록 표 재계산, 통계 검정, 계산량 검산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2402.07865에서 받을 수 있습니다). 논문 그림은 레이더 차트인 그림 6, 7만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **학습과 12개 벤치마크 평가의 재현은 이 분석에서 하지 않았습니다.** 논문은 8×A100에서 7B 모델을 약 9시간 학습합니다.
- **표 재계산, 검정, 계산량 산술은 CPU에서 수 초입니다.**
- 공식 코드는 실행하지 않았고 가중치도 받지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 평가 묶음, 실험, 한계 |
| `analysis/02_prior_work.md` | 계보, 구성 요소의 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | 문항 수와 MDE, 설계 축별 검정, p 값 재현 시도, 계산량 검산, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (차이 D1–D10) |
| `analysis/parse_tables.py`, `analysis/tables.json` | 논문 부록 표 2–4 파싱 결과 (58행) |
| `analysis/recompute.py` | 재계산, 검정, 시뮬레이션 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 두 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 세 장 (우리가 그림)과 논문 그림 두 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
python3 analysis/recompute.py        # tables.json이 있으면 바로 실행됩니다
python3 analysis/plot_figures.py
python3 report/crop_figures.py       # paper/paper.pdf를 받아 둔 뒤
```

`parse_tables.py`는 `paper/paper.txt`(pdftotext 결과)가 있어야 실행되고, 결과는 `tables.json`에 이미 들어 있습니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 링크 확인). 원문의 저작권 문구는 ICML 표준 "Copyright 2024 by the author(s)"뿐이고 회사 저작권 고지는 없습니다.
- 논문 그림 두 장 (그림 6, 7, 레이더 차트)은 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2402.07865"). 데이터셋 예시, 모델 출력 예시, 혐오 발언 예시가 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/TRI-ML/prismatic-vlms`(MIT)와 `github.com/TRI-ML/vlm-evaluation`(LICENSE 파일 MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.
