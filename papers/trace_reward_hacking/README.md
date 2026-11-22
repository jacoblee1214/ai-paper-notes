# TRACE: Is It Thinking or Cheating? (arXiv:2510.01367, ICLR 2026 구두) 분석 노트

*Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort* (arXiv v4, 2026-03-02)를 그림 라벨 재계산, TRACE 점수의 식 검산, 가상 곡선 토이, 주장별 검증으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2510.01367에서 받을 수 있습니다). 논문에는 표가 없고 코드·점수 로그가 공개되지 않아(공식 코드를 찾지 못함), 그림 수치는 막대 라벨로만 검증했습니다.

## 재현 가능성 (요약)

- **논문 수치 재현은 하지 않았습니다.** RL 학습(GPU)과 모델 실행이 필요하고 코드가 공개되지 않았습니다.
- **그림 라벨 재계산, 식 검산, 토이는 완료했습니다.** CPU에서 수 초입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 네 장 포함) |
| `analysis/01_paper_analysis.md` | 문제, 방법, 실험, 한계, v1과 v4의 차이 |
| `analysis/02_prior_work.md` | 선행 계보와 이름이 같은 다른 논문 |
| `analysis/03_analysis.md` | 점수 정의 검산, 이득 수치 재계산, 표본 수 민감도, 임계값·과잉 사고 토이, 원인 2×2 |
| `analysis/04_claim_verification.md` | 주장별 검증 C1–C11, 재현 제안 |
| `analysis/recompute_trace.py` | 재계산과 토이 (numpy, scipy) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 네 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 두 장 (우리가 그림)과 논문 그림 네 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf pillow
python3 analysis/recompute_trace.py
python3 analysis/plot_figures.py
python3 report/crop_figures.py   # paper/paper.pdf 가 필요 (arXiv:2510.01367v4)
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv abs 페이지 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 논문 그림 네 장 (그림 1의 차트, 8, 9, 10)은 차트만 골랐고 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2510.01367"). 사진, 말풍선, 모델 출력 발췌는 넣지 않았습니다.
