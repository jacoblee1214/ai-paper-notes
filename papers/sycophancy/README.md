# Towards Understanding Sycophancy in Language Models (arXiv:2310.13548, ICLR 2024) 분석 노트

Sharma et al.의 논문(arXiv v4, 2025-05-10)을 그림 벡터 복원, 공개 데이터셋 문항 수 기반 재계산, 공식 데이터 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다 (arXiv:2310.13548, github.com/meg-tong/sycophancy-eval에서 받을 수 있습니다). 논문 그림은 넣지 않았고 그래프 세 장은 논문 수치로 새로 그린 것입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 문제, 측정, 원인 분석, 한계 |
| `analysis/02_prior_work.md` | 계보, 후속 문헌, 자료 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 일관성 표 |
| `analysis/04_code_verification.md` | 공식 데이터·노트북 대조 |
| `analysis/extract_figures.py` | PDF 그림에서 값 복원 (`figure_values.json`) |
| `analysis/recompute.py` | 재계산과 검정 (`recompute_results.json`) |
| `analysis/dataset_audit.py` | 공개 데이터셋 구성 감사 |
| `analysis/toy_bayes_logistic.py` | 선호 회귀 합성 데이터 토이 |
| `analysis/plot_figures.py` | 그래프 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
# 논문 PDF를 paper/paper.pdf에 둡니다 (curl -L -o paper/paper.pdf https://arxiv.org/pdf/2310.13548)
python3 analysis/extract_figures.py
python3 analysis/recompute.py
python3 analysis/toy_bayes_logistic.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 원문 PDF에서 회사 저작권 고지는 찾지 못했습니다.
- 공식 데이터 저장소에는 LICENSE가 없어 이 저장소에 넣지 않았습니다. 데이터셋 감사(`dataset_audit.py`)는 직접 clone한 사본(`repo/`)을 읽습니다.
- 모델 호출과 API 사용은 없습니다.
