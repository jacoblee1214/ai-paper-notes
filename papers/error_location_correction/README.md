# LLMs cannot find reasoning errors, but can correct them given the error location (arXiv:2311.08516) 분석 노트

Tyen et al., *LLMs cannot find reasoning errors, but can correct them given the error location* (ACL 2024 Findings)을 공개 원자료로 재계산해 검증한 노트입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2311.08516에서 받을 수 있습니다). 논문 그림은 싣지 않았고, 우리가 표 6 값에서 그린 그래프 한 장만 있습니다.

## 검증 결과 (요약)

- 표 2(데이터 구성), 표 4 "Overall"(과제 평균) 10/11셀, 표 5 기준선(78.11), 표 6 분모와 값이 재계산과 일치합니다.
- 오답 추적에서 오류 위치를 주면 5/5 과제에서 수정 정답률이 오르고, 무작위 위치보다 큽니다(Fisher p ≤ 0.022). 무작위 기준선은 한 번 실행한 값입니다.
- 분류기 개선은 단어 정렬과 섞인 물건 추적에서만 강합니다.
- 불일치 4건과 모호함 1건을 찾았습니다(`analysis/03_analysis.md` §7). 모델 예측이 공개되지 않아 셀 값 자체는 재계산하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 근거 위치 |
| `analysis/02_prior_work.md` | 계보, 구성 요소 출처, 공식 저장소 정보 |
| `analysis/03_analysis.md` | 원자료 재계산과 검정, 불일치 목록 |
| `analysis/04_claim_verification.md` | 주장별 판정 |
| `analysis/recompute_tables.py` | 재계산 스크립트 (scipy) |
| `analysis/plot_figures.py` | 그래프 생성 (matplotlib) |
| `report/figures/own_delta_ci.png` | 표 6 재구성 그래프 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
# 원자료: raw.githubusercontent.com/WHGTyen/BIG-Bench-Mistake/main/ 의 .jsonl 다섯 개를 data/에 받는다
python3 analysis/recompute_tables.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2311.08516, CC BY 4.0 (arXiv abs 페이지 확인). 원문 PDF에 회사 저작권 고지는 없습니다.
- 데이터셋 저장소: `WHGTyen/BIG-Bench-Mistake`, Apache-2.0. 원자료는 이 저장소에 넣지 않았습니다.
