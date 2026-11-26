# AgentBaton (arXiv:2610.35959) 분석 노트

*AgentBaton: Efficient Model Collaboration in Agentic Tasks* (arXiv v1, 2026-10-25)를 논문 표 재계산, 짝지은 불일치율 시나리오 통계, 비용 식 해체와 민감도로 분석한 자료입니다. 공식 코드, 가중치, 궤적이 공개되어 있지 않아 코드 대조는 하지 못했고 주장 검증(C1–C14)으로 대신했습니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2610.35959에서 받을 수 있습니다). 코드 실행, API 호출, 설치는 하지 않았습니다.

## 요약

- 헤드라인 숫자(비용 평균 65.9%, 12/22칸, 최대 +8.8%p)는 표에서 모두 재현됩니다.
- 문항이 DeepSWE 113개, Terminal-Bench 89개이고 반복 실행이 없어서 "정확도를 지킨다"와 "+8.8%p"는 표본 오차와 구분되지 않습니다 (짝지은 불일치율 0.2에서 p=0.064, Holm 후 유의한 칸 0).
- 표 3과 표 6의 같은 구성에서 값이 다른 칸이 있습니다 (GLM-5.3 단독 61.1 대 69.0 등).
- 방향 비대칭은 정확도가 아니라 비용에서 나오고, 로컬 0원 규약과 비용의 볼록성으로 대부분 설명됩니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 문제, 설계, 결과, 탐지기, 한계 |
| `analysis/02_prior_work.md` | 계보와 가까운 연구, 이 저장소의 다른 분석과의 관계, 자료 표 |
| `analysis/03_analysis.md` | 문항 수 역산, 구간, 짝지은 시나리오 검정, 선택 효과, 비용, 방향 비대칭, 탐지기, 파레토 |
| `analysis/04_claim_verification.md` | 주장 C1–C14와 구현 가능성 |
| `analysis/paper_tables.py` | 논문 표 1, 3, 4, 6을 옮겨 적은 데이터 |
| `analysis/table_check.py`, `power_check.py`, `cost_model.py` | 재계산 스크립트와 `*_results.json` (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
cd analysis && python3 table_check.py && python3 power_check.py && python3 cost_model.py && cd ..
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 올라와 있지만 원문 PDF 첫 쪽에 "© 2026 NVIDIA. All rights reserved."가 있어 논문 그림은 넣지 않았습니다. 그래프는 모두 직접 그렸습니다.
- 공식 코드는 공개되어 있지 않습니다. 하네스 mini-SWE-agent는 MIT입니다.
