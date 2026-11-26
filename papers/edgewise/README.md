# EdgeWise (arXiv:2610.17560) 분석 노트

*EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation* (arXiv v2, 2026-10-15)를 논문 표 재계산, 짝지은 불일치율 시나리오 통계, 비용 식 해체와 민감도, RL 식 검산으로 분석한 자료입니다. 공식 코드, 가중치, 궤적이 공개되어 있지 않아 코드 대조는 하지 못했고 주장 검증(C1–C19)으로 대신했습니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2610.17560에서 받을 수 있습니다). 코드 실행, API 호출, 설치는 하지 않았습니다.

## 요약

- 헤드라인 Δ 8칸과 평균 절감 45.86%는 표에서 재현됩니다. 초록의 62%(SWE)는 최댓값, 34%(TB)는 평균 집계입니다.
- 문항이 TB 89개, SWE 110개이고 Pass@1은 단일 실행으로 보여서 "같거나 낫다"는 표본 오차와 구분되지 않습니다 (불일치율 0.2에서 MDE 13.3%p, 11.9%p. Holm 후 유의한 칸 0).
- 비용 62%·53%·35%는 GPT-5.5 xhigh 기준선 대비이고, high 대비로는 TB −8%, SWE 13%입니다. 같은 노력 수준끼리는 27–53%입니다.
- 핵심 단계는 Write/Edit만 Holm 후 유의(r² = 4%)이고, 단계별 신용 할당(토큰 이점)의 정확도 이득은 6과제(p ≈ 0.24)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 문제, 관찰, 핵심 단계, 위임 방식, 실험, v1 대 v2, 한계 |
| `analysis/02_prior_work.md` | 계보와 가까운 연구, 이 저장소의 다른 분석과의 관계, 자료 표 |
| `analysis/03_analysis.md` | 문항 수 역산, 구간, 짝지은 시나리오 검정, 비용, 핵심 단계, RL 식, AgentBaton 대조, 로컬 배치 |
| `analysis/04_claim_verification.md` | 주장 C1–C19와 구현 가능성 |
| `analysis/paper_tables.py` | 논문 표와 그림 값을 옮겨 적은 데이터 |
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

- 논문은 arXiv에 CC BY 4.0으로 올라와 있지만 원문 PDF 첫 쪽에 "© 2026 NVIDIA. All rights reserved."가 있어 논문 그림은 넣지 않았습니다. 그래프는 모두 직접 그렸습니다(그림 1의 곡선은 눈으로 읽은 값이라 판독 오차가 있습니다).
- 공식 코드는 공개되어 있지 않습니다(프로젝트 페이지 "Code Soon").
