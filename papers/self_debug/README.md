# Teaching Large Language Models to Self-Debug (arXiv:2304.05128) 분석 노트

Chen et al., *Teaching Large Language Models to Self-Debug* (ICLR 2024)를 공개 표와 그림 값으로 재계산하고, 이득을 단계별로 분해하고, 짝지은 McNemar 검정의 가능 범위를 구해 주장을 검증한 노트입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2304.05128에서 받을 수 있습니다). 논문 그림은 싣지 않았고, 우리가 표 2 값에서 그린 그래프 한 장만 있습니다. 논문의 공식 코드와 문항별 결과는 공개되어 있지 않아 문항 단위 재계산은 하지 못했습니다.

## 검증 결과 (요약)

- 표 1·2·3·5, 그림 6·8의 정확도 124개가 모두 문항 수(Spider 1,034 / TransCoder 560 / MBPP 500)의 정수 개수와 맞습니다. 그림 6(b)의 난이도별 값에서 표 2(a)의 정답 841 → 870(+29문항)이 정확히 복원됩니다.
- 이득의 대부분은 통과·실패 판정을 받는 재시도입니다. 단순 피드백이 총이득의 28–81%이고, 코드 설명(rubber duck)이 실행 결과 위에 더하는 몫은 −0.2 ~ +2.0%p입니다.
- TransCoder는 피드백에 쓰는 단위 테스트 10개가 채점 테스트와 같습니다. MBPP는 3개 중 1개만 보여 더 정직한 설정입니다.
- 실행 없는 설정(표 3)은 6개 셀 모두 비짝 95% 구간이 0을 포함하고, MBPP Codex의 단순 피드백은 61.4 → 57.6으로 떨어집니다(논문 언급 없음).
- "10배 표본 효율"은 Spider에서만 성립하고 그 차이는 1문항입니다. TransCoder와 MBPP에서는 성립하지 않습니다.
- 표 2(a)의 Codex 열 기준선 81.3은 32샘플 실행 다수결입니다(그림 6(a)와 일치).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 평가 설정, 실험, 주장과 근거 위치, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 계보, 이후 문헌, 자료 표 |
| `analysis/03_analysis.md` | 재계산과 검정 |
| `analysis/04_claim_verification.md` | 주장별 판정 (C1–C10) |
| `analysis/recompute_tables.py` | 재계산 스크립트 (scipy) |
| `analysis/plot_figures.py` | 그래프 생성 (matplotlib) |
| `report/figures/own_gain_decomposition.png` | 이득 분해 그래프 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute_tables.py   # recompute_tables_results.json
python3 analysis/plot_figures.py
```

입력은 스크립트 안에 옮겨 적은 논문 표와 그림 값뿐이라 데이터를 따로 받을 필요가 없습니다.

## 출처와 라이선스

- 논문: arXiv 2304.05128, CC BY 4.0 (arXiv abs 페이지 확인). 원문 PDF에 회사 저작권 고지는 없습니다.
- 공식 코드는 공개되어 있지 않습니다.
