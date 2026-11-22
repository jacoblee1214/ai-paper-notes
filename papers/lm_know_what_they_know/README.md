# Language Models (Mostly) Know What They Know (arXiv:2207.05221) 분석 노트

Kadavath et al. (Anthropic), *Language Models (Mostly) Know What They Know* (arXiv v4, 2022-11-21)를 논문 표와 그림에서 읽은 값으로 재계산하고 직접 짠 numpy 토이로 검증한 노트입니다. 학회 표기 없음 (arXiv). 논문 원문 PDF는 넣지 않았습니다 (arXiv:2207.05221에서 받을 수 있습니다). 논문 그림은 싣지 않았고 우리가 그린 그래프 두 장만 있습니다. 모델이 내부 모델이고 코드와 결과 파일이 없어 재현은 하지 못했습니다.

## 검증 결과 (요약)

- Table 1과 Fig 14의 52B AUROC는 판독 오차 안에서 일치합니다.
- "OOD AUROC가 크기와 함께 증가"는 Mixed-Arithmetic에서만 단조입니다. LAMBADA와 Python 함수 합성은 3B에서 꺾입니다 (점 4개, 학습 1회).
- P(True)>0.5 조건부 정확도 격차는 LAMBADA에서 12B와 52B가 같습니다(0.105). coverage가 보고되지 않았고, 52B Codex에서는 표본의 53% 이상이 "False"로 분류된 셈입니다.
- Codex의 자기 평가 Brier는 모든 크기에서 기저율 상수 예측기보다 나쁩니다. LAMBADA 12B와 52B는 이깁니다.
- ECE는 완벽히 보정된 예측기에서도 표본이 유한하면 0보다 큽니다(예측 5,000개, 10구간에서 약 0.014).
- Table 3의 시작 모델 효과는 한 시험 세트에서만 보입니다.
- 이후 문헌(초록 수준)은 RLHF 뒤 보정 붕괴와 에이전트의 일관된 과신을 보고합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 지표 정의, 주장과 근거 위치 |
| `analysis/02_prior_work.md` | 계보, 이후 문헌 |
| `analysis/03_analysis.md` | 지표 검산, 표와 그림 값 재계산, 토이 |
| `analysis/04_claim_verification.md` | 주장별 판정 (C1–C12) |
| `analysis/recompute_calibration.py` | 재계산과 토이 (numpy, scipy) |
| `analysis/recompute_calibration_results.json` | 위 스크립트 결과 |
| `analysis/plot_figures.py` | 그래프 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute_calibration.py   # recompute_calibration_results.json
python3 analysis/plot_figures.py
```

입력은 스크립트 안에 옮겨 적은 논문 표와 그림에서 읽은 값뿐이라 데이터를 따로 받을 필요가 없습니다.

## 출처와 라이선스

- 논문: arXiv 2207.05221, CC BY 4.0 (arXiv abs 페이지 확인). 원문 PDF에 회사 저작권 고지는 없습니다.
- 공식 코드는 없습니다.
