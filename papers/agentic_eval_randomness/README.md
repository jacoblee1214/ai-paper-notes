# On Randomness in Agentic Evals (arXiv:2602.07150, ICLR 2026 워크숍) 분석 노트

*On Randomness in Agentic Evals* (Bjarnason, Silva, Monperrus, arXiv v3, 2026-03-25)를 저자들이 공개한 실행별 평가 결과(HF, 5.8 MB)로 재계산하고, 식 11과 표 2를 검산하고, 문항 단위 분산 분해와 이항 모형 시뮬레이션으로 권고를 확인한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2602.07150에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 재현 가능성 (요약)

- **궤적 생성은 재현하지 않았습니다.** 8×A100 vLLM, 문항별 컨테이너, 상용 API가 필요하고 논문은 약 3,500 GPU시간을 썼습니다.
- **공개 평가 결과의 재계산, 식 검산, 시뮬레이션은 CPU에서 수 초에서 30초면 됩니다.**
- 공식 코드(ASSERT-KTH/agentic-evals-lab, 라이선스 파일 없음)는 실행하지 않고 정적으로만 읽었고 이 저장소에는 넣지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` | 문제, 방법, 결과, 주장과 근거 대응, 한계 |
| `analysis/02_prior_work.md` | 계보(비결정성, Error Bars, Precipice), 발표 이후 연구, 자료 표 |
| `analysis/03_analysis.md` | 표 1·3 재계산, pass@k, 문항 분산 분해, 표 2 검산, 문항 표본 오차, 온도 효과 검정 |
| `analysis/04_code_verification.md` | 공식 코드·데이터 정적 대조 (일치 K1–K7, 차이 D1–D7) |
| `analysis/recompute_runs.py` | 표 1·3, pass@k, 분산 분해, 온도 효과 (numpy, scipy) |
| `analysis/power_check.py` | 식 11과 표 2, 정확한 t, 이항 시뮬레이션, 최소 탐지 효과 |
| `analysis/empty_patches.py` | 실행별 빈 패치 수와 해결률 |
| `analysis/plot_figures.py` | 그래프 세 장 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 두 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 세 장 (우리가 그림)과 논문 그림 두 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
# data/ 에 HF ASSERT-KTH/agentic-evals-artifacts (CC BY 4.0) 의 평가 결과 JSON을 받는다.
#   각 run 디렉터리에서 크기 100 KB 미만인 *.json (SWE-bench 하네스 결과, resolved_ids 포함), 12개 설정 x 10 run, 합계 약 6 MB
python3 -I analysis/recompute_runs.py
python3 -I analysis/power_check.py
python3 -I analysis/empty_patches.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv abs 페이지 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 두 장 (그림 1, 2)은 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2602.07150"). 모델 추론 텍스트 상자와 검정력 곡선은 넣지 않았습니다.
- 공식 저장소는 라이선스 파일이 없어 코드를 넣지 않았습니다. 결과 데이터는 HF 카드 기준 CC BY 4.0이지만 이 저장소에는 넣지 않았습니다.
