# With Little Power Comes Great Responsibility (arXiv:2010.06595) 분석 노트

Card et al., *With Little Power Comes Great Responsibility* (EMNLP 2020)를 구현 모드로 분석한 노트입니다. 짝지은 비교(McNemar)의 검정력과 최소 검출 효과(MDE)를 식으로 유도해 검산하고, numpy·scipy 최소 구현과 pytest 24개로 확인했으며, 저자가 공개한 데이터로 논문의 표 2·9·10, 그림 4·9·10·14, 사람 평가 메타분석, lme4 결과를 다시 계산했습니다. 논문 원문 PDF와 논문 그림, 저자 코드는 넣지 않았습니다 (원문은 arXiv:2010.06595에서 받을 수 있고, 저자 코드는 github.com/dallascard/NLP-power-analysis, Apache-2.0). numpy를 쓴 이유: 통계 방법이라 신경망이 필요 없습니다.

## 검증 결과 (요약)

- 짝지은 검정의 검정력은 (n, 정확도 차이 Δ, 불일치율 d)로 정해지고 MDE ≈ 2.80·√(d/n)입니다 (α = 0.05, 검정력 80%). 닫힌 형태를 유도했고 시뮬레이션과 맞습니다.
- 표 9의 회귀·이항 MDE 18개 중 16개, 번역 "2000문장에서 약 75%"(0.743), EMNLP 2019 메타분석 7개 수치, 그림 14의 세 비율이 재현됩니다.
- 부록 C(n = 500: 검정력 0.25, Type-M 1.9)는 본문의 χ² 식이 아니라 정확 검정으로 맞습니다 (공식 노트북이 statsmodels 기본값 사용).
- 표 10의 "46%"는 코드에서 검정력 80% 미만의 비율입니다. 검정력 충분은 57개 중 31개(54%)입니다 (저자 확인은 받지 않음).
- 그림 6의 "평균 차이"는 코드상 ±1 부호화의 계수라 두 모델 평균 차이의 절반이고, 평가자 3명의 t > 1.96 기준은 거짓 양성이 많습니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법, 주장과 근거 위치, 한계 |
| `analysis/02_prior_work.md` | 계보, 이후 문헌, 이 저장소의 관련 분석, 자료 표 |
| `analysis/03_analysis.md` | 식 유도, 최소 구현 확인, 논문 수치 재계산 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C10, 공식 코드 대조 D1–D8 |
| `analysis/minimal_impl.py` | 최소 구현 (numpy, scipy) |
| `analysis/test_minimal_impl.py` | pytest 24개 (약 4초) |
| `analysis/recompute.py`, `recompute_results.json` | 논문 수치 재계산과 결과 |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pytest
python3 -m pytest analysis/test_minimal_impl.py -q
git clone --depth 1 https://github.com/dallascard/NLP-power-analysis repo   # recompute.py의 입력(공개 CSV)
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2010.06595, arXiv 비독점 배포 라이선스(arXiv abs 페이지 확인). 원문 PDF에 회사 저작권 고지는 없습니다. 논문 그림은 싣지 않았고 그래프 두 장은 모두 우리가 그렸습니다.
- 저자 저장소의 데이터는 읽기만 했고 이 폴더에 넣지 않았습니다.
