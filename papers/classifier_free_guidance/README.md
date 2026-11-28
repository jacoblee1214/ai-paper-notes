# Classifier-Free Diffusion Guidance (arXiv:2207.12598, NeurIPS 2021 워크숍) 분석 노트

*Classifier-Free Diffusion Guidance*(Ho & Salimans)의 식과 해석을 구현 모드로 분석한 노트입니다. **학습은 하지 않았습니다.** 신경망 대신 2차원 클래스 조건부 가우시안 혼합의 정확한 조건부·무조건 점수로 식 (6)과 알고리즘 2를 PyTorch(float64, CPU)로 구현하고 검산했습니다(pytest 15개). 저자 공식 코드는 확인되지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스, 원문은 arXiv:2207.12598에서 받을 수 있습니다). 그래프는 모두 직접 그렸습니다.

## 결과 요약

- 식 (6)은 암시적 분류기 p(z|c)/p(z)로 안내한 것과 정확히 같습니다(3.5e-14, 1.3e-13). 논문의 w와 흔한 γ 규약은 γ = 1 + w.
- 안내된 장의 포커-플랑크 잔차는 (sigma^2/2) w (1+w) |grad log p(c|z)|^2 + (lambda만의 함수)입니다. 이 노름이 z에 무관할 때(분산이 같은 가우시안)만 일관되고, 그 외에는 끝점이 p(x|c)p(c|x)^w가 아닙니다.
- 토이(w=2, T=256): CFG-DDIM의 재현율 0.501(의도한 표적 0.943), w=8에서 지지집합 밖 표본 25.7%(표적 0.05%). 같은 식 (6)이 DDIM과 조상 표집에서 다른 분포를 만듭니다. 단계 수를 1024까지 늘려도 불일치가 남고(sW1 0.564, 바닥 0.022), 점수만 일관된 것으로 바꾸면 0.042로 사라집니다.
- 구간 안내를 토이에 옮기면 높은 잡음에서 안내를 끄는 것만으로 지지집합 밖 비율이 1.46%에서 0.08%로 줄어듭니다(w=4).
- 논문 표 1·2의 산술을 다시 계산했습니다. 본문의 "FID monotonically decreasing" 서술은 표와 맞지 않고, 그림 5 범례(T=512)는 표 2(T=1024)와 다릅니다. FID와 IS는 재계산할 수 없습니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_claim_verification.md` | 논문 분석, 계보, 식 검산과 토이 결과와 표 재계산, 이후 문헌의 주장 검증 C1-C10 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | 정확한 점수 모델, CFG, DDIM·조상 표집, 정확한 기울임 격자, 오라클 점수, 포커-플랑크 잔차와 pytest 15개 |
| `analysis/derivation_checks.py` | 항등식, 가우시안 닫힌 식, 잔차 닫힌 식 검산 |
| `analysis/toy_experiments.py` | w 스윕, 단계 수, 오라클, 구간 안내 |
| `analysis/paper_arithmetic.py` | 표 1·2 재계산 |
| `analysis/plot_figures.py`, `figures/` | 직접 그린 그래프 다섯 장 |
| `analysis/*_results.json` | 위 스크립트의 결과 |

## 재실행

```bash
pip install torch numpy scipy matplotlib pytest
cd analysis
python3 -m pytest -q test_minimal_impl.py     # 약 5초
python3 derivation_checks.py                   # 약 20초
python3 toy_experiments.py                     # 약 7분(CPU, 메모리 약 0.5 GB)
python3 paper_arithmetic.py && python3 plot_figures.py
```

`toy_experiments.py`는 인자로 E1 E2 E3 E4 중 일부만 돌릴 수 있고, 일부만 돌리면 `toy_results_<목록>.json`으로 저장합니다. 올려 둔 `toy_results.json`은 E1, E2+E4, E3를 따로 돌려 합친 것입니다. 모든 계산은 float64 CPU이고 난수 시드는 스크립트와 결과 json에 있습니다.

## 출처와 라이선스

- 논문: arXiv 2207.12598v1, arXiv 비독점 배포 라이선스(CC 아님). 그래서 논문 그림과 원문 PDF는 이 저장소에 없습니다.
- 저자 공식 코드는 확인되지 않았습니다.
