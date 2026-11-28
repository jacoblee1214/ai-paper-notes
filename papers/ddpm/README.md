# DDPM (arXiv:2006.11239, NeurIPS 2020) 분석 노트

*Denoising Diffusion Probabilistic Models*(Ho, Jain, Abbeel)의 식과 주장을 구현 모드로 분석한 노트입니다. **학습은 하지 않았습니다.** 신경망 대신 가우시안 혼합 데이터에서 닫힌 식으로 나오는 정확한 잡음 예측기 eps*로 식 (4)-(14)와 알고리즘 1·2를 PyTorch(float64, CPU)로 구현하고 검산했습니다(pytest 16개). 공식 TensorFlow 코드(`hojonathanho/diffusion`, 라이선스 파일 없음)는 정적으로만 읽었고 이 저장소에는 넣지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스, 원문은 arXiv:2006.11239에서 받을 수 있습니다). 그래프는 모두 직접 그렸습니다.

## 결과 요약

- 식 (4), (6), (7), (10)-(12)는 1e-13 ~ 4e-10 안에서, 식 (3) = (5)와 식 (16)은 몬테카를로 표준오차 안에서 일치합니다. eps* = -sqrt(1-abar) * 점수(5e-14). 알고리즘 2는 역방향 SDE의 이산화입니다.
- L_simple은 진짜 VLB 가중 대비 t<100을 최대 49배 낮춥니다. 정확한 eps*에서 하한의 62-79%가 t<=100에 쌓입니다.
- sigma_t^2 = beta_t는 N(0,I)에서, beta~_t는 점질량에서 최적이고 일반 데이터에서는 그 사이입니다. 논문의 "비슷한 결과"는 T=1000에서만 맞고, 단계 수가 줄면 방향이 데이터에 따라 뒤집힙니다(시드 8개, 홀름 보정). 변분 하한은 beta_t가 낫습니다.
- 정확한 denoiser에서 하한 간격은 T=1000에서 0.0097 nat(넓은 2차원 혼합)이고 단계 수에 거의 반비례합니다.
- 논문 산술(L_T, 율-왜곡 §4.3, 표 1·2·4, 모수 수 35.7M·114M)을 다시 계산했습니다. FID와 IS는 재계산할 수 없습니다.
- 공식 코드에는 논문에 없는 t=1 분산(beta~_2), x_0 클리핑, 학습률 웜업과 경사 클리핑이 있습니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_claim_verification.md` | 논문 분석, 계보, 식 검산과 토이 결과, 공식 코드 정적 대조 D1-D15와 이후 문헌의 주장 검증 C1-C8 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | 정확한 denoiser, 알고리즘 1·2, 변분 하한 항, 슬라이스 W1과 pytest 16개 |
| `analysis/derivation_checks.py` | 식 (3)-(16)과 sigma^2 최적성, 스케줄 산술 |
| `analysis/toy_experiments.py` | sigma^2 x 단계 수 비교, L_simple 바닥, 하한 간격, 식 (16) |
| `analysis/paper_arithmetic.py` | 표 1-4, 4.3절, 부록 B 산술과 U-Net 모수 수 |
| `analysis/plot_figures.py`, `figures/` | 직접 그린 그래프 네 장 |
| `analysis/*_results.json` | 위 스크립트의 결과 |

## 재실행

```bash
pip install torch numpy scipy matplotlib pytest
cd analysis
python3 -m pytest -q test_minimal_impl.py     # 약 4초
python3 derivation_checks.py                   # 약 20초
python3 toy_experiments.py                     # 약 6분(CPU, 메모리 약 0.6 GB)
python3 paper_arithmetic.py && python3 plot_figures.py
```

모든 계산은 float64 CPU이고 난수 시드는 스크립트와 결과 json에 있습니다.

## 출처와 라이선스

- 논문: arXiv 2006.11239v2, arXiv 비독점 배포 라이선스(CC 아님). 그래서 논문 그림과 원문 PDF는 이 저장소에 없습니다.
- 공식 코드 `hojonathanho/diffusion`은 라이선스 파일이 없습니다. 읽기만 했고 복사하거나 실행하지 않았습니다.
