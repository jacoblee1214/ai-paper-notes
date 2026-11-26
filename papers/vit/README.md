# ViT (arXiv:2010.11929) 분석 노트

*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy 외, Google, ICLR 2021 구두 발표)을 구현 모드로 분석한 노트입니다. 식과 산술을 검산하고(파라미터 수, 연산량, 표 6의 exaFLOPs), 패치 임베딩·[CLS]·학습 1D 위치 임베딩·pre-norm 인코더·위치 임베딩 2D 보간을 PyTorch로 직접 짜 pytest 26개로 확인했으며, 합성 이미지 토이로 "귀납 편향이 적은 모델은 데이터가 적으면 불리하다"의 방향만 확인했습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC가 아니라서, 원문은 arXiv:2010.11929에서 받을 수 있습니다). 공식 코드(`google-research/vision_transformer`, Apache-2.0)는 정적으로 읽고 대조만 했고 이 폴더에 넣지 않았습니다.

## 검증 결과 (요약)

- **표 1**: B/16 86.57M, H/14 632.05M은 맞고, 표의 L 307M은 L/16(304.3M)이 아니라 L/32(306.5M)의 개수에 맞음. 해석식은 구현 `numel`과 정확히 일치.
- **표 6 exaFLOPs**: 6 x MAC x 303M x 에폭 식이 ViT 6개 행을 0.25% 안에서 재현(B/32만 +2.0%). "연산 2–4배 적다"는 표 6 재계산에서 5개 평균 2.8–3.0배, ImageNet 1.8–2.3배.
- **연산량**: 224에서 attention 행렬곱은 층 연산의 3–4%(비 = T/(2D+M)), `FlopCounterMode`와 정확히 일치.
- **토이**(합성 도형, 시드 5개): 증강 없는 ViT는 모든 n에서 CNN에 뒤졌고 격차는 n=600의 46.5 ± 1.5%p에서 n=20000의 10.7 ± 0.8%p로 줄었으나 교차는 없음. 이동 증강이 격차의 약 70%를 메움(n=600). 모델별 튜닝이 없고 논문 재현이 아님.
- **이후 문헌**: DeiT, AugReg, Better plain ViT로 "데이터가 적으면 못한다"는 레시피 의존으로 수정, "[CLS]와 GAP는 비슷하다"는 ImageNet-1k에서 GAP +1.5–1.8%p로 수정, 큰 ViT의 고노름 토큰(register)은 원전이 몰랐던 문제.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(식 1–4, 부록 식 5–8), 실험, 주장 S1–S9와 근거 위치, 한계 |
| `analysis/02_prior_work.md` | 앞선 시도와 이후 계보(DeiT, AugReg, Swin, MAE, FlexiViT, register 등), 자료 표 |
| `analysis/03_analysis.md` | 파라미터 수·토큰 수·연산량 검산, 표 6 재계산, 최소 구현 설명, 토이 실험 결과 |
| `analysis/04_claim_verification.md` | 이후 문헌의 확인·수정 C1–C9, 공식 코드 대조 D1–D8 |
| `analysis/minimal_impl.py` | ViT 최소 구현과 해석식 (PyTorch) |
| `analysis/test_minimal_impl.py` | pytest 26개 (약 2초, CPU) |
| `analysis/formula_checks.py`, `formula_results.json` | 표 1·6 재계산, 해상도별 비용 |
| `analysis/toy_experiment.py`, `toy_results.json` | 합성 도형 토이: CNN 대 ViT, 이동 증강 (MPS 약 25분) |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install torch numpy scipy matplotlib pytest
python3 -m pytest analysis/test_minimal_impl.py -q        # 26 passed
cd analysis
python3 formula_checks.py       # 수 초
python3 toy_experiment.py       # 학습 세트 6가지 x 4팔 x 시드 5개 (MPS 약 25분, CPU면 훨씬 오래 걸림)
python3 plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(CC가 아님)이므로 논문 그림은 넣지 않았고, 그래프는 모두 직접 그렸습니다.
- 공식 코드 `github.com/google-research/vision_transformer`는 Apache-2.0이며 이 저장소에 넣지 않았고 실행하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다. 토이 데이터는 코드가 직접 만든 합성 이미지입니다.
