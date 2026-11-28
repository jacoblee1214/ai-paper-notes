# 02. 계보와 관련 분석, 자료 표

## 1. 계보

```
Sohl-Dickstein 2015 (diffusion probabilistic model, 하한 L)
Vincent 2011 (denoising score matching) ─┐
Song & Ermon 2019/2020 (NCSN, 어닐링 Langevin) ─┤
                                         ▼
                        DDPM (Ho 2020): ε-예측, L_simple, 알고리즘 1·2          ← 이 분석
                                         │
   ┌───────────────┬──────────────┬──────┴────────┬──────────────┬─────────────┐
   ▼               ▼              ▼               ▼              ▼             ▼
 Nichol 2021     DDIM 2020/21   Score SDE 2020/21   VDM 2021     CFG 2022    Latent Diffusion 2021
 (σ 학습, 코사인,  (결정적 표본,    (SDE·확률 흐름 ODE,  (SNR 형태,     (조건부     (잠재 공간)
  L_hybrid)        단계 수 절감)    VP SDE = DDPM)       학습 스케줄)    표본 품질)
                                         │
                                         ▼
                      EDM 2022 → flow matching·rectified flow → SD3(2024) → MeanFlow(2025)
                                         │
                                         └→ Diffusion Policy(2023): DDPM 식을 행동 시퀀스에 그대로 사용
```

## 2. 구성 요소별 출처 (논문의 인용 번호)

| 구성 요소 | 출처 |
|---|---|
| 확산 확률 모델, forward 사슬, 변분 하한(식 3, 5), 사후(식 6–7) | Sohl-Dickstein 외 2015 [53]. 부록 A는 "완전성을 위해 포함"이라고 밝힘 |
| ε-예측과 여러 잡음 크기의 denoising score matching | Vincent 2011 [61], Song & Ermon 2019/2020 [55, 56] |
| 표본기와 Langevin 동역학의 닮음 | Song & Ermon [55] (논문이 주장하는 새 연결은 식 12의 *하한과의* 대응) |
| U-Net(Wide ResNet), 그룹 정규화, self-attention, 시간 사인 임베딩 | PixelCNN++ [52], U-Net [48], Wide ResNet [72], GroupNorm [66], 비국소 신경망·Transformer [63, 60] |
| 이산 디코더(식 13) | PixelCNN++의 이산화 분포 [52], VAE 디코더 [34] |
| 점진적 손실 압축, 자기회귀 해석 | minimal random coding [19, 20], conceptual compression [18], subscale 순서 [38] |
| 평가: FID, IS | TTUR [21], Salimans 외 [51] (D10: 공식 코드는 TF-GAN 구현) |

## 3. 가까운 대안과 이후 논문이 바꾼 것

| 논문 | arXiv | 바꾼 것 | 이 폴더와의 연결 |
|---|---|---|---|
| Improved DDPM (Nichol & Dhariwal) | 2102.09672 | σ² 보간 학습, 코사인 스케줄, L_hybrid, 중요도 표본으로 NLL 2.94 | `04` C1–C3, C5 |
| DDIM (Song 외) | 2010.02502 | 같은 학습으로 결정적·적은 단계 표본(η = 0) | `04` C1 (η = 1.0이 DDPM, σ̂ = √β) (같은 묶음에서 작성 중) |
| Score SDE (Song 외) | 2011.13456 | 연속시간 SDE, DDPM은 VP SDE의 이산화, 확률 흐름 ODE | `04` C2, C4. `../score_sde/` |
| VDM (Kingma 외) | 2107.00630 | SNR 형태의 연속시간 하한, 스케줄 학습 | `04` C6, `03` §2.3 |
| Classifier-Free Guidance (Ho & Salimans) | 2207.12598 | 조건부와 비조건부의 ε 외삽 | (같은 묶음에서 작성 중) |
| EDM (Karras 외) | 2206.00364 | 전처리·가중·표본기의 설계 공간 정리 | `reading_list.md` 진행 순서 60 |
| SD3 | 2403.03206 | ε-예측 가중 등을 한 식으로 통합하고 시간 분포 비교 | `../sd3_rectified_flow/` (`03` §1의 통합 손실 식에서 ε-예측은 c(t) = 1) |
| MeanFlow | 2505.13447 | 평균 속도로 1단계 생성 | (같은 묶음에서 작성 중) |

## 4. 이 저장소의 다른 분석, 이웃 저장소와의 관계

- `../sd3_rectified_flow/`: 가중 w_t, 시간 표본 분포, 정확한 속도장 토이. 이 분석의 정확한 denoiser 토이(ε*)와 같은 기법(혼합 데이터의 닫힌 식)을 서로 다른 목적(속도장 대 ε)에 쓴 것입니다. ε-예측의 가중 표현(`03` §1.2)은 거기서 인용하고 여기서는 되풀이하지 않았습니다.
- `../score_sde/`, `../ddim/`, `../classifier_free_guidance/`, `../meanflow/`: 같은 묶음에서 작성 중입니다.

## 5. 자료 표

| 자료 | 내용 | 라이선스 | 비고 |
|---|---|---|---|
| github.com/hojonathanho/diffusion | 공식 TensorFlow 코드 | **없음**(GitHub `license: null`, LICENSE 파일 없음) | 별 5,372, 포크 495, 생성 2020-06-19, master HEAD `1e0dceb`(2020-06-21), API의 `pushed_at` 2023-08-29(내용은 확인 안 함). 얕은 clone 2.1 MB. 실행하지 않음 |
| hojonathanho.github.io/diffusion | 프로젝트 페이지 | | 이번에 열지 않음 |
| 사전학습 모델 | README의 Dropbox 링크 | | 받지 않음 |
| arXiv 2006.11239 | v1 2020-06-19, v2 2020-12-16 | arXiv 비독점 배포(CC 아님) | PDF 10.3 MB |
| proceedings.neurips.cc | NeurIPS 33:6840–6851 | | 학회 확인 |
| 이후 문헌 arXiv 라이선스 | N&D, Score SDE, EDM, DDIM, CFG: 비독점 배포. VDM: CC BY 4.0 | | 그림 재사용 여부 판단용 |
