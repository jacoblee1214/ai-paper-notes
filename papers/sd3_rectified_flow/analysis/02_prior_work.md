# 02. 계보, 구성 요소의 출처, 관련 자료

## 1. 계보 (이 논문이 어디에 서 있나)

```
Sohl-Dickstein 2015 · DDPM (Ho 2020) · score SDE (Song 2020) ──┐
LDM (Rombach 2022, 잠재 공간 diffusion, ε/linear)               │
EDM (Karras 2022, F-예측, lognormal σ, 가중 통일)               ├─→ 통합 손실 L_w (Kingma & Gao 2023)
Flow matching (Lipman 2023) · Rectified flow (Liu 2022)         │      │
  · Stochastic interpolants (Albergo & Vanden-Eijnden 2022)   ──┘      ▼
                                                              이 논문: RF + 시간 표본기 π(t) 비교(61개)
DiT (Peebles & Xie 2023) · PixArt-α(CrossDiT) · UViT (simple diffusion, Hoogeboom 2023)
        └──→ MM-DiT (모달리티별 가중치, 어텐션에서만 결합)
simple diffusion (Hoogeboom 2023): 해상도에 따른 log-SNR 이동 ──→ 식 (23)–(25)
ViT-22B류 QK-norm (Dehghani 2023, Wortsman 2023) ──→ 그림 5
```

- **통합 손실의 의미.** 이 논문이 새로 만든 식은 거의 없습니다. 식 (13)은 Kingma & Gao의 가중 ELBO 틀이고, 기여는 "RF 손실 = 어떤 w_t이고 π(t)를 바꾸면 w가 어떻게 바뀌는가"(식 18)와 그 대규모 비교입니다. 우리 검산(`03` §1)은 이 식들이 맞고 (상수 배 제외) 논문이 쓴 w_t가 정확히 어떤 상수 배인지 보였습니다.
- **시간 이동의 선행.** 논문이 직접 "similar to (Hoogeboom et al., 2023)"라고 적은 로그 SNR 이동(식 25)입니다. 우리 `03` §3에서 이 이동이 어떤 가정에서 정확하고 어디서 어긋나는지 보였습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문의 변경 |
|---|---|---|
| 잠재 공간 이미지 diffusion | LDM (Rombach 2022) | 채널 16, 정규화 상수는 본문에 없음(`04` D9) |
| 텍스트 조건: 고정 사전학습 인코더 3개 | Imagen·eDiff-I (T5), SDXL (CLIP 2개) | 세 개를 모두 쓰고 인코더마다 46.4% 독립 드롭아웃 |
| 변조(adaLN)와 패치 토큰 | DiT | 시간과 c_vec(풀링 텍스트)을 합쳐 변조 |
| 텍스트-이미지 결합 | DiT의 단일 가중치 연결, PixArt의 교차 어텐션 | **모달리티별 가중치 + 합친 어텐션** (MM-DiT) |
| 선형 보간 forward와 속도 회귀 | RF·FM·SI | 시간 분포 π(t) 설계 |
| 로짓정규 시간 분포 | Atchison & Shen 1980 | RF 학습에 도입, (m,s) 격자 탐색 |
| 해상도별 시간 이동 | simple diffusion | 상수 이미지 가정으로 재유도, 학습·샘플링 모두 적용 |
| QK 정규화 | ViT 학습 안정화 문헌 | 두 스트림 모두, RMSNorm |
| 합성 캡션 | DALL-E 3 | CogVLM으로 50:50 |
| DPO | Diffusion-DPO (Wallace 2023) | LoRA rank 128 |
| 평가 | GenEval, T2I-CompBench, Parti prompts 사람 평가, CLIP score, CLIP 특징 FID | 정식화 순위는 비지배 정렬 |

## 3. 가까운 대안과 후속

- **SiT (Ma 2024).** 클래스 조건 ImageNet에서 flow 정식화를 비교한 선행 연구(논문 §1이 인용). 이 논문은 이를 텍스트 조건·대규모로 넓혔습니다.
- **EDM 계열(Karras 2022).** 표 1에서 최하위 군(edm(−1.2,1.2)가 15.58). 우리 환산(`03` §1.4)에서 edm(−1.2,1.2)는 t ≈ 0.07(데이터 쪽)에 몰린 밀도에 해당하지만, 같은 쪽으로 치우친 eps/linear는 6위라 치우침만으로 순위를 설명하지 못합니다.
- **후속(이 폴더에서 정적으로 확인한 것).** SD3.5(추론 참조 코드, `04`), Hugging Face diffusers의 SD3 학습 스크립트(로짓정규·mode·CosMap 구현과 `FlowMatchEulerDiscreteScheduler`의 정적 shift, 해상도 따라 이동하는 `use_dynamic_shifting` 옵션)(`04` D11–D14).
- **1-step·평균 속도 계열.** MeanFlow(`../meanflow/`)는 이 논문의 "큰 모델일수록 적은 스텝에서 덜 열화"(표 6)와 대비됩니다.

## 4. 이 저장소의 다른 분석과의 관계

| 폴더 | 관계 |
|---|---|
| `meanflow/` | 속도장 학습의 한 스텝 생성 쪽. 이 논문의 RF 손실이 출발점 |
| `ddpm/`, `score_sde/`, `ddim/` | ε-예측·v-예측·확률 흐름 ODE의 원 논문. `03` §1의 변환식(ε ↔ v ↔ velocity)이 가리키는 쪽 |
| `classifier_free_guidance/` | 이 논문이 쓰는 안내 방식(텍스트 인코더별 드롭아웃 46.4%)의 원 논문 |
| `../transfusion/` | 한 Transformer에서 텍스트 AR + 이미지 diffusion. 이미지 쪽 손실은 ε-예측·코사인, 텍스트-이미지 가중치 공유. SD3는 모달리티별 가중치를 쓰는 쪽이라 Transfusion의 "공유 가중치" 가설과 직접 대비됨 |
| `../flashattention/` | MM-DiT의 합친 어텐션 길이는 이미지 토큰 + 154(텍스트) 토큰, 고해상도에서 이미지 토큰이 지배 |

그 밖에 `streamingllm/`, `pagedattention/`, `speculative_decoding/`, `gptq/`, `awq/`는 LLM 추론 쪽이라 직접 관련이 없습니다(큰 모델일수록 적은 스텝에서 덜 열화한다는 표 6은 "스텝 수 vs 품질"로 speculative decoding의 "스텝 수 vs 정확도"와 비슷한 축의 이야기일 뿐입니다).


## 5. GitHub·자료 표 (조회 2026-11-28, `gh api`)

| 자료 | 라이선스 | 별 | 마지막 push | 용도 |
|---|---|---|---|---|
| github.com/Stability-AI/sd3.5 | MIT(코드), 가중치는 별도 | 1,545 | 2025-01-08 | 추론 전용 참조 구현. `repo/`에 얕게 clone(HEAD `8565799a3b41`), 정적 읽기만 |
| github.com/Stability-AI/generative-models | MIT | 27,349 | 2025-12-16 | SDXL·SVD 계열 저장소(88 MB, clone하지 않음) |
| github.com/huggingface/diffusers | Apache-2.0 | 34,985 | 2026-11-28 | `training_utils.py`, `train_dreambooth_lora_sd3.py`, `scheduling_flow_match_euler_discrete.py`를 `gh api`로 읽음(clone 없음, main `509b7658`) |
| proceedings.mlr.press/v235/esser24a.html | — | — | 2024-07-08 | 공식 판본 |
| 논문 학습 코드·데이터·가중치 | 해당 없음 | — | — | 논문이 공개를 약속했으나 학습 코드는 확인하지 못함 |
