# 02. 계보: ViT 이전과 이후

ViT의 기여는 새 부품이 아니라 **"패치 + 표준 Transformer + 큰 데이터"의 조합이 통한다는 증명**입니다. 부품은 대부분 이미 있었고, 이후 논문들은 (1) 데이터를 덜 쓰게 하거나 (2) 계층·국소성을 되돌리거나 (3) 자기지도·언어 감독으로 사전학습을 바꾸거나 (4) 토큰화(패치 크기, 해상도)를 유연하게 하거나 (5) 학습된 ViT의 내부 문제를 고쳤습니다. 이후 논문이 무엇을 바꿨는지는 아래 표에 정리했고, 주장별 판정은 `04`에 있습니다.

## 1. 앞선 시도 (구성 요소별 출처)

| 구성 요소 | ViT 이전의 출처 | ViT가 바꾼 것 |
|---|---|---|
| Transformer 인코더(pre-norm, MSA, GELU MLP) | Vaswani et al. 2017, BERT(Devlin 2019), pre-norm(Wang 2019; Baevski & Auli 2019) | 그대로. BERT-B/L 크기를 따르고 H를 추가(§4.1, 표 1) |
| [class] 토큰 | BERT의 [CLS] | 그대로. 부록 D.3에서 GAP도 같다고 확인 |
| 이미지에 attention | Non-local(Wang 2018), DETR(Carion 2020), 국소 attention(Parmar 2018; Hu 2019; Ramachandran 2019; Zhao 2020), Sparse Transformer(Child 2019), 축 방향(Ho 2019; Wang 2020a) | 특수 패턴을 버리고 **전역 dense attention**을 패치 열 위에 둠. 비용은 패치 단위로 시퀀스를 줄여 해결 |
| **패치 단위 토큰** | **Cordonnier et al. 2020**(arXiv:1911.03584, ICLR 2020)은 2×2 패치에 전체 self-attention을 얹고, 다층 self-attention이 합성곱을 표현할 수 있음을 보임. 논문 §2는 "ViT와 매우 비슷하다"고 인정 | 패치 16×16으로 키우고 중해상도 이미지까지 다루며 **대규모 사전학습으로 CNN과 겨룸**. 차이는 규모와 증명된 결과 |
| 픽셀 시퀀스 생성 사전학습 | iGPT(Chen et al. 2020a) | 픽셀 대신 패치. iGPT 선형 프로브 최대 72%(논문 §2) 대비 지도 사전학습으로 88.55% |
| 대규모 이미지 사전학습 | BiT(Kolesnikov 2020), Noisy Student(Xie 2020), JFT(Sun 2017) | 같은 데이터에서 ResNet 대신 Transformer를 학습. 같은 Adam, weight decay 0.1 레시피를 ResNet에도 적용해 비교 |
| 해상도 높여 미세조정 | Touvron 2019, BiT | 위치 임베딩 2D 보간(§3.2) |

## 2. 이후: ViT 학습 레시피와 데이터 효율

| 논문 | 한 일 | ViT 원전의 어떤 주장을 바꿨나 |
|---|---|---|
| **DeiT** (Touvron 외, arXiv:2012.12877) | ImageNet-1k만으로, 증강·정규화·**증류 토큰**을 얹어 86M ViT가 83.1% top-1(초록). 단일 컴퓨터에서 3일 미만 | "ImageNet 규모에서는 ResNet보다 몇 %p 낮다"(S2) |
| **How to train your ViT?** (AugReg, Steiner 외, arXiv:2106.10270) | 데이터량, 증강, 규제, 모델 크기, 연산의 상호작용을 50,000개 이상의 모델로 연구. ImageNet-21k에서 증강과 연산을 늘리면 **JFT-300M으로 학습한 모델과 같거나 앞선다**(초록) | "큰 비공개 데이터가 필요하다"(S3) |
| **Better plain ViT baselines for ImageNet-1k** (Beyer 외, arXiv:2205.01580) | 원래 레시피를 몇 곳 고쳐(표준 증강, 2D sincos 위치, GAP 등) 90 에폭에 76% 이상, 300 에폭에 80%(초록) | "규제를 정교하게 해야 한다"는 통념과 [CLS], 학습 1D 위치 임베딩(S5, S6) |
| **ResNet strikes back** (Wightman 외, arXiv:2110.00476) | 같은 최신 레시피를 ResNet-50에 적용해 224 해상도에서 80.4%(초록) | 비교 기준선의 강도. S2, S4의 "ResNet이 약하다"는 암묵 가정 |
| **Scaling Vision Transformers** (Zhai 외, arXiv:2106.04560, CVPR 2022) | 오차율, 데이터, 연산의 관계를 위아래로 확장. 20억 파라미터 모델이 ImageNet 90.45%(초록) | S8의 "포화하지 않는다" |
| **Early Convolutions Help Transformers See Better** (Xiao 외, arXiv:2106.14881, NeurIPS 2021) | 16×16 stride 16 patchify 대신 3×3 stride-2 conv 몇 개를 쓰면 최적화 안정성이 크게 늘고 ImageNet-1k 정확도가 ~1–2% 오름(초록) | 선형 패치 사영(식 1)이 최적이 아닐 수 있음 |
| **Do Vision Transformers See Like CNNs?** (Raghu 외, arXiv:2108.08810) | ViT 표현이 층마다 더 균일하고, 낮은 층의 전역 attention과 skip connection이 핵심이라고 분석. 분류 방식(CLS 대 GAP)이 공간 정보 보존에 영향 | S7, S6 |

## 3. 이후: 구조를 바꾼 계열

| 논문 | 바꾼 것 |
|---|---|
| **Swin** (Liu 외, arXiv:2103.14030) | 이동 윈도 attention과 계층 구조로 선형 복잡도, 검출·분할의 백본이 됨. ViT가 말한 "특수 attention은 가속기에 나쁘다"를 구현으로 반박 |
| **ConvNeXt** (Liu 외, arXiv:2201.03545, CVPR 2022) | ViT의 설계(patchify stem, 큰 커널, LN, GELU 등)를 순수 합성곱에 옮겨 같은 성능. "Transformer라서 낫다"의 약화 |
| **NaViT** (Dehghani 외, arXiv:2307.06304) | 고정 해상도 리사이즈 대신 **원래 해상도와 종횡비**로 시퀀스 패킹(§3.2 보간 문제를 구조로 회피) |
| **FlexiViT** (Beyer 외, arXiv:2212.08013) | 학습 때 패치 크기를 무작위로 하면 한 가중치로 여러 패치 크기를 쓸 수 있음. "패치 크기 = 속도·정확도 손잡이"를 모델 하나로 |

## 4. 이후: 사전학습 목적을 바꾼 계열

| 논문 | 바꾼 것 |
|---|---|
| **BEiT** (Bao 외, arXiv:2106.08254) | 패치를 이산 시각 토큰으로 바꾸고 마스킹된 패치의 토큰을 예측(BERT식). ViT §4.6의 마스크 패치 예측의 후속 |
| **MAE** (He 외, arXiv:2111.06377) | 패치 75%를 가리고 보이는 패치만 인코더에 넣어 픽셀을 복원. ViT §4.6의 "자기지도는 지도에 4% 뒤짐"을 크게 개선 |
| **CLIP** (`../clip/`) | 이미지-텍스트 대조 사전학습의 ViT 백본. ViT는 시각 쪽 구조 |
| **SigLIP** (진행 순서 43), **DINOv2** (진행 순서 44) | 오늘 VLM·로봇 정책이 쓰는 비전 인코더. 구조는 ViT 그대로, 목적 함수와 데이터가 다름. 이 저장소의 분석은 폴더가 생기면 `../siglip/`, `../dinov2/`로 연결 |

## 5. 이후: ViT 내부 현상과 토큰 문제

| 논문 | 바꾼 것 |
|---|---|
| **Vision Transformers Need Registers** (Darcet 외, arXiv:2309.16588) | 큰 ViT(지도·자기지도)의 특징 지도에 **높은 노름의 토큰**이 낮은 정보량 배경 영역에 생겨 내부 계산용으로 쓰임. 입력에 추가 토큰(register)을 주면 사라지고 attention 지도가 부드러워짐. 전역 정보를 모을 자리가 [CLS] 하나뿐인 원전 설계와의 관계는 우리의 해석이지 논문이 증명한 것이 아님(`04` C7) |
| **Massive Activations in LLMs** (Sun 외, arXiv:2402.17762, COLM 2024) | 소수의 활성이 비정상적으로 크다는 LLM 현상. ViT에서도 확인했다고 초록에 적음 |
| **Vision Transformers Don't Need Trained Registers** (Jiang 외, arXiv:2506.08010, NeurIPS 2025 스포트라이트) | 고노름 활성을 만드는 소수 뉴런을 찾아 **재학습 없이** 고노름을 옮김(초록) |

## 6. 이 저장소의 다른 분석과의 관계

| 폴더 | 관계 |
|---|---|
| `../attention_sink_emerges/` | LLM의 첫 토큰 attention sink와 key bias로서의 해석. register 문제와 같은 계열(`04` §C7) |
| `../spike_sparse_sink/` | pre-norm Transformer의 massive activation(spike)과 sink의 구조적 연결. ViT는 pre-norm이라 같은 메커니즘 후보(`04` §C7) |
| `../prismatic_vlms/`, `../paligemma/`, `../qwen2_vl/` | 비전 인코더로 ViT 계열(SigLIP, DINOv2, 원 해상도 ViT)을 쓰는 VLM. ViT 패치 크기와 해상도가 토큰 수와 지연을 정함 |
| `../transfusion/` | 이미지를 패치 토큰으로 바꿔 한 Transformer에서 텍스트와 함께 처리. 패치 임베딩 아이디어의 생성 쪽 확장 |
| `../cambrian1/`, `../eyes_wide_shut/` | 비전 인코더 선택과 CLIP 인코더의 한계(작성 중이거나 다른 세션이 작업 중) |
| `../clip/` | 같은 묶음의 ViT 백본 사용 논문 |

## 7. 자료 표 (2026-11-26 `gh api`)

| 저장소 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|
| google-research/vision_transformer | Apache-2.0 | 12,785 | 2026-10-01 | 원전. JAX/Flax. 미세조정과 체크포인트만. 얕은 clone 5.4 MB, 커밋 `64801f1` |
| google-research/big_vision | Apache-2.0 | 3,580 | 2025-05-19 | 사전학습 코드(JFT 포함 설정), FlexiViT, Better plain ViT |
| facebookresearch/deit | Apache-2.0 | 4,346 | 2024-03-15 | DeiT |
| huggingface/pytorch-image-models (timm) | Apache-2.0 | 37,313 | 2026-11-21 | ViT의 사실상 표준 PyTorch 구현과 레시피 |
| microsoft/Swin-Transformer | MIT | 16,082 | 2024-07-24 | Swin |
| facebookresearch/mae | NOASSERTION | 8,362 | 2024-07-23 | MAE (라이선스 자동 판별 안 됨, 직접 확인 필요) |
| facebookresearch/dinov2 | Apache-2.0 | 13,571 | 2026-11-05 | DINOv2 (register 변형 포함) |

arXiv 번호와 초록은 arXiv API로 확인했습니다. 이후 문헌 표의 학회 표기 중 DeiT(ICML 2021), AugReg(TMLR 2022), Better plain ViT(기술 노트)는 기억에 의존하므로 인용 전에 공식 목록에서 확인해야 합니다(우리가 공식 목록으로 확인한 것은 CVPR 2022 Scaling ViT, NeurIPS 2021 Early Convolutions, COLM 2024 Massive Activations, NeurIPS 2025 Test-time registers처럼 arXiv 코멘트에 적힌 경우뿐입니다).
