# 02. 계보와 관련 연구: DINOv2

arXiv 번호, 제목, 초록, 학회 코멘트는 arXiv API로 확인했고 이후 문헌 설명은 **초록 수준**입니다. 본문을 읽은 것은 DINOv2 논문, 공식 코드, 이 저장소의 다른 분석(`vit/`, `prismatic_vlms/`, `cambrian1/`, `eyes_wide_shut/`)에 정리된 내용뿐입니다. 이번 묶음의 다른 논문은 폴더 이름으로만 겁니다.

## 1. 구성 요소별 출처

| DINOv2의 요소 | 출처 | 논문이 바꾼 것 |
|---|---|---|
| 학생-교사 자기 증류, 교사 출력 중심화와 날카롭게 하기, 멀티 크롭 | DINO (Caron et al., arXiv:2104.14294, ICCV 2021) | 전역 크롭 2개 + 지역 크롭, [CLS]끼리의 교차 엔트로피(그대로). 중심화를 SK로 대체, 프로토타입 수를 128k로 |
| 가린 패치 단위 손실, 교사를 온라인 토크나이저로 | iBOT (Zhou et al., arXiv:2111.07832) | 머리를 묶지 않음(untying). iBOT은 공유가 낫다고 했고 DINOv2는 규모가 커지면 반대라고 서술(§4) |
| EMA 교사 | MoCo (He et al., arXiv:1911.05722, CVPR 2020), BYOL (Grill et al., arXiv:2006.07733) | 모멘텀 0.994 → 1 코사인(표 16). DINO·BYOL의 기본값은 0.996 → 1로 알고 있으나 우리는 두 원문을 읽지 않음 |
| Sinkhorn-Knopp 중심화 | SwAV (Caron et al., arXiv:2006.09882, NeurIPS 2020): 온라인 군집 배정에 균등 분할 제약 | 교사 softmax-centering을 SK로 교체(Ruan et al., arXiv:2211.09981, ICLR 2023의 권고, §4). 3회 반복 |
| KoLeo 정규화 | Sablayrolles et al., "Spreading vectors for similarity search" (arXiv:1806.03198, ICLR 2019): Kozachenko-Leonenko 엔트로피 추정량에서 온 균일 퍼짐 정규화 | 가중치 0.1, 전역 크롭 [CLS], 기기 안의 표본끼리 |
| 마스크 이미지 모델링 계열 | MAE (He et al., arXiv:2111.06377): 픽셀 재구성, 미세조정이 필요한 특징 | 픽셀 대신 프로토타입 분포를 예측(iBOT 방식)해 동결 특징으로도 쓸 수 있게 |
| ViT 백본 | ViT (arXiv:2010.11929, `vit/`) | 패치 14, LayerScale, SwiGLU(처음부터 학습), ViT-g 임베딩 1536 |
| 학습 후반 고해상도 | FixRes (Touvron 2019), FlexiViT 등을 논문이 인용 | 518에서 1만 스텝(§4, B.2) |
| 검색 기반 데이터 정제 | 복사 탐지(Pizzi 2022), CCNet식 NLP 정제(Wenzek 2020)에서 영감 | 외부 인코더·메타데이터 없이 자기지도 임베딩의 유사도만 사용 |
| 증류 | Hinton 2015, Duval 2023 | 손실을 바꾸지 않고 교사를 동결 ViT-g로, 학생의 EMA를 평가 |

## 2. 가까운 대안

| 대안 | 무엇이 다른가 | DINOv2와의 관계 |
|---|---|---|
| CLIP (`clip/`), OpenCLIP, EVA-CLIP | 이미지-텍스트 대조 | 논문의 주 비교 대상(표 4–11). 데이터와 목적이 함께 다름 |
| SigLIP (`siglip/`) | 시그모이드 대조 손실 | 논문은 비교하지 않음(2023-04). Prismatic·OpenVLA·Cambrian-1이 SigLIP과 DINOv2를 함께 씀 |
| MAE | 픽셀 재구성 | 표 4, 10, 11에서 MAE가 가장 낮은 편. 동결 특징으로는 약함 |
| I-JEPA | 잠재 공간 예측(마스크) | Cambrian-1 `cambrian1/`의 비교 대상. 이 논문은 인용만 함 |
| MSN, EsViT, Mugs | 프로토타입·계층 변형 | 표 4의 자기지도 기준선 |

## 3. 이후 문헌이 바꾼 것

| 이후 연구 | 바꾼 것 | 근거 수준 |
|---|---|---|
| **Vision Transformers Need Registers** (Darcet, Oquab, Mairal, Bojanowski, arXiv:2309.16588, ICLR 2024) | DINOv2 ViT에서 배경 패치에 **고노름 토큰**이 생겨 밀집 특징·attention 지도가 지저분해짐을 보이고, 입력에 register 토큰을 더해 해결. 공식 코드에 `num_register_tokens`와 "with registers" 모델이 있음(우리가 코드 확인). 저자 4명(arXiv API)이 모두 DINOv2 저자 | 초록, 코드, `vit/` 04 C7의 정리 |
| **Vision Transformers Don't Need Trained Registers** (arXiv:2506.08010, NeurIPS 2025 스포트라이트) | CLIP, DINOv2에서 소수 "register 뉴런"이 고노름 활성을 이상치 토큰에 몰아줌을 찾고 재학습 없이 같은 효과 | 초록 |
| **DINOv3** (arXiv:2508.10104, 기술 보고서) | 데이터·모델 규모를 더 키우고, 긴 학습에서 밀집 특징이 나빠지는 문제를 Gram anchoring으로 해결한다고 서술. 해상도·모델 크기·텍스트 정렬을 위한 사후 단계 | 초록만 읽음. 세부 수치와 DINOv2 대비 표는 읽지 않음 |
| **Scaling Language-Free Visual Representation Learning** (Web-SSL, arXiv:2504.01017) | CLIP과 시각 SSL을 **같은 MetaCLIP 데이터**로 학습해 통제한 비교. SSL이 데이터·모델에 더 잘 스케일하고 7B까지 포화하지 않으며 VQA에서 CLIP 수준에 도달한다고 보고 | 초록만 읽음 |
| **DINOv2 Meets Text (dino.txt)** (arXiv:2412.16334) | 동결 DINOv2에 텍스트 인코더를 정렬(LiT 방식)해 제로샷 분류·개방 어휘 분할. [CLS]와 패치 평균을 이어 정렬 | 초록만 읽음 |
| **Perception Encoder** (arXiv:2504.13181, NeurIPS 2025 구두) | 대조 이미지-언어 학습만으로 분류·검색·밀집 과제까지 최고 수준의 임베딩이 나오며, 그 임베딩이 **마지막 층이 아니라 중간층**에 있다고 보고. DINOv2식 SSL이 밀집 과제에 필수라는 가정에 대한 반례 후보 | 초록만 읽음 |
| **Prismatic VLMs** (`prismatic_vlms/`) | VLM에서 DINOv2를 SigLIP과 채널 방향으로 결합. SigLIP과는 이득(위치 지정 +11.6점), CLIP과는 이득 없음 | 이 저장소 분석(03) |
| **Cambrian-1** (`cambrian1/`) | 언어 지도 인코더가 자기지도(DINOv2 포함)보다 VLM 벤치마크 16개 모두에서 낫다. 통제 쌍에서 DINOv2 추가 효과 −0.01 | 이 저장소 분석(03) |
| **Eyes Wide Shut?** (`eyes_wide_shut/`) | DINOv2를 "CLIP이 못 구별하는 쌍"의 선별 기준과 보강 특징으로 씀. 이득이 DINOv2 특이적인지 SSL 일반인지 구분되지 않음 | 이 저장소 분석(03) |
| **DINO-WM** (arXiv:2411.04983), **OpenVLA** (arXiv:2406.09246) | DINOv2 패치 특징을 세계 모델의 관측 표현으로(DINO-WM, ViT-S/14 동결), SigLIP과 함께 VLA 백본으로(OpenVLA) | 공개본 밖의 분석을 읽음 |
| **Depth Anything** (arXiv:2401.10891, CVPR 2024) | 초록은 "사전학습 인코더의 의미 사전을 보조 지도로 물려받게 한다"고 쓰며 DINOv2를 이름으로 적지는 않음. 본문은 읽지 않았으므로 DINOv2 기반 여부는 확인하지 않음 | 초록만 읽음 |

## 4. 이 저장소의 다른 분석과의 관계

- `vit/`: 백본의 원전. 이 폴더의 `minimal_impl.py` 백본은 그 구현을 복사해 줄인 것(출처 주석). `vit/` 04 C7이 registers를 이미 정리했습니다.
- `clip/`, `siglip/`: 텍스트 지도 쪽 비교 대상. CLIP의 동결 특징과 선형 프로빙을 이 논문이 직접 비교합니다.
- `prismatic_vlms/`, `cambrian1/`, `eyes_wide_shut/`: VLM에서 DINOv2가 어떻게 쓰이고 평가되는지의 3점 자료. `04` C1, C11에서 인용합니다.
- `attention_sink_emerges/`, `spike_sparse_sink/`: LLM의 첫 토큰 attention sink와 massive activation. register 문제와 현상이 비슷하지만 **같은 메커니즘이라는 근거는 없습니다**(`04` C7).

## 5. 자료 표 (2026-11-26, `gh api`)

| 저장소 | 라이선스(GitHub) | 별 | 마지막 push | 비고 |
|---|---|---|---|---|
| facebookresearch/dinov2 | Apache-2.0 | 13,571 | 2026-11-05 | 크기 약 2.9 MB. README는 "code and model weights are released under the Apache License 2.0"(일부 확장 모델은 비상업 라이선스) |
| facebookresearch/dino | Apache-2.0 | 7,605 | 2024-07-03 | DINO 원 코드 |
| bytedance/ibot | Apache-2.0 | 777 | 2022-04-14 | iBOT |
| facebookresearch/swav | NOASSERTION | 2,091 | 2023-04-13 | SwAV(SK 원본) |
| facebookresearch/mae | NOASSERTION | 8,362 | 2024-07-23 | MAE |
| facebookresearch/dinov3 | 별도(Other) | 11,809 | 2026-11-02 | 우리는 읽지 않음 |

학회 표기 중 DINO(ICCV 2021), MoCo(CVPR 2020), SwAV(NeurIPS 2020)는 arXiv 코멘트에 적힌 것이고 공식 목록으로 확인하지 않았습니다. Registers의 ICLR 2024는 `vit/` 분석이 논문 본문의 "Published as a conference paper at ICLR 2024" 표기를 읽은 것입니다.
