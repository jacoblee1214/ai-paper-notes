# 02. 계보와 관련 자료: SigLIP (arXiv:2303.15343)

- **근거 수준.** 이후 문헌은 arXiv 초록·메타데이터(arXiv API, 2026-11-26)와 이 저장소의 다른 분석, 공식 코드(`google-research/big_vision`)·OpenCLIP 소스의 정적 읽기에서 가져왔습니다. 본문을 읽지 않은 곳은 "초록 수준"이라고 표시했습니다. 웹 검색은 1회 썼습니다(독립 비교 찾기).
- 이 저장소의 다른 분석은 폴더명으로만 가리킵니다: `clip/`, `paligemma/`, `prismatic_vlms/`, `eyes_wide_shut/`, `cambrian1/`, `qwen2_vl/`. `dinov2/`, `blip2/`, `llava/`, `vit/`도 같은 묶음에서 분석을 마쳤습니다.

## 1. 계보: SigLIP이 기대는 것

| 갈래 | 논문 | SigLIP이 가져온 것, 바꾼 것 |
|---|---|---|
| softmax 대조 이미지-텍스트 | CLIP(2103.00020, `clip/`), ALIGN(2102.05918, 초록 수준) | 두 방향 softmax InfoNCE를 **쌍별 sigmoid**로 교체. 배치 전체 정규화와 온도 상한(CLIP은 100)이 사라지고 편향 b가 생김. 이미지·텍스트 인코더 구조는 ViT + Transformer로 같은 계열 |
| 이미지 타워 고정 | LiT(Zhai 외 2022, 본 논문 참고문헌 [59]) | **SigLiT**: 같은 틀(고정한 이미지 임베딩을 미리 계산, 텍스트 타워만 학습)에 sigmoid 손실 적용. 같은 저자 중복(Zhai) |
| 쌍별 이진 손실 | Hadsell 외 2006(차원 축소, 본 논문 [19]에서 유일한 선행 sigmoid형 손실로 인용), 분류에서의 sigmoid(Beyer 외 2020 ReaL, Wightman 외 2021 ResNet strikes back) | 본 논문 §2가 직접 인용한 선행. 이미지-텍스트 대조에서 sigmoid를 처음 대규모로 시험했다는 것이 논문의 위치 |
| 큰 배치 메모리 | GradCache(2101.06983, 초록 수준: 대조 손실과 인코더의 역전파를 분리해 배치 차원의 메모리를 거의 일정하게) | 본 논문은 인용하지 않음. softmax에서도 큰 배치를 메모리 제약 안에서 쓰는 방법이 이미 있었다는 점에서 SigLIP의 "메모리 이득"은 손실 고유의 것이 아님(§2의 Inf-CLIP) |
| 연산 효율 | FLIP(토큰 가림), BASIC, LAION | 본 논문이 비교 대상으로 들었음(§2, L89–L97) |
| 안정화 | β₂ = 0.95(MAE, iGPT 참고문헌 [20], [9]) | 큰 배치의 기울기 스파이크 대응(§4.7) |
| 판 확장 | 크기·모양 최적화 ViT: SoViT/So400m(2305.13035, "Getting ViT in Shape", 초록 수준: 모양까지 포함한 스케일링 법칙으로 400M 모델이 두 배 큰 모델과 경쟁) | v4에서 추가된 표 3의 SO(400M). 같은 저자 겹침 |

한 줄 요약: **구조와 데이터 파이프라인은 CLIP·LiT 계열 그대로이고, 바뀐 것은 손실(두 번의 softmax 대신 쌍별 sigmoid + 학습되는 편향)과 그것이 가능하게 한 배치·메모리 설계**입니다.

## 2. 이후 논문이 무엇을 바꿨나

| 이후 연구 | 위치 | 확인 수준 | SigLIP 쪽 주장에 대한 의미 |
|---|---|---|---|
| **SigLIP 2**(2502.14786) | 후속 | 초록 수준 | 같은 sigmoid 이미지-텍스트 목적에 캡션 사전학습, 자기증류·마스크 예측, 온라인 데이터 큐레이션, 다국어와 해상도 변형, 크기 B·L·So400m·g를 합친 레시피. 손실 자체는 유지하되 **손실만으로는 위치·밀집 특징이 부족했음**을 인정하는 방향(초록: "localization and dense features" 개선) |
| **So400m / Getting ViT in Shape**(2305.13035) | 병행·후속 | 초록 수준 | 표 3의 SO400M은 sigmoid가 아니라 **모델 모양**이 만든 이득이 섞임. 표 3의 ImageNet 83.2%는 손실 효과가 아님 |
| **Inf-CLIP**(2410.17243) | 후속 | 초록 수준 | softmax 손실도 타일 단위 계산과 링 통신으로 |B|×|B| 행렬 없이 계산해 "CLIP-ViT-L/14를 배치 4M·12M으로 정확도 손실 없이" 학습한다고 보고. **메모리 O(|B|²) 제거는 sigmoid만의 장점이 아님**(`04` C5) |
| **Image Captioners Are Scalable Vision Learners Too**(2306.07915, `clip/`에서 인용) | 병행 | 초록 수준 | 대조(softmax든 sigmoid든) 대신 캡션만으로 경쟁적인 시각 인코더. 목적 함수 선택이 유일한 길이 아님 |
| **SigCLR**(2410.17427) | 후속 | 초록 수준 | 이미지 쪽 SimCLR에 로지스틱 손실. CIFAR-10·100, Tiny-IN에서 경쟁적이고 학습되는 편향의 중요성을 확인하지만 **온도는 고정해야 잘 된다**고 보고(SigLIP은 t도 학습). 비전 전용 설정이라 직접 재현은 아님 |
| **Analysis of Using Sigmoid Loss for Contrastive Learning**(2402.12613) | 후속 | 초록 수준 | sigmoid 손실의 최적 임베딩 기하를 이론으로 분석(double-Constant Embedding Model). `03` §1.4의 직교 임베딩 최적 분리와 같은 종류의 접근 |
| **On the Similarities of Embeddings in Contrastive Learning**(2506.09781) | 후속 | 초록 수준 | 미니배치가 작을수록 음성 쌍 유사도의 분산이 커져 성능이 떨어진다는 이론. softmax 쪽에서 "작은 배치가 나쁜 이유"를 설명하는 후보이고 SigLIP의 "작은 배치에서 sigmoid가 낫다"와 같은 방향의 설명을 줌(직접 비교는 안 함) |
| **Perception Encoder**(2504.13181) | 후속 | 초록 수준 | "단순한 시각-언어 대조 학습"만으로 강한 범용 임베딩이 나오나 **최고의 임베딩은 출력층이 아니라 중간층**에 있다는 주장. 어떤 대조 손실을 썼는지는 초록에 없어 읽지 않았고 비교하지 않음 |

## 3. SigLIP을 인코더로 쓴 계보 (이 저장소 안)

- **PaliGemma**(`paligemma/`): SigLIP So400m/14 + Gemma-2B. 사전학습에서 인코더를 동결하지 않고 함께 학습. Fuyu식 인코더 없음과의 비교(그림 8)에서 SigLIP 인코더가 표본 효율의 격차를 만든다고 해석하되 SigLIP이 40B 쌍을 미리 봤다는 교란이 있음(`paligemma/analysis/03_analysis.md`).
- **Prismatic VLMs**(`prismatic_vlms/`): 인코더 통제 비교에서 CLIP·SigLIP이 ImageNet-1k ViT·DINOv2보다 낫고 DINOv2 + SigLIP 융합이 SigLIP 단독보다 낫다(p = 0.00164). 단 SigLIP은 SO400M, CLIP은 ViT-L이라 파라미터와 데이터가 다른 비교이고, `Prism_13B_SigLIP` 설정에 CLIP이 쓰인 것으로 보이는 설정 오류(D4)가 있음.
- **Cambrian-1**(`cambrian1/`): 인코더 23종 비교에서 SigLIP SO400M 등을 평가하고 최종 구성에 SigLIP ViT-SO400M/14@384를 쓴다. SigLIP + ConvNeXt에 DINOv2를 더해도 평균 −0.01, DINOv2 쪽에 ConvNeXt를 더하면 +2.91.
- **Eyes Wide Shut**(`eyes_wide_shut/`): MMVP-VLM에서 SigLIP이 OpenAI CLIP보다 높다(+18.5, p = 0.001, 135쌍). CLIP에 불리하게 선별된 쌍이라 과대일 수 있음.
- **Qwen2-VL**(`qwen2_vl/`): 자체 ViT(동적 해상도)를 씀. 고정 해상도 CLIP/SigLIP 계열을 비판 대상으로.

## 4. 자료 표

| 자료 | 라이선스 | 별 | 마지막 push | 메모 |
|---|---|---|---|---|
| [google-research/big_vision](https://github.com/google-research/big_vision) | Apache-2.0 | 3.6k | 2025-05-19 | 커밋 `0127fb6`. SigLIP 트레이너(`trainers/proj/image_text/siglip.py`), 두 타워 모델(`models/proj/image_text/two_towers.py`), 데모 노트북. 저장소 6.4 MB |
| [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | 저장소 메타데이터상 NOASSERTION | 14.3k | 2026-11-21 | 커밋 `57e8d63`. `SigLipLoss`(dist_impl: bidir, shift, reduce, gather, chunk_size), `--siglip`일 때 t′ = ln 10, b = −10 |
| ICCV 2023 (CVF Open Access) | | | | pp. 11975–11986, 구두 발표는 arXiv 주석(ICCV'23 Oral) |
| 체크포인트(공식, 가중치) | big_vision 노트북의 `gs://big_vision/siglip/` | | | 이번에 받지 않음. 노트북 목록에 B/16(224–512), L/16(256·384), So400m/14(224·384), i18n 변형 |
