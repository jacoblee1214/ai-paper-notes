# 02. 선행 연구, 구성 요소의 출처, 관련 분석, 자료

이 문서에서 다른 논문의 내용은 arXiv 초록과 Cambrian-1 v2의 참고문헌·본문 문장에서 읽은 것만 적습니다(웹 검색 0회, arXiv API로 번호와 제목을 대조). 이번 묶음의 다른 논문은 폴더 이름으로만 겁니다. `prismatic_vlms/`, `paligemma/`, `qwen2_vl/`, `transfusion/`, `eyes_wide_shut/`는 만들어져 있고, `vit/`, `clip/`, `siglip/`, `dinov2/`, `blip2/`, `llava/`는 이번 묶음에서 만들어질 폴더입니다.

## 1. 계보: 같은 질문을 다룬 설계 비교

| 논문 | 무엇을 통제해 비교했나 | Cambrian-1과의 관계 |
|---|---|---|
| LLaVA (arXiv:2304.08485, NeurIPS 2023 구두), LLaVA-1.5 (arXiv:2310.03744, CVPR 2024 하이라이트) | 투영층(MLP)과 2단계 학습(연결기만 학습 후 연결기+LLM 학습, 시각 인코더 동결) 레시피 | Cambrian-1의 기본 설정이 이 레시피입니다(§3.3). 어댑터 데이터를 0.5M(LLaVA 558K)에서 1.2M(ShareGPT4V-PT)으로 늘려 "연결기 사전학습 필요" 주장을 재검증하고, 737K 혼합은 LLaVA-665K의 확장입니다 (부록 E.5) |
| LLaVA-NeXT (블로그, arXiv 없음) | AnyRes 타일링, 시각 토큰 2,880개(672px) | 최종 모델의 비교 기준선입니다(표 9). Cambrian은 576토큰으로 OCR·비전 중심에서 앞선다고 주장합니다 |
| Prismatic VLMs (arXiv:2402.07865, ICML 2024; `prismatic_vlms/`) | LLaVA-1.5 재현 설정에서 이미지 처리, 인코더, 결합, LM, 학습 단계, 동결을 한 번에 하나씩 변경, 12개 벤치마크 | **같은 질문의 직접 경쟁자.** Cambrian-1 §3.3이 Prismatic의 두 결론을 이름을 들어 반박합니다: 연결기 사전학습 생략 주장과 시각 인코더 미동결이 해롭다는 주장([64]). 대조는 `03` §8 |
| MM1 (arXiv:2403.09611) | 초록: 이미지 인코더, 연결기, 사전학습 데이터 혼합을 절제. "이미지 인코더, 해상도, 이미지 토큰 수는 큰 영향, 연결기 설계는 상대적으로 무시할 만하다" | 연결기 중요도 결론이 Cambrian-1의 SVA 주장과 긴장합니다. 다만 MM1은 사전학습 규모의 평가이고 Cambrian-1은 고해상도·다중 인코더 집계라 영역이 다릅니다(초록만 읽음, 본문 미확인). Cambrian-1 본문은 MM1([99])을 "시각 인코더를 동결할지" 논쟁의 한쪽으로(§2, §3.3) 인용하고 표 9의 비교 모델(MM-1-8B/30B)로 씁니다 |
| Idefics2, "What matters when building vision-language models?" (arXiv:2405.02246) | 초록: 사전학습 모델, 구조, 데이터, 학습 방법 전반의 실험으로 설계 결정을 정당화(초록만 읽음) | 같은 문제의식(근거 없는 설계 결정 비판)입니다. Cambrian-1은 [73]을 표 4의 Resampler 기준선 출처로만 인용하고([11, 73]) 설계 결론을 비교하지 않습니다 |
| Eyes Wide Shut? (arXiv:2401.06209, CVPR 2024 구두; `eyes_wide_shut/`) | CLIP이 비슷하게 보는 이미지 쌍(CLIP-blind pairs)으로 MMVP를 만들고, DINOv2 특징을 섞는 A-MoF와 I-MoF를 제안 | **같은 저자군의 앞선 작업**(Tong, LeCun, Xie 외). Cambrian-1의 MMVP 열, 비전 중심 군집, 인코더 결합(§3.5의 "A-MoF [127] 방식")의 출처입니다 |
| Mini-Gemini (arXiv:2403.18814) | 초록: 시각 토큰 수를 늘리지 않고 추가 고해상도 인코더로 정제 | SVA와 같은 문제(고해상도 특징을 적은 토큰에 넣기)를 다룹니다. 표 9의 비교 기준선이고 ConvNeXt 다단계 특징 연결 방식도 이 논문 방식을 따랐다고 부록 F가 적습니다([78]) |
| Honeybee C-Abstractor (arXiv:2312.06742), MobileVLM V2 LDPv2 (arXiv:2402.03766) | 지역성을 보존하는 압축 연결기 | v2 표 5의 36토큰 압축 비교 대상입니다. SVA가 General +1.1–3.0, OCR +2.7–5.3, 비전 중심 +2.6–3.2 앞섭니다(`03` §6) |
| BLIP-2 Q-Former(원형), Flamingo Perceiver Resampler | 학습 가능한 질의로 시각 토큰을 고정 개수로 압축 | SVA의 기반 아이디어(질의 + 교차 어텐션. 본문은 Resampler [6], Q-Former 계열 Qwen-VL [11]·InstructBLIP [37]을 인용)이고 표 4의 Resampler 기준선입니다([11, 73]). SVA는 질의마다 부분 영역을 고정해 공간 구조를 보존하는 점이 다릅니다 |

## 2. 이미지 없이 풀리는 벤치마크 비판

| 논문 | 내용 | 관계 |
|---|---|---|
| Making the V in VQA Matter (arXiv:1612.00837) | 언어 prior를 가려 VQA의 이미지 이해 역할을 높임 (Cambrian-1 참고문헌 [51]) | 문제의 원형. Cambrian-1은 "이미지 있음/없음" 비교로 같은 질문을 MLLM 벤치마크에 던집니다 |
| MMStar, "Are We on the Right Way for Evaluating Large Vision-Language Models?" (arXiv:2403.20330) | 초록: 많은 문항에서 시각 내용이 불필요하고(예: GeminiPro가 MMMU에서 이미지 없이 42.9%, 여섯 벤치마크에서 무작위보다 평균 24% 이상) 학습 데이터 누출도 있다고 보고, 시각 필수 문항만 모은 벤치마크를 제안 | **3개월 앞선 같은 결론.** Cambrian-1은 [26]으로 인용하되 본문에서 비교하지 않습니다. Cambrian-1의 차별점은 23개 인코더 모델에 대한 평균과 군집화, 우연 수준과의 3자 비교입니다 |
| MMMU-Pro (arXiv:2409.02813) | 초록: 텍스트 모델이 풀 수 있는 문항을 걸러내고 선택지를 늘림 | Cambrian-1 이후의 같은 방향 처방입니다. Cambrian-1이 이미지 없이 풀린다고 지목한 MMMU(5점 미만 차이)를 직접 고친 후속입니다 |
| Vision language models are blind (arXiv:2407.06581) | 초록: 단순한 저수준 시각 과제 7개(BlindTest)에서 VLM이 평균 58.07% | 비전 중심 평가가 필요하다는 같은 방향의 증거입니다. CV-Bench는 반대로 이미지 정답이 고전 비전 주석에서 오는 대규모 객관식입니다 |
| V*Bench (arXiv:2312.14135), BLINK (arXiv:2404.12390, ECCV 2024) | 고해상도·세부 시각 검색, 사람에게 쉬운 인지 과제 모음 | 비전 중심 평가의 이웃입니다. Cambrian-1의 평가 스위트가 두 벤치마크를 지원합니다(`eval/`) |

## 3. 후속

- **Cambrian-S** (arXiv:2511.04670, 2025-11-06, 같은 프로젝트 사이트): 초록에 "spatial supersensing" 개념과 VSI-SUPER 벤치마크(VSR 장기 시각 공간 회상, VSC 연속 시각 공간 계수)를 제안한다고 적혀 있습니다(초록만 읽음). Cambrian-1이 이미지 한 장의 비전 중심 평가를 다뤘다면 이쪽은 영상으로 옮겼습니다. 이 분석은 Cambrian-S를 읽지 않았습니다.
- MMMU-Pro, MMStar 같은 필터링 벤치마크가 Cambrian-1이 비판한 점을 벤치마크 쪽에서 해결하는 후속 흐름입니다.

## 4. 구성 요소별 출처

| 요소 | 출처 | 비고 |
|---|---|---|
| 시각 인코더 | CLIP (`clip/`), SigLIP (`siglip/`), DFN-CLIP, EVA-CLIP, OpenCLIP, DINOv2 (`dinov2/`), I-JEPA, MAE, MoCo v3, SAM, MiDaS, Stable Diffusion, ViT 지도 (`vit/`) | 표 10의 23개 백본. Cambrian-1 자신은 인코더를 만들지 않고 비교합니다 |
| 2단계 학습 | LLaVA (`llava/`) | 어댑터 → 지시 튜닝 |
| 인코더 결합 | Eyes Wide Shut의 A-MoF | 576토큰으로 보간 후 채널 결합 |
| SVA | 교차 어텐션 질의 (Resampler [6], Q-Former 계열 [11, 37], 원형은 BLIP-2) + 공간 부분 영역 고정(C/D-Abstractor [21], LDPv2 [35]와 대비) | 식 (1) |
| 데이터 | LLaVA-665K, ShareGPT4V, ALLaVA, DVQA, DocVQA 등(그림 9, 표 17) + 자체 데이터 엔진 161K | 데이터 균형은 CLIP·MetaCLIP의 출처별 상한 아이디어를 따름([110, 139]) |
| 평가 | MMBench, SEED-Bench, MME, GQA, ScienceQA, MMMU, MathVista, AI2D, ChartQA, OCRBench, TextVQA, DocVQA, MMVP, RealWorldQA(xAI), V*Bench, CV-Bench(자체) | 표 26이 프롬프트를 나열 |

## 5. 이 저장소의 관련 분석과의 관계

- **`prismatic_vlms/`:** 같은 질문(동결, 정렬 단계, 인코더 결합)의 직접 경쟁 논문입니다. 결론이 갈리는 곳(동결 해제, 단일 단계)과 설정 차이는 `03` §8의 표입니다. Prismatic 분석에서 쓴 MDE·Wilson 방법을 이 분석도 그대로 따랐습니다(같은 q=0.2).
- **`eyes_wide_shut/`:** MMVP와 A-MoF의 출처입니다. Cambrian-1은 이 벤치마크에서 이미지 없음 점수가 정확히 0인 점(쌍 지표의 필연, `03` §1.3)을 "시각 근거화가 중요하다"의 증거로 썼는데, 그 해석의 검토는 `eyes_wide_shut/` 분석과 맞대어 읽어야 합니다.
- **`paligemma/`, `qwen2_vl/`:** 다른 설계(PaliGemma의 SigLIP 단일 인코더와 prefix-LM, Qwen2-VL의 동적 해상도)입니다. Cambrian-1의 576토큰 고정 격자와 대비되는 해상도 전략의 한 쌍이고, 해상도·토큰 수에 대한 Cambrian-1의 결론("고해상도가 차트·비전 중심을 올린다")은 `03` §3.3에서 표의 범위로 제한해 읽습니다.
- **`transfusion/`:** 연속 패치 early fusion으로, Cambrian-1이 다루는 late fusion(인코더 → 연결기 → LM)과 정반대 결합 시점입니다. 직접 비교는 없습니다.
- **`construct_validity/`:** Cambrian-1의 §3.1("이 벤치마크가 정말 비전을 재는가")은 구성 타당도의 과제 타당도 문제입니다. 이미지 없음 비교와 CV-Bench 선택지 단서(`03` §1, §5)는 `construct_validity/`의 점검 항목(현상 정의, 과제가 현상을 재는지)을 MLLM에 적용한 예로 읽을 수 있습니다. Cambrian-1은 정의를 명시하지 않고 군집(General, Knowledge, OCR & Chart, Vision-Centric)을 사후 라벨로 붙입니다.
- **`eval_error_bars/`:** 짝지은 차이와 군집 표준오차의 틀입니다. 이 분석의 범주 평균 MDE는 벤치마크 표본이 독립이고 문항이 쌍체라는 가정이며 군집 보정은 하지 않았습니다. CV-Bench 3D는 이미지 한 장에 문항 1.4–2.7개(`03` §5)라 군집 보정이 필요합니다.
- **`little_power/`:** MDE 2.80·√(q/n)의 출처 방법(쌍 McNemar 검정력)입니다. 불일치율 q는 공개되지 않아 0.2를 대표값으로 씁니다(Prismatic 분석과 같음).

## 6. 공식 자료 표

| 자원 | 위치 | 라이선스 | 크기·상태 | 우리의 사용 |
|---|---|---|---|---|
| 코드 | github.com/cambrian-mllm/cambrian | Apache-2.0 | 별 2,011, 생성 2024-06-17, 최신 커밋 `539ffc3`(2024-10-30), GitHub API size 17.1 MB, 얕은 복제 작업 트리 4.9 MB | 정적 읽기만 (`repo/`, gitignore) |
| 모델 (8B, 13B, 34B) | huggingface.co/nyu-visionx/cambrian-{8b,13b,34b} | Apache-2.0 | 8B 마지막 수정 2024-06-28 | **가중치 받지 않음.** `config.json` 세 개(각 약 1 KB)만 읽어 `04` D1, D2 확인 |
| 학습 데이터 | huggingface.co/datasets/nyu-visionx/Cambrian-10M | Apache-2.0 표기 | HF 저장소 사용 용량 약 980 GB | 받지 않음 |
| CV-Bench | huggingface.co/datasets/nyu-visionx/CV-Bench | Apache-2.0 표기 (원 이미지는 ADE20K, COCO, Omni3D 라이선스를 따를 것으로 보이나 확인하지 않음) | 2,638행, 이미지 포함 parquet 185 MB + 220 MB | **질문·답 주석 jsonl 두 개(0.8 MB + 1.1 MB)만 `data/`에 받음**(gitignore). 이미지는 받지 않음 |
| 평가 데이터(ADE, COCO) | HF `SaiCharithaAkula21/benchmark_ade_manual`(633행), `benchmark_coco_filtered`(805행) | 표기 없음 | 개인 계정 저장소 | 행 수만 datasets-server API로 확인 (`04` D7) |
| 프로젝트 사이트 | cambrian-mllm.github.io | - | - | 열지 않음 |
| 학회 | neurips.cc/virtual/2024/oral/97972, proceedings.neurips.cc (NeurIPS 37) | - | 2026-11-26 확인 | `01` §0 |
