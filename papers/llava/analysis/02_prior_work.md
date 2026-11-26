# 02. 계보와 관련 연구: 이후 논문이 무엇을 바꿨나

이 문서에서 다른 논문의 내용은 **arXiv 초록**(arXiv API, 2026-11-26)과 **이 저장소의 완료된 분석**(`prismatic_vlms/`, `cambrian1/` 등)에서 읽은 것만 적습니다. LLaVA-1.5는 PDF 본문(표 2, 3, 4)을 읽었습니다. 그 밖의 논문은 초록 수준이라 "초록에 따르면"으로 표시합니다. 웹 검색은 쓰지 않았습니다. 이번 묶음의 다른 논문은 폴더 이름으로만 겁니다.

## 1. 선행 연구(LLaVA가 가져온 것)

| 선행 | 가져온 것 | LLaVA에서의 자리 |
|---|---|---|
| CLIP (arXiv:2103.00020, `clip/`) | 이미지·텍스트 쌍으로 정렬된 시각 인코더 ViT-L/14 | 동결된 시각 타워 g (§4.1). 투영 대상은 "문장으로 가리킬 수 있는 특징"이라는 CLIP의 성질에 기댑니다 |
| ViT (arXiv:2010.11929, `vit/`) | 패치 격자 특징 | 마지막 층 앞의 **패치 토큰 전부**(CLS 제외)가 LM 입력이 됩니다 |
| Vicuna, Alpaca, GPT-4-LLM, Self-Instruct (§1, §2 인용) | 기계가 만든 지시 데이터로 LLM을 지시 튜닝 | 이 방법을 시각으로 옮긴 것이 시각 지시 튜닝입니다(§1 기여 1) |
| Flamingo (arXiv:2204.14198) | 초록에 따르면 사전학습된 비전·언어 모델을 잇고 이미지·텍스트가 섞인 열을 받음. 시각 토큰 수를 고정하는 Perceiver Resampler와 gated cross-attention으로 LM에 주입 | LLaVA는 정반대를 고릅니다. 토큰을 줄이지 않고 입력 임베딩 열에 이어 붙이며 LM 구조는 건드리지 않습니다(§2, §4.1) |
| BLIP-2 (arXiv:2301.12597, `blip2/`) | 동결 인코더와 동결 LLM 사이의 Q-Former(질의 32개) | 논문은 Q-Former를 "더 정교한 방식"으로 언급하고 미룹니다(§4.1). 표 5에서 BLIP-2(38.1)와 비교합니다 |
| LLaMA-Adapter, OpenFlamingo, PaLM-E (§2) | LLM에 이미지를 쓰게 하는 시도, 로봇용 LMM | 지시 데이터로 튜닝되지 않은 대조군 |

## 2. 동시기와 직후 (같은 아이디어를 다른 방식으로)

| 논문 | 요지(초록 기준) | LLaVA와의 관계 |
|---|---|---|
| **MiniGPT-4** (arXiv:2304.10592, 2023-04-20) | 초록: 동결 시각 인코더와 동결 Vicuna를 **투영 층 하나**로 정렬. 짧은 캡션으로만 학습하면 반복·파편화된 문장이 나와서 2단계에서 정제한 상세 설명 데이터로 미세조정 | LLaVA 공개 3일 뒤. 같은 "투영 하나 + 두 단계" 틀이지만 데이터가 GPT-4가 만든 지시가 아니라 정제된 상세 캡션입니다. 2단계에서 LLM 동결 여부도 다릅니다 |
| **InstructBLIP** (arXiv:2305.06500) | 초록: 공개 데이터 26개를 지시 형식으로 바꾸고 **지시를 읽는 Q-Former**를 도입. 13개 보류 데이터에서 BLIP-2를 앞섬. ScienceQA(이미지 맥락) 90.7% | 학술 VQA 데이터 혼합과 Q-Former 경로. LLaVA-1.5는 같은 데이터 일부를 쓰며 "병목은 구조가 아니라 데이터·형식 지시"라고 반박합니다(아래) |
| **ShareGPT4V** (arXiv:2311.12793) | 초록: GPT4-Vision의 고품질 캡션 100K를 1.2M으로 확장해, 정렬(사전학습)과 SFT에 넣으면 LLaVA-7B 등이 MME와 MMBench에서 오른다고 보고 | LLaVA의 1단계 데이터(CC-595K, 단순 캡션)가 병목이라는 쪽의 후속. 정렬 데이터의 질이 중요하다는 증거선 |

## 3. LLaVA를 직접 고친 후속 (이름과 무엇을 바꿨나)

| 후속 | 바꾼 것 | 근거와 판 |
|---|---|---|
| **LLaVA-1.5** (arXiv:2310.03744, CVPR 2024 하이라이트) | (1) 선형 투영 → **2층 MLP**(Linear-GELU-Linear), (2) CLIP-ViT-L **336px**(시각 토큰 576), (3) 학술 VQA 데이터(VQAv2, OKVQA, OCR 데이터, 영역 VQA, GQA)와 ShareGPT 텍스트, (4) 짧은 답용 **형식 프롬프트** "Answer the question using a single word or phrase.", (5) 13B | 본문 읽음. **표 2가 증분 절제**: 행 2(+형식 프롬프트) MME 1323.8, 행 3(+MLP 투영) 1355.2(+31.4), GQA 46.8 → 47.3, MM-Vet 26.3 → 27.8. 행 0의 원 LLaVA는 MME 809.6. 모든 행이 **단일 실행**입니다. 1단계는 같은 558K 정렬 데이터, 지시 데이터는 665K입니다 |
| LLaVA-1.5-HD | 이미지를 인코더 원래 해상도(336)의 격자로 쪼개 각각 인코딩, 합치고, 224로 줄인 전체 이미지 특징을 이어 붙여 전역 문맥 제공 | 본문 읽음(그림 2). 전역 문맥 절제 표: GQA +0.9, MME +71, MM-Vet +3.2 (7B, 단일 실행) |
| **LLaVA-NeXT**(블로그), **AnyRes** | 격자 타일링과 줄바꿈 토큰으로 가변 해상도 | 공식 저장소의 `anyres`, `mm_patch_merge_type`(`spatial_unpad`) 코드를 정적으로 확인(`03` §5). `cambrian1/`이 시각 토큰 2,880개로 인용 |
| **LLaVA-OneVision** (arXiv:2408.03326) | 초록: LLaVA-NeXT 블로그 시리즈의 데이터·모델·시각 표현 통찰을 모아 단일 이미지·다중 이미지·영상을 한 모델로 | 초록만 읽음 |
| **Video-LLaVA** 등 | 영상 토큰화 | 이 폴더 범위 밖 |

## 4. 같은 질문을 통제 실험으로 다룬 후속 (이 저장소의 분석으로 읽음)

| 논문 | LLaVA의 어떤 선택을 시험했나 | 결과(해당 분석에서) |
|---|---|---|
| **Prismatic VLMs** (arXiv:2402.07865, `prismatic_vlms/`) | LLaVA-1.5를 기준으로 한 번에 하나씩 바꿈: **단일 단계**(정렬 단계 생략), 시각 백본 동결 해제, 해상도, 융합 인코더 | 단일 단계는 평균 z +0.15로 **공식 대 재현 LLaVA-1.5 격차(±0.17)와 같은 크기**라 "유지"는 지지, "개선"은 약함(`prismatic_vlms/analysis/03` §2). 시각 백본을 동결 해제하면 평균 z −1.15로 **나빠진다**고 보고해 LLaVA의 "시각 인코더 동결"을 지지합니다(아래 Cambrian-1이 반박). 융합 투영기가 코드에서 3층이라 교란이 있습니다 |
| **Cambrian-1** (arXiv:2406.16860, `cambrian1/`) | 같은 레시피에서 어댑터(1단계) 데이터 양, 동결 해제, 인코더 23종, 연결기(SVA) | 시각 인코더 동결 해제는 13개 모델 평균 +1.20(11개 상승)으로 **Prismatic과 반대 방향**(`cambrian1/analysis/03` §3.5). 어댑터 데이터 0 → 0.5M은 16개 인코더 평균 +3.22(14개 상승)이지만 **OpenAI CLIP-336에서는 +0.66**(표본 오차 수준)이라 Prismatic 주장은 그 설정에서는 반박되지 않습니다. 0.5M → 1.2M은 +1.56(13개 상승). 인코더마다 정렬 단계의 필요가 다릅니다(`cambrian1/analysis/03` §3.4) |
| **MM1** (arXiv:2403.09611) | 초록: 이미지 인코더, 연결기, 사전학습 데이터 혼합 절제. **이미지 인코더·해상도·이미지 토큰 수의 영향이 크고 연결기 설계는 상대적으로 무시할 만하다** | LLaVA의 "연결 방식" 선택(선형 대 MLP 대 Q-Former)이 성능에 주는 몫이 작다는 방향. 사전학습 규모가 달라 직접 비교는 못 합니다(초록만 읽음) |
| **Eyes Wide Shut?** (arXiv:2401.06209, `eyes_wide_shut/`) | CLIP-blind 쌍으로 MLLM의 시각 실패를 측정 | LLaVA의 "bag of patches" 한계 문장(§5)을 설계 쪽에서 설명하는 후속 증거선. 인코더 특징이 병목입니다 |
| **POPE** (arXiv:2305.10355, EMNLP 2023) | 초록: LVLM의 물체 환각을 폴링 기반 질의(예/아니오)로 측정. 시각 지시에 자주 나오거나 이미지 물체와 함께 나오는 물체가 환각되기 쉽다 | LLaVA-Bench는 환각을 직접 재지 않습니다(부록 A도 후속 과제로 적음). LLaVA-1.5 표 4에서 POPE(random/popular/adversarial)가 원 LLaVA-7B 76.3/72.2/70.1 → LLaVA-1.5-13B 87.1/86.2/84.5로 보고됩니다(본문 읽음) |

## 5. 로봇·VLA로 간 갈래

- **VLA의 백본이 이 레시피를 이어받습니다.** OpenVLA(arXiv:2406.09246)는 Prismatic-7B(DINOv2 + SigLIP 패치 특징 + 2층 MLP 투영 + Llama 2)를 백본으로 쓰고, Prismatic은 LLaVA-1.5 레시피(패치 토큰 전부를 MLP로 투영해 앞에 붙임)를 재현해 설계를 비교한 것입니다. 즉 "모든 패치를 그대로 넘기는" 연결 방식이 LLaVA → LLaVA-1.5 → Prismatic → OpenVLA로 전해졌습니다(`prismatic_vlms/analysis/02`).
- PaliGemma(`paligemma/`)도 SigLIP-So400m과 Gemma를 **선형층**으로 잇습니다(그 폴더의 요약). 선형 대 MLP는 백본이 달라 한 논문에서 쟀지 않았습니다.
- 대조되는 연결: Q-Former(BLIP-2, 32토큰), Perceiver resampler(Flamingo), 공간 인식 집계(Cambrian-1의 SVA), 동적 해상도(Qwen2-VL, `qwen2_vl/`), 연속 패치 early fusion(Transfusion, `transfusion/`).

## 6. 구성 요소별 출처

| LLaVA의 요소 | 출처 | 이후 변화 |
|---|---|---|
| 시각 타워 CLIP ViT-L/14, 끝에서 두 번째 층 | CLIP, ViT. 층 선택은 LLaVA의 경험적 선택(§5.2 가설) | Prismatic도 "끝에서 두 번째 층"을 쓰고(`prismatic_vlms/analysis/01`), LLaVA-1.5도 같음. 독립 절제는 없음(`04` C6) |
| 패치 토큰 전부 + 선형 투영 | 새로움(ViT 패치를 단어 임베딩 공간에 사상) | MLP(LLaVA-1.5), 다중 인코더 융합(Prismatic), 쿼리 집계(Cambrian), 압축 |
| 시각 토큰을 텍스트 앞에 이어 붙임 | Prefix 방식 | 거의 모든 후속이 유지. PaliGemma는 접두에 양방향 어텐션 |
| 답 토큰에만 손실, `<STOP>`도 예측 | Vicuna·Alpaca 지시 튜닝의 관행 | 그대로 |
| 1단계(투영만) + 2단계(투영 + LM) | MiniGPT-4와 동시기. 논문은 "compatible visual tokenizer" 비유 | Prismatic(생략 가능), Cambrian(인코더 의존) |
| GPT-4로 만든 시각 지시 데이터 158K | Self-Instruct와 GPT-4-LLM의 시각판 | 학술 VQA 혼합(LLaVA-1.5), 고품질 캡션(ShareGPT4V), 데이터 엔진(Cambrian) |
| GPT-4 판정 상대 점수 평가 | Vicuna의 평가 프로토콜 | MT-Bench(`llm_as_judge/`)가 판정 편향을 분석. POPE, MMBench 등 객관식·이진 평가가 늘어남 |

## 7. 자료 표

| 자료 | 라이선스 | 크기·상태 | 비고 |
|---|---|---|---|
| github.com/haotian-liu/LLaVA | Apache-2.0 | 별 25,092, 포크 2,774, 마지막 push 2024-08-12, 저장소 14 MB(얕은 clone 33 MB), 읽은 커밋 `c121f04`(2024-05-12) | 정적으로만 읽음. 이슈 1,140건 열림, 보관(archive)되지는 않음 |
| HF `liuhaotian/LLaVA-Instruct-150K` | cc-by-4.0(HF 메타데이터) | 받지 않음. `docs/Data.md`에 150K 229 MB, 80K 229 MB 표기 | 데이터는 OpenAI 약관을 따르라는 고지가 README에 있음 |
| HF `liuhaotian/LLaVA-Pretrain` | other | 받지 않음 | 558K(LCS) 정렬 데이터 |
| HF `liuhaotian/llava-bench-in-the-wild` | apache-2.0 | 질문 8 KB, 맥락 8 KB만 받음(이미지 24장은 받지 않음) | 60문항 확인 |
| 모델 가중치 | 기반 LM 라이선스(Llama, Llama-2 커뮤니티) | 받지 않음 | README의 고지 |
| github.com/LLaVA-VL/LLaVA-NeXT | Apache-2.0 | 별 4,734, 마지막 push 2026-06-15 | 후속 코드. 읽지 않음 |
| github.com/TRI-ML/prismatic-vlms | MIT | 별 1,026, 마지막 push 2024-07-04 | `prismatic_vlms/`에서 읽음 |

## 8. 이 저장소의 다른 분석과의 관계

- **`blip2/`**: 같은 질문(시각 토큰을 LM에 어떻게 넘기나)의 정반대 답입니다. LLaVA는 모든 패치(시각 토큰 256개 또는 576개)를 선형·MLP로 투영해 넘기고, BLIP-2는 Q-Former가 32개로 압축합니다. 이 폴더의 토이에 위치·개수 질문이 있는 것은 그쪽 토이와 같은 종류의 질문으로 비교하기 위해서입니다(`03` §6.4).
- **`prismatic_vlms/`, `cambrian1/`**: 1단계 필요성, 동결, 연결기 용량의 통제 실험(위 §4). 이 폴더의 04가 두 분석의 결론을 LLaVA의 주장에 연결합니다.
- **`clip/`, `vit/`, `siglip/`, `dinov2/`, `eyes_wide_shut/`**: 시각 타워가 무엇을 담는지. LLaVA의 층 선택 가설("마지막 층은 전역적")은 `clip/`의 CLS 풀링 목적과 이어집니다(`03` §6.3의 토이 진단).
- **`llm_as_judge/`(24번)**: LLaVA-Bench의 GPT-4 판정 구조(참조 위치 고정, 같은 계열 모델이 참조를 쓰고 판정)에 대한 편향 근거. `eval_error_bars/`, `little_power/`: 상대 점수와 ScienceQA 정확도 차이의 불확실성 계산(`03` §7, `04`).
- **`paligemma/`, `qwen2_vl/`, `transfusion/`**: 연결 방식의 다른 선택(선형 + 접두 양방향, 동적 해상도 토큰, 연속 early fusion).
