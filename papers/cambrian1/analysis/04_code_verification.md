# 04. 논문 ↔ 공식 코드 대조

**대상:** `cambrian-mllm/cambrian` 커밋 `539ffc3254bba004e5d012b65c0ad6cb308897c5`(2024-10-30 병합, Apache-2.0). `repo/cambrian/`에 얕게 복제(작업 트리 4.9 MB)해 **정적으로만 읽었고 실행하지 않았습니다.** 모델 가중치는 받지 않았고 HF `config.json` 세 개(각 약 1 KB)와 CV-Bench 질문·답 주석 jsonl 두 개(합 1.9 MB)만 받았습니다. 논문 판본은 arXiv v2입니다.

**저장소 구성.** `cambrian/model/`(모델: `cambrian_arch.py` 653줄, `vision_sampler.py`(SVA), `multimodal_encoder/` 인코더 관련 파이썬 파일 22개, `multimodal_projector/`), `cambrian/train/`(TPU TorchXLA 학습 `train_tpu.py`, `train_fsdp.py`, 트레이너), `scripts/cambrian/`(최종 모델 6개: 8B/13B/34B 연결기 사전학습과 지시 튜닝), `eval/`(26개 벤치마크 평가·채점 스크립트, 2024-09-09 공개), `dataengine/`(Wikipedia 데이터 엔진), `docs/TPUs_Torch_XLA.md`. 학습은 TPU 전제이고 README는 "GPU training code will be released soon"이라고 씁니다.

## 1. 일치하는 것

| 항목 | 논문 | 코드 | 일치 |
|---|---|---|---|
| 식 (1)의 단일 softmax (모든 인코더 키를 이어 한 번에) | 식 (1) | `VisionCrossAttentionLayer`(`layer_type="joint"` 기본값, `vision_sampler.py` 248, 408행)의 `MultiKVCrossAttention`이 인코더별 K, V를 이어 붙여 한 번에 SDPA | 일치. 인코더별 어텐션 후 학습 가중치로 섞는 `VisionAggregationLayer`(330행)는 코드에 있으나 기본값이 아니고 최종 모델은 `joint`를 씁니다 |
| 질의 (i, j)가 부분 영역만 본다 | §4 | `rearrange_vision_tower_features_*`(`cambrian_arch.py`)가 특징 맵을 (L, m, L, m, C)로 보고 (L², m², C)로 재배열해 질의마다 m²개 키만 줌. `sva_toy.py`에서 이 재배열이 식의 마스크 어텐션과 수치적으로 같음을 확인 | 일치 (`03` §6.1) |
| 질의 격자 L=24 → 576토큰, ConvNeXt m=4 | 부록 F | 스크립트 `query_num_list '[576]'`, `vision_tower_aux_token_len_list '[576,576,576,9216]'` (9216 = 96²) | 일치. LLM 입력 위치는 줄바꿈 토큰 24개가 더해져 600 |
| 위치 인코딩 학습 가능, m_k>1일 때만 | 부록 F | `pos_embed_i = nn.Parameter(randn(kv_size², hidden))`, `kv_size>1`일 때 (`vision_sampler.py`의 `pos_embed_{i}`) | 일치 |
| ConvNeXt 다단계 특징 연결 | 부록 F | `clip_convnext_encoder.py` 70–77행: 4개 stage 은닉 합 `192+384+768+1536`(L), `384+768+1536+3072`(XXL)=5,760 | 일치 |
| D=3, G=1 | §4, 부록 F | 스크립트 `connector_depth 3`, `num_query_group 1` | 일치 |
| 인코더 목록 | 표 10 | `multimodal_encoder/`에 OpenAI CLIP, DFN, EVA, SigLIP, OpenCLIP ConvNeXt, DINOv2, MoCo, MAE, I-JEPA, SAM, MiDaS, Stable Diffusion, 지도 ViT 래퍼가 있음 | 일치(래퍼 존재만 확인, 가중치 미확인) |
| 2단계 학습, 2.5M 어댑터 | §6, 부록 F | 사전학습 스크립트 `alignment_2.5m.jsonl`, 지시 튜닝 `Cambrian7M_withsystemprompt.jsonl`, `--pretrain_mm_mlp_adapter` 연결 | 일치 |
| 시각 토큰 수 576 | 표 9 | 위와 같음 | 일치 |
| MME 점수 정의 | §3.1 | `mme_test.py`: 범주 점수 = 100·(정확도 + acc+), Perception 10개 합(2,000 만점). acc+는 `idx%2` 홀짝으로 같은 이미지 질문 쌍을 짝지음 | 표준 정의와 같음(acc+를 `idx % 2` 홀짝으로 계산하므로 결과 파일이 질문 쌍 순서로 정렬돼 있어야 함) |
| MMVP 쌍 지표 | §3.1 | `mmvp_test.py`: 인접 두 문항을 둘 다 맞아야 정답(쌍 정확도) | 일치 (`03` §1.3의 MMVP 0점 해석 근거) |
| MMBench 원형 평가 | - | `tabulate.py`가 `100x_circular_accuracy` | MMB 우연 수준 2.6의 근거 |
| CV-Bench 지표 | 부록 C | HF README의 합성식 ½·((ADE+COCO)/2 + Omni3D). 평가 코드는 ADE, COCO, Omni를 따로 `100x_accuracy`로 기록하고(`tabulate.py`) 합성은 하지 않음 | 식 일치, 합성은 사용자 몫 |

## 2. 차이 D1–D10

| ID | 논문 | 코드·공개 자료 | 영향 |
|---|---|---|---|
| **D1** | DINOv2는 **ViT-L/14@518** (표 10, 표 23 각주, §6) | 스크립트 세 개와 HF `config.json`의 `mm_vision_tower_aux_list`가 모두 `facebook/dinov2-giant-res378`(ViT-g/14, 378px, 패치 729개를 576토큰으로 보간해 사용, `dino_encoder.py`의 `interpolate`) | 논문의 인코더 ablation(표 3, 14)의 DINOv2는 ViT-L이고 공개 모델은 giant입니다. 최종 모델의 DINOv2 기여를 논문 표로 설명할 수 없습니다 |
| **D2** | in-LLM 교차 어텐션 stride는 8B/13B/34B에서 **3/4/9** (부록 F), 층 개수는 적지 않음 | 스크립트·HF config: stride **3/4/7**, 개수 **10/10/9** (8B: 층 0, 3, …, 27, 13B: 0, 4, …, 36, 34B: 0, 7, …, 56). 모두 LLM 층 수 안 | 34B의 stride가 논문(9)과 코드(7)에서 다릅니다. 층 개수는 논문 어디에도 없어 SVA 파라미터 규모를 논문만으로 알 수 없습니다(`03` §6.1은 코드 기준 산출) |
| **D3** | 표 23: 어댑터 lr **1e-4**, 지시 튜닝 lr **2e-5**(34B 포함 모든 최종 모델), 배치 512/1024 | `scripts/cambrian/*`와 README 표: 연결기 단계 기본 lr **1e-3**이고 `vision_sampler*`, `vision_query` 이름의 파라미터만 **1e-4**(`cambrian_trainer.py` 289–316행; `mm_projector_aux_*`, 최종 `mm_projector`, `image_newline`은 1e-3). 지시 튜닝 lr **4e-5**(8B, 13B), 2e-5(34B). 배치는 README 512/512/1024로 논문과 같음 | 8B·13B 지시 튜닝 lr이 논문의 두 배입니다. 최종 모델을 논문 표 23대로 재학습하면 다른 모델이 됩니다. 어느 쪽이 실제 사용값인지 공개 자료만으로는 README·스크립트가 우선(HF 모델 `training_args.bin`은 읽지 않음) |
| **D4** | "질의에 전역 풀링 특징을 이어 붙임"(부록 F, 풀링 대상은 시각 특징) | `cambrian_arch.py` 376–377행: `if aux_i == 0: global_context_feature = image_aux_features.mean(1)`. **첫 번째 인코더(SigLIP)의 평균 하나**만 쓰고 이것이 모든 격자 위치와 모든 LLM 안 SVA 층에 같은 값으로 들어감 | 논문 서술("시각 특징")은 인코더 전체로 읽히나 코드는 SigLIP 하나. 인코더 순서를 바꾸면 모델이 달라집니다 |
| **D5** | 식 (1): W^Q, W_k^K, W_k^V ∈ R^{C×C} 세 종류의 행렬만 서술 | `VisionCrossAttentionLayer` 한 층에: 전역 문맥 투영 `proj_context`, 질의·문맥 결합 `proj_in`(q+C → C), 출력 MLP `proj_out`, LayerNorm, 어텐션 출력 투영 `o_proj`, K·V·Q 앞 LayerNorm, 16헤드, 잔차 연결이 있고, 입력 쪽에 인코더별 투영 MLP `mm_projector_aux_i`(은닉 차원 → C)가 있음 | 층당 파라미터는 식 (1) 최소(9.4M)의 1.67배(연결기 층 15.8M). 파라미터·성능 비교(표 4의 Concat 대 SVA)에서 연결기 용량이 통제되지 않았을 수 있습니다. 논문 표 4–6에는 SVA의 파라미터 수가 없습니다 (`03` §6.1은 8B 기준 SVA 총 302.8M) |
| **D6** | 퍼지 매칭으로 채점하고 GPT-3.5·4로 검증 (§6, 부록 G.2, 표 24) | 퍼지(문자열 유사도) 매칭 함수는 코드에 없음. 객관식 8개 벤치마크(ADE, COCO, Omni, AI2D, MMBench-EN, MMMU, SEED, V*)는 공통 `extract_mcq_answer`: 소문자화 후 "answer is"/"answer:" 뒤를 취하고 **길이가 1 초과면 첫 글자만** 정답 글자로 봄(예: `ade_test.py` 30행). GPT 채점 스크립트는 `ade/gpt_eval.py`, `omni/gpt_eval.py` 둘뿐 | 표 24의 "퍼지 매칭 정확도"가 공개 채점 코드와 같은 것인지 확인할 수 없습니다. 첫 글자 규칙은 선택지 글자 대신 단어로 답한 경우 첫 글자가 우연히 선택지 글자와 같으면 정답, 다르면 오답이 됩니다(예: 답이 "bookcase"이면 B로 채점). 선택지 글자를 쓰도록 유도한 프롬프트(표 26)가 이 위험을 줄입니다 |
| **D7** | CV-Bench는 공개 HF 저장소 `nyu-visionx/CV-Bench` | 평가 코드는 **개인 계정 HF 저장소**를 읽음: `ade_eval.py` 75행 `SaiCharithaAkula21/benchmark_ade_manual`(633행), `coco_eval.py` 76행 `SaiCharithaAkula21/benchmark_coco_filtered`(805행), Omni는 `omni_eval.py` 137행의 **로컬 경로 기본값** `/scratch/eb3174/datasets/vision-benchmark/3D_VQA.json`(공개되지 않음) | 행 수는 CV-Bench의 ADE 633, COCO 805와 같고 Omni는 HF CV-Bench의 3D 1,200행으로 대체해야 합니다. 개인 저장소 내용이 공식 CV-Bench와 문항 단위로 같은지는 대조하지 않았습니다. 사용자가 3D를 평가하려면 코드 수정이 필요합니다 |
| **D8** | §3의 인코더 23종·어댑터 0/0.5/1.2M·동결 해제·5M 실험(표 12–16, 그림 5, 7) | 공개된 학습 스크립트는 **최종 모델 6개뿐.** 737K·1.2M·5M 혼합 파일, 인코더별 실험 스크립트, 동결 해제 실험 설정은 저장소에 없음(플래그 `unfreeze_mm_vision_tower`, `mm_vision_tower_lr`는 `cambrian_trainer.py`와 `train_tpu.py`에 있음) | 표 12–16, 19, 20은 공개 코드로 재현할 수 없습니다(코드와 모델 래퍼는 있으나 설정과 데이터 혼합이 없음) |
| **D9** | 데이터 균형 t는 250k–350k에서 elbow (§5.2, 표 7) | README "Empirically, we found that setting t to 350k yields the best results". README의 표 8 유사 표는 논문 표 8과 값이 다름: README Cambrian-7M 평균 54.8, Vision-Centric 49.5, Cambrian-10M 53.8/47.6, LLaVA-665K 40.4/31.0 대 논문 55.9/54.1, 54.8/51.4, 40.7/32.0 | README가 CV-Bench를 넣기 전 값이거나 다른 열 구성으로 보임. 논문 표 8만 인용하고 README 값은 쓰지 않습니다. 데이터 필터링(상한 t 적용) 코드는 저장소에 없음(데이터는 이미 처리된 JSONL로만 공개) |
| **D10** | 최종 모델 연결기 비교 기준선(Concat, Resampler, C-Abstractor, LDPv2) | 코드에는 `CAbstractor`(`multimodal_projector/builder.py` 75행), `hybrid_encoder.py`(결합), SVA만 있음. Resampler와 LDPv2의 구현은 찾지 못함 | 표 4, 5의 기준선 일부는 공개 코드로 재현할 수 없습니다 |

## 3. 코드에서 발견한 그 밖의 사항

- **학습·추론 경로 분리.** `cambrian_arch.py`는 `IS_XLA_AVAILABLE`이면 학습용 재배열(`rearrange_vision_tower_features_train`, 데이터 로더가 만든 마스크 사용)을, 아니면 추론용(`..._inference`, `unmask_attention_mask`로 pad 영역을 마스크)을 씁니다. 같은 모델이 두 경로에서 입력 마스킹이 다르게 구현돼 있습니다(`image_aspect_ratio pad`). 이 분석에서는 두 경로의 수치 동치를 확인하지 않았습니다.
- **평가 데이터 출처 일부가 제3자 저장소**입니다(GQA는 `lmms-lab/GQA` 등). 재현 시 데이터 버전 고정이 필요합니다.
- **TPU 의존.** 학습은 TorchXLA + FSDP이고 추론 경로는 GPU(`.cuda()`)를 가정합니다(평가 스크립트의 `.cuda()` 호출).

## 4. 구현 가능성 판정 (24 GB 맥, MPS, CUDA 없음)

| 작업 | 판정 | 근거 |
|---|---|---|
| 표 재계산, 통계, SVA 산술, CV-Bench 주석 분석 | **완료** | `analysis/*.py` (CPU, 수 초) |
| SVA 식 (1) 구현 동치 토이 | 완료 | `analysis/sva_toy.py` (torch CPU) |
| 모델 학습(최종 모델 TPU-V4-512 4일 미만, §3 실험 TPU-V4-128 24시간 미만) | **불가** | 부록 A. GPU 학습 코드도 미공개 |
| §3 인코더·동결 실험 재현(표 12–16) | **불가** | 학습이 필요하고 설정·혼합이 미공개 (D8) |
| 공개 체크포인트 추론: Cambrian-1-8B | **불가 또는 매우 조건부** | HF safetensors가 **33.3 GB(fp32)**. bf16 변환 약 16 GB에 시각 타워 네 개(ConvNeXt-XXL, SigLIP-SO400M, CLIP-L, DINOv2-giant, 약 3B 파라미터, bf16 약 6 GB)를 더하면 24 GB를 넘거나 거의 찹니다. 가중치와 타워를 따로 내려받아야 하고 코드가 CUDA를 전제 |
| 공개 체크포인트 추론: Cambrian-1 Phi-3 3.8B | 조건부 (다운로드와 MPS 포팅, 승인 필요) | HF safetensors 16.4 GB(fp32), bf16 약 8 GB. 시각 타워 약 6 GB 추가. 이번에는 받지 않음. 포팅 일수 미측정 |
| CV-Bench 평가 | 조건부 (이미지 약 405 MB 다운로드, 모델 추론 필요) | 질문·답 주석은 받았고 이미지는 받지 않음. 모델 실행이 필요해 위 추론 판정과 같음 |
| CV-Bench Count 단서 검증(선택지 구조만 보는 소형 분류기) | 가능 (CPU) | 이미지 없이 주석만으로 가능 |
| 공식 코드 실행 | 승인 후 | 이번에는 실행하지 않음 |
