# 04. 논문 대 공식 코드 대조

대상은 네 가지이고 모두 **정적으로만** 읽었습니다. 설치, import, 실행, 가중치 다운로드는 하지 않았습니다.

| 대상 | 판 | 용도 |
|---|---|---|
| `QwenLM/Qwen2-VL` (현재 `QwenLM/Qwen3-VL`로 이름이 바뀜) | `9658872` (2026-01-30), 얕은 복제 195 MB | `qwen-vl-utils`(전처리 유틸), `qwen-vl-finetune/qwenvl/data/rope2d.py`(학습용 위치 번호), README |
| `huggingface/transformers` `models/qwen2_vl/`, `vision_utils.py` | main `24a90da` (2026-11-26), 필요한 파일만 raw 수신 | 모델 구현, 이미지·영상 전처리 |
| transformers `modeling_qwen2_vl.py` | tag `v4.45.0` (2024-09, 모델이 처음 들어간 판) | main과의 변경 대조 |
| Hub `Qwen/Qwen2-VL-{2B,7B,72B}-Instruct`의 `config.json`, `preprocessor_config.json` (각 1 KB) | 2026-11-26 | 구조 설정, 전처리 기본값 |

모델 구현은 공식 저장소에 없고 transformers에 있습니다. 공식 저장소는 학습용 위치 번호 계산(`rope2d.py`)과 유틸을 가집니다. 학습 코드 전체(데이터, 학습 루프)는 이 저장소의 `qwen-vl-finetune`이 공개한 미세조정용이고, 논문의 사전학습 코드는 공개되지 않았습니다.

## 1. 일치한 것

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 비전 인코더 → 병합 → LLM | §2.1, 그림 2 | `Qwen2VLModel`: `visual`(`Qwen2VisionTransformerPretrainedModel`) + `language_model` | 일치 |
| ViT 2D-RoPE, 절대 위치 임베딩 제거 | §2.1 | `Qwen2VLVisionRotaryEmbedding`, 패치 임베딩에 위치 항 없음 | 일치 |
| 시간 방향 3D 합성곱, 깊이 2 | §2.1 | `PatchEmbed`: `Conv3d`, 커널 [2, 14, 14], stride 같음, 편향 없음 | 일치 |
| 이미지 = 같은 프레임 두 장 | §2.1 | `patchify`가 시간 방향으로 `expand`(복제)해 t = 1 격자 | 일치 |
| 2×2 병합 | §2.1 | `PatchMerger`, 격자가 (h/2, w/2, 2, 2) 블록 순서라 연속 4개가 이웃 | 일치 (D4) |
| 224² → 66토큰 | §2.1 | `smart_resize` → 8×8 = 64, + 시작·끝 2 | 일치 (`03` §2) |
| 텍스트에서 M-RoPE = 1D-RoPE | §2.1 | 세 성분이 같으면 `recomposition_frequencies`가 1D와 같은 cos, sin | 일치 (직접 구현으로 비트 동일, `03` §1.3) |
| 파라미터 (ViT 675M, LM 1.5B/7.6B/72B) | 표 1 | config로 센 값이 Hub safetensors 총합과 정확히 일치 | 일치 (`03` §3.1) |
| 영상 프레임 상한 768 | §3.2.6 | `Qwen2VLVideoProcessor.max_frames = 768`, `qwen-vl-utils` `FPS_MAX_FRAMES = 768`, 기본 fps 2.0 | 일치 |
| 문맥 길이 32K | §3.2.7 | config `max_position_embeddings = 32768` | 일치 (D9) |
| 학습용 위치 번호 | §2.1 | 공식 `rope2d.py`의 `get_rope_index_2`가 "다음 구간 = 지금까지 최대 번호 + 1" | 일치 (`03` §1.4) |
| 슬라이딩 윈도우 | (논문에 없음) | config `use_sliding_window: false`라 전 층 full attention | 영향 없음 |

## 2. 차이 (D1–D10)

**D1. M-RoPE의 채널 분할이 논문에 없습니다 (논문에 없음, 영향 큼).** 논문은 "회전 임베딩을 시간·높이·너비 세 성분으로 분해"한다고만 씁니다(§2.1). 코드와 config(2B, 7B, 72B 공통)의 `mrope_section = [16, 24, 24]`가 head_dim 128의 64쌍을 시간 16, 높이 24, 너비 24로 자릅니다(비율 25%, 37.5%, 37.5%). 배치는 청크 방식(0–15 시간, 16–39 높이, 40–63 너비)이라 너비 축이 가장 느린 주파수를 받습니다(`03` §1.2, §1.8). base 주파수는 `rope_theta = 10⁶`입니다.

**D2. ViT의 세부 구조가 논문에 없습니다 (논문에 없음).** config: depth 32, embed 1280, heads 16(head_dim 80), mlp ratio 4, 활성화 `quick_gelu`, patch 14, spatial merge 2, temporal patch 2. 논문은 "약 675M"과 "DFN에서 초기화"만 씁니다(§2.1, §2.2). 블록 32개 + 패치 임베딩은 631.2M으로, ViT-H/14 규모와 맞습니다. DFN 가중치와의 대응은 확인하지 못했습니다.

**D3. ViT 2D-RoPE의 주파수 배분이 논문에 없습니다 (논문에 없음).** `compute_axial_rope_parameters`: `spatial_dim = head_dim // 2 = 40`, `inv_freq = 1 / 10000^(arange(0, 40, 2) / 40)` (20개), 높이 위치용과 너비 위치용에 같은 주파수를 쓰고 이어 붙인 뒤 한 번 더 이어 붙여 head_dim 80을 채웁니다(`recomposition_frequencies`). 위치 번호는 병합 전 패치 격자의 행·열이고, 2×2 블록 순서로 나열됩니다(`get_vision_position_ids`). 직접 구현으로 평행 이동 불변을 확인했습니다(`03` §1.7).

**D4. 병합기는 "하나의 MLP 층"이 아니라 LayerNorm + 2층 MLP입니다 (논문과 약간 다름, 영향 작음).** §2.1은 "simple MLP layer"라고 씁니다. 코드의 `PatchMerger`는 `LayerNorm(1280)` → 인접 4개 이어 붙임(5,120) → `Linear(5120, 5120)` → GELU → `Linear(5120, LM 폭)`입니다. 7B에서 약 44.6M, 72B에서 68.2M 파라미터로 LM 폭에 따라 커지므로 "ViT는 LLM 크기와 무관하게 675M"은 병합기를 뺀 631.2M에 대해서만 정확합니다(`03` §3.1).

**D5. `smart_resize`의 가장자리 동작이 두 구현에서 다릅니다 (영향 작음, 드문 경우).**
- transformers: `h_bar = round(h / 28) * 28`. 한 변이 14 px 이하이면 Python `round`가 0.5를 짝수로 보내 0이 되고, 넓이가 `min_pixels`보다 작다고 판단해 `ceil(h·β/28)·28`로 키우는 가지에 들어갑니다. 14×2000은 28×672로 줄어 가로가 3분의 1로 깎입니다(`03` §2.2). 비율이 200을 넘으면 오류.
- `qwen-vl-utils`: `max(factor, round_by_factor(·))`로 0을 막습니다.
- 기본값도 다릅니다. transformers 코드는 `max_pixels = 28·28·1280`(1,280토큰), Hub의 `preprocessor_config.json`은 `12,845,056`(16,384토큰), `qwen-vl-utils` HEAD는 이미지 최대 16,384토큰입니다. 같은 이미지가 어떤 경로로 들어가느냐에 따라 토큰 수가 2배 이상 갈릴 수 있습니다(1344²: 1,225 대 2,304).

**D6. 영상 뒤 텍스트의 시작 위치가 transformers main과 공식 학습 코드에서 다릅니다 (논문 §2.1과 다름, 영향은 미확인).**
- 논문(§2.1), 공식 `rope2d.py` `get_rope_index_2`, transformers v4.45.0: `st_idx = llm_pos_ids_list[-1].max() + 1`. 즉 영상의 (시간, 높이, 너비) 번호 중 **최대값 + 1**에서 다음 텍스트가 시작합니다.
- transformers main `Qwen2VLModel.get_rope_index`: `current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size`. 시간 격자 길이 t'를 무시합니다. 같은 변경이 `qwen2_5_vl`과 `qwen3_vl`의 같은 줄에도 있습니다.
- t' > max(h', w')인 영상(예: 2 fps, 60초, 448×448: t' = 60 > 16)에서만 값이 갈리고, 이때 main은 영상의 시간 번호와 겹치는 번호에서 텍스트를 시작합니다(`03` §1.6). 이미지에서는 갈리지 않습니다. 직접 구현으로 두 규칙의 출력을 비교했을 뿐 코드를 실행하지는 않았습니다. 학습은 `rope2d.py` 쪽 규칙을 썼다고 보이므로(학습 코드 전체가 공개된 것은 아님) HF main으로 긴 영상을 추론하면 학습 분포와 다른 위치를 줄 수 있습니다. 확정 전에 transformers의 해당 파일 변경 이력과 이슈를 확인해야 합니다. 이 분석은 변경 이력을 보지 않았습니다.

**D7. 시간 번호는 "프레임마다"가 아니라 "2프레임 묶음마다" 늘어납니다 (표현 차이, 영향 작음).** 논문 §2.1: "영상에서 시간 ID는 프레임마다 증가". 코드: `video_grid_thw`의 t는 프레임 수의 절반이고 `get_vision_position_ids`의 시간 번호는 `arange(t') * time_interval`(Qwen2-VL에서 `time_interval = 1`)이라 2프레임마다 1입니다. Qwen2.5-VL 코드는 `time_interval = tokens_per_second × second_per_grid_ts`로 절대 시간에 맞춥니다(`02` §3).

**D8. "영상당 토큰 16,384 제한"의 구현이 공개되어 있지 않습니다 (논문에만 있음).** §2.1: 학습 때 영상당 총 토큰을 16,384로 제한하며 프레임 해상도를 동적으로 조절한다고 합니다. 공개 코드에 이를 구현한 곳은 없습니다. `qwen-vl-utils`는 `MODEL_SEQ_LEN = 128000`(환경 변수)에서 `total_pixels = MODEL_SEQ_LEN·28·28·0.9`로 프레임당 최대 넓이를 정하는 추론용 예산을 둡니다(HEAD 기준). transformers의 영상 프로세서에도 `cap_pixels_per_frame`(기본 미설정, 경고에 "v5.22에서 기본값이 True가 됨"), `max_video_tokens = 128000`가 있습니다. 즉 16,384는 학습 설정이고 공개 추론 경로의 영상 예산은 다릅니다. 학습 토큰 예산이 추론과 다르면 영상 길이 일반화(그림 5)의 의미가 달라집니다.

**D9. 80K 토큰 추론(그림 5)의 설정이 문서화되어 있지 않습니다.** config `max_position_embeddings`는 32,768이고 논문 그림 5는 최대 80K토큰 추론을 보입니다. M-RoPE는 위치 번호를 압축하므로 영상 구간의 번호는 수백 안이라(`03` §1.9) rotary의 위치 한도에는 걸리지 않지만 KV 길이 80K를 어떻게 처리했는지(메모리, 프레임·해상도 설정)는 코드와 논문 어디에도 없습니다. 현재 저장소 README의 긴 문맥 설명(YaRN, 256K)은 Qwen3-VL용입니다.

**D10. 저장소가 Qwen3-VL로 바뀌었습니다 (재현성 단서).** `QwenLM/Qwen2-VL`이 `QwenLM/Qwen3-VL`로 이름이 바뀌어 README, 쿡북, `qwen-vl-utils`(HEAD, `image_patch_size` 인자와 이미지 최대 16,384토큰, 영상 128–768토큰 상수)가 후속 모델 기준입니다. `image_patch_size`의 기본값은 14라 병합 단위 28은 Qwen2-VL과 같지만, 이 분석이 읽은 `qwen-vl-utils`는 Qwen2-VL 시기의 판이 아니라서 영상 예산 상수(`03` §2.3)가 당시 값과 다를 수 있습니다.

## 3. 구조 요약 (config)

| | 2B | 7B | 72B |
|---|---|---|---|
| hidden / FFN / 층 / heads / KV heads | 1,536 / 8,960 / 28 / 12 / 2 | 3,584 / 18,944 / 28 / 28 / 4 | 8,192 / 29,568 / 80 / 64 / 8 |
| head_dim | 128 | 128 | 128 |
| vocab | 151,936 | 152,064 | 152,064 |
| 임베딩 공유 | 예 | 아니오 | 아니오 |
| `rope_theta`, `mrope_section` | 10⁶, [16, 24, 24] | 같음 | 같음 |
| 가중치 | bf16, Apache-2.0 | bf16, Apache-2.0 | bf16, "other" (Qwen 라이선스) |
| ViT | depth 32, embed 1280, heads 16, patch 14, merge 2, temporal 2 | 같음 | 같음 |

## 4. 구현 가능성 (이 맥, 24 GB, MPS)

| 작업 | 판정 | 근거 |
|---|---|---|
| 토큰 수·위치 번호·파라미터 산술, 표 재계산, 통계 | 완료 | `analysis/*.py`, `03` |
| 2B 추론 (bf16 약 4.4 GB) | 조건부 | 가중치 다운로드 승인 필요, transformers·torchvision 등 설치 필요, MPS 정확도 확인 필요. 이번에는 하지 않음 |
| 7B 추론 (bf16 약 16.6 GB) | 조건부, 빠듯함 | 24 GB에서 시각 토큰 수천 개를 넣으면 ViT attention 활성화와 KV가 더해짐 |
| 72B | 불가 | 약 147 GB |
| D6의 실측(main 대 v4.45.0 대 `rope2d.py`의 위치 번호 출력 비교) | 조건부, 저렴 | 모델 가중치 없이 `get_rope_index` 함수만 부르면 되나 transformers 설치(승인 필요)가 요구됨. 직접 구현(`mrope_check.py`)으로 규칙만 대조해 두었음 |
| 학습, 12개 이상 벤치마크 평가 | 불가 | GPU, 데이터 |

## 5. 후속 모델 코드와의 변경 요약 (참고용, 정적 확인)

| | Qwen2-VL | Qwen2.5-VL | Qwen3-VL |
|---|---|---|---|
| 시간 번호 | 2프레임 묶음마다 +1 | `tokens_per_second × second_per_grid_ts` (절대 시간) | 프레임마다 타임스탬프 텍스트, 시간 격자 1 |
| M-RoPE 채널 | 청크 [16, 24, 24] | 청크, 기본값 [16, 24, 24] (`recomposition_frequencies`가 같은 규칙) | **인터리브** [24, 20, 20] |
| ViT | 전 층 전체 attention(프레임 단위) | 창 attention + 일부 층만 전체(`window_size`, `fullatt_block_indexes`) | 읽지 않음(`deepstack_features` 존재) |

이 표는 transformers main의 모델 파일에서 확인한 것이고 후속 논문 본문은 읽지 않았습니다.
