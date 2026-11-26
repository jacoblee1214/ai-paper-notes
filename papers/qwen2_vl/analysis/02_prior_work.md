# 02. 선행 연구와 계보

## 1. 이 논문이 서 있는 자리

Qwen2-VL은 "비전 인코더 → 연결기 → LLM"이라는 Qwen-VL의 틀을 유지하면서 **시각 토큰의 개수(해상도)와 위치 번호를 구조로 풀었습니다.** 새 부품의 대부분은 기존 아이디어를 가져와 조합한 것이고(아래 표), 논문이 직접 서술하는 새 요소는 naive dynamic resolution의 구성(패킹 + 2×2 병합 MLP)과 M-RoPE입니다.

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| RoPE | RoFormer (arXiv:2104.09864, Neurocomputing 2024로 목록 기재) | M-RoPE는 회전 각을 세 축으로 나눈 확장입니다(`03` §1). 논문은 Su 2021, 2024(블로그)를 인용 |
| ViT 2D-RoPE | 논문은 RoFormer와 Su(2021)를 인용. 축 분할(절반은 높이, 절반은 너비)은 코드에서 확인(`04` D3) | 절대 위치 임베딩 제거 |
| 가변 해상도·패킹 | NaViT (Patch n' Pack, arXiv:2307.06304, NeurIPS 2023 ✓) | 논문이 Dehghani 외 2024로 인용(§2.1) |
| ViT 초기값 | DFN (Data Filtering Networks, arXiv:2309.17425) | §2.2 |
| 3D 합성곱 영상 토큰 | Carreira와 Zisserman 2017 (I3D), ViViT (Arnab 외 2021) | §2.1 |
| 기본 틀과 3단계 학습 | Qwen-VL (arXiv:2308.12966) | §2.2 "Following Qwen-VL" |
| LLM | Qwen2 | 1.5B, 7.6B, 72B |
| 위치 지정 표기 | Qwen-VL의 박스 토큰, 좌표 [0, 1000) 정규화 | §2.2.1 |
| 사내 이전 판 | Qwen-VL Plus, Qwen-VL Max | v2 각주 1: naive dynamic resolution은 이미 여기에 구현됐고 Qwen2-VL에서 개선 |

## 2. 가까운 대안: 가변 해상도를 푸는 방법들

| 방식 | 대표 | 토큰 수 | Qwen2-VL과의 관계 |
|---|---|---|---|
| 고정 해상도 | CLIP/SigLIP 계열 LLaVA-1.5 (`llava/`, `prismatic_vlms/`) | 해상도마다 고정(224²→256, 336²→576) | Qwen2-VL이 §1에서 비판하는 대상 |
| 타일링 | LLaVA-NeXT AnyRes(블로그), InternVL 1.5 (arXiv:2404.16821, 기술 보고서) | 타일 수 × 타일당 토큰 + 썸네일 | 비율에 맞춰 격자로 자르는 방식. 타일 사이 연결은 LLM 위치 번호에만 의존. 이 분석에서 본문을 읽지 않았고 목록 기재만 씀 |
| 원본 해상도, 패치 패킹 | NaViT (비전 모델), **Qwen2-VL** | 격자 크기에 비례 | ViT 안에서 비율을 유지하고 LLM에 가변 길이로 넣는 첫 대형 공개 VLM 사례로 논문이 소개. 비교 실험은 고정 토큰 수 대 동적(표 7)뿐이고 타일링과의 직접 비교는 없습니다 |
| 문서·스크린샷 | Pix2Struct (arXiv:2210.03347, ICML 2023로 arXiv comment 기재) | 가변 해상도 패치 + 2D 위치 | 가변 해상도 패치 입력의 선례. Qwen2-VL이 문서 벤치마크에서 강한 이유의 한 요인으로 논문이 암시(§3.2.2)하나 절제는 없음 |
| 이후 | LLaVA-OneVision (arXiv:2408.03326, 학회 표기 없음): 단일 이미지, 여러 이미지, 영상을 AnyRes 토큰화 하나로 다룸(목록 기재, 본문 미독) | | |

- **표 7의 "고정 토큰 수"는 타일링이 아니라 정사각 한 장 리사이즈**입니다(해당 토큰 수가 정사각 격자임을 `03` §2에서 확인). 그래서 이 절제는 "동적 대 AnyRes"가 아니라 "원본 비율 유지 대 고정 크기"를 가릅니다.

## 3. 위치 인코딩 계보

| 논문·모델 | 위치 처리 | 근거 |
|---|---|---|
| RoFormer (RoPE) | 1D 회전, 쌍 단위 | 원전 |
| ViT (`vit/`) | 학습된 절대 위치 임베딩, 해상도가 바뀌면 보간 | 목록 기재 |
| Qwen-VL (전작) | 입력 크기를 고정한 이미지를 어댑터로 고정 개수 토큰으로 줄이고, 그 토큰을 LLM의 1D 순서로 넣는 것으로 기억함(확인 필요) | arXiv:2308.12966. 본문은 이 분석에서 읽지 않음. Qwen2-VL §1은 "고정 해상도"가 전작의 한계라고 씀 |
| **Qwen2-VL (이 논문)** | LM은 M-RoPE(시간·높이·너비 세 번호), ViT는 2D-RoPE | `03` §1, `04` |
| Qwen2.5-VL (arXiv:2502.13923) | M-RoPE의 시간 번호를 **절대 시간**에 맞춤: `time_interval = tokens_per_second × second_per_grid_ts` (HF 코드, `04` §5). ViT에 창 attention(`fullatt_block_indexes`, `window_size`) | 코드 정적 확인(`modeling_qwen2_5_vl.py`). 논문 본문은 읽지 않음 |
| Qwen3-VL (arXiv:2511.21631) | M-RoPE 채널을 **인터리브**(`[24, 20, 20]`, T·H·W를 3개 단위로 번갈아). 영상은 타임스탬프 텍스트로 구분 | 코드 정적 확인(`modeling_qwen3_vl.py`, 공식 저장소 `rope2d.py`의 `get_rope_index_3`). 논문 본문은 읽지 않음 |
| Round and Round We Go (arXiv:2410.06205, ICLR 2025 포스터 ✓로 목록 기재) | RoPE가 거리 감쇠보다 주파수 대역 사용 때문에 유용하다는 분석 | 목록 기재만. 본문 미확인. `03` §1.7의 "채널 분할과 주파수 대역" 논점과 이어짐 |

- 우리 `03` §1.7이 보인 "너비 축이 가장 느린 주파수 24쌍을 받는다"는 Qwen2-VL의 청크 배치와, Qwen3-VL이 인터리브로 바꾼 코드의 차이를 직접 계산한 것입니다. Qwen3-VL이 바꾼 이유는 이 분석에서 확인하지 못했습니다(논문 미독).

## 4. 이후의 같은 질문

- **Qwen2.5-VL, Qwen3-VL** (진행 순서 161, 162번): 위 표의 변경에 더해 Qwen3-VL은 DeepStack 특징(코드에 `deepstack_features`)을 씁니다. 두 논문의 본문은 아직 분석하지 않았습니다.

## 5. 같은 묶음·같은 저장소의 분석과의 관계

| 폴더 | 관계 |
|---|---|
| `../prismatic_vlms/` | 같은 VLM 설계 공간을 통제 비교한 논문. Prismatic은 해상도를 224/336/384로 고정하고 letterbox 대 naive resize를 비교했습니다. Qwen2-VL의 동적 해상도는 그 축 바깥(비율 유지 + 가변 길이)입니다. 표 7은 해상도 효과가 벤치마크마다 다르다(InfoVQA·OCRBench는 크게, MMMU는 거의 0)고 보여 Prismatic의 해상도 결론(평균 z +0.41)과 같은 방향이지만, 7B 한 모델의 단일 실행이라 통제 수준은 Prismatic보다 낮습니다 |
| `../paligemma/` | 고정 해상도 224/448/896 체크포인트를 따로 두는 대안(`paligemma/`는 같은 묶음의 다른 분석, 읽지 않음). Qwen2-VL은 한 체크포인트가 여러 해상도를 처리 |
| `../transfusion/`, `../cambrian1/`, `../eyes_wide_shut/`, `../vit/`, `../clip/`, `../siglip/`, `../dinov2/`, `../blip2/`, `../llava/` | 같은 묶음의 다른 분석. 이 분석에서는 폴더 이름으로 연결만 합니다 |
| `../eval_error_bars/` | 군집 표준오차와 쌍체 비교. 표 2의 문항들은 같은 이미지·문서에서 여러 문항을 만든 군집이 있으나(DocVQA, ChartQA 등) 우리 `03` §4의 구간은 반영하지 못했습니다 |
| `../little_power/` | MDE = 2.80·√(불일치율/n)의 출처. 표 7과 8의 작은 차이를 판정하는 데 씁니다(`03` §4) |
| `../construct_validity/` | 표 2의 21개 벤치마크가 "지각 능력"을 재는지 질문할 때. 이 논문의 "문서 이해 우위" 같은 주장이 어떤 구성 개념을 재는지 |

## 6. 공식 저장소와 자료

| 저장소 | 관계 | 라이선스 | 크기, 별, 마지막 push | 기준 | 상태 |
|---|---|---|---|---|---|
| [QwenLM/Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) | 논문이 가리키는 저장소. **현재 `QwenLM/Qwen3-VL`로 이름이 바뀌어 같은 저장소를 가리킵니다** (`gh api`의 full_name이 `QwenLM/Qwen3-VL`) | Apache-2.0 | 138 MB(API), 별 20,261, 2026-01-30 | `9658872` (2026-01-30) | 얕은 복제, 정적 읽기. 작업 트리 195 MB |
| [huggingface/transformers](https://github.com/huggingface/transformers) `models/qwen2_vl/` | 모델·전처리 구현(논문이 직접 가리키지 않음) | Apache-2.0 | 파일 5개 + `vision_utils.py`만 raw로 내려받음(합 약 130 KB) | main `24a90da` (2026-11-26) | 정적 읽기 |
| transformers v4.45.0 `modeling_qwen2_vl.py` | 모델이 처음 들어간 판(2024-09) | Apache-2.0 | 약 90 KB | tag v4.45.0 | 정적 읽기. main과의 차이 대조용 |
| Hugging Face `Qwen/Qwen2-VL-{2B,7B,72B}-Instruct` | 공개 가중치 | 2B·7B Apache-2.0, 72B "other"(Qwen 라이선스, API 표기) | `config.json`과 `preprocessor_config.json`만 받음(각 1 KB). 가중치 안 받음 | 2026-11-26 조회 | `03` §3의 파라미터 검산에 safetensors 총 파라미터 수(API 메타데이터)만 씀 |

- 받은 크기: `repo/` 195 MB(Qwen2-VL 얕은 복제 195 MB, transformers 파일 132 KB, v4.45.0 파일 88 KB). 논문 PDF 36 MB(25 MB 초과라 gitignore, README의 재다운로드 명령).
- 모델 소스는 이 저장소가 아니라 transformers에 있습니다. 공식 저장소는 `qwen-vl-utils`(전처리 유틸)와 `qwen-vl-finetune`(`rope2d.py`에 학습용 위치 번호 계산)을 가집니다(`04`).
