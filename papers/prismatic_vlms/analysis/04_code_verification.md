# 04. 논문 대 공식 코드 대조

대상: `repo/prismatic-vlms`(커밋 `874c5bb`, 2024-05-16, MIT)와 `repo/vlm-evaluation`(커밋 `d9f5896`, 2024-09-12, LICENSE 파일 MIT). 얕은 복제를 **정적으로만** 읽었습니다. 설치, import, 실행, 체크포인트 다운로드는 하지 않았습니다. 평가 결과 파일은 공개되지 않아 점수 재계산은 논문 부록 표(`03`)로 했습니다.

## 1. 일치한 것

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 구조 | 비전 백본 → 패치별 투영기 → LM, 프롬프트 왼쪽에 이어 붙임 (§2, A.2) | `PrismaticVLM`, 임베딩 앞에 연결 (`prismatic.py`) | 일치 |
| 특징 층 | 끝에서 두 번째 층 (A.2) | `get_intermediate_layers(n={len(blocks)-2})` (`base_vision.py`, `dinosiglip_vit.py`) | 일치 |
| 비융합 투영기 | 2층 GELU MLP (A.2) | `MLPProjector`: Linear-GELU-Linear (`nn_utils.py`) | 일치 |
| 하이퍼파라미터 | 배치 128, lr 2e-5, wd 0.1, warmup 0.03, cosine, AdamW (표 1) | `LLaVa_v15_Reproduction_7B` 미세조정 항목 동일. 정렬 단계는 배치 256, lr 1e-3(A.3과 같음) | 일치 |
| 단일 단계 | 정렬 생략 (§4.1) | `arch_specifier = "no-align+..."`, `Exp_7B_One_Stage` | 일치 |
| 미동결 | 전체 미세조정 (§4.1) | `--stage full-finetune`에서 `vision_backbone.requires_grad_(True)` | 일치 |
| 학습 시간 | 1.25, 1.5, 2, 3 epoch | `finetune_max_steps` 6,500, 7,800, 2, 3 epoch. 665,298/128×1.25 = 6,497 | 일치 (`03` §5.3) |
| 프롬프트 | Vicuna는 시스템 프롬프트 + USER/ASSISTANT, base LM은 `In: ... Out:` (A.2) | `vicuna_v15_prompter.py`, `base_prompter.py` | 일치 |
| 생성 | greedy (B.1) | `do_sample=False`, `max_new_tokens=128` 기본 (`vlm_eval/models/prismatic.py`) | 일치. 128토큰 제한은 논문에 없음 |
| 위치 지정 지표 | IoU 0.5, OCID-Ref는 0.25 (B.1) | `refcoco.py`, `ocidref.py` (잘못 파싱한 박스는 오답으로 셈) | 일치 |
| 평가 문항 수 | 표준 분할 (§3) | `expected_examples`: VQAv2 214,354, GQA 12,578, VizWiz 4,319, TextVQA 5,000, RefCOCO계 26,488, OCID-Ref 18,342, VSR 1,222, POPE 9,000, TallyQA 38,589, AI2D 15,501 | 일치(`03` §1.1) |

## 2. 차이 (D1–D10)

**D1. 융합 투영기는 2층이 아니라 3층입니다 (논문과 다름, 영향 큼).** §4.2는 "투영기(2층 MLP)의 입력 차원만 늘린다, 비용은 미미하다"고 씁니다. 코드의 `FusedMLPProjector`는 Linear(입력 → 4×입력) - GELU - Linear(4×입력 → LM 차원) - GELU - Linear(LM → LM)입니다(`nn_utils.py` 38–53행). 모든 융합 설정(`fused-gelu-mlp`)이 이것을 씁니다. 7B에서 융합 7,139만 대 비융합 약 2,100만 파라미터(`03` §5.1). 융합 이득(위치 지정 +11점)에는 두 번째 인코더의 정보와 투영기 용량 증가가 섞여 있고 논문 비교는 이를 분리하지 않았습니다. DINOv2+CLIP이 이득이 없었다는 사실은 용량만으로 설명되지 않음을 시사하지만(같은 3층), 이 분석에서 분리 실험은 하지 않았습니다.

**D2. "DINOv2"는 레지스터 4개 변형입니다.** `vit_large_patch14_reg4_dinov2.lvd142m`(`dinov2_vit.py`, `dinosiglip_vit.py`; 코드 주석에 "Using DINOv2 w/ Registers", arXiv:2309.16588). 논문 본문은 레지스터를 언급하지 않습니다. DINOv2 단독 224px에서 positional embedding은 518 기준 사전학습에서 `img_size` 인자로 보간됩니다.

**D3. 닫힌 집합 벤치마크는 첫 토큰 확률의 argmax입니다 (논문에 서술 없음).** VSR, POPE, TallyQA, AI2D는 `return_string_probabilities`로 "True/False", "Yes/No", 선택지 글자의 첫 토큰 확률을 구해 후보끼리 정규화한 뒤 argmax를 정답으로 씁니다(`prismatic.py` `generate_batch` 527–530행, 각 harness의 scorer). 부록 B.1은 "객관식으로 형식을 바꿔 정확도"만 씁니다. 영향: (a) 모델이 후보 밖 토큰을 내도 점수에 안 드러남, (b) VSR에서 51.47%는 항상 같은 쪽 확률이 큰 모델(`03` §1.2), (c) TallyQA는 A–P 16개라 정답 16 이상은 표현 불가(데이터 없이 비율 확인 못 함), (d) AI2D는 harness가 A–D 4개 확률을 늘 비교하므로 선택지가 4개보다 적은 문항에서 존재하지 않는 글자가 뽑힐 수 있음(코드상 가능성이며 데이터 없이 빈도는 확인하지 못함).

**D4. `Prism_13B_SigLIP` 설정의 비전 백본이 CLIP입니다 (설정 오류로 보임).** `vision_backbone_id: str = "clip-vit-l-336px"` (`models.py`의 `Prism_13B_SigLIP`, 7B와 Controlled 13B는 `siglip-vit-so400m-384px`). 논문 표 2–4 "Prism-SigLIP 13B" 행이 이 설정으로 학습됐는지는 알 수 없습니다. 정황: 7B의 SigLIP-CLIP TextVQA 격차는 +6.9점, 13B는 +0.9점입니다(Controlled는 +5.5 대 -1.6). 13B 비Controlled의 격차 붕괴는 오류와 맞지만, 설정이 올바른 13B Controlled에서도 SigLIP이 CLIP에 TextVQA -1.6, RefCOCO -11.7점으로 지므로 오류만으로 설명되지 않습니다(`03` §2).

**D5. 평가 생성 설정과 배치.** 평가는 배치 1만 지원합니다(`evaluate.py` 주석 "batch size 1 until LLaVa/HF LLaMa fixes bugs"). `generate_batch`에서 루프 안에서 `pixel_values = pixel_values[idx]`로 변수를 덮어써 배치 2 이상에서는 깨집니다(정적 읽기에서 보이는 잠재 문제, 배치 1에서는 무해). `temperature=1.0`이 `do_sample=False`와 같이 전달됩니다(HF가 무시하고 경고). 점수에는 영향 없음.

**D6. 해상도 비교는 체크포인트가 다릅니다.** 224px 실험은 `clip-vit-l`(`vit_large_patch14_clip_224.openai`)과 `siglip-vit-so400m`(224), 336/384px 실험은 `vit_large_patch14_clip_336.openai`와 `..._siglip_384`입니다. 224px 설정은 `LLaVa_v15_Reproduction_7B`의 `letterbox`를 상속합니다. 384/14는 정수가 아니라 27×27(729) 격자로 합성곱이 우하단 6픽셀을 버립니다.

**D7. OpenVLA가 쓴 224px PRISM 모델은 논문 표에 없습니다.** `prism-dinosiglip-224px+7b`(resize-naive, `llama2-7b-pure`, `no-align+fused-gelu-mlp`, 2 epoch, `models.py` 475–490행 근처 "Inference-Optimized")는 코드에만 있고 논문 표 2–4에는 해상도 384의 융합 모델만 있습니다. 224px 융합 모델의 VLM 벤치마크 점수는 이 논문에서 확인할 수 없습니다.

**D8. 공개된 점수 집계 스크립트가 논문의 "12개 평균"과 다르고, 검정 코드는 없습니다.** `scripts/compute_z_scores.py`: `TASK_LIST`는 7개 작업(VQAv2, GQA, VizWiz, TextVQA, RefCOCO, OCID-Ref, POPE)이고 `METRICS_TO_AVERAGE_OVER`는 TextVQA-OCR과 TextVQA-Pure를 둘 다 포함하며 RefCOCO+, VSR, TallyQA, AI2D가 없습니다. 입력 경로는 작성자 개인 데이터 경로입니다. 표준편차는 `np.std`(모집단)로 `03`과 같게 썼습니다. 1-sided Fisher T-test(부록 B.2)는 두 저장소 어디에도 없습니다(`ttest`, `scipy` 검색 0건).

**D9. 이미지 전처리 구현 세부.** letterbox는 `int((max_wh - w)/2)`로 내림하므로 짝수 차이가 아니면 정사각이 1픽셀 모자라고(이후 Resize가 흡수), 패딩 색은 검정이 아니라 정규화 평균×255입니다(`LetterboxPad`). SigLIP과 IN1K의 기본 timm 변환은 더 크게 리사이즈한 뒤 크롭하므로 코드가 `Resize(default_image_size)`로 고쳤습니다. "resize-crop"은 이 수정 뒤의 짧은 변 리사이즈 + 중앙 크롭입니다.

**D10. 투영기 초기화 시드가 백본 차원에 묶여 있습니다.** `torch.manual_seed(vision_backbone.embed_dim)`(`prismatic.py`). 같은 백본끼리는 같은 초기화지만 다른 백본 간 비교는 초기화 난수도 다릅니다(`03` §2).

## 3. 라이선스와 재사용

- `prismatic-vlms`: MIT (LICENSE 첫 줄 확인). `vlm-evaluation`: LICENSE 파일 MIT, GitHub API 메타데이터는 NOASSERTION. 두 저장소의 코드는 `repo/`(gitignore)에만 두고 게시본에 넣지 않았습니다.
- 모델 가중치와 데이터셋 라이선스(LLaVA 혼합의 ShareGPT, GPT-4 생성 데이터)는 확인하지 않았습니다. 가중치를 쓸 때 Llama-2 라이선스를 따로 확인해야 합니다.

## 4. 구현 가능성 (이 맥, Apple Silicon 24 GB)

| 작업 | 판정 | 근거 |
|---|---|---|
| 부록 표 재계산, 검정, 계산량 산술 | 완료 | `recompute.py` |
| 학습 재현 (논문: 7B 약 9시간, 8×A100) | 불가 | GPU 필요. 코드는 FSDP와 CUDA 전제 |
| 공개 체크포인트 추론 (7B bf16 약 15 GB 가중치) | 조건부 | 가중치 다운로드와 MPS 포팅이 필요하며 이 분석에서는 받지 않음. 승인 필요 |
| 평가 하네스 실행 | 조건부 | 데이터셋 다운로드(VQAv2 이미지만 수십 GB 규모, 확인 안 함)와 승인 필요 |
| 공식 코드 실행 | 승인 후 | 설치와 실행은 하지 않음 |
