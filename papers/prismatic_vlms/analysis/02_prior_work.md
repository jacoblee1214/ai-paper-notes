# 02. 선행 연구와 계보

## 1. 이 논문이 서 있는 자리

Prismatic은 "패치를 토큰으로 투영해 LM에 넣는" 단순 구조(LLaVA 계열)를 고정하고, 그 위의 선택지를 같은 데이터와 같은 평가로 하나씩 바꿔 보는 **통제 비교 논문**입니다. 새 구조를 제안하지 않습니다. 구성 요소의 출처는 다음과 같습니다.

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 패치-토큰 VLM, MLP 투영기, 2단계 학습 | LLaVA (arXiv:2304.08485, NeurIPS 2023 구두), LLaVA-1.5 (arXiv:2310.03744, CVPR 2024) | 이 논문의 기준 설정이 LLaVA-1.5 재현입니다(§4, 그림 4) |
| 데이터 혼합 | LLaVA-1.5 665K + 558K, LVIS-Instruct4V, LRV-Instruct | 부록 A.1 |
| 비전 백본 | CLIP (arXiv:2103.00020), SigLIP (arXiv:2303.15343), DINOv2 (arXiv:2304.07193), ViT (arXiv:2010.11929) | 각각 `clip/`, `siglip/`, `dinov2/`, `vit/` |
| 특징 융합 | Kerr et al. LERF(ICCV 2023), Kobayashi et al. (2022)의 특징장 증류 | §4.2에서 DINOv2+CLIP 결합의 근거로 인용 |
| 대조 대상 | InstructBLIP (arXiv:2305.06500), PaLI-3 (arXiv:2310.09199), Fuyu-8B | InstructBLIP은 Q-Former 구조라 위치 지정 평가는 N/A (표 3) |
| LM | Llama-2, Vicuna v1.5, Mistral 7B | |

## 2. 이후의 같은 질문 (계보)

| 논문 | 무엇을 바꿨나 | Prismatic과의 관계 (근거) |
|---|---|---|
| MM1 (arXiv:2403.09611) | 사전학습 단계에서 해상도, 인코더, 연결기, 데이터 비율을 절제 | 목록 기재: 연결기 종류보다 해상도와 시각 토큰 수가 중요. Prismatic은 연결기를 MLP로 고정했고 해상도 효과는 같은 방향입니다(`03` §3). 학회 미확인 |
| Idefics2 (arXiv:2405.02246) | 완전 자기회귀 결합과 교차 어텐션 결합을 같은 조건으로 비교 | Prismatic이 §6에서 제외한 축(Perceiver/교차 어텐션)입니다 |
| Cambrian-1 (arXiv:2406.16860, NeurIPS 2024 구두) | 비전 인코더 20여 종 비교(초록), 비전 중심 평가 | 초록은 자기지도·강한 지도·결합을 비교했다고만 합니다. 동결·정렬 단계 결론이 Prismatic과 같은지는 이 분석에서 확인하지 못했습니다(초록만 읽음). `cambrian1/` |
| Eyes Wide Shut? (arXiv:2401.06209) | CLIP이 구별 못 하는 이미지 쌍, DINOv2 특징 섞기 | DINOv2를 보조로 섞는 같은 발상입니다(Prismatic이 이를 설계 축으로 통제 비교). `eyes_wide_shut/` |
| LLaVA, BLIP-2 | 투영기(MLP) 대 Q-Former | 연결기 대안. `llava/`, `blip2/` |
| PaliGemma, Qwen2-VL | 해상도(가변 해상도), 위치 인코딩, 백본 | Prismatic이 다루지 않은 축입니다. `paligemma/`, `qwen2_vl/` |
| Transfusion | 이산 토큰 없는 early fusion | late fusion(패치 투영)의 대안. `transfusion/` |
| OpenVLA (arXiv:2406.09246, CoRL 2024) | Prismatic의 DINOv2+SigLIP 224px 7B를 VLA 백본으로 | 4절 |

- 이 표의 "무엇을 바꿨나"는 `reading_list.md`와 초록 수준의 기재입니다. 본문 결과를 대조한 것은 아닙니다(웹 검색 한도 안에서 초록과 목록 기재만 사용).

## 3. 같은 저장소의 평가 통계 분석과의 관계

| 분석 | 관계 |
|---|---|
| `../eval_error_bars/` (Adding Error Bars to Evals) | 군집 표준오차와 쌍체 차이. 이 논문의 벤치마크 점수는 같은 이미지의 여러 문항(VQAv2, RefCOCO 계열)을 포함해 군집이 있으나 그 효과를 반영하지 않았고, 우리 `03`의 구간도 반영하지 못했습니다 |
| `../little_power/` (With Little Power Comes Great Responsibility) | 이 논문의 12개 벤치마크는 문항 수가 1,222–214,354로 다릅니다. MDE는 문항 수와 불일치율로 정해지고, VSR(1,222문항)은 효과가 3.6점 이상이어야 검출됩니다(`03` §1) |
| `../construct_validity/` (Measuring what Matters) | "12개 벤치마크의 z 평균"이 무엇을 재는지, 퇴화한 VSR, 합친 TextVQA 두 변형의 문제와 닿습니다(`03` §3.5) |

## 4. 로봇 쪽 연결: OpenVLA

- OpenVLA(arXiv:2406.09246)는 Prismatic 7B(SigLIP+DINOv2 융합, Llama-2 7B, 224px)를 백본으로 씁니다. 코드에서 이 설정은 `prism-dinosiglip-224px+7b`(resize-naive, `llama2-7b-pure`, `no-align+fused-gelu-mlp`, 2 epoch)입니다(`prismatic/conf/models.py`). **이 224px 모델은 Prismatic 논문의 표 2–4에 없습니다.** 논문의 융합 모델은 384px(DINOSigLIP)와 336px(DINOCLIP)뿐이고, 224px는 코드에 "Inference-Optimized"라고만 있습니다 (`04` D7).

| 선택 | Prismatic의 결론 | OpenVLA의 선택과 그 분석의 평가 |
|---|---|---|
| 시각 백본 | 미동결이 크게 나쁨(그림 5, 위치 지정 -11 – -41점) | OpenVLA는 **미동결**이 필수라고 씀(§3.4, 부록 D.3: 동결 46.7% 대 미세조정 80.0%, 우리 분석은 방향 일관·소표본으로 "부분") |
| 해상도 | 336/384가 224보다 유의하게 나음(p=6.05e-4) | OpenVLA는 224와 384 차이가 없어 224를 택함(§3.4, 소규모 실험이라 표 없음) |
| 인코더 융합 | DINOv2+SigLIP이 SigLIP보다 낫다(p=0.00164) | OpenVLA 분석에서 DINOv2 추가 효과는 +5.0점이나 p=0.52로 "근거 약함"(C16) |
| 백본 선택 | (해당 없음) | Prismatic이 LLaVA보다 약 10점 높았다는 소규모 실험(§3.4, 표 없음) |

  - VLM 벤치마크에서의 설계 결론이 행동 예측(VLA)으로 그대로 옮겨지지 않는 세 곳(동결, 해상도, 융합 이득의 크기)이 있습니다. 이 불일치가 이 논문을 VLA 백본 선택의 근거로 쓸 때 가장 중요한 단서입니다.

## 5. 공식 저장소와 자료

| 저장소 | 관계 | 라이선스 | 크기, 별, 마지막 push | 기준 커밋 | 상태 |
|---|---|---|---|---|---|
| [TRI-ML/prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms) | 학습 코드, 모델 레지스트리 | MIT (LICENSE 파일, GitHub API도 MIT) | 3,751 KB, 별 1,026, 2024-07-04 | `874c5bb` (2024-05-16) | 정적 읽기 (`repo/`, gitignore) |
| [TRI-ML/vlm-evaluation](https://github.com/TRI-ML/vlm-evaluation) | 평가 하네스 | MIT (LICENSE 파일. GitHub API는 NOASSERTION) | 7.9 MB 작업 트리, 별 140, 2024-09-17 | `d9f5896` (2024-09-12) | 정적 읽기 (`repo/`) |
| 공개 모델 체크포인트 | PRISM 외 학습 모델 전부(논문 주장) | 확인 안 함 | 받지 않음 | | 미확인 |

- 평가 결과 파일(모델별 점수)은 별도 저장소로 공개되지 않았고, 논문 부록 표 2–4가 유일한 공개 수치입니다. 재계산은 이 표로 했습니다.
- 받은 크기: `repo/` 8.5 MB (두 저장소 합). 논문 PDF 9.4 MB.
