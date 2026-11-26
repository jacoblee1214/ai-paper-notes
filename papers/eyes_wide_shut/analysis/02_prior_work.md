# 02 계보와 관계: Eyes Wide Shut? (arXiv:2401.06209)

이 문서의 arXiv 번호와 제목은 arXiv API로 대조했습니다(2026-11-26). 후속 연구의 내용은 이 저장소의 완료된 분석(읽기만)과, `cambrian1/` 원문 PDF의 MMVP 언급을 읽은 것입니다. 웹 검색은 쓰지 않았습니다.

## 1. 계보

| 계열 | 논문 | 이 논문과의 관계 |
|---|---|---|
| CLIP의 구성성·바인딩 한계 | Winoground(2204.03162, CVPR 2022), "When and why VLMs behave like bags-of-words"(ARO, 2210.01936, ICLR 2023), SugarCrepe(2306.14610, NeurIPS 2023) | 논문 §5가 인용. 이 연구들은 **텍스트 쪽**(바인딩, 어순)의 실패를 다루고, 이 논문은 **이미지 쪽**(같은 임베딩인 다른 이미지)의 실패를 다룹니다. 둘 다 CLIP이 하나의 벡터로 장면을 압축해서 생기는 한계라는 같은 뿌리입니다. 새 벤치마크 대부분이 사람이 만든 정답에 기대는 점도 같습니다 |
| 같은 저자의 실패 자동 발굴 | Mass-Producing Failures of Multimodal Systems with Language Models(2306.12105, NeurIPS 2023, Tong·Jones·Steinhardt) | 논문 [57]. "임베딩 공간의 잘못된 일치(erroneous agreement)로 실패를 대량으로 찾는다"는 발상의 원형. 텍스트 임베딩에서 이미지 임베딩으로 옮긴 것이 CLIP-blind 쌍 |
| 평가 방법의 선행 | ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy(2311.09215), POPE(2305.10355), HallusionBench(2310.14566) | 논문 [61], [27], [28]. ImageNet 정확도 너머의 평가가 필요하다는 주장(§6) |
| 비전 인코더의 대안 | DINOv2(2304.07193, `dinov2/` 예정), SLIP, MAE, MoCo v3(논문 [39][18][17]) | MoF의 재료. DINOv2는 선별 기준(§2.1)과 MoF(§4)에 모두 쓰임 |
| MLLM 기준선 | LLaVA, LLaVA-1.5(2310.03744), InstructBLIP(2305.06500), BLIP-2(`blip2/` 예정) | 평가 대상이자 MoF의 틀(§4.1) |

## 2. 같은 시기의 "두 인코더 결합" (겹치거나 먼저 나온 것)

논문은 CLIP과 DINOv2 특징을 섞는 다른 연구를 인용하지 않습니다(GitHub 이슈 #18은 COMM(arXiv:2310.08825)이 2023-10에 이미 같은 결합을 제안했다고 지적, 우리는 COMM 원문을 읽지 않았고 외부 보고입니다).

| 방법 | 결합 위치 | 결과 | 출처 |
|---|---|---|---|
| **Eyes Wide Shut** I-MoF | 토큰 **교대 배치**(어댑터 뒤, 시퀀스 길이 2배) | 224²에서 MMVP +3.3(LLaVA-1.5), 표본 오차 안(`03` §6) | 이 논문 §4.3 |
| **Prismatic VLMs** | 패치 특징을 **채널 방향으로 이어 붙임**(패치 수 그대로, 투영기 입력만 2배) | DINOv2+SigLIP은 유의하게 좋아지고 DINOv2+CLIP은 아님(p=0.37) | `prismatic_vlms/` 그림 7 |
| **Cambrian-1** SVA | 학습 가능한 질의가 여러 인코더 특징에 교차 주의 | CV-Bench 등 시각 중심 벤치마크 | `cambrian1/` (작성 중, 링크만) |
| Eagle(2408.15998), SPHINX(2311.07575), DeepSeek-VL(2403.05525) | 인코더 혼합 | 이 문서에서 읽지 않음 | 링크 대상 아님 |

핵심 대조: **Prismatic은 DINOv2+CLIP 융합이 이득 없다고 보고**(`prismatic_vlms/` 03의 검정에서 p=0.37)한 반면 이 논문은 같은 두 인코더(CLIP ViT-L + DINOv2 ViT-L)로 MMVP에서 이득을 봅니다. 두 연구는 벤치마크가 다르므로 모순이 아닙니다. Prismatic의 벤치마크는 VQA류 평균이고 이 논문은 CLIP-blind 쌍으로 만든 시험입니다. **DINOv2는 선별 때 이미 쓰였으므로 이 논문의 이득은 DINOv2에 유리한 시험에서의 이득**이기도 합니다(`03` §3). 두 논문을 함께 읽을 때 이 구분이 중요합니다.

## 3. 이 논문을 쓴 후속 연구와 평가 연구

| 연구 | MMVP 사용 | 출처 |
|---|---|---|
| PaliGemma(2407.07726) | MMVP 47.3(448px), 45.3(224px)이 GPT-4V 38.7·Gemini 40.7보다 높다고 "SOTA by a large margin" | `paligemma/` 03 §3.4: Gemini 대비 Fisher p=0.30. 이 문서 `03` §2에서 150쌍 구간으로 다시 확인 |
| Cambrian-1(2406.16860) | MMVP를 "시각 중심" 평가 범주에 포함. 본문에 "MMVP는 시각을 끈 성능이 무작위보다 눈에 띄게 낮다"는 서술이 있고 CV-Bench를 새로 만들며 MMVP(300문항)보다 8.8배 크다고 함 | `cambrian1/paper/paper.pdf`를 읽음. 분석은 작성 중 |
| Qwen2-VL, Prismatic, Transfusion | 원문에 MMVP·Eyes Wide Shut 언급 없음 | 각 폴더 `paper.txt` 검색 |

"시각을 끈 성능이 무작위보다 낮다"는 Cambrian-1의 관찰은 `03` §1의 구조 분석(이미지를 보지 않는 응답은 쌍 단위에서 0점, 상한 25%)과 같은 현상입니다.

## 4. 이 저장소의 평가·통계 분석과의 관계

| 폴더 | 관계 |
|---|---|
| `little_power/` | 짝지은 비교의 MDE ≈ 2.80·√(불일치율/n). `03` §2, §6에서 150쌍 MMVP에 그대로 적용 |
| `eval_error_bars/` | 군집 표준오차와 짝지은 차이. MMVP의 한 쌍(두 질문)은 군집이고, 쌍 단위 채점이 군집 처리를 대신함 |
| `construct_validity/` | "이 시험이 재려는 구인을 재는가". MMVP는 **CLIP ViT-L 임베딩이 구별하지 못하는 쌍**이라는 선별 때문에 "일반 시각 능력"이 아니라 "특정 인코더의 맹점"을 재는 시험에 가깝고, 사람 정답이 쉬운 것(95.7%)은 반증이 아니라 선별이 사람에게 불리하지 않았다는 확인(`03` §3, §7) |
| `llm_as_judge/` | 공식 채점 스크립트가 GPT-4를 판정자로 씀(`gpt-4-0314`, 온도 0.2, 예/아니오). 위치·장황함 편향 완화(순서 교환, 참조 답)가 없고 판정 일치율 검증도 없음(`04` D2) |
| `paligemma/` | MMVP 점수의 출처 중 하나 |
| `prismatic_vlms/` | 융합 방식(이어 붙이기)이 다름. 위 §2 |

## 5. 로봇 쪽 VLA와의 관계

OpenVLA의 백본(Prismatic, DINOv2+SigLIP)은 이 논문의 MoF와 같은 계열의 선택입니다. OpenVLA 부록의 "DINOv2 추가" 절제(+5.0%p)도 재계산하면 표본 오차 안이라 이 논문의 I-MoF 이득(`03` §6.4)과 증거 수준이 같습니다. 이 저장소가 아닌 로봇 분석 쪽에 겹치는 논문은 없습니다.

## 6. GitHub·자료 표

| 항목 | 값 |
|---|---|
| 코드 `tsb0601/MMVP` | 별 364, 포크 12, 마지막 push 2024-01-27, 커밋 `7635005`, 이슈 30건(열린 것 다수), **루트 LICENSE 파일 없음**(README는 MIT라 쓰지만 GitHub 감지 없음, 하위 `LLaVA/LICENSE`는 Apache-2.0), 크기 약 1.4 MB(clone 3.6 MB) |
| 데이터 `MMVP/MMVP` | 라이선스 MIT(카드), 300장 224² 이미지 + `Questions.csv`(300행), 약 3.6 MB, 마지막 수정 2024-06-01 |
| 데이터 `MMVP/MMVP_VLM` | MIT(카드), 270장 224² 이미지 + `Questions.csv`(270행), 마지막 수정 2024-01-10 |
| 모델 `MMVP/MoF_Models` | I-MoF LLaVA-1.5 13B 가중치(README). 받지 않음 |
| 프로젝트 | tsb0601.github.io/mmvp_blog |
| 학회 | CVPR 2024 구두. CVF Open Access 페이지, DOI 10.1109/CVPR52733.2024.00914 |
