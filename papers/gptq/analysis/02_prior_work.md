# 02. 계보, 구성 요소의 출처, 이 저장소의 다른 분석과의 관계

확인일 2026-11-28. arXiv 번호·제목은 arXiv API로 확인했습니다. 후속 논문의 내용은 **초록과 제목 수준 또는 기억에 의한 것**이고 본문은 읽지 않았습니다(표의 "확인 수준" 열). 저장소 정보는 `gh api`로 확인했습니다.

## 1. 구성 요소별 출처

| 구성 요소 | 출처 | GPTQ에서의 역할 |
|---|---|---|
| 2차 정보로 가중치 하나를 제거하고 나머지를 보상 | Optimal Brain Damage(LeCun et al., 1989), Optimal Brain Surgeon(Hassibi & Stork, 1993; 논문 참고문헌) | 식 (2)의 보상 갱신 δ가 OBS 공식. 유도는 `03` §1 |
| OBS를 양자화에 적용 (OBQ) | Frantar & Alistarh, *Optimal Brain Compression*(arXiv:2208.11580). 논문 본문에서는 OBQ로 부름 | GPTQ의 직접 모체. 탐욕 순서와 행별 독립 풀이, O(d_row·d_col³) |
| 층별 재구성 목적(식 1), 격자 고정, 층 입력은 부분 양자화된 모델의 것 | AdaRound(arXiv:2004.10568), BRECQ(arXiv:2102.05426), AdaQuant(arXiv:2006.10518) | 같은 "층별 PTQ" 틀. AdaRound·BRECQ는 SGD로 반올림 방향을 학습하는 계열이라 큰 모델에는 느림 |
| 행별 비대칭 min-max 격자와 RTN | LLM.int8()(arXiv:2208.07339), ZeroQuant(arXiv:2206.01861) | 비교 기준선. 논문이 "RTN = LLM.int8()의 가중치 양자화와 같은 격자"로 정의 |
| 그룹(연속 g개마다 격자) | nuQmm(= LUT-GEMM, arXiv:2206.09557)이 쓰는 가중치 그룹화 | 표 5·7. GPTQ는 격자 매개변수를 갱신된 가중치로 잡음 |
| Cholesky | 수치선형대수의 표준(LAPACK) | Step 3. H⁻¹의 가우스 소거 = Cholesky 인자라는 관찰(`03` §1.4) |
| 같은 저자들의 Hessian 기반 가지치기 | SparseGPT(arXiv:2301.00774, 2023-01) | GPTQ와 같은 층별 Hessian 도구를 가지치기에 적용한 자매 논문(초록) |

## 2. GPTQ 이후의 가까운 방법

| 방법 | 핵심 | GPTQ와 다른 점 | 확인 수준 |
|---|---|---|---|
| SmoothQuant(arXiv:2211.10438, 2022-11) | 활성 이상치를 가중치 쪽으로 옮겨 **W8A8**(활성도 양자화) | GPTQ는 가중치만. 속도는 정수 연산으로 얻음 | 제목·초록 |
| **AWQ**(arXiv:2306.00978, MLSys 2024) | 활성 크기가 큰 채널의 가중치를 보호(스케일링)하는 가중치 4비트 양자화. **역전파·재구성 없이** | GPTQ는 재구성 오차로 보상, AWQ는 채널 스케일. **직접 비교 대상**: `../awq/` (같은 묶음에서 작성 중). 둘 다 보정 데이터가 작고 가중치만 양자화 | 제목·초록 |
| QuIP(arXiv:2307.13304), QuIP#(arXiv:2402.04396) | 가중치와 Hessian의 incoherence를 만드는 무작위 직교 변환 후 적응 반올림(LDLQ), 2비트급 | GPTQ의 층별 2차 보상을 이론적으로 정리한 변형으로 알려짐(초록). 2비트에서 GPTQ보다 나은 결과를 주장 | 제목·초록 |
| OmniQuant(arXiv:2308.13137) | 학습 가능한 클리핑·등가 변환을 블록 단위로 최적화 | GPTQ보다 비싸지만 낮은 비트에서 정확하다고 주장 | 제목·초록 |
| QLoRA(arXiv:2305.14314)의 NF4 | 4비트 정규 분포 격자 위에서 양자화된 기본 모델에 LoRA 학습 | 추론용 PTQ가 아니라 학습용. 격자(NF4)가 균일 격자가 아님 | 제목·초록 |
| llama.cpp의 GGUF 양자화 | 블록별 격자와 여러 조합(k-quants 등), 맥 Metal 실행 | 이 저장소의 24 GB 로컬 기기용 2비트급 양자화 사례 분석이 이 계열의 IQ2_XXS를 다룸 | 이 저장소 문서 |

## 3. 구현·도구 (GitHub API 2026-11-28)

| 저장소 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|
| IST-DASLab/gptq (공식) | Apache-2.0 | 2401 | 2024-03-27 | 논문 코드. OPT·BLOOM·LLaMA, CUDA 3비트 커널. HEAD `2d65066`(2023-07-11) |
| AutoGPTQ/AutoGPTQ | MIT | 5066 | 2025-04-11 | **아카이브됨.** HF transformers 통합의 기반이었음 |
| ModelCloud/GPTQModel | NOASSERTION(복합 라이선스일 가능성, 확인 안 함) | 1296 | 2026-11-28 | 활발히 유지되는 후속 구현 |
| qwopqwop200/GPTQ-for-LLaMa | Apache-2.0 | 3069 | 2024-07-13 | act-order, true-sequential이 먼저 정리된 곳(공식 README가 언급) |
| ggml-org/llama.cpp | MIT | 136459 | 2026-11-28 | GGUF 양자화와 Metal 실행. GPTQ 형식과는 다른 계열 |
| mit-han-lab/llm-awq | MIT | 3668 | 2025-07-17 | AWQ 공식 구현 |

## 4. 이 저장소의 다른 분석과의 관계

| 분석 | 관계 |
|---|---|
| `../flashattention/`, `../pagedattention/`, `../streamingllm/` | 같은 6축(효율·시스템)의 추론 쪽 노드. GPTQ는 **가중치 메모리와 읽기 대역폭**을 줄이고, PagedAttention은 KV 메모리 관리, StreamingLLM은 KV 정책, FlashAttention은 attention의 I/O를 줄임. 24 GB 머신의 메모리 예산을 세 축(가중치 / KV / 활성)으로 나누면 각각의 노드가 됨(`03` §6) |
| `../speculative_decoding/` | 디코딩 지연을 줄이는 다른 길(가중치를 읽는 횟수를 줄임 대 가중치 크기를 줄임). 둘 다 메모리 대역폭 병목이라는 같은 진단에서 출발 (같은 묶음에서 작성 중) |
| `../awq/` | **직접 비교.** AWQ는 "재구성 없이 채널 스케일"이고 GPTQ는 "2차 정보로 보상". 소형 모델 측정은 같은 보정·평가 설정으로 나란히 볼 수 있음 (같은 묶음에서 작성 중) |
| `../attention_sink_emerges/`, `../spike_sparse_sink/` | LLM의 활성 이상치(sink, 거대 활성)가 양자화 오차의 주요 원인이라는 연결. GPTQ는 활성 이상치를 직접 다루지 않고 Hessian의 대각 크기(act-order)와 dampening으로만 반영(`03` §3) |
| `../eval_error_bars/`, `../statistical_precipice/` | 이 분석의 bootstrap·Holm 절차의 출처 |

## 5. 이 논문의 위치

- **계보 한 줄:** OBD/OBS(1989–93) → OBQ·OBC(2022-08, 1억 파라미터급 정확 PTQ) → **GPTQ(2022-10, 고정 순서 + 지연 갱신 + Cholesky로 175B까지)** → 활성 인지(AWQ)·변환(QuIP, OmniQuant)·활성까지 양자화(SmoothQuant) → 실행 형식(GGUF, MLX)과 HF 통합.
- **분류 제안(리뷰용):** 가중치 PTQ를 (a) 반올림만(RTN), (b) **2차 보상형(GPTQ, OBQ)**, (c) 스케일·변환형(AWQ, SmoothQuant, QuIP), (d) 학습형(AdaRound, BRECQ, OmniQuant, QAT)으로 나누면 GPTQ는 (b)의 기준 노드입니다.
