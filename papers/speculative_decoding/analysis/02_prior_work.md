# 02. 계보와 이웃 연구

## 1. 이 논문이 서 있는 곳

```
투기적 실행(Burton 1985; 분기 예측)  ──┐
                                      ├─> 확률적 투기적 실행 + speculative sampling (Leviathan et al., 2022-11)
적응형 계산 / 조기 종료 ────────────────┤        └─ 같은 시기: Chen et al. 2023-02 (arXiv:2302.01318)
Blockwise Parallel Decoding (Stern 2018) ┘
        │
        └─ 후속(초안 만들기와 검증을 바꾼 것): SpecInfer(트리), Medusa(여러 헤드), EAGLE(특징 수준 초안),
           Lookahead, 자기 투기, 초안 증류, 온라인 초안, 다중 초안 최적 수송(SpecTr)
```

핵심 기여는 알고리즘 자체보다 **"어떤 초안 분포 q라도 출력 분포는 p 그대로"라는 증명 가능한 구성**(부록 A.1)과, 이득을 α(Thm 3.5)와 c(Def 3.7)로 나눠 쓴 분석(Thm 3.8)입니다. 이후 문헌의 대부분은 두 부품 중 q(초안을 어떻게 만드나)나 검증(한 줄을 어떻게 병렬로 확인하나)을 바꿨고, 수락 규칙은 그대로 씁니다(`04` C1, C2).

## 2. 구성 요소별 출처

| 요소 | 출처 | 이 논문과의 관계 |
|---|---|---|
| 투기적 실행 | Burton 1985, Hennessy & Patterson 2012 (논문 인용) | 프로세서의 분기 예측에서 가져와 확률 설정으로 일반화(§1) |
| 병렬 디코딩 | Stern et al., arXiv:1811.03115 (NeurIPS 2018, Blockwise Parallel Decoding) | 여러 토큰을 병렬로 디코딩하지만 greedy 전용이고 별도 학습 필요(논문 §5) |
| 입력 복사 | Sun et al. 2021, SAD (arXiv:2106.04970) | 입력을 출력으로 복사하는 경우만, 일반 근사 모델 불가(논문 §5) |
| 수락·재표집 규칙 | 이 논문 §2.3 (Chen et al.이 같은 규칙을 독립 유도) | p′ = norm(max(0, p − q)) |
| 큰 모델의 메모리 병목 관찰 | 논문 §1, §3.4 | 산술이 아니라 대역폭이 병목이라는 가정이 Thm 3.8의 "병렬 검증은 공짜"의 근거. `flashattention`이 이 병목을 attention에서 따로 분석 |
| α의 정의 | Thm 3.5, Cor. 3.6 | Σ min(p, q) = 1 − TV. 총변동 거리와 같은 양을 새 이름 D_LK로 부름 |

## 3. 같은 시기·이후 문헌 (arXiv API로 제목·날짜 확인)

| 연구 | 무엇을 바꿨나 | 요점 |
|---|---|---|
| Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv:2302.01318 (2023-02-02, DeepMind) | **동시 독립 작업.** 같은 수락·재표집 규칙(초록: "modified rejection sampling")을 쓴 Chinchilla 70B 실험 | 분산 설정에서 2–2.5배, 분포는 "하드웨어 수치 범위 안에서" 보존(초록). Leviathan 논문도 §6에서 "독립 구현이 2–2.5배"라고 인용. 둘의 관계: 알고리즘은 같고, Leviathan은 γ·c·α 분석과 T5 실측, Chen은 대형 모델 분산 실측 |
| Miao et al., *SpecInfer*, arXiv:2305.09781 (ASPLOS 2024) | 초안을 **토큰 트리**로 만들고 트리 attention으로 한 번에 검증 | 단일 줄보다 수락 길이가 늘어남 |
| Spector & Ré, *Staged Speculative Decoding*, arXiv:2308.04623 | 초안 모델 자체를 또 다른 초안으로 가속(논문 §6의 계층 아이디어) | 소규모 배치·온디바이스 |
| Zhang et al., *Draft & Verify*, arXiv:2309.08168 (ACL 2024) | 별도 초안 없이 같은 모델의 층을 건너뛰어 초안 | 추가 모델·학습 불필요 |
| Zhou et al., *DistillSpec*, arXiv:2310.08461 | 초안을 목표 분포로 **증류**해 α를 올림 | 논문 §6의 "Mq를 α로 직접 최적화"에 해당 |
| Liu et al., *Online Speculative Decoding*, arXiv:2310.07177 | 서빙 중 초안을 온라인으로 갱신 | 분포 이동 대응 |
| Sun et al., *SpecTr*, arXiv:2310.15141 (NeurIPS 2023) | 초안 여러 줄을 **최적 수송**으로 검증하는 수락 규칙 | 단일 초안 규칙은 다중 초안에서 최적이 아님 |
| Cai et al., *Medusa*, arXiv:2401.10774 | 목표 모델에 **여러 디코딩 헤드**를 붙여 초안 모델을 없앰 | 헤드를 학습해야 하고, 초록 기준 초안 모델 확보·유지 문제를 피함 |
| Li et al., *EAGLE*, arXiv:2401.15077 | 토큰이 아니라 **특징 수준**에서 자기회귀로 초안 | 초록: 특징 수준의 불확실성을 이전 토큰으로 해소 |
| Fu et al., *Lookahead Decoding*, arXiv:2402.02057 | 초안 모델 없이 야코비 반복과 n-gram 풀로 초안 | 논문 §3.6의 n-gram·복사형 "무비용 근사 모델"의 후속 |

이 표의 요점은 위 초록에서 읽은 범위만 적었습니다(본문은 읽지 않았고, 수치 비교는 하지 않았습니다). 이후 문헌이 이 논문의 주장을 확인·수정·반박한 부분은 `04`에 정리했습니다.

## 4. 이 저장소·이웃 저장소와의 관계

| 대상 | 관계 |
|---|---|
| `../flashattention/` | 같은 "메모리 병목" 관찰에서 출발. FlashAttention은 정확한 계산에서 HBM 접근을 줄이고, 이 논문은 정확한 분포에서 목표 모델 호출 수를 줄인다. 둘 다 정확성을 지키는 효율화이고 서로 곱해서 쓴다 |
| `../streamingllm/` | 근사(윈도 + 싱크)로 긴 스트림을 처리. 출력이 달라지는 대신 메모리가 고정된다는 점에서 이 논문과 정반대 설계 선택(분포 보존 vs 메모리 상한) |
| `../pagedattention/` | 서빙에서 KV 캐시 관리. 투기적 디코딩은 거절된 위치의 KV를 폐기해야 하므로 캐시 관리와 맞물린다. 같은 묶음에서 작성됨 |
| `../gptq/`, `../awq/` | 가중치 양자화로 메모리 이동량 자체를 줄이는 보완적 방법(논문 §5: "함께 쓸 수 있음"). |
| `../ddpm/`, `../ddim/`, `../classifier_free_guidance/` 등 확산 계열 | 확산 샘플러의 병렬화는 자기회귀와 다른 문제(단계 간 의존)지만 "표본 분포를 바꾸지 않는 가속"이라는 틀은 비교 대상이다. |

## 5. GitHub·자료 표

공식 구현은 공개되어 있지 않은 것으로 보입니다(`gh search repos`와 논문 본문 확인, `04` 서두). 아래는 독립 구현·이후 구현입니다(2026-11-28 `gh api`).

| 저장소 | 성격 | 라이선스 | 별 | 마지막 push |
|---|---|---|---|---|
| romsto/Speculative-Decoding | 이 논문의 독립 구현 | MIT | 117 | 2024-12-02 |
| feifeibear/LLMSpeculativeSampling | 독립 구현(저장소 설명이 Leviathan 논문 제목) | Apache-2.0 | 925 | 2024-08-22 |
| huggingface/transformers | assisted generation(투기적 디코딩) 구현 포함 | Apache-2.0 | 169,309 | 2026-11-27 |
| vllm-project/vllm | 서빙 엔진, 투기적 디코딩 지원 | Apache-2.0 | 97,716 | 2026-11-28 |
| FasterDecoding/Medusa | Medusa 헤드 | Apache-2.0 | 2,776 | 2024-06-25 |
| SafeAILab/EAGLE | EAGLE | NOASSERTION | 2,586 | 2026-02-20 |
| hao-ai-lab/LookaheadDecoding | Lookahead | Apache-2.0 | 1,345 | 2025-03-06 |

transformers와 vLLM에서 이 논문의 알고리즘이 어떻게 구현되어 있는지는 코드를 읽지 않았습니다.
