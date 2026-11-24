# 02. 선행 연구, 계보, 관련 분석

## 1. 계보: sink가 어디서 왔고 이 논문이 무엇을 더했나

| 단계 | 논문 | 이 논문과의 관계 |
|---|---|---|
| 관찰 | Bondarenko et al., *Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing* (arXiv:2306.12929) | 이상치가 "잔차를 갱신하지 않으려는 헤드"에서 생긴다는 가설과 clipped softmax·gated attention. 이 논문의 부록 A.3이 인용 |
| 관찰 | Darcet et al., *Vision Transformers Need Registers* (arXiv:2309.16588) | ViT의 고노름 토큰. 같은 현상이 비전에도 있다는 근거 |
| 활용 | Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM, arXiv:2309.17453, ICLR 2024) | "attention sink" 명칭과 첫 토큰 KV 유지. 이 논문의 출발점이며 학습형 sink 토큰(표 4 둘째 행)을 구조적 bias로 재해석한다 |
| 원인 후보 | Sun et al., *Massive Activations in LLMs* (arXiv:2402.17762, COLM 2024); Cancedda, *Spectral Filters, Dark Signals, and Attention Sinks* (arXiv:2402.09221) | 첫 토큰의 큰 은닉 norm과 sink의 관계. 이 논문은 key·value norm이 작고 각도가 sink를 만든다고 재진술(그림 2) |
| **출현 조건** | **이 논문 (arXiv:2410.10781, ICLR 2025)** | 사전학습 설정(데이터량, 학습률, weight decay, 손실, PE, bias, 정규화)을 하나씩 바꿔 "언제 생기나"를 경험적으로 정리 |
| 동시기 | Guo et al., *Active-Dormant Attention Heads* (arXiv:2410.13835) | bigram-backcopy 장난감에서 상호 강화 메커니즘을 이론·실험으로. 논문 본문이 동시 연구로 인용 |
| 동시기 | Ramapuram et al., *Theory, Analysis, and Best Practices for Sigmoid Self-Attention* (arXiv:2409.04431) | 정규화 없는 sigmoid attention의 이론과 학습 안정성. 이 논문 부록 E가 "자세한 분석은 이쪽"이라고 인용 |
| 동시기 | Kaul et al., *From Attention to Activation* (arXiv:2410.17174) | softmax-off-by-one과 OrthoAdam으로 활성 이상치 완화 |
| **이후: 방법** | Qiu et al., *Gated Attention for Large Language Models* (arXiv:2505.06708, NeurIPS 2025 구두) | 값 쪽 헤드별 sigmoid gate로 sink를 없앰. 이 논문의 "정규화 의존을 풀면 sink가 사라진다"를 구조 변경으로 실현한 쪽 |
| **이후: 설명** | Barbero et al., *Why do LLMs attend to the first token?* (arXiv:2504.02732) | sink를 과혼합(over-mixing) 회피 장치로 해석하고 문맥 길이·깊이·패킹의 영향을 봄. 학회는 COLM 2025라는 2차 자료 한 곳뿐(공식 확인 안 함) |
| **이후: 해부** | Peng et al., *How Attention Sinks Emerge in LLMs: An Interpretability Perspective* (arXiv:2603.06591) | 첫 토큰 sink가 깊은 층에서 먼저 생기고 앞 두 층의 회로로 수렴한다는 기계적 해석(초록 수준만 확인) |
| **이후: 종합** | *The Spike, the Sparse and the Sink* (arXiv:2603.05498, ICML 2026) | massive activation과 sink의 관계를 해부. 이 저장소 28번 `spike_sparse_sink/`에서 분석 예정(링크만) |

- 이 논문이 새로 보탠 것: (1) 출현 시점 곡선(그림 4), (2) 데이터량·무작위화·고정 토큰으로 위치가 데이터가 정한다는 실험(표 10), (3) K bias로 sink를 "bias 위치"로 옮기는 설계 공간(표 4, 5, 12, 13), (4) 정규화 척도와 커널 변경(표 6, 14, 15).
- 이 논문이 새로 보태지 않은 것: sink 현상 자체, massive activation의 존재, softmax-off-by-one의 완화 효과(부록 A.3이 앞선 연구로 인용).

## 2. 구성 요소별 출처

| 요소 | 출처 |
|---|---|
| 학습 코드 | TinyLlama(lit-gpt), RegMix (Liu et al., 2024) |
| 60M 설정 | RegMix의 d=768, L=10 설정(절 4) |
| 데이터 | Pile (17개 도메인 혼합, `regmix-data-sample`) |
| 평가 | Pile-CC 검증 손실(RegMix와 같은 관행) |
| 지표 | 논문이 정의. 임계값은 StreamingLLM의 attention 시각화를 수량화한 것 |
| 커널 변형 | 선형 attention(Katharopoulos et al., 2020)의 elu+1 |

## 3. 이 저장소의 관련 분석과의 관계

이 저장소에는 아직 sink를 직접 다루는 분석이 없다.

- `spike_sparse_sink/` (진행 순서 28번, 작성 중): massive activation과 sink의 관계. 이 논문의 "KV 크기 작음 + 각도 큼" 설명(그림 2)과 맞는지 확인할 대상으로 링크만 건다.
- 같은 논문 군(attention 내부 해석) 안의 기준 역할: 이 논문의 임계값 지표와 "key bias로 작동" 정의는, 다른 논문이 "이 토큰이 sink인가"를 말할 때 쓸 수 있는 판정 기준(지표 α_1, key 각도, value norm 비, 정규화 의존)이다.

## 4. 로봇 쪽 분석과의 관계


- 그 논문(flow 기반 VLA)은 행동 청크 양 끝 토큰에 몰리는 attention을 sink로 해석하지 말고 **RoPE가 만든 위치 효과**로 보라고 주장한다. 근거는 (i) 상위 층 RoPE를 빼면 끝점 질량이 균일 기준 0.040으로 돌아가고, (ii) 끝점 value의 norm 비가 약 1.0이며 출력 기여가 평균 이상이어서 sink의 조건(작은 value, 작은 출력 기여)을 만족하지 않는다는 것이다.
- 이 논문의 기준으로 보면 그 주장은 일관된다. 이 논문이 말하는 sink는 **작은 value norm(그림 2, 11–14), massive activation, 데이터가 정한 위치(표 10)**의 묶음이다. 끝점 질량이 RoPE 위치 효과이고 value가 작지 않다면 이 정의의 sink가 아니다.
- 다만 이 논문은 위치를 정하는 요인으로 "데이터와 손실"을 말하고 PE는 영향이 없다고 하는데(표 3), GPT2-XL에서는 첫 위치 임베딩 p_1이 sink 위치를 정했다(표 7). 즉 PE가 sink를 **만들지는 않지만 위치를 고정할 수는 있다**. 로봇 쪽 결과의 "RoPE가 만든 위치 효과"는 이 후자의 경우와 가까운 형태일 수 있다(추정이며 검증하지 않았다).
- 같은 종류의 비교를 공개본에도 남기기 위해, 위 비교는 해당 논문 이름 없이 "RoPE 위치 효과로 설명된 끝점 질량" 사례로 03의 §4에도 적는다.

## 5. GitHub와 자료 (확인 2026-11-24)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 공식 코드 | https://github.com/sail-sg/Attention-Sink | MIT | 164 | 2025-07-08 | 크기 6.4 MB. 결과·로그 없음 |
| 이 논문의 ICLR 페이지 | https://iclr.cc/virtual/2025/poster/30847 | - | - | - | 형식 표기 Poster, OpenReview id 78Nn4QJTEN |
| 데이터 `regmix-data-sample` | HF `sail/regmix-data-sample` | 미확인 | - | - | 약 20 GB(README). 받지 않았다 |
| 학습 코드 근간 | TinyLlama(`jzhang38/TinyLlama`), RegMix(`sail-sg/regmix`) | 각각 Apache-2.0 계열(미확인) | - | - | 받지 않았다 |

## 6. 학회 확인

- ICLR 2025. `proceedings.iclr.cc/paper_files/paper/2025/hash/f1b04face60081b689ba740d39ea8f37-Abstract-Conference.html`이 같은 제목으로 존재한다(2026-11-24 확인).
- `iclr.cc/virtual/2025/poster/30847`은 "ICLR Poster"로 표기한다. arXiv comment의 "ICLR 2025 (Spotlight)"는 공식 페이지에서 확인되지 않았다.
