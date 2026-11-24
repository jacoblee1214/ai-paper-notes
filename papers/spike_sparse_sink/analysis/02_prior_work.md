# 02. 계보와 관련 연구

확인 방식: arXiv 번호와 제목은 abs 페이지(2026-11-24)에서 확인했습니다. 학회·내용은 논문 본문 참고문헌과 제 기억에 의존한 곳이 있고, 그런 곳은 "미확인"으로 적었습니다. 발표 이후 논문은 제목만 확인했고 내용은 읽지 않았습니다.

## 1. 계보

| 시기 | 연구 | 이 논문과의 관계 |
|---|---|---|
| 2021–22 | Kovaleva et al. "BERT Busters"(ACL-IJCNLP 2021), Dettmers et al. LLM.int8()(NeurIPS 2022, arXiv:2208.07339), Puccetti et al.(EMNLP 2022) | outlier 차원·outlier feature의 기원. 이 논문 §5가 "massive activation의 선조"로 인용. 원 연구의 관심은 양자화 |
| 2023 | Bondarenko et al. "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing"(NeurIPS 2023) | outlier를 "head가 아무것도 안 하려는 시도"로 해석(clipped softmax, gated attention). 이 논문의 "sink = 학습된 gate" 해석과 같은 계열 |
| 2023 | Xiao et al. StreamingLLM, arXiv:2309.17453 (ICLR 2024) | sink 발견·명명. 첫 토큰 KV 유지로 스트리밍. 이 논문은 "sink가 단거리 의존을 돕는다"는 StreamingLLM 계열 주장을 표 8로 검증했다고 서술 |
| 2023 | Darcet et al. Vision Transformers Need Registers, arXiv:2309.16588 (ICLR 2024) | ViT에서 고노름 토큰(sink 유사)이 생기고 register 토큰으로 해소. 이 논문 본문 참고문헌에는 없음 |
| 2024 | Sun et al. Massive Activations in LLMs, arXiv:2402.17762 (COLM 2024) | massive activation 정의·층별 양상·sink와의 동시 출현을 보고. 이 논문의 "spike"의 출발점이며 5가지 성질(i)–(v)는 이 연구의 관찰 정리 |
| 2024 | Guo et al. Active-Dormant Attention Heads, arXiv:2410.13835 | extreme-token 현상의 메커니즘(작은 모델 이론). 이 논문 §5가 인용 |
| 2024–25 | Yu et al. Super Weight(2024), Oh et al. House of Cards(arXiv:2410.01866) | 특정 가중치가 massive activation을 만든다. 이 논문의 "step-up 블록의 고이득 가중치"와 같은 관찰 |
| 2024 | Kaul et al. From Attention to Activation, arXiv:2410.17174 | 적응 최적화 동역학을 원인으로 보고 OrthoAdam 제안. 이 논문이 "joint emergence의 다른 설명"으로 인용 |
| 2025 | Gu et al. When Attention Sink Emerges in LMs, arXiv:2410.10781 (ICLR 2025) | sink 출현 조건(최적화, 데이터, 손실 함수, 모델 구조)을 사전학습 실험으로 조사. **이 논문의 sink ratio 정의의 출처**(식 27–28). 폴더 [attention_sink_emerges](../attention_sink_emerges/)(27번)가 이 논문을 다룸 |
| 2025 | Qiu et al. Gated Attention for LLMs, arXiv:2505.06708 (NeurIPS 2025) | SDPA 뒤 head별 sigmoid gate가 sink와 massive activation을 없앰. 이 논문의 표 7이 같은 설계를 재현·확장(gate 조건 변인) |
| 2025 | Barbero et al. Why do LLMs attend to the first token?, arXiv:2504.02732 | 첫 토큰 sink를 과혼합(over-mixing) 방지로 설명. 이 논문 본문 참고문헌에는 없음 |
| 2025 | Queipo-de Llano et al. Attention Sinks and Compression Valleys, arXiv:2510.06477 | massive activation이 compression valley를 만든다는 통합 이론. 이 논문이 "보완"으로 인용 |
| 2025 | Owen et al. A Refined Analysis of Massive Activations, arXiv:2503.22329 | DynamicTanh 등 원소별 변환이 massive activation을 막는다고 보고. 이 논문 표 5의 DynamicTanh 결과가 같은 방향 |
| 2026-03 | **이 논문** (ICML 2026) | pre-norm이 동시 출현을 가능케 하는 구조적 조건이라는 주장, 개입 실험 |

## 2. 가까운 대안 설명과 비교

| 설명 | 주장 | 이 논문과의 차이 |
|---|---|---|
| softmax 합=1 제약 | sink는 "남는 attention을 버릴 곳" (Miller 2023 "off by one", Gu et al. 2025 등) | 이 논문은 sink를 attention 공간 차원, 학습 문맥 분포, 조건부 gate 유무로 설명하고 softmax 자체는 조작하지 않음 |
| 위치·마스크 기반 | 첫 토큰은 인과 마스크 때문에 모든 질의가 볼 수 있는 유일한 토큰 (Barbero 2025 계열) | 이 논문은 표 2에서 위치 0이 토큰 종류보다 중요하다고 보이지만 위치 인코딩을 조작한 실험은 없음 |
| 최적화 동역학 | Adam 계열의 구조 때문 (Kaul 2024) | 표 3은 β2, 학습률, weight decay에 spike가 민감함을 보이나 최적화기를 바꾼 실험은 없음 |
| 정규화·잔차 누적 | pre-norm이 값을 무한히 쌓게 한다 | **이 논문의 핵심.** 표 5의 Sandwich, DynamicTanh |

## 3. 구성 요소별 출처

| 요소 | 출처 |
|---|---|
| sink ratio | Gu et al. 2025 (식 27–28) |
| step-up/step-down, 5 성질 | Sun et al. 2024 (그림 1이 같은 방법으로 상위 3 채널 추적) |
| 방향성 이차 증폭기 | 이 논문(Yu et al. 2024의 collinear gate/up 행 관찰을 이차형식으로 정리) |
| Sandwich norm | Ding et al. CogView(NeurIPS 2021) |
| QKNorm | Olmo 3 보고서(arXiv:2512.13961)를 인용 |
| DynamicTanh | Zhu et al. CVPR 2025 (Transformers without Normalization). 저자 중 Zhu·Chen 등이 겹침 |
| gated attention 변형 | Qiu et al. 2025 |
| 문맥 길이 분포 실험 | Xiao et al. 2024(DuoAttention, arXiv:2410.10819) 동기 |

## 4. 이 저장소·이웃 저장소의 관련 분석

- [attention_sink_emerges](../attention_sink_emerges/): sink 출현 조건 연구(Gu et al., ICLR 2025). 이 논문은 그 sink ratio를 쓰고 그 결과(최적화·데이터·구조 의존)를 확장합니다. 두 분석의 세부 비교는 하지 않았고 링크만 둡니다.
- flow 기반 VLA의 행동 청크 끝점 attention을 분석한 로봇 쪽 연구는 "attention을 많이 받는 위치가 곧 sink는 아니다"라는 주의를 줍니다. 이 논문의 sink 판정(spike 동반, 거의 상수인 key)과 그 연구의 판정(value 노름, 출력 기여)을 함께 써야 위치 효과와 sink를 구분할 수 있다는 것이 우리의 정리입니다.
- 이 저장소 평가 쪽 폴더들과는 직접 관계가 없습니다.

## 5. 발표 이후 연구 (제목만 확인, 내용 미독)

| arXiv | 제목 | 이 논문과 관련 추정 |
|---|---|---|
| 2603.17771 | Attention Sinks Induce Gradient Sinks: Massive Activations as Gradient Regulators in Transformers | 같은 현상의 학습 동역학 설명. 이 논문의 "암묵적 파라미터" 해석과 비교 대상 |
| 2604.14722 | A Mechanistic Account of Attention Sinks in GPT-2: One Circuit, Broader Implications for Mitigation | 작은 모델에서 회로 수준 설명 |
| 2610.00423 | The Life Cycle of a Massive Activation: Stochastic Birth, Weight-Decay-Driven Growth, and Competitive Consolidation | weight decay와 massive activation. 이 논문 표 3의 weight decay 0에서 spike 12,275와 연결될 수 있음 |
| 웹 검색 결과에만 제목이 나온 것 | 2609.35630 (Writing with Read-Blindness), 2609.31956 (MLP Gate Thresholds) | 제목만 보고 관련 후보로 적음 |

## 6. 자료 표 (확인 2026-11-24)

| 자료 | 링크 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| 논문 (arXiv v1) | arxiv.org/abs/2603.05498 | CC0 1.0 (공개 도메인 헌정) | - | 2026-03-05 제출 | abs 페이지의 라이선스 링크 `creativecommons.org/publicdomain/zero/1.0/` |
| PMLR | proceedings.mlr.press/v306/sun26d.html | - | - | 2026-09-29 | 제목이 "Anatomy of Massive Activations and Attention Sinks"로 줄어 있음 |
| ICML 포스터 | icml.cc/virtual/2026/poster/63610 | - | - | - | 검색 결과 요약 기준 2026-07-08 포스터, 세션 시각 표기는 일관되지 않음 |
| 공식 코드 | github.com/savinasun/SpikeSparseSink | 라이선스 파일 없음 | 3 | 2026-03-06 | README가 "§4 ablation의 공식 PyTorch 구현"이라고 밝힘. 18 KB, 클론 후 468 KB. 학습 코드만 |
