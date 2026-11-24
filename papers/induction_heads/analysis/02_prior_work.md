# 02. 계보와 관련 자료: In-context Learning and Induction Heads (arXiv:2209.11895)

이후 문헌의 서술은 **arXiv 초록과 메타데이터(arXiv API, 확인일 2026-11-24)**에 근거합니다. 본문까지 읽지 않았으므로, 초록에 없는 세부 주장은 쓰지 않았습니다. 학회 표기는 arXiv의 comment·journal-ref 필드와 Crossref에서 확인한 것만 적었습니다.

## 1. 계보: 이 논문이 기대는 것

| 구성 요소 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 잔차 흐름, QK·OV 회로, head 합성(Q-, K-, V-composition), attention-only 1·2층 모델의 해부 | Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*(Transformer Circuits Thread, 학회지 아님) | induction head의 발견과 "이전 토큰 head + K-composition" 설명을 그대로 가져옴. 이 논문의 Argument 5는 이 해부의 요약(L769–L817) |
| ICL을 "문맥 위치가 뒤일수록 손실이 줄어드는 정도"로 보는 정의 | Kaplan et al. 2020, scaling laws (arXiv:2001.08361) | ICL 점수의 근거. 1층 모델이 스케일링 법칙에서 벗어난다는 관찰도 가져와 induction head 부재로 설명(L1120–L1124) |
| few-shot ICL의 부상 | Radford et al. 2019, Brown et al. 2020 (GPT-3) | 미시적 정의로 소개하고 쓰지는 않음(L142–L162) |
| 학습 궤적을 함수 공간(출력 벡터)에서 PCA로 비교 | Erhan et al. 2010 | per-token loss 벡터의 PCA(L251–L264) |
| 학습 중 갑작스러운 능력 출현 | Brown et al. 2020, Power et al. 2022(grokking), Nakkiran et al.(이중 하강) | 단계 변화를 grokking에 가장 가깝다고 보고, "처음으로 메커니즘으로 설명한 단계 변화"라고 주장(L1143–L1148) |
| 사전 head 분석(이전 토큰 head 등) | Voita et al. 2019, Clark et al. 2019 | "이전 토큰 head는 여러 모델에 보편적"(L1191–L1196) |
| 위치 정보를 잔차에 더하지 않는 방식 | Press et al. 2020 (Shortformer) | 소형 모델의 위치 처리(L892–L894, 각주 25) |

이 논문이 **새로 한 것**은 (1) induction head를 구조 분석이 아닌 행동(반복 무작위 열에서의 prefix matching과 copying)으로 재정의해 큰 모델에도 쓸 수 있게 한 것, (2) ICL 증가와 head 형성이 같은 학습 구간에 일어난다는 거시 관찰, (3) smeared key라는 구조 개입, (4) head ablation 데이터(소형 모델), (5) 13B 모델에서 번역·패턴 매칭을 하는 head의 사례입니다.

## 2. 가까운 대안 설명

| 설명 | 요지 | 이 논문과의 관계 |
|---|---|---|
| 메타 학습 / 암묵적 베이즈 / 경사하강 | ICL은 문맥으로 잠재 과제를 추론하거나(Xie et al. 2021), 순전파에서 경사하강을 흉내 낸다(von Oswald et al., arXiv:2212.07677) | 논문은 Xie et al.과의 긴장을 "합성 데이터라 transformer가 이점이 없을 것"으로 정리(L1088–L1094). von Oswald의 구성은 선형 회귀 같은 단순 과제에서 나왔고 자연 텍스트의 ICL 점수와 직접 맞닿지는 않음 |
| 위치 임베딩 기반 induction("pointer arithmetic") | GPT-2에서 Q-composition과 위치 임베딩으로 구현(L805–L817) | 같은 행동의 다른 구현. 논문이 직접 언급 |
| function vector head | 일부 head가 과제를 요약한 벡터를 운반(Todd et al., arXiv:2310.15213, ICLR 2024) | 이 논문 이후의 대안. 04 C4 |

## 3. 이후 문헌 (논문이 연 연구선, 초록 기준)

학회 표기는 arXiv 메타데이터 필드에서 확인한 것이고, 비어 있으면 "미확인"입니다.

| 줄기 | 논문 (arXiv, 학회 표기) | 초록이 말하는 것 |
|---|---|---|
| 학습 역학 이론 | Bietti et al., *Birth of a Transformer: A Memory Viewpoint* (2306.00802, NeurIPS 2023) | 합성 설정의 2층 transformer에서 전역 bigram은 빨리, in-context bigram용 induction head는 더 느리게 학습. 가중치 행렬을 연상 기억으로 해석 |
| | Edelman et al., *The Evolution of Statistical Induction Heads* (2402.11004, NeurIPS 2024로 Crossref에서 확인) | 마르코프 체인 과제에서 균등 → unigram → bigram 단계를 거치며 빠른 단계 전이 |
| | Chen et al., *Unveiling Induction Heads: Provable Training Dynamics* (2409.10559, 학회 미확인) | n-gram 마르코프 데이터에서 2층 attention 모델의 경사 흐름 수렴을 증명 |
| | Nichani et al., *How Transformers Learn Causal Structure with Gradient Descent* (2402.14735, ICML 2024) | 2층에서 잠재 인과 그래프를 첫 attention 층에 학습함을 증명 |
| 출현 조건과 데이터 | Chan et al., *Data Distributional Properties Drive Emergent In-Context Learning* (2205.05055, NeurIPS 2022 구두) | burstiness, 많은 희귀 클래스 같은 데이터 성질이 ICL을 일으킴 |
| | Reddy, *The mechanistic basis of data dependence and abrupt learning in an in-context classification task* (2312.03002, 학회 미확인) | 최소 attention-only 네트워크에서 ICL은 induction head의 급작스러운 출현에서 오고, 이후 in-weights 학습과 경쟁 |
| | Singh et al., *The Transient Nature of Emergent In-Context Learning* (2311.08360, 학회 미확인) | 합성 데이터에서 ICL이 생겼다가 사라질 수 있음 |
| | Singh et al., *What needs to go right for an induction head?* (2404.07129, 학회 미확인) | 합성 데이터에서 induction head 하위 회로의 출현 역학을 광유전학식 개입으로 분석 |
| 인과 효과(큰 모델) | Crosbie & Shutova, *Induction Heads as an Essential Mechanism for Pattern Matching in ICL* (2407.07011, 학회 미확인) | Llama-3-8B, InternLM2-20B에서 induction head를 최소 ablation하면 추상 패턴 과제에서 ICL 성능이 최대 약 32% 하락 |
| 대안 메커니즘 | Todd et al., *Function Vectors in LLMs* (2310.15213, ICLR 2024) | 소수의 attention head가 과제 표현(function vector)을 운반 |
| | Yin & Steinhardt, *Which Attention Heads Matter for ICL?* (2502.14010, ICML 2025) | 12개 모델에서 few-shot ICL 성능은 주로 FV head에 달렸고(큰 모델일수록), 많은 FV head가 학습 중 induction head로 시작해 전환 |
| | Akyürek et al., *In-Context Language Learning: Architectures and Algorithms* (2401.12973, 학회 미확인) | 형식 언어 문맥 학습에 대한 모델 문제 계열 도입 |
| | Minegishi et al., *Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence* (2505.16694, ICML 2025) | 답이 문맥에 있는 경우만 induction이 설명한다고 짚고, 메타 학습형 ICL에서는 다단계 회로 출현을 보고 |
| 변형 | Ren et al., *Identifying Semantic Induction Heads to Understand ICL* (2402.13055, Findings of ACL 2024로 Crossref에서 확인) | 구문·지식 그래프 관계를 회상하는 의미적 induction head가 ICL의 출현과 상관 |
| | Wu et al., *Retrieval Head Mechanistically Explains Long-Context Factuality* (2404.15574, 학회 미확인) | 문맥에서 정보를 가져오는 희소한(5% 미만) retrieval head |
| 일반화 | Tigges et al., *LLM Circuit Analyses Are Consistent Across Training and Scale* (2407.10827, NeurIPS 2024) | 70M–2.8B 모델에서 과제 능력과 회로 구성요소가 비슷한 토큰 수에서 출현 |

## 4. 자료 표

| 자료 | 라이선스 | 별 | 마지막 push | 비고 (`gh api`, 2026-11-24) |
|---|---|---|---|---|
| 논문 자체 | CC BY 4.0 (arXiv 라이선스 페이지) | - | - | 공식 학습 코드·가중치 없음. 학습한 모델은 Anthropic 내부 모델이고 데이터는 비공개 |
| anthropics/PySvelte | Apache-2.0 | 228 | 2021-12-22 (보관됨) | 논문의 대화형 시각화 도구. 학습 코드가 아님 |
| TransformerLensOrg/TransformerLens | MIT | 약 4.1k | 2026-11-23 | GPT 계열 모델의 회로 분석 라이브러리. 외부 복제(Lieberum, 논문 Comments)와 후속 분석의 표준 도구. 우리는 설치하지 않았고 실행하지 않았음 |
| callummcdougall/CircuitsVis | MIT | 9 | 2023-11-07 | 시각화 |

## 5. 이 저장소의 다른 분석과의 관계

- [`attention_sink_emerges/`](../../attention_sink_emerges/) (27번): 같은 "소형 토이에서 현상이 생기는가" 질문에 대해 그쪽은 4층 토이에서 sink가 생기지 않은 음성 결과였습니다. 이 분석은 자연 텍스트 2층 토이에서 같은 종류의 음성 결과를 얻었습니다(`03` §6). 반대로 합성 반복 데이터에서는 단계 변화가 재현됩니다. 둘을 합치면 "토이에서 보이는 단계 변화는 데이터가 그 구조를 강하게 요구할 때만 보인다"는 읽기가 나옵니다.
- [`spike_sparse_sink/`](../../spike_sparse_sink/) (28번, 작성 중): 링크만 둡니다.
- [`cot_faithfulness/`](../../cot_faithfulness/) (13번), [`lm_know_what_they_know/`](../../lm_know_what_they_know/) (14번): 같은 Anthropic 계열이지만 비공개 모델의 출력 행동을 보는 논문이라 이 논문의 회로 분석과 방법이 다릅니다. 공통점은 둘 다 모델 내부를 열지 못하고 외부 관찰에 의존한다는 점이고, 이 논문은 반대로 내부 개입(ablation, 구조 변경)이 가능한 소형 모델을 쓴 덕에 인과 주장이 가능했습니다. 한 가지 연결: Kadavath et al.의 자기 평가가 문맥에서 정보를 가져오는 능력(ICL)에 기대는 정도는 이 논문이 직접 다루지 않습니다.
- `toy_superposition/` (30번), `sparse_autoencoders/` (31번): 예정이라 링크만 둡니다. 31번이 쓸 자연 텍스트 모델은 이 폴더의 `artifacts/`에서 만듭니다(README).
