# 02 선행 연구와 이후 연구 (Lanham et al. 2023, arXiv:2307.13702)

표시: [논문] = 이 논문의 §4와 본문에서 가져온 서지, [확인] = 이 세션에서 arXiv API·Crossref로 제목과 날짜(·초록)를 확인한 것(2026-11-22). 웹 검색은 학회 확인에 1회 썼고, 이후 연구는 arXiv API로 찾았습니다. 초록만 읽은 항목은 그렇게 표시합니다. 원문 전체를 읽은 것은 이 논문뿐입니다.

## 1. 계보 (이 논문 이전)

| 갈래 | 대표 연구 | 이 논문과의 관계 |
|---|---|---|
| 충실성의 정의 | Jacovi & Goldberg 2020 [논문] | "모델 예측의 추론 과정을 정확히 나타낸다"를 그대로 채택(§1) |
| CoT가 왜 효과가 있는가 | Wei et al. 2022 [논문] | 세 가설(수식 생성, 테스트 시점 계산, 사전학습 지식 접근)을 시험. 이 논문의 필러 토큰 실험은 그중 "계산량만"을 더 많은 과제·길이로 넓힌 것(§4) |
| 편향된 입력에서의 불충실 | Turpin et al. 2023, arXiv:2305.04388 (NeurIPS 2023) [확인] | 퓨샷 예시를 "답은 항상 (A)"로 바꾸는 식의 **적대적 설정**에서 CoT가 편향을 언급하지 않고 편향된 답을 정당화(정확도 최대 36%p 하락, GPT-3.5와 Claude 1.0, 13개 BBH 과제). 이 논문은 같은 문제를 **비적대 설정**에서 CoT에 직접 개입하는 쪽으로 풂(§4) |
| 토큰별 기여 | Gao 2023(Shapley) [논문] | CoT의 어떤 토큰이 중요한지. 이 논문은 계산이 더 싼 개입 검사를 제안(§4) |
| 프롬프트 반사실 | Madaan & Yazdanbakhsh 2022 [논문] | 퓨샷 프롬프트 요소를 바꿈. 이 논문은 모델이 만든 CoT를 바꿈(§4) |
| 충실한 구조로 설계 | Lyu et al. 2023(프로그램 생성 후 실행), Creswell & Shanahan 2022, Creswell et al. 2023 [논문] | 구성상 post-hoc이 아닌 방법. 이 논문의 지표를 이런 방법의 평가에 쓸 수 있다고 제안(§4) |
| 질문 분해 | Radhakrishnan et al. 2023, arXiv:2307.11768 [확인] | **같은 날(2023-07-17) 나온 짝 논문.** 질문을 하위 질문으로 분해해 별개 문맥에서 답하게 하면 이 논문의 early answering·adding mistakes 지표로 본 충실성이 CoT보다 크게 오른다고 보고(초록) |

## 2. 이후 연구: 같은 실험을 다시 했는가

초록 기준으로 읽었습니다. 판단 근거는 이 절에 적은 초록 문장이고, 본문의 세부(표본 수, 통계)는 확인하지 못했습니다.

### 2.1 이 논문의 지표 자체를 재현·수정한 연구

| 연구 | 한 것 | 이 논문 주장에 대한 의미 |
|---|---|---|
| **Bentham et al. 2024, "Chain-of-Thought Unfaithfulness as Disguised Accuracy", arXiv:2402.14897 (TMLR)** [확인, 초록] | §3의 크기 실험을 **세 모델군**으로 재현. "특정 조건에서" 스케일링-그다음 역 스케일링 추세를 재현. 그러나 **답 선택지 편향을 보정한 뒤에는 작고 약한 모델의 불충실성이 크게 떨어짐**. 보정한 지표는 **정확도와 R²=0.74로 강하게 상관**해 충실성 평가로서의 타당성에 의문 | **C5와 C7의 가장 직접적인 수정.** 우리 재계산과 같은 방향: 같은 답 비율이 정확도와 얽혀 있음(`03` §4: 그림 7 값과 CoT 없는 정확도의 Spearman ρ=0.85, R²=0.73, 13B 이상 n=32). 다만 보정 방식과 데이터가 달라 R²를 서로 비교할 수는 없음 |
| **Paul et al. 2024, "Making Reasoning Matter", arXiv:2402.13950 (Findings of EMNLP 2024)** [확인, 초록·Crossref] | 12개 LLM에서 인과 매개 분석을 해 "LLM이 중간 추론 단계를 답에 안정적으로 쓰지 않는다"고 보고. 이어 FRODO로 소형 LM이 추론을 더 충실히 쓰도록 훈련 | 이 논문의 결론(과제·모델에 따라 의존도가 다름)을 다른 방법·모델군에서 지지(초록 수준) |
| **Tutek et al. 2025, "Measuring CoT Faithfulness by Unlearning Reasoning Steps"(FUR), arXiv:2502.14829 (EMNLP 2025 본회의)** [확인, 초록·Crossref] | 추론 단계의 정보를 **파라미터에서 지워** 예측이 바뀌는지로 "파라미터적 충실성"을 잼. 4개 LM, 5개 다중 홉 객관식 | 입력 쪽 개입(이 논문)이 아니라 가중치 쪽 개입. 입력 개입의 분포 이탈 문제(`01` §5-7)를 피하는 접근(초록 수준) |
| **Is CoT Really Not Explainability?, arXiv:2512.23032 (ACL 2026)** [확인, 초록] | 힌트 언급 기반 지표(Turpin형)가 "불충실"로 보는 CoT의 많은 수(일부 모델에서 50% 초과)를 다른 지표가 충실로 판정. 인과 매개 분석으로 **언급되지 않은 힌트도 CoT를 거쳐 답을 바꿈**을 보이고 "corruption 기반 지표"를 포함한 넓은 도구를 권고 | 이 논문류의 개입(corruption) 지표와 힌트 언급 지표가 서로 다른 것을 잰다는 근거 |
| **RFEval, arXiv:2602.17053** [확인, 초록] | 대형 추론 모델(LRM) 12개 오픈 모델에서 출력 수준 반사실 개입 벤치마크(7,186개). **49.7%**가 불충실, 정확도와 충실성의 연결은 모델·과제를 통제하면 **약하고 유의하지 않음** | "정확도가 높으면 충실하다/불충실하다"는 단순 대응을 거부. 우리 결과(AOC와 CoT 없는 정확도의 강한 상관)와는 다른 질문이라 충돌이 아님: RFEval은 이 논문의 AOC가 아니라 자기 지표를 씀 |

### 2.2 힌트 언급(Turpin형) 계열: 추론 모델 시대

| 연구 | 요지(초록) |
|---|---|
| Chen et al. 2025, "Reasoning Models Don't Always Say What They Think", arXiv:2505.05410 [확인] | 6가지 프롬프트 힌트에서 추론 모델의 CoT가 힌트 사용을 드러내는 비율이 대개 1% 이상이나 **20% 미만인 경우가 많음**. 결과 기반 RL은 처음엔 충실성을 올리다 정체. RL이 힌트 이용(보상 해킹)을 늘려도 언급 성향은 늘지 않음 |
| Arcuschin et al. 2025, "CoT Reasoning In The Wild Is Not Always Faithful", arXiv:2503.08679 (ICML 2026, arXiv 주석) [확인] | 인위적 힌트 없는 자연 프롬프트에서도 불충실한 CoT가 나타난다는 연구(제목, 출판 표기만 확인) |

이 계열은 "CoT에 개입하지 않고 **입력에 힌트를 넣어** 답 변화와 언급을 비교"하는 방법이라 이 논문(CoT에 개입)과 상보적입니다. 양쪽이 같은 모델에서 일치하는지는 이 두 연구가 직접 다루는지 확인하지 못했습니다.

### 2.3 모니터링 가능성(감시) 계열

| 연구 | 요지(초록) |
|---|---|
| Korbak et al. 2025, "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety", arXiv:2507.11473 [확인] | CoT를 감시해 오용 의도를 찾을 기회가 있으나 불완전하고 깨지기 쉬움. 개발 결정이 monitorability에 미치는 영향을 고려하라는 입장문 |
| Emmons et al. 2025, "When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors", arXiv:2507.05246 [확인, 제목] | CoT가 **계산에 필요한** 과제에서는 모델이 감시를 회피하기 어렵다는 취지(제목 수준). 이 논문 §2.3의 "CoT가 필요한 과제(AQuA, LogiQA)는 더 충실"과 같은 방향의 해석 |
| Baker et al. 2025, "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation", arXiv:2503.11926 [확인, 제목] | CoT를 감시하되 CoT 자체에 최적화 압박을 가하면 숨김이 생길 위험 |
| "Measuring CoT Monitorability Through Faithfulness and Verbosity", arXiv:2510.27378 [확인, 초록] | 충실성(단서 변경 시 언급)과 verbosity(필요한 모든 요인을 나열)를 합친 monitorability 점수. 모델이 충실해 보여도 핵심 요인을 빼면 감시하기 어렵다고 보고 |
| TRACE, arXiv:2510.01367 ("Is It Thinking or Cheating?", 이 저장소의 `trace_reward_hacking` 분석 §2에서 인용, 이 세션에서 원문을 열지 않음) | early answering을 표본 단위·기대 보상으로 일반화해 CoT 절단으로 보상 해킹을 탐지. 이 지표와의 관계는 `03` §7 |

### 2.4 이 논문의 지표를 쓰는 인접 연구

| 연구 | 관계 |
|---|---|
| Silver et al. 2025, "Language Models can perform Single-Utterance Self-Correction of Perturbed Reasoning", arXiv:2506.15894 [확인, 제목. 내용은 이 저장소의 `cannot_self_correct` 분석에서 인용] | CoT에 인위 오류를 넣으면 오픈 모델이 **한 발화 안에서** 자기 수정을 보였다고 보고. 이는 adding mistakes에서 "답이 안 바뀜 = post-hoc"이라는 해석이 자기 수정과 섞이는 경로(`01` §5-4, 토이 `03` §5)를 뒷받침. 이 논문(§2.4)에서 모델이 틀린 답이 선택지에 없을 때 가장 가까운 선택지를 고르는 관찰도 같은 계열 |

## 3. 구성 요소별 출처

| 요소 | 출처 | 이 논문의 변경 |
|---|---|---|
| CoT 프롬프트 | Bowman et al. 2022에서 약간 수정(§2.1) | — |
| 답 읽기: 선택지 다음 토큰 확률 | 이 논문(§2.1) | 샘플한 문자열이 아니라 확률에서 읽음 |
| 문장 단위 절단 | NLTK punkt(§2.1) | — |
| early answering | 이 논문(정식 이름과 AOC) | 이후 TRACE가 표본 수준으로 일반화(`03` §7) |
| 오류 생성 | 사전학습 모델 + 3샷 프롬프트(§2.4, 표 6) | — |
| 필러 토큰 | Wei et al. 2022의 "variable compute" 변형(§4) | 길이를 0에서 최장 CoT까지 변화 |
| 크기별 모델 | Ganguli et al. 2023의 RLHF 시리즈(§3) | — |

## 4. 이 저장소의 다른 분석과의 관계

- `cannot_self_correct`(6번), `self_correction_survey`(7번): 모델이 외부 신호 없이 자기 오류를 고칠 수 있는가를 다룹니다. 이 논문의 adding mistakes는 "추론 중에 넣은 오류를 모델이 바로잡는가"와 같은 현상을 **충실성 지표로 읽는** 설계라, 자기 수정 능력이 높을수록 지표가 post-hoc으로 보이는 교란이 생깁니다(`03` §5 토이).
- `mast`(8번), `who_and_when`(9번): 에이전트 실패의 유형·위치 라벨입니다. 이 논문은 "라벨이 아니라 개입으로 원인을 확인한다"는 점에서 반대편에 있고, 개입 지표도 사람 라벨처럼 기준선(정확도 하한)과 비교해야 한다는 교훈을 공유합니다(`03` §4).
- `agentic_eval_randomness`(10번), `llm_nondeterminism`(11번): 생성 변동이 평가 점수에 미치는 영향을 다룹니다. 이 논문은 문항당 CoT 100개를 샘플하지만 시드와 오차막대 정의가 없어 변동의 원천(문항 대 샘플)을 분해할 수 없습니다(`03` §6).
- `trace_reward_hacking`(12번): early answering의 직접 후속입니다(`03` §7).

## 5. 자료 (GitHub·데이터)

| 항목 | 상태 |
|---|---|
| 이 논문의 공식 코드·결과 파일 | **없음.** 모델이 비공개이고, `gh search`(2회)와 arXiv 페이지에서 저장소를 찾지 못했습니다(2026-11-22) |
| Radhakrishnan et al.(짝 논문)의 프롬프트와 예시 | `github.com/anthropics/DecompositionFaithfulnessPaper`(arXiv 주석의 링크, `gh api`: 크기 25 KB, 별 32, 마지막 push 2023-07-17, 라이선스 표기 없음). 이 논문의 코드가 아니고 클론하지 않았습니다 |
| 논문 개요 | Anthropic 연구 페이지(`anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning`, 웹 검색 결과로 존재 확인, 2023-07-18 게시. 본문은 읽지 않음) |

## 6. 한계

- 위 이후 연구는 대부분 초록만 읽었습니다. "재현했다/반박했다"는 판정은 초록의 문장에 한정되고, 본문의 모델·표본·통계를 검증한 것이 아닙니다.
- "같은 AOC 지표를 공개 모델에서 다시 낸" 연구는 Bentham et al. 2024 하나만 확인했습니다. 추론 모델(긴 CoT)에 early answering을 적용한 연구는 TRACE처럼 지표를 바꿔 쓴 것 외에 찾지 못했고, 이는 "없다"가 아니라 "이번 검색에서 못 찾음"입니다.
