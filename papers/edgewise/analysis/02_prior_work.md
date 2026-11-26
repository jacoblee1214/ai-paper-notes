# 02. 계보와 관련 연구

arXiv 메타데이터(제목, 날짜, 초록)는 arXiv API로, 저장소 정보는 `gh api`로 확인했습니다(2026-11-26). 웹 검색은 쓰지 않았습니다. "확인함"은 이 세션에서 제목·초록을 읽었다는 뜻이고 "논문 인용"은 EdgeWise의 참고문헌 표기를 그대로 옮겼다는 뜻입니다.

## 1. 한 줄 위치

에이전트 평가·효율 축(10축)에서 **한 과제 안에서 단계마다 모델을 고르는** 연구입니다. 질의 단위 라우팅·캐스케이드에서 출발해 다단계 라우팅(Router-R1, MERA)과 에이전트 라우팅(SWE-Router, MTRouter)을 거치며, 새로 보탠 것은 **라우터를 따로 두지 않고 SLM 자신이 "위임" 행동을 내도록 SFT 후 RL로 학습하고, 위임 결정마다 비평가 기준으로 신용을 나누는** 점입니다. 같은 그룹의 AgentBaton(244번)과는 방향이 반대이고(`03` §7), 같은 그룹의 SLM 입장 논문(arXiv:2506.02153)에 대한 실험적 응답으로 읽힙니다.

## 2. 계보

### 2.1 질의 단계 라우팅과 캐스케이드

| 연도 | 연구 | 이 논문과의 관계 |
|---|---|---|
| 2023-05 | FrugalGPT (arXiv:2305.05176, 확인함) | 싼 모델부터 호출하고 점수가 낮으면 큰 모델로 올리는 캐스케이드. **EdgeWise는 인용하지 않음** |
| 2023-10 | AutoMix (arXiv:2310.12963, 확인함, 논문 [7]) | 작은 모델의 답을 자기 검증으로 점검해 큰 모델로 넘김. "작은 모델 자신이 필요를 판단"한다는 점에서 자기 위임의 선행 |
| 2024-04 | Hybrid LLM (arXiv:2404.14618, 확인함) | 작은(엣지) 모델과 큰(클라우드) 모델 중 질의 난이도 예측으로 선택. 엣지-클라우드 구도가 같으나 **EdgeWise는 인용하지 않음** |
| 2024-05 | Ramírez 외, 불확실성 기반 두 단계 선택 (arXiv:2405.02134, COLM 2024, 확인함, 논문 [6]) | 작은 모델의 불확실성으로 큰 모델 호출 결정 |
| 2024-06 | RouteLLM (arXiv:2406.18665, 확인함, 논문 [18]) | 선호 데이터로 라우터 학습 |
| 2026-07 | Conformal Cascade (arXiv:2607.25018, 확인함, 논문 [22]) | 정합 예측으로 에스컬레이션 시점에 정확도 보장 |
| 2026-08 | RLCascadeRouter (arXiv:2608.15817, 확인함, 논문 [23]) | 모델 호출을 MDP로 보고 RL로 성능-비용 최적화 |

### 2.2 생성 중 라우팅 (토큰·추론 단위)

Speculative Decoding (arXiv:2211.17192, 확인함: 작은 모델이 초안, 큰 모델이 검증), RSD(arXiv:2501.19324, 확인함, 논문 [14], 표 1 기준선), CITER(arXiv:2502.01976, 확인함, 논문 [24]), R2R(논문 [25]). 이들은 한 번의 생성 안에서의 라우팅이고, EdgeWise의 위임 단위는 에이전트 한 단계(행동 하나)입니다. RSD는 보상 모델 기반 디코딩이라 에이전트용 기준선으로 쓰려면 재구현이 필요합니다(`04` C7).

### 2.3 다단계·에이전트 라우팅

| 연구 | 내용 | 관계 |
|---|---|---|
| Router-R1 (논문 [8], NeurIPS 2025), MERA (arXiv:2608.10333, 확인함), MasRouter (ACL 2025, 논문 [26]) | 다중 홉 QA와 다중 에이전트에서의 모델 선택 | EdgeWise가 "긴 과제·모델 풀 의존"을 비판하는 대상 |
| TwinRouterBench (arXiv:2605.18859, 확인함, 논문 [10]) | 에이전트 라우팅의 정적·동적 평가 | "기존 평가가 짧은 과제"라는 근거 |
| MTRouter (ACL 2026, 논문 [16]), STEER (AAAI 2026, 논문 [15]) | 이력을 쓰는 턴 단위 라우팅 / 신뢰도 기반 단계 라우팅 | 표 1의 기준선 |
| ToolOrchestra (arXiv:2511.21689, 확인함, 같은 NVIDIA 그룹) | 작은 오케스트레이터가 모델·도구를 고르도록 RL로 학습 | 학습되는 결정자라는 점은 같고, 모델 풀에 묶임 |
| **SWE-Router** (arXiv:2607.00053, 확인함) | 싼 모델이 몇 턴 탐색한 뒤 **부분 궤적을 읽고** 계속할지 비싼 모델로 올릴지 결정. 부분 궤적 조건부가 질의 조건부보다 낫다는 정리 | 가장 가까운 선행. 같은 SWE 도메인. **EdgeWise는 인용하지 않음** |

### 2.4 한 과제 안의 인계 (같은 시기)

| 연구 | 내용 | 관계 |
|---|---|---|
| The Handoff Tax (arXiv:2608.24358, 확인함) | 인계 방향·시점·인터페이스 비교. 약→강은 비용만 추가, 인터페이스 우열이 방향에 따라 뒤바뀜 | `../agent_baton/`에서 다룸 |
| TACIT-Switch (arXiv:2608.27911, 확인함) | 약→강 **한 번 영구 전환** 시점을 중도 절단 데이터로 학습 | EdgeWise는 "복귀하는 위임"이라 구조가 다름 |
| AgentBaton (arXiv:2610.35959, 244번) | 강→약 한 번 인계 시점을 2B 탐지기로 학습 | 방향 반대. `../agent_baton/`, 서로 인용하지 않음 |

### 2.5 도움 요청을 배우는 에이전트 (언제 물어볼지)

| 연구 | 내용 | 관계 |
|---|---|---|
| KnowNo, *Robots That Ask For Help* (arXiv:2307.01928, 확인함) | 정합 예측으로 LLM 계획자가 불확실할 때만 사람에게 도움 요청(로봇) | 로봇 쪽. EdgeWise의 위임 시점 결정과 같은 문제를 **보정된 불확실성**으로 품 |
| R2V-Agent, *Teaching SLMs When to Ask for Help* (arXiv:2605.16604, 확인함) | 증류된 SLM + 교사 LLM + 단계별 검증기 + **위험 보정된 단계 라우터**가 잔여 실패 위험으로 에스컬레이션 | **가장 가까운 동시대 연구.** 라우터를 따로 두는 점이 EdgeWise와 다름. EdgeWise는 인용하지 않음 |
| ADAPT, *Learning When to Ask: Advisor-Aware Post-Training for SLM Agents* (arXiv:2611.27443, 확인함, 2026-11) | SLM을 강한 모델의 조언을 묻고 따르도록 다단계 후처리 학습. 환경의 결과만으로 학습 | 같은 문제. 정책 롤아웃과 결과 감독만 쓰고 비평가 LLM이 없음. EdgeWise보다 늦음 |
| CARL, *Knowing When to Ask: Segment-Level Credit Assignment for LLM Tool Use* (arXiv:2605.27788, 확인함) | 자기 롤아웃으로 비평가를 학습해 도구 사용 경계에서 **구간별 신용**을 부여, 외부 판정 없이 | EdgeWise의 위임 결정별 신용 할당(식 2, 3)과 직접 비교 대상 |
| Ask-to-Act (arXiv:2504.00907, 확인함), Good Time to Ask (arXiv:2206.10606, 확인함), When to Ask for Help: Proactive Interventions (arXiv:2210.10765, 확인함) | 모호한 지시에서 질문, 탐색에서 도움 요청, RL에서 개입 요청 | 에이전트/로봇에서 "언제 물을지" 학습의 계보 |
| 이 저장소 `../lm_know_what_they_know/` (arXiv:2207.05221) | 사전학습 모델의 P(IK), P(True) 자기 평가 | 자기 위임이 의존하는 자기 평가의 기초 연구 |

### 2.6 같은 그룹(NVIDIA LFI)의 이웃

| 연구 | 관계 |
|---|---|
| Small Language Models are the Future of Agentic AI (Belcak 외, arXiv:2506.02153 v3, 확인함, 진행 순서 247) | 에이전트 호출 대부분은 좁고 반복적이라 SLM이 충분하다는 입장과 "SLM 우선, 필요할 때만 LLM"의 이질적 시스템 구성을 제안. **EdgeWise는 그 입장을 코딩 에이전트에서 시험한 첫 실험**에 해당합니다(Belcak이 공저). 이 저장소에서는 아직 분석하지 않았습니다 |
| AgentBaton (244번) | 방향 반대(`03` §7) |
| CPU 위 ARC (진행 순서 245) | 링크만: `../cpu_arc/` |

### 2.7 구성 요소의 출처

| 구성 요소 | 출처 |
|---|---|
| 정책 학습 | GRPO(그룹 상대 이점). 정확한 변형(클립, KL, 토큰 평균/시퀀스 평균)은 논문에 없음 |
| 비평가에 의한 신용 할당 | 위임 결정별 ±1 이점(식 3). CARL(구간별 신용)과 유사 |
| 학습 데이터 | SWE-smith (arXiv:2504.21798, 확인함) |
| 평가 | Terminal-Bench 2.1, SWE-rebench (arXiv:2505.20411, 확인함; "-03"은 월별 판으로 보이나 확인하지 못함) |
| 하네스 | Terminus 2, OpenHands, π(Pi), mini-SWE-agent |
| 서빙 | vLLM 0.19.1, DGX Spark |

## 3. 이 저장소의 다른 분석과의 관계

| 폴더 | 관계 | 대조 |
|---|---|---|
| `../agent_baton/` (244) | 같은 그룹, **방향 반대**. 같은 TB 2.1 사용. 기준선 선택, 로컬 0원 규약, 비용이 뒤쪽 턴에 몰리는 구조가 같은 유형의 문제 | 이 논문 `03` §4, §7. 차이: 여기서는 노력 수준 곡선이 있어 최선 기준선 대비를 계산할 수 있음(TB −8%, SWE 13%, 같은 수준끼리 TB 27–39%, SWE 29–53%) |
| `../agents_that_matter/` | 비용·정확도 파레토, 단순 기준선, 비용 보고 | EdgeWise는 파레토 그림(그림 1)과 무작위 위임 기준선을 둔 점에서 권고를 따름. 반면 고정 간격·행동 유형 규칙 기준선이 없고, 학습 비용을 비용에 넣지 않음. 그 논문은 164문항 HumanEval에서 MDE 약 3%p였는데 이 논문은 89·110문항이라 MDE 13.3%p·11.9%p(불일치율 0.2) |
| `../agentic_eval_randomness/` | SWE-Bench-Verified 10회 반복의 해결률 SD 0.7–1.8%p(n=500) | EdgeWise는 n=89·110에 단일 실행(Pass@1)이라 SD가 훨씬 큼. 우리가 역산한 같은 구성의 실행 간 불일치율(0.06–0.20)과 맞음(`03` §1.2) |
| `../eval_error_bars/` | 같은 문항의 짝지은 차이 SE | 과제별 결과가 없어 불일치율 시나리오로 대신(`03` §3) |
| `../little_power/` | MDE ≈ 2.80·√(불일치율/n) | TB 13.3%p, SWE 11.9%p(0.2). EdgeWise의 표 1 차이 대부분이 아래에 있음 |
| `../swe_bench/`, `../swe_agent/` | SWE-bench 채점, 하네스(ACI)의 뿌리. swe_agent는 같은 설정 6회에서 과제 단위로 한 번이라도 푼 98문항 중 약 78문항이 실행마다 갈렸음 | Pass@3(한 번이라도 성공)은 이 변동에 유리함. EdgeWise의 Pass@3 "동률"이 가진 의미(`03` §2) |
| `../lm_know_what_they_know/` | 사전학습 모델의 자기 평가(P(IK), P(True))는 학습 분포 안에서 구분하고 밖에서는 보정이 나쁨 | 자기 위임이 "스스로 언제 넘길지"를 배우는 방식은 보정이 아니라 비평가 기준(행동 유형)과의 일치 보상. 분포 밖(다른 하네스) 전이는 표 7이 부분적으로 지지(`03` §6.3) |
| `../self_repair/` | 피드백(진단) 품질이 수리 이득을 정함 | 위임된 한 단계가 SLM의 이후 진행에 주는 "피드백"으로 읽을 수 있음 |
| `../mast/`, `../who_and_when/` | 실패는 어느 단계에서 생기나, 결정적 오류 단계 식별의 어려움 | EdgeWise의 핵심 단계는 "어려운 단계"이지 실패 단계가 아님. 실패 단계를 맞히는 문제를 피함 |
| `../cpu_arc/` | 링크만 | |


## 4. 자료 표 (GitHub, 2026-11-26 확인)

| 자산 | 라이선스 | 상태 |
|---|---|---|
| **EdgeWise 공식 코드·가중치** | — | **없음.** 프로젝트 페이지 "Code Soon". NVlabs·NVIDIA·nv-dler와 이름 검색에서 저장소 없음 |
| [SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith) (학습 데이터 생성) | MIT | ★837, push 2026-11-23, 14.5 MB |
| [SWE-rebench/SWE-rebench-V2](https://github.com/SWE-rebench/SWE-rebench-V2) | MIT | SWE-ReBench-03이 이 저장소의 부분집합인지 확인하지 못함 |
| [harbor-framework/terminal-bench](https://github.com/harbor-framework/terminal-bench), [harbor](https://github.com/harbor-framework/harbor) | Apache-2.0 | ★1,135 / ★7,467, push 2026-11-25. Terminal-Bench 2.1이 어느 쪽인지 확인하지 못함(이전 이름 laude-institute/terminal-bench는 harbor-framework/terminal-bench-1로 이동) |
| [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | MIT | ★94,356, push 2026-11-26, 703 MB |
| [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | MIT | ★10,089, push 2026-11-23, 20 MB |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) (서빙) | Apache-2.0 | ★97,581, push 2026-11-26, 384 MB |

## 5. 읽는 순서 제안

1. 이 논문 §2–3과 표 1, 그림 1·4. 2. SWE-Router와 R2V-Agent(라우터를 따로 두는 가장 가까운 동시대 연구). 3. CARL과 ADAPT(위임·도움 요청의 신용 할당). 4. `../agent_baton/`(반대 방향). 5. SLM 입장 논문(247번).
