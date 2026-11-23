# 02 계보와 관련 연구

arXiv 메타데이터(제목, 날짜, 초록)는 arXiv API로, 저장소 정보는 `gh api`로, 학회는 OpenReview API로 확인했습니다. 웹 검색은 쓰지 않았습니다. 아래 "확인함"은 이 세션에서 해당 출처를 읽었다는 뜻이고, "논문 인용"은 이 논문의 참고문헌 표기를 그대로 옮겼다는 뜻입니다.

## 1. 한 줄 위치

에이전트 벤치마크의 **평가 관행 비판**입니다. 정확도 단일 지표 대신 비용을 함께 보는 파레토 비교와 단순 기준선(재시도, 온도 올리기, 모델 단계 올리기)을 요구하고, 홀드아웃 설계와 재현성을 문제 삼습니다. 이 저장소의 축 11(평가 방법론)에서 "에이전트 평가"를 맡습니다.

## 2. 계보

### 2.1 비용·효율을 평가 축으로 삼는 흐름

| 연도 | 연구 | 이 논문과의 관계 |
|---|---|---|
| 2019 | Dodge 외 *Show Your Work: Improved Reporting of Experimental Results* (arXiv:1909.03004, 확인함) | 예산(하이퍼파라미터 탐색 횟수)에 따른 성능 곡선을 보고하자는 요구. 이 논문은 같은 문제의식을 에이전트의 추론 비용으로 옮김 |
| 2019 | Schwartz 외 *Green AI* (arXiv:1907.10597, 확인함) | 정확도와 함께 계산 비용을 평가 축으로 삼자는 제안. 이 논문은 4절에서 연산량·파라미터 같은 대리 지표 대신 달러 비용을 하위 개발자용 평가에 쓰자고 구분 |
| 2022 | Liang 외 *Holistic Evaluation of Language Models* (HELM, arXiv:2211.09110, 확인함) | 정확도 외 효율 등을 함께 보고하는 다차원 평가와 표준 harness. 이 논문 6절은 HELM과 LM Evaluation Harness가 모델 평가의 표준화를 이뤘지만 에이전트에는 부족하다고 주장 |
| 2024-07 | **이 논문** (arXiv:2407.01502, 2024-07-01) | 에이전트에 파레토 관점, 단순 기준선, 홀드아웃 수준 분류를 적용 |
| 2025-10 | Kapoor, Stroebl 외 *Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation* (arXiv:2510.11977, 확인함, 같은 저자 팀) | 이 논문 6절이 "ripe area for future work"라고 한 표준 에이전트 평가 harness를 실제로 만든 후속. 초록 기준 9개 모델 × 9개 벤치마크에서 21,730회 rollout, 총비용 약 40,000 USD, LLM 보조 로그 검사로 벤치마크를 HuggingFace에서 검색하는 행동 등을 발견. 이 논문의 주장이 큰 규모에서 어떻게 유지되는지는 이 문서에서 읽지 않았음 |

### 2.2 단순 기준선과 반복 샘플링

- **반복 샘플링이 정확도를 올린다**는 사실은 이 논문 이전부터 알려져 있었습니다(논문 인용: Chen 외 [6] *Are More LLM Calls All You Need?*(arXiv:2403.02419), Li 외 [26] *More Agents Is All You Need*(arXiv:2402.05120), Li 외 [27] AlphaCode의 재시도 1,000회·백만 회 결과, Sun 외 [48]). 이 논문의 기여는 새 사실이 아니라 **"복잡한 에이전트 논문들이 이 기준선을 비교하지 않았다"** 는 점을 같은 비용 축 위에서 보인 것입니다.
- **Large Language Monkeys** (Brown 외, arXiv:2407.21787, 2024-07-31, 확인함). 이 논문 직후(30일 뒤)에 나온, 반복 샘플링의 커버리지가 샘플 수에 로그 선형으로 늘어난다는 연구입니다. 자동 검증기가 있는 영역(코딩, 증명)에서 커버리지 증가가 곧 성능 증가이고, 검증기 없는 영역에서는 다수결·보상 모델이 수백 샘플에서 정체한다고 보고합니다. 이 논문의 Retry·Warming이 통하는 조건("정답 신호가 쉽게 있는 환경", 2.1절)과 같은 조건을 더 큰 규모로 확인한 것으로 읽힙니다.
- **Agentless** (Xia 외, arXiv:2407.01489, 2024-07-01, 확인함). 이 논문과 같은 날 올라온, 에이전트의 복잡도가 필요한가를 묻는 연구입니다. 국소화, 수정, 패치 검증의 세 단계 단순 절차로 SWE-bench Lite에서 32.00%와 0.70 USD를 보고합니다(초록). 이 논문이 SWE-bench 같은 어려운 과제를 "열린 질문"으로 남겼는데, 같은 시기의 다른 증거입니다.

### 2.3 에이전트 벤치마크 결함 지적

- **AgentBench** (Liu 외, arXiv:2308.03688, 확인함): 이 논문이 벤치마크 조사의 출처로 삼은 모음 중 하나(부록 C).
- **Establishing Best Practices for Building Rigorous Agentic Benchmarks (ABC)** (Zhu 외, arXiv:2507.02825, 확인함): 에이전트 벤치마크의 과제 설정·보상 설계 결함(SWE-bench Verified의 불충분한 테스트, τ-bench가 빈 응답을 성공으로 셈)이 성능을 최대 100% 상대 오차로 틀릴 수 있다고 보이고 체크리스트를 제안합니다(초록). 이 저장소의 `swebench_solved_issues/`, `impossiblebench/`와 같은 계열이며, 이 논문의 6절(재현성)과 5절(지름길)을 벤치마크 설계 쪽으로 확장한 후속으로 읽힙니다. 진행 순서 23번에서 따로 분석합니다.
- **Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science*** (arXiv:2207.07048, 확인함): 같은 저자의 선행. 재현성과 누출 문제를 ML 기반 과학 전반에 적용한 연구로, 이 논문 6절의 재현성 관심의 출처 격입니다(내용은 초록 수준으로만 알고 있음).

## 3. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| HumanEval | Chen 외 2021 (*Evaluating Large Language Models Trained on Code*; 논문 참고문헌 목록에 있음) | 164문항, 예시 테스트는 docstring에만 있고 3문항(HumanEval/32, 38, 50)은 없음 |
| LDB | Zhong 외, arXiv:2402.16906 (확인함), 코드 Apache-2.0 | 실행 추적 기반 디버거 |
| LATS | Zhou 외, arXiv:2310.04406 (확인함), 코드 MIT | MCTS 기반 에이전트 |
| Reflexion | Shinn 외, arXiv:2303.11366 (확인함), 코드 MIT | 반성 텍스트로 재시도 |
| DSPy | Khattab 외 (논문 인용 [24]), MIT | 프롬프트·시연 컴파일러 |
| Optuna | Akiba 외 (논문 인용 [2]) | 다목적 하이퍼파라미터 탐색 |
| HotPotQA | Yang 외 (논문 인용 [60]) | 다중 홉 QA |
| NovelQA | Wang 외 (논문 인용 [53]) | 장문 소설 QA |
| WebArena, STeP | Zhou 외 [66], Sodhi 외 [47] | 웹 에이전트 벤치마크와 하드코딩 정책을 쓰는 에이전트 |
| 파레토 전선, 볼록 제약 | 고전 다목적 최적화 | 각주 2의 "두 에이전트를 확률 p로 섞는다" 논리가 볼록 제약의 근거 |

## 4. 이 저장소의 다른 분석과의 관계

| 폴더 | 관계 |
|---|---|
| `self_debug/` (15번) | Self-Debug의 이득을 분해해 "통과·실패 판정을 받는 재시도가 이득의 대부분"이라는 결론을 냈습니다. 이 논문의 Retry·Warming이 LDB·Reflexion과 같은 정확도라는 결과와 같은 방향입니다. 우리 재계산(`03` §3.2)에서 LDB (GPT-4, GPT-3.5)의 디버깅 순수 이득은 시드 대비 +1.34%p(구간 −0.12 ~ +3.17)이고 단순 재시도는 +2.32%p(구간 +0.73 ~ +4.15)입니다. 독립 근거가 모델·벤치마크·신호가 다른 쪽에서 같은 방향입니다. |
| `swe_agent/` (16번) | 이 논문 6절이 SWE-agent의 4 USD 비용 상한과 "전체 평가 한 번에 8,000 USD 이상"을 예로 들어 반복 실행이 어렵다고 합니다. `swe_agent/`는 같은 설정 6회 실행의 변동(17.94±0.49%p)과 어블레이션이 단일 실행이라는 점을 확인했습니다. 이 논문의 "비용 때문에 구간이 없다"의 구체적 사례입니다. |
| `swe_bench/` (17번) | 이 논문 표 A4가 SWE-bench를 과제 특정(task-specific) 벤치마크로 분류하고 홀드아웃(보지 않은 저장소)이 "분포 내 샘플"이라 수준에 맞지 않다고 판정합니다. SWE-bench 자체의 채점 코드와 판본 차이는 `swe_bench/`가 다룹니다. |
| `agentdebug/` (18번) | 같은 시도 수의 Best-of-N이 강한 두 백본에서 디버깅 이득의 75–77%를 설명합니다. 이 논문의 "재시도가 SOTA 구조와 동률"이 코드 생성이 아닌 ALFWorld의 실패 교정에서도 보이는 사례입니다. |
| `agentic_eval_randomness/` (10번) | 같은 설정 10회 반복에서 표준편차 0.7–1.8%p. 이 논문의 5회 실행 SD(0.5–1.9%p)와 같은 범위이고, 이 논문의 구간은 실행 간 변동만 담습니다(`03` §3.1). 반복 횟수와 검정력의 관계(10회면 약 2%p 탐지)가 이 논문의 "5회로 충분한가"의 답이 됩니다. |
| `llm_nondeterminism/` (11번) | 온도 0도 결정적이 아니라는 근거. 이 논문의 Retry(온도 0 반복)가 의미 있는 전제(부록 A.1)입니다. |
| `statistical_precipice/` (19번) | 소수 실행의 구간 추정(층화 부트스트랩)과 이진 점수의 한계. 이 논문의 5회 t 구간 대신 과제 단위 부트스트랩을 쓰는 우리 계산의 방법적 근거입니다. |
| `little_power/` (20번) | 짝지은 비교의 MDE ≈ 2.8·√(d/n). 우리가 HumanEval 164문항에서 "3%p 미만의 차이는 구분되지 않는다"를 계산할 때 같은 식을 썼습니다(`03` §3.2). |
| `eval_error_bars/` (21번) | 군집 표준오차와 쌍체 차이. 진행 중이라 링크만 겁니다. |
| `impossiblebench/`, `swebench_solved_issues/` | 벤치마크 채점의 결함(테스트 악용, 약한 숨은 테스트). 이 논문의 6절 사례(부분 테스트 평가, 오기된 로그)와 같은 계열입니다. |


## 5. 자료 표

| 항목 | 라이선스 | 별 | 마지막 push | 크기 | 비고 |
|---|---|---|---|---|---|
| `benediktstroebl/agent-evals` (이 논문 코드) | MIT | 29 | 2025-05-28 | 84.5 MB | 결과 jsonl·로그 포함. 작업 트리 1.5 GB |
| `benediktstroebl/agent-eval-webapp` | 없음 | 0 | 2024-09-27 | 41 KB | 단가를 바꿔 비용·정확도를 다시 보는 웹앱 |
| `FloridSleeves/LLMDebugger` (LDB) | Apache-2.0 | 586 | 2024-09-10 | 4.3 MB | 이 논문 코드의 바탕 |
| `lapisrocks/LanguageAgentTreeSearch` (LATS) | MIT | 866 | 2024-07-30 | 24 MB | |
| `noahshinn/reflexion` | MIT | 3,439 | 2025-01-14 | 9.1 MB | |
| `asappresearch/webagents-step` (STeP) | MIT | 41 | 2024-07-21 | 3.1 MB | |
| `stanfordnlp/dspy` | MIT | 39,191 | 2026-11-21 | 188 MB | 이 논문의 결합 최적화는 수정한 DSPy에 구현했고 공식 DSPy로의 통합은 "계획"이라고 적었음(부록 I) |
| `princeton-pli/hal-harness` (HAL 후속) | API상 없음 | 308 | 2026-07-01 | 4.4 MB | 후속 harness. 이번에 clone하지 않음 |

(2026-11-23 `gh api` 기준.)

## 6. 판본과 학회

- arXiv는 v1(2024-07-01)뿐이고 라이선스는 `nonexclusive-distrib/1.0`(CC BY 아님).
- **학회: TMLR 2025.** OpenReview 포럼 `Zy4uFzMviZ`(제목 AI Agents That Matter, venue "Accepted by TMLR", venueid TMLR, 담당 편집자 Nino Vieillard, 공식 BibTeX `kapoor2025ai`, 게재 처리 시각 2025-06-13). reading_list 14절의 "mlanthology.org(2차 색인)"은 OpenReview로 확인했습니다. 코드 저장소 README 첫 줄도 `[TMLR 2025]`입니다.
- OpenReview의 변경 요약에 따르면 TMLR 최종본은 (1) 문제를 지적할 뿐 해법은 제안하지 않는다는 점과 (2) 사람 참여 평가는 논의하되 해법은 범위 밖이라는 점을 명확히 하고, 인용을 고치고, 카메라 레디 형식으로 갱신했습니다. 최종본 PDF는 받지 못해(OpenReview 403) 이 분석의 수치는 arXiv v1 기준입니다.
