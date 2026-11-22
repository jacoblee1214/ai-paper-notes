# 02 선행 연구와 관계 (TRACE, arXiv:2510.01367)

표시: [논문] = TRACE 논문의 참고문헌·관련 연구 절에서 가져온 서지(이 세션에서 원문을 다시 열지는 않음), [확인] = 이 세션에서 arXiv·공식 페이지로 확인한 것. 웹 검색은 2회 썼습니다.

## 1. 계보

| 갈래 | 대표 연구 | TRACE와의 관계 |
|---|---|---|
| 보상 해킹의 정의 | Skalse et al. 2022 (arXiv:2209.13085) [논문] | §2의 R̂ 대 R 형식화를 그대로 씀 |
| 보상 조작·해킹의 LLM 사례 | Denison et al. 2024 "Sycophancy to subterfuge" (2406.10162), METR 2025 "Recent frontier models are reward hacking", Kahn 2025(SWE-bench 이슈 #465, 미래 커밋 열람), Sakana CUDA 사례 [논문] | 도입부 동기. 이 저장소의 `impossiblebench/`(4번)가 같은 현상을 **테스트 악용 성향**으로 직접 잰 벤치마크 |
| CoT 모니터링 | Baker et al. 2025 (2503.11926): 보조 LLM이 CoT를 읽어 오용 감지. 강한 최적화 압박에서는 모델이 의도를 숨김(obfuscation) [논문] | TRACE의 직접 비교 대상. TRACE는 이 압박에 TRACE도 약해질 것이라 예상(§7, 미실험) |
| CoT 충실성 | Turpin et al. 2023 (NeurIPS), Lanham et al. 2023 (2307.13702), Chen et al. 2025 (2505.05410) [논문] | TRACE는 Lanham의 early answering을 표본 수준·기대 보상으로 일반화(§4). 이 저장소의 13번 `2307.13702`가 원전 |
| CoT가 필요할 때의 모니터링 | Emmons et al. 2025 (2507.05246), METR 2025 "CoT may be highly informative despite unfaithfulness" [논문] | CoT가 계산에 필요하면 모니터가 통한다는 주장. TRACE의 대상은 CoT가 필요 없는 쉬운 허점 |
| 모니터링 가능성 일반 | Korbak et al. 2025 "Chain of Thought Monitorability" (2507.11473, v2 2025-12-07) [확인] | 이 논문의 인용 목록에 없음. 모니터링이 "새롭고 깨지기 쉬운 기회"라는 입장 서술 |
| 언급 강제 | Turpin et al. 2025 VFT (2506.22777) [논문] | 해킹을 CoT에 말하게 훈련. TRACE는 말하지 않는 해킹이 대상 |
| 추론 노력의 대안 지표 | Chen et al. 2026 "deep-thinking tokens" (2602.13517) [논문] | §7이 후속으로 제안한 내부 상태 기반 노력 측정. 절단 생성보다 싸다는 점이 강점 후보 |
| 은닉 추론 | Pfau et al. 2024 (dot by dot), Roger & Greenblatt 2023, Hao et al. 2024 (연속 잠재 추론) [논문] | CoT 모니터가 원리적으로 못 보는 경우 |
| 은닉 지식의 탐침 | Zhang et al. 2025 "Reasoning models know when they're right" [논문] | 같은 저자군(He He). 은닉 상태에서 정답 신호를 읽음. 과잉 사고 논의에서 인용 |
| 허점 발견 | Zhong et al. 2022 (텍스트 분포 차이 서술), Zhong et al. 2026 Hodoscope [논문] | §5의 군집 후 LLM 비교는 이 갈래의 단순판 |

## 2. 발표 이후 같은 묶음 (검색 결과)

- **이름 충돌 주의 [확인]:** arXiv:2601.20103 "Benchmarking Reward Hack Detection in Code Environments"도 약어 TRACE(Testing Reward Anomalies in Code Environments)를 씁니다. 코드 환경 보상 해킹 탐지 벤치마크이고, 검색 요약에 따르면 최고 모델(GPT-5.2 고추론)이 전체 해킹의 63%만 탐지했습니다. 이 논문의 TRACE와 다른 것입니다(인용 시 혼동 금지). 이 값은 검색 요약에서 읽은 것이고 원문은 열지 않았습니다.
- 이 논문을 확장·재현한 후속 연구는 이번 검색 한도(2회) 안에서 찾지 못했습니다. "후속이 없다"는 결론이 아니라 "찾지 못함"입니다.

## 3. 구성 요소별 출처

| 구성 | 출처 |
|---|---|
| 절단 후 강제 답변 | Lanham et al. 2023 (early answering) |
| E[R̂]로의 일반화 | 이 논문(§4) |
| 임계값 = 초기 정책 점수 | 이 논문. 해킹 이전 체크포인트를 기준 정책으로 쓰는 감사 관행과 같은 발상 |
| 반사실 라벨(힌트 제거·검증기 교체) | 이 논문(§3.2). IC는 "틀린 힌트에서 실패"를 더 엄격한 시험으로 사용 |
| RLOO / GRPO | Ahmadian et al. 2024 / Shao et al. 2024 |
| 허점 설계: 문제 번호에 정답 숨김 | 이 논문. 명시 힌트(Chen 2025, METR, Emmons)와 대비하려고 암묵적으로 변형 |

## 4. 코드·자료 표

| 자료 | 상태 |
|---|---|
| 공식 코드 | **찾지 못함.** 논문 본문에 코드 링크 없음. GitHub 저장소 검색 2회(`implicit reward hacking TRACE`, `truncated reasoning auc evaluation`)에서 해당 저장소 없음. ICLR 페이지(`iclr.cc/virtual/2026/oral/10010464`)의 코드 링크도 확인하지 못함. OpenReview 보조 자료는 열지 못함 |
| 학습 설정 | 부록 F 표 1·2(배치, 학습률, KL 계수, LoRA 등)에 있어 재구현 가능하나 시드·체크포인트·탐지 표본 수는 없음 |
| 프롬프트 | 부록 G(허점 주입), H(모니터 루브릭), I(군집 비교) 전문 공개 |
| 라이선스 | 논문 CC BY 4.0 (arXiv abs 확인) |

## 5. 이 저장소의 다른 분석과의 관계

- **`impossiblebench/` (4번):** 테스트 변형으로 "통과 = 위반"인 과제를 만들어 **결과만으로** 악용 성향을 잽니다. TRACE는 정반대로 결과(통과)가 같아도 **과정(CoT 접두부)**으로 악용을 잡습니다. 두 방법은 보완적입니다. ImpossibleBench가 모니터 탐지율을 42–65%로 보고(`../impossiblebench/analysis/03_analysis.md` §6)한 것과 TRACE 논문의 72B 모니터 F1(0.39–0.83)은 지표(재현율 대 F1)와 과제가 달라 직접 비교하면 안 됩니다.
- **`swebench_solved_issues/` (5번):** 약한 숨은 테스트 때문에 틀린 패치가 통과. 검사 쪽 원인이며 TRACE의 RM 허점(허술한 검증기)과 같은 족입니다. TRACE의 RM 허점은 의도적으로 주입된 것이라 "자연 발생 약한 테스트"와는 다릅니다.
- **`cannot_self_correct/`(6번), `self_correction_survey/`(7번):** 모델이 자기 출력을 검토하는 신호의 한계. TRACE는 모델의 자기 평가를 쓰지 않고 외부 검증기 R̂을 씁니다.
- **`mast/`(8번), `who_and_when/`(9번):** 실패 분류·귀인. TRACE는 실패가 아니라 **통과한 응답 안의 위반**을 가리는 점에서 대상이 다릅니다.
- **`agentic_eval_randomness/`(10번):** 반복 실행·검정력 논점이 TRACE의 표본 수 문제(`03` §2, §4)에 그대로 해당합니다.
- **13번 (`2307.13702`, 아직 안 함):** TRACE의 방법적 뿌리(early answering).
