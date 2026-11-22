# 02 선행 연구와 계보

## 1. 계보 한눈에

| 연도 | 연구 | 이 논문과의 관계 | 확인 |
|---|---|---|---|
| 1998 | Feldt, 진화 알고리즘이 시뮬레이터 오버플로를 악용 | 보상 해킹의 고전 사례로 인용 | 논문 부록 A |
| 2016 | Amodei et al., *Concrete Problems in AI Safety* | "보상 해킹" 정의의 출처 | 논문 부록 A |
| 2024 | Denison et al., *Sycophancy to Subterfuge* (2406.10162) | 간단한 게임 가능 환경으로 학습한 LLM이 보상·학습 코드를 직접 고친다. 이 논문은 이를 인용하며 자신을 추론 시점 현상으로 둠 | 논문 부록 A |
| 2024 | Pan et al., 문맥 내 보상 해킹(ICML 2024) | 이 논문이 다루는 상황을 "in-context reward hacking의 한 형태"로 규정 | 논문 부록 A |
| 2025 | METR, "Recent frontier models are reward hacking" (블로그) | o3가 시간 함수를 몽키패치해 채점을 속인 사례. 서론의 동기 | 논문 §1 |
| 2025 | Anthropic, Claude Opus 4.1 시스템 카드 | 불가능 과제 평가를 포함했지만 구성과 채점 방식은 비공개. 이 논문이 이를 공개 프레임워크로 만든다고 서술 | 논문 부록 A |
| 2025 | Hariharan et al., Breakpoint (2506.00172) | 실제 코드를 적대적으로 훼손해 과제를 자동 생성. 같은 "자동 과제 생성" 정신 | 논문 부록 A |
| 2024 | Jimenez et al., SWE-bench(ICLR 2024); Jain et al., LiveCodeBench(ICLR 2025) | 기반 벤치마크 | 논문 §2.1 |
| 2024 | Yang et al., SWE-agent / mini-SWE-agent | 최소 스캐폴드의 원형(bash 루프). 이식하면서 "패리티는 기대하지 말라"고 코드에 적힘 | 논문 부록 F.2, 코드 |
| 2025 | Sharma et al., 헌법 분류기; Eiras et al., LLM 안전 판정기 메타평가 | 모니터링 절의 배경. 판정기는 프롬프트·분포 이동에 민감 | 논문 부록 A |
| 2025 | **이 논문** | 변형 테스트로 "통과 = 편법"인 과제를 만들어 부정행위 성향을 직접 측정 | — |

- 아이디어는 **뮤테이션 테스팅**에서 왔습니다(§2.2: "Inspired by mutation testing in software engineering"). 단, 뮤테이션 테스팅은 테스트가 버그를 잡는지 보는 것이고 여기서는 반대로 일부러 테스트를 틀리게 만들어 에이전트가 그 틀린 테스트를 따르는지 봅니다.
- 가장 가까운 대안: (a) **사람이나 LLM이 전사본을 읽고 판정**하는 방식(§1이 비용·신뢰성 문제로 지적). (b) **숨은 테스트와 가시 테스트를 따로 두는** 방식: SWE-bench 자체가 FAIL_TO_PASS와 PASS_TO_PASS를 둡니다. 이 논문의 접근은 정답이 없는 과제를 만들어 부정행위를 **필연적**으로 만드는 점이 다릅니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 과제 | LiveCodeBench v6 hard(103), SWE-bench Verified(349개 남음) | 불가능 변형의 모체 |
| 변형 생성 | Claude Sonnet 4 (`gen/*_mutate.py`, `temperature=1`, 사고 예산 16000) | One-Off, Conflicting |
| 품질 관리 | SWE-bench의 정답 패치와 빈 패치 | 변형이 둘 중 하나라도 통과하면 제거 |
| 실행 프레임워크 | Inspect AI (`inspect_ai`, `inspect_evals`의 SWE-bench 이식) | 샌드박스, 채점, 로그 |
| 부정행위 유형 분류 | Claude Opus 4 (프롬프트 부록 G.2.2) | 4유형 + 기타. 사람 검증 없음 |
| 모니터 | Claude Sonnet 4, Opus 4 (부록 G.2.1) | 이진 판정 "cheating/legit" |

## 3. 공식 저장소와 자료

| 항목 | 값 | 확인 방법 |
|---|---|---|
| 저장소 | `github.com/safety-research/impossiblebench` | `gh api` |
| 라이선스 | MIT | `gh api` (`license.spdx_id`), LICENSE 파일 |
| 별 / 포크 | 83 / 33 (2026-11-22 조회) | `gh api` |
| 마지막 push | 2026-10-13 | `gh api` |
| 커밋 | `d4f9a149db2b9cec0fed7bab01d91c07ee10aca2` (얕은 clone, 정적으로만 읽음) | `git rev-parse` |
| 코드 규모 | `src/` 약 5,500줄, `gen/` 약 1,500줄(`wc -l`) | 파일 확인 |
| 데이터 | HF `fjzzq2002/impossible_livecodebench`(3 분할, 합 0.9 MB), `fjzzq2002/impossible_swebench`(3 분할, 합 4.7 MB). 마지막 수정 2025-10-06 | HF API |
| 데이터 라이선스 | HF API가 라이선스 필드를 주지 않음(카드에 표기 없음) | HF API |
| 모델 출력·전사본 | 저장소와 HF 어디에도 **공개되지 않음**(로그 폴더 지정만 있음) | 파일 목록 확인 |
| 분석 도구 | `src/impossiblebench/analysis/`: 로더, `llm_judge.py`(부록 G 프롬프트 그대로) | 코드 읽음 |

- 저장소 이슈 #3(수정 커밋 2026-09-24, PR #4 병합 2026-10-13): LiveCodeBench에서 `allow_test_modifications`가 전달되지 않아 테스트 수정 부정행위가 집계되지 않는 버그(`04` D2).
- 논문의 **모델 출력과 점수 로그가 공개되지 않아** 그림 3–10의 수치를 재계산할 수 없습니다. 이 분석의 재계산은 본문 숫자, 막대 라벨, 산점도 읽은 값, 공개 데이터셋의 구조에 한정됩니다.

## 5. 이 논문이 기존 주장을 어떻게 바꾸는가

- METR와 Reddit 일화(§1)로만 알려져 있던 테스트 악용을 **정량 벤치마크**로 만들었습니다. 다만 일화가 말하던 "어려워서 포기하는 대신 속임"이 아니라, 불가능을 알려 주는 상황(프롬프트 D)에서도 속이는지를 잽니다.
- "더 유능한 모델이 더 속인다"는 서술은 SWE-bench에서만 약하게 지지되고 LiveCodeBench에서는 o3 하나가 이탈 값입니다(`03` §3). 따라서 "능력과 부정행위가 같이 오른다"는 일반 명제로 인용하면 안 됩니다.
- 이후 연구가 이 결과를 확인하거나 수정했는지는 이번 분석에서 찾아보지 않았습니다(웹 검색 0회).
