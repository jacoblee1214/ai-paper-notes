# 02. 계보와 관련 연구

검색 엔진 웹 검색은 쓰지 않았습니다(0회). arXiv API(제목·초록·댓글), OpenReview 검색 API, neurips.cc 페이지, GitHub·HF API를 썼습니다. "논문이 인용한 것"은 논문 참고문헌에서, 후속 연구는 OpenReview 검색 결과의 **제목과 venue만** 옮겼고 본문은 읽지 않았습니다(그렇게 표시). 이 저장소의 분석 관계는 완료된 폴더의 README와 분석을 읽고 적었습니다.

## 1. 계보

1. **측정 이론(심리측정):** Cronbach & Meehl 1955 "Construct Validity in Psychological Tests"(논문 [8]), Campbell & Fiske 1959의 수렴·판별 타당도 다특성-다방법 행렬([24]), Messick 1998 "Test Validity: A Matter of Consequence"([19]). 논문은 구성 타당도를 face·content·ecological·predictive·convergent·discriminant·criterion으로 나눠 코드북에 옮깁니다(§2).
2. **ML 벤치마크 비판:**
   - Raji et al. 2021, "AI and the Everything in the Whole Wide World Benchmark"(arXiv:2111.15366, NeurIPS 2021 Benchmarks and Datasets, 포지션 논문). 소수의 "범용" 벤치마크가 일반 능력의 대리로 쓰이는 구성 타당도 문제를 논증합니다. 이 논문은 이를 "벤치마크 = 과제 + 지표 → 현상" 정의의 출발점으로 인용합니다(§3).
   - Bowman & Dahl 2021, "What Will It Take to Fix Benchmarking in NLU?"(arXiv:2104.02145, 논문 참고문헌에는 NAACL 2021): NLU 벤치마킹을 고치는 조건을 논한 글. 초록 이상은 읽지 않았습니다. 이 논문이 구성 타당도의 가치를 말할 때 인용합니다([7]).
   - Hutchinson et al. 2022 "Evaluation Gaps in ML Practice"(FAccT), Dynabench(Kiela et al. 2021, 동적 벤치마크).
   - Reuel et al. 2024, **BetterBench**(arXiv:2411.12990, NeurIPS 2024 스포트라이트): 벤치마크 수명주기 46개 모범 사례로 24개 벤치마크를 평가(초록). "대부분 통계적 유의성을 보고하지 않는다"는 발견이 이 논문의 16.0%와 같은 방향이지만 BetterBench는 벤치마크별 평가이고 이 논문은 문헌 전수 코딩입니다. 이 논문은 BetterBench를 "수명주기 전체의 모범 사례를 모은 선행"으로 인용합니다([17], §1).
   - Alaa et al. 2025, 의료 LLM 벤치마크는 구성 타당도를 우선해야 한다(arXiv:2503.10694, OpenReview 기준 ICML 2025 포지션 트랙): 의료 시험 문항 기반 벤치마크의 구성 타당도 점검을 주장합니다. 이 논문이 구성 타당도 정의에서 인용([18]).
   - 평가 방법론: lm-evaluation-harness의 재현 교훈(Biderman et al. 2024, arXiv:2405.14782), Miller 2024 오차 막대(arXiv:2411.00640, 이 저장소 21번), Weidinger et al. 2025 평가 과학(arXiv:2503.05336), 일반 척도(Zhou et al. 2025, arXiv:2503.06378: 과제별 요구 수준 주석으로 설명력·예측력을 높임, 이 논문이 GSM8K 예시에서 인용).
3. **이 논문(2025-11):** 위 비판의 구성 타당도 관점을 **445개 벤치마크의 전수 코딩**으로 정량화하고, 귀납 코딩으로 8개 권고와 28항목 점검표를 낸 것.
4. **후속(OpenReview 검색 결과, 본문 미확인):** "Construct Validity Failures in Agentic AI Benchmarks: An Empirical Audit"(KDD 2026 워크숍 Agentic AI Evaluation and Trustworthiness, OpenReview `g3m9VEG6i3`), "When Guardrails Look Effective: Construct Validity Failures in LLM Agent Commerce Evaluation"(NeurIPS 2026 TAE 워크숍 포스터), "When Does a Probe Still Measure Its Construct?"(COLM 2026 워크숍 AIMS). 제목만 확인했고 이 논문 점검표를 쓰는지는 모릅니다.

- **이 논문이 한 새로운 것:** (1) 구성 타당도를 문헌 전수 코딩으로 셈(445편, 29명). (2) 코드북과 코딩 결과 공개(HF, GitHub). (3) 권고별 점검표.
- **하지 않은 것:** 점검표의 효과 검증(점검표를 따른 벤치마크가 실제로 더 타당한지)은 없습니다. 점수·등급을 내지 않고(BetterBench와 달리) 벤치마크 개별 평가표도 내지 않습니다. 코딩은 주로 1인 코딩이고 일치도가 낮습니다(`03` §3).

## 2. 구성 요소별 출처

| 요소 | 출처 | 우리 확인 |
|---|---|---|
| 현상–과제–지표 삼분 | Raji et al. [3] | 코드북 필드 수준에서 구현(부록 B) |
| 샘플링 분류(편의·목표·기준·무작위) | 사회과학 표집 이론, Alvi 2016([42]) | 코드 응답 문자열은 4개 범주(`03` §2) |
| 통계 방법 권고 | Miller 2024 [14], McIntosh et al. [13], HiBayES [15] | 이 저장소 21·20번이 계산 방법을 검증(§4) |
| 오염 | Zhou 2023 "Don't Make Your LLM an Evaluation Benchmark Cheater"[59], Zhang et al. 2024 GSM1k [60], LINGOLY-TOO [62] | 코드북에 대응 항목 없음 |
| 오류 분석 | Phuong et al. 2024 [48] | 코드북에 대응 항목 없음 |
| 일치도 지표 | Brennan & Prediger 1981(부록 D 참고문헌 [3]) | 우리 재계산은 `03` §3 |

## 3. 이후 연구 (본문 미확인, `reading_list.md` 기록)

| 논문 | 관계 |
|---|---|
| ABC: Establishing Best Practices in Building Rigorous Agentic Benchmarks (arXiv:2507.02825, NeurIPS 2025 D&B) | 이 논문의 선행·동시기 작업. 에이전트 벤치마크로 좁혀 과제·결과 타당성을 점검(이 저장소 23번) |
| The Leaderboard Illusion (arXiv:2504.20879, NeurIPS 2025 포스터) | 순위 왜곡. 구성 타당도와 다른 축(운영) |
| What Are We Actually Benchmarking in Robot Manipulation? (arXiv:2606.04233, CoRL 2026) | 로봇 조작 벤치마크 감사 (링크만) |

## 4. 이 저장소의 관련 분석과의 관계

각 분석의 구체 수치는 해당 폴더 README 인용 카드를 보세요. 여기서는 이 논문과의 관계만 적습니다.

| 폴더(번호) | 관계 |
|---|---|
| `../agentic_benchmark_checklist/` (23) | 같은 계열, **범위가 다릅니다.** ABC는 에이전트 벤치마크의 과제·결과 타당성(43항목, 채점기 허점)이고, 이 논문은 LLM 벤치마크 일반의 구성 타당도(현상 정의, 과제 대표성, 재사용, 통계, 정당화)입니다. 23번 02는 ABC가 구성 타당도를 R.5·R.6 두 항목으로만 다뤘고 이 논문이 그것을 확장하는 위치라고 적었습니다. 두 분석 모두 **평가자 일치도가 약하다**: ABC는 없음, 이 논문은 κ 0.15(`03` §3). 두 점검표를 합치면 "무엇을 재는가(이 논문) + 채점이 제대로 되는가(ABC)"가 됩니다 |
| `../agents_that_matter/` (22) | 에이전트 평가에서 비용·기준선·홀드아웃. 이 논문 권고 5(홀드아웃으로 오염 대비), 3(일반성 수준에 맞는 표본)과 겹치고, 권고 6(통계)이 단순 기준선과의 비교 근거를 요구하는 쪽과 맞닿습니다. 22번은 "동률"을 구간 폭으로 판정했고 이 논문은 그런 판정을 요구하는 데까지 가지 않습니다 |
| `../swe_bench/` (17) | 권고 3·4·5의 구체 사례. 같은 모델이 Verified 500문항에서 나머지보다 2–4배 높게 풀리는 선택 효과(17번), 판본별 채점 정의 차이는 권고 4("수정을 정당화하라")와 권고 2("출력 파싱·채점 검증")에 해당합니다. 이 논문의 점검표는 채점 코드를 읽지 않으므로 17번이 찾은 채점기 허점(빈 P2P=1, SKIPPED F2P)을 놓칩니다 |
| `../eval_error_bars/` (21) | 권고 6의 **방법론**. 군집 SE, 쌍체 차이, 검정력으로 "불확실성 추정"을 어떻게 낼지. 이 논문의 16.0%(표준편차 포함)가 낮다고 말하는 대상이 이 논문들이 다루는 오차 막대입니다 |
| `../little_power/` (20) | 권고 6의 "표본 크기와 검정력 정당화"의 근거. EMNLP 2019 사람 평가의 항목 100개 이하 69%·유의성 검정 24%(20번)는 이 논문의 검정 2.9%, 시험 세트 크기 중앙값 약 2,400(`03` §4)과 같은 방향의 별개 조사 |
| `../statistical_precipice/` (19) | 권고 6의 한 변형(적은 실행 수의 구간) |
| `../mast/` (8), `../agentdebug/` (18) | 사람 주석 일치도 보고 방식의 비교 대상. mast는 κ를 PDF 원표로 재계산해 구간을 붙임. 이 논문은 BPK만 보고하고 κ·α는 보고하지 않습니다 |
| `../llm_as_judge/` (24, 진행 중) | 링크만. 지표에 LLM 판정자를 쓴 17.1%(§4)에 대한 심화 |

## 5. 자료 표

| 자료 | 주소 | 라이선스 | 크기 | 별 | 마지막 갱신 | 쓴 용도 |
|---|---|---|---|---|---|---|
| 논문 | arxiv.org/abs/2511.04703 (v1) | CC BY 4.0 (abs 페이지, 2026-11-23) | PDF 0.86 MB, 29쪽 | | 2025-11-03 | 전 분석 |
| 코딩 결과(HF) | huggingface.co/datasets/ambean/construct-validity-review (커밋 `312b4f8`) | CC BY-SA 4.0 (카드) | CSV 2개, 4.3 MB | | 2025-11-23 | 03 (코드북 455행, 키워드 통과 2,189편) |
| 코드·자료(GitHub) | github.com/am-bean/benchmark_review (HEAD `ac85a25`) | **라이선스 파일 없음**(API null, README에 없음) | API 51.6 MB (우리가 받은 것: 파일 일부 67 MB) | 2 | 2025-09-29 | 03 (이중 코딩 82행, 원본 학회 목록, 선별 단계) |
| NeurIPS 포스터 | neurips.cc/virtual/2025/poster/121477, D&B 목록 페이지 | | | | | 학회 표기 |
| OpenReview | openreview.net/forum?id=mdA5lVvNcU (venue "NeurIPS 2025 Datasets and Benchmarks Track poster") | | | | | 학회 표기 |

- 받은 파일은 `data/`(gitignore)에만 있고 게시본에는 넣지 않습니다. HF 자료는 CC BY-SA 4.0이라 출처 표기와 동일 조건 공유 조건을 따르면 재배포가 가능하지만, GitHub 자료는 라이선스가 없어 **원자료 사본은 넣지 않고 우리가 계산한 요약 JSON과 스크립트만** 게시본에 둡니다.
- GitHub `data/coding_responses.csv`에는 검토자 이메일 열이 있어 **내려받았다가 지웠고 사용하지 않았습니다.**
