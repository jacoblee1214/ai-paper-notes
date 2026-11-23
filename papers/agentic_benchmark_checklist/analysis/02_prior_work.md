# 02. 계보와 관련 연구

검색 엔진 웹 검색은 쓰지 않았습니다(0회. arXiv API, GitHub API, neurips.cc와 OpenReview 페이지 조회는 사용). 아래 "논문이 인용한 것"은 논문 참고문헌(`paper/paper.txt`)에서, "이 저장소의 분석"은 완료된 분석 폴더에서, 나머지 후속 연구는 이 저장소의 읽기 목록(11축, 10축)의 기록에서 옮겼고 본문은 읽지 않았습니다(그렇게 표시).

## 1. 계보

1. **벤치마크 품질 일반:** 라벨 노이즈·표준 프로토콜·구성 타당도·오염(논문 §2 [17, 18, 20, 41, 64, 75, 91]), BetterBench [65], How should I build a benchmark [10].
2. **에이전트 벤치마크 개별 결함 보고:** SWE-bench Verified 재주석(OpenAI [14])과 UTBoost [87], KernelBench 채점 허점 [35], WebArena 속도 제한 [32], BIRD 잡음 [62, 80], AI Agents That Matter [32].
3. **ABC (2025-07):** 위 둘을 점검표로 묶음.
4. **후속:** 구성 타당도 서베이(25번), SWE-bench 해결 패치 재검사(5번), 벤치마크 감사류.

- **이전의 한계(§2 "Issues in Evaluating AI Agents"):** 위 개별 결함 보고는 한 벤치마크의 한 문제를 보였고, "실행 가능한 체계적 지침"은 없었다고 논문이 말합니다. ABC는 그 지침을 43개 항목으로 낸 것입니다.
- **ABC가 한 새로운 것:** (1) 타당도를 과제와 결과로 나눈 틀(그림 1). (2) 채점 방식별(문자열·판정자·단위 테스트·퍼징·상태 일치) 항목 체계. (3) 10개 벤치마크에 일괄 적용해 숫자를 만든 평가표. (4) 보고(R) 항목: 트리비얼 에이전트(R.13), 결함의 정량 영향(R.9), 신뢰구간(R.10).
- **ABC가 하지 않은 것:** 점검표의 타당성 검증(항목을 쓰면 실제 평가가 좋아지는가)은 CVE-Bench 한 건뿐입니다. 평가자 간 일치도가 없습니다(`03` §3). 구성 타당도(무엇을 재는가)는 R.5·R.6 두 항목에 그치고 [65]를 인용합니다.

## 2. 구성 요소별 출처 (표 4와 우리 확인)

| 영역 | 출처 | 우리 확인 |
|---|---|---|
| 문자열 일치 O.a·b | GAIA [47], WebArena [93], τ-bench [86] | 일치(표 4) |
| 부분 문자열의 "모두 나열" O.b.2 | τ-bench의 DB 쏟기 실험(이 논문의 새 실험) | `03` §4.1 정확 재현(항공) |
| LLM 판정자 O.c | WebVoyager [25], Ziems et al. [98] | 판정자 정확도 근거 없음을 WebArena에서 지적하나 표 11은 O.c.1=1(`03` §2) |
| 단위 테스트 O.d | SWE-bench Verified [14], UTBoost [87], 커버리지 [94] | O.d.2는 3/3 위반 |
| 퍼징 O.e | KernelBench [59], Lange et al. [35], 퍼징 서베이 [95] | KernelBench 31%는 재계산 불가(`04` D6) |
| 상태 일치 O.g | τ-bench, OSWorld, WebArena | O.g.3, O.g.2 위반 사례 |
| 도구·환경 T.1–T.6 | SWE-Lancer [48], Kapoor et al. [32] | 항목 번호 판본 불일치(`04` D1) |
| 보고 R | MLE-bench [11], LiveBench [78], Dorner & Hardt [17](신뢰구간, 비용 대비 라벨 품질) | R.10·R.13은 9/10 위반 |

## 3. 이후 연구 (본문 미확인, `reading_list.md` 기록)

| 논문 | 관계 |
|---|---|
| Measuring what Matters: Construct Validity in LLM Benchmarks (arXiv:2511.04703, NeurIPS 2025 D&B △) | 구성 타당도를 8가지 권고로. ABC의 R.5·R.6을 확장 (25번, `construct_validity/` 예정, 링크만) |
| AI Agents That Matter (arXiv:2407.01502) | 비용·기준선·holdout. ABC가 R.5와 T.2의 출처로 인용 [32] (22번, `../agents_that_matter/`, 링크만. 다른 에이전트가 작성 중) |
| Are "Solved Issues" in SWE-bench Really Solved Correctly? (arXiv:2503.15223) | O.d.2 위반의 정량: 해결 패치 7.8%가 개발자 테스트 전체에서 실패, 29.6%가 정답과 다르게 동작 (5번 `../swebench_solved_issues/`) |
| SWE-Bench+ (arXiv:2410.06992), UTBoost (arXiv:2506.09289) | 해답 누출·약한 테스트. UTBoost가 ABC 저자들의 같은 연구실이며 ABC가 [87]로 직접 인용 |
| Holistic Agent Leaderboard (arXiv:2510.11977) | 표준 하네스와 로그 분석으로 에이전트의 지름길 탐지(초록). ABC의 T.10을 로그에서 찾는 쪽 |
| The Leaderboard Illusion (arXiv:2504.20879) | 순위 왜곡. ABC가 다루지 않는 리더보드 운영 축 |

## 4. 이 저장소의 관련 분석과의 관계

각 분석에서 **구체적 수치는 해당 폴더의 README 인용 카드**를 보세요. 여기서는 ABC 항목과의 관계만 적습니다.

| 폴더(번호) | 관계 | ABC 항목 |
|---|---|---|
| `../swe_bench/` (17) | 채점 코드 정적 대조에서 빈 P2P=1, SKIPPED F2P, 모델의 테스트 파일 수정, 판본별 "적용" 정의 차이를 찾았습니다. ABC는 SWE-bench에 T.5·T.10을 1점 줘서 이를 놓쳤습니다(`04` 2절 대조표) | T.5, T.10, O.d.2 |
| `../swebench_solved_issues/` (5) | 숨은 테스트의 약함을 해결 패치 877개로 정량. ABC의 O.d.2=0이 가리키는 문제 크기 | O.d.1, O.d.2 |
| `../impossiblebench/` (4) | 테스트와 모순되는 명세로 테스트 악용 성향을 잼. 읽기 전용 테스트에서도 부정행위가 남아 T.5를 이진 점검이 아니라 사용 여부까지 봐야 함 | T.5, O.i.1 |
| `../swe_agent/` (16) | 에이전트 쪽 인터페이스. 하네스 판본·도구 설정에 따라 해결률이 바뀌므로 R.6(평가 대상이 모델인가 에이전트 프레임워크인가)이 의미를 가짐 | R.6 |
| `../agentic_eval_randomness/` (10) | 같은 설정 10회 반복에서 pass@1이 2.2–6.0%p 흔들림. ABC R.10의 구체 형태 | R.10 |
| `../eval_error_bars/` (21) | 군집 SE, 쌍체 차이, 검정력. ABC R.10을 어떻게 계산할지의 방법 | R.10, R.9 |
| `../statistical_precipice/` (19), `../little_power/` (20) | 실행 수가 적을 때의 구간, 검정력. R.10의 기술적 기반 | R.10 |
| `../mast/` (8), `../agentdebug/` (18) | 주석자 간 일치도를 측정하는 방법. ABC는 이 일치도가 **없습니다**(`03` §3) | 해당 없음 |
| `../agents_that_matter/` (22) | 링크만 (다른 에이전트가 작성 중) | R.5, R.13 |
| 25번 `construct_validity/` (예정) | 링크만 | R.5, R.6 |

## 5. 자료 표

| 자료 | 주소 | 라이선스 | 크기 | 별 | 마지막 push | 쓴 용도 |
|---|---|---|---|---|---|---|
| 논문 | arxiv.org/abs/2507.02825 (v5) | arXiv 비독점 배포(`nonexclusive-distrib/1.0`) | PDF 1.2 MB, 39쪽 | | 2025-08-07 | 전 분석 |
| ABC 저장소 | github.com/uiuc-kang-lab/agentic-benchmarks | **파일 없음**(API null) | 16.7 MB(clone 95 MB) | 66 | 2025-07-31 | 04, 03 (정적) |
| ABC 웹사이트 | uiuc-kang-lab.github.io/agentic-benchmarks | 확인하지 않음 | | | | 논문이 안내(§3). 열지 않음 |
| τ-bench | github.com/sierra-research/tau-bench (커밋 `14bf0ef`) | MIT | 6.1 MB(clone 65 MB) | 1.5k | 2026-03-18(main) | 과제 정의 정적 읽기 |
| NeurIPS 포스터 | neurips.cc/virtual/2025/poster/121769 | | | | | 학회 표기 확인 |
| OpenReview | openreview.net/forum?id=E58HNCqoaA | | | | | 학회 표기(`venue`), 제목 변경 확인 |
