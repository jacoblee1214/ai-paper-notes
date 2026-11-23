# 02. 계보와 관련 연구: SWE-agent

## 1. 계보 (이 논문이 기대는 것)

| 구성 요소 | 출처 | SWE-agent에서의 역할 |
|---|---|---|
| thought + action 반복 | ReAct (Yao et al., arXiv:2210.03629, ICLR 2023) | 매 단계 생각과 명령을 내는 에이전트 루프(§3) |
| 셸 상호작용 환경, Docker | InterCode (Yang et al. 2023, 논문 참고문헌 [59]) | 환경 구현의 기반이자 Shell-only 기준선(부록 A.2, §4) |
| 과제와 숨은 테스트 | SWE-bench (Jimenez et al., arXiv:2310.06770, ICLR 2024) | 평가 데이터와 판정 방식. 17번으로 분석 예정이라 링크만 둔다: `swe_bench/`(예정) |
| 비대화형 기준선 | SWE-bench의 BM25 RAG | 표 1의 비교 대상(전체 1.31/3.79%, Lite 2.67/4.33%) |
| 실행 피드백으로 코드 수정 | Self-Debug 등 실행 기반 수정 계열(논문 [39]) | 서론에서 "코드 생성에 실행 피드백이 효과적"이라는 출발점 |
| linter 가드레일 | 사람용 IDE의 구문 검사 | 편집 도구에 flake8 일부 규칙을 넣음(`04` §2) |

저자들이 내세우는 새로움은 구성 요소가 아니라 **"인터페이스 자체를 설계 대상으로 삼고 어블레이션한다"**는 틀(ACI)이다(§1). 저자가 쓴 표현은 "SWE-agent is the first work to explore language agents for end-to-end SE"(§6.2)이다. 같은 시기에 이슈 해결을 다룬 AutoCodeRover(2024-04-08)가 있으므로 "첫 번째"는 조심해서 인용한다.

## 2. 같은 시기와 가까운 대안

| 방법 | arXiv | 아이디어 | SWE-agent와의 차이 |
|---|---|---|---|
| AutoCodeRover | 2404.05427 | AST 기반 코드 검색 API와 스펙트럼 기반 결함 위치 | 도구를 프로그램 구조에 맞춰 설계. 모델이 셸을 직접 쓰지 않음 |
| Agentless | 2407.01489 | 위치 찾기 → 수정 → 패치 검증의 3단계, 에이전트 없음 | 초록 기준 Lite 32.00%(96개)에 $0.70. "복잡한 에이전트가 정말 필요한가"를 묻는다 |
| OpenHands (OpenDevin) | 2407.16741 | 에이전트 개발 플랫폼, 샌드박스, 다중 에이전트 | 범용 플랫폼. SWE-bench는 평가 대상 중 하나 |
| Aider 등 상용 도구 | (논문 아님) | 편집 형식(diff, 줄 범위)을 설계 | 편집 인터페이스 설계라는 같은 문제의식 |

Agentless의 Lite 32.00%는 그 논문의 초록 값이고, 이 분석에서 재계산하지 않았다. SWE-agent의 GPT-4 Turbo Lite 18.00%(2024-04)와 비교할 때 모델과 시기가 다르다.

## 3. 이후 (SWE-agent 자신과 주변)

- **SWE-agent 1.0 (2025-02-13, v1.0.0).** 저장소가 크게 바뀌었다. 설정이 `config/`의 yaml과 `tools/`의 도구 묶음(`windowed`, `windowed_edit_linting`, `search`, `submit`, `review_on_submit_m`, `filemap` 등)으로 나뉘고, 논문 시점의 구성은 `config/sweagent_0_7/`로 남아 있다(2026-11-23 GitHub API 확인). 논문 시점 코드는 v0.3.0(커밋 `43b8de5`, 2024-05-01)으로 대조했다.
- **mini-swe-agent.** 같은 팀의 후속이다. 저장소 README는 "SWE-agent의 도구 중심 설계는 2024년에 중요했지만, LM이 강해지면서 상당 부분이 필요 없다"고 쓴다. `mini`는 **bash 외 도구가 없고**, 기록이 선형이며, 행동마다 독립된 `subprocess.run`으로 실행한다. 현재 README는 SWE-bench Verified에서 74% 넘는 점수를 주장하고, SWE-agent 저장소 README는 "mini-swe-agent가 SWE-agent를 대체했고 성능이 맞먹으면서 훨씬 단순하다"고 쓴다(둘 다 저자 측 주장이며 이 분석에서 재현하지 않았다). SWE-bench 공식 사이트에는 mini-swe-agent로 여러 LM을 비교하는 "Bash Only" 리더보드가 있다(웹 검색으로 확인). 이것은 **원 논문의 핵심 주장(ACI가 성능을 크게 좌우한다)의 적용 범위가 모델이 강해지면 좁아진다**는 저자 측 증거이지만, 같은 모델에서 도구 유무를 맞춘 어블레이션은 확인하지 못했다.
- **SWE-bench 평가의 신뢰성.** 해결로 판정된 패치의 상당수가 숨은 테스트가 약해서 통과한 것이라는 후속 연구가 있다(§4의 `swebench_solved_issues`).
- **반복 실행 변동.** 에이전트 평가를 한 번만 돌리면 변동이 크다는 후속 연구가 있다(§4의 `agentic_eval_randomness`).

## 4. 이 저장소의 다른 분석과의 관계

| 분석 | 관계 |
|---|---|
| `../swebench_solved_issues/`(5번) | 해결로 판정된 패치의 7.8%가 개발자 테스트 전체에서 실패하고 29.6%가 정답과 다르게 동작한다(표 1, 2, Verified의 세 도구 877개). SWE-agent GPT-4(2024-04, 전체·Lite)는 그 표본에 없다. 같은 비율을 그대로 옮기면 안 되지만, **SWE-agent의 12.47%와 18.00%도 "숨은 테스트 통과율"이지 "정답 수정률"이 아니다.** Verified는 문항을 걸러 낸 집합이라 전체·Lite에서는 약한 테스트의 비중이 다를 수 있고 방향도 알 수 없다. 이 분석은 SWE-agent의 패치를 다시 검사하지 않았다 |
| `../impossiblebench/`(4번) | 테스트와 명세가 모순일 때의 편법 성향. SWE-agent는 테스트 파일 편집을 막는 장치가 없다(차단 목록은 vim, git 등 `04` D8). SWE-bench 판정은 채점 때 테스트 파일을 정답 테스트 패치로 덮어쓰므로 테스트 수정은 통과로 이어지지 않는다(SWE-bench 원 논문 설계, 17번에서 확인 예정). 이 분석은 그 사실을 SWE-agent 코드에서 확인하지 않았다 |
| `../self_repair/`(1번) | 자기 수리의 병목은 피드백 품질이라는 결과. SWE-agent의 linting 편집 가드는 "결정적이고 신뢰할 수 있는 진단"이 수리를 돕는 같은 방향의 사례이나, 표 3의 효과는 단일 실행이라 3%p 차이가 약하다(`03` §3). 편집 회복 실패가 미해결의 23.4%(그림 8)라는 점은 회복 단계가 병목임을 보인다 |
| `../agentic_eval_randomness/`(10번) | 같은 설정 반복 실행의 변동. 표 10의 6회 실행(±0.49%p)과 비교할 가치가 있다(`03` §5) |
| `../llm_nondeterminism/`(11번) | temperature 0에서도 출력이 달라진다. SWE-agent는 temperature 0.0으로 돌렸는데 표 10의 6회 실행이 서로 다르다(문항별로 갈림) |
| `../cannot_self_correct/`(6번), `../self_correction_survey/`(7번) | SWE-agent는 외부 신호(실행 결과, linter)로 고치는 구조이고 내재적 자기 수정과 다르다. 다만 에이전트가 만든 재현 스크립트가 "검증"이 되므로 재현 스크립트가 틀리면 거짓 확신이 가능하다(논문은 이를 측정하지 않았다) |
| `../mast/`(8번), `../who_and_when/`(9번) | 실패 유형 분류의 비교 대상. SWE-agent의 그림 8 분류(9범주)는 GPT-4o가 자동 분류하고 사람 라벨 15개로 검증했다(87%, 13/15, 구간 62–96%). 주석자 간 일치도는 없다 |
| `../self_debug/`(15번) | 같은 계열(실행 피드백 수정). 작업 중이라 링크만 둔다 |
| `swe_bench/`(17번, 예정) | 원 벤치마크. 분석 후 연결 |


## 5. 자료

| 자료 | 주소 | 라이선스 | 별 | 마지막 push | 크기 | 비고 |
|---|---|---|---|---|---|---|
| SWE-agent | github.com/SWE-agent/SWE-agent | MIT | 20,856 | 2026-11-16 | 72 MB | v0.3.0(`43b8de5`)을 `repo/`에 얕게 clone(4.2 MB), 실행 안 함 |
| mini-swe-agent | github.com/SWE-agent/mini-swe-agent | MIT | 9,988 | 2026-11-16 | 20 MB | README만 읽음 |
| SWE-bench | github.com/SWE-bench/SWE-bench | MIT | 6,235 | 2026-09-18 | | 17번에서 분석 |
| SWE-bench/experiments | github.com/SWE-bench/experiments | (확인 안 함) | | 2026-09-03 | 348 MB(전체) | 문항별 결과(`evaluation/{lite,test}/20240402_sweagent_*/results/`)만 `gh api`로 받음(0.7 MB) |
| Agentless | github.com/OpenAutoCoder/Agentless | MIT | 2,138 | 2024-12-22 | | 읽지 않음 |
| AutoCodeRover | github.com/AutoCodeRoverSG/auto-code-rover | NOASSERTION | 3,104 | 2026-11-01 | | 읽지 않음 |
| OpenHands | github.com/OpenHands/OpenHands | MIT | 94,118 | 2026-11-23 | | 읽지 않음 |

(별과 push 날짜는 2026-11-23 `gh api`. 논문 라이선스는 arXiv abs 페이지 기준 CC BY 4.0. OpenReview 기록: 포럼 `mXpq6ut8J3`, NeurIPS 2024 포스터.)

## 6. 리뷰에서의 위치

10축(에이전트)의 "도구 설계·하네스" 갈래다. "에이전트 성능이 모델만이 아니라 인터페이스에도 달려 있다"는 근거로 쓰이며, 쓸 때는 (a) 2024년 초 모델 기준, (b) 어블레이션이 단일 실행, (c) 저자 후속의 "도구 대부분이 불필요" 수정을 함께 적는다.
