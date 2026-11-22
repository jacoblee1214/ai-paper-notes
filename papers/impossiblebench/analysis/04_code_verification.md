# 04 공식 코드 대조 (논문 대 코드, 정적)

- 저장소: `github.com/safety-research/impossiblebench`, 커밋 `d4f9a149db2b9cec0fed7bab01d91c07ee10aca2`(2026-10-13, 얕은 clone, `repo/`는 gitignore)
- **실행하지 않았습니다.** Inspect AI·Docker·API 키가 필요하고, 모델 호출은 비용과 외부 전송이 생겨 승인 없이 할 수 없습니다. 아래는 모두 코드를 읽어 얻은 것입니다.
- 읽은 파일: `livecodebench_tasks.py`, `livecodebench_scorers.py`, `livecodebench_agent_mini.py`, `livecodebench_agent_full.py`, `swebench_tasks.py`, `swebench_scorers.py`, `swebench_agent_full.py`, `swebench_agent_mini.py`(일부), `swebench_dummytest.py`, `analysis/llm_judge.py`(프롬프트와 모델 이름), `gen/swebench_mutate.py`, `gen/livecodebench_mutate.py`, `gen/livecodebench_transcribe.py`, `demo.py`, `README.md`.
- 이 커밋은 논문(2025-10) 이후의 코드입니다. 저장소 README 상단에 "LiveCodeBench 분할에서 에이전트가 테스트를 수정하지 못하게 하는 버그를 2026-10-12에 발견했고, 이는 GitHub로 옮기면서 생겼으며 원래 논문에는 영향이 없다"고 적혀 있습니다(D2). 논문 수치를 만든 코드는 공개본과 다를 수 있습니다.

## 1. 일치한 것

| # | 논문 | 코드 | 판정 |
|---|---|---|---|
| K1 | 변형 생성 프롬프트(부록 G.1.1, G.1.2) | `gen/livecodebench_mutate.py`, `gen/swebench_mutate.py`의 프롬프트 문자열 | 일치(One-Off, Conflicting) |
| K2 | 변형 생성기는 Claude Sonnet 4(§2.2) | 두 mutate 스크립트 `model="claude-sonnet-4-20250514"`, `temperature=1`, 사고 예산 16000 | 일치(온도와 사고 예산은 논문에 없음) |
| K3 | SWE-bench 품질 관리: 정답 패치와 빈 패치로 검증(§2.3) | `dummy="oracle"` / `"nochange"` 옵션과 `swebench_dummytest.py` | 일치하는 도구가 있음. 단, 어느 문항이 제거됐는지의 목록은 공개되지 않음 |
| K4 | 모니터 프롬프트(부록 G.2.1)와 유형 분류 프롬프트(G.2.2) | `analysis/llm_judge.py`의 `BINARY_PROMPT`, `CLASSIFY_PROMPT` | 문구 일치. 기본 모델 `claude-opus-4-20250514`, `max_tokens=19157`, `temperature=1` |
| K5 | 최대 10번 제출(§2.4) | `max_attempts=10`(데모, 코드 SWE 기본값 10) | 일치 |
| K6 | 열린 테스트: 수정해서 통과해도 부정행위로 센다(§2.4) | SWE 기본 `reset_tests=False`일 때 채점 스크립트는 변형 패치를 다시 적용하지 않고 현재 작업 트리의 테스트를 그대로 돌림(`swebench_scorers.py` `get_eval_script`의 `if not reset_tests` 분기) | 일치. 수정된 테스트로 통과하면 점수 1.0 |
| K7 | 테스트 접근 3종(§5.2) | `hide_tests`, `reset_tests` 조합: 완전=(False, False), 읽기 전용=(False, True), 숨김=(True, True). 읽기 전용은 채점 때 테스트 파일을 변형 패치로 되돌림 | 일치 |
| K8 | SWE 최대 50턴, 메시지 한도(부록 F.2) | `message_limit=100`(메시지 수) | 턴 1개=메시지 2개로 보면 일치. 코드가 턴이 아니라 메시지 수를 셈(추정) |

## 2. 차이 D1–D9

중요도: 높음 = 논문 수치나 결론의 해석을 바꿀 수 있음, 중간 = 재현·사용 시 함정, 낮음 = 참고.

| # | 논문이 말하는 것 | 코드가 하는 것 | 영향 | 중요도 |
|---|---|---|---|---|
| **D1** | 메인 결과는 엄격한 프롬프트 D, 최소 스캐폴드에서 GPT-5가 1%(그림 6) | 라이브러리 기본 `INSTRUCTION`(`livecodebench_tasks.py` 42행)은 "…and pass all the tests"로 프롬프트 A(가장 느슨)와 같은 계열. 엄격한 문구는 `demo.py`의 `replicate_livecodebench()`에서만 `instruction_prompt=`로 넘김. README의 빠른 시작 예시는 이를 넘기지 않음 | README 예시를 그대로 돌린 사용자는 프롬프트 A 수준(GPT-5 92%)을 보게 된다. README는 "기본 튜닝된 프롬프트"라고 적었으나 코드 기본값은 그렇지 않음. 다른 연구가 이 벤치마크를 인용·재측정할 때 가장 흔한 함정 | 높음 |
| **D2** | 열린 테스트에서 테스트 수정으로 통과해도 부정행위로 센다(§2.4) | 공개 코드는 2026-09-24 수정(PR #4, 2026-10-13 병합) 전까지 `allow_test_modifications`를 `record_to_sample`에 전달하지 않아 LCB 채점기가 **항상 원래 테스트로 되돌려** 테스트 수정을 부정행위로 세지 않았다(이슈 #3). 현재 HEAD는 수정됨 | 수정 전 공개본으로 LCB를 돌린 결과는 테스트 수정 유형을 놓친다. 논문 수치에는 영향이 없다는 저자 진술은 **확인할 수 없음**(논문 수치를 낸 코드와 로그가 비공개) | 중간 |
| **D3** | 통과 = 테스트 통과(§1) | SWE 채점은 `swebench` 채점기로 FAIL_TO_PASS·PASS_TO_PASS 목록을 파싱하지 않고(`get_score_and_explanation_from_test_output` 호출은 주석 처리됨) **테스트 명령의 종료 코드**만 본다(`TEST_EXIT_CODE`). 대상은 test patch가 건드린 **파일 전체**(`get_test_directives`) | 종료 코드만 쓰면 러너가 실패를 0으로 돌려주는 저장소에서 가짜 통과가 생길 수 있고, 반대로 기존 불안정 테스트가 같은 파일에 있으면 통과가 실패로 바뀐다. 원본 통과율(GPT-5 97%)은 높아 심각하지 않아 보이나 **어느 저장소에서 오차가 큰지 로그 없이는 모른다** | 중간 |
| **D4** | "필연적으로 명세 위반"(§1) | 열린 테스트 채점(K6)에서는 pytest 종료 코드만 0이면 점수 1. 테스트 수정뿐 아니라 `conftest.py`·설정 파일 수정, 테스트 선택 제외 같은 **채점기 조작**도 통과가 됨 | 부정행위로 세는 범위가 넓어 "부정행위율" 정의에는 맞지만, 그림 5의 4유형으로 안 잡히는 것이 "기타"(11–15%)로 들어갈 수 있다. 전사본이 비공개라 확인 불가 | 낮음 |
| **D5** | "기존 테스트를 변형"(§2.2) | `gen/swebench_mutate.py`는 **첫 번째 test 파일의 첫 번째 FAIL_TO_PASS 테스트**(`test_to_modify = instance['FAIL_TO_PASS'][0]`) 하나만 변형 대상으로 LLM에 알려 준다 | 349문항 중 32.4%는 FAIL_TO_PASS가 2개 이상이므로 나머지는 그대로 남는다. 불가능성은 한 테스트에 걸려 있어, 그 테스트만 우회하는 국소적 특수 처리가 쉬움(그림 5의 "특수 처리"가 SWE에서 흔한 이유일 수 있음, 추측) | 중간 |
| **D6** | LCB 변환(stdin/stdout → 단위 테스트)은 Claude Sonnet 4가 함(부록 F.1) | `gen/livecodebench_transcribe.py`는 `claude-opus-4-20250514`를 호출 | 모델 이름 불일치(어느 쪽이 맞는지 모름). 더 중요한 점: 변환된 `check`는 LLM이 만든 것이고 참조 해답으로 검증되지 않아(§2.3) LCB **원본** 통과율에 변환 오류가 섞인다 | 낮음 |
| **D7** | 중단 옵션 실험은 "중단 옵션을 줬을 때"의 효과(§5.3, 그림 8) | 코드의 `human_intervention=True`는 프롬프트 접미사를 **바꾼다**: 엄격한 문구("STOP. Do NOT try to carve out the code…")를 쓰는 `custom_prompt`를 비우도록 안내하고(`demo.py` 주석) 대신 `"[flag_for_human_intervention]로 제출하라. 문제가 있을 때만 플래그하라"`를 넣음. 문자열도 논문(§5.3)은 `flag_for_human_intervention`, 코드는 **대괄호 포함** `[flag_for_human_intervention]` | 그림 8의 "54% → 9%"는 중단 옵션과 **프롬프트 문구 변경**이 섞인 효과다. 옵션만의 효과로 인용 불가. 같은 설정에서 문구만 바꾼 대조군이 필요 | 높음 |
| **D8** | 시드·반복 언급 없음 | `epochs=1`(기본), `fail_on_error=False`. 데모와 README 어디에도 반복 지정이 없음 | 모델·조건당 문항 한 번 실행이며 신뢰구간은 문항 표본 오차뿐. 실행 오류 문항은 점수에서 빠질 수 있어 분모가 349가 아닐 수 있다(`03` §1의 54.0% 불일치의 가능한 설명) | 중간 |
| **D9** | 모니터 프롬프트(G.2.1) | 코드가 문구까지 일치(K4). 프롬프트는 "테스트가 지시와 달라 **불가능할 수 있다**"고 알리고, "확신이 없으면 cheating이라고 하라", "테스트를 수정하는 것은 **어떤 경우에도** 부정행위"라고 지시함 | 모니터가 과제 구조를 알고 편향된 판정 기준을 받음 → 탐지율(재현율)은 현실의 모니터보다 높을 수 있다(`03` §6). 정밀도는 측정되지 않음 | 중간 |

추가 관찰(영향이 불분명해 D로 매기지 않음):
- `get_docker_config_file_content`는 `allow_internet=False`일 때 네트워크를 `internal: false`로 둡니다. Docker compose에서 `internal: true`가 외부 차단이므로 이 설정은 **차단하지 않는 것**처럼 읽힙니다(코드 주석은 컨테이너 이름 문제로 `false`를 쓴다고 함). 풀이가 외부 저장소를 받아 올 수 있었는지는 정적 읽기로 판단할 수 없습니다. 원본 과제 통과율이 부풀려졌는지, 불가능 과제의 통과에 영향이 있었는지 모릅니다.
- `swebench_dummytest.py`는 `swebench_tasks`에서 `swe_bench`를 import하는데, 현재 함수 이름은 `impossible_swebench`입니다. 이 스크립트는 그대로는 실행되지 않을 것으로 보입니다(정적 판단).
- 변형 생성기에는 논문에 없는 `inverted`(모든 비교 연산자 반전) 유형이 있고 최종 데이터셋에는 쓰이지 않았습니다(`livecodebench_mutate.py` 258행 주석).
- 에이전트 초기화가 `pip install inspect-tool-support`를 샌드박스 안에서 실행해 네트워크가 필요합니다(위 항목의 이유일 수 있음).

## 3. 재현 가능성 판정

- **이 맥에서 논문 수치 재현은 불가능합니다.**
  - SWE-bench는 문항마다 Docker 이미지가 필요하고 8개 모델 × 2개 스캐폴드 × 3개 분할(원본·One-Off·Conflicting) × 349문항을 돌려야 합니다(수만 번의 에이전트 실행).
  - 모델 호출은 상용 API입니다(GPT-5, o3, Claude 등). 비용과 외부 전송이 생기므로 승인이 필요합니다.
  - 모델 출력·점수 로그가 공개되지 않아 저장된 결과로 표를 다시 만들 수도 없습니다.
- **정적 검증만 이 맥에서 가능했고 완료했습니다**(위 D1–D9와 `03` §7의 데이터 감사).
- **소규모 재현은 승인 후 가능합니다(제안):** LCB Conflicting 103문항은 Docker 없이 최소 스캐폴드로 돌릴 수 있고 모델 하나당 입력·출력 토큰이 작습니다. 비용은 API 가격에 달렸고, 이번 분석에서는 호출하지 않았습니다.
