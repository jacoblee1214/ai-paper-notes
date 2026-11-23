# 04. 논문 대 공식 코드 대조 (D1–D9)

대상: `github.com/ulab-uiuc/AgentDebug` 커밋 `7740fe3a5c4822b2143cbde78ecfffeace0bb166`(MIT, 2026-03-30 push, "docs: update README.md" 한 커밋만 얕게 clone). `repo/`에 두었고 **정적으로만 읽었으며 실행하지 않았습니다**(API 키·환경 설치 필요). 논문 판은 arXiv v1.

저장소 구성: `detector/`(1단계 `fine_grained_analysis.py` 710줄, 2단계 `critical_error_detection.py` 535줄, `error_definitions.py` 191줄), `agentdebug/`(엔진, 환경, 롤아웃, 메모리). `examples/`와 `docs/`는 README가 말하지만 저장소에 없습니다.

| # | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| D1 | 2단계 방식 | §3.2 제목 "Critical Error Detection via Counterfactuals": 각 단계에 교정 행동을 대입해 성공 여부 시험. 알고리즘 1 주석과 그림 15는 "no rollout/counterfactuals", 단일 프롬프트 | `critical_error_detection.py::identify_critical_error`가 `call_llm` 한 번으로 JSON(`critical_step`, `critical_module`, …)을 받음. 반사실 롤아웃 코드 없음 | 본문 §3.2의 서술은 코드와 다르고 알고리즘 1·부록 프롬프트와 같음. 반사실 탐색은 기준선 Brute Force·Binary Search에만 해당 |
| D2 | 성공 라벨 사용 | 알고리즘 1: Eval(τ)=1이면 종료. 그림 15 프롬프트 "TASK RESULT: FAILED" | 1단계가 `task_success`(metadata의 `success`/`won`)를 받고, 2단계 프롬프트에 `TASK RESULT: FAILED`를 고정 삽입. 성공 궤적이면 `None` 반환 | 디버거는 환경의 성공/실패 판정을 입력으로 씀. 실패 판정이 없는 배포 상황에는 그대로 못 씀 |
| D3 | 3단계 루프 | 알고리즘 1 3단계: 재롤아웃 → Eval → UpdateFeedback 반복(최대 I) | 반복 오케스트레이터가 **저장소에 없음**. `agentdebug/environments/base.py`에 재생(replay) 훅(`setup_replay`, `get_debugger_feedback`, 지속 지침 `persistent_guidance`)이 있으나 `setup_replay`를 부르는 곳이 없음. `critical_error_detection.py` 헤더 주석은 "No scoring, no agent feedback - just critical error identification". `UpdateFeedback`에 대응하는 함수 없음 | 하위 과제 성공률(그림 5–7)을 만드는 코드 경로를 코드로 확인할 수 없음. 피드백은 `correction_guidance` 문자열을 관찰 뒤에 이어 붙이는 방식(`env_manager.py`: `obs = obs + "\n\n" + debugger_feedback`)으로 주입됨 |
| D4 | 기준선과 예산 | §4.2: Self-Refine, Vanilla Debugger, ToT, Best-of-N을 "총 토큰 사용량으로 맞춤". §4.1: Direct Prompting, Brute Force, Binary Search | 롤아웃에 `--best_of_n`(기본 1, 온도 1.2)만 있고, 나머지 기준선과 토큰 집계는 없음. `agentdebug/engines`는 `max_tokens` 인자만 다루고 사용량을 기록하지 않음 | 같은 예산이라는 주장은 코드로도 표로도 확인 불가 |
| D5 | 평가 스크립트 | 표 1의 S, S+M, ALL | 정확도 계산·사람 라벨과의 비교 코드 없음. 공개 데이터는 Google Drive(저장소 밖) | 표 1은 코드로 재현 불가. 우리는 표 값 격자 검사와 미러 데이터 분포로만 대조(`03`) |
| D6 | 규칙 기반 개입 | 논문에 없음 | `fine_grained_analysis.py::_check_system_errors`: `step_num >= 30`이면 LLM 호출 없이 `system/step_limit` 오류를 확정(주석 "Common step limit in ALFWorld"). 환경 구분 없이 모든 환경에 적용. 2단계는 step 1의 memory·reflection이 임계로 나오면 최대 2번 재질의(`retry_count < 2`)하고, 그래도 실패하면 `analysis_failure`를 반환 | 그러나 `StepAnalysis`에는 system 필드가 없어(`memory/reflection/planning/action`만) 이 시스템 오류는 `step_summary` 문자열에만 남고 2단계 프롬프트(`errors` 딕셔너리만 사용)에는 전달되지 않음. 즉 1단계의 system 판정(`step_limit`, `tool_execution_error` 등)은 코드상 2단계 입력에 반영되지 않으며, 2단계가 system 모듈을 고르려면 LLM이 스스로 판단해야 함(프롬프트는 system을 허용). ALFWorld 실패는 100%가 30턴까지 감(`03` §3). "step 1에는 memory·reflection이 없다"는 사전 지식이 재질의로 들어가 있어 논문 프롬프트(그림 15)에는 있는 규칙이지만 재질의 절차는 논문에 없음 |
| D7 | 분류 체계 | 표 2: 5모듈 17유형, action은 3개(Planning–Action Disconnect, Format, Parameter) | `error_definitions.py`: memory 3, reflection 4, planning 3, **action 4**(`invalid_action` 추가), system 4, 그리고 **`others` 모듈**(유형 `others`, 2단계 프롬프트가 "Others category … Do NOT ignore"라 안내). 합 18유형 + others. README는 "17 error types". 주석 데이터에도 `invalid_action` 1건 | 문서의 17과 코드의 18이 다름. `others`는 논문의 분류 체계에 없는 탈출구 |
| D8 | 단계 번호 | 알고리즘은 t=1..T. 그림 16 Vanilla 프롬프트는 "step number (0-based)" | 탐지기는 어시스턴트 메시지를 세어 **1-based**(`assistant_count`), 재생 훅은 `debugger_feedback_step`을 "0-based"로 문서화(`base.py`). 둘을 잇는 코드가 없음(D3) | 임계 단계 번호를 피드백 위치로 쓸 때 1칸 어긋날 위험. 어느 쪽이 맞게 연결됐는지 확인 불가 |
| D9 | 모델과 온도 | GPT-4.1, temperature 0(§4.1) | 설정 기본 temperature 0.0, README 예시 모델은 `gpt-4o-mini`. 롤아웃 기본 temperature 0.7, Best-of-N 온도 1.2 | 논문 설정은 코드 기본값이 아니라 사용자가 지정해야 함. 시드 인자는 롤아웃에만 있음 |

## 코드에서 확인된 것 (논문과 같음)

- 모듈형 롤아웃 프롬프트(`<memory>`, `<reflection>`, `<plan>`, `<action>`)가 부록 그림 17–19와 같은 형식(`agentdebug/environments/prompts/`).
- 1단계는 단계×모듈마다 독립 LLM 호출(`detect_module_errors`, 모듈별 "이 출력이 정의와 일치하는가"). 한 궤적의 호출 수는 약 4 × 단계 수(step 1은 memory·reflection 생략)입니다.
- 2단계 프롬프트가 부록 그림 15와 같은 구조("HOLISTIC, GLOBAL perspective", "EARLIEST point", JSON 출력).
- 탐지 결과는 **임계 단계 하나**(알고리즘의 집합 T*, min 연산이 아니라 단일 정수 `critical_step`)입니다.

## 데이터 (D의 일부가 아닌 별도 사항)

- 공식 데이터는 README의 Google Drive 링크뿐이며 이 환경에서 받지 않았습니다. 제3자 HF 미러(`davide221/agenterrorbench`)는 모듈·단계 분포가 논문 그림 11–13과 일치하는 200행이지만 라벨에 결함이 있습니다(`03` §3: 빈 유형 27건, 철자 변형, 궤적당 주석 1개).

## 판정

- **코드는 탐지(1·2단계)까지만 공개**돼 있고, 하위 과제 성공률을 내는 3단계·기준선·평가는 없습니다. 이 논문의 두 번째 기여(회복 실험)는 코드로 재현할 수 없고, 표 1(탐지)도 평가 스크립트가 없어 재채점이 불가합니다.
- 인용할 때 "공식 코드가 공개되어 있다"만 쓰면 오해를 줄 수 있습니다. 공개 범위는 탐지 프롬프트와 환경 재생 훅입니다.

## 재현 제안 (실행하지 않음, 승인 필요)

- 탐지 재현: 미러 또는 공식 데이터의 200궤적에 GPT-4.1을 호출하는 방식이라 **LLM API 필요**(궤적당 약 수십~백 번의 1단계 호출 × 200). CPU로는 실행할 수 없고 비용이 큽니다. 한 환경 50궤적 부분 재현이면 호출 수가 줄지만 승인이 필요합니다.
- 회복 실험 재현: 3단계 루프를 새로 짜야 하며 ALFWorld 설치와 에이전트 LLM 호출이 필요. 필요하다면 같은 예산 기준선과 "피드백 없는 같은 단계 재실행" 대조군을 함께 설계해야 합니다.
