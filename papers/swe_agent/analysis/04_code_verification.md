# 04. 논문과 공식 코드 대조 (정적 읽기)

대상: `SWE-agent/SWE-agent` 태그 **v0.3.0**, 커밋 `43b8de5a634136bc47564c2d07b3e24670451103`(2024-05-01, 논문 v1 제출 5일 전). `repo/`에 얕게 clone했다(4.2 MB). 실행, 설치, Docker, API 호출은 하지 않았다. 논문이 어느 태그로 돌렸는지는 어디에도 적혀 있지 않다. 이 태그를 고른 이유는 릴리스 날짜(2024-05-02)가 논문 제출 직전이기 때문이며, v0.4.0(2024-05-09) 이후와의 차이는 보지 않았다. 따라서 아래 "차이"는 이 태그에서의 차이이고 논문 실험 시점의 코드와 같다는 보장은 없다.

## 1. 일치하는 것

| # | 논문 | 코드 | 위치 |
|---|---|---|---|
| K1 | 창 100줄, 줄 번호 접두, 위아래 생략 줄 수 표시(§3) | `WINDOW: 100`, `_print`가 `[File: path (N lines total)]`, `(k more lines above/below)`, `grep -n $`로 줄 번호 | `config/default.yaml`, `config/commands/defaults.sh` |
| K2 | `edit n:m`이 줄 범위를 한 번에 교체하고 뷰어가 자동 갱신(§3) | `edit()`가 줄 범위 교체 후 `_print` | `config/commands/edit_linting.sh` |
| K3 | 편집에 linter 포함, 오류면 편집 폐기하고 "편집 전후"를 보여 줌(§3, 그림 6) | flake8로 검사 후 오류면 백업에서 복구하고 "would have looked", "original code before your edit"를 출력 | 같은 파일 |
| K4 | 출력이 없으면 전용 메시지(§3) | `next_step_no_output_template`: "Your command ran successfully and did not produce any output." | `config/default.yaml` |
| K5 | 최근 5개 관찰 이전은 한 줄로 접음(§3) | `Last5Observations`: 첫 관찰과 마지막 5개를 남기고 나머지를 "Old output omitted (N lines)"로 | `sweagent/agent/history_processors.py` |
| K6 | 명령 목록(표 4): open, goto, scroll_up/down, search_file, search_dir, find_file, edit, create, submit | 모두 존재 | `config/commands/*.sh` |
| K7 | 데모 1개 사용, 데모 없음 어블레이션(표 1, 3) | `demonstrations:` 한 개(marshmallow-1867, SWE-bench Lite 12개 저장소에 없음) | `config/default.yaml` |
| K8 | 매 단계 thought + command(ReAct) | `ThoughtActionParser`, 시스템 프롬프트가 "discussion과 command 한 개씩" 요구 | `config/default.yaml` |
| K9 | 모델 GPT-4 Turbo = gpt-4-1106-preview, Claude 3 Opus = claude-3-opus-20240229 | `SHORTCUTS`의 `gpt4`와 `claude-opus`가 같은 이름으로 매핑, Claude 응답 상한 4,096 토큰 | `sweagent/agent/models.py` |
| K10 | `submit`이 편집 전체를 패치로 만듦 | `git add -A; git diff --cached > model.patch` | `defaults.sh` |

## 2. 차이 (D1–D9)

| # | 논문 | 코드(v0.3.0) | 영향 |
|---|---|---|---|
| D1 | 검색은 결과가 **50개**를 넘으면 결과를 내지 않고 더 구체적인 쿼리를 제안(§3) | `search_dir`는 일치 **파일 수 100 초과**, `search_file`은 일치 **줄 수 100 초과**일 때 "Please narrow your search"(`search.sh`). 50이라는 숫자는 코드에 없음 | 논문 서술이 코드와 다른 상수. 어블레이션의 "요약 검색"이 어느 쪽인지 코드에서 가릴 수 없음 |
| D2 | 인스턴스당 예산 $4(§4) | 코드 기본값 `per_instance_cost_limit=3.0`(`run.py`), `scripts/run.sh`도 3.00, README 예시 2.00. 반면 README의 평가 로그 예시 폴더명에는 `c-4.00`이 있음 | $4는 사용자가 지정한 값으로 보이며 코드 기본값이 아님. 재현하려면 4.00을 명시해야 함 |
| D3 | temperature 0.0, top-p는 언급 없음(부록 B.1) | `run.py` 기본값 temperature 0.0, **top_p 0.95**. `scripts/run_and_eval.sh`는 temperature 0.2. `models.py` 데이터클래스 기본은 1.0, 1.0 | 논문에 top-p가 없다. 6회 실행 변동(표 10)의 설정에 top-p 0.95가 들어간다고 가정해야 함 |
| D4 | 형식에 어긋난 출력은 오류 메시지를 보내고 **올바른 출력이 나올 때까지** 반복, 지난 오류 메시지는 첫 번째만 남김(§3) | `check_format_and_requery`는 형식 실패와 차단 명령이 합쳐 **3회**(`while ... <= 2`) 넘으면 `exit_format`으로 종료. 재시도 메시지는 `temp_history`에만 쓰이고 영구 기록에 남지 않음(첫 번째도 안 남음) | 논문 서술과 다름. 표 13의 early exit가 전체에서 1건뿐인 것과는 일관됨 |
| D5 | 표 4: `scroll_down`은 "moves the window **up** 100 lines", `scroll_up`은 "down" | 코드 설명은 "moves the window down {WINDOW} lines", 실제 이동은 `WINDOW - OVERLAP` = 98줄(`OVERLAP: 2`) | 표 4의 방향 표기는 오타이고 이동량은 98줄. 모델에게 보이는 설명은 코드 쪽 |
| D6 | linter는 "select errors"를 보여 줌(§3), 종류는 미기재 | `flake8 --select=F821,F822,F831,E111,E112,E113,E999,E902`(미정의 이름 등 일부), **`.py` 파일에만** 적용 | 문법·들여쓰기·미정의 이름만 잡고 의미 오류는 못 잡음. 다른 언어에서는 가드레일이 없음. HumanEvalFix의 JS·Java 결과(표 2)에는 linter가 작동하지 않았을 가능성이 크다(설정이 같다고 부록 B.7이 말함) |
| D7 | 표 3의 변형(반복 검색, 편집 도구 없음, linting 없음 등) | 이 태그의 `config/commands/`에는 `defaults.sh`, `edit_linting.sh`, `search.sh`와 "cursors" 변형뿐이다. 반복(iterative) 검색과 linting 없는 edit 명령 파일은 없다. 창 크기, 전체 기록, 데모 없음은 설정 변경으로, 편집·검색 도구 제거는 `command_files`에서 파일을 빼는 것으로 가능해 보인다(실행하지 않음) | 어블레이션 8칸 중 2칸(반복 검색, linting 없음)은 공개 코드만으로 재구성되지 않는다 |
| D8 | (언급 없음) | 차단 명령 목록: `vim vi emacs nano nohup git`, 단독으로는 `python python3 ipython bash sh exit`. 차단되면 "Interactive operation ... is not supported" | `git`이 막혀 있어 에이전트가 git 기록에서 정답 힌트를 찾을 수 없다. 논문에 적히지 않은 설계 요소이고 결과에 영향을 줄 수 있음 |
| D9 | (언급 없음) | `run.py`에 `"FAIL_endTO_PASS"`라는 오타 키(338행) → 이 조건은 항상 거짓이라 `tests`가 비어 있음. 기본 프롬프트는 `{files}`, `{test_files}`, `{tests}`를 쓰지 않음 | 실행에 영향이 없다(정답 패치나 테스트 목록은 에이전트에 전달되지 않음). 다만 환경의 `record`에는 정답 패치가 들어 있어(`setup_args`) 템플릿을 바꾸면 새어 들어갈 수 있는 구조 |

## 3. 환경 관련 보조 확인

- `reset(apply_test_patch=False)`: 정답 테스트 패치를 에이전트 환경에 미리 적용하는 "oracle setting" 옵션이 있고, 기본은 꺼짐이다. 켜면 `submit`이 `git apply -R`로 되돌린다. 논문 실험이 꺼진 상태였다는 것은 코드만으로 확인할 수 없고(설정 인자를 이 태그에서 못 찾았다) 논문 서술상 꺼진 것으로 보인다.
- 환경 구성 시 `pip install flake8`을 컨테이너 안에서 실행한다(재현에는 네트워크 필요).
- 환경에서 상태를 유지하는 셸 세션(`CURRENT_FILE`, `CURRENT_LINE` 변수)을 쓴다. 후속 mini-swe-agent는 이 상태 유지를 일부러 버렸다(`02` §3).

## 4. main(1.x)과의 차이 (짧게)

현재 main은 논문 구현과 구조가 다르다. 확인한 것은 `gh api`로 본 목록과 설정 일부뿐이다.

- 도구가 `tools/` 아래 묶음(`windowed`, `windowed_edit_linting`, `windowed_edit_replace`, `windowed_edit_rewrite`, `search`, `submit`, `review_on_submit_m`, `filemap`, `forfeit`, `web_browser`, `image_tools` 등)으로 나뉘고 함수 호출(function_calling) 형식을 쓴다.
- `config/sweagent_0_7/`에 0.7 시점 구성(`07.yaml`, `07_fcalling.yaml` 등)이 남아 있다.
- `search_dir`의 파일 수 제한 100은 그대로다(`tools/search/bin/search_dir`).
- README는 개발이 mini-swe-agent로 옮겨 갔고 mini-swe-agent가 "SWE-agent를 대체했으며 성능이 맞먹는다"고 쓴다. 논문의 재현에는 v0.x 태그를 써야 하고, main에서 같은 숫자가 나온다고 가정하면 안 된다.

## 5. 후속 코드가 말해 주는 것

mini-swe-agent(같은 팀)의 README는 "도구 중심 설계는 2024년에 중요했지만 LM이 강해지며 상당 부분이 필요 없다"고 쓴다. 이는 저자 측 의견이며, 같은 모델에서 도구 유무를 맞춘 어블레이션을 이 분석에서 찾지 못했다. 논문의 ACI 원칙이 "모델이 약할 때 크다"는 조건부 주장으로 읽혀야 함을 시사한다.

## 6. 재현 제안 (실행 안 함, 사용자 승인 필요)

| 제안 | 필요한 것 | 비고 |
|---|---|---|
| 공개 궤적(S3, `trajs`)과 평가 로그로 Opus 전체 실행의 해결 수 확정(213 대 241 등) | 다운로드(크기 미확인). 이 분석에서는 안 받음. 크기를 먼저 확인할 것 | CPU로 가능한 읽기 작업. 우리 03 §6의 불일치를 푼다 |
| Lite GPT-4 궤적에서 편집 실패 직후 회복률(90.5 대 57.2%) 재계산 | 궤적 파일 | CPU 가능 |
| v0.3.0 설정으로 소수 인스턴스 실행해 비용·형식 오류 확인 | Docker, SWE-bench 인스턴스 이미지(인스턴스당 수 GB), LLM API | API 승인 필요, GPU 불필요하지만 비용 발생 |
