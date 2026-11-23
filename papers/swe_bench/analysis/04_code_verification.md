# 04. 논문 대 공식 코드 정적 대조

대상 저장소: `SWE-bench/SWE-bench`(MIT, GitHub `size` 80 MB, 별 6,235, 마지막 push 2026-09-18). **코드는 실행하지 않았습니다**(Docker, conda, API 필요). 아래는 모두 정적으로 읽은 결과이며, 동작 설명은 코드를 읽은 추론이지 실행으로 확인한 것이 아닙니다.

## 0. 어느 판본을 읽었나

| 이름 | 커밋 | 날짜 | 고른 이유 |
|---|---|---|---|
| **paper_b829995** | `b8299951846f4b6ef23176f8c137cffda79bc747` ("Update citation") | 2023-10-11 | arXiv v1(2023-10-10) 바로 다음 날. **저장소에 태그가 없는 시기**라 태그가 아닌 커밋입니다. `repo/paper_b829995/`(1.7 MB) |
| **v2.0.0** | `3730591e9affc4005653e230ef342ab2baa01e4d` | 2024-06-27 | 저장소의 **가장 오래된 태그**(`gh api repos/SWE-bench/SWE-bench/tags`, v2.0.0 ~ v5.0.1, 43개). Docker 하네스가 들어온 첫 태그. `repo/v2.0.0/`(64 MB, 번들 데이터 46 MB 포함) |
| **main** | `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` (v5.0.1은 2026-08-18) | 2026-11-23 조회 | 현재 하네스. 복제하지 않고 `채점·실행 파일 5개만 gh api로 받음` |

- **논문이 쓴 태그라는 기록은 없습니다.** 논문과 저장소 어디에도 하네스 판본이 적혀 있지 않습니다. 논문은 conda 환경 기반 평가(부록 A.3의 "executable contexts as conda envs")를 설명하므로, 논문 시점의 하네스는 `paper_b829995`처럼 conda 방식이고 v2.0.0 이후의 Docker 방식과는 다릅니다.
- 저장소 변경 기록(`v2.0.0/CHANGELOG.md`)은 "1.0.1(2024-03-31)이 첫 작동 버전이며 그보다 오래된 판본은 쓰지 말 것을 강하게 권한다"고 씁니다. 즉 **논문 시점 하네스는 저자들 스스로 작동하지 않는다고 여기는 판본**입니다. 논문 수치가 어떤 판본으로 채점되었는지는 코드에서 알 수 없습니다. 공개 결과 파일(`SWE-bench/experiments`)은 Docker 이후 채점입니다(`03` §2의 "applied" 정의 차이).
- 줄 번호는 위 세 판본 기준입니다.

## 1. 채점 흐름 (논문 부록 A.4 대 코드)

| 논문 A.4 | paper_b829995 | v2.0.0 | main |
|---|---|---|---|
| 1–3 기준 커밋 + 환경 + 설치 | `engine_evaluation.py:106` reset, `:122` install. `context_manager.py:354–385` `git restore`, `git clean -fdx`, `.gitignore` 경로 삭제 | 컨테이너 이미지 | 컨테이너 생성 `run_evaluation.py:288` |
| 4 테스트 패치 적용 | `engine_evaluation.py:123` | `test_spec.py:219–221` (eval 스크립트 안) | 데이터셋의 `eval_script` |
| 5 모델 패치 적용 | `engine_evaluation.py:124` (**테스트 패치 다음**) | `run_evaluation.py:116–132` (**eval 스크립트 앞**) | `run_evaluation.py:288–336` |
| 6 패치 고치기 | `engine_evaluation.py:109–118`, `utils.py:369–396` | 없음 | `GIT_APPLY_CMDS` 4단계 (`run_evaluation.py:54–59`) |
| 7 테스트 실행 | `context_manager.py:513–552` | `test_spec.py:250` | eval 스크립트 |
| 판정 | `metrics/getters.py:45–66`, `metrics/report.py:29–105`, `metrics/metrics.py:78–94` | `grading.py:44–84`, `87–142`, `199–216` | `grading.py:85–110`, `309` |

## 2. 차이와 확인 사항 (D1–D9)

### D1. 모델 패치와 테스트 패치의 적용 순서: 논문 시점과 이후가 정반대

16번 분석(`swe_agent`)에서 넘어온 질문입니다: "채점할 때 모델 패치를 적용한 뒤 테스트 파일을 정답 테스트 패치로 덮어쓰거나 되돌리는가?"

- **논문 시점(`engine_evaluation.py:120–127`):** `run_install_task` → `apply_patch(test_patch)`(123행) → `apply_patch(prediction)`(124행) → `run_tests_task`(125행). 테스트 패치가 **먼저**, 모델 패치가 **나중**입니다. 논문 A.4도 4단계 테스트 패치, 5단계 모델 패치로 같은 순서입니다. 덮어쓰기나 되돌리기는 없습니다. 적용은 `git apply -v <patch>`(`context_manager.py:487`)이고 실패하면 로그에 `Patch Apply Failed; (<유형>)`을 남기고 `False`를 돌려줍니다(502–505행).
- **v2.0.0 이후:** 모델 패치가 **먼저**(`run_evaluation.py:116–132`) 컨테이너에 적용되고, 그 뒤 eval 스크립트(`test_spec.py:211–253`)가 **테스트 파일을 기준 커밋 상태로 되돌린 다음**(`git checkout <base_commit> <테스트 패치가 `--- a/`로 가리키는 파일>`, 216–218행) 테스트 패치를 적용하고(219–221), 테스트를 돌리고(250), **끝에서 다시 같은 되돌리기**(251)를 합니다. 주석(251행)은 "저장소를 이전과 같은 상태로 남긴다"입니다.
- 결론: **논문 시점에는 모델이 테스트 파일을 고쳐도 되돌리지 않고, v2.0.0부터는 테스트 패치가 건드리는 파일에 한해 모델의 변경을 버립니다.** 같은 모델 패치라도 하네스 판본에 따라 결과가 달라질 수 있고, 어느 판본이 논문 수치를 낸 것인지 모르는 이상(§0) 모델이 테스트를 고친 경우의 논문 수치는 해석할 수 없습니다.

### D2. 모델이 테스트 파일을 고쳤을 때

세 판본의 정적 읽기입니다. 실행해 본 것은 없습니다.

- **논문 시점:**
  - 모델 패치가 테스트 패치가 바꾼 줄과 겹치면 `git apply`가 전체 패치를 원자적으로 거부해 해당 문항은 0점입니다. 첫 시도(`pred_try`)는 **깨끗한 기준 커밋**에 모델 패치를 적용해 보고 되돌린 뒤(`engine_evaluation.py:109–117`), 테스트 패치 뒤에 다시 적용합니다. 그래서 기준 커밋에서는 적용되지만 테스트 패치와 충돌하는 모델 패치가 존재할 수 있고 그 경우 0점입니다.
  - 겹치지 않는 테스트 파일 변경(같은 파일의 다른 위치, 테스트 패치가 건드리지 않은 `conftest.py`나 보조 파일)은 **그대로 남아 실행됩니다.** 테스트 명령이 실행하는 파일은 테스트 패치가 가리킨 파일뿐이지만(`utils.py:187–215`, `context_manager.py:100–102`), 이 파일이 import하는 보조 모듈의 변경은 적용됩니다.
- **v2.0.0:** `DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"`(`test_spec.py:27`)로 되돌릴 파일을 고릅니다. 이 정규식은 **기준 커밋에 이미 있던 파일만** 잡습니다(새로 만든 테스트 파일의 헤더는 `--- /dev/null`). 따라서 (a) 기존 테스트 파일에 대한 모델 변경은 버려지지만, (b) 테스트 패치가 **새로 만드는** 파일을 모델이 같은 경로로 먼저 만들어 두면 되돌리기 대상에 없고 `git apply -v`는 "이미 존재" 오류를 낼 수 있는데, eval 스크립트는 `set -uxo pipefail`로 `-e`가 없어(`test_spec.py:51`) 실패해도 계속 진행하고 테스트가 모델이 만든 파일로 실행될 수 있습니다. 이 경로가 실제로 어떻게 동작하는지는 실행하지 않아 모릅니다. 노출 규모는 공개 데이터로 셀 수 있습니다: 테스트 패치가 새 파일을 만드는 문항은 113개(4.9%), 그중 기존 파일 변경 없이 새 파일만 만드는 문항이 17개입니다(`recompute_results.json`의 `test_patch_structure`). 모델이 이 경로를 실제로 쓴 사례가 있다는 증거는 없습니다. (c) 테스트 패치와 상관없는 보조 파일(`conftest.py` 등)의 모델 변경은 되돌리기 대상이 아닙니다.
- **main:** eval 스크립트가 데이터셋 필드로 옮겨졌습니다(`swebench/harness/utils.py:213–269`). 같은 파일의 `record_test_exit_code` 독스트링(225–227행)이 "eval 스크립트는 테스트 파일을 되돌리는 `git checkout`으로 끝난다"고 쓰는 것으로 v2.0.0 방식이 이어졌음을 확인했습니다. 스크립트 본문은 데이터셋에 있어 이 분석에서 읽지 못했습니다.

### D3. 패치 고치기와 "적용됨" 판정

- 논문 A.4 6단계: 적용이 실패하면 불필요한 문맥 줄을 지우고 헤더를 다시 계산해 한 번 더 시도합니다. 코드 `engine_evaluation.py:109–118`과 `utils.py:369–396`(`extract_minimal_patch`)이 일치합니다.
- **채점 쪽(`metrics/getters.py:60`)은 로그에 `Patch Apply Failed`가 한 번이라도 있으면 "적용 안 됨"으로 처리합니다.** 첫 시도(`pred_try`)가 실패하고 고친 패치가 성공한 경우에도 첫 시도의 실패 줄이 로그에 남으므로(`context_manager.py:502`), 이 코드 경로만 읽으면 **고친 패치는 채점에서 적용 안 됨이 되는 것으로 보입니다.** 이는 공개 결과 파일에서 확인한 사실과 맞습니다: 2023년 4개 모델의 `applied` 수가 표 14의 Applies − Fixed와 정확히 같습니다(`03` §2). 단 이는 정적 추론입니다. 논문이 낸 수치는 이 경로로 계산하지 않았을 수 있습니다.
- v2.0.0은 패치 고치기가 없고 `git apply` 한 번만 씁니다(`run_evaluation.py:116–132`). main은 `git apply`, `--3way`, `--reject`, `patch --fuzz=5`를 차례로 시도하고 마지막에 역적용 검사까지 합니다(`run_evaluation.py:54–59`, `301–336`). 즉 **"패치가 적용된다"의 정의가 세 판본에서 모두 다릅니다.** 논문의 % Apply와 이후 리더보드의 % Apply는 비교할 수 없습니다.

### D4. 테스트 명령과 PASS_TO_PASS의 범위

- 테스트 명령은 저장소별 테스트 도구 + **테스트 패치가 건드린 파일 목록**입니다(`context_manager.py:100–102`, `utils.py:187–215`: `diff --git a/.* b/(.*)`에서 비테스트 확장자 제외, django는 모듈 경로로 변환). **전체 테스트 스위트를 돌리지 않습니다.**
- 따라서 PASS_TO_PASS는 "기존에 통과하던 모든 테스트"가 아니라 "테스트 패치가 건드린 파일 안에서 통과하던 테스트"입니다. 논문 본문 §2.3 "median of 51 additional tests ... whether prior functionality is properly maintained"는 이 범위를 말하지 않습니다. 이 때문에 정답 패치가 고친 코드가 다른 모듈에 끼치는 영향은 회귀 검사에 잡히지 않을 수 있습니다(이론적 경로이며 실제 빈도는 이 분석에서 측정하지 못함). 이후 문헌의 약한 테스트 결과(`03` §6)와 같은 방향입니다.

### D5. 테스트 상태 판정: 없는 테스트는 실패 (논문과 일치), 단 주석이 반대

- `getters.py:127–131`: `test_passed = case in sm and sm[case] == PASSED`, `test_failed = case not in sm or sm[case] in {FAILED, ERROR}`. 없는 테스트는 실패로 칩니다. 논문 A.4 "If a test is missing or has a non-pass status, it is considered a fail"과 일치합니다.
- 다만 `report.py:56–57`의 주석은 "Assume silent success for now (test case not in eval_sm)"로 반대 동작을 말합니다(코드는 실패 처리). v2.0.0 `grading.py:119–120`에도 그대로 남아 있습니다. 코드가 맞고 주석이 낡았습니다.

### D6. 빈 집합 규칙과 SKIPPED: 논문에 없는 두 허점

- **비어 있는 PASS_TO_PASS는 만점 1입니다.** `metrics.py:27–31`, v2.0.0 `grading.py:178–196`은 `total == 0`이면 `return 1`이고 "TODO: Don't factor in p2p metrics" 주석이 있습니다. 공개 test 분할에서 P2P가 빈 문항은 67개(2.9%)이며 이 문항에는 회귀 검사가 없습니다(`03` §8).
- **SKIPPED 상태의 FAIL_TO_PASS 테스트는 성공도 실패도 아닙니다.** 논문 시점 `test_failed`는 FAILED·ERROR만 실패로 세고(`getters.py:129–131`), SKIPPED는 어느 목록에도 들어가지 않습니다. 그러면 모든 F2P 테스트가 SKIPPED일 때 F2P의 성공+실패 합이 0이어서 위 `return 1`이 적용되어 `RESOLVED_FULL`(`metrics.py:90`)이 될 수 있습니다(P2P도 통과해야 함). **main 쪽 주석이 이 허점을 직접 인정합니다**: `grading.py:106` "a skipped F2P test is not a resolution; without this, a patch that makes every F2P test skip lands in neither list and scores RESOLVED_FULL", 그리고 P2P는 SKIPPED를 회귀로 안 보는 별도 함수 `test_maintained`(`grading.py:92–102`). 논문 제출 결과에서 이 경로로 해결 판정된 문항이 있었는지는 모릅니다(실행 로그가 없음).

### D7. 해결 정의

- `metrics.py:78–94`: F2P 비율 = 1 그리고 P2P 비율 = 1이면 `RESOLVED_FULL`, F2P가 0과 1 사이이고 P2P = 1이면 `RESOLVED_PARTIAL`, 그 밖은 `RESOLVED_NO`. 논문의 "모든 F2P·P2P 통과"(A.4)와 FULL이 일치합니다. 논문 부록 C.5의 여섯 분류(Resolved, Breaking Resolved, Partially Resolved, Work in Progress, No-Op, Regression, 표 22–23)는 **공개 코드 어디에도 구현이 없습니다**(`paper_b829995`, `v2.0.0/swebench` 검색에서 해당 이름 없음). 표 23의 숫자는 코드로 재현할 수 없고 수치 검증이 불가능합니다.

### D8. 데이터 판본

- 논문 표 11과 공개 데이터가 맞지 않는 곳(xarray의 P2P 평균 297.5 대 543.1, requests의 F2P 7.6 대 7.0, 편집 줄 평균)은 데이터 쪽 차이입니다. v2.0.0에 번들된 `data/swebench_updated.json`(2024-06-27)은 현재 HF 데이터와 2,291/2,294문항이 F2P·P2P·패치·테스트 패치까지 같고(`03` §1), xarray는 이미 543.1입니다. 즉 **논문 이후 2024-06 이전에 데이터가 한 번 개정되었고 이후는 안정**입니다. `paper_b829995`에는 데이터가 없어(HF에서 받는 구조) 논문 당시 데이터와 직접 비교하지 못했습니다. 개정 내용을 설명하는 변경 기록은 없습니다.

### D9. 검색과 생성 설정

- **BM25 색인은 테스트 파일을 제외합니다.** `inference/make_datasets/bm25_retrieval.py:131`의 `list_files(cm.repo_path, include_tests=False)`. 논문 D.1은 파일 경로를 문서 앞에 붙인다는 것만 쓰고 테스트 제외는 oracle("excluding test files")에만 쓰여 있습니다. 질의당 상위 20개를 받습니다(275행).
- **생성 온도:** 논문 D.2는 "greedy decoding for all models"입니다. `inference/run_model.py:124–125`의 기본값은 `temperature=0.2`, `top_p=0.95`이고 `run_live.py:220`은 `temperature=0, top_p=1`입니다. 논문 수치를 낸 인자는 저장소에 기록되어 있지 않아, 어느 쪽이 쓰였는지 확인할 수 없습니다. 실행 스크립트가 온도를 지정했을 수 있습니다.

## 3. 요약

| 논문의 주장 | 코드 대조 결과 |
|---|---|
| 모델 패치는 테스트 패치 적용 뒤에 붙인다(A.4 4–5단계) | 논문 시점 코드와 일치. v2.0.0 이후는 순서가 뒤바뀌고 테스트 파일을 되돌림 (D1) |
| 모델이 테스트를 고쳤을 때의 처리 | 논문에 없음. 시점별로 다르고 허점이 있을 수 있음 (D2) |
| 해결 = F2P와 P2P 전체 통과 | 코드 일치. 단 빈 P2P는 1, SKIPPED F2P는 판정 구멍 (D6, D7) |
| "회귀 검사 중앙값 51개" | 테스트 패치가 건드린 파일 안의 테스트만 (D4) |
| % Apply | 판본마다 정의가 다름 (D3) |
| 표 22–23 분류 | 코드에 구현 없음 (D7) |

## 4. 재현 제안 (실행하지 않았고 승인이 필요)

- **R1:** 공식 Docker 하네스로 Lite의 정답 패치를 채점해 `resolved` 300/300이 되는지, 그리고 일부러 만든 패치(테스트 파일 수정, 모든 F2P를 skip 처리, 빈 패치)에 대한 판정을 확인합니다. Docker, 수십 GB 이미지, CPU 시간이 필요해 이 저장소의 CPU·MPS 토이 규칙 밖입니다(맥에서는 x86 이미지 에뮬레이션). 승인과 자원 확인이 먼저입니다.
- **R2:** D6의 SKIPPED 허점은 하네스 없이 `grading.py`의 순수 함수(`get_eval_report`, `get_resolution_status`)에 가짜 상태 맵을 넣어 확인할 수 있습니다. 다만 공식 코드를 불러 실행하는 것이라 이 저장소 규칙상 승인 후에 합니다. 이번에는 하지 않았습니다.
