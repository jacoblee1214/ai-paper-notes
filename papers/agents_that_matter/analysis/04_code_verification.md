# 04 논문 대 공식 코드 정적 대조

대상: `github.com/benediktstroebl/agent-evals`(논문 부록 I 각주 19, MIT). 커밋 `5ce7c0c`(2025-05-28, `--depth 1`로 clone). 저장소는 `repo/`에 두었고 **실행하지 않았습니다**(API 호출, 설치 없음). 대조는 코드와 저장된 결과 파일을 읽은 것입니다.

| 항목 | 값 |
|---|---|
| gh API 크기 | 84.5 MB(전송 약 84 MB), 작업 트리 1.5 GB(LATS 출력 로그 931 MB, LDB 319 MB, STeP 68 MB 등), 커밋 52개(clone은 얕아 이력은 읽지 않음) |
| 라이선스 | MIT. 하위 디렉터리는 LDB(Apache-2.0), LATS(MIT), Reflexion(MIT), STeP(MIT), WebArena(별도 `webarena_license.txt`)의 복사본 |
| 구성 | `LLMDebugger/`(LDB + 우리가 본 기준선 `simple*.py`), `LanguageAgentTreeSearch/`, `reflexion/`, `webagents-step/`, `webarena/`, `HotpotQA/`, `NovelQA/`, `humaneval_eval.ipynb`, `hotpotqa_eval.ipynb`, `plot_data_0409.csv`, `plot_data_0613.csv`, `pareto_frontier_example.ipynb` |
| 결과 | 각 에이전트 디렉터리의 `output_data/`에 실행 5회의 jsonl(과제별 `is_solved`)과 로그. 웹아레나 궤적은 Dropbox(받지 않음) |
| 웹앱 | `benediktstroebl/agent-eval-webapp`(41 KB, 라이선스 없음, 2024-09-27) |

## 논문과 일치하는 항목

| 번호 | 논문 | 코드 위치 | 판정 |
|---|---|---|---|
| K1 | Retry: 최대 5번, 온도 0, 예시 테스트 실패 시에만 재시도(부록 A.1) | `LLMDebugger/programming/simple_repeat.py` `while step < 5`, `exe.execute(cur_func_impl, tests_i)`로 중단, `get_seed` 기본 `temperature=0.0` | 일치 |
| K2 | Warming: 온도 0, 0.3, 0.3, 0.5, 0.5 | `simple_incr_temp.py` `step in [0]` 0.0, `[1, 2]` 0.3, `[3, 4]` 0.5 | 일치 |
| K3 | Escalation: Llama-3 8B → GPT-3.5 → Llama-3 70B → GPT-4, 온도 0 | `simple_boosting.py` `while step < 4`: llama3-8b, gpt-3.5-turbo-0125, llama3-70b, gpt-4-turbo-2024-04-09 | 일치 |
| K4 | 세 기준선은 예시 테스트로만 멈추고 최종 채점은 전체 테스트 | 위 세 파일에서 `tests_i = item['given_tests']`를 중단 조건에, `item["test"]`를 `is_solved`에 사용. 기본 `pass_at_k=1`이라 숨은 테스트가 재시도를 조기 종료시키지 않음(`main.py` 기본값, 실행 스크립트가 지정하지 않음) | 일치. 정보 누출 없음 |
| K5 | 실행 5회 | `LLMDebugger/programming/run_*.sh`가 `run1 … run5`를 순회하고, Reflexion과 LATS는 `output_data/`에 `run1`–`run5` 폴더가 있음(스크립트는 읽지 않음) | 일치 |
| K6 | LDB 최대 반복 10, 온도 0 | `main.py --max_iters` 기본 10, `run_ldb.sh`가 `--max_iters "10"`, `py_generate.py`의 디버그 호출이 `temperature=0` | 일치 |
| K7 | LATS 반복 8, 생성 온도 0.8 | `run_lats_gpt4.sh`의 `--max_iters "8"`, `py_generate.py:267` 기본 `temperature: float = 0.8` | 일치(내부 테스트 수와 확장 계수는 읽지 않음) |
| K8 | Reflexion 반복 2 | `reflexion/programming_runs/run_reflexion_gpt4.sh` `--max_iters "2"` | 일치 |
| K9 | 단가: GPT-4-turbo 10/30, GPT-3.5-0125 0.5/1.5, Llama-3-70B 0.9/0.9 USD per 1M | `humaneval_eval.ipynb`의 `model_prices`. Llama-3-8B 0.2/0.2, gpt-4-0613 30/60, gpt-3.5-0613 0.5/1.5도 여기 있음 | 일치. 논문에 없는 8B 단가가 코드에 있음 |
| K10 | HotPotQA 훈련 50 + 개발 50 + 시험 200, 홉 2, 홉당 문서 2, 최대 8개 시연 | `hotpotqa_eval.ipynb`와 `HotpotQA/.../joint_optim/*.ipynb`: `HotPotQA(train_seed=10, train_size=50, eval_seed=2000, dev_size=250, test_size=0)`, `devset = dev[:50]`, `testset = dev[50:]`, `max_bootstrapped_demos=8` | 일치(논문의 "훈련 100 중 50 + 50"은 훈련 50과 개발 50으로 구현) |
| K11 | 결합 최적화는 정확도와 프롬프트 시연 토큰 수를 Optuna로 최적화, 후보 중 개발 정확도 최고를 보고 | `joint_optim` 결과 JSON의 후보 4개 각각 `[trainset_acc, demos_prompt_tokens]`, 논문 표 A3의 결합 행은 후보 2(개발 최고)와 일치(`03` §6) | 일치 |
| K12 | NovelQA RAG: text-embedding-3-large, 청크 1,000자, 10개 청크 | `NovelQA-langchain-gpt4.ipynb`: `chunk_size=1000`, `k=10`, `OpenAIEmbeddings(model="text-embedding-3-large")` | 일치 |
| K13 | 표 A1·A2·A3·A5의 값 | `plot_data_*.csv`, `HotpotQA/.../*eval_results.json`에서 재계산 | 일치(`03` §1, §6, §7) |

## 차이와 문제 (D)

| 번호 | 내용 | 영향 |
|---|---|---|
| D1 | CSV와 `output_data`의 모델 이름 `A+B`는 **디버거 + 시드**의 순서입니다(`gpt-3.5-turbo+gpt-4-turbo` = GPT-3.5로 디버깅, 시드는 GPT-4 zero-shot). 그림 1 캡션은 "생성 모델, 디버그 모델" 순서라 표기가 뒤집혀 있습니다. 파일 이름 `probs._ldb_10_gpt-3.5-turbo-0125_pass_at_1_seed_probs._simple_10_gpt-4-turbo-2024-04-09_...`와 토큰 열로 확인 | 저장소를 따로 읽는 사람이 LDB 변형의 모델을 거꾸로 읽기 쉬움 |
| D2 | LDB의 `total_cost`에는 **시드 프로그램 생성 비용이 합쳐져** 있습니다(토큰 곱을 뺀 잔차가 같은 실행 번호의 시드 실행 비용과 정확히 같음, 최대 오차 0.0000). 논문 본문 2절은 LDB의 비용 구성을 설명하지 않습니다 | "LDB (GPT-4, GPT-3.5) 2.19 USD"의 88%(1.93)가 시드. 디버깅만의 비용은 0.26 USD(`03` §2) |
| D3 | `LDB (Reflexion, GPT-3.5)`의 시드는 `humaneval-py._reflexion_2_gpt-4-turbo-2024-04-09…`, 즉 **GPT-4로 만든 Reflexion 프로그램**입니다. `run_ldb_reflexion_seed.sh`는 시드 경로를 `…/reflexion/output_data/reflexion/$dataset/$seedmodel/…`로 받고 두 번째 인수로 시드 모델을 줌. 논문 표 A6과 부록 E.1이 이미 지적한 LDB 논문의 문제를 저장소가 그대로 재현합니다 | 표의 괄호 이름만 보면 GPT-3.5 구성으로 읽힘. 비용 4.19 USD의 3.90이 GPT-4 Reflexion |
| D4 | Reflexion 출력 jsonl은 161행이 MultiPL-E 형식(`HumanEval_23_strlen`)이고 3행이 `HumanEval/32`, `/50`, `/38`로 **끝에 따로 붙어** 있습니다. 예시 테스트가 없는 3과제(논문 6절)가 이것입니다. 과제 키를 정규화하지 않으면 다른 에이전트와 짝지을 수 없음 | `extract_pertask.py`에서 이름을 정규화했고, 3과제만의 영향은 작음(`03` §3.2) |
| D5 | 그림 1의 파레토 점선은 `humaneval_eval.ipynb`에서 **좌표를 손으로 적은 값**입니다(`pareto_x = [0.054854, 0.273124, 2.448440, 6.355392]`, `pareto_y = [0.739024, 0.85, 0.931707, 0.932927]`). 값은 GPT-3.5, Escalation, Warming, LDB (GPT-4)의 평균과 같고, 같은 평균에 대해 `pareto_frontier_example.ipynb`의 알고리즘(상부 볼록 껍질 후 비지배 필터)을 돌리면 같은 네 점이 나옵니다(우리 이식 확인) | 그림이 자료에서 자동 계산되지 않음. 결과는 일치하나 자료가 바뀌면 점선이 따라가지 않음 |
| D6 | HotPotQA의 **최적화 지표**(`validate_context_and_answer_and_hops`: 정답 일치 + 정답이 문맥에 포함 + 홉 조건)와 **보고 지표**(`gold_passages_retrieved`: 정답 문서를 모두 검색)가 다릅니다. 개발 정확도(0.16–0.30)는 앞의 값이고 시험 정확도(0.44–0.62)는 뒤의 값입니다. 본문 3.1절은 평가 지표만 설명합니다 | 후보 선택(개발 최고)과 보고 지표가 정렬되어 있지 않음(`03` §6) |
| D7 | `LanguageAgentTreeSearch/programming/run_lats_gpt4.sh`는 모델 `gpt-4-1106-preview`와 이름 `gpt4_mcts_3_full`로 적힌 예시 스크립트이고, 논문의 gpt-4-turbo-2024-04-09 결과를 만든 명령은 저장소에 따로 없습니다(모델 인수만 바꾸면 됨) | 설정 재현은 가능하나 스크립트가 결과와 직접 대응하지 않음 |
| D8 | `run_ldb_reflexion_seed.sh`는 클러스터의 절대 경로 `/scratch/gpfs/bs6865/agent-eval/…`를 씁니다. `README.md`의 링크 일부는 익명 리뷰용 `anonymous.4open.science` 주소 | 그대로는 다른 기기에서 실행되지 않음, README 링크 일부는 만료 가능 |
| D9 | NovelQA 코드는 모델을 `gpt-4-turbo` 별칭으로 부르고(`novelQA-gpt-4.py:61`, 노트북 `ChatOpenAI(model="gpt-4-turbo")`) 날짜 판을 고정하지 않습니다. 논문은 "GPT-4"라고만 적음. 롱컨텍스트 비용은 `cost_calc.ipynb`가 tiktoken으로 센 토큰 수 × 단가로 계산한 값이고 청구액이 아님 | 별칭은 시간에 따라 다른 모델을 가리킬 수 있음. 비용은 계산값 |
| D10 | 저장소 README는 "원 저자 코드에 가한 변경은 모두 커밋 이력에 있다"고 하나, 이번 clone은 얕아(`--depth 1`) 이력을 읽지 않았습니다. 52개 커밋이 있다는 것만 API로 확인 | 변경 내용은 확인하지 못함 |
| D11 | 저장소 README: LATS의 일부 실행은 중간에 끊겨 이어서 돌려 jsonl `acc`가 부정확하고 `.log`가 정확하다고 안내합니다. GPT-4와 GPT-3.5의 0409/0125 5회 실행은 jsonl `is_solved` 합계가 CSV와 맞았습니다(`03` §1) | 우리 분석은 영향 없음. 다른 LATS 폴더(0613, `gpt-4-turbo-0125` 등)는 읽지 않음 |

## 읽지 않은 것

- `webagents-step/`(STeP)의 정책 코드, `webarena/`의 평가기와 Reddit 자동 로그인 코드, 표 A6의 "open todo comment" 확인. 후속으로 정적 읽기가 가능합니다.
- LATS의 `mcts.py` 내부 테스트 수와 확장 계수(논문 A.1의 "GPT-3.5는 6개, GPT-4는 4개 내부 테스트").
- WebArena 궤적(Dropbox).
- 저장소의 커밋 이력(D10).

## 재실행 제안(실행하지 않음)

- 공식 코드 실행은 모두 OpenAI·Together API를 호출합니다(논문 부록 F: GPU 불필요). 한 번의 HumanEval 실행은 Warming 약 2.4 USD, LDB (GPT-4) 약 6.4 USD, LATS (GPT-4) 약 135 USD입니다(저자 보고, 우리 재계산). 승인이 필요한 일이라 제안만 합니다.
