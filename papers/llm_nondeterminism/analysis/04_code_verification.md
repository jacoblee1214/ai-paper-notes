# 04. 논문 ↔ 공식 저장소 대조 (D1–D10)

대상: `github.com/breckbaldwin/llm-stability`, 커밋 `c089c02`(2025-11-11), Apache-2.0. 저장소는 정적으로 읽었고 **공식 코드는 실행하지 않았습니다**(`evaluate.py`가 `datasets`를 불러 네트워크에 닿고 모델 래퍼는 API 키가 필요함). 대신 답 추출과 지표 규칙을 `analysis/recompute_runs.py`에 직접 다시 구현해 공개 결과와 대조했습니다. 공개 데이터는 `experiments/v3/runs.tgz`(87 MB, 풀면 2.0 GB), `experiments/v2/runs.tgz`(28 MB)입니다.

| ID | 논문 | 코드·데이터 | 판정 |
|---|---|---|---|
| D1 | TARr@N, TARa@N, BestAcc, WorstAcc 정의(§5.2) | `evaluate.py`의 `TACr`, `TACa`, `best_possible_count`, `worst_possible_count`가 정의대로 계산. 우리 재구현이 `stability_eval.csv` 114개 배치에서 TACr, TACa, 실행별 정답 수, Best, Worst와 **전부 일치**(114/114) | 일치 |
| D2 | TARa는 "추출한 답이 같다"(§5.2) | 답을 하나로 정하지 못한 응답(복수 답 `LookupError`, 또는 답 없음)은 `answer.add(idx)`로 고유 값이 되어 불일치로 셈. 답 없음은 문항당 첫 실패만 넣어 10회 모두 실패면 일치로 셈. 논문 80개 배치에서 원출력이 10회 모두 같은데 TARa가 불일치인 문항 62개. 표 3 GPT-3.5 ruin names에서 TARr 97.2% > TARa 95.6% | **논문에 없는 규칙.** TARa는 파서 품질에 의존(`03` §5) |
| D3 | 빈 응답 언급 없음 | `pd.isna(row['response'])`이면 정답도 답 집합 원소도 아니므로 건너뜀. 논문 배치에서 빈 응답 167개(GPT-3.5 126, GPT-4o 41). 표 3 GPT-4o 논리 연역 zero-shot에서 Best−Worst > 100−TARa | **논문에 없음**(`03` §5) |
| D4 | 답 추출: "많은 부분이 하드코딩"(§8) | `helper_functions.parse_parenthesized_answers`: 마지막 줄 → 마지막 3줄 → 전체의 순서로 "the answer is (X)", `(X)` 정확 일치, 대소문자 무시, 단어 경계 순으로 찾고, 한 단계에서 둘 이상이 잡히면 복수 답으로 실패. 전체 응답 165,700개(논문 배치) 중 복수 답 5,029개(3.0%), 답 없음 598개. 모델별 복수 답: Mixtral 2,072, Llama-8B 1,139, Llama-70B 1,043, GPT-4o 454, GPT-3.5 321. 한 조건에서 최대 16.2% | 일치(코드가 서술한 대로). 복수 답 응답은 오답으로 셈 |
| D5 | 모델 "Llama-3-8B-Instruct", "Mixtral-8x7B-Instruct", "GPT-3.5 Turbo", "GPT-4o"(§5.1). 버전·날짜 없음 | `models/llama3-8b.py`의 현재 모델명은 **`Meta-Llama-3.1-8B-Instruct-Turbo`**(Llama-3는 주석 처리), 70B는 `Meta-Llama-3-70B-Instruct-Turbo`, Mixtral은 `mistralai/Mixtral-8x7B-Instruct-v0.1`(Together AI). GPT는 Azure OpenAI 배포(`api_version` 2023-07-01-preview, 2024-04-01-preview). 코드는 `max_tokens`를 지정하지 않음 | **불일치 또는 불명.** 논문의 "Llama-3-8B"가 실제로 3인지 3.1인지 코드로 확정되지 않음(8B 실행은 2025-02, 코드의 현재 값은 3.1). `max_tokens` 무제한이라 출력 길이는 모델에 맡김 |
| D6 | 로컬 GPU에서 최적화 없이 돌린 Llama3-8B가 결정적(§8.1). 표·설정 없음 | 논문 본문에 결과 표가 없다. 저장소 `experiments/shared_vs_one_job/`: Penn State 서버의 단일 작업 Llama-8B 1과제(대학 수학 zero-shot, 100문항 × 10회)에서 TARr 100%, TARa 96%, 실행별 정답 31로 동일. 비교 대상은 공유 API(Together)의 같은 조건으로 TARr 3%, 정답 수 30–38. 단일 작업 쪽 온도는 **1e-14**(공유 쪽은 0.0). 노트북 결론은 "premature to declare" | **약하게 지지.** 조건 1개이고 두 쪽이 호스팅 환경 외에 모델 판본·양자화·온도 값도 다를 수 있다. 같은 문항 100개 × 10회지만 문항 100개가 사실상 표본 크기 |
| D7 | GPT-3.5 파인튜닝(2-fold)이 불안정을 줄임(§8.1). 표 없음 | `experiments/fine_tuned/stability_eval.csv`: 논리 연역 few-shot(N=7)에서 TARr 99.6%, TARa 99.6%, 정확도 BestAcc 44.4%/WorstAcc 44.0%. navigate few-shot(N=6)에서 TARr 100%, TARa 100%, 정확도 65.2%. 비교용 기본 모델(2024-12, top-p 0.0)은 TARr 79.6%와 86.8%, 정확도 약 90%와 96%. 실행 수가 10이 아니라 7과 6 | **지지하지만 해석이 어려움.** 파인튜닝 후 정확도가 크게 낮아져(44.4% 대 90%) 모델이 거의 한 답으로 수렴했을 수 있다. 안정성 향상이 불안정 감소인지 성능 저하의 부산물인지 구분 안 됨 |
| D8 | "T=0, top-p 1, 시드 고정"(§5) | 코드는 `temperature=0.0, seed=12, top_p=config['top_p_k']`를 API로 보냄. 공개 데이터의 설정 열: 2024-12 GPT 배치는 `top_p_k=0.0`, 2025-02 이후 배치는 `1.0`. 표 2·3은 모두 `top_p_k=1.0` 배치와 일치(80/80) | 표는 서술과 일치. 단 **2024-12 배치는 top-p 0**이고 논문에 쓰이지 않음(`03` §3 C). 시드는 보냈으나 제공자가 지켰는지는 검증되지 않음 |
| D9 | 8과제 × 5모델 × 2설정 = 80조건, 조건당 10회(§6). 문항 수 표 1 | 80개 조건 모두 10회 배치가 있음. 공개 배치는 114개(두 번 돌린 34개 포함). 문항 수가 표 1과 같음(250, 250, 250, 250, 165, 100, 282, 110). Llama-8B 전문 회계 few-shot에는 5회짜리 배치 하나가 더 있고 표는 10회짜리와 일치. 압축을 풀면 비어 있는 폴더가 4개(Llama-70B 도형 few, Llama-8B 논리 연역 few·navigate 0·few의 일부 날짜, 실패한 실행으로 보임) | 일치. 일부 폴더가 비어 있음(공개 결과 CSV에는 영향 없음) |
| D10 | 공개 주소 | v5 초록에 `github.com/breckbaldwin/llm-stability`. 게재본 초록은 `github.com/Anonymous`로 남아 있음. 저장소 README가 v1은 코드가 없고, v2·v3 코드가 재정비된 것이라고 안내(레포의 `release 1.0`은 v2용) | 일치. 게재본과 v5의 표 수치는 같음 |

## 재현 가능성 판정

- **공개 원출력에서 표 2·3의 모든 수치를 다시 얻습니다.** 이 부분은 API 호출 없이 재현되었고(80/80), 그림 1·3·4도 같은 값에서 나옵니다.
- **다시 돌려 같은 수치를 얻는 것은 기대할 수 없습니다.** 호스팅 모델은 버전이 바뀌고 TARr 자체가 배치마다 수십 %p 달라집니다(`03` §3 C). 새 API 실행은 비용과 외부 전송이 생겨 승인 후에만 합니다.
- 모델 래퍼는 사내 Azure 엔드포인트 이름을 하드코딩해 그대로는 쓸 수 없습니다.

## 데이터 크기

- 저장소 clone 344 MB(GitHub API 185 MB, `experiments/`에 실행 압축 파일 다수).
- 압축 해제: v3 2.0 GB, v2 192 MB. `data/`(gitignore)에 있고 커밋하지 않습니다.
