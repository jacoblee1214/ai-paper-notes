# 04. 논문과 공식 코드 대조 (FastChat `fastchat/llm_judge/`)

- 저장소: `github.com/lm-sys/FastChat`, 커밋 `587d5cfa1609a43d192cedb8441cac3c17db105d`(2025-06-02), Apache-2.0. `--sparse`로 `fastchat/llm_judge/`만 받았습니다(`repo/` 전체 1.4 MB(.git 포함), `fastchat/llm_judge/` 592 KB).
- **정적 읽기만 했고 실행하지 않았습니다**(OpenAI·Anthropic API 필요, 약관과 비용). 읽은 파일: `common.py`(711줄), `gen_judgment.py`, `show_result.py`, `compute_agreement.py`, `gen_model_answer.py`(온도 부분), `clean_judgment.py`, `data/judge_prompts.jsonl`, `data/mt_bench/question.jsonl`, `data/mt_bench/reference_answer/gpt-4.jsonl`, `download_mt_bench_pregenerated.py`.
- 공개 데이터는 코드로 확인한 것이 아니라 `03`에서 직접 계산한 값과 맞춰 봤습니다.

## 1. 논문과 같은 것 (K)

| # | 논문 | 코드 | 판정 |
|---|---|---|---|
| K1 | 쌍 비교 기본 프롬프트(그림 5): "Avoid any position biases… Do not allow the length of the responses to influence… Do not favor certain names…", 출력 `[[A]]`/`[[B]]`/`[[C]]` | `judge_prompts.jsonl`의 `pair-v2`와 `run_judge_pair`의 `[[A]]`, `[[B]]`, `[[C]]` 파싱 | 일치 |
| K2 | 단일 채점 프롬프트(그림 6), `[[rating]]` | `single-v1`, `one_score_pattern` | 일치 |
| K3 | 순서를 바꿔 두 번 판정, 두 번 모두 이긴 쪽만 승(§3.4 "conservative") | `play_a_match_pair`가 `(answer_1, answer_2)`와 `(answer_2, answer_1)` 두 번 호출하고 `g1_winner`, `g2_winner`를 저장. `show_result.py`가 `g1 != g2`이면 tie로 셈 | 일치 |
| K4 | 다중 턴은 전체 대화를 한 프롬프트에 넣고 둘째 질문에 집중(§3.5, 그림 9) | `pair-v2-multi-turn`, `single-v1-multi-turn` 프롬프트와 `multi_turn=True` | 일치 |
| K5 | 참조 답 안내(§3.4): 판정자가 먼저 푼 답을 참조로 제시 | `ref_based=True`, `ref_answer_1/2`; 참조 답 파일이 `reference_answer/gpt-4.jsonl`(판정자 이름과 같은 파일) | 일치 |
| K6 | 일치도 정의(부록 D.3): G4-사람은 GPT-4 한 표와 사람 표 각각, 사람-사람은 서로 다른 두 사람 쌍, S1은 비일관을 동점으로 셈 | `compute_agreement.py`의 `get_mt_bench_agreement`, `convertvote`(`"tie" in vote`면 tie), `equalvote` | 일치. 이 규칙으로 표 5의 표 수와 비율을 모두 재현함(`03` §1.1) |
| K7 | 표 8 점수 = 160개(80문항 x 2턴) 평균 | `show_result.py`의 "Average" 블록 | 일치(공개 판정 파일에서 GPT-4 8.991 등 4개 모델 값 재현, `03` §5.1) |

## 2. 논문과 다르거나 논문에 없는 것 (D)

| # | 논문 | 코드 또는 데이터 | 영향 |
|---|---|---|---|
| D1 | 표 5는 판정자 6종(G4-Pair, G4-Single, Human 등)을 보고 표 13은 Claude, Author, Human-M까지 | 공개 `compute_agreement.py`는 `gpt4-pair`와 `human`/`author` 이름만 처리하고 `Human-M`, `G4-Single`, Claude는 계산하지 못함. HF 판정 파일도 GPT-4 쌍 비교(`gpt4_pair`)와 사람(`human`)만 있음 | 표 5의 G4-Pair/Human 열과 Author는 재현했지만 G4-Single·Claude 열은 재현할 수 없음. `Human-M`은 우리 구현으로 S1 n을 정확히 맞췄고 S2는 4개 차이(`03` §1.1) |
| D2 | 본문은 사람이 GPT-4 판정을 본 뒤 바꿀 수 있다고만 함(§4.2, 부록 C.1) | 수집 코드와 "노출 후 최종 표"인지 "노출 전 표"인지는 공개 코드·데이터 카드에 없음 | 일치도가 부풀었을 가능성을 배제할 수 없음. 가정 시나리오 76.7%(`03` §1.3) |
| D3 | §3.4는 "수학" 질문에 참조 답을 쓰는 예를 듦. 표 4는 수학 10문항 | `NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]`. 참조 답 파일 30개(수학·추론·코딩 각 10), 공개 단일 채점 파일에서 `single-math-v1`과 `single-math-v1-multi-turn`이 각각 1,020개(30문항 x 34모델) | **MT-bench 80문항의 3/8(30문항)이 판정자 GPT-4 스스로 만든 참조 답에 의존.** 본문에 어느 범주에 참조를 썼는지, 참조의 정답 검증 여부가 없음. 같은 모델이 참조를 만들고 채점하므로 GPT-4의 오답이 참조일 때 GPT-4 채점이 일관되게 틀릴 수 있음(자기 일관성) |
| D4 | 표 8은 160개 평균 | `show_result.py`는 `score != -1`(점수 추출 실패)을 평균에서 빼지만, 공개 판정 파일에는 -1이 4개 모델에 1개씩 남아 있음. 우리 평균은 -1을 포함해 표 8과 일치(표 8의 네 모델에는 -1이 없음) | 해당 4개 모델에서 점수 0.04–0.06 차이(예: claude-instant-v1 7.850 대 7.906). SE(0.2–0.3)보다 훨씬 작음 |
| D5 | 판정 온도·반복·모델 버전이 본문에 없음(부록 B가 "gpt-4-0314로 만든 예시"라고만 함) | 판정 호출은 `temperature=0, max_tokens=2048`(OpenAI), 1024(Anthropic). 호출은 항목당 한 번이고 반복이 없음. 공개 판정 파일의 `tstamp`는 2023-06-20–23(대부분)과 2023-07-19로, 판정 모델 스냅샷 이름은 `judge` 필드에 `["gpt-4", "single-v1"]`로만 남음 | 같은 입력의 GPT-4 호출이 T=0에서도 달라질 수 있음은 이 저장소 11번이 다른 호스팅 모델에서 보임. 판정자 쪽 분산이 보고에 없고 스냅샷이 특정되지 않아 재생성한 점수가 논문 점수와 맞는다는 보장이 없음 |
| D6 | 표 2·11: "GPT-4가 비슷한 답에서 65% 일관" | 공개 쌍 판정 파일에는 표 2·9의 비슷한 답 쌍(GPT-3.5를 T=0.7로 두 번 호출) 데이터가 없음. 표 11의 세 쌍은 공개 파일로 근사 비교 가능: 논문 대비 1턴 일관이 3.8–5.0%p 낮음(Claude 63.7 대 67.5, Vicuna 67.5 대 73.8, LLaMA 93.8 대 98.8, `03` §3) | 표 2의 위치 편향 실험(80개 쌍 x 판정자 3 x 프롬프트 4)은 원자료가 없어 산술 검산만 가능 |
| D7 | "Chatbot Arena 3K표를 무작위 추출"(§4.1) | `lmsys/chatbot_arena_conversations`는 게이트(자동 승인·약관 동의), 33K 대화이고 논문이 쓴 3K 부분집합의 표시도 없음. GPT-4 판정 결과 파일은 이 데이터에 없음 | 표 6, 그림 4는 데이터로 재계산 불가. 우리는 받지 못했고 논문 숫자만 검토 |
| D8 | 일치도에 표본 오차 없음 | 코드는 비율만 출력(`ratio`) | 구간은 우리가 붙임(`03` §1) |
| D9 | 논문은 temperature를 언급하지 않음 | **모델 답 생성 온도가 범주마다 다름**: `temperature_config`에서 writing, roleplay 0.7, stem, humanities 0.1, 나머지 0.0(`gen_model_answer.py`가 이 설정을 씀, 그 외 범주는 0.7) | 문항 점수가 범주별로 다른 샘플링 온도에서 나옴. 모델 간 비교는 같은 규칙이지만, 재생성 점수의 분산 원천이 됨(문항 안 샘플링 잡음은 본문에서 논의되지 않음) |

## 3. 구현 노트 (코드 읽기에서)

- 판정 파싱은 정규식 `\[\[(\d+\.?\d*)\]\]`(실패하면 `[x]` 백업, 그래도 실패하면 -1). 쌍 비교는 `"[[A]]" in judgment` 먼저 검사하고 그다음 `[[B]]`, `[[C]]`이므로 판정문에 `[[A]]`와 `[[B]]`가 모두 나오면 항상 A가 됩니다(우선순위가 암묵적).
- 단일 채점을 쌍 비교로 바꿀 때 동점 임계값은 `TIE_DELTA = 0.1`입니다. 점수가 정수와 .5 단위라서 사실상 "같은 점수면 동점"입니다. 논문은 이 변환 규칙을 적지 않았습니다(표 5의 G4-Single 열).
- Anthropic 판정자에게는 시스템 프롬프트를 사용자 메시지 앞에 `[Instruction]`으로 붙입니다(`run_judge_pair`). OpenAI 판정자는 시스템 메시지로 둡니다. 판정자 간 비교는 프롬프트 배치도 달라서 순수한 모델 효과가 아닙니다.
- `compute_agreement.py`의 S2 정의는 "tie 또는 tie (inconsistent)를 양쪽에서 뺀다"이며 논문 표 5의 설명과 같습니다.

## 4. 재현 계획 (제안만, 승인 필요)

| 항목 | 비용·요건 | 이 맥에서 |
|---|---|---|
| 공개 판정 파일 재계산(`03`) | 완료 | 가능 |
| MT-bench 판정 재생성(GPT-4 단일 채점, 80x2x모델) | OpenAI API 호출, 승인과 비용 필요(모델 1개당 약 160번 호출, 입출력 토큰 수천) | 호출은 가능하나 GPT-4 스냅샷이 달라 논문 점수와 정확히 맞지 않음 |
| 같은 입력을 5–10번 반복 호출해 판정자 쪽 분산 측정 | API 호출, 승인 필요 | 가능 |
| 위치 편향 표 2 재현(비슷한 답 쌍) | API 호출 + 답 생성 | 오픈 모델 답 생성은 GPU, 판정은 API |
| Arena 표 6 재계산 | HF 약관 동의 후 약 33K 대화 다운로드 | 계정과 약관 동의가 필요해 승인 필요 |
