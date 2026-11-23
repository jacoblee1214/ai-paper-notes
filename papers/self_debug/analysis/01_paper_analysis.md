# 01. 논문 분석: Teaching Large Language Models to Self-Debug

Chen, Lin, Schärli, Zhou (Google DeepMind, UC Berkeley). arXiv:2304.05128, 분석 판은 v2(2023-10-05, 최신)이고 v1(2023-04-11)과의 차이는 §7에 있습니다. ICLR 2024 (§8). 쪽·표·그림 번호는 v2 기준입니다.

## 1. 문제

LLM은 복잡한 프로그래밍 과제에서 한 번에 맞는 코드를 내기 어렵습니다 (Sec 1). 기존 해법은 두 갈래입니다. (가) 여러 샘플을 뽑아 실행 결과로 다시 순위를 매기는 방법(MBR-Exec, Coder-Reviewer, LEVER)은 수십 개 샘플이 필요합니다. (나) 학습된 코드 수리 모델은 추가 학습이 필요합니다. 선행 연구는 "외부 피드백(단위 테스트, 사람 지시) 없이는 LLM이 코드를 못 고친다"고 보았습니다 (Chen et al. 2023a, Sec 1).

이 논문의 질문: **학습 없이 few-shot 프롬프트만으로, 모델이 자기 코드를 실행 결과와 스스로 쓴 설명으로 고치게 할 수 있는가.**

## 2. 방법 (Sec 3, Figure 1)

한 번의 디버깅 턴은 세 단계입니다.

1. **Generation**: 문제 설명에서 후보 프로그램을 만듭니다.
2. **Explanation**: 코드를 자연어로 설명하거나 샘플 입력에 대한 실행 추적(trace)을 만듭니다.
3. **Feedback**: 코드가 맞는지에 대한 메시지를 만듭니다. 모델이 판단하거나 단위 테스트로 외부에서 만듭니다.

종료 조건은 피드백이 "맞다"고 하거나 최대 디버깅 턴(10, Sec 5) 도달입니다. 한 턴이 세 단계를 다 갖지는 않습니다. 피드백은 네 종류입니다 (Sec 3).

| 종류 | 내용 | 실행기 사용 |
|---|---|---|
| Simple | "맞다/틀렸다. 고쳐라" 한 문장 | Spider는 모델이 판단(실행 없음), TransCoder·MBPP는 단위 테스트 실행으로 판정 |
| UT | 실패한 단위 테스트의 입력, 기대값, 실제 결과 (Figure 5) | 사용 |
| Expl | 코드를 줄 단위로 설명(rubber duck debugging). Spider는 질문 요약, 실행 표 2행, SQL 설명, 비교로 구성 (Figure 3) | 설명은 모델이 생성. UT와 함께 쓰면 실행 결과 포함 |
| Trace | 모델이 줄 단위로 실행을 흉내 내 쓴 추적 | 추적은 모델이 생성(실제 중간 상태가 아님) |

초기 후보가 여럿이면 오류 없이 실행되는 것 중 실행 결과가 가장 많은 것을 골라(실행 다수결) 그 하나에 디버깅을 적용합니다. 단위 테스트가 문제에 있으면 통과하지 못하는 후보를 먼저 거릅니다 (Sec 2). 식은 없고 절차와 프롬프트로 이루어진 방법입니다. 단위 테스트 통과의 정의 P(i_k)=o_k, ∀k만 Sec 2에 있습니다.

## 3. 적용과 평가 설정 (Sec 4, 5)

| 과제 | 벤치마크 | 문항 수 | 단위 테스트 | 디버깅 중 실행기가 보는 것 |
|---|---|---|---|---|
| text-to-SQL | Spider dev | 1,034 (논문에는 없음. HF `xlangai/spider` validation 분할로 확인) | 없음 | SQL을 실행해 반환 표의 처음 2행을 프롬프트에 넣음 (Sec 4.1) |
| C++→Python | TransCoder | 560 (Sec 4.2) | 문제당 10개, **문제 설명에 모두 포함** | 10개 전부. 하나라도 틀리면 디버깅, 전부 통과하면 종료 |
| text-to-Python | MBPP test | 500 (Sec 4.3) | 문제당 3개. **첫 번째만 프롬프트에 포함하고 나머지 2개는 숨겨 평가** | 보이는 1개 |

모델은 code-davinci-002(Codex), gpt-3.5-turbo, gpt-4, StarCoder 15.5B입니다. 초기 생성은 한 개일 때 greedy(τ=0), 여럿일 때 τ=0.7입니다. 설명, 피드백, 새 프로그램은 모두 greedy입니다 (Sec 5). 시드, 반복 횟수, 토큰 비용, API 모델 스냅샷은 적혀 있지 않습니다. Spider의 채점 지표(실행 정확도 여부)도 본문에 서술이 없습니다.

비교 기준선은 학습된 모델(T5-3B + N-best reranking, LEVER)과 프롬프트 기반 재순위(Coder-Reviewer, MBR-Exec)입니다. TransCoder와 MBPP는 코드 LLM이 이미 최고 성능이라 부록 B로 미룹니다 (Sec 5).

## 4. 실험 결과 (표 번호는 v2)

| 표·그림 | 내용 | 핵심 값 |
|---|---|---|
| Table 1(a) | Spider dev | T5-3B+N-best 80.6, LEVER 81.9, MBR-Exec 75.2, Self-Debug Codex 81.3 → +Expl 84.1 |
| Table 1(b) | MBPP | MBR-Exec 63.0 (n=25), Reviewer 66.9 (n=25), LEVER 68.9 (n=100), Codex n=10 72.2 → Simple 73.6, UT 75.2, UT+Expl 75.6 |
| Table 2 | 피드백 종류별 x 모델 (실행 포함) | TransCoder Codex 80.4 → 92.5, GPT-4 77.3 → 90.4. MBPP Codex 61.4 → 70.8 |
| Table 3 | 단위 테스트 **실행 없이** | TransCoder Codex +3.5, GPT-3.5 0.0, GPT-4 +1.1. MBPP Codex Simple는 61.4 → 57.6 |
| Fig 6(a) | Spider 초기 샘플 수 1·8·16·32 | 기준선 77.5/78.7/80.7/81.3, Self-Debug 80.8/82.5/82.9/84.1 |
| Fig 6(b) | Spider 난이도별 | extra hard 63.9 → 72.9 (+9.0) |
| Table 4 | 고쳐진 오류 유형 | Spider: WHERE 조건 25.7 등. TransCoder 출력 불일치 61.9 / 런타임 오류 38.1. MBPP 69.2 / 30.8 |
| Fig 8 | TransCoder 턴 수와 샘플 수 | 첫 턴에서 대부분 개선. 25샘플 기준선 96.3 |
| Table 5 | 초기 코드를 다른 모델이 디버깅 | Codex가 GPT-3.5의 코드를 UT+Expl로 92.7 |
| Table 6 (부록 B) | TransCoder 선행 모델 | PaLM-Coder 55.1 대비 80.4 |

저자의 서술 (Sec 5.1, 5.2): (1) Spider에서는 simple 피드백만으로 나아지지 않고(모델이 맞고 틀린 SQL을 구분 못 함) 코드 설명이 일관되게 2–3%p 올립니다. (2) 단위 테스트 실행 정보가 피드백에 들어갈 때 이득이 가장 큽니다. (3) 실행 없이도 Codex는 최대 5%p, GPT-4는 MBPP에서 3.6%p 오릅니다(5.2.2). (4) 첫 디버깅 턴이 개선 대부분을 만듭니다. (5) 실패한 예측을 재사용하므로 표본 효율이 10배 이상 좋습니다(abstract, 5.2.1).

## 5. 주장과 근거 위치

| # | 주장 | 위치 | 비고 |
|---|---|---|---|
| S1 | few-shot 프롬프트만으로 학습 없이 자기 코드를 고친다 | Abstract, Sec 3, 표 2 | 실행 포함 설정에서 성립. 실행 없는 설정은 S5 |
| S2 | Spider에서 코드 설명이 기준선을 2–3%p, extra hard에서 9%p 올린다 | Abstract, Sec 5.1, 5.2.3, Fig 6 | `03` §5와 §2에서 검산 |
| S3 | TransCoder·MBPP에서 최대 12%p | Abstract, Table 2 | 표에서는 GPT-4 TransCoder가 +13.1이라 문구보다 큼 (`03` §4) |
| S4 | 표본 효율: 10배 이상 많은 샘플을 쓴 기준선과 같거나 낫다 | Abstract, Sec 5.2.1, Fig 6(a) | Spider에서만 직접 성립 (`04` C4) |
| S5 | 외부 진단 없이 실행 결과와 자기 설명만으로 잘못을 찾는다(rubber duck) | Abstract(v2), Sec 3, 5.2.2 | 실행을 빼면 이득이 대부분 구별 안 됨 (`03` §3, `04` C2) |
| S6 | 더 풍부한 피드백이 일관되게 낫다 | Sec 5.1 | 표 2에서 설명의 몫은 작고 trace는 일관되지 않음 (`04` C5) |
| S7 | 최신 성능(state-of-the-art) | Abstract, Table 1 | 비교 기준선이 서로 다른 논문의 수치 (`04` C6) |

## 6. 한계 (논문이 쓴 것, 쓰지 않은 것)

논문은 한계 절이 없고 결론(Sec 7)에서 설명 능력 개선, 잠재 버그 설명 추가를 향후 과제로 말합니다. 논문 안에서 이미 보이는 약점은 다음과 같습니다.

- GPT-4가 Spider에서 Codex보다 한참 낮고 few-shot을 덜 따른다고 인정합니다(Sec 5.1). 실행 없이는 GPT-3.5와 GPT-4가 "자기 초기 예측에 과신한다"고 적습니다(Sec 5.2.2).
- MBPP에서 설명의 이득이 작은 이유를 세 가지로 듭니다: 단위 테스트가 이미 정오를 알려 줌, 설명을 넣으면 예시 수가 6개에서 3개로 줄어듦, 초기 코드가 멀면 설명을 쓰지 않고 새로 씀(Appendix C.2).
- 시드, 분산, 토큰·호출 비용, 모델 스냅샷은 보고하지 않습니다. 공식 코드가 없습니다(`04` §1).

## 7. v1과 v2의 차이

| 항목 | v1 (2023-04-11) | v2 (2023-10-05) |
|---|---|---|
| 모델 | Codex만 | Codex, GPT-3.5, GPT-4, StarCoder (Table 2, 3, 5) |
| 초록의 rubber duck 서술 | "without any feedback on the code correctness or error messages, the model is able to identify its mistakes by explaining the generated code" | "without any **human** feedback ..., by **investigating the execution results** and explaining the generated code" |
| 피드백 종류 | Simple, UT, Expl | Trace 추가 |
| 실행 없는 설정 | 없음 | Table 3 추가 |
| 오류 유형 분석 | 없음 | Table 4 |
| 표 1(a) | Codex 81.3 / Simple 81.3 / +Expl 84.1 | 표 2(a)로 이동. Baseline 81.3 / Simple 81.3 / +Expl 84.1 |
| Spider 기준선 설명 | Sec 5.2.1: "Self-Debug 전 Codex 결과가 이미 최고 성능에 맞먹는데 이는 더 나은 프롬프트 설계 덕" | 이 문장 없음 |
| 쪽 수 | 68 | 78 |

수치(Codex의 Spider, TransCoder, MBPP 표)는 v1과 v2가 같습니다. 바뀐 것은 초록의 서술입니다. v2는 사람 피드백이 없다는 점을 "사람"으로 한정하고 실행 결과를 보는 것을 명시했습니다. 이것이 Huang et al.(2310.01798, v1 2023-10-03)이 지적한 "외부 신호가 들어간 자기 수정" 문제와 같은 시기에 나온 서술 조정인지는 알 수 없습니다. 날짜가 비슷하다는 사실만 확인했고 이유는 확인하지 못했습니다.

## 8. 학회 표기

proceedings.iclr.cc/paper_files/paper/2024 목록에서 제목과 저자를 확인했고(해시 `2460396f2d0d421885997dd1612ac56b`, 쪽 8746–8825), iclr.cc/virtual/2024/poster/18880의 제목이 "ICLR Poster Teaching Large Language Models to Self-Debug"입니다 (2026-11-23). 따라서 ICLR 2024 포스터입니다.
