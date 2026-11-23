# 02. 계보와 관련 연구

분석일 2026-11-23. 이 문서의 이후 문헌은 초록 수준으로 읽은 것이 많아 어디까지 읽었는지 표시합니다. 웹 검색 2회, arXiv API 조회, 원문 PDF 1편(LDB)을 썼습니다.

## 1. 계보: 이 논문이 서 있는 자리

| 갈래 | 대표 연구 | Self-Debug와의 관계 |
|---|---|---|
| 실행 결과로 후보를 고르는 재순위 | MBR-Exec (Shi et al. 2022), Coder-Reviewer (Zhang et al. 2022), CodeT (Chen et al. 2023, arXiv:2207.10397), LEVER (Ni et al. 2023) | 수십 개 샘플이 필요한 방식. Self-Debug가 직접 비교하는 기준선이고 초기 후보 선택(실행 다수결)에는 이 계열을 그대로 씀 (Sec 2) |
| 학습된 코드 수리 | Gupta et al. 2020, Wang et al. 2018, Self-Edit (arXiv:2305.04087, ACL 2023), Chen et al. 2023a (자연어 피드백 학습, arXiv:2303.16749, TMLR) | 수리 모델을 따로 학습. Self-Debug는 학습 없이 few-shot (Sec 6) |
| 프롬프트 기반 자기 피드백 | Reflexion (arXiv:2303.11366), Self-Refine (arXiv:2303.17651), RCI (Kim et al. 2023), CRITIC (arXiv:2305.11738, ICLR 2024) | 같은 시기(2023-03 ~ 05)의 동시대 연구. Self-Debug는 코드와 **실행기**가 있는 과제에 특화 |
| 설명을 중간 산출물로 쓰는 프롬프트 | 사고 연쇄(Wei et al. 2022), PAL, Program of Thoughts | 줄 단위 코드 설명이 사고 연쇄와 정신이 같다고 논문이 말함 (Sec 6) |

구성 요소별 출처: 실행 다수결 선택은 MBR-Exec 계열 (Sec 2). rubber duck debugging은 *The Pragmatic Programmer* (Hunt & Thomas 2000)에서 가져온 비유입니다. 새로운 점은 (1) 설명, 실행 결과, 모델의 정오 판단을 한 루프로 묶고 (2) 이것을 few-shot으로 가르치며 (3) 실패한 예측을 버리지 않고 재사용하는 데 있습니다.

## 2. 이 저장소의 관련 분석과의 관계

### Self-Repair (`../self_repair/`, 1번)

Olausson et al.(arXiv:2306.09896, ICLR 2024)은 이 논문을 "가장 가까운 선행"으로 지목합니다. 두 논문은 같은 구조(생성 → 실행 → 피드백 → 수리)를 다르게 평가합니다.

| | Self-Debug | Self-Repair (Olausson) |
|---|---|---|
| 비교 기준선 | 샘플 수를 늘린 재순위(표본 효율) | 같은 샘플 수(예산)의 i.i.d. 추출 |
| 예산 단위 | 샘플 수 (피드백 호출 비용은 별도로 안 셈) | 토큰 수 (피드백·수리 호출까지 포함) |
| 피드백 출처를 분리했는가 | 아니오. 같은 모델이 설명·피드백·수리를 모두 함 | 예. 피드백 모델과 수리 모델을 분리 (GPT-4 피드백 대 자기 피드백) |
| 결론 | 이득이 크다 (최대 +12%p) | 같은 예산에서 이득은 작고 피드백(진단) 품질이 병목 |

수치로 대조하면 두 결과가 부딪히지 않습니다. Self-Repair는 GPT-3.5의 자기 피드백 정규화 값이 0.91–1.04(1은 같은 예산의 i.i.d. 추출)였고 GPT-4 피드백을 쓰면 1.09–1.14였으며, 사람 피드백은 GPT-4 자기 피드백보다 통과율이 33.3%에서 52.6%로 높았습니다. Self-Debug의 큰 이득은 (가) 기준선이 **greedy 한 번**이고 (나) 실행 결과(단위 테스트)가 피드백에 들어가기 때문입니다. 이 문서의 `03` §3에서 이득의 대부분이 "통과·실패 판정을 받는 재시도"라는 것을 분해했습니다. 이 분해는 Self-Repair의 "피드백 모델의 품질이 병목"이라는 결과와 같은 방향입니다. 다만 Self-Debug는 같은 토큰 예산의 i.i.d. 기준선을 두지 않아 Self-Repair의 질문(예산을 맞추면 이득이 남는가)에는 답하지 않습니다 (`04` C4).

### 오류 위치와 교정 (`../error_location_correction/`, 2번)

"LLM은 추론 오류를 못 찾지만 위치를 알려 주면 고친다"(arXiv:2311.08516)는 탐지와 수정을 분리해야 한다는 결과입니다. Self-Debug의 Spider 실험이 이에 해당합니다. 모델이 맞고 틀린 SQL을 구분하지 못해 Simple 피드백이 소용없고, 설명으로 모델이 불일치를 찾게 하면 겨우 오릅니다 (논문 Sec 5.1). 다만 이 저장소의 2번 분석은 추론(수학) 과제이고 Self-Debug는 코드입니다. 대응은 구조 수준입니다.

### Huang et al., 자기 수정의 한계 (`../cannot_self_correct/`, 6번)

Huang et al.(arXiv:2310.01798)은 외부 신호가 없는 내재적 자기 수정은 정확도를 올리지 못하고 선행 연구의 이득은 정답 라벨로 루프를 멈춘 덕이라 지적합니다. 이 비판이 Self-Debug에 닿는 정도는 설정마다 다릅니다.

| 설정 | 루프 안의 신호 | Huang의 분류에서 | 이득 |
|---|---|---|---|
| TransCoder UT | 채점에 쓰는 **같은 10개 테스트** | 정답 신호(오라클)로 멈춤에 가까움 | +3.6 ~ +13.1%p |
| MBPP UT | 채점 테스트 3개 중 **1개만** 보이고 2개는 숨김 | 부분 신호. 정직한 평가 | +6.0 ~ +9.4%p |
| Spider +Expl | 정답 신호 없음. 실행 결과 표 2행만 | 내재적에 가장 가까움 | Codex +2.8, 나머지 모델 +0.2 ~ +1.1 |
| Table 3 (실행 없음) | 없음 | 내재적 | Codex 최대 +4.8, GPT-3.5 0 ~ +1.6, GPT-4 +0.9 ~ +3.6 |

Huang의 GSM8K GPT-3.5 내재적 설정은 정답을 오답으로 바꾸는 비율이 오답을 고치는 비율과 비슷해 이득이 없었습니다(정답의 11.6%를 깨뜨리고 오답의 31.5%를 고침). Self-Debug의 TransCoder는 실행 게이트(통과하면 디버깅 안 함) 때문에 정답을 깨뜨릴 수 없는 구조입니다. 그래서 이득이 크게 나오는 것은 구조상 당연하고 "자기 수정 능력"의 증거라기보다 "정오 판별 신호가 공짜로 주어졌을 때의 재시도 이득"입니다 (`03` §3, `04` C2).

### 자기 수정 서베이 (`../self_correction_survey/`, 7번)

Kamoi et al.(TACL 2024)은 Self-Debug를 표 1에서 한 줄(외부 도구 피드백, 코드 실행기)로 분류하고 "신뢰할 외부 피드백이 있는 과제에서는 자기 수정이 잘 된다"는 근거로 씁니다. 7번 분석의 V7이 같은 표 2와 표 3을 재계산해 두 가지를 확인했습니다: (1) 실행을 빼면 6개 셀(TransCoder 3, MBPP 3) 모두 비짝 95% 구간이 0을 포함합니다. (2) TransCoder는 평가 테스트가 프롬프트에 노출되어 서베이의 "비현실적(오라클 사용)" 부류에 가깝습니다. 이 문서의 `03` §4는 그 비짝 구간을 그대로 재현했고(MBPP Codex UT +8.0 [2.1, 13.8], TransCoder Codex UT +11.2 [7.2, 15.3] 등) 짝지은 McNemar 가능 범위를 추가했습니다. 서베이가 한 줄로 묶은 것을 이 문서는 세 가지 신호(통과·실패 판정, 실행 결과 내용, 모델 설명)로 가릅니다.

## 3. 이후 문헌

읽은 정도를 괄호에 적었습니다. 설정이 다른 비교(모델, 벤치마크)는 값을 직접 맞대지 않습니다.

| 연구 | 발표 | 내용 | Self-Debug 주장과의 관계 | 읽은 정도 |
|---|---|---|---|---|
| LDB, *Debug like a Human* (arXiv:2402.16906) | arXiv (학회 표기 미확인) | 실행을 기본 블록 단위로 쪼개 중간 변수 값을 넣음 | **독립 재구현.** 공개 코드가 없어 논문 지침대로 Self-Debug를 다시 구현했다고 적음(§3.2). 같은 시험 틀에서 Self-Debug(+Expl) GPT-3.5 TransCoder +3.6, MBPP +6.8, HumanEval +7.3. 그러나 StarCoder HumanEval에서는 −0.6(+Expl)과 0.0(+Trace)이라 "모델의 설명이 오히려 방해한다"고 해석. 시드 프로그램이 강하면(GPT-4 87.2, Reflexion 91.5) Self-Debug 이득은 +1.2, +0.6으로 줄어듦(표 2) | 본문 표 1·2, 3.2–3.3절 읽음 |
| *Revisit Self-Debugging with Self-Generated Tests* (arXiv:2501.12793, ACL 2025 main, aclanthology 2025.acl-long.881) | ACL 2025 | 테스트가 주어지지 않아 모델이 만든 테스트로 디버깅. 사후 실행(post-execution) 대 실행 중(in-execution) | 자기 생성 테스트의 편향 때문에 사후 실행 디버깅이 쉬운 문제에서 어려움. 중간 상태만 쓰는 방식은 편향을 줄임 | 초록만 읽음 |
| LeDex (arXiv:2405.18649, NeurIPS 2024 논문, arXiv 주석 기준) | NeurIPS 2024 | 설명과 수정을 SFT와 RL로 학습 | 프롬프트 방식(Self-Debug)이 작은 오픈 모델에서 잘 안 된다고 지적. 학습으로 pass@1 최대 +15.9 | 초록만 읽음 |
| RLEF (arXiv:2410.02089, Meta) | arXiv | 공개 테스트의 실행 피드백을 여러 턴에 걸쳐 쓰고 숨은 테스트 보상으로 RL | 실행 피드백 자체를 학습 목표로 삼음. Self-Debug의 프롬프트 방식을 학습으로 대체 | 검색 결과 요약만 읽음 |
| MURPHY (arXiv:2511.07833), ReflexiCoder (arXiv:2603.05863, Findings of ACL 2026) | 2025-11, 2026 | 다턴 GRPO로 실행 피드백 자기 수정 학습, 외부 피드백 없이 반성하도록 RL | 프롬프트로는 부족해 학습으로 이동하는 흐름 | 검색 결과 요약만 읽음 |
| 한 번 수정 비교 연구 (preprints.org 202608.0854, 비심사) | 프리프린트 | Self-Refine 대 Self-Debug를 모델 7개에서 비교 | Self-Debug가 비용 대비 이득이 더 좋으나 초기 정확도가 높으면 이득이 작음 | 검색 결과 요약만 읽음. 비심사라 인용 금지 |
| Is Self-Repair a Silver Bullet? (arXiv:2306.09896, ICLR 2024) | ICLR 2024 | 위 §2 | 같은 예산에서 이득이 작음 | |

해석: 이후 연구에서 Self-Debug 자체를 같은 조건으로 다시 돌린 것은 LDB 한 편뿐이고, 거기서 (가) 실행 포함 이득은 재현됩니다(GPT-3.5 TransCoder +3.6은 이 논문의 GPT-3.5 UT +3.6과 같은 크기, MBPP +6.8도 이 논문의 +6.6과 비슷). (나) 약한 모델(StarCoder)과 강한 시드에서는 줄거나 사라집니다. LDB의 TransCoder 기준선 GPT-3.5 82.3은 이 논문의 89.1과 달라, 같은 이름의 기준선이 구현마다 다름을 보여 줍니다. LDB도 TransCoder의 단위 테스트를 모두 모델에 보이게 합니다(3절 각주 3). 즉 평가 테스트가 노출되는 구조는 이 계보에서 관행입니다.

## 4. 저장소·자료 표

| 자료 | 주소 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| Self-Debug 공식 코드 | 없음 | | | | `gh api`로 google-deepmind, google-research, 저자 계정 후보를 확인하고 저장소 검색을 했으나 찾지 못함 (2026-11-23). LDB 논문도 "공개 코드 없음"이라고 적음 |
| LDB 저자 저장소 | github.com/FloridSleeves/LLMDebugger | Apache-2.0 | 586 | 2024-09-10 | 4.3 MB. Self-Debug가 아니라 LDB 구현. 이번에 clone하지 않았고 Self-Debug 재구현이 들어 있는지 확인하지 못함 |
| Spider (dev) | HF `xlangai/spider` | | | | validation 1,034, train 7,000 (datasets-server 크기 조회) |
| MBPP | HF `google-research-datasets/mbpp` | | | | full test 500 (논문의 500과 일치), sanitized test 257 |
| TransCoder | Roziere et al. 2020 | | | | 논문이 560 문항 사용이라 적음. 데이터 직접 확인하지 못함 |

## 5. arXiv 메타데이터

- arXiv:2304.05128, cs.CL (보조 cs.AI). v1 2023-04-11, v2 2023-10-05(최신). 저자 Xinyun Chen, Maxwell Lin, Nathanael Schärli, Denny Zhou. 라이선스 CC BY 4.0 (abs 페이지 확인).
- ICLR 2024 포스터. proceedings 쪽 8746–8825.
