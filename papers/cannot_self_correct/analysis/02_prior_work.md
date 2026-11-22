# 02. 계보와 관련 자료

## 1. 논문이 스스로 밝힌 계보 (Sec 1, 2)

원문 인용을 따랐고, 우리가 새로 확인한 것은 아닙니다. 확인한 항목은 `04`에 있습니다.

| 흐름 | 대표 연구 | 이 논문과의 관계 (저자 서술) |
|---|---|---|
| 자기 피드백으로 반복 수정 | Self-Refine (Madaan et al. 2023), Welleck et al. 2023, Constitutional AI (Bai et al. 2022), Ganguli et al. 2023 | 스타일·안전 같은 목표에서는 효과가 있을 수 있다고 인정 (Sec 7). Self-Refine의 CommonGen-Hard 향상은 초기 프롬프트가 약해서라고 반박 (Sec 5) |
| 자기 수정 + 정답 라벨 | RCI (Kim et al. 2023), Reflexion (Shinn et al. 2023) | 향상이 정답 라벨로 멈춘 덕. 논문의 가장 핵심 반박 (Sec 3) |
| 다중 에이전트 토론 | Du et al. 2023, Liang et al. 2023, Chen et al. 2023a (ReConcile) | 같은 응답 수의 자기 일관성(Wang et al. 2022)과 비교하지 않았다 (Sec 4) |
| 외부 피드백·도구를 쓴 수정 | CRITIC (Gou et al. 2023), Self-Debug (Chen et al. 2023b), Olausson et al. 2023, 검증기 학습 (Cobbe et al. 2021, Lightman et al. 2023) | 외부 신호가 믿을 만하면 수정이 된다. 이 논문이 권하는 방향 (Sec 6) |
| 같은 방향의 선행 결과 | Gou et al. 2023, Zhou et al. 2023a (코드 인터프리터 자기 검증), Wang et al. 2023a (오해를 부르는 피드백에 쉽게 흔들림) | 이 논문과 같은 결론이므로 새로운 결론이 아니라 모호함 정리가 기여 (Sec 7) |
| 조사 | Pan et al. 2023 (피드백 출처에 따른 분류) | 내부 피드백 대 외부 피드백 구분을 빌려 옴 (Sec 2) |

- 이 논문의 대안 중 가장 가까운 것: **Olausson et al. 2023(코드 자기 수리)과 CRITIC**. 둘 다 "외부 신호가 있을 때"에 초점을 둡니다. 이 논문은 "외부 신호가 없을 때"를 부정적으로 측정한 쪽입니다.

## 2. 구성 요소별 출처와 자료 상태

| 구성 요소 | 출처 | 상태 |
|---|---|---|
| GPT-3.5-Turbo (0613, 0301), GPT-4 (2023/08/29 접근), GPT-4-Turbo (gpt-4-1106-preview) | OpenAI API | 지금은 모두 폐기되었을 가능성이 큽니다(확인하지 않음). 재현은 API 비용과 모델 불일치 때문에 근사치만 가능 |
| Llama-2-70b-chat | Touvron et al. 2023 | 공개 가중치. 70B라 이 맥에서 실행 불가 |
| GSM8K 테스트 1,319문항 | Cobbe et al. 2021 | 공개 데이터 |
| CommonSenseQA dev 1,221문항 | Talmor et al. 2019 | 공개 데이터 |
| HotpotQA 100문항 | Yang et al. 2018, 표본은 Shinn et al. 2023과 같음 | 100문항 목록이 Reflexion 저장소에 있음(확인하지 않음) |
| 토론 프롬프트 | Du et al. 2023 (같은 프롬프트 사용) | 원 논문 부록 |
| CommonGen-Hard | Madaan et al. 2023 (Self-Refine 논문이 만든 확장판) | 문항 수는 이 논문에 없음 |
| 이 논문의 코드·응답·결과 | | **공개되지 않음** (본문에 링크 없음, GitHub 검색에서 찾지 못함) |

## 4. 논문 이후 문헌

웹 검색 1회와 arXiv API 초록으로 모았습니다. 본문은 읽지 않았으므로 아래 요약은 **초록에 적힌 주장**입니다.

| 연구 | arXiv | 이 논문과의 관계 |
|---|---|---|
| Kamoi et al., *When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey* (TACL 2024) | 2406.01297 | 확인·정리: (1) 프롬프트된 LLM의 피드백으로 자기 수정에 성공한 선행 연구는 "자기 수정에 특별히 적합한 과제"를 빼면 없다, (2) 신뢰할 외부 피드백이 있는 과제에서는 잘 된다, (3) 대규모 미세조정이 자기 수정을 가능하게 한다. 이 논문의 결론을 일반화하되 조건(미세조정)을 더함 |
| Zhang et al., *Understanding the Dark Side of LLMs' Intrinsic Self-Correction* (ACL 2025) | 2412.14959 | 확인·원인 분석: "라벨 없는 자기 수정은 실패한다"를 출발점으로 삼고, 단순 사실 질문에서 답이 흔들리고(프롬프트 편향), 복잡한 과제에서 인지 편향이 생긴다고 해석. 질문 반복과 소량 SFT로 완화 |
| Liu et al., *Large Language Models have Intrinsic Self-Correction Ability* | 2406.15673 | **조건부 반박:** 온도 0과 공정한 프롬프트 두 조건에서 내재적 자기 수정이 여러 모델에서 나타난다고 주장. 이 논문의 GPT-3.5·GPT-4(온도 1)와 조건이 다름. 이 논문 표 3, 4의 온도 0 모델(GPT-4-Turbo, Llama-2)도 하락 |
| Li et al., *Confidence Matters: Revisiting Intrinsic Self-Correction* | 2402.12563 | **조건부 수정:** 모델이 자기 응답의 "확신"을 이해하고, 이를 무시하면 과도하게 스스로를 비판한다. 확신 평가를 끼운 IoE 프롬프트로 일관된 향상을 주장 |
| Kumar et al., *Training Language Models to Self-Correct via RL* (SCoRe) | 2409.12917 | **조건부 수정:** 자기 수정이 "현대 LLM에서 대체로 비효율적"이라는 전제를 받아들이고, 자기 생성 데이터만으로 멀티턴 온라인 RL을 해 MATH와 HumanEval에서 자기 수정을 15.6%·9.1% 개선했다고 보고. 프롬프트가 아니라 학습으로 풀 수 있다는 방향 |
| Stechly et al., *GPT-4 Doesn't Know It's Wrong* | 2310.12397 | **확인 (다른 과제):** 그래프 색칠에서 GPT-4의 자기 비판이 도움이 안 되고, 외부 정답 검증기가 있어도 비판 내용이 중요하지 않다. 향상은 상위 k 후보에 정답이 있었기 때문 |
| Valmeekam et al., *Can LLMs Really Improve by Self-critiquing Their Own Plans?* | 2310.08118 | **확인 (계획):** GPT-4 자기 비판이 계획 생성 성능을 떨어뜨리고, 검증기가 오탐(false positive)을 많이 낸다 |
| 후속 벤치마크: CorrectBench (Can LLMs Correct Themselves?) | 2510.16062 | 세 과제에서 내재적·외부·미세조정 방식을 평가. 자기 수정이 정확도를 올리지만 추론 LLM에서는 추가 수정의 이득이 제한적이고 시간 비용이 크다고 보고 |
| Self-Correction Bench | 2507.02778 | 같은 오류를 외부(사용자 귀속)와 내부(모델 귀속)로 주입했을 때 14개 비추론 모델에서 "자기 수정 사각지대" 64.5%. 활성화 실패와 지식 부족을 분리 |
| Silver et al. (2025), *Single-Utterance Self-Correction of Perturbed Reasoning* | 2506.15894 | **조건부 반박:** CoT에 인위 오류를 넣었을 때 오픈 가중치 모델들이 한 발화 안에서 자기 수정을 보인다고 보고. 이 논문의 "처음 응답을 고치는" 설정과는 다른 과제 |

- 정리: 이 논문의 핵심(프롬프트만 쓴 내재적 자기 수정, 라벨 없음, 추론 과제)은 이후 연구에서도 **대체로 유지**됩니다. 반박은 **조건이 다른 곳**(온도 0과 공정한 프롬프트, 확신 기반 프롬프트, RL로 학습, 오류가 주입된 추적)에서 나옵니다. 이들이 이 논문을 직접 반박하는지는 본문을 읽어 확인해야 합니다(초록만 읽음).
- 이 목록은 웹 검색 1회와 arXiv 초록으로 만든 것이라 완전하지 않습니다. 인용 전에 Kamoi et al.의 서베이와 Google Scholar의 피인용 목록으로 보충해야 합니다.
- 우리가 확인한 것은 시험 설정의 일관성(`03`)이고, 후속 연구들의 수치는 재계산하지 않았습니다.

## 5. 학회와 판본

| 항목 | 값 | 확인 |
|---|---|---|
| 학회 | ICLR 2024 | `proceedings.iclr.cc/paper_files/paper/2024` 목록에서 제목과 저자(Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Yu, Xinying Song, Denny Zhou)가 함께 나옴 (2026-11-22). arXiv 주석은 "ICLR 2024" |
| 판본 | v1 2023-10-03, v2 2024-03-14 (최신) | arXiv API |
| 분류 | cs.CL (주), cs.AI | arXiv API |
| 라이선스 | CC BY 4.0 | arXiv abs 페이지 |

## 6. 관련 저장소 (논문 자체의 코드는 없음)

`gh api`로 2026-11-22에 확인했습니다. 정적으로만 읽었습니다.

| 저장소 | 별 | 라이선스 | 마지막 push | 대조한 커밋 |
|---|---|---|---|---|
| `noahshinn/reflexion` | 3,439 | MIT | 2025-01-14 | `218cf0ef1df84b05ce379dd4a8e47f17766733a0` |
| `posgnu/rci-agent` | 241 | MIT | 2024-05-01 | `31bf737922fa0337d03367a844de5cf1aff4ac52` |
