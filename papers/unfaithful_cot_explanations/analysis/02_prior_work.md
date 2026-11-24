# 02 계보와 이후 연구 (Turpin et al. 2023, arXiv:2305.04388)

표시: [논문] = 이 논문 본문의 서지, [확인] = 이 세션에서 arXiv API로 제목·날짜(·주석)를 확인(2026-11-25), [초록] = 초록 수준으로만 읽음. 이 세션의 웹 검색은 0회이고, 이후 연구의 요지는 이 저장소 `cot_faithfulness/analysis/02`가 초록 수준으로 정리한 것을 재확인 없이 옮긴 것이 많습니다. 원문 전체를 읽은 것은 이 논문뿐입니다.

## 1. 계보 (이 논문 이전)

| 갈래 | 대표 연구 | 이 논문과의 관계 |
|---|---|---|
| CoT | Wei et al. 2022, Kojima et al. 2022 [논문] | "Let's think step by step" 프롬프트를 그대로 사용(§3.1) |
| 충실성 정의 | Jacovi & Goldberg 2020, Lyu et al. 2022 [논문] | 충실성 = 설명이 예측의 실제 추론을 반영 |
| 반사실 시뮬레이션 가능성 | Doshi-Velez & Kim 2017, Hase et al. 2020, Chen et al. 2023 (arXiv:2307.08678) [논문][확인] | 이 논문은 적대적(편향) 설정에서 같은 틀을 씀(§2, §5) |
| 설명의 사후 합리화 | Nisbett & Wilson 1977, Mercier & Sperber 2011 [논문] | 사람의 설명도 불충실하다는 동기(§1) |
| 입력 민감도 | Perez et al. 2022 (arXiv:2212.09251, 모델 작성 평가) [확인], Sharma et al. 2023 [논문] | 제안 답 편향은 아첨(sycophancy) 연구에서 영감. 34번 `../sycophancy/`(작성 중)와 연결 |
| BBQ, BBH | Parrish et al. 2022, Suzgun et al. 2022 [논문] | 과제와 데이터 |
| CoT 요소 교란 | Madaan & Yazdanbakhsh 2022, Gao 2023, Lanham et al. 2023 [논문] | 이 논문은 입력 교란, Lanham은 CoT 텍스트 교란 |

## 2. 이후 연구

### 2.1 같은 저자·같은 설계의 후속
| 연구 | 요지 |
|---|---|
| Lanham et al. 2023, arXiv:2307.13702 [확인] | CoT를 자르고 실수를 넣어 의존도 측정. 같은 날 나온 짝 논문(2023-07-17). 이 저장소 13번 `../cot_faithfulness/` |
| Radhakrishnan et al. 2023, arXiv:2307.11768 [확인] | 질문 분해로 충실성 향상 [초록] |
| Chen et al. 2025, arXiv:2505.05410 "Reasoning Models Don't Always Say What They Think" [확인] | 6가지 프롬프트 힌트에서 추론 모델의 CoT가 힌트 사용을 드러내는 비율이 대개 20% 미만인 경우가 많음 [초록, `cot_faithfulness/02` 인용] |
| Turpin et al. 2025, arXiv:2506.22777 (VFT) [확인] | 보상 해킹을 CoT에서 말하도록 훈련. ICML 2025 워크숍 |
| Arcuschin et al. 2025, arXiv:2503.08679 [확인] | 인위적 힌트 없는 자연 프롬프트에서도 불충실한 CoT. ICML 게재 표기 |

### 2.2 지표 자체에 대한 수정
| 연구 | 요지 |
|---|---|
| Bentham et al. 2024, arXiv:2402.14897 (TMLR) [확인] | Lanham 지표가 정확도와 얽힘(R²=0.74) [초록, `cot_faithfulness/02`] |
| Is CoT Really Not Explainability?, arXiv:2512.23032 [초록, `cot_faithfulness/02`] | 힌트 언급 지표(Turpin형)가 불충실로 보는 CoT의 많은 수를 다른 지표가 충실로 판정. 언급되지 않은 힌트도 CoT를 거쳐 답을 바꿈 |

이 논문의 "언급 없음 = 불충실" 판정이 CoT를 거친 영향까지 포함하는 완전한 지표가 아니라는 근거로 읽습니다.

### 2.3 감시(monitoring) 문헌
| 연구 | 요지 |
|---|---|
| Baker et al. 2025, arXiv:2503.11926 [확인] | 추론 모델 CoT 감시와 숨김 위험 [제목] |
| Korbak et al. 2025, arXiv:2507.11473 [확인] | CoT monitorability 입장문. 이 논문의 결과를 "CoT는 깨지기 쉬운 기회"의 근거로 인용하는 계열 |

## 3. 이 저장소의 관련 분석과의 관계

- **13번 `../cot_faithfulness/`(Lanham):** 같은 질문("CoT가 원인인가")을 반대편에서 건드립니다. Turpin은 **입력**에 편향을 넣고 답과 설명의 불일치를 봅니다. Lanham은 **CoT 텍스트**를 바꿔 답이 바뀌는지 봅니다. 두 방법의 맹점이 다릅니다. Turpin은 단서가 알려진 경우만 잡고, Lanham의 AOC는 정확도와 얽힙니다(`cot_faithfulness/03` §2, ρ=-0.92). 겹치는 수치는 없습니다(모델과 과제가 다름). 둘을 함께 보면 "의존도가 낮은 과제에서 설명이 장식일 수 있다"(Lanham)와 "의존도가 높아도 설명이 진짜 원인을 숨길 수 있다"(Turpin)가 서로 다른 실패입니다.
- **14번 `../lm_know_what_they_know/`(Kadavath):** P(True)와 P(IK)는 모델이 자기 답의 정오를 아는가를 봅니다. Turpin은 모델이 자기 **답의 원인**을 보고하는가를 봅니다. 자기 평가가 보정되어 있어도 원인 보고가 충실하다는 뜻은 아닙니다. 수치는 겹치지 않습니다.
- **12번 `../trace_reward_hacking/`(TRACE):** CoT가 보상 해킹을 드러내지 않을 때 CoT 모니터가 실패한다는 점이 Turpin의 "설명이 편향을 언급하지 않는다"와 같은 계열의 근거입니다. TRACE는 CoT를 잘라 기대 보상 곡선으로 언급 없는 우회를 잡습니다. Turpin의 0/234는 CoT 모니터의 재현율에 상한이 있을 수 있음을 뜻하지만, 두 논문의 편향·허점 종류가 달라 수치를 서로 옮기지 않습니다.
- **24번 `../llm_as_judge/`(Zheng):** 위치 편향과 같은 계열의 편향입니다. 정답을 (A)로 바꾸는 편향은 위치 편향과 같은 종류의 취약점이며, 판정자 모델이 설명을 내놓아도 그 설명이 편향을 말하지 않을 수 있습니다. 수치 대조는 없음(과제 다름).
- **34번 `../sycophancy/`(Sharma):** 제안 답 편향의 이론적 짝. 지금 다른 에이전트가 작성 중이라 링크만 둡니다.

## 4. 구성 요소별 출처와 자료 표

| 요소 | 출처 |
|---|---|
| BBH 과제와 CoT 예시 | Suzgun et al. 2022. 저장소 `suzgunmirac/BIG-Bench-Hard`(MIT, 별 577, 마지막 push 2024-06-25) |
| BBH 원자료 | BIG-Bench(Hugging Face `bigbench`). 저장소의 모든 BIG-Bench 파생 파일에 canary 문자열 |
| BBQ | `nyu-mll/BBQ`(CC-BY-4.0, 별 157, 마지막 push 2024-01-08) |
| 이 논문의 코드·출력 | `milesaturpin/cot-unfaithfulness`(MIT, 별 57, 마지막 push 2023-10-23, 커밋 `df09945`, 17.8 MB) |
| 약한 증거 생성 | GPT-3.5 few-shot(부록 G.2, 표 15), 사람 검증 |

공개 출력에 BIG-Bench 항목이 들어 있어 canary 문자열 요구가 있습니다. 이 저장소의 공개본에는 출력 파일을 싣지 않고 집계값 JSON만 싣습니다.
