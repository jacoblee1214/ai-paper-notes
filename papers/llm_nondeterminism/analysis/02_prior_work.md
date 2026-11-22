# 02. 계보와 관련 자료

## 1. 논문이 밝힌 계보 (§2)

| 흐름 | 대표 연구 | 논문과의 관계 (저자 서술) |
|---|---|---|
| 평가 재현성 | Biderman et al. 2024 (lm-evaluation-harness 교훈) | 평가 도구와 모범 사례를 주지만 불안정(instability)은 다루지 않음 |
| 샘플링 변동 | Song et al. 2024 (*The Good, The Bad, and The Greedy*, arXiv:2407.10457) | 온도 1에서 변동을 분석. 이 논문은 변동을 최소화한 T=0을 본다 |
| 단일 모델 불안정 | Ouyang et al. 2025 (ChatGPT 코드 생성의 비결정성, TOSEM) | 모델 하나, 과제 하나, 온도를 바꿔 가며 봄 |
| 입력 변형 견고성 | Sehwag et al. 2019, Hancox-Li 2020, Rauber et al. 2017 | 입력이 조금 바뀔 때의 견고성이고, 이 논문은 입력이 같을 때 |
| 디코딩 자유도 | Holtzman et al. 2020 | 같은 입력에서 다른 출력이 나올 수 있다고만 언급 |

- 가장 가까운 선행은 Song et al. 2024입니다. "비결정성을 평가에서 무시하지 말라"는 같은 방향이고, 이 논문은 호스팅 API의 T=0을 정량화했다는 점이 다릅니다. (논문이 인용한 두 선행의 본문은 우리가 읽지 않았습니다. arXiv 번호만 확인했습니다.)

## 2. 구성 요소별 출처와 자료 상태

| 구성 요소 | 출처 | 상태 |
|---|---|---|
| BBH 4과제 | Suzgun et al. 2023, HF `lukaemon/bbh` | 공개, MIT(저장소 `tasks/ruin_names.py` 주석) |
| MMLU 4과제 | Hendrycks et al. 2021 | 공개. MMLU 오류 문항은 Gema et al. 2024가 지적했으나 이 논문은 그대로 사용 |
| GPT-3.5, GPT-4o | Azure OpenAI 배포(`AppliedAI-gpt-35-turbo`, `AppliedAI-gpt-4o`) | 비공개 API. 논문에 버전 날짜 없음. 지금 같은 결과를 얻는다는 보장 없음 |
| Llama-3-70B, Llama-3-8B, Mixtral-8x7B | Together AI 서버리스 | 코드에 `Meta-Llama-3-70B-Instruct-Turbo`, `Meta-Llama-3.1-8B-Instruct-Turbo` 등이 적혀 있으나 논문은 "Llama-3-8B"라고만 씀(`04` D5) |
| 실행별 원출력 | 공식 저장소 `experiments/v3/runs.tgz`(87 MB, 풀면 2.0 GB) | **공개.** 114개 배치(논문이 쓴 80개 배치의 응답이 165,700개) |
| 평가 코드 | 공식 저장소 `evaluate.py`, `helper_functions.py`, `tasks/*.py` | 공개. 우리는 정적으로 읽고 답 추출 규칙을 직접 다시 구현해 대조(`04`) |

## 3. 공식 저장소와 자료 표

| 자료 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|
| github.com/breckbaldwin/llm-stability (분석 기준, 크기 185 MB) | Apache-2.0 | 8 | 2025-11-11 | 후속 작업의 본거지. 이 분석의 커밋은 `c089c02` |
| github.com/Comcast/llm-stability | Apache-2.0 | 9 | 2025-04-23 | 원 저장소. README가 breckbaldwin으로 옮겼다고 안내 |
| arXiv:2408.04667 | CC BY 4.0 | | v5 2025-04-02 | |
| ACL Anthology 2025.eval4nlp-1.12 | Anthology 페이지의 라이선스는 확인하지 않음(arXiv 라이선스만 확인) | | 2025-12 | Eval4NLP 2025, pp. 135–148 |

(별과 push 날짜는 GitHub API, 2026-11-22.)

## 4. 이후 연구 (2025–2026)

이 논문은 "T=0에서도 다르다"를 측정했고 원인은 추측했습니다. 이후 연구가 원인을 실험으로 좁혔습니다.

| 연구 | 내용 | 이 논문과의 관계 |
|---|---|---|
| Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference* (2025-09 블로그) | 주된 원인은 "부동소수점 비결합성 + 동시성"이 아니라 **서버 부하에 따라 배치 크기가 달라지고 커널의 수치가 배치 크기에 의존하는 것(batch invariance 부재)**이라고 주장. RMSNorm, 행렬곱, 어텐션을 배치 불변으로 만든 커널을 공개(`thinking-machines-lab/batch_invariant_ops`, MIT). 같은 프롬프트 1,000회에서 기본 vLLM은 서로 다른 완성 80종, 배치 불변 커널은 1종이라는 2차 출처 보도가 있음(우리는 원문 수치를 확인하지 않음) | 이 논문의 "입력 버퍼 공유" 추측과 같은 방향의 더 구체적인 설명. 이 논문은 같은 방향을 실험하지 않았다 |
| vLLM `VLLM_BATCH_INVARIANT=1` (문서, 베타) | 결정적 커널과 일부 최적화 비활성. NVIDIA 컴퓨트 능력 8.0 이상 GPU 등 조건이 있음 | 자가 호스팅에서는 해결책이 있다. 호스팅 API 사용자는 쓸 수 없다 |
| SGLang `--enable-deterministic-inference` (LMSYS 블로그, 2025-09-22) | 청크 프리필, CUDA 그래프, 라딕스 캐시와 호환되는 결정적 추론. 일반 모드보다 느림(벤치에서 라딕스 캐시 끔) | 위와 같음 |
| Yuan et al., *Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference* (arXiv:2506.09501, NeurIPS 2025 구두) | 배치 크기, GPU 수, GPU 종류가 달라지면 bf16 greedy에서도 응답이 달라지고, 추론 모델에서는 정확도 최대 9%, 응답 길이 9,000 토큰 차이. 부동소수점 비결합이 원인이라 보고 FP32 연산 LayerCast를 제안 | 이 논문의 원인 추측(§8.1 "연속 배칭, 청크 프리필, 프리픽스 캐싱")을 실험으로 뒷받침하는 쪽 |
| Baldwin 저장소의 후속 노트북(`experiments/shared_vs_one_job`, `stochastic_non_determinism`, 2025-07) | 단일 작업 로컬 Llama-8B 1조건에서 TARr 100%, 공유 API 3%. 실행 수가 늘수록 비결정 문항이 늘어난다는 관찰 | 논문 §8.1의 로컬 실험 주장의 근거(`04` D6) |
| *On Randomness in Agentic Evals* (arXiv:2602.07150) | SWE-Bench Verified에서 60,000 궤적, 모델 3개×스캐폴드 2개. T=0에서도 단일 실행 pass@1이 2.2–6.0%p 변하고 표준편차가 1.5%p를 넘음. 궤적은 토큰의 처음 몇 %에서 갈라지고 전략 차이로 증폭됨 | 이 논문의 문항 단위·짧은 답 측정을 **긴 에이전트 궤적**으로 옮긴 후속. 이 논문은 한 문항 안의 갈라짐(raw string)을 보고, 이쪽은 궤적 전체의 결과 분산을 본다. 해당 폴더 `../agentic_eval_randomness/`(10번) 참조 |

- 위 표의 Thinking Machines 블로그, vLLM·SGLang 문서, Yuan et al.은 웹 검색 결과 요약으로 확인했고 전문을 정독하지 않았습니다. 수치는 인용하기 전에 원문을 확인해야 합니다.
- 이 분석의 토이(`analysis/batch_invariance_toy.py`)가 확인한 것은 "같은 행렬곱이 배치 크기에 따라 비트가 달라질 수 있다"는 메커니즘 하나입니다(`03` §8).

## 5. 이 저장소의 다른 분석과의 관계

| 분석 | 관계 |
|---|---|
| `agentic_eval_randomness/` (10번) | 같은 질문(같은 설정 반복 시 결과 분산)을 에이전트 궤적에서 다룬 후속. 이 논문은 단문 객관식에서 문항 단위 갈라짐, 저쪽은 pass@1 분산과 검정력. 둘을 이으면 "문항 단위 갈라짐이 궤적 길이에 따라 증폭"이라는 설명이 된다 |
| `impossiblebench/`, `swebench_solved_issues/` | 벤치마크 점수를 1회 실행으로 해석하는 위험이라는 같은 문제의식. 이 논문의 TARa·Best/Worst는 "한 번의 점수"가 가리는 폭의 크기를 준다 |
| `mast/` | LLM 판정자(o1, 온도 1.0, 단일 호출)의 라벨에 같은 비결정성이 들어간다. 이 논문은 T=0에서도 판정자 재실행이 다를 수 있다는 근거 |
| `cannot_self_correct/`, `self_correction_survey/` | 자기 수정 성능 비교가 단일 실행 정확도에 의존한다. 차이가 비결정성의 범위(이 논문의 최대-최소 1–15%p)보다 작으면 확정할 수 없다 |
