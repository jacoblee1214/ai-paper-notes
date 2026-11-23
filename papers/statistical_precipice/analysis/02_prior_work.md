# 02. 계보와 관련 연구

## 1. 이 논문이 서 있는 자리

### 1.1 이전: RL 평가 관행 비판 (논문 부록 A.3과 §1 인용 기준)

| 선행 | 한 일 | 이 논문과의 차이 (논문 A.3 서술) |
|---|---|---|
| Henderson et al. 2018 (Deep RL that matters), Colas et al. 2018·2019 | 단일 과제에서 시드 수·유의성 검정·부트스트랩 구간 지침 | 단일 과제의 평균 구간. 과제당 3–5회면 부트스트랩이 부정확해 Atari 100k에서 20–30회가 필요(그림 A.18). 이 논문은 과제를 모아 M·N개로 층화해 5–10회로 낮춤(그림 A.19) |
| Machado et al. 2018 (ALE 재검토) | sticky action 등 ALE 평가 프로토콜 정리 | 이 논문은 ALE 200M 평가에 sticky action 55게임 프로토콜을 따름(그림 9) |
| Chan et al. 2020 | 안정성·위험 지표 | 평가 자체의 신뢰성이 아니라 알고리즘 신뢰성. 성능 프로필은 단일 과제에도 쓸 수 있다고 연결 |
| Jordan et al. 2020 | 하이퍼파라미터 튜닝 없는 "완전한" 알고리즘을 게임이론적으로 평가, 과제당 1,000–10,000회 권고 | 튜닝이 끝난 뒤의 성능을 소수 실행으로 보는 목적이 다름 |
| Recht 2019 | Welch t 검정 기반 성능 프로필 | 편향 추정, 이상값에 약함, 효과 크기 무시, 정규성 가정, 비교 집합에 순위가 의존(A.3의 다섯 가지 지적) |
| Dolan & Moré 2002 | 최적화 소프트웨어 벤치마크의 성능 프로필(경험적 누적분포) | 이 논문은 구간 밴드를 붙인 "꼬리 분포" 형태로 RL에 이식 |
| Bellemare et al. 2013 (ALE) | 과제 평균 점수의 분포(중앙값의 일반화) | 편향된 추정량(그림 7 오른쪽)이라며 실행 점수 분포로 대체 |
| Bouthillier et al. 2021 | ML 벤치마크의 분산 원천, P(A>B) 기준 | 개선 확률과 0.75 기준을 가져옴(A.7) |

실제 이 논문의 새로움은 **새 통계량**이 아니라 (1) 대규모 사례로 점추정이 얼마나 위험한지 보인 것, (2) 소수 실행에서 작동하도록 층화 부트스트랩과 IQM, 실행 점수 분포를 한 묶음으로 처방하고 라이브러리(`rliable`)로 낸 것입니다. IQM은 통계학의 25% 절사 평균이고 부트스트랩, 프로필, Mann–Whitney U는 모두 기존 도구입니다.

### 1.2 이후: 채택과 수정

- **채택(확인한 것):** 후속 논문이 rliable의 지표(중앙값, IQM, 평균, 최적성 격차와 95% 층화 구간)를 쓴다. 예: Robust Offline Imitation Learning from Diverse Auxiliary Data (arXiv:2410.03626, 부록 A.3), Deep RL Agents are not even close to Human Intelligence (arXiv:2505.21731, IQM과 95% CI를 주 지표로). 두 논문 원문에서 해당 절을 확인했습니다(`04` C6).
- **수정(확인한 것):** Patterson et al. *Empirical Design in Reinforcement Learning* (JMLR 2024, arXiv:2304.01315, §2.6)은 소수 실행의 부트스트랩 신뢰도와 IQM의 쓰임새를 단서와 함께 다시 다룹니다(`04` C2, C3, C9).
- 후속 문헌의 전체 조사는 하지 않았습니다. 검색 4회 안에서 위 세 편만 원문으로 확인했습니다.

### 1.3 분류 축

- 11축(평가). 유형은 방법론·평가 통계(새 모델 없음).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 우리 구현 |
|---|---|---|
| 25% 절사 평균 | 통계학의 trimmed mean | `iqm`, scipy `trim_mean`과 같은 절사 규칙 |
| 부트스트랩, 백분위 구간 | Efron 1979/1994, 논문 [29] | `stratified_resample`, `bootstrap_ci` |
| 성능 프로필 | Dolan & Moré 2002 | `performance_profile` |
| Mann–Whitney U | Mann & Whitney 1947 | `prob_improvement`, scipy와 대조 |
| 층화 재추출 | 과제별 독립 복원추출 | 같음 |

## 3. 이 저장소의 관련 분석

이 논문은 "실행(시드)이 적을 때 어떻게 보고하는가"를 다룹니다. 이 저장소의 에이전트 평가 분석들이 같은 문제를 LLM 쪽에서 각자 부분적으로 만졌습니다.

- `agentic_eval_randomness/` (On Randomness in Agentic Evals): 같은 설정을 10회 돌려 단일 실행 pass@1이 2.2–6.0%p 흔들림을 보이고 검정력으로 필요한 실행 수를 계산. 이 논문의 "실행 수 대 신뢰구간 폭" 문제의 LLM 에이전트판이다. 차이: 그 논문은 한 벤치마크(500문항)의 반복을 다루고 문항 군집 오차를 빼며, 이 논문은 과제(M)를 모은 집계와 층화 재추출을 다룬다.
- `llm_nondeterminism/` (Non-Determinism of "Deterministic" LLM Settings): temperature 0에서도 실행마다 달라져 "실행"이 시드 고정으로 통제되지 않는다. 이 논문의 각주 3("시드 고정은 모든 무작위성을 통제하지 못한다, GPU 비결정성")과 §6의 "시드 고정으로 해결되지 않는다"를 LLM 쪽에서 뒷받침한다.
- `swe_agent/` (SWE-agent): 표 3의 어블레이션 한 칸은 단일 실행이라 3.0%p 차이를 같은 설정 반복의 불일치로 가늠해야 했다. 이 논문의 도구를 쓰면 그 일이 층화 부트스트랩 구간과 개선 확률로 정리된다(`03` §6). 그 분석에서는 쌍체 비교(문항 짝)를 썼는데, 이 논문의 개선 확률은 쌍체를 쓰지 않는 Mann–Whitney 형태라는 차이가 있다.
- `little_power/` (With Little Power Comes Great Responsibility, 이후 분석 예정): 검정력 분석 쪽. 이 논문은 유의성 검정을 피하고 구간을 권고한다(§2 Remark)는 점에서 다른 입장.
- `eval_error_bars/` (Adding Error Bars to Evals, 이후 분석 예정): LLM 평가의 오차막대(문항 군집 표준오차 등). 이 논문의 층화 부트스트랩과 직접 비교 대상.

## 4. 자료

| 자료 | 주소 | 라이선스 | 크기·상태 (2026-11-23 확인) |
|---|---|---|---|
| 논문 | arXiv:2108.13264 (v4), proceedings.neurips.cc 2021 | CC BY 4.0 | PDF 22.8 MB |
| 코드 `rliable` | github.com/google-research/rliable | Apache-2.0 | 마지막 push 2024-08-12, 별 876, 열린 이슈 5, GitHub 표시 크기 1.9 MB(clone 5.0 MB), 커밋 `3ccd9f4` |
| 의존 라이브러리 `arch` | pip `arch >= 5.3.1, < 8.0` (setup.py) | GitHub 판정 NOASSERTION(미확인) | 신뢰구간 계산(`IIDBootstrap`)을 위임 |
| 개별 실행 점수 | `gs://rl-benchmark-data` (논문 부록 A.1) | - | **받을 수 없음.** 이슈 #31(2025-04)에서 403, 2026-11-23에 버킷을 조회하니 논문 데이터가 없고 무관한 웹사이트 파일 11개만 있음(`04` C8) |
| Colab | bit.ly/statistical_precipice_colab | - | 확인하지 않음 |
