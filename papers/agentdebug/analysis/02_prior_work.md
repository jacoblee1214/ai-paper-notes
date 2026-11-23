# 02. 계보, 관련 연구, 자료 표

## 1. 계보

| 줄기 | 대표 | AgentDebug와의 관계 |
|---|---|---|
| 언어 피드백으로 재시도 | Reflexion (Shinn et al., arXiv:2303.11366, 논문 참고문헌), Self-Refine (Madaan et al., 2023, 논문 참고문헌 목록에는 없고 기준선 이름으로만 등장) | AgentDebug의 기준선이자 출발점. 차이는 (가) 피드백을 에이전트 자신이 아니라 별도(더 강한) LLM이 궤적 전체를 보고 쓴다는 점, (나) 피드백을 임계 단계에 꽂고 그 단계부터 재실행한다는 점 |
| 실패 분류 | MAST (다중 에이전트, 14모드), 단일 에이전트 오류 연구(Ji et al. 2024, Ning et al. 2024, VeriLA, 논문 §6) | 분류를 모듈(memory/reflection/planning/action/system)로 쪼갠 점이 다르고, 분류가 곧 프롬프트의 정의 목록이 됨 |
| 실패 귀인 | Who&When(어느 에이전트가 언제, 논문 참고문헌), AgenTracer | 임계 단계 찾기 지표(S)는 Who&When의 step-level 정확도와 같은 종류. 다만 AgentErrorBench는 단일 에이전트 궤적 |
| 오류 위치 제공 후 수정 | Tyen et al. 2024 (이 저장소 `error_location_correction`) | "찾기가 병목, 위치를 주면 고친다"의 에이전트 버전. 본 논문은 찾기와 고치기를 한 파이프라인으로 묶고 고치기 효과만 하위 과제 성공률로 잼 |
| 테스트 시간 스케일링 | Best-of-N, ToT | 같은 예산 기준선으로 쓰임(그러나 예산 단위는 §3 참고) |

## 2. 이 저장소의 관련 분석과의 관계

### `../mast/` (8번)
MAST는 다중 에이전트 7개 시스템 1,642 trace에서 14모드 3범주 분류를 만들고 사람 주석 κ로 정의를 고정했습니다(최종 14모드 사람 κ 0.73, 4 trace, 구간 0.56–0.89; 중간판 κ 0.88). AgentDebug는 단일 에이전트 3개 환경의 200 궤적과 5모듈 17유형입니다. 두 분류의 항목별 대응은 이 문서에서 검증하지 않았습니다. 신뢰도 보고 방식은 다릅니다. MAST는 라운드별 쌍별 κ를 원표로 공개했고 AgentDebug는 κ 0.55 한 값만 적고 이중 주석 궤적 수와 구간이 없습니다.

### `../who_and_when/` (9번)
Who&When은 단계 수준 귀인이 어렵고(손수 만든 시스템에서 GPT-4o가 3.5–8.8%), "항상 마지막 단계" 같은 단순 규칙이 12.1%로 모든 방법과 유의하게 다르지 않음을 보였습니다. AgentErrorBench의 AgentDebug step 정확도 35–58%는 그보다 높지만 환경이 다르고 단일 에이전트입니다. 같은 검증 틀로 우리는 상수 추측 기준선(표본 내 최적 단계)을 계산해 ALFWorld 15%, WebShop 18%, GAIA 26%를 얻었습니다(`03` §4). 즉 Direct Prompting의 GAIA 26%는 상수 추측과 같고, ALFWorld·WebShop에서 Direct Prompting은 상수 추측보다 높습니다.

### `../agentracer/` (3번)
AgenTracer는 반사실 재실행과 결함 주입으로 정답 귀인 라벨을 만들었습니다. AgentDebug의 §3.2 문장("반사실 대입으로 임계 단계를 찾는다")은 AgenTracer와 비슷한 방법을 가리키지만 실제 탐지기는 단일 LLM 호출입니다(`04` D1). 사람 주석은 재실행으로 검증된 것이 아니라 읽고 판단한 라벨입니다. AgenTracer 분석의 "정답 G를 줘도 항상 좋아지지는 않는다"는 관찰은 AgentDebug의 피드백이 정답(사후 궤적)에서 만들어진다는 점과 함께 읽어야 합니다.

### `../error_location_correction/` (2번)
오류 위치가 주어지면 수정이 좋아진다는 결과(표 6, 오라클 위치 대 무작위 위치)의 에이전트판입니다. AgentDebug는 위치를 LLM이 찾아 주는데, 우리가 확인할 수 있는 것은 위치 정확도(표 1)이지 "틀린 위치를 줬을 때"의 효과가 아닙니다. 그 대조(오라클 위치, 무작위 위치, 예측 위치)가 이 논문에 없습니다. Tyen의 설계가 바로 그 빠진 대조군입니다.

### `../self_debug/` (15번)
Self-Debug 분석은 이득을 "통과·실패 판정을 받는 재시도"와 "추가 설명"으로 분해했고, 총이득의 28–81%가 단순 재시도에서 왔습니다. AgentDebug의 그림 5에 같은 분해를 적용하면(`03` §5) Qwen3 두 모델에서는 이득의 75–77%가 단순 Best-of-N으로 설명되고 GPT-4o-mini에서만 29%입니다. 두 논문 모두 재시도 횟수와 진단 정보를 같은 예산에서 분리하지 않았고, 두 논문 모두 환경(또는 테스트)의 성공 판정을 루프 안에서 씁니다.

## 3. 발표 이후 연구 (2026-11-23까지, 웹 검색 3회 + arXiv API)

- **AgentDebugX** (arXiv:2607.18754, 2026-07, EMNLP 2026 Demo 표기, 같은 저자군의 후속 도구): Detect, Attribute, Recover, Rerun 폐루프. DeepDebug가 Who&When에서 strict 귀인 정확도 55.8% → 63.6%라고 보고(초록). 저장소는 MIT, 별 76, 2026-10-20 push. 독립 평가는 찾지 못했습니다.
- **AgentRx** (Microsoft, arXiv:2602.02475, EMNLP Findings 2026): 170 궤적, 11개 과제 설정, 임계 단계와 9범주 분류를 주석한 벤치마크와 진단 프레임워크.
- **TrajDebug** (arXiv:2608.06346): "임계 오류 탐지"를 같은 문제 이름으로 다루며 TRAJERRBENCH(검색 결과 요약 기준 486 주석 궤적, τ²-Bench·SWE-Bench Pro)를 냄(초록 일부만 확인).
- **CUADebug** (arXiv:2608.02643): 컴퓨터 사용 에이전트용. 5범주 30하위유형, 204 OSWorld 실패 궤적, ReAct식 진단 에이전트(초록).
- **CatchBench** (arXiv:2608.22808): 실패를 잡을 수 있는 정보 상태(PRE/LIVE/POST)별 평가. AgentDebug를 논의(검색 결과 기준).
- 검색 결과에 한 비평이 "근본 원인 표적 여부만 바꾸는 단일 요인 절제가 없다"고 적었다는 요약이 있었으나 그 원문은 확인하지 못했습니다. 우리 결론(`03` §6)은 독립적으로 도출한 것입니다.

세부 내용은 모두 초록 수준이며, 각 논문의 본문 표는 읽지 않았습니다.

## 4. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 모듈형 롤아웃(memory/reflection/plan/action) | 저장소 `agentdebug/rollout/rollout.py`에 `OpenManusAgent = UnifiedAgent` 별칭이 있고(저자 소속에 OpenManus), ALFWorld·WebShop 환경 코드를 그대로 가져와 포함 | 논문은 출처를 적지 않음 |
| 환경 | ALFWorld(Shridhar 2020), WebShop(Yao 2022), GAIA(Mialon 2023) | 논문 §4.1 |
| 임계 단계 개념 | Who&When의 decisive error step, AgenTracer | 논문 참고문헌에 Who&When 있음 |
| 반사실 재실행 | Brute Force·Binary Search 기준선으로만 사용 | 본문 §3.2는 사용한다고 서술 |

## 5. GitHub·자료 표

| 자료 | 주소 | 라이선스 | 별 | 마지막 push | 크기 | 비고 |
|---|---|---|---|---|---|---|
| 공식 코드 | github.com/ulab-uiuc/AgentDebug | MIT | 109 | 2026-03-30 | 10.3 MB(API), clone 작업트리 약 32 MB | 커밋 `7740fe3a`. 얕게 clone해 정적으로만 읽음. 논문은 "code and data will be available"이라 했고 README가 같은 주소를 가리킴 |
| 공식 데이터 | Google Drive 폴더(README 링크) | 미확인 | | | | 이 환경에서 받지 않음. 폴더 라이선스 표기 없음 |
| 비공식 HF 미러 | huggingface.co/datasets/davide221/agenterrorbench | 카드에 없음 | | 2025-10-07 | 18.3 MB(parquet 3개), 200행 | 제3자 업로드. 공식과 같은지는 보증되지 않지만 모듈·단계 분포가 논문 그림 11–13과 거의 같음(`03` §3). `data/`에 두며 커밋하지 않음 |
| 후속 도구 | github.com/AgentDebugX/AgentDebugX | MIT | 76 | 2026-10-20 | 9 MB | 이번 분석 대상 아님 |
| 이름이 비슷한 다른 저장소 | `AgentDebug`로 검색되는 무관한 저장소 다수(agiresearch/AgentDebugger 등) | | | | | 이 논문과 관계 확인하지 않음 |
