# 04. 코드 대조: 논문 ↔ 공식 저장소

- 저장소: https://github.com/bingreeky/AgenTracer, 커밋 `256b19e53e8c7341822e5925c481fb942f606ed8`(`repo/`, gitignore)
- 방법: 정적 읽기만 했습니다. 실행, 설치, API 호출은 하지 않았습니다.
- 읽은 파일: `README.md`, `data/` 목록, `MetaGPT/Who_When_Data_Pipeline/`의 `diagnose_fail_log_metagpt.py`(행 194–430, 725–752), `attack_success_log_metagpt.py`(제목과 주입 프롬프트 일부), `universal_framework/universal_framework.py`(행 1040–1075), `universal_framework/attack_processor.py`(행 52–140)

## 1. 저장소에 무엇이 있는가

- 파일 목록(최상위 `README.md`, `data/tracertraj-code-test.parquet`, `assets/framework.png`, `MetaGPT/`)을 보면 저장소는 **MetaGPT 한 시스템에 대한 데이터 파이프라인**입니다.
- 학습 코드(GRPO, verl 설정, 보상 함수 식 11–12), AgenTracer-8B 가중치, 나머지 다섯 다중 에이전트 시스템(AutoGen, AgentPrune, AFlow, OWL, smolagents)의 실행 코드는 없습니다.
- 데이터 릴리스(zip)는 별도로 받았습니다(`data/release/`, gitignore). 구조는 `03_analysis.md` §4에 있습니다.

## 2. 대조 결과

| ID | 논문 | 저장소 | 판정 | 근거 위치 |
|---|---|---|---|---|
| D1 | 반사실 재실행: 분석기가 행동 a'_t를 제안하고 재실행해 성공하면 그 단계가 결정적 오류(식 4–7, Alg. 1 1–8행) | 재실행과 오라클 수정을 하는 코드가 없습니다. `diagnose_fail_log_metagpt.py`는 실패 로그를 LLM에 주고 `mistake_step`과 `mistake_agent`를 받습니다. 검증은 그 단계가 존재하는지, 주체가 라벨과 같은지뿐입니다(행 326–358) | **다름(핵심).** 저장소 방법은 재실행이 아닌 LLM 판단입니다. 정답은 프롬프트에 들어갑니다(행 395). 그러니 정답을 본 LLM이 라벨을 만든 셈입니다 | `diagnose_fail_log_metagpt.py` 326–358, 395 |
| D2 | 결함 주입: 성공 궤적의 한 단계를 Π로 망가뜨리고, 실패가 나면 그 단계가 정답(식 8–9) | 주입 프롬프트는 LLM이 공격 단계와 내용을 고르게 합니다(`attack_success_log_metagpt.py`). 결과 기록은 `is_correct: False`로 고정됩니다(`universal_framework.py` 1063–1067 주석: "Attack successful, result is incorrect") | **부분 일치.** 주입 후 실패를 확인하는 코드는 이 경로에서 보이지 않습니다. 확인 위치는 상류의 evalplus 단계인데, 정적 읽기로는 끝까지 추적하지 못했습니다 | `universal_framework.py` 1040–1075, `attack_processor.py` 52–140 |
| D3 | Alg. 1 Part 2: 무작위 K개 단계 중 첫 성공 주입 | 주입 단계를 무작위로 고르는 코드가 없고, LLM이 `attack_step`을 고릅니다 | **다름.** 단계 선택 방식이 다릅니다 | `attack_success_log_metagpt.py` 프롬프트 |
| D4 | 보상 r = I_format·[λr_step + (1−λ)r_agent], 가우시안 σ=1, λ=0.5(식 11–12, §5.1) | 해당 코드 없음 | 검증 불가 | 저장소 파일 목록 |
| D5 | 학습: GRPO, KL 제거, 동적 클리핑, verl, batch 32, rollout 8, lr 1e-6(§4.2, §5.1) | 해당 코드 없음 | 검증 불가 | 저장소 파일 목록 |
| D6 | 여섯 개 시스템, 여섯 개 벤치마크에서 궤적 수집(§5.1) | MetaGPT만 있음. 다른 시스템 로그 생성 코드 없음 | **일부만.** 공개 데이터는 여러 시스템을 포함하지만 릴리스에는 시스템 이름이 적히지 않습니다 | README, `universal_framework` |
| D7 | 자기 개선 파이프라인(§5.3): 피드백을 주입하고 3회 반복 | `diagnose_fail_log_metagpt.py` 737–750은 `monitor.correction_successful = True`를 **테스트 결과 없이** 설정합니다. 주석은 "can be further refined based on test results"라고 합니다 | **다름(주의).** 저장소의 "수정 성공"은 테스트로 확인하지 않습니다. 논문 §5.3의 측정과 같은 것이 아닙니다 | `diagnose_fail_log_metagpt.py` 725–752 |
| D8 | 공개 데이터는 논문과 같은 데이터(§5.1, 부록 표 3) | 릴리스 README: "not exactly identical ... includes additional samples"(v1.0.0) | **다름.** 행 수와 주체 라벨 형식이 논문과 다릅니다(`03_analysis.md` §4) | `data/release/unz/.../README.md` |
| D9 | 평가 설정 w/ G, w/o G(§5.1) | 저장소 평가 코드 없음 | 검증 불가 | — |

## 3. 우리 쪽에서 확인된 것

1. **정답을 프롬프트에 넣는 방식.** 저장소의 진단 프롬프트는 `Ground Truth:`를 넣습니다(`diagnose_fail_log_metagpt.py` 395). 논문의 w/ G 설정과 같은 조건입니다. 라벨 생성 단계에서 정답을 쓰면, 평가 설정(w/o G)과 라벨이 같은 정보원을 공유하지 않을 수 있습니다. 주석이 정답 지식으로 채워졌다면 w/o G 평가가 주석과 같은 편향을 가질 수 있습니다. 이것은 추측이므로 `04`의 판정 목록에 "확인 필요"로만 적습니다.
2. **검증이 얕음.** 저장소는 라벨이 궤적 안에 존재하는지와 주체가 맞는지만 봅니다. 결정적 오류가 실제로 실패의 원인인지(재실행 확인)는 보지 않습니다. 이것이 논문 Alg. 1과 가장 큰 차이입니다(D1).
3. **재실행이 없는 자기 개선.** 수정 성공 표시가 테스트 없이 `True`입니다(D7). 논문 §5.3의 성과 수치(+4.8~14.2%)를 이 코드로 재현할 수 없고, 이 코드가 그 수치를 만들었다는 근거도 없습니다.

## 4. 판정 목록 (C 형식, 04_claim_verification 대신 여기 적음)

- C1 (결정적 오류 라벨이 반사실 재실행에서 나옴): **저장소에서 지지되지 않음**(D1)
- C2 (결함 주입이 정답 라벨을 만듦): **부분 지지**. 실패 확인 코드가 이 경로에서 보이지 않음(D2)
- C3 (자기 개선 +4.8~14.2%): **저장소로 검증 불가**(D7, 테스트 없는 성공 표시)
- C4 (가중치 비공개): **확인됨**(README)

## 5. 재현 가능 여부

- 학습과 추론에는 가중치와 학습 코드가 필요합니다. 둘 다 없으므로 재현은 불가능합니다.
- 주석 파이프라인(MetaGPT 실행)은 API 키와 MetaGPT 실행 환경이 필요합니다. 승인 없이는 하지 않습니다(`CLAUDE.md`의 API 규칙). 제안으로만 남깁니다.
