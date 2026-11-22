# 04. 공식 코드·데이터 정적 대조

대상: `github.com/multi-agent-systems-failure-taxonomy/MAST` 커밋 `a70542e541b2104ef8fcd785778179e173fb8d70`(2025-07-23), 얕은 clone(`repo/`, gitignore). **실행하지 않았고** 정적으로만 읽었습니다. 공개 데이터는 HF `mcemri/MAD`(현 `MAST-Data`)를 받아 재계산했습니다(`03`).

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `llm_judge_pipeline.ipynb` | o1 호출, 응답 파싱, 모드별 비율 집계. 셀 11개 |
| `taxonomy_definitions_examples/definitions.txt`, `examples.txt` | 모드 정의(132줄)와 예시(1,115줄)를 판정 프롬프트에 붙임 |
| `traces/` | 원 trace(AG2 7,184 json, AppWorld 16, HyperAgent 513, MagenticOne_GAIA 4,535, math_interventions 600, mmlu 91, OpenManus_GAIA 30, programdev 389). 1.9 GB 중 1.1 GB가 MagenticOne_GAIA |
| `inter_annotator_agreement_annotations/*.pdf` | 사람 주석표 PDF 4개(`round1`, `round_2`, `round3`, `new_benchmarks`) |
| 없는 것 | κ 계산 코드, 표 2 평가 코드, 라벨 정제·집계 코드, 그림·표 생성 스크립트, 라이선스 파일 |

## 논문과 코드·데이터의 차이

| # | 항목 | 논문 | 코드·데이터 | 영향 |
|---|---|---|---|---|
| D1 | 재현 범위 | 표 2(판정자 κ 0.77), IAA κ 0.88·0.79 | 판정자 호출과 파싱 코드만 있고 **κ 계산·표 2 평가 코드가 없습니다.** held-out 목록도 없습니다 | 표 2와 IAA는 공개 파일에서 새로 계산해야 합니다(`03`) |
| D2 | 모드 3.2·3.3 이름 | 3.2 "No or Incomplete Verification", 3.3 "Incorrect Verification"(그림 1, HF README) | 노트북 프롬프트는 3.2 "No or Incorrect Verification", 3.3 "Weak Verification". `definitions.txt`는 3.2 "Weak Verification"(검증이 있으나 불충분), 3.3 "No or Incorrect Verification"(검증 누락·불이행). **같은 노트북 안에서 3.2와 3.3의 의미가 뒤바뀌어 있습니다** | 판정자가 3.2·3.3을 어느 의미로 표시했는지 알 수 없습니다. 공개 데이터의 3.2/3.3과 표 6이 어긋나는 것(`03` §4.3)과 방향이 맞습니다. 원인 단정은 못 합니다 |
| D3 | 2.6 이름 | "Reasoning-Action Mismatch" | 프롬프트·정의 "Action-Reasoning Mismatch" | 이름만 다름 |
| D4 | 프롬프트 예시 응답 | 14모드 | 프롬프트 안의 예시 답안에 **존재하지 않는 코드 `1.6`과 `2.7`**이 있고 정의 목록에는 없습니다 | 예시가 이전 분류(17모드 중간판)에서 온 흔적. 판정자 혼동 가능성 |
| D5 | 파싱 | 설명 없음 | `{mode}`를 정규식에 이스케이프 없이 넣고(`1.1`의 점이 임의 문자) 실패하면 조용히 `0`("no")으로 둡니다 | 파싱 실패가 "실패 없음"으로 기록됩니다. 공개 데이터에는 완전한 `null` 4행만 남아 있습니다 |
| D6 | 판정 설정 | o1 | `model='o1'`, `temperature=1.0`, 단일 호출, 시드 없음. 재현을 위한 반복 호출이나 다수결이 없습니다. 입력이 1,048,570자를 넘으면 **뒤를 잘라냅니다** | 판정 일관성이 보고되지 않았습니다. 공개 데이터에서 잘림 대상은 3 trace(ChatDev Qwen)로 영향은 작습니다 |
| D7 | 성공 여부도 LLM이 판정 | 표 1에서 "HE"(사람이 과업 완료 확인) | 프롬프트가 모드와 함께 "과업이 성공했는가"도 o1에게 묻습니다 | 성공 라벨 출처가 판정자인지 사람인지 데이터에서 분리되지 않습니다. 공개 데이터에는 성공 라벨이 없습니다 |
| D8 | trace 메타데이터 | 표 1: OpenManus는 ProgramDev | 저장소 폴더 `OpenManus_GAIA`(30), 사람 주석 파일은 GAIA, HF 데이터는 ProgramDev | 어느 쪽이 맞는지 `03` §1에서 정리 |
| D9 | 사람 주석 공개본 | 본문 "21 traces" | HF 19행(고유 18), 원 주석표와 셀 일치 76–89%(라운드 2·3·새 벤치마크), PDF `round1`은 `new_benchmarks`와 본문이 같은 복사본 | **라운드 1 원자료가 공개되지 않았습니다.** HF 파일은 κ 재계산에 쓰면 안 됩니다 |
| D10 | 공개 라벨 대 논문 | 그림 1, 표 6, 표 8 | 모드 점유율, 표 6의 20/56칸, 표 8이 재현되지 않습니다(`03` §4) | 공개 라벨이 논문 이후에 다시 주석되었거나 다른 판정 설정일 가능성 |
| D11 | 문서 | README | 링크는 arXiv v2를 가리키고 "1K traces"라고 쓰며(HF는 1,642), HF 이름이 `MAD`에서 `MAST-Data`로 바뀜 | 사소 |
| D12 | 라이선스 | 코드·데이터 공개 | 저장소에 LICENSE가 없습니다. HF 데이터는 CC BY 4.0 | 코드 재사용 조건이 불명확합니다 |

## 일치하는 것

- 노트북 프롬프트는 14모드 각각에 yes/no를 요구하고 한 문장 요약을 요구합니다(§3.3의 설명과 같음).
- 공개 데이터의 구성별 trace 수는 표 1과 모두 일치합니다(합 1,642).
- 노트북은 정의와 예시를 프롬프트 끝에 붙이는 few-shot 설계입니다(표 2 "o1 few-shot"과 같은 구성으로 보입니다. 예시 선택 기준은 코드에 없음).

## 실행 제안 (하지 않았음, 승인 필요)

- **판정자 재현 (API 비용)**: 사람 주석이 있는 trace 4–15개에 o1 판정을 여러 번 돌려 반복 일관성과 κ를 측정. o1 호출이라 비용과 외부 전송이 생깁니다. 필요하면 trace 수와 호출 수를 먼저 보여 드립니다.
- 노트북 실행은 하지 않았고 필요한 패키지(openai, pandas, tiktoken)도 설치하지 않았습니다.
