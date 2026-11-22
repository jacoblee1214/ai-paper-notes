# 02. 계보와 관련 자료

## 1. 논문이 스스로 밝힌 계보 (Sec 2)

논문 Sec 2 기준입니다. 원문 인용을 그대로 따랐고, 우리가 새로 확인한 것은 아닙니다.

| 흐름 | 논문에 나온 대표 연구 | 이 논문과의 차이 (저자 서술) |
|---|---|---|
| 프로그램 합성에 LLM | Chen et al. 2021 (Codex, pass@k), Li et al. 2022 (AlphaCode), Austin et al. 2021 | 저자들은 입출력 예시 전체를 가정합니다. 공개 테스트와 숨은 테스트를 나누지 않습니다. |
| 코드 수리 (사람 코드) | Long & Rinard 2016, Le Goues et al. 2021 등 | 이 논문은 LLM이 만든 코드를 텍스트 피드백으로 고칩니다. |
| 자동 합성 코드의 사후 수리 | Gupta et al. 2020 (DSL), Le et al. 2022 (CodeRL 계열), Yasunaga & Liang 2020, 2021 | 텍스트 피드백을 쓰는 점이 다릅니다. |
| 동시대 LLM 자기 수리 | Zhang et al. 2023 (APPS, 피드백 없는 수리, 비용 무시), Chen et al. 2023b (Self-Debug, 코드 수리 with 텍스트 피드백), Peng et al. 2023 (검색 보조 환각 수정), Madaan et al. 2023 (Self-Refine) | 논문은 피드백 단계를 떼어 그 중요성을 따로 봅니다. Chen et al. 2023b와 가장 가깝고, 모델과 목적이 다릅니다. |
| 조사 | Pan et al. 2023 | 저자들이 이 논문의 보완 자료로 지목합니다. |

## 2. 가까운 논문 (이번에는 분석하지 않음)

| 논문 | arXiv | 관계 |
|---|---|---|
| LLMs cannot find reasoning errors, but can correct them given the error location | 2311.08516 | 오류 위치를 주면 수정은 되지만 찾는 것이 어렵다. 진단과 수정 분리라는 같은 질문. |
| Large Language Models Cannot Self-Correct Reasoning Yet | 2310.01798 | 외부 오라클 없는 자기 수정의 한계. |
| Teaching Large Language Models to Self-Debug | 2304.05128 | 논문 Sec 2가 가장 가깝다고 지목한 연구. |


로봇 쪽 저장소에는 이 논문과 겹치는 분석이 없습니다(arXiv 번호로 대조).

## 3. 구성 요소별 출처와 자료 상태

| 구성 요소 | 상태 |
|---|---|
| 실험 모델 GPT-3.5, GPT-4 | API 모델입니다. 논문은 고정 스냅샷(gpt-3.5-turbo-0301, gpt-4-0314)을 씁니다. 지금은 쓸 수 없는 스냅샷일 가능성이 큽니다 (확인하지 않음). |
| CodeLlama-13b-instruct | 공개 가중치 (huggingface.co/codellama/CodeLlama-13b-Instruct-hf, 논문 각주 5). 크기가 커서 이번에는 받지 않았습니다. |
| APPS 300과제 | 논문 부록 H에 과제 번호가 있습니다. 데이터셋 자체는 받지 않았습니다. |
| HumanEval | 공개 데이터이지만 저장소는 수정본(assertion 메시지 포함)을 따로 씁니다. 수정본은 암호화된 파일로 제공되며, 이번에는 받지 않았습니다. |
| 인간 피드백 데이터 (16명, 40프로그램) | 저장소에 없습니다 (README: IRB 때문에 미공개). 부록 E 표만 있습니다. |

## 4. 공식 저장소 (`github.com/theoxo/self-repair`)

`gh api`로 2026-11-22에 확인했습니다. 대조 내용은 `04_code_verification.md`에 있습니다.

| 항목 | 값 |
|---|---|
| 별 | 15 |
| 마지막 push | 2024-05-02 |
| 보관 상태 | archived (읽기 전용) |
| 라이선스 | GitHub 메타데이터 없음, 저장소 안에 LICENSE 파일 없음 (재사용 조건 불명) |
| 대조한 커밋 | `fcda7a193a732d89fc1f610074c719682228fbc4` (2024-05-02, 얕은 clone) |
| 큰 파일 | `*.json`, `*.jsonl`, `*.tar.bz2`가 git-lfs 포인터입니다. APPS 결과 tarball 198 MB, HumanEval 28 MB. |

## 5. 논문 이후 문헌 (확인하지 않음)

이 문서는 2024년 이전 계보와 저장소 자료만 봤습니다. 웹 검색을 쓰지 않았으므로 논문 발표 후의 자기 수리 연구(피드백 품질을 직접 재는 연구 등)는 조사하지 않았습니다. 인용 전에 이 부분을 채워야 합니다.

## 6. arXiv 메타데이터

- 제목: Is Self-Repair a Silver Bullet for Code Generation?
- 제출: 2023-06-16 (v1), 판본 v5 2024-02-02
- 분류: cs.CL (주), cs.AI, cs.PL, cs.SE
- 주석: "Accepted to ICLR 2024. Added additional Code Llama experiments and fixed a data processing error harming Code Llama's reported self-repair performance on HumanEval"
- 저자: Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, Armando Solar-Lezama
