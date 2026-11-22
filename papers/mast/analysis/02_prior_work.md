# 02. 선행 연구, 이후 연구, 자료 정보

## 1. 계보

| 시기 | 연구 | MAST와의 관계 |
|---|---|---|
| 1967 | Glaser & Strauss, Grounded Theory | 분류 도출 방법(§3.1) |
| 2023 | Zheng 외, LLM-as-a-judge(MT-Bench) | 판정자 파이프라인의 근거(§3.3, 참고문헌 [21]) |
| 2024 | Kapoor 외 *AI agents that matter*, Agentless | MAS가 단순 기준선보다 이득이 작다는 동기(§1) |
| 2024 | Bansal 외(MSR TR), 사람-에이전트 소통 과제 목록 | 사람과 에이전트 상호작용용 목록. MAST는 자율 MAS 실행 중 실패에 초점(§2.3) |
| 2024 | AgentEval, AGDebugger | 평가 기준 정의·디버깅 도구(§2.3). 분류 체계가 아님 |
| 2025-05 | Who&When (arXiv:2505.00212) | MAST가 "동시 연구"로 인용(§2.3, [42]). **과업 단위 귀인**(어느 에이전트, 어느 단계)이고 MAST는 **패턴 분류**입니다. 서로 다른 질문 |
| 2025-09 | AgenTracer (arXiv:2509.03312) | 위 귀인 문제를 학습으로 푼 후속. 평가 벤치마크는 Who&When, MAST는 아닙니다(이 저장소 `agentracer/`) |

- **MAST 대 Who&When.** MAST는 trace에 14모드가 있는지를 이진으로 달고(여러 모드 동시 가능, 단계 위치 없음), Who&When은 실패를 일으킨 단일 에이전트와 단계를 정답으로 둡니다. 둘을 합치면 "어떤 모드가 어느 단계에서" 정도의 라벨이 되지만 현재 공개 라벨은 그렇게 연결되어 있지 않습니다.
- **MAST 대 AgenTracer.** AgenTracer는 훈련 데이터 구성에서 MAST를 벤치마크 목록에 올릴 뿐(이 저장소 `agentracer/analysis/02_prior_work.md`) 모드 라벨을 쓰지 않습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 |
|---|---|
| Cohen's κ 해석(0.8 이상 강함, 0.9 이상 거의 완전) | McHugh 2012, v1·v2 본문 인용 |
| 판정자 모델 o1과 few-shot 설계 | OpenAI o1, 공식 노트북 `llm_judge_pipeline.ipynb` |
| 시스템 구성 | MetaGPT, ChatDev, HyperAgent, AppWorld, AG2(MathChat), Magentic-One, OpenManus(표 3) |
| 벤치마크 | ProgramDev(자체 30문항)와 ProgramDev-v2(자체 100문항), SWE-Bench Lite, GSM-Plus, OlympiadBench, MMLU, GAIA, AppWorld Test-C |

## 3. 발표 이후 연구 (웹 검색 2회와 arXiv API로 확인)

| 연구 | 관계 | 확인 범위 |
|---|---|---|
| **AdaMAST** (arXiv:2607.16387, 2026-07, 저자에 MAST 제1저자 포함) | MAST의 손수 만든 분류를 trace에서 **유도한 적응형 분류**로 바꾸는 후속. 초록에 따르면 "사람 주석 없이" 만든 어휘가 전문가 주석과 손수 만든 기준 어휘보다 더 가깝게 맞는다고 주장 | 초록만 읽음. 본문 수치는 확인하지 못함 |
| **Holistic Evaluation and Failure Diagnosis of AI Agents** (arXiv:2605.14865) | TRAIL 벤치마크(GAIA·SWE-bench의 span 단위 오류 주석)에서 span 단위 평가로 SOTA를 주장. MAST와 직접 비교는 하지 않았을 수 있음 | 초록만 읽음 |
| MAST 직접 비판·재분석 | 검색에서 찾지 못함 | 검색 2회 한도 안. **없다는 뜻이 아님** |

- 이 표의 두 항목은 초록 수준입니다. 인용하기 전에 본문을 읽어야 합니다.
- 아래 `03`의 재계산(공개 IAA 파일과 원 주석표의 불일치 등)이 선행 보고와 겹치는지는 조사하지 않았습니다.

## 4. 이 저장소와 이웃 저장소의 관련 분석

- `../agentracer/`(3번, 완료): 실패 귀인. MAST를 벤치마크 목록에서만 만납니다.

## 5. 자료 정보

| 자료 | 주소 | 라이선스 | 비고 |
|---|---|---|---|
| 공식 저장소 | `github.com/multi-agent-systems-failure-taxonomy/MAST` | 없음(GitHub API `license: null`) | 별 441, 마지막 push 2025-07-23, 최종 커밋 `a70542e`(2025-07-23). 작업 트리 1.9 GB(대부분 `traces/MagenticOne_GAIA` 1.1 GB) |
| MAST-Data | `huggingface.co/datasets/mcemri/MAD` → `mcemri/MAST-Data` | CC BY 4.0 | `MAD_full_dataset.json` 199.6 MB(1,642행), `MAD_human_labelled_dataset.json` 2.7 MB(19행). 최종 수정 2026-08-13 |
| 프로젝트 | `sky.cs.berkeley.edu/project/mast/` | 확인 안 함 | |
| 논문 | arXiv v3 | CC BY-NC-ND 4.0 | 그림 게시 불가 |
