# 02. 계보와 관계: AgenTracer

## 1. 계보

| 연도 | 논문 | 관계 |
|---|---|---|
| 2025-03 | MAST, Why Do Multi-Agent LLM Systems Fail? (arXiv:2503.13657) | 실패 모드 14종. 주석자 일치도와 LLM 판정자 확장. AgenTracer가 §2에서 "14개 패턴"으로 인용 |
| 2025-05 | Who&When, Which Agent Causes Task Failures and When? (arXiv:2505.00212) | 실패 귀인 벤치마크. AgenTracer의 **평가 벤치마크**. 공식 초록: 127 LLM 다중 에이전트 로그, 세부 주석 연결 |
| 2025-09 | **AgenTracer** (arXiv:2509.03312) | 반사실 재실행과 결함 주입으로 라벨 생성, Qwen3-8B를 GRPO로 학습 |

## 2. 가까운 대안

- **Who&When 원 논문(arXiv:2505.00212)**: 같은 벤치마크에서 LLM에 한 번에 판정을 맡기는 "all-at-once"를 씀. AgenTracer는 같은 설정(§5.1)을 따르므로 비교가 공정합니다.
- **CollabUIAgents(He et al., arXiv:2502.14496)**: AgenTracer §2가 가장 가까운 선행으로 꼽습니다. LLM이 이진 보상을 줍니다. AgenTracer는 이를 "신뢰성이 의심스럽다"고 비판합니다.
- **AgentDebug(arXiv:2509.25370, 이 목록 10축)**: 같은 시기 근본 원인 단계 찾기. 읽기 목록에서 근본 원인 진단과 가장 가깝다고 표시한 논문입니다. 두 논문을 함께 읽어야 합니다. 아직 분석하지 않았습니다.
- **TRAIL(arXiv:2505.08638)**: 궤적에서 오류 위치 찾기 벤치마크. 이 목록에서 같은 축에 있습니다.

## 3. 구성 요소 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 결정적 오류의 정의(식 4–5) | Zhang et al. 2025c(Who&When) 인용 | AgenTracer §3 "Following (Zhang et al., 2025c)" |
| 반사실 재실행 | 본 논문 | 기존 방법이 아닌 자체 설계 |
| 결함 주입 | 본 논문 | 교란 연산자 Π는 DeepSeek-R1 기반 |
| GRPO | Guo et al. 2025(DeepSeek-R1) | KL 항 제거, 동적 클리핑은 Liu et al. 2025 인용 |
| 자기 개선 비교 | Self-Refine(Madaan 2023), CRITIC(Gou 2024) | 둘 다 GPT-4.1로 실행 |
| 벤치마크 | MAST, Who&When, MBPP+, KodCode, Blackjack, GAIA, MATH, GSM8K, HumanEval+, MATH-500 | 목록 10축과 11축에 대부분 있음 |



## 5. 공식 저장소와 자료 정보 (GitHub API, 2026-11-22 확인)

| 항목 | 값 |
|---|---|
| 저장소 | https://github.com/bingreeky/AgenTracer |
| 확인한 커밋 | `256b19e53e8c7341822e5925c481fb942f606ed8` (2026-08-27, "Update data and model release status") |
| 별 | 114 |
| 마지막 push | 2026-08-27 |
| 라이선스 | **없음**(GitHub API: `license: null`). 최상위에 LICENSE가 없고, `MetaGPT/` 하위 파일만 MetaGPT의 라이선스를 가짐 |
| 데이터 릴리스 | `data-v1.0.0`, zip 24 MB(압축 해제 약 140 MB). 1 GB 미만이므로 받아 계산함 |
| 모델 가중치 | 공개하지 않음. README: "we currently have no plans to release the AgenTracer-8B model weights" |

공식 저장소는 `repo/`에 얕게 clone했습니다(gitignore). 실행하지 않고 정적으로만 읽었습니다(`04_code_verification.md`).

## 6. 학회

ICLR 2026 포스터. 공식 페이지 `iclr.cc/virtual/2026/poster/10007726`에서 제목, 저자, 포스터 형식을 확인했습니다(2026-11-22). 자세한 내용은 README의 `venue_note`를 봅니다.
