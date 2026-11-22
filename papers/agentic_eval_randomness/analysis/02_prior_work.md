# 02. 선행 연구, 후속 연구, 자료

## 1. 계보: 이 논문이 서 있는 자리

"평가 점수의 흔들림을 재고 보고 방식을 고치자"는 흐름 위에 있고, 새 점은 **다단계 에이전트 궤적**에서 대규모로 쟀다는 것입니다.

| 줄기 | 대표 문헌 | 이 논문과의 관계 |
|---|---|---|
| 결정적 설정의 비결정성 | Atil et al., *Non-Determinism of "Deterministic" LLM Settings* (arXiv:2408.04667, 2024-08, Eval4NLP 2025 워크숍) | 5개 LLM, 8개 과제, 10회 반복에서 temperature 0 정확도가 최대 15% 달라지고 최선과 최악 격차가 최대 70%라고 보고합니다(초록). 이 논문은 같은 질문을 단발 생성이 아니라 에이전트 루프로 옮겨 12개 설정에서 확인합니다. 추론 엔진 원인은 He & Lab(2025), Yuan et al.(2025)을 인용합니다(§2.4.1, §4.1) |
| 평가 통계 | Miller, *Adding Error Bars to Evals* (arXiv:2411.00640, 2024-11, 학회 표기 없음) | 문항을 모집단에서 뽑은 표본으로 보고 Var(μ̂) = (Var(x) + E[σ_i²])/n으로 나눕니다(§3). 문항당 K번 재표집하면 두 번째 항만 1/K로 줄고, 두 모델 비교는 문항 쌍체(식 7), 검정력은 최소 탐지 효과로 계획합니다(§5). 이 논문은 "반복 실행과 신뢰구간을 요구하는 제안"의 하나로 한 줄 인용합니다(§4.2). 문항 모집단 항 Var(x)와 쌍체 비교는 이어받지 않았고, 이 논문의 σ는 "문항 고정, 실행 간"만 잽니다 |
| 반복 부족 비판 | Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (arXiv:2108.13264, NeurIPS 2021, 우수 논문상) | 적은 실행으로 보고하는 관행을 비판하고 IQM과 층화 부트스트랩 구간을 제안합니다. 이 논문은 이 문헌을 "계산이 비싼 벤치마크에서 실행 수를 줄이는 관행"의 근거로 인용합니다(§4.2) |
| pass@k 계열 | Chen et al. (2021) pass@k, Yao et al. (2025) pass^k | 식 3의 추정량. pass@k는 낙관 경계, pass^k는 비관 경계(§2.3) |
| 에이전트 평가 안정성 | Mustahsan et al. (2025) 급내 상관, Heineman et al. 신호 대 잡음비, Shen et al. (2026, SWE-bench Verified 표준편차 중앙값 1.2%) | §4.1이 직접 관련 연구로 듭니다 |
| 재현성 일반 | Henderson et al. (2018), Dodge et al. (2019), Biderman et al. (2024) | 보고 표준화 제안(§4.2) |

식 11의 필요 실행 수는 교과서적인 두 표본 검정의 표본 크기 공식입니다. 새로운 통계 방법이 아니라 이 분산 값 위에 적용한 것입니다.

## 2. 구성 요소별 출처

- **모델:** Qwen3-32B(Yang et al., 2025), DeepSWE-preview(Luo et al., Qwen3-32B 미세조정), Devstral-2-123B(Rastogi et al., 2025). 온도는 각 모델 카드의 권장값입니다(표 4).
- **스캐폴드:** nano-agent(저자 자체 최소 스캐폴드, 별도 저장소), R2E-Gym(Jain et al., DeepSWE-preview 학습에 사용된 스캐폴드).

## 3. 발표 이후 같은 질문을 다룬 연구 (arXiv API로 존재 확인, 2026-11-22)

| 논문 | 내용 | 이 논문과의 관계 |
|---|---|---|
| *Identical Runs, Different Results: Benchmarking AI Coding Agents on Open-Weight Models* (arXiv:2609.33812, Ariño de la Rubia & Pafka, 2026-09) | 한 ML 과제에서 에이전트 6개 × 오픈 가중치 모델 6개, 584회 실행, 여섯 조합은 52회씩 반복 | 같은 현상을 "실행 간 차이가 이 과제에서 무엇을 뜻하나"로 반복 횟수를 늘려 확인한다고 초록이 밝힙니다. 본문은 읽지 않았습니다 |
| *A Trust Layer for Agent Evaluation* (arXiv:2610.07274, Sediqin et al., 2026-10) | 점수가 "두 번째 실행에서도 유지되는가"를 포함한 신뢰성 점검 틀 | 검색 결과 요약에서 이 논문을 인용한다고 했을 뿐 확인하지 않았습니다 |
| *Engineering Reliable Coding Agents* (arXiv:2608.13867, Jarmak, 2026-08) | 164편을 종합한 단행본 | 마찬가지로 인용 여부만 검색 요약에 의존합니다 |

세 편 모두 본문을 정독하지 않았으므로 인용할 때는 직접 확인해야 합니다(웹 검색 1회, 요약만 봄). 이 논문 자체의 반박이나 수정 논문은 찾지 못했습니다.

## 5. 자료 표

| 자료 | 위치 | 크기 | 라이선스 | 비고 |
|---|---|---|---|---|
| 논문 PDF (v3) | arxiv.org/abs/2602.07150v3 | 0.8 MB | CC BY 4.0 (arXiv abs 확인 2026-11-22) | `paper/` |
| 코드 `ASSERT-KTH/agentic-evals-lab` | github.com/ASSERT-KTH/agentic-evals-lab | 저장소 3.2 MB (API), clone 1.7 MB | 라이선스 파일 없음 (API `license: null`) | 별 9, 마지막 push 2026-03-20, 커밋 e1036f6. 학습(GRPO·SFT)과 평가 프레임워크 |
| 스캐폴드 `ASSERT-KTH/nano-agent` | github.com/ASSERT-KTH/nano-agent | 0.4 MB | 라이선스 파일 없음 | 별 8, 마지막 push 2026-01-16. 위 저장소가 `andre/fix/thinkink-param` 브랜치를 의존성으로 고정. 읽지 않았습니다 |
| 결과 `ASSERT-KTH/agentic-evals-artifacts` | huggingface.co/datasets/ASSERT-KTH/agentic-evals-artifacts | 약 48 GB(궤적 포함) | CC BY 4.0 (카드) | 설정 12개 × 10 실행. 이 분석은 실행별 평가 결과 JSON 120개(5.8 MB)만 받았습니다 |
| 학회 | iclr.cc/virtual/2026/10016323 | | | Poster in Workshop: *Agents in the Wild: Safety, Security, and Beyond* (페이지 curl 확인) |
