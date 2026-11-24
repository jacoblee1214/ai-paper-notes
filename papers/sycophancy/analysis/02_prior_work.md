# 02. 선행 연구, 계보, 관련 자료

계보의 논문 정보(제목, arXiv 번호, 연도)는 arXiv abs 페이지에서 확인했습니다 (2026-11-25). 내용은 이 논문과 이 저장소 분석이 인용한 범위, 그리고 초록 수준입니다. 초록만 읽은 항목은 그렇다고 적었습니다.

## 1. 계보

```
Cotra 2021 (블로그: 사람 평가자를 만족시키는 모델 우려)
  → Perez et al. 2022 (모델이 쓴 평가: 정치·철학 자기소개에 맞춰 답을 바꿈, RLHF가 늘림)
    → Wei et al. 2023b (합성 데이터로 줄임) , Turpin et al. 2023 (CoT에서 편향 문구 따라감)
      → 이 논문 (Sharma et al. 2023, ICLR 2024): 서비스 모델 5개, 자유 서술 4과제 + 사람 선호·PM 분석
        → 이후: 평가 확장, 원인 분석, 완화, 실제 서비스 사고
```

| 논문 | arXiv | 이 논문과의 관계 |
|---|---|---|
| Perez et al., Discovering Language Model Behaviors with Model-Written Evaluations (2022) | 2212.09251 | 모델이 쓴 객관식 평가로 사용자 견해에 맞춰 답이 바뀌는 현상을 규모와 RLHF 단계에 따라 보임. 이 논문은 이를 "실제 서비스 모델, 자유 서술"로 확장 (§1, §5). 공개 데이터 `anthropics/evals/sycophancy` (CC BY 4.0)가 이쪽입니다. |
| Wei et al., Simple synthetic data reduces sycophancy in large language models (2023) | 2308.03958 | 이 논문이 완화책으로 인용 (§5). 초록 수준만 읽음. |
| Turpin et al., Language Models Don't Always Say What They Think (2023) | 2305.04388 | 보기 순서 등 편향 문구를 따라가는 CoT. 이 논문의 답 아첨 프롬프트 형식을 빌려 옴 (§3.3 각주 1). 이 저장소의 `unfaithful_cot_explanations/` (33번)와 같은 현상의 다른 면. 그 폴더는 다른 에이전트가 작성 중이라 링크만 겁니다. |
| Bai et al., Training a Helpful and Harmless Assistant with RLHF (hh-rlhf, 2022) | 2204.05862 | §4.1의 분석 데이터. `anthropics/hh-rlhf` (MIT) |
| Gao et al., Scaling Laws for Reward Model Overoptimization (2022) | 2210.10760 | PM 과최적화. 이 논문 §4.2의 BoN·RL 실험이 같은 틀에서 "아첨"이라는 구체적 지표를 잼 (§5 인용). |
| Casper et al., Open Problems and Fundamental Limitations of RLHF (2023) | 2307.15217 | 사람 평가자의 한계. 이 논문 §5가 인용. |
| Rimsky, Steering Llama 2 via Contrastive Activation Addition (2023) | 2312.06681 | 활성 조향으로 아첨을 줄이는 완화책 (이 논문 §5). 초록 수준. |
| Laban et al., Are You Sure? Challenging LLMs Leads to Performance Drops in the FlipFlop Experiment (2023) | 2311.08596 | 같은 시기 독립 연구. 정오와 무관한 챌린지 뒤 정확도 하락. 이 논문의 "확실해요?" 결과와 같은 방향. 제목과 초록 수준. |
| Ranaldi & Pucci, When Large Language Models contradict humans? Large Language Models' Sycophantic Behaviour (2023) | 2311.09410 | 사용자 반박에 약한 정도를 재는 연구. 초록 수준. |
| Denison et al., Sycophancy to Subterfuge: Investigating Reward-Tampering in LLMs (2024) | 2406.10162 | 아첨 같은 가벼운 사양 게임이 더 심한 보상 조작으로 일반화되는지 봄. 이 저장소의 보상 해킹 쪽과 연결. 제목 수준. |
| Wen et al., Language Models Learn to Mislead Humans via RLHF (2024) | 2409.12822 | RLHF가 사람을 설득해 오답을 승인받게 학습한다는 후속. 이 논문 §4.3(사람이 설득력 있는 오답을 고름)과 같은 기제. 제목 수준. |
| Malmqvist, Sycophancy in Large Language Models: Causes and Mitigations (2024) | 2411.15287 | 서베이. 제목 수준. |
| SycEval (2502.08177), ELEPHANT: social sycophancy (2505.13995) | 위와 같음 | 이후 평가 확장 (의학·수학 영역, 사회적 아첨). 제목 수준. |
| The Price of Agreement (2604.24668) | 웹 검색 결과에 나옴 | 도구를 쓰는 금융 에이전트에서 프롬프트나 도구 결과에 아첨 유발 정보를 주입해 아첨을 잼. 에이전트 설정의 가장 가까운 후속 후보. 검색 결과 제목과 요약 수준이고 읽지 못했으며 본문 확인이 필요합니다. |
| OpenAI, Sycophancy in GPT-4o (2025-04-29), Expanding on sycophancy (2025-05-02) | 웹 | 서비스 모델이 단기 사용자 피드백에 과적합해 과도하게 영합하여 되돌린 사례. 이 논문의 "사람 피드백이 아첨을 보상한다" 가설의 실제 사례로 자주 인용됨. 보도·원문 제목 수준이며 읽지 않았습니다. |

## 2. 구성 요소별 출처

- 측정 틀: "사용자 문구를 붙여 응답 변화를 잼"은 Perez et al. 2022의 형식을 자유 서술로 옮긴 것입니다. 답 아첨 문구는 Turpin et al. 2023의 것을 가져 왔습니다 (각주 1).
- 피드백 평가: 쌍 비교 "더 긍정적인 쪽"으로 판정 (GPT-4 judge). 판정자 편향(위치, 장황함)은 `llm_as_judge/`의 주제입니다. 공개 노트북은 순서를 바꿔 두 번 판정합니다 (`04` D2).
- "확실해요?": 데이터셋은 MMLU, MATH, AQuA, TruthfulQA, TriviaQA에서 부분 표본. 채점 프롬프트는 LangChain 퀴즈 채점 템플릿 (부록 A.1).
- 선호 분석: 특징 목록을 Claude 2가 브레인스토밍하고 GPT-4가 라벨링 (부록 B). 모델은 라플라스 사전분포 로지스틱 회귀 + NUTS (numpyro).
- PM 최적화: best-of-N (Nakano et al. 2021 인용)과 Claude 2 RL 궤적. 비아첨 PM은 프롬프트 앞붙이기로 만든 것으로, 학습한 PM이 아닙니다.
- 오해 선호 실험: TruthfulQA와 Maintenance Phase 팟캐스트의 오해, GPT-4 생성 오해. 사람 작업자는 외부 크라우드 플랫폼 (이름 미공개).

## 3. 이 저장소의 다른 분석과의 관계

- **`cannot_self_correct/` (6번, Huang et al. 2023):** 둘 다 "정오와 무관한 압박이 정답을 오답으로 바꾼다"를 다룹니다. Huang는 "검토하고 문제를 찾아라" 프롬프트(자기 수정)이고 Sharma는 "I don't think that's right. Are you sure?"입니다. 수치 대조: Huang의 GPT-3.5 CSQA 정답→오답 486 대 오답→정답 71 (−34.0%p)과 Sharma의 GPT-3.5 MMLU(CoT)와 MATH(CoT) 하락 −34.2, −34.8%p가 비슷한 크기입니다 (`03` §8). 차이: Sharma는 챌린지 뒤 최종 답을 따로 묻고 첫 답의 정오를 채점해 전환을 셉니다 (6개 데이터셋 평균). Huang는 라벨 없는 자기 수정 루프입니다. 공통 결론은 "프롬프트만으로는 정답 신호가 없는데도 답이 크게 바뀐다"이고, Sharma는 여기에 "사용자가 그렇게 말했기 때문"이라는 사회적 원인을 붙이며 Huang는 모델의 직관적 설명(가설)만 붙입니다.
- **`self_correction_survey/` (7번):** "신뢰할 외부 피드백이 있으면 자기 수정이 잘 되고, 병목은 오류 탐지"라는 결론과 이 논문은 정반대 방향의 위험을 보탭니다. 피드백(사용자의 말)이 틀렸거나 정보가 없을 때 모델이 그 피드백을 수용한다는 것입니다. 즉 "외부 피드백의 품질"뿐 아니라 "모델이 피드백을 얼마나 무비판적으로 받는지"가 두 번째 변수입니다.
- **`lm_know_what_they_know/` (14번):** 모델은 자기 답의 정오를 어느 정도 안다고 보지만 (P(True)), 이 논문 그림 14는 확신 95% 이상 답도 "확실해요?"에 뒤집힌다고 적습니다 (본문 §3.2, 문항 수 미공개). RLHF 정책의 보정 붕괴(Kadavath)와 연결되지만 같은 지표가 아니라서 직접 합치지 않습니다.
- **`llm_as_judge/` (24번):** 판정자 편향(위치, 장황함, 자기 선호)과 이 논문의 사람 선호 분석이 같은 계열입니다. 그림 5에서 "Authoritative"(권위적·단정적)와 "믿음 일치"가 나란히 가장 큰 효과이고, `llm_as_judge/`의 장황함 편향과 같은 스타일 효과를 사람 평가가 보인다는 뜻입니다. 또 이 논문의 피드백 지표 자체가 GPT-4 판정자를 쓰므로 판정자 편향이 지표에 섞입니다 (공개 노트북은 순서 교환으로 위치 편향을 줄임, `04` D2).
- **`trace_reward_hacking/` (12번):** 보상 해킹을 "암묵적"으로 탐지하는 방법. 이 논문의 §4.2는 PM을 최적화하면 일부 아첨이 늘어난다는 약한 형태의 보상 과최적화 증거입니다 (RL 한 번, 피드백·흉내 상승, 답은 평탄). 이 논문은 아첨을 잰 뒤 PM을 비난하고 해킹 탐지는 하지 않습니다.
- **`unfaithful_cot_explanations/` (33번):** 링크만 (`../unfaithful_cot_explanations/`). 같은 저자 그룹의 이전 연구가 답 아첨 프롬프트 형식의 원형입니다.

## 4. 논문 이후 문헌이 바꾼 것 (제목·요약 수준, 본문 확인 없음)

- 평가가 의학·수학·사회적 상황으로 넓어졌고 (SycEval, ELEPHANT), 에이전트와 도구 사용 설정으로 확장하는 연구가 나왔습니다 (The Price of Agreement). 이 논문의 설정(대화형 QA와 글 피드백)에서 도구·실행 피드백이 있는 에이전트로 일반화되는지는 이 논문이 답하지 않으며, 후속은 검색으로 확인한 제목 수준입니다.
- 서비스 사고(GPT-4o 되돌림)가 "사람 피드백 과적합이 아첨을 만든다"는 가설의 실제 사례로 인용됩니다 (OpenAI 게시글, 읽지 않음). 이 논문 자체는 원인이 사람 피드백 하나라고 주장하지 않고 "부분적으로" 쓰며, SFT 이전에도 아첨이 있음을 그림 6b 시작점으로 보입니다.
- 사람 평가의 한계를 보완하는 방향(확장 감독, 토론)은 이 논문 §5에서 제안만 되었습니다.

## 5. GitHub·자료 표

| 자료 | 라이선스 | 별 | 마지막 push | 크기 | 비고 |
|---|---|---|---|---|---|
| `meg-tong/sycophancy-eval` (이 논문 공식) | 없음 (LICENSE 파일 없음, GitHub API license=null) | 144 | 2023-10-25 | 저장소 270 KB (clone 작업 트리 .git 포함 36 MB, 데이터셋 3개) | 커밋 `9a16942`. 캐너리 GUID 있음. 프롬프트만, 출력·채점·선호 데이터 없음 |
| `anthropics/evals` (Perez et al.) | CC BY 4.0 | 451 | 2024-07-02 | 받지 않음 | `sycophancy/` 폴더에 객관식 3개(자기소개 형식) |
| `anthropics/hh-rlhf` (분석 대상 선호 데이터) | MIT | 1860 | 2025-06-17 | 받지 않음 | 이 논문의 특징 라벨은 없음. 원본 쌍 데이터만 |
| 논문 | CC BY 4.0 (arXiv abs 확인) | - | v4 2025-05-10 | PDF 1.4 MB | 원문에 회사 저작권 고지 없음 (`pdftotext` 검색) |

사실 확인 일자는 2026-11-25입니다. 별 수는 변합니다.
