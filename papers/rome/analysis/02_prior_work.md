# 02. 계보, 가까운 대안, 이후 문헌, 관련 분석

범위: 이 논문이 무엇을 이어받았는지(§1), 가까운 방법과의 관계(§2), 이후 무엇을 바꿨는지(§3), 이 저장소의 다른 분석과의 관계(§4), 자료 표(§5). 이전 문헌의 서술은 논문 §4(Related Work)와 §3 본문에 근거합니다. 이후 문헌은 2026-11-25에 arXiv API로 받은 초록·코멘트와, 아래에 "본문 확인"이라고 적은 것은 PDF 본문을 읽은 것만 근거로 하고, 나머지는 "초록 기준"으로 표시했습니다.


## 1. 계보

### 1.1 인과 매개 분석에서 인과 추적으로

- **Pearl(2001)의 직접·간접 효과**, 그리고 이를 LM에 처음 쓴 **Vig et al.(2020a,b)**의 뉴런·attention head 매개 분석(성별 편향)이 틀입니다(§2.1, §4). ROME의 새 점은 **개별 hidden state 벡터를 매개체로 하는 쌍(손상 + 복원) 개입**과, 이를 (토큰 × 층) 격자 전체로 훑어 평균(AIE)을 내는 것입니다(§4: "paired interventions that allow explicit measurement of causal indirect effects of individual hidden state vectors").
- 이웃: Finlayson et al.(2021, 구문 일치), Feder et al.(2021, 표현·가중치 개입 틀), Elazar et al.(2021b, 정보 삭제). 모두 §4의 서술.
- 이후 이 방법은 **activation patching**, **denoising**이라는 이름으로 퍼졌고, 잡음(GN)과 대칭 토큰 교체(STR) 같은 손상 규칙의 선택이 결과를 바꾼다는 방법론 비판이 나왔습니다(`04` C5).

### 1.2 MLP = 키-값 기억

- **Geva et al.(2021)**: 트랜스포머 MLP가 키-값 기억(첫 층 뉴런 = 키, 둘째 층 = 값)이라고 봄. ROME은 뉴런 단위가 아닌 **선형 연상 기억 W K ≈ V**로 모델링하고, 이는 Geva의 뉴런별 관점과 다르다고 직접 밝힙니다(§3, 각주 6 인근).
- **Kohonen(1972), Anderson(1972)**의 선형 연상 기억, 의사역행렬 해 W = V K⁺(§3.1).
- **Bau et al.(2020, 생성 모델 다시 쓰기)**: 새 키-값 쌍을 제약 최소제곱으로 넣는 방식과 rank-one 갱신 규칙을 제안. ROME은 이를 완전연결층의 닫힌 형식으로 옮겼습니다(§3.1, App. A: "rank-one update rule that was proposed in Bau et al. (2020)"). 즉 **갱신식 자체는 새롭지 않고**, 새 것은 (k*, v*)를 인과 추적이 가리킨 자리에서 고르는 방식(식 (3), (4))입니다.

### 1.3 모델 편집의 이웃 방법 (논문이 비교한 것, §3.2, §4, App. E)

| 방법 | 아이디어 | 논문 속 결과 (Table 4, GPT-2 XL) |
|---|---|---|
| FT | MLP 한 층을 미세조정 | ES 100.0, NS 40.4(특이성 실패) |
| FT+L | L∞ 노름 제약 미세조정(Zhu et al. 2020) | PS 48.7(일반화 실패) |
| KN | knowledge neurons(Dai et al. 2022): 기울기 기여로 뉴런을 골라 임베딩을 더함 | ES 28.7(편집 실패) |
| KE, MEND | 하이퍼네트워크가 편집 갱신을 예측(De Cao et al. 2021, Mitchell et al. 2021) | 일반화 또는 특이성 한쪽이 무너짐 |
| ROME | 인과 추적이 가리킨 층에 rank-one | S 89.2 |

## 2. 가까운 대안과 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| (토큰 × 층) 인과 격자, AIE | 이 논문 | 기반은 Pearl 2001, Vig et al. 2020 |
| 손상 = 주어 임베딩에 가우스 잡음 3σ | 이 논문 | 후속 비판: Zhang & Nanda 2023(`04` C5) |
| MLP를 선형 연상 기억으로 | Kohonen 1972, Anderson 1972, Geva et al. 2021(뉴런별 관점은 다름) | |
| rank-one 제약 최소제곱 갱신 | Bau et al. 2020 | 식 (2), 부록 A는 고전적 등식 제약 최소제곱 |
| C = K Kᵀ를 위키백과 토큰으로 추정 | 이 논문 | 정규화·백색화 역할(`03` §2.4) |
| v* = 목적어 확률 극대화 + KL 항 | 이 논문 | KL은 "essence drift" 억제(식 4b) |
| 일반화·특이성·일관성·유창성 평가 틀 | CounterFact(이 논문), ParaRel(Elazar et al. 2021a) 기반 | §3.3 |

## 3. 이후 무엇을 바꿨나 (자세한 검증은 `04`)

| 주제 | 문헌 | 한 줄 요약 (본문 확인 / 초록 기준) |
|---|---|---|
| 많은 사실 한 번에 | MEMIT (Meng et al. 2022, arXiv:2210.07229, ICLR 2023) | 같은 저자들이 여러 층에 편집을 분산해 GPT-J, GPT-NeoX에서 수천 개 연상을 한 번에 넣음 (초록 기준). 이 논문 §3.7도 후속으로 인용 |
| 국소화가 편집을 예측하는가 | Hase et al. 2023 (arXiv:2301.04213, NeurIPS 2023 Spotlight) | 인과 추적 효과가 편집 성공과 거의 상관 없음, 편집 층 선택이 R² 0.947을 설명 (본문 확인) |
| 사실 회상의 메커니즘 | Geva et al. 2023 (arXiv:2304.14767, EMNLP 2023) | 주어 마지막 위치의 풍부화(초기 MLP) → 관계 정보 → 예측 위치가 attention head로 속성을 추출. ROME의 early site를 확인하면서 attention의 역할을 더 크게 봄 (초록 기준) |
| 편집 뒤 연쇄 추론 | MQuAKE (Zhong et al. 2023, arXiv:2305.14795, EMNLP 2023), RippleEdits (Cohen et al. 2023, arXiv:2307.12976, TACL 2024) | 편집한 사실 자체는 회상하지만 다단계 질문과 파급 사실에서 크게 실패 (초록 기준) |
| 반복 편집 | Gupta et al. 2024 (arXiv:2401.07453, ACL 2024 Findings), Gupta et al. "Rebuilding ROME" (arXiv:2403.07175, EMNLP 2024) | ROME·MEMIT 순차 편집이 점진적 망각 뒤 급격한 붕괴. 일부 "disabling edit"는 ROME 구현의 키 벡터 비대칭 사용 때문 (본문 일부 확인) |
| 일반 능력 훼손 | Gu et al. 2024 (arXiv:2401.04700, EMNLP 2024) | 편집이 추론, NLI, QA 일반 능력을 해칠 수 있어 정규화를 제안 (초록 기준) |
| 평가·종합 | Yao et al. 2023 (arXiv:2305.13172, EMNLP 2023) | LLM 편집 문제, 방법, 기회의 종합과 비교 (초록 기준) |
| 패칭 방법론 | Zhang & Nanda 2023 (arXiv:2309.16042, ICLR 2024), Heimersheim & Nanda 2024 (arXiv:2404.15255) | 잡음(GN) 대 대칭 교체(STR), 지표 선택이 국소화 결과를 바꿈. 모범 사례 권고 (본문 일부 확인, 후자는 초록 기준) |

## 4. 이 저장소의 다른 분석과의 관계

- **`../toy_superposition/`(30번)**: 특징이 차원보다 많아 방향에 **겹쳐** 저장되면 키 공간이 직교하지 않습니다. ROME의 최소제곱 구조에서 C = KKᵀ는 키들의 비직교성(겹침)을 보정하는 항이고, C = I(직교 키 가정)이면 k*와 겹치는 기존 키의 출력이 바뀝니다(`03` §2.2, §2.4). 이 폴더의 토이에서 C = I와 추정한 C를 비교했습니다(`03` §4.5). 다만 이 토이의 MLP 키가 superposition 장난감처럼 희소·독립 특징의 겹침인지는 확인하지 않았습니다. 같은 문제가 아닙니다.
- **`../spike_sparse_sink/`(28번)**, **`../attention_sink_emerges/`(27번)**: 중간층 hidden state의 극단적 activation(massive activation)이 어떤 위치와 층에 몰리는지를 다룹니다. 인과 추적은 "어느 상태가 예측에 결정적인가"를 개입으로 묻고, 싱크 분석은 "어느 상태가 크고 특이한가"를 관측합니다. 두 지도가 겹치는지(예: 첫 토큰 싱크가 주어 위치 복원에서 어떤 역할인지)는 확인한 바 없습니다. 비교하려면 같은 모델에서 둘 다 재야 합니다.
- **`../induction_heads/`(29번)**, **`../sparse_autoencoders/`(31번)**: 작성 중이거나 예정이라 링크만 둡니다. 29번의 작은 attention 모델과 이 폴더의 토이는 모델 구조가 다릅니다(여기는 MLP가 있는 6층, 합성 사실 암기).

## 5. 자료 표 (조회 2026-11-25)

| 자료 | 주소 | 라이선스 | 비고 |
|---|---|---|---|
| 논문 | arXiv:2202.05262 (v5) | arXiv 비독점 배포 라이선스 | CC 아님 |
| 공식 코드 | github.com/kmeng01/rome | MIT (GitHub API) | 별 795, 포크 172, 마지막 push 2024-04-20, 마지막 커밋 2022-10-14(`0874014`), 저장소 크기 약 23 MB(API KB 23173), clone 작업 트리 1.2 MB |
| 공식 페이지 | rome.baulab.info | 확인 안 함 | 논문이 코드, 데이터셋, 시각화, 데모를 둔다고 적음 |
| NeurIPS 2022 | proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html | | DOI 10.52202/068431-1262, OpenReview 노트 `-h6WAS6eE4`("NeurIPS 2022 Accept") |
| CounterFact | 논문 §3.3, 공식 저장소 `dsets/` | 저장소와 같다고 보이나 확인 안 함 | 21,919 레코드 |
