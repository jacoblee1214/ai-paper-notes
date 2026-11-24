# 02. 계보와 관련 자료: Sparse Autoencoders Find Highly Interpretable Features in Language Models (arXiv:2309.08600)

이후 문헌의 서술은 **arXiv 초록과 메타데이터(arXiv API, 2026-11-25 확인)** 또는 아래에 적은 게시 페이지에서 읽은 대목에만 근거합니다. 본문 전체를 읽지 않았으므로 초록에 없는 세부는 쓰지 않았습니다. 학회 표기는 arXiv comment 필드와 OpenReview에서 확인한 것만 적었습니다.

## 1. 계보: 이 논문이 기대는 것

| 구성 요소 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 희소 사전 학습(sparse dictionary learning): 데이터를 과완비 사전의 희소 선형 결합으로 씀 | Olshausen & Field 1997(*Vision Research* 37(23)), 2004; Lee et al. 2006(효율적 희소 코딩 알고리즘, NeurIPS 19) | 문제 정의 그대로(L71–L75, L93–L95). 식 (4)는 L1 희소 코딩의 목적 함수와 같음 |
| 희소 코딩이 정답 사전을 복원한다는 이론 | Wright & Ma 2022(교재); Qu et al. 2019(arXiv:1912.02427, 과완비 표현 학습의 최적화 지형) | "L1 재구성이 정답 특징을 복원할 수 있다"의 근거(L143–L145). 부록 C.4에서 ℓ4 노름과 첨도로 연결 |
| 활성에 희소 사전 학습을 적용 | Yun et al. 2021(arXiv:2103.15949): 모든 잔차 층에 동시 적용 | 가장 가까운 선행 연구로 명시(L76–L78) |
| superposition 가설과 다의성의 원인 | Elhage et al. 2022b = 30번 `toy_superposition/`(arXiv:2209.10652) | 동기(L54–L71) |
| 토이 데이터에서 SAE가 정답 특징을 복원 | **Sharkey, Braun, Millidge 2022**(AlignmentForum 중간 보고, 2022-12-13, Conjecture 소속). 단위 구에 무작위로 뽑은 G개 정답 방향, 평균 5개 동시 활성, 확률 지수 감쇠 옵션, 정답 복원 지표로 **MMCS** 도입. L1 계수에 "골디락스 구간"이 있음. 이 논문 본문 방법 절의 "Building on Sharkey et al. (2023)"(L76) | 이 논문은 정답을 모르는 실제 모델만 다루므로 **정답 복원 근거를 이 중간 보고에 빌려 옴**(L143–L144). MMCS 정의는 이 글(우리 03 §3) |
| 자동 해석(autointerpretability) | Bills et al. 2023(OpenAI): GPT-4가 뉴런 설명을 쓰고 시뮬레이터가 activation을 예측해 상관으로 채점 | §3의 핵심 평가. 시뮬레이터를 GPT-3.5로 바꾼 이유는 로그확률 API(L572–L575) |
| 활성 패칭과 회로 탐색 | Vig et al. 2020(인과 매개), Wang et al. 2022(IOI, arXiv:2211.00593), Conmy et al. 2023(ACDC) | §4 |
| 잔차 흐름의 기저는 특별하지 않음, 단 이상치 차원 존재 | Elhage et al. 2023(privileged basis), Dettmers et al. 2022 | 기저를 기준선으로 쓰는 이유(L52–L53, 부록 C.2) |
| 아키텍처로 희소성을 유도하는 대안 | SoLU(Elhage et al. 2022a), L1 활성 벌점, 가지치기 | 대안으로 소개하고 사후 분해를 택한 이유(L390–L397) |

이 논문이 **새로 한 것**: (1) SAE를 실제 언어모델(Pythia-70M/410M) 활성에 적용해 기저·무작위·PCA·ICA와 자동 해석 점수로 비교, (2) IOI 과제에서 SAE 특징으로 반사실 행동의 원인 특징을 지목, (3) 아포스트로피·닫는 괄호 같은 사례 연구와 층 간 회로 트리. 식 (1)–(4)의 오토인코더 자체는 Sharkey et al.의 것이고, 손실에 정규화 사전과 ReLU를 쓰는 것은 표준 희소 코딩의 신경망 근사입니다.

### 1.1 거의 동시에 나온 병행 연구

Anthropic의 *Towards Monosemanticity*(Bricken et al., Transformer Circuits Thread, **2023-10-04 게시**, 이 논문 v3 제출일과 같은 날)는 이 논문을 "Sharkey et al.의 중간 보고를 거쳐 나온 매우 비슷한 주제의 병행 연구"로 소개합니다(게시 페이지에서 확인). 차이는 다음과 같습니다(같은 페이지에서 읽은 것).

| 항목 | 이 논문(v3) | Towards Monosemanticity |
|---|---|---|
| 대상 | Pythia-70M/410M 잔차 흐름(주), MLP(부록) | 1층 트랜스포머의 MLP 활성(512 뉴런), 데이터 80억 점 |
| 사전 크기 | R ∈ {0.5…8}(주로 1–4) | 확장 계수 1×(512)에서 256×(131,072)까지, 상세 분석은 4,096개 |
| 인코더/디코더 | **묶음**(MLP에서만 비묶음) | **비묶음**: 인코더·디코더 코사인 중앙값이 약 0.5라서 묶으면 안 된다고 주장 |
| 죽은 특징 | 재초기화는 "앞으로의 계획"(부록 E) | **주기적 재샘플링**(오래 안 켜진 뉴런의 인코더를 재구성이 나쁜 데이터점으로 재설정) |
| 편향 | 인코더 편향만 | 입력에서 디코더 편향을 빼는 "pre-encoder bias", 기하 중앙값으로 초기화 |
| 평가 | 자동 해석 점수, IOI 패칭 | 사람이 한 개별 특징 분석, 자동 해석, 특징 분할(feature splitting) |

## 2. 가까운 대안과 이론

- **L1 대신 다른 희소화**: 아래 §3의 TopK, Gated, JumpReLU. 모두 이 논문 식 (4)의 L1 벌점이 수축 편향과 L0와의 불일치를 낳는다는 문제의식에서 나옵니다.
- **정답 복원의 관점**: SAE의 인코더가 한 번의 ReLU로 희소 코드를 구하는 "상각된(amortized) 추론"이라 정확한 L1 최소화(lasso)보다 약합니다. 이 차이는 우리 `03` §5의 합성 실험에서 직접 드러납니다(superposition이 4배일 때 복원 실패).
- **다른 해석 접근**: 뉴런 기저 해석(Bills 2023), 방향 탐지 프로브, 활성 조향 벡터(difference-in-means 등). 이후 문헌은 이들이 SAE보다 낫거나 같다고 보고함(§3).

## 3. 이후 문헌 (초록 기준)

1. **같은 방법의 확장과 병행**: Bricken et al. 2023(위), Templeton et al. 2024 *Scaling Monosemanticity*(Claude 3 Sonnet 중간층, SAE 1M/4M/34M 특징; 30번 `toy_superposition/analysis/04` C5가 이미 정리).
2. **아키텍처·학습 개선**: Gated SAE(Rajamanoharan et al., arXiv:2404.16014): "L1 벌점이 수축을 포함한 여러 편향을 낳는다"고 보고 *어느 방향을 쓸지*와 *얼마나 쓸지*를 분리, 7B 모델까지 시험해 수축이 해결되고 비슷한 재구성에 절반의 켜진 특징으로 충분하다고 주장. JumpReLU SAE(arXiv:2407.14435, Gemma 2 9B): ReLU를 불연속 JumpReLU로 바꾸고 STE로 L0를 직접 학습해 "수축 같은 문제를 피함". TopK SAE(Gao et al., OpenAI, arXiv:2406.04093): k-희소 오토인코더로 희소도를 직접 지정하고 죽은 잠재 변수를 줄이는 수정을 더해 크기에 대한 깨끗한 스케일링 법칙을 보임, GPT-4 활성에 1,600만 잠재 변수·400억 토큰. BatchTopK(arXiv:2412.06410), Matryoshka SAE(arXiv:2503.17547).
3. **특징 분할과 흡수**: Chanin et al.(arXiv:2409.14507, NeurIPS 2025 구두): 사전이 커지면 계층적 특징이 쪼개지고("math" → "algebra", "geometry"), 부모 특징이 **자식에게 흡수**되어 있어야 할 곳에서 안 켜진다. 희소성 최적화 자체가 원인이고 크기·희소도 조절로는 해결 안 됨. Matryoshka SAE는 이를 줄이려는 시도.
4. **특징의 불안정성**: Paulo & Belrose(arXiv:2501.16615): 시드만 다른 SAE에서 131K 잠재 변수의 약 30%만 공유, **ReLU+L1 SAE가 TopK보다 시드 간 안정**. Song et al.(arXiv:2505.20254): 특징 일관성 지표 PW-MCC를 제안하고 TopK SAE에서 0.80을 달성 가능. Leask et al.(arXiv:2502.04878, ICLR 2025): SAE 잠재 변수는 정규적 분석 단위가 아님.
5. **해석 지표의 타당성**: Heap et al.(arXiv:2501.17727): 무작위 초기화 트랜스포머의 SAE도 자동 해석 점수와 재구성 지표가 훈련된 모델과 비슷한 경우가 많다 → 높은 점수만으로 "계산과 관련된 특징을 찾았다"고 할 수 없다. 이 논문의 §3(자동 해석)이 이 비판의 표적. Korznikov et al.(arXiv:2602.14111): 정답을 아는 합성 설정에서 설명 분산 71%인데 정답 특징의 9%만 복원, 무작위 방향으로 제약한 SAE 기준선이 해석 0.87 대 0.90으로 맞먹음. SAEBench(arXiv:2503.09532, ICML 2025): SAE 평가 모음.
6. **하류 과제**: Kantamneni et al.(arXiv:2502.16681)은 프로빙 네 영역(데이터 부족, 클래스 불균형, 라벨 잡음, 공변량 이동)에서 SAE가 기준선을 일관되게 못 넘는다고 보고. AxBench(Wu et al., arXiv:2501.17148): 조향에서 프롬프팅과 미세조정이 앞서고 개념 탐지에서 difference-in-means가 최고, SAE는 경쟁력이 없다. 반대로 Marks et al. Sparse Feature Circuits(arXiv:2403.19647, ICLR 2025)는 SAE 특징으로 회로를 찾고 분류기 일반화를 개선(SHIFT)하는 응용을 보임.
7. **다차원 특징**: Engels et al.(arXiv:2405.14860, ICLR 2025): 요일·월 같은 원형(2차원) 특징. SAE가 1차원 방향을 가정하는 데 대한 수정(30번 `04` C4).
8. **로봇·VLA로의 이식(우리 연구와 관련)**: Swann et al. *Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models*(OpenReview 기준 CoRL 2026 포스터; 초록 기준): VLA의 은닉 활성에 SAE를 학습해 동작 원형과 의미 개념에 해당하는 특징을 찾고 LIBERO 시뮬레이션과 DROID 실물에서 조향. 본문은 읽지 않았고 초록만 근거로 했습니다. Häon et al.(arXiv:2509.00328, CoRL 2025)은 SAE가 아니라 FFN 활성을 토큰 임베딩 기저로 사영해 의미 방향을 찾고 Pi0·OpenVLA를 조향.

## 4. 이 저장소의 다른 분석과의 관계

| 폴더 | 관계 |
|---|---|
| [`toy_superposition/`](../../toy_superposition/) (30번) | superposition 가설의 근거. 이 글의 `generate_batch`(희소 U[0,1] 특징)를 복사해 정답 있는 시험대로 썼습니다(`03` §5). 30번 `04` C5, C6은 SAE의 후속 문헌을 이미 정리해 두었고 `04`의 C 절은 그것과 겹치지 않는 항목(수축, 시드 안정성의 직접 측정, MSE vs 해석성)을 위주로 합니다 |
| [`induction_heads/`](../../induction_heads/) (29번) | 입력 모델과 activation(`artifacts/`)을 제공합니다. 29번이 남긴 과제(층 1 head 출력 방향과 SAE 특징의 겹침)를 `03` §8에서 실행했습니다 |
| [`spike_sparse_sink/`](../../spike_sparse_sink/) (28번) | massive activation은 소수 채널이 큰 값을 독점하는 현상이라 SAE 학습에서 재구성 오차를 지배할 수 있는 후보입니다. 29번 모델의 활성에서는 위치 0(BOS)의 노름이 다른 위치의 약 2.3–2.7배(22.7 대 8.5, 25.3 대 11.2)에 그치고 massive 수준은 아니어서(`03` §6) 위치 0만 학습·평가에서 제외했습니다. 큰 모델에서의 영향은 측정하지 않았습니다 |
| [`rome/`](../../rome/) (32번) | 링크만. 지식이 저장되는 층·MLP를 위치 지정하는 방법으로, SAE 특징 수준의 위치 지정과 대비되는 주제입니다 |

## 5. 자료 표

| 자료 | 라이선스 | 규모 | 마지막 갱신 | 비고 |
|---|---|---|---|---|
| arXiv:2309.08600 (v3) | CC BY 4.0 (arXiv abs 페이지) | PDF 1.8 MB, 20쪽 | v3 2023-10-04 | ICLR 2024 포스터(OpenReview `F76bwRSLeK`) |
| `HoagyC/sparse_coding` (GitHub) | **라이선스 파일 없음**(GitHub API `license: null`, 작업 트리에도 LICENSE 없음) | 별 315, 저장소 크기 7.96 MB, 작업 트리 9.0 MB(clone 후 `du -sh`) | 마지막 push 2023-11-10 | HEAD `69c5ae06813ee77bafa679dfacee33b21395d8e4`. 정적으로 읽음, 실행 안 함(`04` D 절). README가 "직접 학습하려면 `sparse_autoencoder` 라이브러리를 쓰라"고 안내 |
| 29번 `induction_heads/artifacts/` | 이 저장소 산출물 | 132 MB | 2026-11-25 | 읽기만, 커밋·게시 안 함 |
