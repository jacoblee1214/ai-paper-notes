# 01. 논문 분석: Toy Models of Superposition

Elhage, Hume, Olsson, Schiefer, Henighan, Kravec, Hatfield-Dodds, Lasenby, Drain, Chen, Grosse, McCandlish, Kaplan, Amodei, Wattenberg, Olah. Transformer Circuits Thread, 2022-09-14 게시. arXiv:2209.10652v1(2022-09-21, cs.LG, 유일한 판, 2026-11-24 확인). 소속은 Anthropic, Harvard. 원문은 HTML 글이고 PDF는 그것을 옮긴 것입니다.

**인용 위치 표기.** 이 글에는 절 번호와 식 번호가 없습니다. 이 저장소의 문서는 소제목 경로로 위치를 적습니다(예: "§Demonstrating Superposition > The Loss"). `paper/paper.txt`에서 소제목으로 찾을 수 있습니다. 식은 본문에 번호 없이 놓여 있어 "식(모델)", "식(손실)", "식(차원수)"처럼 부릅니다.

## 1. 문제

신경망의 뉴런은 한 가지 개념에 대응하기도 하지만, 특히 언어 모델에서는 서로 무관한 입력에 반응하는 다의 뉴런(polysemantic neuron)이 흔합니다(§Introduction, §Empirical Phenomena). 이 글은 그 원인으로 **겹쳐 담기(superposition) 가설**, 곧 "모델이 가진 뉴런 수보다 많은 특징을 거의 직교인 방향에 담는다"를 세우고, 장난감 모델에서 실제로 일어남을 보이려 합니다.

가정(§The Superposition Hypothesis, §Summary: A Hierarchy of Feature Properties):

- **분해 가능성**: 표현이 독립적으로 이해할 수 있는 특징들로 나뉜다.
- **선형성**: 특징은 활성화 공간의 방향 W_i이고, 여러 특징이 켜지면 Σ x_i W_i로 표현된다.
- **superposition**: W^T W가 가역이 아니면(특징 수 > 차원 수) superposition이 있다. 기저에 정렬되면 W_i가 원-핫이다.

두 힘의 경쟁이 핵심입니다. 더 많은 특징을 담는 **이득**과, 특징들이 직교로 들어가지 못해 생기는 **간섭**입니다. 희소한 특징은 동시에 켜질 일이 적어 간섭 비용이 작고, 출력의 ReLU가 작은 잡음을 걸러 줍니다.

## 2. 방법(토이 모델)

### 2.1 데이터 (§Demonstrating Superposition > Experiment Setup > The Feature Vector)

n개 특징 x ∈ R^n. 각 x_i는 **확률 S_i로 0**, 아니면 [0,1] 균등분포. 특징마다 중요도 I_i. 실험은 대부분 S_i = S로 같게 둡니다(각주 8: 균등분포는 임의의 선택).

### 2.2 모델 (§The Model)

- 선형 모델: h = Wx, x' = W^T h + b = W^T W x + b
- ReLU 출력 모델: x' = ReLU(W^T W x + b)
- W ∈ R^{m×n}, m < n. 복원에 W^T를 다시 쓰는 이유는 방향의 모호성을 없애려는 것입니다(각주 9: W가 정규직교이면 W^T = W^{-1}, 압축 센싱의 "거의 직교" 직관).
- 이 글은 편향을 `+b`로 쓰지만 단계도 절에서는 `ReLU(W^T W x − b)`라고 적었습니다(부호 표기가 일관되지 않음, 학습되는 값이라 결과에는 영향 없음).

### 2.3 손실 (§The Loss)

L = Σ_x Σ_i I_i (x_i − x'_i)².

### 2.4 이론 (§Mathematical Understanding)

- 선형 모델: PCA와 같다고 말하고(상위 m개 주성분), Saxe et al.의 선형 신경망 동역학과 연결합니다. "특징 이득"과 "간섭"이 경쟁하지만 선형 모델에서는 m개를 넘겨 담을 가치가 없다고 각주 12에서 말합니다(유도는 주지 않음).
- ReLU 모델: 입력을 희소 패턴별로 쪼개 L = (1−S)^n L_n + … + (1−S)S^{n−1} L_1 + S^n L_0. S → 1이면 L_1과 L_0가 지배합니다. L_0은 양의 편향에 대한 벌점 Σ ReLU(b_i)², L_1(입력 하나만 켜짐)은 "특징 이득" + "간섭" 두 항이며, 간섭 항은 일반화된 **Thomson 문제**(구 위에 점 배치)와 닮았습니다. 식 자체는 그림(이미지)으로 실려 PDF 텍스트에 없습니다. ReLU가 **음의 간섭을 공짜**로 만들고, 음의 편향이 작은 양의 간섭을 사실상 음의 간섭으로 바꾼다고 설명합니다.

### 2.5 단계도 (§Superposition as a Phase Change)

- n=2, m=1에서 특징 1의 중요도 1.0, 특징 2의 중요도 0.1–10, 희소도 1.0–0.01을 훑고 10개 모델의 평균(최악 모델 제외)으로 "특징 2를 안 배움 / 겹쳐 배움 / 전담 차원으로 배움"을 분류합니다.
- 이론("토이의 토이"): 세 가지 해 W=[1,0], W=[0,1], W=[1,−1](대척점)의 손실을 닫힌 형식으로 구해 비교합니다. 손실 곡선이 교차해 **1차 상전이**라고 주장합니다. 닫힌 형식 자체는 별도 노트북에만 있고 본문에는 없습니다.
- n=3, m=2에서는 네 가지 해(W ⊥ [0,0,1] 등)를 비교합니다.

### 2.6 기하 (§The Geometry of Superposition)

- 균일 중요도·희소도, n=400, m=30. D* = m/‖W‖_F²("특징당 차원")이 1과 1/2 근처에서 "끈적"하다고 보고합니다.
- **특징 차원수** D_i = ‖W_i‖² / Σ_j (Ŵ_i·W_j)² (§Feature Dimensionality). 대척점 쌍이면 1/(1+1)=1/2. 점이 모이는 분수: 3/4(정사면체), 2/3(삼각형), 1/2(대척점), 2/5(오각형), 3/8(정사각 반프리즘), 0(안 배움). Thomson 문제의 해와 대응하고, 정다면체가 아닌 해는 텍검(tegum) 곱으로 이해합니다.
- **다각체와 저랭크 행렬의 대응**: W^T W와 n개 점의 배치가 같은 것(§Aside: Polytopes and Low-Rank Matrices).
- 비균일(§Non-Uniform Superposition): 한 특징만 바꾸면 다각형이 부드럽게 찌그러지다 임계점에서 다른 다각체로 "튄다"(오각형 → 두 대척점 쌍). 상관된 특징은 직교(다른 텍검 인자)로, 반상관 특징은 겹쳐서 음의 간섭으로 배치된다고 합니다.

### 2.7 그 밖의 절

| 절 | 주장 |
|---|---|
| §Superposition and Learning Dynamics | 학습 중 특징 차원수가 "에너지 준위"처럼 점프하고 그때 손실이 계단형으로 떨어진다. 상관 특징에서는 기하 변환의 순서가 보임. 분명한 조사는 후속으로 미룸 |
| §Relationship to Adversarial Robustness | superposition이 생기면 L2 적대적 공격에 대한 취약성이 3배 넘게 커지고 취약성이 특징/차원(= 1/D)과 함께 간다. 적대적 학습으로 줄일 수 있으나 공격이 입력 노름의 80%여야 완전히 없어짐 |
| §Superposition in a Privileged Basis | 은닉층에 ReLU를 넣으면 특징이 뉴런에 정렬되고, 한 뉴런에 주특징 + 부특징이 섞이는 다의 뉴런이 생김 |
| §Computation in Superposition | h = ReLU(W1 x), y' = ReLU(W2 h + b)로 abs(x)를 계산하게 학습(n=100, m=40). 희소도가 커지면 뉴런 수보다 많은 특징의 절댓값을 계산함. 비대칭 superposition + 억제 모티프 |
| §The Strategic Picture | 해결책 세 갈래: (1) superposition 없는 모델(은닉층 L1 규제, MoE 등), (2) 사후에 과완비 기저를 찾기(희소 코딩), (3) 혼합. 2번은 "수백만 뉴런 × 수십억 토큰 행렬의 극단적 과완비 분해"라는 공학 과제로 서술 |
| §Discussion, §Open Questions | 장난감 모델의 타당성 근거는 다의 뉴런이 있다는 정성적 일치. 일반화될 것 같은 것: superposition, 다의/단의 뉴런 공존, 적대적 예제와의 연결. 불확실: 기하와 학습 동역학 |
| Appendix | 비선형 압축 예시. 압축 센싱 하한(RIP, m = Ω(k log(n/k)))을 장난감 모델에 적용한 정리 1 |

## 3. 실험 요약과 규모

- 모델은 n = 2, 3, 5, 20, 80, 100, 200, 400, 은닉 m = 1, 2, 3, 5, 20, 30, 40. **전부 합성 데이터이고 실제 모델에서 측정한 것이 없습니다.**
- 옵티마이저·스텝·배치 같은 학습 설정은 글에 없고 노트북(`anthropics/toy-models-of-superposition`)에 AdamW lr 1e-3, 배치 1024, 스텝 1만이 기본값으로 있습니다(`04` D 절, 정적으로 읽음).
- 단계도의 모델 수: 격자 점마다 10개. 시드·분산은 보고하지 않습니다.
- m = 2에서는 경사 하강이 어렵고 "모델을 여러 번 학습해 손실이 가장 낮은 해를 취했다"고 적었습니다(§Perturbing a Single Feature의 재구현 참고).

## 4. 주장과 근거 위치

| # | 주장 | 위치 | 근거의 성격 |
|---|---|---|---|
| 1 | 선형 모델은 상위 m개 특징만 배운다(PCA) | §Basic Results, §Mathematical Understanding, 각주 12 | 실험 + Saxe 인용 (유도는 없음) |
| 2 | ReLU 출력 모델은 희소해지면 특징을 m개보다 많이 겹쳐 배운다 | §Basic Results | 실험(n=20, m=5와 n=80, m=20) |
| 3 | superposition은 특징 이득-간섭의 경쟁으로 이해된다 | §Mathematical Understanding | L_1 식 + Thomson 문제 비유 |
| 4 | 특징 하나의 상태(안 배움/겹침/전담)는 중요도와 희소도의 함수이고 상전이(1차)가 있다 | §Superposition as a Phase Change | n=2·m=1, n=3·m=2의 실험과 세 해/네 해 이론 |
| 5 | 균일 superposition에서 특징 차원수는 1, 3/4, 2/3, 1/2, 2/5, 3/8 등에 머문다(Thomson 해와 대응) | §Feature Dimensionality, §Why These Geometric Structures? | n=400, m=30 산점도, 기하 그래프 |
| 6 | 학습 중 차원수가 점프하고 손실이 계단형으로 준다 | §Superposition and Learning Dynamics | 단일 사례 시각화 |
| 7 | superposition이 적대적 취약성을 키운다 | §Relationship to Adversarial Robustness | 한 모델 설정의 공격 실험 |
| 8 | 신경망은 특징이 superposition일 때도 계산(절댓값)을 할 수 있다 | §Computation in Superposition | 학습된 모델의 가중치 시각화 |
| 9 | 장난감 모델이 실제 모델의 다의 뉴런을 설명한다 | §Discussion | 정성적 일치(InceptionV1 깊이별, 초기 MLP 뉴런)만, 측정 없음 |
| 10 | 독립 재현: 기본 결과, 차원수, 에너지 준위 점프 | §Replication | OpenAI 정렬팀 구성원 두 명이 재현했다고 기재(코드와 수치는 글에 없음) |

## 5. 한계(저자 인정과 우리 지적)

저자 인정(§Discussion, §Open Questions): 장난감 모델의 현실성 불명, 기하·학습 동역학은 장난감에 특이할 수 있다("too elegant to be true"), 닫힌 형식 해는 n=2·m=1만 부분적으로 있다, "실제 모델의 특징 중요도·희소도 곡선을 추정할 수 있나", "superposition이 규모를 키우면 사라지나"가 열려 있다.

우리 지적:

1. **정량 주장 중 본문에서 식으로 닫힌 것이 거의 없습니다.** 1번 주장의 선형 모델 결과는 각주로만 있고, 3번의 L_1 식은 그림으로 실려 있고, 4번의 이론은 노트북에 있습니다. 이 저장소가 `03`에서 직접 유도·확인했습니다.
2. **"희소도"가 두 가지 의미로 쓰입니다.** S는 특징이 0일 확률이고 그림 축은 1−S 또는 1/(1−S)입니다. 입력이 희소한 것(특징의 켜짐 확률)과 뉴런 활성이 희소한 것(MoE 논의)은 다릅니다.
3. **상전이 판정이 "10개 평균에서 최악 제외"에 의존하고 시드 분산이 없습니다.** 우리는 `03`에서 격자 일치율과 경계 위치로 다시 봅니다.
4. **실제 모델 증거가 없습니다.** 장난감에서 일어남 ≠ 실제 모델에서 일어남. 저자도 §Discussion에서 "토이 모델이 해당 현상과 일치"한다고만 합니다. 이후 문헌은 `04`.
5. **데이터가 독립 균등 + 같은 희소도**라 특징 간 상관·계층 구조가 없는 경우가 기본입니다(상관은 별도 절의 질적 실험).
6. **적대적 취약성 주장(7번)은 한 설정의 분석적 공격이라** 일반 주장으로 읽기엔 약합니다. 저자도 "추측을 삼가겠다"고 적었습니다.
7. 논문 PDF는 HTML 글의 인쇄본이라 인터랙티브 그림은 정지 이미지이고, 본문 식 일부가 이미지라 텍스트로 추출되지 않습니다.

## 6. 판본

arXiv에는 v1만 있습니다(2022-09-21). 글의 게시일은 2022-09-14(Transformer Circuits). 이 저장소의 `paper/paper.pdf`는 arXiv v1입니다. 이후 수정 이력(웹 글의 정오표)은 확인하지 않았습니다.
