---
title: "확산 모델을 SDE로 통일하면 무엇이 정확하고 무엇이 조건부인가: 정확한 점수로 다시 읽은 Score SDE"
date: 2026-11-28
venue: "ICLR 2021"
venue_note: "구두 발표. iclr.cc 공식 일정 데이터(iclr-2021-orals-posters.json)에서 decision Oral, 세션 Outstanding Paper Session 1로 편성된 것을 확인(OpenReview 포럼 PxTIG12RRHS). 수상 공지 페이지는 열지 못했고 우수 논문 세션 편성으로 확인했습니다. 분석 판은 arXiv v2(2021-02-10)입니다."
tags: [생성 모델, 확산 모델, SDE, 확률 흐름 ODE, 예측기-교정기, flow matching, 통계, 구현 모드]
paper: "Score-Based Generative Modeling through Stochastic Differential Equations (arXiv:2011.13456)"
code: "https://github.com/yang-song/score_sde_pytorch (저자 공식, PyTorch, Apache-2.0; 원본 JAX는 yang-song/score_sde)"
summary: "SMLD와 DDPM을 연속 SDE의 두 이산화로 통일한 Score SDE의 식들을, 신경망 대신 가우시안 혼합의 닫힌 식 점수로 검증했습니다. 핵에서 유도한 식 9, 11, 12와 sub-VP 분산 상한, 확률 흐름 ODE의 연속 방정식, 식 39의 정확한 우도(최대 오차 7e-7)는 맞았습니다. 같은 NFE에서 PC 표집기의 이점은 2차원 정확한 점수에서는 NFE 2000에서 표본 잡음과 구별되지 않았고, 논문 표 1은 VP의 DDPM형 예측기에서 5회 표집 잡음 안이었습니다. 초록의 FID 2.20, IS 9.89, 2.99 bits/dim은 서로 다른 두 모델의 값입니다. PF-ODE 속도는 flow matching의 주변 속도이고 flow matching 손실은 가중치 g^4/4의 점수 매칭입니다. 인페인팅의 관측 대체 근사는 다봉 토이에서 크게 틀렸습니다. 학습, 설치, 다운로드, 공식 코드 실행은 하지 않았습니다."
---

# 확산 모델을 SDE로 통일하면 무엇이 정확하고 무엇이 조건부인가

*Score-Based Generative Modeling through Stochastic Differential Equations*(Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, arXiv:2011.13456, ICLR 2021 구두·우수 논문 세션)는 SMLD(NCSN)와 DDPM을 연속 시간 SDE의 이산화로 통일하고, 같은 주변분포의 결정적 ODE로 정확한 우도와 인코딩을 얻습니다. 이 글은 식을 직접 검산하고 논문 표를 통계적으로 다시 읽은 기록입니다. **모델 학습, 가중치·데이터 다운로드, 설치, 공식 코드 실행, API 호출은 하지 않았습니다.** 논문 그림은 arXiv 라이선스가 CC BY가 아니라서 싣지 않았습니다.

## 학습 없이 확인하는 방법

데이터가 가우시안 혼합이면 모든 시각 $t$에서 $p_t=\sum_kw_k\mathcal N(s\mu_k,s^2\Sigma_k+\sigma^2I)$이고 점수가 닫힌 식입니다. 점수 오차가 없으니 SDE 풀이기, 교정기, 확률 흐름, 우도, 조건부 생성의 성질만 따로 볼 수 있습니다. 2차원, 성분 3개(가중치 0.5/0.3/0.2), PyTorch float64, CPU, pytest 40개입니다.

## 식은 맞습니다

- 섭동 핵 $\mathcal N(s(t)x_0,\sigma(t)^2I)$에서 $g^2=d\sigma^2/dt-2f\sigma^2$로 유도한 $f,g$가 식 9(VE), 11(VP), 12(sub-VP)와 1e-9 이내입니다. 오일러-마루야마 몬테카를로로 핵의 평균·분산도 맞습니다.
- 이산 DDPM 핵은 VP 핵으로 정확히 $1/N$ 속도로 수렴합니다($N=1000$에서 8e-4). 그림 5의 "거의 일치"의 수치판입니다.
- sub-VP는 VP보다 분산이 작은데, 차이는 초기 분산과 무관하게 정확히 $e^{-B}-e^{-2B}$입니다(최대 1/4).
- 논문 안에 작은 불일치가 있습니다. VE 핵이 식 29에서는 $\sigma^2(t)-\sigma^2(0)$, 식 31에서는 $\sigma^2(t)$로 $\sigma_{\min}^2=10^{-4}$만큼 다릅니다.

## 확률 흐름 ODE와 정확한 우도

순방향 포커-플랑크 방정식을 $\partial_tp=-\nabla\!\cdot[(f-\tfrac12g^2\nabla\log p)p]$로 쓰면 식 13이 곧 연속 방정식입니다. 정확한 점수에서 $\partial_tp$와 $-\nabla\!\cdot(p\tilde f)$가 $10^{-6}\max p$ 이내로 맞고, 확률 흐름(RK4)과 역방향 SDE(오일러)의 $t=0.4$ 평균·공분산이 참 분포와 표준오차 안에서 같습니다. 인코딩은 결정적이고 왕복 오차가 5e-9입니다.

- **VE의 사전분포.** $p_T$ 대신 $\mathcal N(0,\sigma_{\max}^2I)$에서 출발하면 확률 흐름은 4.8 표준오차 어긋나고 SDE는 1.1 표준오차입니다. VP·sub-VP는 해당 없습니다.
- **식 39.** $\frac{d}{dt}\log p_t(x(t))=-\nabla\!\cdot\tilde f$이므로 $\log p_0=\log p_T+\int\nabla\!\cdot\tilde f$. 정확한 $p_T$에서 최대 오차 7e-7(RK4 오차). 가우시안 사전분포를 쓰면 VE에서 최대 0.26 nats, VP에서 0.01 nats.
- **허친슨.** 2차원 라데마허 프로브의 분산은 $4M_{12}^2$($M=\int J\,dt$)로 이론 0.733, 경험 0.735 nats². 가우시안 프로브는 VP에서 3.6배, VE에서 42배 분산이 큽니다.

## 같은 계산량에서 PC는 얼마나 나은가

점수 호출 횟수(NFE)를 맞춰 비교했습니다(시드 12, $n=4000$, 슬라이스 W1, Holm 보정). 표본 잡음 바닥은 0.038입니다.

- NFE 2000에서는 예측기 단독, 교정기 단독, PC가 모두 바닥과 구별되지 않습니다(0.031–0.044). 정확한 점수와 2차원에서는 1000스텝 이산화 오차가 표본 잡음보다 작아서입니다.
- NFE 100에서 교정기 단독은 PC보다 0.06–0.07 나쁩니다. 확률 흐름 예측기는 교정기와 함께일 때 크게 좋아집니다(VP NFE 100: 0.162에서 0.056).
- 교정기의 신호 대 잡음비 규칙 $\epsilon=2\alpha(r\|z\|/\|s\|)^2$에서 $r$은 스텝의 신호/잡음 노름 비 그 자체이고, 정확한 점수에서도 정상 분산을 $v\to v(1+r^2)$로 부풀립니다(r=0.4에서 측정 1.155, 예측 1.160).

**논문 표 1을 다시 읽으면.** 표는 학습된 모델 하나에서 표집만 5회 반복한 값입니다. Welch 검정과 Holm 보정(20개 비교)에서 VE의 세 예측기와 VP의 확률 흐름은 PC가 유의하게 낫지만(Holm $p<2\times10^{-4}$), VP의 조상·역확산은 5회 표집 잡음 안입니다(PC1000 3.18±.01 대 P2000 3.19±.02, $p=0.76$). "역확산이 조상 표집보다 항상 낫다"는 방향은 6개 비교 모두 맞지만 유의한 것은 2개입니다. 학습 시드 변동은 표에 없습니다.

## 초록의 세 수치는 한 모델이 아닙니다

FID 2.20과 IS 9.89는 NCSN++ cont. (deep, VE)(표 3, PC 표집, 학습 중 FID 최소 체크포인트)이고, 2.99 bits/dim은 DDPM++ cont. (deep, sub-VP)(표 2, ODE, 마지막 체크포인트)입니다. 같은 sub-VP 모델의 FID는 표 2에서 2.92, 표 3에서 2.41입니다. 표 2의 DDPM ELBO 값은 이산 데이터 기준이고 나머지는 균일 역양자화 기준이라 직접 비교가 아닙니다. sub-VP가 VP보다 NLL이 낮은 것은 세 쌍 모두 같은 방향(-0.14–-0.16)이지만 부호 검정 $p=0.25$이고 시드가 없습니다. 후속으로 최대우도 학습(Song 2021b)이 CIFAR-10에서 2.83을 보고했습니다(arXiv 초록 기준).

## flow matching으로 가는 다리

핵 $x_t=s x_0+\sigma\epsilon$에서 PF-ODE 속도는

$$\dot x=\dot s\,E[x_0|x_t]+\dot\sigma\,E[\epsilon|x_t],\qquad E[\epsilon|x]=-\sigma\nabla\log p_t,\ E[x_0|x]=\tfrac{x+\sigma^2\nabla\log p_t}{s}$$

로, 조건부 흐름의 속도를 $x_t$로 조건 기댓값 낸 flow matching 주변 속도장입니다. VE, VP, sub-VP, 직선 경로에서 점수와 독립인 사후 평균 공식으로 따로 계산한 값과 1e-8 이내입니다. 점수 모델에서 속도를 만들어 flow matching 손실을 쓰면

$$\|v_\theta-v_{\rm target}\|^2=\tfrac{g(t)^4}{4}\,\|s_\theta-\nabla\log p_{0t}\|^2$$

즉 식 7의 $\lambda(t)=g^4/4$입니다(논문 권장 $\sigma^2$, 우도 가중 $g^2$와 다름). 직선 경로에서는 $\epsilon$-오차의 가중이 $1/(1-t)^2$로, 같은 묶음의 SD3 분석에서 계산한 값과 같습니다.

정확한 점수의 확률 흐름을 균일 격자 오일러로 풀면(NFE 5–200) VP·sub-VP는 20스텝에서 표본 잡음 바닥(0.038)에 닿고, 직선 경로는 100–200스텝, VE는 200스텝에서도 바닥의 약 2배입니다. "직선 경로가 스텝 효율이 좋다"는 직관은 이 토이·이 격자에서는 확인되지 않았습니다. 격자와 스케줄이 이산화 오차를 정한다는 EDM·DPM-Solver 계열의 관점과 방향이 같습니다(후속 논문은 초록만 확인).

## 조건부 생성: 식은 정확하고 근사는 조건부입니다

식 49의 분해는 베이즈 규칙의 분류기로 3.6e-15 이내에서 정확하고, 클래스 조건부 표집이 성분 분포와 W1 0.014–0.018로 맞습니다. 그러나 인페인팅(식 50)의 "알려진 좌표를 섭동된 관측으로 덮어쓰기"는 날카로운 다봉 조건부에서 크게 틀렸습니다. $x_1=0$을 관측하면 정답은 한 모드(평균 3.0, 표준편차 0.51)인데 근사는 평균 0.53의 여러 모드 표본을 만들었고 W1이 2.47 [2.45, 2.50]입니다. 정확한 1차원 조건부 점수를 쓴 대조군은 0.011입니다. 잡음이 클 때 섭동 관측이 모드 정보를 거의 갖지 않아 $x_2$가 먼저 모드를 정한다는 것이 추정 기작이며 검증하지 않았습니다. 논문은 이 근사의 오차 조건을 논하지 않습니다.

## 공식 코드와의 대조

`score_sde_pytorch`(커밋 `cb1f359`, 정적 읽기만): 신호 대 잡음비 스텝 규칙, 시간 절단(VP·sub-VP $10^{-3}$, VE $10^{-5}$, 학습·우도 $10^{-5}$), RK45 atol=rtol=1e-5, sub-VP의 핵과 확산이 논문과 같습니다. 차이는 (D2) PC 루프가 교정 후 예측 순서라 알고리즘 의사코드와 다르고, (D3) 시간 격자 `linspace(T, eps, N)`와 오일러 스텝 `dt=-1/N`이 $N\epsilon=1$일 때만 일치하며(VE는 마지막 구간 시각 오차 최대 약 $10^{-3}$), (D4) VP·sub-VP의 마지막 잡음 제거는 트위디가 아닌 평균 스텝입니다(같은 크기의 계수).

## 인용할 때

- 초록의 세 수치는 두 모델의 값입니다. 2.99 bits/dim은 이후 최대우도 학습에 갱신되었습니다.
- 표 1·2·3은 학습 시드 반복이 없고 표 3은 FID 최소 체크포인트입니다.
- "확률 흐름이 SDE보다 나쁘다"는 풀이기·격자에 달렸습니다.
- 식 6, 13, 39, 49는 정확한 점수에서 정확합니다. 인페인팅 근사는 다봉 조건부에서 틀릴 수 있습니다.

코드와 결과: 최소 구현 `analysis/minimal_impl.py`, 검사 `analysis/test_minimal_impl.py`, 표 재계산 `analysis/paper_tables.py`.

## 자료

- [최소 구현](analysis/minimal_impl.py)
- [01. 논문 분석: Score-Based Generative Modeling through Stochastic Differential Equations](analysis/01_paper_analysis.md)
- [02. 계보, 가까운 대안, 구성 요소의 출처, 관련 분석](analysis/02_prior_work.md)
- [03. 분석: 식 유도 검산, 정확한 점수 최소 구현, 논문 표 재계산](analysis/03_analysis.md)
- [04. 주장 검증(C)과 코드 대조(K, D)](analysis/04_claim_verification.md)
