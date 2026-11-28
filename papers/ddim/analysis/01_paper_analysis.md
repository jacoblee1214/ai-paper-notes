# 01. 논문 분석: Denoising Diffusion Implicit Models (arXiv:2010.02502)

- 저자: Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford). ICLR 2021 포스터(iclr.cc 공식 일정 데이터에서 `decision: Poster`, "Poster Session 06", OpenReview 포럼 `St1giarCHLP`).
- 분석 판: arXiv v4(2022-10-05, 22쪽). 판 이력은 v1 2020-10-06, v2 2021-11-04, v3 2022-06-09, v4 2022-10-05이고 arXiv 코멘트는 "ICLR 2021; updated connections with ODEs at page 6, fixed some typos"입니다. 즉 §4.3의 ODE 연결(식 14–15, 명제 1)이 나중 판에서 갱신됐습니다. v1 본문은 받지 않아 판 사이 수식 변화는 대조하지 않았습니다.
- 표기: 논문의 $\alpha_t$는 DDPM의 누적곱 $\bar\alpha_t$입니다(부록 C.2). 이 문서는 논문 표기 대신 $\bar\alpha_t$를 씁니다.

## 1. 문제

DDPM은 표본 하나에 $T=1000$번 순차 호출이 필요합니다(2080 Ti에서 32×32 5만 장에 약 20시간, 256×256 5만 장에 약 1000시간, §1). 이유는 생성 과정이 "마르코프 순방향 과정의 역"을 근사하도록 묶여 있어 순방향 길이만큼 단계가 필요하다는 것입니다. 질문은 **학습은 그대로 두고, 생성 과정만 바꿔 단계 수를 줄일 수 있는가**입니다.

## 2. 방법 (식 번호는 논문 그대로)

**(a) 관찰: DDPM 목적은 주변분포만 쓴다 (§3).** $L_\gamma$(식 5)는 $q(x_t|x_0)$(식 4)에만 의존하고 결합분포 $q(x_{1:T}|x_0)$에는 의존하지 않습니다. 같은 주변분포를 갖는 결합분포가 많으므로 비마르코프 순방향 과정을 고를 수 있습니다.

**(b) 비마르코프 족 $\mathcal Q$ (식 6–7).** $\sigma\in\mathbb R_{\ge0}^T$로 색인되는
$$q_\sigma(x_{t-1}|x_t,x_0)=\mathcal N\!\Big(\sqrt{\bar\alpha_{t-1}}x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\tfrac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},\ \sigma_t^2I\Big),\quad q_\sigma(x_T|x_0)=\mathcal N(\sqrt{\bar\alpha_T}x_0,(1-\bar\alpha_T)I).$$
평균의 계수는 모든 $t$에서 $q_\sigma(x_t|x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$이 되도록 고른 것입니다(보조정리 1, 부록 B 식 22–28). 순방향 $q_\sigma(x_t|x_{t-1},x_0)$는 베이즈로 얻으며(식 8) $x_0$에도 의존해 마르코프가 아닙니다. $\sigma\to0$이면 $x_0,x_t$를 알 때 $x_{t-1}$이 결정됩니다.

**(c) 생성 과정 (식 9–10).** $f_\theta^{(t)}(x_t)=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta^{(t)}(x_t))/\sqrt{\bar\alpha_t}$(식 9)로 $x_0$를 예측하고 $p_\theta^{(t)}(x_{t-1}|x_t)=q_\sigma(x_{t-1}|x_t,f_\theta^{(t)}(x_t))$(식 10, $t=1$은 $\mathcal N(f_\theta^{(1)}(x_1),\sigma_1^2I)$)로 둡니다.

**(d) 정리 1 (§3.2, 부록 B 식 29–35).** 변분 목적 $J_\sigma$(식 11)가 모든 $\sigma>0$에 대해 $L_\gamma+C$와 같습니다. 따라서 $\gamma=1$로 학습한 DDPM 모델이 모든 $\sigma$의 해로 쓰입니다(파라미터를 $t$ 사이에 공유하지 않을 때 최적해가 $\gamma$에 무관하기 때문). 각주 5: $\sigma=0$은 정리에 포함되지 않고 $\sigma$를 아주 작게 해 근사한다고 적습니다.

**(e) 표집식 (식 12).**
$$x_{t-1}=\sqrt{\bar\alpha_{t-1}}\underbrace{\tfrac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta^{(t)}(x_t)}{\sqrt{\bar\alpha_t}}}_{\text{예측 }x_0}+\underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\ \epsilon_\theta^{(t)}(x_t)}_{x_t\text{를 향하는 방향}}+\sigma_t\epsilon_t.$$
$\sigma_t=\sqrt{(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)}\sqrt{1-\bar\alpha_t/\bar\alpha_{t-1}}$이면 DDPM, $\sigma_t=0$이면 DDIM(암묵 확률 모델, 표본이 $x_T$의 결정적 함수)입니다.

**(f) 가속 (§4.2, 부록 C.1 식 52–59).** 부분 수열 $\tau=[\tau_1<\dots<\tau_S]$ 위에서만 순방향 과정을 정의합니다. 나머지 $t\notin\tau$는 별 모양(스타 그래프)으로 $x_0$에만 연결해 목적에서 $x_0|x_t$ 항으로 처리합니다. 같은 모델을 그대로 씁니다. 식 12의 $t-1$을 $\tau_{i-1}$로 바꾸기만 하면 됩니다.

**(g) ODE 연결 (§4.3, 식 13–15, 명제 1, 부록 B 식 36–51).** 식 12를 $\bar x=x/\sqrt{\bar\alpha}$, $\bar\sigma=\sqrt{(1-\bar\alpha)/\bar\alpha}$로 다시 쓰면 $\bar x_{t-\Delta t}=\bar x_t+(\bar\sigma_{t-\Delta t}-\bar\sigma_t)\epsilon_\theta$(식 13)이고, 이는 $d\bar x=\epsilon_\theta(\bar x/\sqrt{\bar\sigma^2+1})\,d\bar\sigma$(식 14)의 오일러 스텝입니다. 최적 $\epsilon_\theta$에서 이 ODE는 Song 등(2021, 같은 해 병행 연구)의 VE SDE 확률 흐름 ODE와 같다는 것이 명제 1입니다. 그러나 논문은 "우리는 $d\sigma(t)$에 대해, Song 등은 $dt$에 대해 오일러 스텝을 밟는다"고 설명하고, 확률 흐름 ODE의 오일러 스텝을 식 15($\tfrac12(\bar\sigma_{t-\Delta t}^2-\bar\sigma_t^2)\,\bar\sigma_t^{-1}\epsilon_\theta$ 꼴)로 적으며 두 업데이트가 스텝이 적을 때는 다르다고 말합니다. 결정적 ODE이므로 $x_0\to x_T$ 인코딩이 가능합니다.

**(h) 실험용 매개변수화 (식 16, 부록 D.3).** $\sigma_{\tau_i}(\eta)=\eta\sqrt{(1-\bar\alpha_{\tau_{i-1}})/(1-\bar\alpha_{\tau_i})}\sqrt{1-\bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i-1}}}$, $\eta=1$이 DDPM, $\eta=0$이 DDIM. $\hat\sigma_{\tau_i}=\sqrt{1-\bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i-1}}}$는 DDPM의 더 큰 분산($\beta_t$)이며 Ho 등의 구현이 CIFAR10 표본에만 썼다고 적습니다. $\hat\sigma$ 갱신은 방향 항은 $\eta=1$ 그대로이고 잡음 계수만 $\hat\sigma$로 바뀝니다.

**(i) 부분 수열 선택 (부록 D.2).** 선형 $\tau_i=\lfloor ci\rfloor$, 이차 $\tau_i=\lfloor ci^2\rfloor$, $\tau_{-1}$이 $T$ 가까이 오도록 $c$를 고름. CIFAR10은 이차, 나머지는 선형(해당 데이터에서 FID가 약간 낫다고만 적음).

## 3. 실험 (표·그림 번호)

모델은 전부 $T=1000$, $\gamma=1$로 학습된 DDPM 모델입니다. CIFAR10, LSUN Bedroom·Church는 Ho 등의 사전학습 체크포인트이고 CelebA 64×64는 저자가 직접 학습했습니다(부록 D.1).

| 위치 | 내용 | 값 |
|---|---|---|
| 표 1 | CIFAR10·CelebA FID, $S\in\{10,20,50,100,1000\}$ × $\eta\in\{0,0.2,0.5,1,\hat\sigma\}$ | CIFAR10 $\eta=0$: 13.36/6.84/4.67/4.16/4.04, $\eta=1$: 41.07/18.36/8.01/5.78/4.73, $\hat\sigma$: 367.43/133.37/32.72/9.99/3.17. CelebA $\eta=0$: 17.33/13.73/9.17/6.53/3.51 |
| 그림 3 | 같은 $x_T$, $S\in\{10,100\}$, $\eta$별 표본(CIFAR10, CelebA) | 정성적 |
| 그림 4 | 5만 장 생성 시간 대 $S$ | 선형 |
| 그림 5 | 같은 $x_T$, 서로 다른 $S$의 DDIM 표본 | 정성적 일관성 |
| 그림 6 | $x_T$ 구면 보간(slerp), $S=50$ | 정성적 |
| 표 2 | CIFAR-10 시험 집합 재구성 오차(화소 $[0,1]$ 척도 차원당 MSE, $10^{-4}$ 반올림), $S=10..1000$ | 0.014/0.0065/0.0023/0.0009/0.0004/0.0001/0.0001 |
| 표 3(부록) | LSUN Bedroom·Church FID, DDIM 대 DDPM | Bedroom DDIM 16.95/8.89/6.75/6.62, DDPM(η=1) 42.78/22.77/10.81/6.81, 1000단계 DDPM 6.36. Church DDIM 19.45/12.47/10.84/10.58, DDPM 51.56/23.37/11.16/8.27, 1000단계 7.89 |

## 4. 주장과 근거 위치

| # | 논문의 주장 | 위치 | 이 저장소의 확인 |
|---|---|---|---|
| 1 | 임의의 $\sigma$에서 $q_\sigma$의 주변분포가 DDPM과 같다 | 보조정리 1, 식 22–28 | 해석과 몬테카를로로 확인(`03` §1.1) |
| 2 | $\eta=1$이 DDPM의 조상 표집 | §4.1, 식 12, 16 | 1000단계 전부에서 분산 1e-17, 계수 1e-13 일치(`03` §1.2) |
| 3 | $J_\sigma=L_\gamma+C$, 같은 모델이 모든 $\sigma$를 푼다 | 정리 1, 식 29–35 | **결론은 성립, 부록 식 30–35의 $\gamma_t$는 틀림**(`03` §1.3) |
| 4 | DDIM은 재매개화 ODE의 오일러 | 식 13–14 | 1e-13 수준에서 스텝별 일치(`03` §1.4) |
| 5 | 그 ODE는 VE SDE의 확률 흐름 ODE와 같다 | 명제 1, 식 36–51 | 유도 재현, 수치는 `score_sde`가 함 |
| 6 | 식 15(확률 흐름 오일러)는 스텝이 적을 때 DDIM과 다르다 | §4.3 | 확인: $S=10$에서 거리 10배(`03` §2.4) |
| 7 | 스텝이 적을 때 $\eta=0$이 가장 좋고 $\eta$가 커질수록 나쁘다 | §5.1, 표 1 | 정확한 $\epsilon^*$ 토이에서 $\eta\in\{0,.2,.5,1\}$ 순서는 맞고 $\hat\sigma$는 조건부(`03` §2.1) |
| 8 | 10×–100× 가속 | 초록, §1, §5.1 | 표 1 재계산: CIFAR10은 10–20×, CelebA·Church는 아님(`03` §3.2) |
| 9 | 같은 $x_T$는 $S$와 무관하게 같은 고수준 특징(일관성) | §5.2, 그림 5 | 토이에서 라벨 일치 97.6–99.7%(`03` §2.6) |
| 10 | $x_T$ 보간이 의미 있는 보간을 준다 | §5.3, 그림 6 | 토이에서 98.6% 경로가 봉우리를 되돌아가지 않음(`03` §2.6) |
| 11 | $S$가 크면 재구성 오차가 매우 작다 | §5.4, 표 2 | 토이 기울기 -1.07, 표 2 기울기 -1.19(`03` §2.5, §3.3) |

## 5. 한계와 읽을 때의 주의

- **논문이 직접 적은 한계:** 이산 데이터용 비마르코프 확장은 실험하지 않았습니다(부록 A). 다단계 방법(Adams–Bashforth 등)이 스텝 수를 더 줄일 수 있을 것이라고만 적고 시도하지 않았습니다(§6).
- **표 1·3은 FID 한 줄입니다.** 표집 반복, 신뢰구간, 학습 시드 반복이 없고, FID 표본 수는 본문에 없습니다(공식 코드의 `sample_fid`는 50,000장). 작은 차이($\eta=0$ 대 0.2)의 순위는 이 표로는 가릴 수 없습니다(`03` §3.4).
- **"$\hat\sigma$이 1000단계에서 DDIM보다 약간 나쁘다"는 서술은 수치와 다릅니다.** §5.1은 CIFAR10 1000단계에서 DDIM이 $\hat\sigma$보다 "marginally worse"라 쓰는데 표 1은 4.04 대 3.17(-21%), CelebA 3.51 대 3.26(-7%)입니다.
- **일관성·보간은 정성적 그림**(그림 5, 6)이고 정량 지표가 없습니다.
- **재구성 오차(표 2)는 인코더가 따로 있는 것이 아니라 같은 ODE의 왕복**이라 "오차가 작다"는 DDIM의 ODE 이산화 오차가 작다는 뜻이지 잠재 변수가 사전분포를 따른다는 뜻이 아닙니다(`03` §2.5에서 토이로 분리).
- **$\gamma_t$ 오류:** 부록 식 30–35의 중간 식에서 평균 계수가 빠져 최종 가중이 틀립니다. 정리의 존재 주장(어떤 $\gamma>0$이 있다)은 유지됩니다(`03` §1.3).
- 한 번의 학습과 단일 모델 계열(U-Net, 픽셀 공간)에서의 결과라, 잠재 공간 모델이나 큰 해상도, 조건부 생성에서 $\eta$·격자의 효과는 이 논문에서 알 수 없습니다.
