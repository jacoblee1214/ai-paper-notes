# 04. 코드 대조와 이후 문헌의 확인·수정

## 0. 대상과 범위

- 코드: `github.com/ermongroup/ddim`, 커밋 `e8df66cf02e457719930748b5c26e37e3126dfdd`(2024-07-26, PR #38 병합 커밋, 얕은 clone이라 이력은 한 건), MIT, 얕은 clone(280 KB)을 `repo/`에 두고 **정적으로만 읽었고 실행하지 않았습니다**(실행은 사용자 승인이 필요한 일). 읽은 파일: `functions/denoising.py`(67줄), `runners/diffusion.py`(384줄), `main.py`, `functions/losses.py`, `functions/ckpt_util.py`, `configs/cifar10.yml`, `README.md`. `models/diffusion.py`(U-Net)는 훑기만 했습니다.
- 이 저장소의 분석 코드(`minimal_impl.py`)는 공식 코드를 복사하지 않고 논문 식에서 새로 썼습니다. 공식 코드의 격자를 비교하려고 `make_taus_official`로 규칙만 다시 적었습니다.

## 1. 논문 식과 맞는 부분 (K)

| # | 논문 | 코드 | 판정 |
|---|---|---|---|
| K1 | 식 12, 16: $x_{\tau_{i-1}}=\sqrt{\bar\alpha_{\tau_{i-1}}}f+\sqrt{1-\bar\alpha_{\tau_{i-1}}-\sigma^2}\,\epsilon_\theta+\sigma\epsilon$, $\sigma=\eta\sqrt{(1-\bar\alpha_{\tau_{i-1}})/(1-\bar\alpha_{\tau_i})}\sqrt{1-\bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i-1}}}$ | `generalized_steps`: `x0_t=(xt-et*sqrt(1-at))/sqrt(at)`, `c1=eta*sqrt((1-at/at_next)*(1-at_next)/(1-at))`, `c2=sqrt((1-at_next)-c1**2)`, `xt_next=sqrt(at_next)*x0_t+c1*randn+c2*et` | 일치. $c_1$ 안의 곱을 풀면 식 16과 같음(`(1-at/at_next)`=$\bar\alpha$ 비, 항 순서만 다름) |
| K2 | $\bar\alpha_0:=1$ | `seq_next=[-1]+list(seq[:-1])`, `compute_alpha`가 $\beta$ 앞에 0을 붙여 `t=-1`이 $\bar\alpha=1$ | 일치. 마지막 스텝의 $\sigma$는 0 |
| K3 | $\eta=0$이 기본 | `main.py --eta` 기본 0.0, `--sample_type` 기본 `generalized` | 일치 |
| K4 | 학습은 DDPM과 같음($\gamma=1$, $L_1$) | `losses.noise_estimation_loss`: $\|\epsilon-\epsilon_\theta(\sqrt{\bar a}x_0+\sqrt{1-\bar a}\,\epsilon)\|^2$ | 일치(스케일 D8 참조) |
| K5 | 보간은 구면 선형(부록 D.5 식 67) | `sample_interpolation.slerp`: `theta=acos(sum(z1*z2)/(norm*norm))`, $\sin((1-\alpha)\theta)/\sin\theta\,z_1+\sin(\alpha\theta)/\sin\theta\,z_2$ | 일치. 11점($\alpha=0,0.1,..,1$) |
| K6 | 스케줄 $\beta:10^{-4}\to0.02$, $T=1000$ | `configs/cifar10.yml` `linear`, 1e-4, 0.02, 1000 | 일치(float32) |
| K7 | 사전학습 모델은 DDPM 체크포인트 사용 | `ckpt_util.py`가 DDPM TF 체크포인트를 PyTorch로 옮긴 사본(주석 "pesser/pytorch_diffusion")을 가리킴 | 일치 |

## 2. 논문과 다른 부분 (D)

| # | 논문 | 코드 | 영향 |
|---|---|---|---|
| **D1** | 부록 D.2: 선형 $\tau_i=\lfloor ci\rfloor$, $\tau_S$가 $T$ 근처 | `seq=range(0, T, T//S)`(0 기준 색인). 논문 색인(+1)으로는 $\tau=\{1,1+\text{skip},\dots\}$이고 **마지막이 $T$가 아니라 $T-\text{skip}+1$**. $S=10$이면 901($\bar\alpha=2.7\times10^{-4}$, 논문 격자의 $4.0\times10^{-5}$와 다름) | 시작 잡음 수준이 낮은 쪽으로 어긋남(초기 $x_T\sim\mathcal N(0,I)$는 그대로). 정확한 $\epsilon^*$ 토이에서 SW1 영향은 보이지 않음(0.0680 대 0.0683, `03` §2.3). **또 $T//S$ 정수 나눗셈이라 $S$가 1000의 약수가 아니면 실제 스텝 수가 $S$가 아님:** $S=30$→31, 64→67, 300→334(`range` 길이 산술) |
| **D2** | 부록 D.2: 이차 $\tau_i=\lfloor ci^2\rfloor$, $\tau_S\approx T$ | `seq=(linspace(0, sqrt(0.8*T), S))**2`를 `int`로. **마지막이 $0.8T$=800**(논문 색인 801, $\bar\alpha=1.5\times10^{-3}$)이고 앞쪽이 **중복**($S=50$에서 고유 49, $S=100$에서 96) | 중복 스텝은 $\bar\alpha$가 같아 항등 변환이라 모델 호출만 낭비. 800에서 시작하는 격자는 CIFAR10의 "이차"가 이 코드로 만든 값이면 논문 서술과 다른 격자. 토이에서는 SW1 차이 없음(0.0395 대 0.0401, $S=10$) |
| **D3** | 식 16은 $\eta$ 하나로 $\sigma$ 정의, $\hat\sigma$은 부록 D.3 | $\eta$ 경로(`generalized_steps`)와 $\hat\sigma$ 경로(`ddpm_steps`, `--sample_type ddpm_noisy`)가 **별개 함수**이고 $\eta=1$은 $\hat\sigma$이 아님. README: "$\eta$=1 is one type of DDPM", 큰 분산은 `ddpm_noisy` | 논문 표 1의 $\eta=1$ 열과 $\hat\sigma$ 열이 서로 다른 코드 경로 |
| **D4** | 부록 D.3: $\hat\sigma_{\tau_i}=\sqrt{1-\bar\alpha_{\tau_i}/\bar\alpha_{\tau_{i-1}}}$, 마지막 스텝에도 잡음이 있다(식 그대로 $\bar\alpha_{\tau_0}=1$이면 $\sqrt{1-\bar\alpha_{\tau_1}}$) | `ddpm_steps`: `mask = 1 - (t == 0)`로 **t=0(논문 색인 1)의 스텝 잡음을 지움** | 코드의 uniform 격자는 $\tau_1=1$이라 그 스텝 잡음이 원래 0.01이어서 영향이 작다. 논문 선형 격자($\tau_1=T/S$)였다면 $S=10$에서 잡음 0.32가 마지막에 붙으므로 마스크 유무가 결정적(토이 SW1 0.149 대 0.067, `03` §2.1). 표 1의 $\hat\sigma$ 값이 어느 쪽으로 만들어졌는지 본문·코드만으로는 알 수 없음 |
| **D5** | $x_0$ 예측에 클리핑 언급 없음 | `ddpm_steps`는 `x0_from_e=clamp(x0_from_e,-1,1)`, `generalized_steps`는 클리핑 없음 | 표 1에서 $\hat\sigma$ 열만 클리핑 유무가 다른 설정이면 $\hat\sigma$ 대 $\eta$ 비교가 표집기 외의 요인과 섞임. 어느 쪽 값인지 알 수 없음 |
| **D6** | 표 2: $x_0\to x_T\to x_0$ 재구성 오차 | **저장소에 인코딩(역방향) 코드와 재구성 오차 계산이 없음**(`encode`, `invert`, `reconstruct` 검색 결과 없음). `sample_sequence`는 스텝마다 예측 $x_0$만 저장 | 표 2는 이 코드로 재현할 수 없고 인코딩 이산화(§2.5의 전진 오일러 선택)가 무엇인지도 코드에서 확인 안 됨 |
| **D7** | 표 1: FID | `sample_fid`는 `--fid`일 때 PNG(8비트, `save_image`가 [0,1] 클리핑) 50,000장을 저장할 뿐 **FID 계산 코드 없음**. FID 표본 수 50,000은 여기서만 확인됨 | FID 구현·참조 통계는 외부 도구 의존(어떤 구현인지 논문·코드에 없음) |
| **D8** | 부록 D.1: 학습은 $\gamma=1$ | 학습 루프는 **대조(antithetic) 시간 표집**(`t=cat([t, T-t-1])`)과 화소 합 손실(`sum(dim=(1,2,3)).mean(0)`) 사용 | 논문에 없는 분산 감소 기법. 사전학습 DDPM 체크포인트를 쓰는 표 1·3에는 영향이 없고 CelebA 학습에만 해당 |
| **D9** | 표 1: CelebA 64×64 포함, 직접 학습 | `configs/celeba.yml`은 있으나 `ckpt_util.py`에 CelebA 체크포인트 없음(cifar10, lsun_bedroom, lsun_cat, lsun_church만) | CelebA 열은 학습 없이는 재현 불가. `lsun_cat`은 논문에 없음 |
| **D10** | (환경) | `generalized_steps`·`ddpm_steps`가 `xt = xs[-1].to('cuda')`로 CUDA를 하드코딩, `Diffusion.__init__`은 CUDA 없으면 CPU로 폴백 | CPU 전용 환경에서는 그대로 실행되지 않음(이 분석은 실행하지 않았으므로 확인은 코드 읽기까지) |

## 3. 이후 문헌의 주장과 대조 (C)

**읽은 범위:** 아래 C 항목은 arXiv API로 받은 **초록 문장**만 근거입니다(본문 미확인). 수치는 초록에 적힌 값이고, 모델·데이터·평가가 달라 이 논문 표와 직접 비교할 수 없습니다.

| # | 주장·질문 | 출처 | 상태 | 비고 |
|---|---|---|---|---|
| C1 | 정리 1의 가중치 $\gamma_t=1/(2d\sigma_t^2\bar\alpha_t)$(식 35) | 이 저장소의 유도 (`03` §1.3) | **수정** | 올바른 값은 $\gamma_t=\tfrac{A_t^2(1-\bar\alpha_t)}{2\sigma_t^2\bar\alpha_t}$이고 $\sigma$가 DDPM 분산이면 Ho 등 식 12 가중과 일치. 정리의 결론은 유지. 후속 논문이 이를 다룬 것은 확인하지 않음 |
| C2 | DDIM은 고차 ODE 풀이기로 개선된다 | PNDM(arXiv:2202.09778): "DDIMs are simple cases of pseudo numerical methods", 50단계로 1000단계 DDIM보다 높은 품질(20배 가속). DPM-Solver(2206.00927): 10–20번 평가로 CIFAR10 FID 4.70(10), 2.87(20) | **갱신** | 표 1의 DDIM CIFAR10 FID 13.36(10), 6.84(20)은 이후 풀이기보다 열세. DDIM의 "스텝 수 기준선" 역할은 유효하지만 최선이 아님. 우리 토이의 $-0.86$ 기울기(1차 미만)와 §1.4의 점질량 정확성은 고차 풀이기의 동기와 방향이 같음 |
| C3 | 스텝을 줄이는 길은 결정적 표집뿐인가 | Improved DDPM(2102.09672): 분산을 학습하면 "order of magnitude fewer forward passes"에서 품질 손실이 미미. Analytic-DPM(2201.06503): 최적 역방향 분산의 해석식, 20–80배 가속 | **수정** | $\eta=0$ 대 $\eta=1$의 이분법이 아니라 **분산 선택이 변수**. 정확한 $\epsilon^*$ 토이에서도 $\hat\sigma$(마지막 잡음 제거)가 $S=20$–100에서 $\eta=0$보다 좋았음(`03` §2.1). 학습된 모델에서의 일반성은 확인하지 않음 |
| C4 | 설계 공간 | EDM(2206.00364): 설계 선택을 분리, 35번 평가로 CIFAR-10 FID 1.97(무조건) | **갱신** | 표집기 격자·전처리·확률성의 분리. 본문은 읽지 않아 "확률적 표집기" 세부는 확인하지 않음 |
| C5 | "결정적 표집기의 스텝 수 하한" | Progressive Distillation(2202.00512): 결정적 표집기를 절반 단계 모델로 증류. Consistency Models(2303.01469): 1단계 FID 3.55(CIFAR-10) | **갱신** | DDIM 10단계 FID 13.36보다 낮은 값이지만 학습이 필요하고 모델·평가가 달라 직접 비교 불가. DDIM의 "학습 없이" 장점의 대가를 보여 주는 대안 |
| C6 | ODE의 대응: 가우시안 경로의 흐름 매칭 | Flow Matching(2210.02747): "general family of Gaussian probability paths ... subsumes existing diffusion paths". 이 저장소 `../score_sde/` `03` §7이 PF-ODE 속도 $=\dot sE[x_0|x]+\dot\sigma E[\epsilon|x]$를 닫힌 식으로 확인 | **확인(연결)** | DDIM 식 14의 ODE는 같은 대상의 한 매개화. 그 분석은 "직선 경로가 스텝 효율이 좋다"는 직관이 정확한 점수 오일러 토이에서 확인되지 않았다고 보고했고 이 분석의 DDIM $\bar\sigma$-오일러도 20스텝에서 바닥에 닿지 않아(`03` §2.1, 바닥의 2.3배) 서로 다른 데이터라 직접 비교하지 않음 |
| C7 | 식 15(PF 오일러)는 DDIM과 다르다 | 논문 §4.3 | **확인** | 정확한 $\epsilon^*$에서 $S=10$에 DDIM이 10배 정확(`03` §2.4). $t$ 기준 오일러(Score SDE 설정)와는 비교하지 않음 |

## 4. 확인하지 못한 것

- 공식 코드가 논문 표 1·2·3의 수치를 낸다는 것(실행하지 않음, 체크포인트는 받지 않음, D6·D7·D9).
- 표 1의 $\hat\sigma$ 값이 `ddpm_noisy`(클리핑·마스크 포함)로 만든 것인지(D4, D5).
- C2–C6 문헌의 본문 수치(초록만 확인).
