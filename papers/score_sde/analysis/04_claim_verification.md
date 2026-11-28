# 04. 주장 검증(C)과 코드 대조(K, D)

이 논문은 공식 코드가 있어서 §2에서 논문 ↔ 코드를 대조하고(K: 일치, D: 차이·주의), §1에서는 이후 문헌이 논문의 주장을 확인·수정·확장한 곳(C)을 적습니다.

- **후속 논문은 arXiv 초록(2026-11-28 arXiv API)으로만 확인했고 본문은 읽지 않았습니다.** 아래 수치는 초록에 적힌 값입니다. 본문 확인이 필요한 항목에는 "초록 기준"이라고 적었습니다.
- 코드: `yang-song/score_sde_pytorch` 커밋 `cb1f359`(2022-11-10)를 `--depth 1`로 clone해 정적으로만 읽었습니다. 실행하지 않았습니다. JAX 원본(`yang-song/score_sde`)은 읽지 않았습니다.

## 1. 이후 문헌의 확인·수정·확장 (C1–C7)

| # | 이 논문의 주장 | 이후 문헌 | 이 저장소의 검증 | 판정 |
|---|---|---|---|---|
| C1 | 연속 목적(식 7, $\lambda\propto\sigma^2$)로 학습하고 확률 흐름 ODE로 정확한 우도를 구해 2.99 bits/dim, "최대우도 학습 없이도" 최고 기록 (§4.4, 표 2) | Song et al. 2021b (arXiv:2101.09258, NeurIPS 2021 스포트라이트): 특정 가중 $\lambda=g^2$에서 목적이 NLL의 상한이 되고 최대우도 학습이 우도를 꾸준히 개선, CIFAR-10 2.83, ImageNet 32 3.76 bits/dim (초록 기준) | 손실 가중의 동치 관계를 유도(`03` §7.3): flow matching은 $\lambda=g^4/4$, 논문의 권장은 $\sigma^2$, ELBO는 $g^2$ | **갱신.** 2.99는 우도 기록이 아니게 됨. "가중치가 우도를 정한다"는 방향이 확인됨 |
| C2 | VP 대비 sub-VP가 같은 구조에서 더 나은 NLL (표 2, 세 쌍 -0.14–-0.16) | Kingma et al. 2021 VDM (arXiv:2107.00630, NeurIPS 2021): 연속 시간 VLB는 잡음 스케줄에 불변이고 SNR의 양 끝점만 영향을 줌 (초록 기준). 우도 최적화는 스케줄 학습과 함께 | 논문 표에서 세 쌍 부호 검정 p=0.25, 시드 반복 없음(`03` §5.2) | **조건부.** VLB의 스케줄 불변성은 ELBO에 대한 것이고 ODE 우도는 별개라 정면 충돌은 아님(추정). sub-VP 우위가 학습 잡음 안인지는 미확인 |
| C3 | 확률 흐름 샘플은 SDE 풀이기보다 FID가 나쁘고(부록 D.4) PC 표집기가 최선(표 3, FID 2.20, NFE 2000) | Karras et al. 2022 EDM (arXiv:2206.00364, NeurIPS 2022): 설계 공간을 분리해 표집·학습·사전조건을 바꾸고 35 NFE에서 CIFAR-10 FID 1.79(조건부)/1.97(무조건) (초록 기준) | 같은 NFE 비교에서 정확한 점수라면 예측기 선택이 표본 잡음 안(`03` §4.2). 확률 흐름 오일러 오차는 SDE·격자에 크게 의존(`03` §7.5) | **수정.** 표 3의 2.20은 최고가 아니고, "ODE가 SDE보다 나쁘다"는 풀이기·격자 선택에 달렸습니다. 조건 없이 인용하지 말 것 |
| C4 | 확률 흐름 ODE를 RK45(atol=rtol=1e-5)로 풀면 NFE를 줄일 수 있음 (그림 3, 부록 D.4) | Lu et al. 2022 DPM-Solver (arXiv:2206.00927, NeurIPS 2022): 선형 부분을 해석적으로 풀고 지수 가중 적분으로 단순화, 10 NFE FID 4.70, 20 NFE 2.87 (초록 기준). Zhang & Chen 2022 DEIS (arXiv:2204.13902, arXiv 코멘트상 ICLR 카메라레디): 반선형 구조의 지수 적분기, 10 NFE FID 4.17 (초록 기준) | 오일러 NFE 5–200에서 VP·sub-VP는 NFE 20에서 표본 잡음 바닥, VE는 200에서도 바닥의 약 2배(`03` §7.5) | **확장.** 방향 확인(스케줄이 이산화 오차를 정함) |
| C5 | PF-ODE는 SDE와 같은 주변분포를 가짐 (식 13, 부록 D.1) | Lipman et al. 2022 Flow Matching (arXiv:2210.02747): 가우시안 확률 경로의 일반 족이 기존 확산 경로를 특수한 경우로 포함, 확산 경로에 FM을 쓰면 더 안정적인 학습 (초록 기준) | PF-ODE 속도 $=\dot sE[x_0|x]+\dot\sigma E[\epsilon|x]$, VE, VP, sub-VP, 직선 경로에서 닫힌 식 혼합으로 1e-8 이내 확인. 손실은 점수 매칭의 가중치 $g^4/4$ 판(`03` §7.1–7.3) | **확인(수식).** "FM이 더 효율적"이라는 우위 주장은 이 토이에서 확인되지 않음(균일 격자 오일러에서는 VP가 직선 경로보다 스텝 효율이 좋음, `03` §7.5) |
| C6 | 결정적 인코딩, 코드가 유일하게 정해짐 (부록 D.5) | Liu et al. 2022 Rectified Flow (arXiv:2209.03003): 직선 경로로 π₀, π₁ 사이를 수송하는 ODE를 학습하고 재정류로 더 곧은 경로, 거친 이산화로 시뮬 가능 (초록 기준). Albergo & Vanden-Eijnden 2022 (arXiv:2209.15571, ICLR 2023): 가우시안 base에서 속도로 확산 모델의 점수를 구성 가능, SDE 없이 ODE 수준에서 작업 (초록 기준) | 정확한 점수에서 인코딩은 결정적이고 가역(왕복 오차 5e-9). 점수와 속도의 변환식을 닫힌 식으로 확인(`03` §2.3, §7.1) | **확장.** 인코딩의 유일성은 정확한 점수에서는 정의상, 학습된 모델에서는 실험적(그림 7–8, 이 저장소 미검증) |
| C7 | 인페인팅·컬러화는 식 49–50의 근사로 되고(§5, 부록 I) 한 모델로 여러 역문제를 푼다 | (이 저장소가 후속 문헌을 확인하지 않음) | 다봉 토이에서 "알려진 부분을 섭동 관측값으로 대체"하는 근사가 정확한 조건부와 크게 어긋남: $c=0$에서 W1 2.47, 정확한 조건부 점수 대조군은 0.011(`03` §6) | **이 저장소 발견, 문헌 확인 필요.** 근사의 오차 조건을 후속 문헌에서 찾아야 함 |

- C1–C6의 "초록 기준" 열은 논문 본문의 수치·조건을 확인하지 않았습니다. 인용할 때 해당 논문 본문에서 다시 확인하세요.
- DDIM(Song et al., arXiv:2010.02502, ICLR 2021)은 같은 시기 논문이고 확률 흐름의 이산화로 볼 수 있다는 연결은 DDIM 분석(`../ddim/`)에서 다뤘습니다. 그쪽에서 DDIM(η=0)이 재매개화 ODE의 오일러 스텝과 스텝별 1e-13에서 같음을 확인했습니다.

## 2. 코드 대조 (`score_sde_pytorch`, 커밋 `cb1f359`, 정적 읽기)

### 2.1 논문과 일치 (K)

| # | 항목 | 논문 | 코드 |
|---|---|---|---|
| K1 | **신호 대 잡음비 스텝 규칙** | 알고리즘 4–5: $\epsilon=2\alpha(r\|z\|/\|s\|)^2$, 배치 평균 노름 | `sampling.py` `LangevinCorrector.update_fn`: `step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha`, 노름은 `.mean()`, `x_mean = x + step_size*grad; x = x_mean + sqrt(step_size*2)*noise`. $\alpha$는 VP·sub-VP에서 이산 `alphas[timestep]`, VE에서 1. 우리 구현(`minimal_impl.corrector_step`)과 수식 일치 |
| K2 | **우도 적분의 $\epsilon$과 시간 절단** | 학습·우도 $\epsilon=10^{-5}$, 표집 VP $10^{-3}$, VE $10^{-5}$ (부록 C, D.2, D.4) | `likelihood.py` `get_likelihood_fn(..., eps=1e-5)`. `run_lib.py`(90–98, 208–216)의 `sampling_eps`: VP·sub-VP 1e-3, VE 1e-5. `losses.py` `get_sde_loss_fn(eps=1e-5)` |
| K3 | 확률 흐름 풀이기 | scipy RK45, atol=rtol=1e-5 (부록 D.2, D.4) | `sampling.get_ode_sampler`, `likelihood.get_likelihood_fn` 기본값 `rtol=1e-5, atol=1e-5, method='RK45'` |
| K4 | 허친슨 추정의 프로브 분포 | 식 40: $E[\epsilon]=0$, $\mathrm{Cov}=I$ 이면 됨 (분포는 지정하지 않음) | 기본값 `hutchinson_type='Rademacher'`. 우리 토이에서 같은 평균 오차에 가우시안 프로브의 분산이 3.6배(VP), 42배(VE) 큼(`03` §3.2) |
| K5 | sub-VP 핵과 확산 | 식 12, 34: 표준편차 $1-e^{-B}$, 확산 $\sqrt{\beta(1-e^{-2B})}$ | `sde_lib.subVPSDE.marginal_prob`: `std = 1 - exp(2*log_mean_coeff)`, `sde`: `discount = 1 - exp(-2*beta_0*t - (beta_1-beta_0)*t**2)`, `diffusion = sqrt(beta_t*discount)` |
| K6 | Table 2의 NLL 설정 | "최대우도 학습 없이" 2.99 (§4.4) | `configs/subvp/cifar10_ddpmpp_deep_continuous.py`: `continuous=True`, `reduce_mean=True`, `likelihood_weighting`은 기본값 False(`default_cifar10_configs.py`). 평가는 `run_lib.py` 233–235에서 `uniform_dequantization=True` |
| K7 | 논문의 VE 최고 모델 설정 | NCSN++ cont. (deep, VE), PC 표집 (표 3) | `configs/ve/cifar10_ncsnpp_deep_continuous.py`: `predictor='reverse_diffusion'`, `corrector='langevin'`, `snr=0.16`(기본), `n_steps_each=1` |
| K8 | 우도 단위 변환 | 비트/차원, 균일 역양자화 데이터 | `likelihood.py`: `bpd = bpd + (7. - inverse_scaler(-1.))`. 데이터가 $[-1,1]$이면 야코비안 $2^{-d}$를 8비트로 보정해 +7, $[0,1]$이면 +8. 산술로 맞음, 실행하지 않음 |

### 2.2 차이·주의 (D)

| # | 논문 | 코드 | 영향 |
|---|---|---|---|
| D1 | **VE 핵의 분산 규약.** 식 29: $\sigma^2(t)-\sigma^2(0)$, 식 31: $\sigma_{\min}^2(\cdot)^{2t}$ | `sde_lib.VESDE.marginal_prob`: `std = sigma_min * (sigma_max/sigma_min)**t` (식 31) | 분산 차이 $\sigma_{\min}^2=10^{-4}$. 논문 안의 두 식이 이 만큼 다르고(`03` §1.1) 코드는 후자. 학습·표집에는 사실상 무영향, 엄밀한 우도의 $t=\epsilon$ 정의에는 해당 |
| D2 | **PC 루프 순서.** 알고리즘 2–3: 예측기로 $x_i$를 만든 뒤 수준 $i$에서 교정 | `sampling.get_pc_sampler`: 매 반복에서 `corrector_update_fn` → `predictor_update_fn` 순서 (같은 $t$에서 교정이 먼저) | 첫 반복에서 사전 표본 $x_T$를 $t=T$에서 교정하고, 마지막 예측 뒤 $t=\epsilon$ 수준 교정은 없음. 교정과 예측 호출 횟수(NFE=$N(n_{\rm steps}+1)$)는 같지만 알고리즘 의사코드와 순서가 다름 |
| D3 | 균일 시간 격자와 오일러-마루야마 | `timesteps = torch.linspace(sde.T, eps, sde.N)`(간격 $(T-\epsilon)/(N-1)$)인데 `EulerMaruyamaPredictor`는 `dt = -1/self.rsde.N` | 점수가 평가되는 시각과 스텝 크기가 어긋남: $N\epsilon=1$이면 정확히 일치(VP·sub-VP: $N=1000,\epsilon=10^{-3}$), VE는 $N\epsilon=0.01$이라 마지막 구간에서 시각 오차가 최대 약 $10^{-3}$($\sigma$ 비 약 0.9%). 산술로 계산했고 실행하지 않음 |
| D4 | **마지막 잡음 제거.** 부록 G: "모든 실험에서 한 번의 트위디 잡음 제거" | `denoise=True`면 마지막 예측기의 `x_mean`을 반환. VE 역확산 예측기는 $t=\epsilon$에서 `adjacent_sigma=0`이라 `x + sigma_min^2 * score`이고 이는 정확히 트위디. VP·sub-VP의 오일러-마루야마 예측기(§4.4가 연속 모델에 사용)는 `x + drift*dt`의 평균 스텝 | VP: 평균 스텝의 점수 계수 $\beta(\epsilon)/N\approx1.2\times10^{-4}$, 트위디 계수 $1-e^{-B(\epsilon)}\approx1.1\times10^{-4}$로 같은 크기이지만 정확히 트위디는 아님(산술 계산) |
| D5 | 표 2의 FID는 블랙박스 ODE 표집(부록 D.4) | `configs/` 아래 모든 설정 파일이 `sampling.method = 'pc'`이고 `'ode'`를 쓰는 설정은 없음(grep). `sampling.get_sampling_fn`에는 `'ode'` 분기가 있음 | 표 2의 FID(ODE) 수치를 내려면 명령줄 설정 덮어쓰기가 필요하고, 저장소의 설정 파일만으로는 그 수치를 만드는 구성이 드러나지 않음(실행하지 않음) |

- D1–D4는 모두 정적 읽기와 산술이고 코드를 실행하지 않았습니다.
- 공식 체크포인트(Google Drive, 크기 미확인)를 받아 표 2·3을 재현하는 일은 이 저장소 규칙상 승인 없이 하지 않습니다.

## 3. 인용할 때 붙일 단서 정리

1. 2.99 bits/dim은 "최대우도 학습 없이"의 값이고 이후 최대우도 학습이 더 낮춥니다(C1).
2. 초록의 IS 9.89·FID 2.20(VE)과 2.99(sub-VP)는 다른 두 모델의 값입니다(`03` §5.2).
3. 표 1·2·3은 학습 시드 반복이 없고, 표 3은 FID 최소 체크포인트입니다. VP의 DDPM형 예측기에서 PC의 이득은 표 1 안에서 5회 표집 잡음 안입니다(`03` §5.1).
4. 확률 흐름의 샘플 품질은 풀이기와 격자에 달렸습니다(C3, C4).
5. 인페인팅 근사는 날카로운 다봉 조건부에서 틀릴 수 있습니다(C7, `03` §6).
