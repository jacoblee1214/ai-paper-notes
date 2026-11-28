# 02. 계보, 가까운 대안, 구성 요소의 출처, 관련 분석

## 1. 계보 (이 논문이 어디에 서 있나)

```
Sohl-Dickstein 2015 확산 확률 모형 ──┐
Vincent 2011 잡음 제거 점수 매칭 ────┤
Song & Ermon 2019 NCSN ──────────────┼─→ Ho 2020 DDPM (마르코프 순방향, 1000단계 조상 표집, L_simple)
Mohamed & Lakshminarayanan 2016 ─────┘                │
 (암묵 확률 모델)                                      ▼
                    이 논문 (Song, Meng, Ermon 2021, ICLR 2021 포스터)
                    비마르코프 순방향 족 q_sigma, 같은 학습(L_1)으로 결정적 표집(DDIM), 가속 부분 수열, 인코딩
                                           │  (병행: Song 2021 Score SDE의 확률 흐름 ODE)
      ┌──────────────┬────────────────┬────┴───────────────┬──────────────────────┐
      ▼              ▼                ▼                    ▼                      ▼
 Improved DDPM   PNDM, DPM-Solver,  Analytic-DPM        Progressive           Flow Matching,
 (학습된 분산)   DEIS, EDM          (최적 분산 해석)    Distillation,         Rectified Flow
 Nichol 2021     (고차 풀이기)      Bao 2022            Consistency Models    ──→ SD3, MeanFlow
                 Liu/Lu/Karras 2022                     (스텝 증류)
```

- **이 논문이 새로 한 것.** (1) DDPM 목적이 주변분포 $q(x_t|x_0)$에만 의존한다는 관찰로 같은 학습 목적을 공유하는 비마르코프 순방향 과정의 족 $q_\sigma$를 만들고(식 6–7, 정리 1), (2) 그 족의 생성 과정에서 $\sigma=0$이면 표본이 $x_T$의 결정적 함수가 되는 암묵 확률 모델(DDIM)을 얻고(식 12), (3) 길이 $S<T$의 부분 수열에서 같은 모델로 표집해 스텝 수를 줄이고(§4.2, 부록 C.1), (4) 이를 재매개화한 ODE의 오일러 스텝으로 해석해 인코딩·보간·재구성을 얻었습니다(§4.3, §5.3–5.4). 학습 목적, 모델 구조, 스케줄은 DDPM 것을 그대로 썼습니다(부록 D.1).
- **병행 연구.** Song 등(Score SDE, arXiv:2011.13456)은 같은 해 확률 흐름 ODE를 포커-플랑크 방정식에서 유도했고, 이 논문은 비마르코프 과정에서 얻었습니다. 명제 1은 두 ODE가 같다는 것이고, 식 15는 이산화 방식에 따라 스텝이 적을 때 다르다고 합니다. 유도는 `../score_sde/`(`analysis/03` §2)를 인용하고 여기서 다시 하지 않았습니다. 이 분석은 두 이산화(식 13 대 식 15)의 차이를 정확한 $\epsilon^*$에서 수치로 비교했습니다(`03` §2.4).
- **이후 논문이 바꾼 것.** 요약은 아래 §3과 `04` C1–C8.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 이 논문의 변경 |
|---|---|---|
| 잡음 예측 목적 $L_\gamma$, 선형 $\beta$ 스케줄 $10^{-4}\to0.02$, $T=1000$, U-Net | Ho 2020 (DDPM) | 변경 없음. 사전학습 체크포인트를 그대로 사용(CelebA만 직접 학습) |
| 조상 표집과 사후 분산 $\tilde\beta_t$, 더 큰 분산 $\beta_t$ | Ho 2020 (식 7, §3.2) | $\eta=1$이 $\tilde\beta_t$, $\hat\sigma$이 $\beta_t$에 대응(식 16, 부록 D.3). `../ddpm/`이 이 분산 선택을 정확한 $\epsilon^*$에서 비교 |
| 암묵 확률 모델 | Mohamed & Lakshminarayanan 2016 | 표본이 잠재 변수의 결정적 함수인 확산 모델로 전환 |
| 구면 선형 보간(slerp) | Shoemake 1985 | $x_T$에 적용(부록 D.5 식 67) |
| 신경 ODE·정규화 흐름식 인코딩 | Chen 2018, Dinh 2016 | DDIM의 ODE로 $x_0\to x_T$(§5.4) |
| 확률 흐름 ODE | Song 2021 (병행) | 명제 1에서 동치 증명(부록 B) |
| 다단계 ODE 풀이기 | Butcher & Goodwin 2008 | 미래 과제로만 언급(§6) |

## 3. 가까운 대안 비교

| 방법 | 같은 점 | 다른 점 |
|---|---|---|
| DDPM (조상 표집, $\eta=1$) | 같은 학습과 모델 | 마르코프 역과정, $S$단계로 줄이면 스텝당 Gaussian 근사가 나빠져 품질 급락(표 1) |
| Score SDE의 확률 흐름 ODE | 같은 ODE(명제 1) | $dt$ 또는 $d\bar\sigma^2$로 이산화(식 15), 적응 풀이기(RK45) 사용 |
| Improved DDPM (Nichol & Dhariwal, arXiv:2102.09672) | 스텝 수 감축 | 분산을 학습해 확률적 표집으로 10배 적은 스텝을 달성(초록). DDIM은 학습 없이 $\sigma$ 선택 |
| PNDM (Liu 등, arXiv:2202.09778) | DDIM을 특수한 경우로 포함 | 다단계(선형 다스텝) 풀이기, 50단계로 1000단계 DDIM보다 좋은 FID(초록) |
| DPM-Solver (Lu 등, arXiv:2206.00927) | 확산 ODE 풀이 | 선형 항을 해석적으로 풀고 고차 풀이기, 10–20 평가로 고품질(초록) |
| EDM (Karras 등, arXiv:2206.00364) | 같은 ODE 틀 | 설계 공간 분리, 확률적 표집기와 2차 풀이기, CIFAR-10 35번 평가 FID 1.97(초록) |
| Analytic-DPM (Bao 등, arXiv:2201.06503) | 훈련 없이 $\sigma$를 고름 | 최적 역방향 분산의 해석식, 20–80배 가속(초록) |
| Progressive Distillation (Salimans & Ho, arXiv:2202.00512) | 결정적 표집기가 출발점 | 결정적 표집기를 절반 단계 모델로 증류 |
| Consistency Models (Song 등, arXiv:2303.01469) | PF-ODE 궤적을 따라 같은 해로 가는 사상 | 한 번에 $x_T\to x_0$를 학습, CIFAR-10 1단계 FID 3.55(초록) |
| Flow Matching (Lipman 등, arXiv:2210.02747) | 가우시안 경로의 결정적 ODE | 점수 대신 속도를 회귀(`../score_sde/` `03` §7이 대응 관계를 닫힌 식으로 확인) |

## 4. 이 저장소와 이웃 저장소의 관련 분석

| 분석 | 관계 |
|---|---|
| `../ddpm/` | 같은 묶음에서 작성. 이 논문이 가정하는 DDPM의 식 (4)–(14), 사후 평균·분산(식 7), $\epsilon^*$의 닫힌 식을 거기서 검산. 이 분석은 그 $\epsilon^*$ 틀을 이어 씀 |
| `../score_sde/` | 확률 흐름 ODE의 유도, VP 이산화의 $1/N$ 수렴, 속도-점수 변환. 명제 1의 유도를 거기서 인용 |
| `../sd3_rectified_flow/` | 직선 경로의 ODE 표집. DDIM은 분산 보존 경로 위의 결정적 표집이고 스텝 수 절감의 기준선 |
| `../meanflow/` | 평균 속도로 한 스텝 표집. DDIM의 "스텝 줄이기"를 학습으로 해결한 쪽 |
| `../classifier_free_guidance/` | 가이던스 설정과 표집기(DDIM)가 함께 쓰이는 방식 |


## 5. 자료 표

| 자료 | 내용 | 라이선스 | 크기·별·마지막 push | 확인 |
|---|---|---|---|---|
| arXiv:2010.02502 | 논문 PDF v4 (22쪽, 10.9 MB) | arXiv 비독점 배포 라이선스(CC BY 아님) | v4 2022-10-05 | PDF 받아 `pdftotext` 변환, 회사 저작권 고지 없음 |
| github.com/ermongroup/ddim | 저자 공식 PyTorch 구현(PyTorch 1.6에서 시험) | MIT | 저장소 16 KB(API) · clone 280 KB · 별 1,859 · 포크 228 · 마지막 push 2024-07-26 | 얕은 clone, 커밋 `e8df66c`, 정적으로만 읽음 |
| 사전학습 체크포인트 | `functions/ckpt_util.py`가 가리키는 heibox 주소(DDPM의 TF 체크포인트를 PyTorch로 옮긴 것, pesser/pytorch_diffusion) | 확인 안 함 | 크기 확인 안 함 | 받지 않음 |
| iclr.cc 일정 JSON | `iclr-2021-orals-posters.json`(4.5 MB) | 공개 데이터 | - | `decision: Poster`, "Poster Session 06" 확인 |
| huggingface diffusers `DDIMScheduler` | 저장소 README가 안내하는 이식 | 확인 안 함 | - | 열어 보지 않음 |
