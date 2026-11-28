# 04. 논문 대 공식 코드 대조

대상: `Gsunshine/meanflow` (JAX, 저자 공식, MIT, `repo/`에 `--depth 1` clone, 커밋 `d70cb55d298ee03c53bf6da67bec281082e4e2d9`, 2026-03-13). **정적으로만 읽었고 실행·설치·체크포인트 다운로드는 하지 않았습니다.** 핵심 파일: `meanflow.py` (258줄), `train.py`, `models/models_dit.py`, `configs/default.py`, `configs/run_b4.yml`. README는 "written and tested on TPUs"라고 하고, PyTorch 공식 포트 `Gsunshine/py-meanflow`(CIFAR-10)는 읽지 않았습니다(`02` §4). 공식 저장소가 제공하는 설정은 MF-B/4 하나뿐입니다.

## 1. 일치하는 것

| # | 논문 | 코드 | 판정 |
|---|---|---|---|
| K1 | 식 11 목표 $v_t-(t-r)\,\mathrm{JVP}$, 접선 $[v,0,1]$ | `meanflow.py` `forward`: `jax.jvp(u_fn, (z_t, t, r), (v_g, 1, 0))`, `u_tgt = v_g - clip(t-r,0,1) * du_dt` | 일치. 인자 순서가 $(z,t,r)$로 논문의 $(z,r,t)$와 다르지만 접선 $(v_g,1,0)$이 같은 의미(아래 D7) |
| K2 | stop-gradient (식 9) | `u_tgt = jax.lax.stop_gradient(u_tgt)` | 일치 |
| K3 | 경로 $z_t=(1-t)x+t\epsilon$, $v=\epsilon-x$ | `z_t = (1 - t) * x + t * e`, `v = e - x` | 일치 |
| K4 | 샘플링 식 12, 1-NFE $z_0=z_1-u(z_1,0,1)$ | `solver_step`: `z_t - (t - r) * u`, `_default_schedule`이 `[1.0, 0.0]` | 일치 |
| K5 | 식 19 ($\kappa=0$): $\tilde v=\omega v_t+(1-\omega)u_\theta(z_t,t,t)$ | `guidance_fn`, `kappa == 0`: `v_uncond + omega*(v_t - v_uncond)`, `v_uncond = v_fn(z,t,null)` (`h=0`) | 일치 |
| K6 | 식 21 ($\kappa>0$) | `v_g = omega*v_t + (1-omega-kappa)*v_uncond + kappa*v_cond` | 일치 |
| K7 | 표 4 "CFG triggered if t in [a,b]" | `t_start`, `t_end` 안에서만 $\omega$, $\kappa$ 적용, 밖은 $\omega=1$, $\kappa=0$ | 일치 |
| K8 | 표 집기: 두 값 독립 추출 후 큰 쪽 $t$, 작은 쪽 $r$ | `sample_tr`: `jnp.maximum`, `jnp.minimum`, 로짓 정규 `sigmoid(N*P_std+P_mean)`, 기본 $-0.4$, $1.0$ | 일치 |
| K9 | 최적화(표 4): Adam(0.9,0.95), lr 1e-4 상수, wd 0, 배치 256, EMA 0.9999 | `optax.adamw(lr, weight_decay=0, b2=0.95)`, `run_b4.yml` lr 0.0001, 배치 256, `ema_val` 0.9999 상수, 평가·FID는 EMA 파라미터 | 일치 |
| K10 | 시간 조건 $(t,t-r)$, 각각 위치 임베딩 후 MLP, 합 (부록 A) | `DiT.__call__`: `t_embedder(t)`, `h_embedder(h)`, `y_embedder(y)`를 더해 adaLN 조건 `c = t + h + y`, `u_fn(z,t,r)`가 `net(z, t, t-r)` | 일치 |
| K11 | 구조 DiT (표 4: B/4 12층·768, M/2 16층·1024, L/2 24층·1024, XL/2 28층·1152) | `DiT_B_4` 등 12/768, 16/1024, 24/1024, 28/1152 | 일치. 파라미터·FLOPs 산술은 `03` §6.2 |

## 2. 논문과 다른 점, 논문에 없는 점 (D)

**D1. $r=t$ 표본이 "배치 앞 75%"로 고정됩니다.** `zero_mask = arange(bz) < int(bz*0.75)`로 앞쪽 인덱스의 $r$을 $t$로 덮습니다(독립 베르누이가 아님). 표본이 독립 동분포로 섞여 들어오면 분포상 같고 DistributedSampler로 섞습니다. 변수 이름은 `data_proportion`(=r=t 비율)이라 값 0.75는 "r≠t 25%"를 뜻합니다. ImageNet 설정에서는 논문(표 4)과 일치, CIFAR-10(논문 부록: 비율 75% = $r\ne t$)을 재현하려면 값을 0.25로 바꿔야 합니다. 영향 없음, 혼동 주의.

**D2. 적응 가중의 상수와 합산 방식.** 코드는 `loss = sum_{pixels}(u - u_tgt)^2`, `adp_wt = (loss + norm_eps)^norm_p`, `loss / sg(adp_wt)`, 기본 `norm_eps=0.01`, `norm_p=1.0`. 논문 식 22와 형태는 같고($\|\Delta\|^2$는 픽셀 전체 합) $p=1$은 표 4와 일치하지만, 상수 $c$는 논문이 "e.g., $10^{-3}$"로만 쓰고 표 4에 값이 없는데 코드는 0.01입니다. 큰 $\|\Delta\|^2$(차원 4096의 합)에서는 영향이 작지만 목표가 잘 맞은 구간(오차가 0.01 아래)에서는 가중이 $1/c$ 쪽으로 포화합니다. $p$에 대한 민감도는 표 1e만 있고 $c$에 대한 것은 없습니다.

**D3. 클래스 조건을 버린 표본은 항상 $r=t$입니다 (논문에 없음).** `sample_tr`의 $r=t$ 마스크와 `cond_drop`의 라벨 제거 마스크가 둘 다 배치 앞쪽 인덱스(`arange(bz) < k`)를 씁니다. 제거 개수 `num_drop`(이항, 평균 25.6)은 $r=t$ 구간 192보다 거의 항상 작아서($P[\mathrm{num\_drop}>192]=8.6\times10^{-136}$), 제거된 표본은 모두 $r=t$입니다. 20,000배치를 시뮬레이션해(`code_check.py`, 시드 0) 제거 표본 중 $r\ne t$는 **0개**, 클래스 조건 표본의 $r\ne t$ 비율은 27.8%(논문 서술 25%, 이론값 64/230.4=27.8%)였습니다. 귀결: (i) 무조건부(null 클래스) 평균 속도 $u(z,r,t|\varnothing)$, $r\ne t$는 직접 학습되지 않고, (ii) 식 16·19는 $u(z_t,t,t|\varnothing)$($r=t$)만 쓰고 샘플링은 클래스 조건 모델만 쓰므로 논문의 파이프라인에는 영향이 없어 보입니다. 다만 논문은 "10% 확률로 버림"이라고만 쓰므로 독립 가정과 다르다는 점은 기록할 가치가 있습니다.

**D4. 버려진 표본의 목표는 $\omega=1$ 목표입니다 (논문에 서술 없음).** `cond_drop`이 버려진 표본에서 `v_g = v_t`로 되돌립니다. 무조건부 장은 $\mathbb E_{c|z}[v^{cfg}(z,t|c)]=v(z,t)$라서(`03` §5) 이론과 일관되지만 식 17–19에는 없는 분기입니다.

**D5. CFG 목표 계산이 학습 스텝마다 순전파를 더합니다.** `guidance_fn`은 `kappa == 0`이면 무조건부 순전파 1번, `kappa > 0`이면 무조건부와 조건부 2번을 현재(EMA가 아닌) 파라미터로 돌립니다. 이 호출은 JVP와 별개입니다. 논문 부록 B.4의 오버헤드 16%(B/4 Flow Matching 대비)에는 이 비용이 없습니다. 순전파 1 : 역전파 2 단위로 보면 $\kappa>0$ 학습은 기본 스텝 대비 약 +67%(+2 단위/3 단위) 더 들 것으로 추정합니다. 측정한 값이 아니며 큰 모델·시퀀스에서 달라질 수 있습니다.

**D6. 공유된 설정이 논문의 절제 기본값과 다릅니다.** `default.py`의 방법 기본값은 $\omega=1.0$, $\kappa=0.5$(유효 안내 $\omega'=2.0$)로 CFG가 이미 켜져 있고, `run_b4.yml`은 `num_epochs: 240`입니다. 논문 표 1의 B/4 기본값은 CFG 없음(FID 61.06, 80 에폭)이므로 같은 모델 이름의 설정이 아닙니다. 표 4의 XL/2($\omega=0.2$, $\kappa=1-\omega/\omega'=0.92$, 구간 [0,0.75])나 B/2($\omega=1.0$, $\kappa=0.5$) 같은 다른 크기의 설정 파일은 없어서 큰 모델 결과는 저장소만으로 재현 설정을 알 수 없습니다. README의 MF-B/4 체크포인트 평가 FID 11.4는 논문에 없는 수치이고 학습 설정이 기록되어 있지 않습니다.

**D7. JVP 인자 순서와 접선.** 코드는 $(z,t,r)$ 순서이고 접선이 $(v_g,1,0)$, 논문은 $(z,r,t)$에 $(v,0,1)$입니다. 같은 JVP지만, 네트워크가 $(t, h=t-r)$을 받으므로 $h$의 접선은 $1-0=1$입니다. 표 1b의 변형 이름(예: "(v,1,0)")은 논문의 $(r,t)$ 순서 기준이므로, 코드 인자 순서에 그대로 옮기면 다른 변형을 시험하게 됩니다.

**D8. 시간 입력 스케일과 잠재 처리의 세부.** 시간 임베딩이 $[0,1]$의 $t$와 $h$를 그대로 sin/cos에 넣습니다(DiT 원형처럼 1000배 하지 않음, `max_period=10000`). 논문은 "positional embedding"이라고만 씁니다. 잠재는 캐시한 VAE 사후 분포(평균, 표준편차)에서 **매 스텝 표집**하고 0.18215를 곱합니다(`vae_util.py`). 논문은 "VAE 토크나이저가 32×32×4를 낸다"고만 씁니다. `input_pipeline.py`는 `use_flip=True`로 잠재 데이터셋을 만들지만 `LatentDataset.__getitem__`은 이 플래그를 쓰지 않아 뒤집기 증강은 실제로는 적용되지 않는 것으로 읽힙니다(논문은 ImageNet 증강을 쓰지 않았다고 하지 않음).

**D9. 2-NFE 샘플링의 중간 시각이 기본 설정에 없습니다.** 기본 일정은 `[1.0, 0.0]` 하나이고 `sampling_timesteps`를 따로 주어야 합니다. 논문 표 2의 2-NFE(2.93, 2.20)가 어떤 중간 시각을 썼는지는 논문에도 코드에도 없습니다.

**D10. FID 구현.** `utils/jax_fid`의 JAX Inception 구현과 ADM 참조 통계(README: "followed the ADM repo")를 씁니다. 평가 표본의 라벨은 균등 난수(`randint`)입니다. 이 구현의 수치가 비교 대상 표(DiT, SiT, ADM 등 TF·PyTorch 구현)와 같은 척도인지는 이 분석에서 확인하지 못했습니다(구현 사이 차이의 크기는 미측정).

**D11. EMA 초기화.** `ema_params = update_ema(copy, params, 0)`으로 초기 EMA를 파라미터와 같게 두고 상수 0.9999로 갱신합니다(웜업 없음). 표 4의 EMA 0.9999와 일치하지만 초기 구간에서 EMA는 초기화 값에 오래 끌립니다. 논문에는 평가가 EMA인지 쓰여 있지 않습니다(코드는 EMA).

## 3. 결론

- 핵심 식(JVP 목표, stop-gradient, 샘플링, CFG의 두 목표)은 코드와 논문이 일치합니다(K1–K11).
- 논문에 없는 구현 선택은 D3(라벨 제거와 $r=t$ 마스크의 겹침), D5(학습 비용), D6(설정 기본값), D9(2-NFE 일정)가 해석상 가장 큽니다.
- 논문 수치는 이 저장소의 설정 하나(B/4)로만 재현을 시도할 수 있고, XL/2 3.43은 설정 파일이 없습니다. 실행해 보지 않았으므로 코드가 논문 수치를 낸다는 것은 확인되지 않았습니다.
