# 04. 논문 ↔ 코드 대조 (정적 읽기)

**원칙.** 코드는 실행하지 않았고 가중치도 받지 않았습니다. `repo/`에 얕게 clone한 것은 `Stability-AI/sd3.5` 하나(HEAD `8565799a3b41eb0c7ba976d18375f0f753f56402`, 2025-01-08, MIT, 총 3,285줄)이고, 학습 쪽은 `huggingface/diffusers`(Apache-2.0, main `509b7658e8d99a811050f26ce4ed0a40b65ccfee`)의 세 파일을 `gh api`로 읽어만 보았습니다(clone 없음). 줄 번호는 위 판 기준입니다. 산술이 필요한 곳은 `code_checks.py`(수치만 다시 계산, 저장소를 import하지 않음)로 확인했습니다.

**가장 중요한 사실.** 공식 저장소 `sd3.5`는 README 첫 줄에 "Inference-only tiny reference implementation"이라고 적혀 있습니다. 논문 §3.1(시간 표본기, 손실 가중)과 §5(학습 설정, EMA, DPO)에 해당하는 **학습 코드는 이 저장소에 없습니다.** 따라서 논문의 핵심 기여 중 시간 표본기·손실은 공식 코드로 대조할 수 없고, 제3자(Hugging Face)의 학습 스크립트만 정적으로 대조했습니다(D11–D15).

## 1. 일치하는 것

| 논문 | 코드 | 위치 |
|---|---|---|
| forward z_t = (1−t)x0 + tε, t=0 데이터 / t=1 노이즈 | `noise_scaling`: σ·noise + (1−σ)·latent | `sd3_impls.py` 54–55 |
| 네트워크가 속도 v를 출력, x0 = z − t·v | `calculate_denoised`: x − output·σ, `to_d`: (x − denoised)/σ | `sd3_impls.py` 50–52, 330–332 |
| 은닉 64d, 헤드 d (헤드 크기 64), MLP 4× | `hidden_size = 64 * depth`, `num_heads = depth`, `mlp_ratio=4.0` | `mmditx.py` 743–744 |
| 두 스트림이 각자 가중치, 어텐션에서만 결합 | `JointBlock`이 `context_block`, `x_block`을 따로 두고 `block_mixing`에서 q, k, v를 dim=1로 이어 붙여 한 번 어텐션 | `mmditx.py` 591–648 |
| adaLN 변조(α, β, γ, δ, ε, ζ) | `adaLN_modulation`(SiLU + Linear → 6개 청크: shift/scale/gate × (어텐션, MLP)) | `mmditx.py` 470–484(`n_mods`) |
| 잠재 채널 16, 2×2 패치 | `in_channels=16`, `patch_size`는 체크포인트 가중치 모양에서 읽음 | `sd3_impls.py` 74–75, 110 |
| QK 정규화: 학습 스케일이 있는 RMSNorm, 두 스트림 | `qk_norm == "rms"`이면 `RMSNorm(head_dim, elementwise_affine=True, eps=1e-6)`를 q, k에 적용 | `mmditx.py` 243–257, 282–283 |
| CLIP-L(768) ‖ CLIP-G(1280) 풀링 = c_vec ∈ R^2048 | `torch.cat((l_pooled, g_pooled), dim=-1)` | `sd3_infer.py` 327–329 |
| c_ctxt = [CLIP 채널 이어붙임 77×2048을 4096으로 0 패딩 ‖ T5 77×4096] | `lg_out`을 `pad(0, 4096 − 2048)` 후 `torch.cat([lg_out, t5_out], dim=-2)` | `sd3_infer.py` 325–329 |
| 시간 이동 σ' = αt/(1+(α−1)t) | `sigma()`: `shift * t / (1 + (shift − 1) * t)`, `SHIFT = 3.0` | `sd3_impls.py` 41–48, `sd3_infer.py` 210–211 |
| CFG | `neg + (pos − neg) · cond_scale`, 조건·무조건을 한 배치로 | `sd3_impls.py` 189–215 |

## 2. 차이 D1–D16

| # | 항목 | 논문 | 코드 | 영향 |
|---|---|---|---|---|
| **D1** | 학습 코드 | §5, 부록 B.3에 학습 설정이 있고 초록이 "코드 공개"를 약속 | `sd3.5`는 추론 전용. 손실, 시간 표본기, EMA, DPO, 데이터 파이프라인 없음 | 논문 §3.1의 수치(표 1)는 공식 코드로 재현할 수 없음 |
| **D2** | 이동값 | 1024² 모델에 α=3.0을 학습과 샘플링 모두 사용(§5.3.2) | `SHIFT = 3.0`("publicly released models use 3.0"), `CONFIGS`의 `sd3.5_*`는 3.0이지만 **`sd3_medium`은 `shift: 1.0`**(`sd3_infer.py` 503–507) | 같은 파일 안에서 불일치. 코드만으로는 SD3 Medium의 학습 이동값을 판정할 수 없음(`SHIFT` 상수의 주석과 `CONFIGS` 중 하나가 틀림) |
| **D3** | 샘플러 | 모든 실험이 오일러 이산화(부록 B.3). CFG 1.0–5.0, 5–50스텝 | 기본 `dpmpp_2m`(2차 다단계, 로그 시그마 공간), 스텝 40–50, CFG 4.5–5.0. `euler`는 SD3.5 Large Turbo(4스텝, CFG 1.0)와 ControlNet(60스텝, CFG 3.5)만 | 논문의 "RF는 스텝 수를 줄여도 덜 나빠진다"(그림 3)는 오일러에서의 이야기이고 공개 기본 설정은 아님. 표 6의 5·10·20스텝도 오일러 기준으로 읽어야 함 |
| **D4** | 시그마 격자의 끝점 | 식 (23)을 균일 격자에 적용하면 끝은 t_min | `get_sigmas`가 `sigma_min`(이미 이동된 값)에 1000을 곱해 timestep으로 바꾼 뒤 `sigma()`를 한 번 더 적용. shift=3이면 마지막 0이 아닌 σ가 0.00893으로 표의 최소값 0.00299의 2.98배, shift=1이면 영향 없음(`code_checks.py`) | 마지막 스텝이 한 번 더 이동됨. 영향은 작지만(끝 부근 한 스텝) 식 (23)의 단순 적용이 아님 |
| **D5** | 시간 입력 | "시간의 사인파 인코딩"(그림 2) | `timestep = σ·1000`을 256차원 사인 임베딩 → MLP(`TimestepEmbedder`) | 이동 후의 σ가 입력이므로 네트워크는 **실제 노이즈 수준**을 봄. 이동된 학습·추론 일관성의 전제 |
| **D6** | QK 정규화 | "두 스트림의 Q, K에 학습 스케일 RMSNorm"(§5.3.2) | **헤드 단위**(`head_dim`=64) RMSNorm, eps 1e-6. 체크포인트에 `ln_k.weight`가 있을 때만 켬(없으면 `None`). `"ln"`(LayerNorm) 옵션도 있음 | 논문은 헤드 단위인지 전체 은닉인지 쓰지 않음. 코드는 헤드 단위 |
| **D7** | 마지막 블록 | 기술 없음 | 마지막 블록의 텍스트 스트림은 `pre_only=True`: adaLN이 2청크, MLP와 출력 사영이 없고 텍스트 출력은 버려짐(`block_mixing`에서 `context = None`). 이 때문에 파라미터가 36h²d³ 근사보다 약간 적음(`03` §4.4) | 구조상 자연스러우나 논문에는 없음 |
| **D8** | 블록 세부 | "LayerNorm", MLP 4× | `qkv_bias=True`, GELU(tanh) MLP, 정규화는 비아핀 LayerNorm(eps 1e-6). `rmsnorm`, `swiglu`, `scale_mod_only` 옵션은 기본 꺼짐 | 논문에 활성 함수·편향이 없음. 옵션은 다른 변형용 |
| **D9** | 잠재 정규화 | "평균·표준편차로 정규화(훈련 데이터 일부에서 전역 계산)"(부록 B.2), 값은 없음 | `SD3LatentFormat`: (x − 0.0609)·1.5305 (스칼라 둘) | 값이 논문에 없고 코드에서만 확인됨. 스칼라 한 쌍이라는 점도 논문에 없음 |
| **D10** | 텍스트 길이 | c_ctxt 고정 154×4096 (그림 2: 77 + 77 토큰) | CLIP 쪽 77토큰은 고정이나 T5 토크나이저는 `max_length=99999999`, `min_length=77`: 77보다 긴 프롬프트는 T5 토큰이 늘어 문맥이 154보다 길어짐 | 긴 프롬프트에서 어텐션 시퀀스가 가변. 논문 설명은 상한만 |
| **D11** | (diffusers) 시간 표본기 | logit-normal(m, s): u~N(m,s)를 시그모이드, **m>0이면 노이즈 쪽**(식 19) | `compute_density_for_timestep_sampling("logit_normal")`는 `sigmoid(N(m,s))`를 얻은 뒤 `indices = (u·1000).long()`으로 **내림차순 timestep 표**(index 0 = σ 1.0)를 조회 | 인덱스 방향이 반대라 m≠0이면 부호가 뒤집힘(양의 m이 데이터 쪽). 기본 m=0, s=1은 대칭이라 영향 없음 |
| **D12** | (diffusers) mode | 식 (20) f_mode(u; s) = 1 − u − s(cos²(πu/2) − 1 + u) | `u = 1 − u − mode_scale·(cos(π·u/2)**2 − 1 + u)` (같음), 기본 `mode_scale=1.29` | 일치. 유효 범위 검사는 없음(`03` §1.3: s > 1.7519는 단조가 깨짐) |
| **D13** | (diffusers) 손실 | RF는 속도 MSE `‖v_Θ − (ε − x0)‖²`(식 8), 시간 분포가 가중을 만듦(식 18) | 기본 `precondition_outputs=1`: `model_pred·(−σ) + z`로 x0를 예측해 **x0-MSE**를 계산. 이는 속도 MSE에 σ²를 곱한 것이고 논문의 w^π_t에 t²를 곱한 가중과 같음(`03` §1.2의 규약으로 w = t²·w^π) | 학습 신호가 데이터 쪽(작은 t)에서 억제됨. `precondition_outputs=0`으로 두면 `target = noise − model_input`으로 논문 식과 같음 |
| **D14** | (diffusers) CosMap | 시간 t를 π_CosMap에서 표본하고 가중 1(식 22) | `weighting = 2/(π(1 − 2σ + 2σ²))`을 **균일 표본**에 곱함(`compute_loss_weighting_for_sd3`) | 기댓값은 같음(시간당 가중이 같음). 분산이 큼(균일 표본 + 가중). 논문은 표본 방식 |
| **D15** | (diffusers) 이동 | 학습과 샘플링 모두 α=3.0 | 스케줄러의 `shift`(기본 1.0)를 `timesteps` 표에 적용하고 학습 스크립트가 그 표를 조회하므로 학습 시간도 이동됨. SD3 체크포인트의 `shift` 설정값은 이 저장소에서 확인하지 못함 | 설정에 따라 달라짐 |
| **D16** | (diffusers) 해상도 의존 이동 | 식 (23)의 √(m/n)를 쓰지 않고 α=3.0을 고정 | `use_dynamic_shifting`: t' = e^μ/(e^μ + (1/t − 1)^σ)(`_time_shift_exponential`), σ=1이면 α = e^μ인 식 (23)과 같은 함수 족. μ는 이미지 시퀀스 길이에서 `base_shift`(0.5)·`max_shift`(1.15)로 보간 | 식 (23)을 해상도에 연동한 후속 변형. 이 옵션은 SD3가 아니라 FLUX 계열이 쓰는 것으로 읽힘(미확인) |

추가로 SD3.5 Medium용으로 `x_block_self_attn_layers`(이미지 스트림에 두 번째 자기어텐션 `attn2`, adaLN 9청크, MMDiT-X)와 skip-layer guidance(`SkipLayerCFGDenoiser`)가 있습니다. 둘 다 논문에 없는 후속 변경입니다(SD3.5 공개 이후 추가, 인용 시 "논문의 MM-DiT"와 구분해야 함).

## 3. 요약

- **MM-DiT 본체(결합 어텐션, 모달리티별 가중치, QK 정규화, adaLN, 은닉 64d)와 시간 이동 식은 논문과 일치합니다.** 파라미터 수는 d=24에서 2.03 B, d=38에서 8.06 B로 논문의 "2B, 8B"와 맞습니다(`03` §4.4).
- **논문의 가장 새로운 부분인 시간 표본기와 손실 가중 코드는 공식 저장소에 없고**, 제3자 구현은 기본값이 논문과 다릅니다(D13 σ² 가중, D11 인덱스 방향, D14 CosMap 구현). 논문 수치를 쓰려면 어느 구현의 어느 옵션인지 밝혀야 합니다.
- **재현성의 변수:** 샘플러(D3: 기본이 오일러가 아님), 이동값(D2: Medium 1.0 대 3.0 표기 불일치), 텍스트 길이(D10), 끝점(D4)입니다.
- **하지 않은 것:** 코드 실행, 가중치 다운로드, 파라미터 수의 실제 모델 대조(체크포인트 없음), diffusers SD3 체크포인트의 `scheduler_config.json` 확인.
