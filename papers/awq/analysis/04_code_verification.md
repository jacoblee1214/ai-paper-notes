# 04. 논문 ↔ 공식 코드 대조 (정적)

대상은 `mit-han-lab/llm-awq`(MIT, 별 3,668, 마지막 push 2025-07-17, 보관(archived) 아님, 열린 이슈·PR 197개(GitHub `open_issues_count`))입니다. 커밋 `d6e797a42b9ef7778de8ee2352116e0f48a78d61`(2025-07-17)을 `awq/repo/`에 얕게 받았습니다. 작업 트리 전체는 영상·GIF 때문에 약 190 MB라서 부분 복제(`--filter=blob:none --sparse`)로 `awq/`, `tinychat/` 파이썬 파일, README, LICENSE만 내려받았습니다(1.4 MB). **읽기만 했고 실행하지 않았습니다.** 줄 번호는 이 커밋 기준입니다. 알고리즘 핵심은 `awq/quantize/` 네 파일(`auto_scale.py` 480줄, `auto_clip.py` 98, `quantizer.py` 165, `pre_quant.py` 254)이고, 4비트 저장과 커널은 `qmodule.py`와 `awq/kernels/`, `tinychat/`입니다(커널·TinyChat 속도는 읽지 않았습니다).

우리 구현(`minimal_impl.py`, `run_experiments.py`)이 각 항목을 어떻게 처리했는지를 마지막 열에 적었습니다.

## 차이 목록

| # | 논문(위치) | 코드(파일:줄) | 영향 | 우리 구현 |
|---|---|---|---|---|
| D1 | 양자화기는 식 1: 대칭 absmax, Δ = max\|w\|/2^(N−1) | 기본은 **zero-point 비대칭**: scale = (max−min)/(2^n−1), zero = round(−min/scale)를 [0, 2^n−1]로 자름 (`quantizer.py:61–103`, `entry.py:114`의 `zero_point=not args.no_zero_point`). 대칭 분기는 "we actually never used this" 주석이 있고 정의되지 않은 `min_val`을 단언해(`:77`) 실행하면 오류가 나는 사실상 죽은 코드 | 식 2–3의 Δ'/Δ 논증은 대칭 Δ 기준. 비대칭은 그룹의 max와 min이 둘 다 Δ를 정함(우리 토이: 그룹 최댓값이 바뀐 비율이 s=2에서 대칭 17% 대 비대칭 20%, `03` §1) | 둘 다 구현. 모델 실험은 코드와 같은 비대칭 |
| D2 | 식 5: s = s_X^α | `s = x_max.pow(ratio).clamp(min=1e-4)`에 이어 **`s / sqrt(max(s)·min(s))`로 정규화** (`auto_scale.py:130–131`). 논문에 없음 | 전체에 상수를 곱해도 양자화 결과는 같음(비대칭 양자화는 척도 등변). 수치 범위 관리용. 우리 테스트가 확인(`test_global_rescale_invariance`) | 같은 정규화 사용 |
| D3 | 식 4: ‖Q(W·diag(s))(diag(s)⁻¹X) − WX‖, 층 하나의 출력 | **블록 출력**을 손실로 씀: q·k·v 스케일은 `self_attn` 모듈 출력, gate·up은 `mlp` 출력, o_proj·down_proj·OPT의 fc1·fc2는 층 하나 (`auto_scale.py:109–171`, `module2inspect`). 손실은 MSE 평균(`:139–141`) | q, k 스케일의 효과가 softmax를 거친 뒤 평가됨. 식 4를 그대로 구현하면 q·k 스케일이 어텐션 점수 왜곡을 못 봄. 본문은 이 차이를 말하지 않음 | **층 출력**(식 4 문자 그대로). 모듈 출력 손실은 구현하지 않음 → 우리 AWQ 결과가 코드와 다를 수 있음(`03` §4.2의 한계) |
| D4 | §3.2 "[0, 1]에서 격자 탐색", §5.1 격자 20 | `for ratio in range(n_grid): ratio = ratio / n_grid` → α ∈ {0, 0.05, …, **0.95**}, 1.0은 **제외** (`:128–129`) | 우리 모델에서는 선택된 α가 0.2–0.4 부근이고 0.95 이상은 한 번도 없어서(`03` §4.2) 영향 없음. 다른 모델에서는 모름 | 같은 격자(옵션으로 α = 1 포함) |
| D5 | 식 4의 Q(W·s)(s⁻¹X)를 가중치에 곱하고 입력에서 나눔 | 탐색 중에는 `W ← Q(W·s)/s`로 **가짜 양자화를 되돌려 놓은 형태**로 손실을 계산하고 상태를 복원(`:132–148`). 최종 적용은 `apply_scale`이 s⁻¹를 앞 연산(LayerNorm, 앞 선형 층, GELU 뒤 `ScaledActivation`)에 흡수하고 W·s를 둠(`:373–`) | 정확한 산술에서는 같은 함수. FP16에서는 흡수 위치에서 반올림이 달라짐(`04` D11도 참조) | 탐색과 같은 형태(`scaled_quantize`). 흡수는 구현하지 않고 등가성만 테스트(float64 1e-12) |
| D6 | §3.2 끝 한 문장: "weight clipping to minimize the MSE error of quantization" | `auto_clip_layer`: (출력 행, 그룹)마다 상한을 원래 absmax의 1.0, 0.95, …, **0.55**(10개, `max_shrink=0.5`, `n_grid=20`) 중에서 고름. **출력 오차**를 최소화(입력은 문서에서 512토큰만 균등 추출). 클램프는 **대칭** [−m, m]이고 양자화기는 비대칭. `q_`, `k_`, `query`, `key`, `Wqkv` 층은 건너뜀 ("qk bmm 때문에 정확히 clip하기 어렵다", `auto_clip.py:10–83`) | 논문 문구는 가중치 MSE로도 읽힘. 코드는 입력 의존 출력 MSE. clipping의 기여는 우리 실험에서 큼(`03` §4.2: W3에서 perplexity가 GPT-2 16%, Pythia 45% 추가로 낮아짐). **논문 Table 3의 "AWQ"가 clipping을 포함하는지** 본문에 분리 보고 없음 | 같은 정의(출력 MSE, 대칭 클램프, 10개 후보). 건너뛰는 층은 구현하지 않음(우리 qkv도 clip함) |
| D7 | §3.2: 보정 입력 X는 "작은 보정 집합"의 입력 특징 | 블록 i의 입력은 **앞 블록들이 양자화되기 전 값**을 이어서 계산한 것(`pre_quant.py:193–195`에서 `inps = layer(inps)`는 이 블록의 스케일·clip 적용 전에 계산). 따라서 앞 블록의 양자화 오차가 뒤 블록의 탐색에 전달되지 않음 | 층 단위 탐색이 서로 독립이라는 우리 구현과 같은 조건. GPTQ의 순차 방식(앞 블록 양자화 후 입력)과 다름 | 같은 조건(FP 모델을 한 번 돌려 모든 층 입력을 모음) |
| D8 | 논문은 지원 구조를 열거하지 않음 | 스케일을 흡수할 앞 연산이 있는 곳에만 적용: OPT(네 곳), Llama는 `v_proj`와 `o_proj`의 형상이 같을 때만 v→o 스케일(GQA인 70B는 건너뜀, `:223–229`), BLOOM은 attn 출력 스케일 주석 처리, **GPT-NeoX는 attention 출력(`dense`) 스케일 없음**(`:446–`), GELU·SiLU 뒤는 `ScaledActivation`(추가 곱 연산)으로 처리 | 논문의 "모든 선형 층에 s" 서술보다 코드가 좁음. 실제 AWQ 모델에서 일부 층은 RTN임 | 우리는 qkv, o, fc1, fc2 모두에 적용(Pythia의 `dense`는 코드와 다름) |
| D9 | §5.1: Pile의 작은 보정 집합. Figure 8(a)는 보정 시퀀스 수(×2048 토큰)를 변화 | `entry.py:208`: `n_samples=128, seqlen=512`. 데이터는 `mit-han-lab/pile-val-backup` validation을 시드 42로 섞고, **512토큰을 넘는 문서는 버리고**(`calib_data.py:17`), 남은 것을 이어 붙여 512토큰 블록으로 자름 | 짧은 문서에 치우친 보정. 논문 본문(2048 토큰 단위)과 코드 기본값(512)의 차이. 그림 8과 같은 설정을 재현하려면 인자를 바꿔야 함 | pile-10k를 시드 0으로 섞어 이어 붙인 512토큰 32개(기본값과 다름. 공식 설정은 128개) |
| D10 | 논문은 "LLM을 양자화" | 블록 안의 `nn.Linear`만 양자화(`get_named_linears`). 임베딩, lm_head, 정규화는 FP16 그대로. `quantizer.py:10–11`의 `EMBEDDING_KEYWORDS`, `LM_HEAD_KEYWORDS`는 어디서도 쓰이지 않음(죽은 상수) | 모델 크기 계산에서 임베딩·헤드가 FP16이라는 점이 중요(Llama-2-7B 약 0.52 GB, `03` §6) | 같은 범위 |
| D11 | §4: 4비트 가중치 패킹 | `WQLinear`가 `qweight`(int16 패킹), `scales`(FP16), `scaled_zeros`(FP16, = −zero·scale)를 **그룹마다 FP16 두 개** 저장(`qmodule.py:86–130`, 폭은 `calculate_zeros_width`로 pack_num의 배수로 올림) | 그룹 128에서 가중치당 4 + 32/128 = **4.25비트**(0.25 오버헤드). zero를 4비트로 패킹했다고 가정하면 4.16비트. 모델 크기 산술에 3% 차이(`03` §6) | 산술만 |
| D12 | | `get_weight_scale`(`auto_scale.py:18–25`)은 정의만 있고 호출되지 않음. 초기 판의 가중치 크기 기반 항의 잔재로 보임 | 논문 식 5와 일치(활성만 씀) | 해당 없음 |

## 읽은 범위와 한계

- 읽음: `awq/quantize/{auto_scale,auto_clip,quantizer,pre_quant,qmodule}.py`, `awq/utils/calib_data.py`, `awq/entry.py` 일부(양자화 흐름 195–240줄과 인자).
- 읽지 않음: `awq/kernels/`(CUDA), `tinychat/`의 모델·커널, `awq/quantize/smooth.py`·`w8a8_linear.py`(SmoothQuant 계열), 평가 어댑터. 따라서 TinyChat의 속도, 커널 융합, SIMD 패킹(§4)은 코드로 확인하지 않았습니다.
- 줄 번호는 위 커밋 기준이며 `main` 최신이 아닐 수 있습니다(마지막 push가 2025-07-17이라 같을 가능성이 높지만 확인하지 않음).
- 코드를 실행하지 않았으므로 "코드가 이 값을 낸다"는 주장은 하지 않습니다. D1–D12는 소스 읽기에서 나온 차이입니다.

## 참고: 이웃 구현 (정적 읽기, 같은 날)

| 저장소 | 라이선스 | 별 | 마지막 push | 메모 |
|---|---|---|---|---|
| `mit-han-lab/llm-awq` | MIT | 3,668 | 2025-07-17 | 위 대조의 대상 |
| `ml-explore/mlx-lm` | MIT | 7,778 | 2026-11-24 | `mlx_lm/quant/awq.py`가 같은 탐색 구조(D2의 정규화, 격자 `ratio/n_grid`, 평균 \|x\|, clip 격자 `1 − i/n_grid`)를 MLX로 옮김. 기본 그룹 64, CLI 기본 4비트, 샘플 32, 격자 10(`LEARNED_QUANTS.md`). 04의 D4·D6이 그대로 이어지는지는 읽지 않음 |
| `casper-hansen/AutoAWQ` | MIT | 2,343 | 2025-05-11 | **보관됨**(GitHub `archived: true`) |
| `vllm-project/llm-compressor` | Apache-2.0 | 3,998 | 2026-11-27 | 이 저장소에서 읽지 않음. 별과 push만 확인 |
