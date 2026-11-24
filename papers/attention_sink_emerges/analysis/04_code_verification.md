# 04. 논문과 공식 코드 대조 (`sail-sg/Attention-Sink`)

- 저장소: `github.com/sail-sg/Attention-Sink`, 커밋 `cd141064cc96d6a1e078783042339e38de5a37ba`(2025-07-08), MIT, 별 164. `git clone --depth 1`로 받음(`gh api` 크기 6.4 MB, 작업 트리 21 MB, `.git` 5.6 MB). `repo/`(gitignore)에 있다.
- **정적 읽기만 했고 실행하지 않았다**(A100 40GB 4장 가정, flash-attn·xformers 소스 빌드, Pile 5B 토큰 전처리 약 20 GB, wandb 필요). 읽은 파일: `README.md`, `pretrain/tinyllama.py`, `pretrain/eval_sink_pretrain.py`, `eval_sink_open_source.py`, `lit_gpt/model_attn_sim.py`, `lit_gpt/inference/model_attn_sim_infer.py`, `lit_gpt/config.py`, `random_data.py`, `configs/**/*.yaml` 54개, `scripts/*.md`, `scripts/*/*.sh`.
- 저장소에는 **결과 표, 로그, 체크포인트가 없다**(학습 기록은 wandb). 그래서 03의 재계산은 논문 표 값만 쓴다.

## 1. 논문과 같은 것 (K)

| # | 논문 | 코드 | 판정 |
|---|---|---|---|
| K1 | Sink 지표: α_k = (1/(T−k+1)) Σ_i A_{i,k} (절 3.2) | `eval_sink_pretrain.py`: `ratios = arange(T,0,-1)`을 열 방향으로 두고 `(scores/ratios).sum(dim=-2)` → 열 k를 T−k+1로 나눔. `> epsilon`(0.3), 헤드·층 평균 | 일치 |
| K2 | T=64 | `token_length = 64`, 100개 시퀀스 | 일치(데이터 출처는 D3) |
| K3 | 60M 모델 d=768, L=10, H=8, FFN 1536, Rotary, RMSNorm, 편향 없음 | `config.py` `tinyllama_60M` | 일치. 비임베딩 58.98M(`03` §5) |
| K4 | 배치 1M 토큰, 20k 스텝, 워밍업 100, lr 4e-4 | 설정 yaml: `global_batch_size 512` × 문맥 2048 = 1,048,576 토큰, `max_step 20000`, `warmup_steps 100`, `learning_rate 0.0004` | 일치(스케줄은 D1) |
| K5 | weight decay 기본 0.1, AdamW | `weight_decay = 1e-1`, `AdamW(betas=(0.9,0.95))` | 일치(해석은 D5) |
| K6 | 정규화 없는 sigmoid/elu: `sim`에 sigmoid나 elu+1을 적용하고 Z=1 | `model_attn_sim.py` `eager_scaled_dot_product_attention`: `sigmoid(qk·scale)`, `elu+1`, 하삼각 곱, `attention_normalize`일 때만 행 합으로 나눔 | 일치 |
| K7 | proxy 점수 = sim / Σ sim | `inference/model_attn_sim_infer.py`: `attn_weight_normalized = attn_weight/(sum+1e-8)`를 반환해 지표에 사용 | 일치 |
| K8 | 첫 토큰 무작위화: x1 ~ Uniform(V) | `random_data.py`: 2049 블록의 `[start,end)` 위치를 `randint(0, 50432)`로 교체 | 일치 |
| K9 | 접두 LM(p>1): 앞쪽 손실 제외 | `ignore_loss = PREFIX−1`로 `logits[:, ignore_loss:]`만 손실 | 일치 |
| K10 | 설정 폴더가 논문의 변형 축을 덮음 | `configs/{decay,lr,batch,pos,ffn,kv_bias,prefix,window,attnsim,attnkernel}` | 대부분 일치(D2) |

## 2. 논문과 다르거나 논문에 없는 것 (D)

| # | 논문 | 코드 | 영향 |
|---|---|---|---|
| D1 | "learning rate 4e-4 with cosine scheduling"(절 4) | **모든 설정 yaml이 `min_lr`를 `learning_rate`와 같게 둔다**(54개 중 1B 포함 전부, 예: `min_lr: 0.0004`). `get_lr`는 `min_lr + coeff·(lr − min_lr)`이므로 워밍업 뒤 **상수 학습률**이다. 코드 기본값만 `min_lr=1e-5`이지만 yaml 값으로 덮어쓰인다(`globals()[k] = v`) | 논문의 "cosine"은 이 공개 설정과 다르다. 학습률 효과(표 9)는 상수 학습률 기준이며, 학습 끝에서 lr이 줄지 않으므로 sink 측정 시점의 lr이 논문 서술보다 높다 |
| D2 | 표 9의 학습률: 8e-4(10k, 20k), 4e-4(20k), 2e-4(20k, 40k), 1e-4(20k, 80k) | 공개 `configs/lr/`에는 **1e-3(20k), 1e-4(40k), 1e-5(80k)** 세 개뿐. 8e-4, 2e-4는 없고 1e-4의 스텝이 논문(20k, 80k)과 다르다. `scripts/optimization.md`는 "`$lr`는 1e-3, 1e-4, 1e-5 중에서"라고 안내 | 표 9 일곱 행은 공개 설정만으로 되풀이할 수 없다. 표 14 오른쪽의 sigmoid lr 1e-3, 1e-4는 가능 |
| D3 | 측정: "훈련 데이터에서 100개 시퀀스, T=64, BOS 없음"(절 4) | 측정 코드는 `datasets/probe_valid.jsonl`(**170개 문서**)에서 앞 100개 문서의 **문서 첫 64토큰**을 읽는다(`tokenizer.encode(prompt, eos=False)`, `bos=False`). 학습 입력은 패킹된 2048 조각이라 첫 토큰이 문서 중간일 수 있다. 첫 문서는 PubMed 초록 | 측정 입력의 첫 토큰은 항상 문서의 첫 토큰이고 학습 때와 분포가 같다고 보장되지 않는다. 논문 문장과 코드가 다르다 |
| D4 | "Pile에서 5B 토큰 샘플"(절 4) | 데이터는 `regmix-data-sample`(HF)이고 yaml이 17개 Pile 도메인에 **혼합 가중치**를 준다(arXiv 0.1052, PubMed Central 0.1071, Pile-CC 0.1121, ...). 20k 스텝 × 1,048,576 = **20.97B 토큰**을 읽는다 | 5B 토큰 데이터를 약 4.2회 반복한다(`03` §5). 논문 본문에는 반복 횟수 언급이 없다. 데이터 양 실험은 같은 스텝으로 50M(419회)까지 반복시킨다 |
| D5 | 손실 L + γ‖θ‖²₂ (절 6) | `torch.optim.AdamW(weight_decay=γ)`, 즉 **분리형(decoupled) weight decay**를 `requires_grad`인 모든 파라미터(임베딩·RMSNorm 포함)에 적용. 설정 `decay`는 0, 1e-3, 1e-2, 0.5, 1, 2, 5, **10**(논문 표 2에는 10 없음) | γ의 의미가 L2 정규화 계수와 다르다(AdamW에서 효과는 lr·γ). 표 2의 γ=5.0은 스텝당 lr·γ = 0.002 감쇠. 다른 구현에 옮길 때 같은 γ를 L2로 넣으면 결과가 다르다 |
| D6 | 정규화 없는 attention의 지표(절 7.4) | 지표는 proxy(`attn_weight_normalized`)로 **정규화 없는 모델에서만 별도 반환**. 훈련에는 정규화 없는 값이 쓰인다 | 논문이 밝힌 대로다. 다만 softmax는 이미 합이 1이라 같은 척도가 아니다 |
| D7 | 고정 토큰 실험(표 10 오른쪽) | `pretrain/tinyllama.py`의 `mode == "fix"` 분기가 `train_data = train_data[:, fix_position] = 1`이다. Python의 연쇄 대입이라 `train_data`를 정수 1로 바꾼 뒤 `1[:, p] = 1`을 시도하므로 바로 다음 줄 `train_data[:, 0:block_size]`가 정수에 슬라이스를 하게 되어 **첫 스텝에서 TypeError**가 날 것으로 읽힌다(실행하지 않고 읽기만 함). `validate`의 같은 줄도 같은 형태다 | `scripts/data_distribution/run_fix_token.sh`는 이 경로를 쓴다. 논문의 고정 토큰 결과가 이 공개 코드 그대로 나왔는지 확인할 수 없다 |
| D8 | 시드 정보 없음 | 모든 yaml `seed: 42`, `data_seed 3406`. 논문 표는 변형마다 실행 1회 | 시드 변동을 보고하지 않아 "영향 있음/없음"을 구간으로 가를 수 없다 |
| D9 | 정밀도 언급 없음 | 측정은 `model.to(torch.bfloat16)`에서 attention 확률과 α_1을 bf16으로 합산 | α_1이 임계값 0.3 가까이 있는 헤드는 bf16 반올림(약 3자리)으로 갈릴 수 있으나 평균 지표에는 영향이 작을 것(추정, 확인 안 함) |
| D10 | 공개 모델 측정(표 1, 그림 3, 4) | `eval_sink_open_source.py`는 `tokenizer(prompt, return_tensors="pt")`를 쓴다. HF 토크나이저의 기본 `add_special_tokens=True`이면 LLaMA2/3·Mistral은 **BOS가 앞에 붙는다**(토크나이저 설정을 받지 않아 확인 안 함, 추정) | 사전학습 모델 표 1의 "첫 토큰"이 BOS일 수 있다. 사전학습 60M 모델 측정(BOS 없음)과 조건이 다르다. 표 1의 "반복 토큰" 열 해석에도 BOS 처리가 영향을 준다 |
| D11 | 1B 모델: 22층, 헤드 수 언급 없음 | `tinyllama_1_1b`: n_layer 22, n_head 16, n_embd 2048, FFN 5632, **그룹 쿼리 없음**(MHA). 비임베딩 1.13B, 헤드 352개 | "1B"는 비임베딩 1.13B이다. 헤드가 352개라 지표 한 칸은 0.28 pp |

## 3. 구현 노트

- 평가는 `output_attention=True`일 때만 **eager attention**으로 확률을 만들어 반환한다(100개 × T=64이므로 부담 없음). 학습은 flash-attn.
- sigmoid 분기에서 마스크는 `sigmoid(...) * attention_bias(0/1 하삼각)`로 곱한다. 마스킹된 위치가 sigmoid(−∞)=0이 아니라 0을 곱해 얻은 값이라 수치는 같다.
- `eval_sink_pretrain.py`는 출력 헤더만 보면 `Sink`를 **위치 0–9별**로 출력한다(`metric1[:, :10].mean(dim=0)`). 논문 표의 Sink_1은 그중 첫 값이다.
- 학습 중 sink 곡선(그림 4, 28)을 만드는 코드는 이 저장소에 없다. 체크포인트마다 `eval_sink_pretrain.py`를 돌렸다고 읽힌다.

## 4. 재현 제안 (제안만, 승인 필요)

| 단계 | 내용 | 자원 | 이 맥 가능 |
|---|---|---|---|
| a | 사전학습 공개 가중치(GPT-2, Pythia-14M–160M)로 `eval_sink_open_source.py` 지표만 측정. BOS 유무와 ε, T를 바꿔 D3·D10 점검 | Pythia-14m/70m는 수십–수백 MB, GPT-2는 0.5 GB(다운로드·transformers 설치 필요) | 가능(승인 필요: 설치, 다운로드) |
| b | 공식 코드의 60M 학습을 1/10 규모(2k 스텝, 소량 데이터)로 줄여 sigmoid 대 softmax만 | CUDA 권장, flash-attn 빌드가 막힘 | 어려움 |
| c | D1 확인: 같은 설정에서 코사인 스케줄을 쓴 짧은 학습 | 같음 | 어려움 |
