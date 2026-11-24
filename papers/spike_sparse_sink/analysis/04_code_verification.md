# 04. 논문 ↔ 공식 코드 대조

- 저장소: github.com/savinasun/SpikeSparseSink, 커밋 `884cba1ecd68f7d3690704e7f88fe792c3bc37a2`(2026-03-06), 라이선스 파일 없음, 별 3.
- 크기: `gh api` 18 KB, 얕은 clone 후 작업 트리 468 KB(`du -sh`). `repo/`에 두었고 gitignore입니다. **실행하지 않았고 정적으로만 읽었습니다.**
- README 자체의 범위: "Official PyTorch implementation of the **ablation experiments in Section 4**". 즉 §3(사전학습 모델 해부)와 평가(perplexity, sink ratio, spike 크기) 코드는 범위 밖입니다.
- 구성: `train.py`(FSDP2, 배열 데이터셋, bf16), `helper/`(엔진, 데이터셋), `models/`(변형 10개 파일), `configs/`(base + train_*.yaml 42개).

## 1. 대응이 확인된 것

| 논문 | 코드 | 결과 |
|---|---|---|
| 표 9: 32층, d=4096, 32 head, dhead 128, FFN 11008, 어휘 32000 | `configs/train_0.yaml` model_config | 일치 |
| 표 9: AdamW, β=(0.9,0.95), wd 0.1, clip 1.0, lr 3e-4, warmup 2000, 코사인, min LR 비 0.1 | `train_0.yaml`, `train.py`의 스케줄 | 일치. 워밍업은 `1-(1-x/w)²`(이차형) |
| 표 3 학습률 {7.5e-5, 1.5e-4, 6e-4, 1.2e-3} | `train_1..4` | 일치 |
| 표 3 wd 0 / β2 0.9, 0.999 / min LR 3e-4(=비 1.0) / 200B | `train_11`, `21`, `22`, `41`, `31`(steps 100,000) | 일치 |
| 표 6 head 수 16, 8 / dhead 64..8 / 총용량 고정 6개 | `train_91,92`, `101–104`, `111–115` | 일치(head 수 어긋남 없음) |
| 표 4 GeLU, 선형, attention-only | `train_121–123` | 대응(D3의 크기 차이 참고) |
| 표 5 Sandwich, DynamicTanh | `train_142`, `141` | 대응(D4 참고) |
| 표 7 조건부 gate 3종 | `train_151–153` + `llama_gated_attention.py` | 대응. 입력은 attn_norm을 거친 x, 시그모이드 gate를 SDPA 출력에 곱함 |
| 표 8 문맥 분포 | `train_171–174`, `helper/engine.py`의 `context_length` 슬라이싱 | 대응. 모델은 전 위치를 보고 손실만 [a:b]에서 계산 |
| 시드 | `base.yaml`의 `seed: 2025`, 모든 config가 같은 시드 | **시드 하나 고정** (논문 본문에도 시드 언급 없음) |

## 2. 차이와 미확인 (D1–D8)

| ID | 논문 | 코드 | 영향 |
|---|---|---|---|
| D1 | 표 9: 학습 스텝 5,000, 배치 2M 토큰 → 10B 토큰. 본문 §4.1: 100B 토큰 | `train_0.yaml` num_steps 50000 → 100B (배치 2M이면). 200B 행은 `train_31` 100000 | **표 9의 5,000은 오기로 보인다.** 100B 기준 맞는 값은 50,000. 배치 크기 자체는 config에서 자리표시자(`batch_size`는 GPU당)라 2M은 확인 불가 |
| D2 | 표 5 "QKNorm: 입력 정규화를 q와 k에만" | `llama_qk_output_normalization.py`: `q_norm`, `k_norm`을 **전체 투영 폭**(num_heads×head_dim)에 걸어 쓰고(head별 아님), `attn_norm`, `ffn_norm`을 **입력이 아니라 블록 출력에** 적용(`h = x + attn_norm(attn(x))`, `x = h + ffn_norm(ffn(h))`). 즉 블록 입력 정규화가 전부 없음. FFN과 V 경로는 정규화 안 된 잔차를 그대로 받음 | 표 5의 "Sandwich(QK)"는 "post-norm + qk-norm, pre-norm 없음"에 가깝다. 'spike 92'는 이 구성의 결과이고 "pre-norm에서 QKNorm만 추가"로 읽으면 틀림. 논문이 이 구성을 분명히 서술하지 않음 |
| D3 | §4.2.1 "표준 2층 GeLU, 단일 선형, attention-only" | 선형 FFN은 `Linear(d,d)`. attention-only는 FFN 자리에 Attention 블록. GeLU는 FFN 폭을 16,512로 키워 SwiGLU와 같은 파라미터 수(논문에 서술 없음) | 우리 계산(`recompute_tables.py`): SwiGLU와 GeLU 6.74B, attention-only 4.56B, 선형 2.95B. **표 4의 spike 크기(637–688)와 ppl 12.5는 모델 크기 차이와 섞여 있다** |
| D4 | 표 5 DynamicTanh | `llama_dynamic_tanh.py`는 블록 입력 정규화 2개만 DyT로 바꾸고, **최종 `self.norm`은 RMSNorm으로 남김**. α 초기값 attn 0.8, ffn 0.2 | 최종 RMSNorm이 남아 "정규화 없는 모델"은 아님. spike 153은 블록 내부 정규화 제거만의 결과 |
| D5 | 표 7의 비조건부 3행(channel, head, single)과 static 2행(positional, token embedding) | `train_154`(unconditional head), `155`(unconditional "token", 게이트 모양 `(max_seq, heads, head_dim)` → 위치별 게이트), `156`(임베딩 조건 gate), `157`(unconditional standard이지만 **hidden 3072, 24 head, FFN 8256의 작은 모델**) | 표의 5행에 대응하는 config가 4개이고 "비조건부 single(스칼라)"는 없음. 157은 표 9 기준 모델과 크기가 달라 비교가 어긋남. 어느 config가 어느 행인지는 추정 |
| D6 | 표 8 행: 1/256, 1/1024, 1/4096 | `train_162`(seq 1024), `163`(seq 256)가 대응. **`train_161`(seq 16384)은 있지만 표에 결과가 없음** | 긴 문맥 학습 결과가 보고 안 됨(결과가 불리했는지는 알 수 없음) |
| D7 | spike("최대 activation 크기")와 sink ratio(식 28, ε=0.3, T=64) | **측정 코드 없음.** `Llama.forward(record_mode=True)`가 층별 attention 맵과 hidden을 돌려주는 훅만 있고 지표를 계산하는 스크립트는 없음 | 어느 층·어느 토큰 위치에서 최댓값을 잡는지, 평균이 무엇인지 검증 불가. 표 3–8의 모든 수치는 코드로 재계산 불가 |
| D8 | §3 전체(step-up/down, 이차형식, 고유값, t-SNE) | 코드 없음 | §3 수치는 전부 검증 불가 |
| D9 | `models/__init__.py` | `llama_headqk_output_normalization`, `llama_ffn_only`, `llama_duo_attention` 분기는 있지만 해당 파일이 저장소에 없음 | 논문에 없는 실험 흔적이거나 정리 누락. 해당 config는 없어 현재 사용에는 영향 없음 |

## 3. 기타 관찰

- 데이터: `DistributedDataset`은 토큰화된 Arrow 청크를 받습니다. DCLM의 토큰화·청크 분할(문서 경계, 위치 0이 문서 시작인지) 코드는 없습니다. 이것이 "위치 0 = spike" 실험의 해석에 중요합니다. 문서가 무작위 위치에서 잘렸다면 위치 0은 문서 시작이 아닙니다.
- 어텐션 마스크는 시퀀스 전체에 단순 인과 마스크입니다(문서 경계 마스킹 없음).
- 정밀도: 파라미터 bf16, attention은 SDPA(cuDNN/FlashAttention)에서 bf16으로 변환. `record_mode`는 수동 softmax를 float32로 계산해 학습 경로와 다릅니다.
- 초기화: 임베딩 N(0,1), 나머지 N(0,0.02²), RMSNorm weight 1. 임베딩 초기 크기 1은 보통의 0.02보다 큽니다. spike 출현 시기에 영향이 있을 수 있지만 이를 바꾼 실험은 없습니다(우리 추정).
- 학습에 쓰는 평가 분리: 코드에는 검증 손실 계산이 없고 학습 loss만 로깅합니다. 표의 perplexity가 어떤 데이터(C4)에서 어떤 길이로 계산됐는지 코드로 확인 불가(`§A`는 C4 청크를 모델 문맥 길이에 맞췄다고 서술).

## 4. 판정

- **설정 재현 가능성:** 표 3–8의 대부분(약 40개 행)은 config로 1:1 대응이 확인됩니다. 이 대응은 정적 확인이며 실행 결과는 검증하지 못했습니다.
- **수치 재현 가능성:** 지표 코드와 §3 코드가 없고 시드가 하나여서 수치 재현이나 오차 추정은 불가능합니다.
- **계산 비용(우리 추정, `recompute_tables.py`):** 100B 토큰 한 번에 약 4.0×10²¹ FLOPs(6ND), H100 40% MFU 가정 시 약 2,800 GPU-시간. config 42개(200B 행 포함)이면 10⁵ GPU-시간 규모. 이 저장소 규칙상 재현하지 않습니다.
