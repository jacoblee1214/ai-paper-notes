# 02. 계보와 이웃 연구 (FlashAttention, arXiv:2205.14135)

## 1. 구성 요소별 출처 (논문 §3.1, 부록 A, B.5)

| 구성 요소 | 출처 | 논문이 한 일 |
|---|---|---|
| online softmax (블록 통계 m, ℓ로 softmax 분해) | Milakov & Gimelshein 2018 (arXiv:1805.02867, 논문 [60]). 논문은 같은 분해를 쓴 Reformer [51], Rabe & Staats [66]도 함께 인용 | 이를 attention 출력 O까지 확장(O를 블록마다 재척도해 갱신, Alg. 1 줄 12) |
| 선형 메모리 attention | Rabe & Staats 2021 (arXiv:2112.05682) | 같은 아이디어를 "메모리 접근 수" 관점으로 재정리하고 속도까지 얻음 (부록 B.5의 세 차이: 목표가 접근 수, O를 하나만 유지, 역전파를 해석적으로 단순화) |
| 재계산 | gradient checkpointing (Chen et al. 2016 [10], Griewank & Walther 2008 [34]) | 선택적 checkpointing으로 본다. S, P만 다시 계산하고 블록 임시 출력은 다시 만들지 않는다 |
| IO 복잡도, 메모리 위계 | Aggarwal & Vitter 1988 [1], roofline [85], 작업 집합 모형 [21] (부록 A) | 2단계(HBM/SRAM) 모형으로 attention 분석 |
| 커널 융합 구현 | NVIDIA Apex FMHA (길이 ≤512, head dim 64, A100 전용) (부록 E.4) | FMHA 코드를 출발점으로 타일링·재계산을 더해 긴 시퀀스와 여러 head dim 지원 |
| 블록 희소 패턴 | butterfly (Pixelated Butterfly, Dao et al. [17]), Sparse Transformer [11], Longformer [3], BigBird [92] | 희소 패턴을 새로 제안하지 않고 기존 butterfly를 IO 인식 커널 위에 올림 |

## 2. 가까운 대안 (논문 시점)

- **근사 attention**: Linformer, Performer, Reformer, SMYRF, Local, Linear attention (Table 3). 논문의 주장은 이들이 FLOPs를 줄여도 접근 수 때문에 벽시계 이득이 작다는 것이고, 길이 512–1024 사이에서 일부(Linformer)가 Flash보다 빨라진다고 직접 적는다(§4.3).
- **Rabe & Staats**: 메모리는 줄지만 속도는 표준과 비슷하거나 약간 느리다고 논문이 적는다(부록 B.5). 이 속도 비교는 논문의 주장이며 우리는 확인하지 않았다.
- **Megatron-LM 융합 softmax/마스크, Apex FMHA**: 융합만 하고 역전파용으로 P를 저장(FMHA).

## 3. 이후 계보 (본 폴더에서 확인한 것만)

| 후속 | 바뀐 것 | 확인 |
|---|---|---|
| **FlashAttention-2** (arXiv:2307.08691, 2023-07) | v1이 이론 최대 FLOPs/s의 25–40%에 그친다고 진단. 작업 분할 개선(Q 블록을 바깥 루프로, 스레드 블록 병렬화) | 초록 확인. 공식 코드 구조는 `04` D1 |
| **FlashAttention-3** (arXiv:2407.08608, 2024-07) | Hopper 비동기(TMA, warp specialization), FP8. H100에서 FA2가 35% 활용에 그친다고 진단, 1.5–2.0×, FP16 최대 740 TFLOPs/s(75%) | 초록 확인 |
| **I/O 하한** Saha & Ye (arXiv:2402.07443) | M ≥ d²에서 Flash와 상수 배 안에서 맞는 하한 증명, M < d²에서는 더 나은 알고리즘 제시 | 초록 확인 (`04` C2) |
| **PagedAttention / vLLM** (arXiv:2309.06180) | 추론 시 KV 캐시 메모리 관리(페이지 단위). Flash가 계산을 다룬다면 이쪽은 캐시 배치를 다룬다 | 같은 묶음에서 작성 중: `../pagedattention/` |
| **StreamingLLM** | attention sink로 KV 캐시 유지 | 같은 묶음에서 작성 중: `../streamingllm/` |
| **Is Flash Attention Stable?** Golden et al. (arXiv:2405.02803) | BF16 순전파에서 Flash가 baseline보다 약 한 자릿수 큰 수치 편차(초록) | 초록 확인 (`04` C7) |

(Flash-Decoding, ring attention 등 다른 후속은 이 폴더에서 확인하지 않아 표에 넣지 않았다.)

## 4. 이 저장소의 다른 분석과의 관계

- [`../attention_sink_emerges/`](../../attention_sink_emerges/), [`../spike_sparse_sink/`](../../spike_sparse_sink/): 학습 코드가 flash-attn을 쓴다는 언급이 각 `04`에 있다. **우리 GPT-2 측정(`03` §5)에서 헤드 최대 로짓은 중앙값 약 20, 최대 약 299였고 층 4·2·3이 큰 로짓을 보였다.** attention sink 헤드의 큰 로짓이 fp16 naive attention 오차를 키우는 경로가 되는지는 원인 분리가 안 된 상관(로그 최대 로짓과 로그 오차비의 상관 0.67)이다.
- [`../dinov2/`](../../dinov2/): 자체 FlashAttention류 어텐션을 쓴다는 언급(`01`). high-norm 토큰과의 관계는 이 폴더에서 다루지 않는다.
- [`../prismatic_vlms/`](../../prismatic_vlms/): 해상도 증가의 어텐션 메모리를 FlashAttention류가 줄인다는 언급. 우리 CPU 측정(`03` §7)은 메모리가 N²이 아니라 거의 선형으로 늘어남을 보여 준다.

## 5. 자료 표

| 항목 | 값 (2026-11-28 확인) |
|---|---|
| arXiv | 2205.14135, v1 2022-05-27, v2 2022-06-23, 저자 Dao, Fu, Ermon, Rudra, Ré, cs.LG |
| 라이선스 (arXiv) | 비독점 배포 라이선스(`arxiv.org/licenses/nonexclusive-distrib/1.0/`), CC 아님 |
| 학회 | NeurIPS 2022 포스터 (Poster Session 4), 공식 virtual JSON과 proceedings DOI 10.52202/068431-1189 (쪽 16344–16359, Advances in NeurIPS 35), OpenReview `H4DqfPSibmx` |
| 공식 코드 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention), BSD-3-Clause, 별 25,356, 마지막 push 2026-11-26, 최신 릴리스 v2.8.3.post2(2026-10-13), 저장소 약 25 MB |
| 본 폴더의 clone | HEAD `2676a7be000545279a692d97151eac0762df6842`(2026-11-25), `repo/`, 정적으로만 읽음(`04` D1) |
