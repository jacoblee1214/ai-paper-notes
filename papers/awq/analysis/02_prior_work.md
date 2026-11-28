# 02. 계보와 이웃 연구

## 1. AWQ가 서 있는 자리

가중치 PTQ(학습 후 양자화)는 크게 네 갈래입니다. 아래 분류는 이 저장소의 제안이며 GPTQ 분석(`../gptq/`)의 분류와 맞춘 것입니다.

| 갈래 | 아이디어 | 대표 | AWQ와의 관계 |
|---|---|---|---|
| (a) 반올림 | 그룹별 min/max로 가장 가까운 격자로 | RTN, ZeroQuant | AWQ의 기준선(α = 0이면 정확히 RTN) |
| (b) 2차 보상 | 한 열을 양자화한 오차를 남은 열로 보상(Hessian) | OBQ → GPTQ | AWQ 논문이 가장 가까운 경쟁자로 둠. 보정 데이터에서 가중치 자체를 바꿈(과적합 논쟁) |
| (c) 스케일·변환 | 양자화 전에 등가 변환으로 이상치·중요 채널의 영향을 바꿈 | SmoothQuant, **AWQ**, OmniQuant, QuaRot, SpinQuant | AWQ는 채널별 스케일 하나(α 하나)만 격자 탐색하는 가장 가벼운 쪽 |
| (d) 학습형 | 양자화 파라미터를 경사로 학습 | AdaRound, BRECQ, OmniQuant, DWQ | AWQ는 "역전파 없음"을 장점으로 내세움(§1) |

핵심 새로움은 둘입니다. ① **중요 가중치 채널을 가중치가 아니라 활성으로 고른다**(§3.1, Table 1). ② **그 채널을 FP16 혼합 정밀도로 두는 대신 스케일로 보호한다**(§3.2).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 균일 그룹 양자화, zero-point | Jacob et al. 2018(정수 전용 추론), Dettmers & Zettlemoyer 2022(저비트 스케일링 법칙, 그룹의 이점) | 논문이 인용(§2, §5.1) |
| 채널별 등가 스케일(입력 ÷ s, 가중치 × s) | **SmoothQuant**(Xiao et al. 2022, arXiv:2211.10438), Wei et al. 2022b(Outlier Suppression) | SmoothQuant는 W8A8에서 활성의 이상치를 가중치로 옮김(s = max\|x\|^α / max\|w\|^(1−α)). AWQ는 같은 변환을 **가중치만 양자화하는** 설정에서 가중치 쪽 오차를 줄이는 데 씀. 두 방법의 스케일 식은 다르고(AWQ는 활성만 봄) 목적도 반대 방향의 관점: SmoothQuant는 활성이 양자화되어서, AWQ는 활성 크기가 가중치 오차의 증폭 계수라서 |
| 활성 크기로 가중치 중요도 판단 | LLM.int8()(Dettmers et al. 2022)의 이상치 특징 차원, "massive activations"(Sun et al. 2024) | 논문은 LLM.int8()을 인용(§2). 이상치 차원 분리는 FP16로 두는 쪽(혼합 정밀도), AWQ는 스케일링 |
| 반올림 오차 가정(균등, 0.25) | 양자화 잡음의 고전적 가정 | 식 2의 근거, 우리 토이로 s ≤ 1.5에서 확인(`03` §1) |
| 평균 \|x\| 기반 탐색 공간 | 논문 고유 | 단일 하이퍼파라미터 α, 격자 20 |
| 가중치 clipping | 선행 양자화 문헌 다수(MSE 기반 범위 탐색) | 논문은 한 문장으로만 언급(§3.2). 우리 모델에서는 기여가 scale만큼 큼(`03` §4.1) |
| 온더플라이 INT4 역양자화 커널, SIMD 패킹 | GPTQ 커널, FasterTransformer, exllama, llama.cpp | §2의 "시스템 지원", TinyChat 자체는 논문 고유 |

## 3. 후속 계보 (이 폴더에서 읽은 범위)

논문 v6의 서술(채택 목록, §1: Hugging Face Transformers, TensorRT-LLM, vLLM 등)과 우리가 확인한 이웃 저장소만 적습니다. 후속 논문들의 본문은 이 분석에서 읽지 않았으므로 성능 주장은 인용하지 않습니다.

- **AutoAWQ**(`casper-hansen/AutoAWQ`, MIT, 별 2,343, 마지막 push 2025-05-11): 커뮤니티 구현. GitHub에서 **보관됨**.
- **mlx-lm AWQ**(`ml-explore/mlx-lm`, MIT, 별 7,778, 마지막 push 2026-11-24): `mlx_lm.awq`가 같은 탐색 구조를 MLX 가중치 형식 위에 구현(`04` 참고 표, `03` §7). 가중치 크기 기본값 그룹 64, 격자 10.
- **llm-compressor**(`vllm-project/llm-compressor`, Apache-2.0, 별 3,998, push 2026-11-27): 이 저장소에서 읽지 않음.
- 학술 후속(OmniQuant, QuIP, QuaRot, SpinQuant, KV 양자화 KIVI 등): 목록에는 있으나(`reading_list.md` 14절 리뷰 후보) 이 분석에서 본문을 읽지 않음. AWQ 이후 흐름은 스케일(c)에서 회전(QuaRot, SpinQuant)과 학습형(OmniQuant, DWQ)으로 확장된 것으로 알려져 있으나 **이 분석에서 검증하지 않았습니다**.

## 4. 이 저장소·이웃 저장소와의 관계




읽기만 했습니다. 직접 관련은 없고, 접점은 "로컬 양자화 모델을 에이전트 백본으로 쓸 때 실패 모드가 비트 수에 따라 달라지는가" 하나입니다.

### 4.3 이 저장소의 같은 묶음 분석

| 폴더 | 관계 | 상태 |
|---|---|---|
| `../gptq/` | **직접 비교 대상.** 같은 모델(GPT-2, Pythia-160M)·같은 데이터에서 GPTQ와 RTN 비교. 스크립트와 평가 문서가 달라 수치를 직접 비교하지 않음(`03` §8). AWQ 분석은 (c) 스케일·변환의 기준 노드 | 있음 |
| `../flashattention/` | 효율 축의 다른 병목(attention의 HBM 접근). FlashAttention은 정확한 연산을 유지하며 메모리 이동을 줄이고 AWQ는 가중치 읽기를 줄임. 둘 다 "디코딩은 메모리 병목"이라는 같은 산술 강도 논증을 씀(`03` §6.2) | 있음 |
| `../pagedattention/` | KV 캐시 메모리 관리. AWQ는 가중치 4비트, KV는 FP16 그대로이므로 긴 컨텍스트에서는 KV가 지배(`03` §6.1: 7B, 컨텍스트 4096에서 KV 2.0 GiB 대 가중치 3.7 GiB) | 있음 |
| `../streamingllm/` | 어텐션 싱크. 활성 이상치 채널과 싱크 토큰의 관계는 우리 위치 0 진단(`03` §4.5)에서 AWQ 탐색에 영향이 거의 없다고 확인 | 있음 |
| `../speculative_decoding/` | 디코딩에서 가중치 읽기 횟수를 줄이는 다른 방법. 4비트 모델을 초안 모델로 쓰는 조합이 자연스러우나 이 분석에서 다루지 않음 | 있음 |
| `../sd3_rectified_flow/` | 다른 축(생성). 양자화와 직접 연결 없음 | 있음 |
| meanflow, ddpm, score_sde, ddim, classifier_free_guidance | 다른 축(생성). 직접 연결 없음 | (같은 묶음에서 작성 중) |
| `../attention_sink_emerges/`, `../edgewise/` | 어텐션 싱크의 출현, 에지 에이전트. AWQ의 "온디바이스" 동기와 연결 | 있음(읽지 않음) |

## 5. GitHub·자료 표 (2026-11-28 확인)

| 자료 | 라이선스 | 별 | 마지막 push | 메모 |
|---|---|---|---|---|
| `mit-han-lab/llm-awq` | MIT | 3,668 | 2025-07-17 | 공식 코드, `repo/`에 부분 복제(1.4 MB). 작업 트리 전체는 약 190 MB |
| `mit-han-lab/smoothquant` | MIT | 1,707 | 2024-07-12 | 읽지 않음(별과 push만) |
| `IST-DASLab/gptq` | Apache-2.0 | 2,401 | 2024-03-27 | GPTQ 공식 코드, 비교용 |
| `mit-han-lab/omniserve` | (미확인) | 868 | 2025-03-06 | 같은 연구실의 후속 서빙 시스템으로 보이나 읽지 않음 |
| `casper-hansen/AutoAWQ` | MIT | 2,343 | 2025-05-11 | 보관됨 |
| `ml-explore/mlx` | MIT | 29,310 | 2026-11-28 | 읽지 않음(별과 push만) |
| `ml-explore/mlx-lm` | MIT | 7,778 | 2026-11-24 | `mlx_lm/quant/awq.py` 읽음 |
| `ggml-org/llama.cpp` | MIT | 136,460 | 2026-11-28 | `ggml-common.h`, `ggml-quants.c` 읽음 |
| 논문 | arXiv 비독점 배포 라이선스 | | v6 2026-04-25 | `01` |
| 학회 | MLSys 2024 Poster, Best Paper Award(공식 사이트 표기) | | | `README` venue_note |
