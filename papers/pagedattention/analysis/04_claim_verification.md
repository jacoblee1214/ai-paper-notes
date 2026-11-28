# 04. 주장 검증과 공식 코드 대조 - PagedAttention / vLLM (arXiv:2309.06180)

구현 모드의 04입니다. C는 주장 검증(우리 계산과 시뮬레이션, 가능하면 이후 문헌), D는 논문 알고리즘과 **현재** 공식 코드의 차이입니다. 코드 기준은 `vllm-project/vllm` HEAD `316c6c0d72fd6c26f083d433f93fe5232f87869e`(2026-11-27, 정적 읽기만). 이 코드는 논문 시점(2023)보다 훨씬 진화한 V1 엔진이라 D의 차이 대부분은 오류가 아니라 **진화**이고, 각 항목에 원인을 구분해 적었습니다. 이후 문헌은 이번에 웹 검색을 쓰지 않아 직접 확인하지 못했습니다(Crossref 제목·학회 확인만).

## 주장 검증 C1–C8

| # | 논문 주장(위치) | 판정 | 근거 |
|---|---|---|---|
| C1 | 기존 시스템은 KV 메모리의 20.4–38.2%만 실제 토큰에 쓴다(그림 2) | **조건부 재현** | 그림 11의 평균만 쓴 절단 로그정규 길이로 buddy 포함 Orca 세 변형을 구현하면 20.3, 28.7, 37.9%(`03` §2). Max의 8.9%는 풀 자투리 산술로 정확히 설명(8.85%). Orca는 저자 재구현이고 Pow2·Oracle의 외부 단편화는 buddy 가정과 길이 분산에 민감. 원자료(ShareGPT)로는 시험하지 않음 |
| C2 | vLLM은 낭비가 거의 없다(96.3%, 그림 2). 낭비는 요청당 마지막 블록 하나 안으로 제한(4.3절) | **재현** | 우리 vLLM 96.5% [96.4, 96.6], 요청당 쪽 안 낭비가 (B−1)/2 슬롯(7.5 대 이론 7.5, `03` §3), 테스트로 낭비 < B 확인 |
| C3 | OPT-13B KV는 토큰당 800 KB, 2048 토큰이면 1.6 GB(3절) | **재계산 일치(단위 혼용)** | 819,200 B = 800 KiB, 1.5625 GiB(= 1.68 GB). 표 1의 GB는 GiB로 읽어야 슬롯 수와 맞음(`03` §1) |
| C4 | 병렬 샘플링 6.1–9.8%, 빔 서치 37.6–55.2% 메모리 절약(그림 15, Alpaca) | **메커니즘만 확인, 수치는 미재현** | CoW와 참조 횟수는 테스트로 확인. 병렬 샘플링 수치는 우리가 논문의 절반(3.0–5.7%), 빔은 입력 가정(부모 분포)에 지배됨(`03` §5) |
| C5 | 같은 지연에서 처리량 2–4배(초록, 6절) | **미검증(GPU 필요)** | 본문 배수 2.2배, 4.3배는 그림 13 막대와 일치. 메모리 풀에 더 들어가는 요청 수(우리 2.6배 대 논문 2.2배)만 지지. 곡선과 지연은 재지 않음 |
| C6 | 블록 크기 16이 대부분의 워크로드에서 충분히 크고 낭비가 작다(7.2절) | **부분 확인** | 낭비 쪽은 우리 확인(ShareGPT B=16에서 B=1 대비 토큰 상태 −1.6p, Alpaca −8.2p, `03` §3). "충분히 크다"는 커널 효율이라 확인 불가. 현재 코드는 기본값이 고정 16이 아님(D6) |
| C7 | 재계산 오버헤드는 스와핑 지연의 20%를 넘지 않고 블록 16–64에서 둘이 비슷(7.3절, 그림 19) | **방향만 확인(가정 상수)** | 비용 토이에서 교차가 B≈8–16이고 방향 일치. 20% 수치는 재지 않음. 이후 코드는 재계산 경로만 남아 있음(D3) |
| C8 | PagedAttention은 같은 모델 정확도에 영향이 없다(1절) | **수치 동일성 확인** | 쪽 단위 세 커널이 GPT-2 실제 K/V에서 밀집 어텐션과 일치(`03` §4). 훈련·정확도 영향은 구조상 없음(같은 식) |

## 공식 코드 대조 D1–D8

논문과 가장 가까운 현재 파일: 블록 풀 `vllm/v1/core/block_pool.py`, 관리자 `vllm/v1/core/kv_cache_manager.py`와 `single_type_kv_cache_manager.py`, 스케줄러 `vllm/v1/core/sched/scheduler.py`, 쓰기 커널 `csrc/cache_kernels.cu`, 어텐션 호출 래퍼 `vllm/v1/attention/ops/paged_attn.py`. 논문 시점의 `vllm/core/block_manager.py`는 이 HEAD에 없습니다.

| # | 논문 | 현재 코드 | 차이의 원인 |
|---|---|---|---|
| D1 | 물리 블록 자유 목록 + 블록별 참조 횟수(4.4절) | `KVCacheBlock.ref_cnt`와 `FreeKVCacheBlockQueue`(이중 연결 목록, 블록 중간 O(1) 제거), `get_new_blocks`가 `popleft_n`으로 받아 `ref_cnt=1`로 설정, `free_blocks`가 감소시켜 0이면 자유 큐로 | **진화**: 접두사 캐싱을 위해 해제된 블록도 해시로 살려 두고 LRU로 축출하기 때문(큐 순서 주석: 요청 블록을 뒤집어 해제해 꼬리부터 축출). 논문의 단순 FIFO 자유 목록(우리 `BlockAllocator`)과 의미는 같고 정책이 추가됨 |
| D2 | 중앙 스케줄러의 KV 캐시 관리자가 블록 표를 갖고 모든 작업자에 방송(4.6절) | `KVCacheManager`가 스케줄러(EngineCore)에 있고 스케줄러가 새로 할당된 블록 ID(`new_block_ids`)를 작업자에 보냄 | **유지**: 구조는 같고 전체 블록 표를 매 단계 방송하던 것이 증분 전달로 바뀐 것으로 보임(증분 여부의 작업자 쪽 코드는 미확인) |
| D3 | 선점: 전부 또는 전무, 가장 늦은 요청부터 축출, **스와핑 또는 재계산**(4.5절) | `_preempt_request`: 블록을 전부 해제, `num_computed_tokens = 0`, 상태 `PREEMPTED`, 대기열 맨 앞에 되돌림. 희생자는 `running[-1]`(FCFS) 또는 우선순위 정책에서 `max(priority, arrival_time)`. 스케줄러에서 스와프 경로를 찾지 못함 | **진화(축소)**: V1은 재계산만 남김. `swap_blocks` 커널은 `csrc/cache_kernels.cu`에 남아 있으나 스케줄러가 쓰는지는 미확인. 재계산이 접두사 캐시 적중과 만나면 실제 재계산량이 줄 수 있음(우리 토이에는 없음) |
| D4 | `fork`/`append`/`free`와 블록 단위 CoW로 병렬 샘플링·빔 서치 지원(4.4, 5.2절) | 코어 `vllm/v1/core`에 `fork` API와 빔 서치가 없음(`beam` 문자열 검색에서 파일 없음). CoW는 `_apply_cow`, `take_pending_cow_copies`, `is_block_writable`(`ref_cnt == 1` 이고 해시 없음)로 접두사 부분 적중과 상태 공간 모델용으로만 보임 | **진화**: 다중 샘플·빔은 시퀀스 묶음이 아니라 요청 단위와 상위 계층으로 옮겨진 것으로 보이나 그 경로는 확인하지 못함(미확인). 논문 4.4절의 시퀀스 그룹 단위 공동 선점과도 다름 |
| D5 | 프롬프트 단계에 ⌈n/B⌉ 블록, 이후 마지막 블록이 차면 새 블록(그림 6) | `get_num_blocks_to_allocate`의 `cdiv(num_tokens, block_size)`와 `allocate_new_blocks`의 `num_required_blocks − len(req_blocks)` | **유지**(같은 산술). 추가: 슬라이딩 윈도와 청크 로컬 어텐션용 승인 상한, 추측 디코딩용 lookahead 토큰, 접두사 적중 블록 선 할당 |
| D6 | 기본 블록 크기 16(7.2절) | `CacheConfig.block_size`가 `Field(default=None)`이고 `resolve_kv_cache_block_sizes` 등이 계산. 여러 KV 그룹(하이브리드 모델)이 서로 다른 블록 크기를 가질 수 있음 | **진화**: 백엔드(어텐션 커널)와 모델 종류가 크기를 정함. 현재 기본값이 몇인지는 이 clone에서 확인하지 못함 |
| D7 | 자체 PagedAttention 커널(FasterTransformer 어텐션 변형, 워프가 블록 하나, 5.1절) | `csrc/attention/`에 `attention_kernels.cuh`가 없고 `merge_attn_states.cu`, MLA, octave 계열만 있음. 어텐션 래퍼 `paged_attn.py`는 K 캐시를 `[num_blocks, kv_heads, head_size/x, block_size, x]`(x = 16 B/원소 크기)로 보고 `reshape_and_cache`를 호출. 쓰기 커널은 `reshape_and_cache`, `reshape_and_cache_flash`가 `cache_kernels.cu`에 있음 | **진화**: 원래 커널은 FlashAttention 계열 등 외부 백엔드로 대체되고 vLLM에는 캐시 쓰기와 레이아웃이 남음(대체 경로의 커널 본문은 sparse checkout에 없어 미확인) |
| D8 | 블록 복사 일괄 커널(5.1절 ③) | `csrc/cache_kernels.cu`에 `swap_blocks`, `swap_blocks_batch`(블록 매핑을 CPU 텐서로 받음)가 있음. `copy_blocks` 함수 이름은 이 파일 검색에서 확인하지 못함(주석에 `copy_blocks_kernel` 패턴 언급) | 진화(정확한 위치 미확인) |

## 한 줄 결론

핵심 아이디어(고정 크기 블록 + 블록 표 + 참조 횟수)는 현재 코드에서 그대로 살아 있고 관리 단위는 접두사 해시와 하이브리드 모델 때문에 훨씬 복잡해졌습니다. 논문의 **스와핑 선점**과 **시퀀스 그룹 CoW(빔 서치)**는 현재 코어에서 찾지 못했고, 이것을 근거로 현재 vLLM의 동작을 설명하면 안 됩니다.
