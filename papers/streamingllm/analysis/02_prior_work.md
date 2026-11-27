# 02. 계보, 구성 요소의 출처, 이 저장소의 다른 분석과의 관계

확인일 2026-11-28. arXiv 번호·제목·제출일은 arXiv API로 확인했고, 학회 표기는 이 문서에서 다루지 않습니다. 후속 논문의 내용은 **초록과 제목 수준**으로만 적었고 본문은 읽지 않았습니다(읽은 것은 표시).

## 1. 구성 요소별 출처

| 구성 요소 | 출처 | StreamingLLM에서의 역할 |
|---|---|---|
| 슬라이딩 윈도(최근 L개만 attention) | Longformer(arXiv:2004.05150)의 지역 윈도, Sparse Transformer(arXiv:1904.10509) | 기준선 "window attention"(논문 그림 1b). 논문 부록 B가 직접 이 계열과의 차이를 설명 |
| 전역 토큰 + 지역 윈도 | Longformer·BigBird(arXiv:2007.14062)·ETC의 전역 attention | 구조는 닮았다(고정 몇 토큰 + 윈도). 논문은 이들이 양방향용이고 사전학습 모델에 적용할 수 없다고 구분(부록 B) |
| RoPE | RoFormer(arXiv:2104.09864) | 키를 회전 전에 저장하고 캐시 안 위치로 회전하는 이유(상대 거리만 보므로). 공식 코드는 Llama·GPT-NeoX·Falcon을 위해 attention 순전파를 바꿔치기(`pos_shift/`) |
| ALiBi | arXiv:2108.12409 | 캐시 안에서 연속된 선형 편향을 쓰도록 적용(MPT 실험) |
| 길이 외삽 / 컨텍스트 확장 | Position Interpolation(arXiv:2306.15595) 등 | 직교(orthogonal)라고 논문이 규정. 캐시의 "최근 윈도"를 넓히는 데 같이 쓸 수 있음 |
| 슬라이딩 윈도 + 재계산 | 논문이 정한 오라클 기준선 | 그림 1c. 정확도는 높지만 O(TL²) |
| attention의 "남는 질량" 처리 | Miller, "Attention Is Off by One"(블로그, 2023), Quantizable Transformers(arXiv:2306.12929) | 식 2의 SoftMax₁ = "Zero Sink". 후자는 같은 현상을 양자화 이상치 관점에서 봄 |
| 토큰 하나를 비워서 attention을 흡수시킴 | Vision Transformers Need Registers(arXiv:2309.16588)(거의 같은 시기) | 논문 부록 B가 "register와 sink는 비슷하지만 위치가 다르다"고 구분 |

## 2. 같은 시기·이후의 가까운 방법

| 방법 | 핵심 | StreamingLLM과 다른 점 | 확인 수준 |
|---|---|---|---|
| LM-Infinite(arXiv:2308.16137, 2023-08-30) | Λ자 마스크(처음 몇 토큰 + 윈도)와 거리 상한으로 길이 일반화 | 구조가 거의 같다. 논문 부록 B가 동시 연구로 언급, 차이는 sink 현상 자체를 밝히고 사전학습 싱크를 제안한 것 | 초록 |
| H2O(arXiv:2306.14048), Scissorhands(arXiv:2305.17118) | 중요도(누적 attention)가 큰 토큰을 KV에서 남김 | 고정된 "첫 몇 토큰"이 아니라 데이터에 따라 정함. 초기 토큰이 자주 남을 것이라는 해석은 추측(본문 미확인) | 초록 |
| FastGen(arXiv:2310.01801) | head마다 다른 압축 정책을 프로파일링해 적용 | head 종류별 정책(특수 토큰, 지역, 빈도) | 초록 |
| SnapKV(arXiv:2404.14469) | 프롬프트 끝의 관찰 창으로 head별 중요 KV를 골라 압축 | 프롬프트 압축(스트리밍이 아님) | 초록 |
| InfLLM(arXiv:2402.04617) | 캐시 밖 컨텍스트를 블록 단위 외부 메모리에 두고 검색 | StreamingLLM이 못 하는 "캐시 밖 정보 사용"을 시도 | 초록 |
| RazorAttention(arXiv:2407.15891), DuoAttention(arXiv:2410.10819) | head를 retrieval head / streaming head로 나눠 streaming head에는 싱크 + 윈도만 둠 | StreamingLLM 패턴을 head 단위로 쓰고 일부 head는 전체 캐시 유지 | 초록 |

## 3. sink 현상 자체를 다룬 이후 연구 (이 저장소와 연결)

| 논문 | 무엇을 바꿨나 | 이 저장소의 분석 |
|---|---|---|
| Massive Activations in LLMs(arXiv:2402.17762) | sink가 되는 토큰의 은닉 상태에 아주 큰 값이 있다고 설명(초록) | 이 저장소의 `spike_sparse_sink/`가 다룸 |
| When Attention Sink Emerges(arXiv:2410.10781, ICLR 2025) | 출현 조건(데이터, 손실, 위치 임베딩, 정규화)과 sink = key bias 해석. softmax 정규화를 없애면 sink가 안 생긴다 | `attention_sink_emerges/`(공개 가중치 측정 포함) |
| The Spike, the Sparse and the Sink(arXiv:2603.05498, ICML 2026) | massive activation과 sink의 관계 해부 | `spike_sparse_sink/` |
| Gated Attention for LLMs(arXiv:2505.06708) | attention 출력에 게이트를 넣어 sink 없는 모델을 만든다고 보고(초록) | 논문 3.3절의 "사전학습 단계에서 해결"이라는 방향의 후속 |
| gpt-oss 모델 카드(arXiv:2508.10925) | head별로 학습되는 sink 항을 softmax 분모에 넣는 구조를 쓴다고 알려짐 | 본문을 읽지 않음. 식 2(SoftMax₁)의 학습되는 일반화로 보임(추측) |
| StreamingVLM(arXiv:2510.09608) | 무한 비디오 스트림에 sink + 윈도 KV 캐시를 적용 | 초록만 확인. VLM 쪽 적용 사례 |

이 저장소에서의 위치: StreamingLLM은 sink라는 이름을 처음 붙인 논문이고, 이후 연구는 (1) 왜 생기는가(`attention_sink_emerges/`, `spike_sparse_sink/`), (2) 어떻게 이용하는가(KV 압축 계열), (3) 구조로 없애는가(게이트, 학습되는 분모 항)로 갈라집니다.

## 4. 이 저장소·다른 저장소의 분석과의 관계

- **`../attention_sink_emerges/`**: 같은 현상의 출현 조건 논문. 우리는 그쪽의 공개 가중치 측정(`analysis/sink_real_models_results.json`: 처음 위치가 받는 head 평균 attention α₁, T=1024, 문서 200개)을 그대로 읽어 왔습니다. Pythia-14M 0.017, 70M 0.029, 160M 0.073, GPT-2 0.355(균일 기준선 0.0073). 이 값과 `03`에서 잰 "윈도만 쓸 때 perplexity가 얼마나 터지는가"의 순서가 같은지 봅니다(`03` §3.5: GPT-2에서는 맞고 Pythia 세 모델 사이에서는 맞지 않음).
- **`../spike_sparse_sink/`**: sink와 massive activation의 관계. StreamingLLM의 "초기 토큰이 sink"가 어떤 은닉 상태 구조에서 오는지의 설명 쪽입니다. 여기서는 중복해서 측정하지 않았습니다.
- **`../flashattention/`**: 정확한 attention의 메모리 이동을 줄이는 방법(훈련·prefill에 적용). StreamingLLM은 어떤 K, V를 남길지 바꾸는 근사(디코딩 KV 캐시)라 직교하며, 축 6(효율·시스템)에서 "정확한 계산의 I/O"와 "남기는 토큰 선택"의 두 갈래를 이룹니다.
- **`../pagedattention/`**: (같은 묶음에서 작성 중). KV 캐시의 메모리 배치를 다루는 논문이라 StreamingLLM의 "무엇을 남기나"와 짝입니다.
- **`../speculative_decoding/`**: (같은 묶음에서 작성 중). 디코딩 지연을 줄이는 다른 갈래.
- **`../gptq/`, `../awq/`**: (같은 묶음에서 작성 중). 가중치 양자화로, KV 캐시를 건드리는 StreamingLLM과 겹치지 않고 함께 쓸 수 있는 갈래입니다.

## 5. GitHub·자료 표 (2026-11-28, `gh api`)

| 저장소 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|
| `mit-han-lab/streaming-llm` | MIT | 7,274 | 2024-07-11 | 공식. HEAD `2e50426`. `repo/`에 얕게 clone, 정적으로만 읽음 |
| `tomaarsen/attention_sinks` | Apache-2.0 | 734 | 2024-04-10 | 비공식 HF 호환 구현(확인한 것은 메타데이터뿐) |
| `FMInference/H2O` | 명시 없음 | 542 | 2024-08-01 | 가까운 대안 |
| `Glaciohound/LM-Infinite` | MIT | 152 | 2025-03-13 | 동시 연구의 공식 코드 |
| `mit-han-lab/duo-attention` | MIT | 544 | 2025-02-10 | 후속(같은 그룹) |
| `FasterDecoding/SnapKV` | Apache-2.0 | 340 | 2025-07-10 | 가까운 대안 |

공식 코드의 구성(`streaming_llm/`): `kv_cache.py`의 `StartRecentKVCache`(처음 `start_size`개 + 마지막 `recent_size`개를 이어 붙여 자르기), `enable_streaming_llm.py`(모델 종류별 분기), `pos_shift/modify_{llama,gpt_neox,falcon}.py`(회전 전 키를 캐시, 키는 캐시 안 순번으로 회전). 논문과 코드의 대조는 `04`.
