# StreamingLLM (arXiv:2309.17453) 분석 노트

Xiao et al.(ICLR 2024)의 StreamingLLM을 구현 모드로 분석한 노트입니다. 싱크 토큰 + 최근 윈도의 롤링 KV 캐시(RoPE는 캐시 안 순번으로 회전)를 Pythia와 GPT-2를 한 코드로 돌리는 PyTorch 모듈로 짜서(pytest 22개) dense, 윈도만, 싱크 + 윈도, 슬라이딩 윈도 + 재계산을 pile-10k 이은 스트림 24개의 perplexity로 비교했습니다. 논문 그림 1, 4, 10은 arXiv 라이선스(CC BY 4.0)에 따라 `figures/paper_fig*.png`로 넣었고 나머지는 직접 그린 그래프입니다. 원문 PDF와 공식 코드(`mit-han-lab/streaming-llm`, MIT)는 넣지 않았습니다.

## 결과 요약

- 윈도만 쓰면 모든 설정에서 24개 스트림 전부 나빠지지만(Holm p <= 0.004) 크기는 Pythia-70M 1.09배에서 GPT-2 약 80배(C=128)까지 모델마다 다릅니다. 논문 Llama-2의 350-955배 규모는 소형 Pythia에서 안 나왔습니다.
- 싱크 1개로 대부분 충분했고 4개와의 차이는 0-3%입니다(논문 표 2의 "4개 필요"는 Llama-2의 값). 8개가 4개보다 낫다는 증거는 없습니다.
- 4+윈도는 재계산 기준선과 RoPE 모델에서 0.02 nat 이내로 같습니다(Pythia-70M +0.002 [-0.008, 0.013], Pythia-160M +0.021 [0.003, 0.042]). 처음 4토큰을 줄바꿈으로 바꿔도 +-0.032 nat 이내입니다.
- "캐시 안 위치로 매겨야 한다"는 사전학습 문맥(2048)을 넘는 구간에서만 결정적입니다(Pythia-160M 7.35 -> 122). 학습된 절대 위치 GPT-2에서는 위치가 K/V에 굳어 같은 방법이 T=2048에서 재계산보다 약 18배 나쁩니다.
- 논문 표 재계산: 그림 10의 최대 22.2배는 막대 숫자와 일치, 표 2와 표 6의 Falcon-7B 4+2044가 12.12 대 12.84로 다름, 표 3은 평가 표본 1개, 표 4는 표본 오차 안.
- 사전학습 싱크 토큰(3.3절)은 0.8M 바이트 수준 토이에서 싱크가 생기지 않아 확인하지 못했습니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_claim_verification.md` | 논문 분석, 계보, 실험 결과, 이후 문헌 C1-C9와 공식 코드 정적 대조 D1-D6 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | `MiniLM`, 병렬 `attend()`, `RollingKVCache`, `nll_recompute()`와 pytest 22개 |
| `analysis/run_streaming_ppl.py`, `streaming_ppl_results.json` | 스트리밍 perplexity 격자 |
| `analysis/ablate_sink_pos.py`, `real_model_checks.py`, `sink_alpha_link.py` | 질의 위치 소거, 실제 가중치 정합성, sink 강도와의 관계 |
| `analysis/paper_tables_recalc.py` | 논문 그림 10, 표 2, 3, 4, 6, 7 재계산 |
| `analysis/sink_token_toy.py` | 사전학습 싱크 토큰 토이 |
| `analysis/crop_figures.py`, `plot_figures.py`, `figures/` | 논문 그림 3장과 우리가 그린 그래프 |

## 재실행

모델과 데이터는 로컬 캐시(`EleutherAI/pythia-14m|70m|160m`, `openai-community/gpt2`, 데이터셋 `NeelNanda/pile-10k`)를 씁니다. `crop_figures.py`는 원문 PDF(arXiv 2309.17453v4)를 `paper/paper.pdf`에 받아 두어야 합니다.

```bash
pip install torch transformers numpy scipy pandas matplotlib pyarrow pymupdf pytest
export HF_HUB_OFFLINE=1
cd analysis
python3 -m pytest test_minimal_impl.py -q
python3 run_streaming_ppl.py p14 p70 gpt2
STREAM_DEVICE=cpu python3 run_streaming_ppl.py p160 gpt2long
python3 ablate_sink_pos.py
python3 real_model_checks.py && python3 paper_tables_recalc.py && python3 sink_alpha_link.py
STREAM_DEVICE=cpu python3 sink_token_toy.py
python3 plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2309.17453v4, arXiv 라이선스 CC BY 4.0. 원문에 회사 저작권 고지가 없어 도식·차트 3장만 넣었습니다.
- 공식 코드는 MIT이며 읽기만 했고 복사하거나 실행하지 않았습니다. 이 폴더의 구현은 논문 설명으로 직접 짰습니다.
