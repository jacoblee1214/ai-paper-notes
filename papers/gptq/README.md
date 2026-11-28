# GPTQ (arXiv:2210.17323) 분석 노트

Frantar et al.(ICLR 2023)의 GPTQ를 구현 모드로 분석한 노트입니다. 선형층 하나짜리 PyTorch 구현(`gptq()`, `rtn()`, 순차 OBQ, 탐욕 OBQ; pytest 31개)이 같은 열 순서의 순차 OBQ와 float64에서 1e-8 이내로 일치함을 확인하고, GPT-2 small과 Pythia-160M에 블록 단위 순차 GPTQ와 RTN(4·3비트, 행별 또는 g128)을 적용해 pile-10k held-out 문서 150개로 비교했습니다. 논문 그림 1, 2는 arXiv 라이선스(CC BY 4.0)에 따라 `figures/paper_fig*.png`로 넣었고 나머지는 직접 그린 그래프입니다. 원문 PDF와 공식 코드(`IST-DASLab/gptq`, Apache-2.0)는 넣지 않았습니다.

## 결과 요약

- GPTQ는 8개 설정 중 7개에서 RTN보다 분명히 낮은 perplexity입니다(같은 문서 짝 비교 Holm p <= 2e-4). 3비트 행별에서 GPT-2 37.2 대 260, Pythia-160M 140 대 672. GPT-2 4비트 g128은 경계(Holm 0.031).
- 그러나 4비트도 FP32보다 유의하게 나쁩니다(GPT-2 +4-7%, Pythia-160M +26-44%). 논문의 "거의 무손실"은 175B의 관찰입니다.
- 층 24개 조합 모두 GPTQ < RTN이고, held-out 입력의 층 오차는 보정 입력보다 24-44% 큽니다(일부 층).
- 논문 4절의 "탐욕 순서는 이득이 작다"는 작은 합성 층에서 반대 방향이었고(우리 탐욕 OBQ가 더 나쁨, 보조 휴리스틱 없음), 후속 act-order는 3비트에서 perplexity를 6%(GPT-2)-36%(Pythia) 낮췄습니다.
- dampening은 수치 안정화이자 정규화입니다. 표본이 열보다 적으면 lambda = 0은 Cholesky가 실패했고, 언어 모델 설정에서는 1e-3과 1e-2가 구분되지 않았습니다.
- 논문 산술은 대부분 일치합니다(OPT-175B 3비트 62.0 GiB + KV 9.0 GiB, 표 6 속도비, 그룹 오버헤드 0.02와 0.15비트, 175B 4.2시간은 FP32 피크 2.6시간에 효율 61%). 사소한 불일치: OPT-125M FP16 27.65(표 3) 대 27.66(표 8).
- 16 GiB GPU 한도(RAM의 2/3 가정)에서 4비트 g128 가중치는 KV 2 GiB 포함 약 25.8 B 파라미터까지입니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_code_verification.md` | 논문 분석, 계보, 식 유도와 실험 결과, 공식 코드 정적 대조 D1-D11과 주장 검증 C1-C8 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | GPTQ, RTN, 순차 OBQ, 탐욕 OBQ와 pytest 31개 |
| `analysis/layer_checks.py` | 층 하나 실험(합성, dampening, act-order, GPT-2 실제 층, 그룹 격자 시점, 시간) |
| `analysis/lm_quant.py`, `run_lm_quant.py`, `analyze_lm.py` | 블록 단위 순차 양자화, 설정 격자, 부트스트랩·Holm 분석 |
| `analysis/paper_arithmetic.py` | 논문 표, 메모리, 시간, 24 GB 산술 재계산 |
| `analysis/crop_figures.py`, `figures/` | 논문 그림 2장과 우리가 그린 그래프 |
| `analysis/*_results.json`, `lm_quant_nll.json` | 위 스크립트의 결과와 문서별 NLL |

## 재실행

모델과 데이터는 `openai-community/gpt2`, `EleutherAI/pythia-160m`, 데이터셋 `NeelNanda/pile-10k`를 씁니다(허브 캐시, `HF_HUB_OFFLINE=1`). `crop_figures.py`는 원문 PDF(arXiv 2210.17323v2)를 `paper/paper.pdf`에 받아 두어야 합니다.

```bash
pip install torch transformers numpy scipy pandas matplotlib pyarrow pymupdf pytest
export HF_HUB_OFFLINE=1
cd analysis
python3 -m pytest test_minimal_impl.py -q
python3 layer_checks.py
python3 run_lm_quant.py gpt2 fp rtn_b4_g0 rtn_b4_g128 rtn_b3_g0 rtn_b3_g128 gptq_b4_g0 gptq_b4_g128 gptq_b3_g0 gptq_b3_g128 gptq_b4_g0_act gptq_b3_g0_act gptq_b3_g0_d0.001 gptq_b3_g0_s1 gptq_b3_g0_s2 gptq_b4_g0_s1 gptq_b4_g0_s2
python3 run_lm_quant.py p160 fp rtn_b4_g0 rtn_b4_g128 rtn_b3_g0 rtn_b3_g128 gptq_b4_g0 gptq_b4_g128 gptq_b3_g0 gptq_b3_g128 gptq_b3_g0_act
python3 analyze_lm.py && python3 paper_arithmetic.py && python3 crop_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2210.17323v2, arXiv 라이선스 CC BY 4.0. 원문에 회사 저작권 고지가 없어 도식과 차트 2장만 넣었습니다.
- 공식 코드는 Apache-2.0이며 읽기만 했고 복사하거나 실행하지 않았습니다. 이 폴더의 구현은 논문 설명으로 직접 짰습니다.
