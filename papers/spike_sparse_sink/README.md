# The Spike, the Sparse and the Sink (arXiv:2603.05498, ICML 2026) 분석 노트

*The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks* (arXiv v1, 2026-03-05; PMLR 제목은 "Anatomy of Massive Activations and Attention Sinks")를 표·식 재계산, 인과 주장의 근거 정리, 공식 코드 정적 대조, 작은 토이로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다(arXiv:2603.05498, github.com/savinasun/SpikeSparseSink). 공식 코드는 라이선스 파일이 없고 실행하지 않았습니다. 논문 그림은 그림 1 한 장만 넣었고 출처를 적었습니다.

## 요약

- 인과 주장의 근거는 시드 1개의 7B 사전학습 표 6개이고, spike(massive activation)의 정의와 측정 코드가 없습니다.
- 식 (21)은 틀렸고(올바른 형태 λ d cos²) 표 9의 학습 스텝 5,000은 오기로 보입니다.
- 논문 규모 재현은 하지 않았습니다. 작은 토이는 `analysis/03_analysis.md` §6.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 문제, 주장, 방법과 근거 위치, 표 요약 |
| `analysis/02_prior_work.md` | 계보, 대안 설명, 발표 이후 연구, 자료 표 |
| `analysis/03_analysis.md` | 정의·식 검산, 표 재계산, 인과 주장의 근거, 토이 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조 (D1–D9) |
| `analysis/recompute_tables.py` | 표·식 재계산 (numpy, scipy, 수 초) |
| `analysis/toy_spike_sink.py` | 4층 바이트 LM 토이 (PyTorch, MPS 또는 CPU) |
| `analysis/plot_figures.py` | 그래프 생성 |
| `report/figures/` | 우리가 그린 그래프와 논문 그림 1 (저자 작성, CC0 1.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
python3 analysis/recompute_tables.py
python3 analysis/plot_figures.py
# 토이: torch 필요. 영문 텍스트 파일(수 MB)을 TOY_TEXT로 지정
TOY_TEXT=corpus.txt TOY_LR=6e-3 python3 analysis/toy_spike_sink.py 4000 0,1,2
```

## 출처와 라이선스

- 논문은 arXiv에 CC0 1.0으로 공개되어 있습니다(abs 페이지 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 논문 그림 1(차트)은 출처와 함께 실었습니다("저자 작성, CC0 1.0, 출처 arXiv:2603.05498").
- 공식 코드와 가중치, 데이터는 받아서 넣지 않았고, 코드는 라이선스 파일이 없어 복사하지 않았습니다.
