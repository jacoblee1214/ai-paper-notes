# When Attention Sink Emerges in Language Models (arXiv:2410.10781, ICLR 2025) 분석 노트

Gu et al.의 *When Attention Sink Emerges in Language Models: An Empirical View*(arXiv v2, 2025-03-02)를 논문 표 값 재계산, 지표 식 검산, 공식 코드(`sail-sg/Attention-Sink`) 정적 대조, 직접 짠 소형 토이, 공개 가중치(Pythia, GPT-2) 재측정으로 분석한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다. 논문은 arXiv 비독점 배포 라이선스(CC BY 아님)라 논문 그림도 쓰지 않았고, 그림은 모두 직접 그린 것입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 지표, 결과 위치, 한계 |
| `analysis/02_prior_work.md` | 계보(StreamingLLM에서 Gated Attention까지), 자료 표 |
| `analysis/03_analysis.md` | 지표 식 검산, 표 재계산, 시드 변동 대용 척도, 토이, 공개 가중치 재측정 |
| `analysis/04_code_verification.md` | 코드 대조(K1–K10, D1–D11), 재현 제안 |
| `analysis/metric_check.py`, `tables_recompute.py`, `sink_toy.py`, `sink_real_models.py`와 `*_results.json` | 계산과 결과 |
| `analysis/plot_figures.py`, `report/figures/` | 직접 그린 그림 5장 |

## 재실행

토이와 공개 가중치 재측정은 torch(CPU 또는 MPS/CUDA)가 필요하고(후자는 transformers, pandas, pyarrow도) 나머지는 numpy입니다. 토이는 파이썬 표준 라이브러리 소스를 바이트 코퍼스로 읽으므로 다운로드가 없습니다(시드 3개 x 변형 3개 = 9회, 한 회당 약 2분).

```bash
pip install numpy matplotlib torch transformers pandas pyarrow
python3 -I analysis/metric_check.py
python3 -I analysis/tables_recompute.py
python3 -I analysis/sink_toy.py
python3 -I analysis/sink_real_models.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 비독점 배포 라이선스(CC BY 아님). 논문 그림과 원문 PDF는 넣지 않았습니다.
- 코드: `sail-sg/Attention-Sink`(MIT)는 읽기만 했고 넣지 않았습니다.
- 공개 가중치 재측정: `EleutherAI/pythia-14m`, `pythia-70m`, `pythia-160m`, `openai-community/gpt2`(Hugging Face 캐시에 받아 두어야 하며 이 폴더에는 넣지 않음)와 데이터셋 `NeelNanda/pile-10k`를 씁니다. 라이선스는 각 모델·데이터셋 카드를 확인하세요(이 작업에서 확인하지 않음).
- 데이터: 논문 표 값은 직접 옮겨 적었고, 토이 코퍼스는 설치된 파이썬 표준 라이브러리입니다.
