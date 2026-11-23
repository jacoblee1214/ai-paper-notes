# Judging LLM-as-a-Judge (arXiv:2306.05685, NeurIPS 2023 D&B) 분석 노트

Zheng et al.의 *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*(v4)를 공개 판정 데이터로 재계산하고 공식 코드(FastChat `llm_judge`)와 정적으로 대조한 자료입니다. 논문 원문 PDF와 입력 데이터는 넣지 않았습니다. 논문은 arXiv 비독점 배포 라이선스라 논문 그림은 쓰지 않았고 그림은 모두 직접 그린 것입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과 위치 |
| `analysis/02_prior_work.md` | 계보, 관련 분석, 자료 표 |
| `analysis/03_analysis.md` | 일치도 재계산과 구간, 우연 보정, 편향 표 산술, MT-Bench 점수 |
| `analysis/04_code_verification.md` | 코드 대조 |
| `analysis/agreement_recompute.py` 외 두 스크립트와 `*_results.json` | 재계산과 결과 |
| `analysis/plot_figures.py` | 그림 생성 |

## 재실행

입력은 HF `lmsys/mt_bench_human_judgments`(CC BY 4.0)의 `human`, `gpt4_pair` parquet와 HF space `lmsys/mt-bench`의 `gpt-4_single.jsonl`입니다(`data/hf_mtbench/`, `data/hf_space_mtbench/`에 받아 둠).

```bash
pip install numpy scipy pyarrow matplotlib
python3 -I analysis/agreement_recompute.py
python3 -I analysis/bias_tables.py
python3 -I analysis/mtbench_scores.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 비독점 배포 라이선스(CC BY 아님). 논문 그림과 원문 PDF는 넣지 않았습니다.
- 코드: FastChat(Apache-2.0)는 읽기만 했고 넣지 않았습니다.
- 데이터: 판정 데이터 사본은 넣지 않았고 계산 결과(JSON)만 넣었습니다.
