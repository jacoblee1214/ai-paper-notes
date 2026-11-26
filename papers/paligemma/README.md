# PaliGemma (arXiv:2407.07726) 분석 노트

*PaliGemma: A versatile 3B VLM for transfer* (arXiv v2, 2024-10-10)를 식 검산, 표·그림 값 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 학회 표기 없음(기술 보고서·블로그). 논문 원문 PDF와 논문 그림은 넣지 않았습니다(PDF에 (c) Google DeepMind 표기, arXiv:2407.07726에서 받을 수 있습니다).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 우리가 그린 그래프 네 장) |
| `analysis/01_paper_analysis.md` | 방법, 단계, 실험, 한계, 판 비교 |
| `analysis/02_prior_work.md` | 계보와 자료 표 |
| `analysis/03_analysis.md` | 토큰·파라미터·FLOPs 산술, 표 1 통계, 마스크 절제, 해상도, README 대조, π0 연결 |
| `analysis/04_code_verification.md` | 공식 코드 대조 D1-D12 |
| `analysis/mask_toy.py` | prefix-LM 마스크 식 검산과 numpy 토이(KV 캐시, 정보 흐름, π0 3블록) |
| `analysis/compute_check.py`, `stats_check.py`, `readme_vs_table1.py` | 산술, 통계, README 대조 |
| `analysis/figure_extract.py` | 논문 PDF 벡터 그림에서 값 읽기(논문 PDF 필요) |
| `analysis/hparam_check.py` | 부록 J 대 공식 설정(ast 정적 대조, big_vision clone 필요) |
| `analysis/paper_tables.py` | 논문 표를 옮겨 적은 데이터 |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
curl -sL https://arxiv.org/pdf/2407.07726v2 -o paper/paper.pdf   # figure_extract.py용
git clone --depth 1 https://github.com/google-research/big_vision repo   # hparam_check.py용(커밋 0127fb6)
cd analysis && python3 mask_toy.py && python3 compute_check.py && python3 stats_check.py && python3 readme_vs_table1.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 표기되지만 PDF 첫 쪽에 (c) 2024 Google DeepMind 표기가 있어 논문 그림은 넣지 않았습니다.
- 공식 코드는 Apache-2.0, 가중치는 Gemma 약관입니다. 코드를 실행하거나 설치하지 않았고 가중치와 데이터는 받지 않았습니다.
