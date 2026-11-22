# Non-Determinism of "Deterministic" LLM Settings (arXiv:2408.04667, Eval4NLP 2025 워크숍) 분석 노트

*Non-Determinism of "Deterministic" LLM Settings* (Atil et al., arXiv v5)를 공개 실행별 원출력의 재계산, 과분산 검정, 정규성 재검정, 공식 저장소 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2408.04667에서 받을 수 있습니다). 논문 그림은 차트 두 장(그림 1, 3)만 넣었고 출처를 적었습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 주장과 근거 위치, 판 사이의 차이 |
| `analysis/02_prior_work.md` | 계보, 이후 연구, 저장소 정보 |
| `analysis/03_analysis.md` | 식 검산, 표 2·3 재계산, 과분산 검정, 정규성 재검정, 배치 불변성 토이 |
| `analysis/04_code_verification.md` | 공식 저장소와 논문의 대조(D1–D10) |
| `analysis/*.py`, `*_results.json` | 재계산 스크립트와 결과 |
| `report/figures/` | 우리가 그린 그래프 세 장과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

입력 파일은 이 폴더에 없습니다. 공식 저장소 `github.com/breckbaldwin/llm-stability`(커밋 c089c02)를 clone하고 `experiments/v3/runs.tgz`를 `data/v3/`에, `experiments/v2/runs.tgz`를 `data/v2/`에 풉니다(풀면 약 2.2 GB). 표 대조에는 `paper/paper.txt`(arXiv v5를 `pdftotext -layout`으로 변환)가 필요합니다.

```bash
python3 -I analysis/recompute_runs.py
python3 -I analysis/table_check.py
python3 -I analysis/overdispersion.py
python3 -I analysis/normality_check.py
python3 -I analysis/batch_invariance_toy.py   # torch 필요
python3 -I analysis/make_figures.py
```

## 출처와 라이선스

- 논문: arXiv 라이선스 CC BY 4.0(abs 페이지 확인). 원문 PDF에 저작권 표시가 없습니다. 논문 그림은 "© 저자, CC BY 4.0, 출처 arXiv:2408.04667"로 표기했습니다.
- 공식 저장소는 Apache-2.0입니다(GitHub API). 코드와 원출력은 이 폴더에 넣지 않았고 공식 코드는 실행하지 않았습니다.
