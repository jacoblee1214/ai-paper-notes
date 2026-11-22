# CoT 충실성 측정 (arXiv:2307.13702) 분석 노트

*Measuring Faithfulness in Chain-of-Thought Reasoning*(Lanham et al., 2023)을 논문 PDF의 벡터 그림에서 추출한 값의 재계산, 정확도 하한 유도, numpy 토이, 이후 문헌 정리로 검증한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다.

## 재현 가능성

- 모델은 호출하지 않았습니다. 원 실험은 비공개 모델이라 공식 코드와 결과 파일이 없습니다.
- `analysis/figure_values.json`은 논문 PDF에서 추출한 곡선 값입니다. `extract_figures.py`를 다시 돌리려면 arXiv PDF(`https://arxiv.org/pdf/2307.13702`)를 `paper/paper.pdf`로 받아 두어야 합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 문제, 방법, 주장과 근거 대응, 한계 |
| `analysis/02_prior_work.md` | 계보와 이후 연구 |
| `analysis/03_analysis.md` | 재계산, 정확도 하한, 토이, 표본 오차 |
| `analysis/04_claim_verification.md` | 주장별 판정(C1–C7)과 인용 판정 |
| `analysis/*.py`, `*_results.json` | 추출, 재계산, 토이, 그래프 |

## 재실행

```bash
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2307.13702
python3 analysis/extract_figures.py
python3 analysis/recompute_metrics.py
python3 analysis/toy_truncation.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(nonexclusive-distrib 1.0)라 논문 그림을 넣지 않았습니다. 그래프 세 장은 우리가 그렸습니다.
