# Who&When (arXiv:2505.00212, ICML 2025 스포트라이트) 분석 노트

*Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems*를 표 재계산, 공개 데이터 통계, 단순 기준선 비교, 공식 저장소 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다.

## 재현 가능성

- 모델은 호출하지 않았습니다. 공개 표와 공식 저장소의 공개 데이터(JSON 184개)만 썼습니다.
- 데이터와 코드는 이 폴더에 넣지 않았습니다. 공식 저장소(MIT)를 얕게 clone하면 `analysis/data_stats.py`를 다시 돌릴 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 문제, 정식화, 데이터셋, 방법, 주장과 근거 대응, 한계 |
| `analysis/02_prior_work.md` | 계보, 이후 연구, 라벨 재검토, 저장소 정보 |
| `analysis/03_analysis.md` | 공개 데이터 통계, 표 분모, Random 행, 기준선 비교, 검정 |
| `analysis/04_code_verification.md` | 공식 저장소와 데이터 정적 대조(D1–D10) |
| `analysis/data_stats.py` | 공개 데이터 통계와 단순 기준선(저장소 clone 필요) |
| `analysis/recompute_tables.py` | 표 재계산과 검정(scipy) |
| `analysis/plot_figures.py` | 그래프 세 장(matplotlib) |

## 재실행

```bash
git clone --depth 1 https://github.com/mingyin1/Agents_Failure_Attribution repo   # 커밋 b2bae5c
python3 analysis/data_stats.py
python3 analysis/recompute_tables.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지 확인). 이번 게시본에는 논문 그림을 넣지 않았고, 그래프 세 장은 우리가 그렸습니다.
- 공식 저장소는 MIT입니다. 코드와 데이터는 넣지 않았습니다.
