# AgenTracer (arXiv:2509.03312, ICLR 2026 포스터) 분석 노트

*AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?*를 표 재계산, 공개 데이터 릴리스 계산, 공식 저장소 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스).

## 재현 가능성

- 가중치는 공개되지 않았고(저장소 README), 학습 코드도 없습니다. 재현은 불가능합니다.
- 표 재계산과 공개 데이터(v1.0.0) 통계는 이 폴더의 스크립트로 다시 돌릴 수 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 주장과 근거 대응 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, 공식 저장소 정보 |
| `analysis/03_analysis.md` | 표 셀의 분모 검사, 주장 재현, Fisher 검정, 공개 데이터 통계 |
| `analysis/04_code_verification.md` | 공식 코드와 논문 방법의 대조 |
| `analysis/recompute_tables.py` | 표 재계산과 주장 검색(scipy) |
| `analysis/data_*.py` | 공개 데이터 릴리스 통계, 라벨 일치, 질문 중복 검사 |

## 재실행

```bash
python3 analysis/recompute_tables.py
python3 analysis/data_release_stats.py   # data/release/ 에 릴리스 zip을 풀어 둔 뒤
python3 analysis/data_release_steps.py
python3 analysis/data_label_consistency.py
python3 analysis/data_overlap_check.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스입니다(arXiv abs 페이지 확인). CC BY가 아니므로 논문 그림은 넣지 않았습니다.
- 공식 저장소 bingreeky/AgenTracer에는 LICENSE가 없습니다. 코드를 넣지 않았고 실행하지 않았습니다.
- 공개 데이터 릴리스(v1.0.0)는 저장소 Release에서 받았습니다. 데이터 자체는 이 폴더에 넣지 않았습니다.
