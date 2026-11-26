# CPU-ARC (arXiv:2611.09891) 분석 노트

*CPU-ARC: Agentic Reasoning Caching for on-CPU Reasoning Reuse* (Belcak 외, NVIDIA, arXiv v1, 2026-11-06)를 논문 표 2와 부록 C의 재계산, 통계 검정, 비용·적용률 토이 모형으로 분석한 자료입니다. 공식 코드와 문항별 결과가 공개되어 있지 않아 코드 대조는 없고 주장 검증(C1–C14)을 했습니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2611.09891에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **재현은 하지 않았고 지금 할 수 없습니다.** 코드, 캐시, 파인튜닝한 1B 가중치, 문항별 결과가 공개되어 있지 않습니다.
- **구축 파이프라인은 직접 구현해야 합니다.** 소규모 재현 설계(라벨 누출 대조, 보류 문항 시험, 직접 풀이 기준선, CPU 지연 토이)는 `analysis/04_claim_verification.md` §3에 제안만 했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 직접 그린 그래프 세 장) |
| `analysis/01_paper_analysis.md` | 문제, 방법, 실험 설정과 표 2, 주장 목록, 한계 |
| `analysis/02_prior_work.md` | 계보와 가까운 대안, 이 저장소의 관련 분석, 자료 표 |
| `analysis/03_analysis.md` | 비용 재계산, 토큰·단가 분해, 손익분기, 민감도, 표본 수·구간·동등성, 누출·교란, 적용률 토이 |
| `analysis/04_claim_verification.md` | 주장 C1–C14 판정과 소규모 재현 설계 |
| `analysis/paper_tables.py` | 논문 표 2와 부록 C 표 3–10 수치 |
| `analysis/cost_check.py` | 비용·손익분기·민감도·토큰 비 계산 (numpy 없이 수 초) |
| `analysis/stats_check.py` | 표본 수 복원, Wilson 구간, 동등성·MDE, 정의 일관성, Holm (scipy) |
| `analysis/toy_cache_model.py` | 적용률·재등장 횟수·감지율 토이 모형 (numpy, 1초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |
| `report/figures/` | 직접 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 -I analysis/cost_check.py
python3 -I analysis/stats_check.py
python3 -I analysis/toy_cache_model.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(CC BY 아님)로 공개되어 있고 PDF 첫 쪽에 "© 2026 NVIDIA. All rights reserved." 고지가 있습니다. 그래서 논문 그림은 넣지 않았고, 그래프는 모두 논문 표의 수치에서 직접 그렸습니다.
- 분석에 쓴 수치(표 2, 부록 C)는 논문에서 옮긴 값이며 `analysis/paper_tables.py`에 출처 표 번호와 함께 있습니다.
- 공식 코드는 없고 가중치와 데이터는 받지 않았습니다.
