# 구성 타당도 검토 (arXiv:2511.04703, NeurIPS 2025 D&B) 분석 노트

Bean et al., *Measuring what Matters: Construct Validity in Large Language Model Benchmarks*를 검증 모드로 분석한 노트입니다. 저자가 공개한 코딩 자료로 선별 흐름(표 7·8), 집계 비율(§4, 그림 3), 평가자 간 일치도(표 9)를 다시 계산하고 권고 8개와 집계 근거의 대응표를 만들었습니다. 저자 노트북은 읽기만 했고 LLM 호출은 하지 않았습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2511.04703에서 받을 수 있습니다).

## 검증 결과 (요약)

- 선별 흐름(2,189 → 938 → 846 → 522 → 445) 산술이 맞고 키워드 선별은 ICML 외 학회에서 연도별로 거의 재현됩니다. LLM 선별 검증은 표본 50편이라 재현율 구간이 56–98%입니다.
- 보고 비율은 중복 10행을 포함한 455행을 분모로 해야 재현됩니다(영향 평균 0.36%p).
- 통계 검정은 16.0%가 아니라 2.9%, 현실 환경 비교는 31.2%가 아니라 5.1%, 재사용은 48.1%, 편의 표집은 39.3%입니다.
- 이중 코딩은 41쌍이고 평균 BPK 약 0.50, Cohen kappa 평균 0.15입니다. "현상을 정의했는가" 문항의 BPK는 0.07입니다.
- 권고 8개 중 6개만 코드북 집계에서 나오고 오염 대비와 오류 분석은 근거 항목이 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` ... `04_code_verification.md` | 논문 분석, 계보와 관련 연구, 재계산, 공개 자료 대조 |
| `analysis/recompute_*.py`, `*_results.json` | 재계산 스크립트와 요약 결과 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 두 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
# 저자 자료를 data/ 아래에 받아 둔다(원자료는 이 폴더에 넣지 않았다)
#   HF  : huggingface.co/datasets/ambean/construct-validity-review (CC BY-SA 4.0)
#   GitHub: github.com/am-bean/benchmark_review (라이선스 파일 없음)의 data/ 일부
python3 -I analysis/recompute_funnel.py
python3 -I analysis/recompute_headline.py
python3 -I analysis/recompute_irr.py
python3 -I analysis/recompute_extras.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 원문 PDF에 회사 저작권 고지는 없습니다. 논문 그림 두 장(그림 1, 3)은 저작자 표기와 함께 실었고 "© 저자, CC BY 4.0, 출처 arXiv:2511.04703"를 적었습니다.
- HF 데이터셋은 CC BY-SA 4.0입니다. GitHub 저장소 am-bean/benchmark_review는 라이선스 파일이 없어 원자료 사본을 넣지 않았고 우리가 계산한 요약 JSON과 스크립트만 있습니다.
