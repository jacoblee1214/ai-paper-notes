# MAST (arXiv:2503.13657, NeurIPS 2025 D&B 스포트라이트) 분석 노트

*Why Do Multi-Agent LLM Systems Fail?*를 공개 주석표와 공개 LLM 라벨의 재계산, 표 통계 검정, 공식 저장소 정적 대조로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(CC BY-NC-ND 4.0).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 주장, 판 사이의 차이 |
| `analysis/02_prior_work.md` | 계보, 이후 연구, 자료 정보 |
| `analysis/03_analysis.md` | κ 재계산, 판정자 구간, 공개 라벨 재계산, 통계 검정 |
| `analysis/04_code_verification.md` | 공식 저장소와 논문의 대조(D1–D12) |
| `analysis/*.py`, `*_results.json` | 재계산 스크립트와 결과 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

입력 파일은 이 폴더에 없습니다. 공식 저장소(`multi-agent-systems-failure-taxonomy/MAST`)를 clone하고 `inter_annotator_agreement_annotations/*.pdf`를 `pdftotext -layout`으로 변환해 `.scratch/`에 두고, HF `mcemri/MAD`(현 `mcemri/MAST-Data`)의 `MAD_human_labelled_dataset.json`, `MAD_full_dataset.json`을 `data/hf/`에 둡니다.

```bash
python3 -I analysis/iaa_recompute.py
python3 -I analysis/iaa_pdf_vs_hf.py
python3 -I analysis/ac1_check.py
python3 -I analysis/judge_check.py
python3 -I analysis/mad_recompute.py
python3 -I analysis/stat_checks.py
python3 -I analysis/make_figures.py
```

## 출처와 라이선스

- 논문: arXiv 라이선스 CC BY-NC-ND 4.0(abs 페이지 확인). 그림은 넣지 않았습니다.
- 공식 저장소에는 LICENSE가 없습니다(GitHub API). 코드와 trace는 넣지 않았고 실행하지 않았습니다.
- MAST-Data(HF)는 CC BY 4.0입니다. 데이터 자체는 이 폴더에 넣지 않았습니다.
