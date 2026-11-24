# Language Models Don't Always Say What They Think (arXiv:2305.04388, NeurIPS 2023) 분석 노트

*Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting* (Turpin et al., arXiv v2, 2023-12-09)을 저자가 공개한 모델 출력으로 재계산하고 공식 코드를 정적으로 대조한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2305.04388에서 받을 수 있습니다). 논문 그림은 그림 1(정확도 차트) 한 장만 넣었고 출처를 적었습니다.

## 재현 가능성

- 모델을 다시 부르지 않았습니다. API 호출, 공식 코드 실행, 패키지 설치는 하지 않았습니다.
- 재계산은 저자 저장소(`github.com/milesaturpin/cot-unfaithfulness`, MIT, 커밋 `df09945`)의 공개 출력을 씁니다. 이 저장소에는 BIG-Bench 항목과 canary 문자열이 들어 있어 출력과 데이터는 넣지 않았고 집계 JSON만 넣었습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 그림 1 포함) |
| `analysis/01_paper_analysis.md` | 문제, 방법, 주장과 근거 대응, 한계 |
| `analysis/02_prior_work.md` | 계보, 이후 연구, 이 저장소의 다른 분석과의 관계, 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, paired 검정, 사람 주석 재집계, BBQ 보정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조(K1–K10, D1–D12) |
| `analysis/recompute_bbh.py`, `recompute_bbq.py`, `annotation_checks.py`, `check_against_paper.py` | 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/*_results.json` | 위 스크립트의 결과 |
| `analysis/plot_figures.py`, `report/crop_figures.py` | 그래프 세 장, 논문 그림 1 자르기 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 1(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf pillow
git clone https://github.com/milesaturpin/cot-unfaithfulness repo
git -C repo checkout df099452736946533f59498a90c23be3f09631c4
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2305.04388
python3 -I analysis/recompute_bbh.py
python3 -I analysis/recompute_bbq.py
python3 -I analysis/annotation_checks.py
python3 -I analysis/check_against_paper.py
python3 -I analysis/plot_figures.py
python3 report/crop_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지 확인). 원문 PDF에 ©, copyright, all rights reserved 표시는 없습니다(저자 소속 표기만 있음).
- 논문 그림 1은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2305.04388"). 모델 출력 텍스트가 든 표는 넣지 않았습니다.
- 공식 코드와 데이터는 MIT 라이선스이고 BIG-Bench 파생 항목에 canary 문자열이 있습니다. 이 저장소에는 코드와 출력을 넣지 않았습니다.
