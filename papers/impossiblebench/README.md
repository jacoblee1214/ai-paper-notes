# ImpossibleBench (arXiv:2510.20270, ICLR 2026 포스터) 분석 노트

*ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases* (arXiv v1, 2025-10-23)를 본문 수치 재계산, 통계 검정, 공개 데이터셋 구조 감사, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2510.20270에서 받을 수 있습니다). 논문에는 표가 없고 모델 출력 로그가 공개되지 않아, 그림 수치는 막대 라벨과 본문 숫자, 산점도에서 읽은 값으로만 검증했습니다.

## 재현 가능성 (요약)

- **논문 수치 재현은 이 맥에서 불가입니다.** 8개 모델의 상용 API 호출과 SWE-bench Docker 이미지가 필요하고 점수 로그가 공개되지 않았습니다.
- **정적 코드 대조와 공개 데이터 구조 감사는 완료했습니다.** 공식 코드는 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 네 장 포함) |
| `analysis/01_paper_analysis.md` | 문제, 방법, 실험, 주장과 근거 대응, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소의 출처, 공식 저장소와 자료 표 |
| `analysis/03_analysis.md` | 산술 일치, 효과 검정, 능력-부정행위 상관, 모니터 구간, 데이터셋 구조 감사 |
| `analysis/04_code_verification.md` | 공식 코드 대조(일치 K1–K8, 차이 D1–D9) |
| `analysis/recompute_stats.py` | 비율 재계산과 Fisher·스피어만·Wilson (numpy, scipy, 수 초) |
| `analysis/audit_dataset.py` | 공개 데이터셋의 변형 구조 감사 (pandas, pyarrow, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 네 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 세 장 (우리가 그림)과 논문 그림 네 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy pandas pyarrow matplotlib pymupdf
mkdir -p data/lcb data/swe
for s in conflicting oneoff original; do
  curl -sL "https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench/resolve/main/data/$s-00000-of-00001.parquet" -o data/lcb/$s.parquet
  curl -sL "https://huggingface.co/datasets/fjzzq2002/impossible_swebench/resolve/main/data/$s-00000-of-00001.parquet" -o data/swe/$s.parquet
done
python3 analysis/recompute_stats.py
python3 analysis/audit_dataset.py
python3 analysis/plot_figures.py
python3 report/crop_figures.py   # paper/paper.pdf 가 필요 (arXiv:2510.20270)
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv abs 페이지 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 논문 그림 네 장 (그림 3, 4, 6, 10)은 차트만 골랐고 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2510.20270"). 사진이나 데이터셋 예시 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/safety-research/impossiblebench`에 있고 MIT 라이선스입니다. 이 저장소에는 코드를 넣지 않았고 실행하지 않았습니다.
- 데이터셋(`fjzzq2002/impossible_*`)은 Hugging Face에서 받아 구조만 확인했고 이 저장소에 넣지 않았습니다 (HF 카드에 라이선스 표기 없음).
