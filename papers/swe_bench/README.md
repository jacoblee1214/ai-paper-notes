# SWE-bench (arXiv:2310.06770, ICLR 2024) 분석 노트

HF 데이터셋(전체·Lite·Verified의 test 분할)과 SWE-bench/experiments의 공개 결과 파일(문항별 해결 목록)로 논문의 데이터셋 통계와 표 5를 다시 계산하고, 모델 간 차이를 짝지은 정확 McNemar로 검정하고, 공식 저장소의 세 시점(논문 이틀 뒤 커밋, v2.0.0, main)에서 채점 코드를 정적으로 대조한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다. 논문 그림 두 장(도식과 차트)은 CC BY 4.0이라 출처를 밝히고 넣었습니다.

## 재현 가능성

- 코드 실행, Docker, LLM 호출은 하지 않았습니다. 재계산은 공개 데이터와 공개 결과 파일만 씁니다.
- 데이터(약 15 MB)와 결과 파일(약 2 MB)은 폴더에 넣지 않았습니다. 받는 방법은 아래 재실행에 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 문제, 구성 파이프라인, 채점, 결과표 요약, 한계 |
| `analysis/02_prior_work.md` | 계보와 후속(Lite, Verified, Multimodal, SWE-Gym 등), 테스트 품질 비판 |
| `analysis/03_analysis.md` | 통계 재계산, 구간, 짝지은 검정, 라벨 잡음 |
| `analysis/04_code_verification.md` | 채점 코드의 시점별 정적 대조(D1–D9) |
| `analysis/*.py`, `analysis/*.json` | 재계산·그래프 스크립트와 결과 |

## 재실행

```bash
# 데이터와 결과 파일 받기 (HF 약 15 MB, GitHub API 약 2 MB)
for d in SWE-bench SWE-bench_Lite SWE-bench_Verified; do
  mkdir -p data/$d
  curl -sL -o data/$d/test.parquet https://huggingface.co/datasets/princeton-nlp/$d/resolve/main/data/test-00000-of-00001.parquet
done
for s in test lite verified; do for m in 20231010_rag_claude2 20231010_rag_gpt35 20231010_rag_swellama13b 20231010_rag_swellama7b 20240402_rag_claude3opus 20240402_rag_gpt4; do
  d=data/exp/$s/$m; mkdir -p $d
  gh api -H "Accept: application/vnd.github.raw" repos/SWE-bench/experiments/contents/evaluation/$s/$m/results/results.json > $d/results.json
done; done
# 정적 대조용 저장소 (선택, 실행하지 않음)
git clone --depth 1 --branch v2.0.0 https://github.com/SWE-bench/SWE-bench repo/v2.0.0
python3 -I analysis/recompute_tables.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY 4.0(arXiv abs 페이지). 그림은 도식과 차트 두 장만 출처 표기와 함께 실었습니다.
- 공식 저장소 SWE-bench/SWE-bench는 MIT입니다. 코드는 넣지 않았고 실행하지 않았습니다.
- 데이터셋과 결과 파일은 각각 HF와 SWE-bench/experiments의 것이며 이 폴더에 복사하지 않았습니다.
