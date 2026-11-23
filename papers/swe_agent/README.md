# SWE-agent (arXiv:2405.15793, NeurIPS 2024) 분석 노트

SWE-bench/experiments의 공개 결과 파일(문항별 해결 목록)로 논문 표를 다시 계산하고, 어블레이션 차이를 같은 설정 반복의 불일치와 비교하고, 공식 저장소 v0.3.0을 정적으로 대조한 자료입니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다. 논문 그림 두 장(도식과 차트)은 CC BY 4.0이라 출처를 밝히고 넣었습니다.

## 재현 가능성

- 코드 실행, Docker, LLM 호출은 하지 않았습니다. 재계산은 공개 표와 공개 결과 파일만 씁니다.
- 결과 파일(4세트, 약 0.7 MB)은 폴더에 넣지 않았습니다. 받는 방법은 아래 재실행에 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 문제, ACI 구성, 표·그림 요약, 한계 |
| `analysis/02_prior_work.md` | 계보와 후속(Agentless, mini-swe-agent 등), 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 구간, 짝지은 검정, 어블레이션 잡음, 공개 파일 대조 |
| `analysis/04_code_verification.md` | 공식 코드(v0.3.0)와 논문의 대조(D1–D9) |
| `analysis/*.py`, `analysis/*.json` | 재계산·그래프 스크립트와 결과 |

## 재실행

```bash
# 공개 결과 파일 받기 (GitHub API, 4세트)
for p in lite test; do for m in claude3opus gpt4; do
  d=data/exp/$p/$m; mkdir -p $d
  for f in $(gh api repos/SWE-bench/experiments/contents/evaluation/$p/20240402_sweagent_$m/results --jq '.[]|select(.type=="file")|.name'); do
    gh api -H "Accept: application/vnd.github.raw" repos/SWE-bench/experiments/contents/evaluation/$p/20240402_sweagent_$m/results/$f > $d/$f
  done
done; done
python3 -I analysis/recompute_tables.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY 4.0(arXiv abs 페이지). 그림은 도식과 차트 두 장만 출처 표기와 함께 실었습니다.
- 공식 저장소 SWE-agent/SWE-agent는 MIT입니다. 코드는 넣지 않았고 실행하지 않았습니다.
- 결과 파일은 SWE-bench/experiments 저장소의 것이며 이 폴더에 복사하지 않았습니다.
