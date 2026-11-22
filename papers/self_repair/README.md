# Is Self-Repair a Silver Bullet for Code Generation? (arXiv:2306.09896) 분석 노트

Olausson et al. (ICLR 2024)의 공개 원자료를 다시 계산하고, 과제 단위 불확실성을 붙여 검산한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv에서 받을 수 있습니다). 논문 그림은 넣지 않았고, 그래프는 원자료로 우리가 그렸습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 문제, 방법, 실험, 주장과 근거 위치 |
| `analysis/02_prior_work.md` | 계보, 저장소 표, arXiv 메타데이터 |
| `analysis/03_analysis.md` | 재계산과 검산 (V1–V15) |
| `analysis/04_code_verification.md` | 공식 저장소와 논문의 대조 (D1–D14, 정적 읽기) |
| `analysis/*.py`, `*_results.json` | 재계산 스크립트와 결과 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy matplotlib
python3 -I analysis/recompute_human_study.py
python3 -I analysis/apps_gpt4_exact_check.py <all-gpt4-repair-results-tokens.jsonl>
python3 -I analysis/apps_feedback_check.py <all-chatgpt-repair-results-tokens.jsonl> <mp-chatgpt-mf-gpt4-nf-10-results-tokens.jsonl>
python3 analysis/plot_figures.py
```

APPS 원자료는 공식 저장소의 `paper/data/apps/apps-data.tar.bz2`(git-lfs, 198 MB)에서 풉니다. 인간 피드백 표는 논문 Appendix E에서 옮겼습니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다.
- 공식 저장소(`github.com/theoxo/self-repair`)에는 LICENSE 파일이 없고, 보관(archived) 상태입니다. 코드는 게시하지 않았고 정적으로만 읽었습니다.
- 원자료와 저장소 결과 파일은 재배포하지 않았습니다. 필요한 사람은 저장소에서 받습니다.
