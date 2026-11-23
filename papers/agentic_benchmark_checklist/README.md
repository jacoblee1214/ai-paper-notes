# ABC 점검표 (arXiv:2507.02825, NeurIPS 2025 D&B) 분석 노트

Zhu et al., *Establishing Best Practices in Building Rigorous Agentic Benchmarks*를 검증 모드로 분석한 노트입니다. 논문 부록 D의 점검표 평가 10건을 저자 저장소의 평가표와 항목별로 대조하고 절별 점수와 위반 집계를 다시 계산했으며, τ-bench 과제 정의를 정적으로 읽어 "아무것도 안 하는 에이전트가 통과한다"를 정수 개수로 재현했습니다. 저자 저장소(커밋 `ae11247`)를 정적으로 읽어 논문과 대조했고, 벤치마크 실행·패치 적용·LLM 호출은 하지 않았습니다. 논문 원문 PDF, 논문 그림, 저자 코드는 넣지 않았습니다(원문은 arXiv:2507.02825에서 받을 수 있습니다).

## 검증 결과 (요약)

- 점검표는 43항목(O 20, T 10, R 13)이고 부록 D의 10개 평가표는 저자 저장소 yaml과 항목별로 완전히 일치합니다. 위반 벤치마크 수 7/7/10이 재계산으로 맞습니다. 단 O 점수의 분모가 벤치마크마다 1–10으로 달라 순위용이 아니고, 해당 없음을 1점으로 셉니다.
- τ-bench 항공의 do-nothing 38%, DB 쏟기 40%는 과제 정의에서 19/50, 20/50으로 정확히 재현됩니다. 소매는 정적으로 5/115, 9/115까지만 재현되어 논문 수치(6.0%, 9.6%)와 다릅니다.
- WebArena의 4.4%는 N/A 과제 36/812로 정확히 맞고, OSWorld 13/46(24→37)은 산술이 맞지만 전체 369문항으로는 +3.5%p입니다.
- 부록 F 표 15(BIRD 신뢰구간)는 식대로 재현되지만 구간이 점추정을 포함하지 않는 행이 61개 중 44개입니다. 식이 보정 방향을 뒤집은 것으로 보입니다.
- ABC 문구의 T.5·T.10은 SWE-bench 채점 코드의 허점(SKIPPED F2P, 테스트 파일 조작)을 겨냥하지만 논문의 평가는 둘 다 1점을 주었습니다. 평가자 수와 평가자 간 일치도는 보고되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 문제, 점검표 구조, 평가 결과, 사례, 한계 |
| `analysis/02_prior_work.md` | 계보, 후속 연구, 자료 표 |
| `analysis/03_analysis.md` | 점수표 재계산, τ-bench 정적 재계산, 과대 추정 산술, 표 15 검산 |
| `analysis/04_code_verification.md` | 저자 저장소 정적 대조(D1–D12)와 SWE-bench 채점 허점과의 대조표 |
| `analysis/*.py`, `analysis/*_results.json` | 재계산·그래프 스크립트와 결과 |
| `report/figures/own_*.png` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install pyyaml matplotlib
git clone --depth 1 https://github.com/uiuc-kang-lab/agentic-benchmarks repo   # 커밋 ae11247
mkdir -p data && git clone https://github.com/sierra-research/tau-bench data/tau-bench   && git -C data/tau-bench checkout 14bf0ef52e595922d597a38f32d3e8c0dce3a8f8
# paper/paper.txt: arXiv:2507.02825v5 PDF를 받아 pdftotext -layout paper/paper.pdf paper/paper.txt
python3 -I analysis/parse_assessments.py
python3 -I analysis/recompute_scores.py
python3 -I analysis/recompute_tau_bench.py
python3 -I analysis/recompute_claims.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(arXiv abs 페이지)라 논문 그림은 싣지 않았고 그래프 세 장은 모두 우리가 그렸습니다. 원문 PDF에 회사 저작권 고지는 없습니다.
- 저자 저장소 uiuc-kang-lab/agentic-benchmarks는 라이선스 파일이 없어 코드·점검표 문서·평가표를 복사하지 않았습니다. τ-bench(sierra-research/tau-bench)는 MIT이며 과제 정의를 읽기만 했습니다.
