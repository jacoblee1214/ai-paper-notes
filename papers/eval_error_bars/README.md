# Adding Error Bars to Evals (arXiv:2411.00640) 분석 노트

Evan Miller의 *Adding Error Bars to Evals*에 나오는 식(CLT 표준오차, 군집 표준오차, 짝지은 차이, 문항당 K번 샘플링의 분산 분해, 필요 표본 수)을 번호 그대로 검산하고, numpy·scipy로 구현해(신경망이 없는 통계 방법이라 PyTorch 대신 numpy) 몬테카를로로 성질을 확인한 뒤, 공개된 문항별 결과(SWE-bench BM25 기준선, 반복 실행 결과)에 적용한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2411.00640). 논문에는 그림이 없고, 그림은 모두 직접 그린 것입니다. 논문의 4.2절 산술 예와 5절 문항 수 예(969)에서 오류를 찾았습니다(`analysis/03_analysis.md` §3.3, §5.2).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 틀, 식 번호별 정리, 권고, 한계 |
| `analysis/02_prior_work.md` | 계보, 이후 문헌, 이 저장소의 관련 분석, 자료 표 |
| `analysis/03_analysis.md` | 식 검산, 가정표, 최소 구현·테스트, 공개 결과 적용, 주장 판정 |
| `analysis/04_claim_verification.md` | 이후 문헌의 확인·수정·반박 (C1–C10) |
| `analysis/minimal_impl.py` | 최소 구현 (입력 `scores[question, sample]`과 군집 id) |
| `analysis/test_minimal_impl.py` | pytest 19개 (몬테카를로 포함, 약 3초) |
| `analysis/mc_experiments.py`, `mc_results.json` | 군집 수별 포함률, 짝지은 분산 감소, 검정력, 작은 n 포함률 |
| `analysis/apply_public_results.py`, `apply_results.json` | 공개 문항별 결과에 적용한 계산과 결과 (입력 데이터는 넣지 않음) |
| `analysis/plot_figures.py`, `figures/` | 직접 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pytest
python3 -m pytest -q analysis
python3 -I analysis/mc_experiments.py
```

`apply_public_results.py`는 `data/`에 공개 결과(SWE-bench/experiments의 `results.json`, ASSERT-KTH/agentic-evals-artifacts의 평가 결과 JSON)를 받아 두어야 돌아가며, 계산 결과는 `analysis/apply_results.json`에 들어 있습니다.

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(`nonexclusive-distrib/1.0`)로 공개되어 있고 CC BY가 아닙니다. 논문 그림은 쓰지 않았고(논문에 그림도 없음) 원문 PDF도 넣지 않았습니다.
- 공식 코드는 없습니다.
- 적용 예의 입력 데이터는 이 저장소 밖 공개 결과 파일의 사본이며 이 폴더에는 넣지 않았습니다.
