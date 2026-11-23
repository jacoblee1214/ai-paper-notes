# Deep RL at the Edge of the Statistical Precipice (arXiv:2108.13264, NeurIPS 2021) 분석 노트

IQM, 최적성 격차, 층화 부트스트랩 구간, 성능 프로필, 개선 확률을 numpy·scipy로 직접 구현하고(신경망이 없는 통계 방법이라 PyTorch 대신 numpy), 식을 유도해 검산한 뒤 직접 만든 토이 세계에서 논문의 핵심 주장을 확인한 자료입니다. 공식 라이브러리 `google-research/rliable`은 정적으로만 읽고 대조했습니다. 논문 원문 PDF와 공식 코드는 넣지 않았습니다. 논문 그림은 쓰지 않았고 그림은 모두 직접 그린 것입니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(식 번호), 실험, 한계 |
| `analysis/02_prior_work.md` | 계보, 후속, 이 저장소의 관련 분석, 자료 표 |
| `analysis/03_analysis.md` | 식 유도 검산, 토이 실험 결과, 주장별 판정 |
| `analysis/04_claim_verification.md` | 이후 문헌의 확인·수정(C1–C9), rliable 대조(D1–D10) |
| `analysis/minimal_impl.py` | 최소 구현 (입력은 `scores[run, task]`) |
| `analysis/test_minimal_impl.py` | pytest 15개 (몇 초) |
| `analysis/toy_experiments.py`, `toy_results.json` | 토이 세계 실험(약 1분)과 결과 |
| `figures/` | 직접 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pytest
python3 -m pytest -q analysis
python3 -I analysis/toy_experiments.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다.
- 공식 라이브러리는 Apache-2.0이고 코드는 넣지 않았으며 실행하지 않았습니다.
- 논문의 개별 실행 점수 버킷(`gs://rl-benchmark-data`)은 2026-11-23에 받을 수 없어 논문 데이터는 쓰지 않았습니다.
