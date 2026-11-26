# SigLIP (arXiv:2303.15343, ICCV 2023) 분석 노트

*Sigmoid Loss for Language Image Pre-Training* (Zhai 외, Google DeepMind)를 식 검산, 직접 짠 PyTorch 최소 구현(pytest 18개), 논문 표 재계산, 합성 도형 토이(softmax 대 sigmoid, 6시드), 공식 코드·OpenCLIP 정적 대조로 분석한 자료입니다. 구현 모드라 논문 수치를 재현하지 않았고 토이는 이해용입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv:2303.15343에서 받을 수 있고 arXiv 라이선스는 CC BY 4.0입니다).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(알고리즘 1, 청크 계산), 실험 위치, 한계, 판(v1–v4) 간 차이 |
| `analysis/02_prior_work.md` | 계보(CLIP, LiT, Hadsell 외), 이후 문헌(SigLIP 2, Inf-CLIP, SigCLR 등), 자료 표 |
| `analysis/03_analysis.md` | 식 유도 검산(편향 초기화, 최적 로짓 분리, 메모리 산술), 최소 구현 설명, 논문 표 재계산, 토이 결과 |
| `analysis/04_claim_verification.md` | 이후 문헌과 토이가 확인·수정한 주장 C1–C9, 논문 대 코드 D1–D8 |
| `analysis/minimal_impl.py` | sigmoid 손실, 장치 D개 청크 시뮬레이션, softmax 손실, 손 유도 기울기, 해석값 |
| `analysis/test_minimal_impl.py` | pytest 18개 (CPU 1-2초) |
| `analysis/toy_data_model.py`, `toy_train.py`, `run_experiments.py` | 합성 도형 데이터, 작은 두 타워 모델, 학습과 시드 요약 |
| `analysis/toy_seed*.json`, `toy_results.json` | 토이 결과(시드별, 평균과 t 구간, 짝지은 차이) |
| `analysis/derivation_checks.py`, `derivation_results.json` | 식 검산 수치 |
| `analysis/table_recalc.py`, `table_recalc_results.json` | 논문 표 5·8·9 재계산 |
| `analysis/plot_figures.py`, `figures/` | 그래프 두 장 |

## 재실행

```bash
pip install torch pytest numpy matplotlib scipy
python -m pytest analysis/test_minimal_impl.py -q          # 18 passed
python analysis/derivation_checks.py                        # derivation_results.json (수 초)
python analysis/table_recalc.py                             # table_recalc_results.json (paper/paper.txt가 필요: arXiv에서 받은 PDF의 pdftotext -layout)
for s in 0 1 2 3 4 5; do python analysis/run_experiments.py train --seeds $s --out analysis/toy_seed$s.json; done   # CPU 1스레드, 시드당 약 15-25분
python analysis/run_experiments.py summary                  # toy_results.json
python analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다. 이 저장소의 그래프는 모두 직접 그린 것입니다.
- 공식 코드 `github.com/google-research/big_vision`(Apache-2.0)과 `github.com/mlfoundations/open_clip`은 정적으로만 읽었고 이 저장소에는 넣지 않았습니다. `toy_data_model.py`의 데이터·모델 코드는 이 분석 시리즈의 CLIP 분석 토이에서 복사한 것입니다. 가중치와 데이터셋은 받지 않았고 토이 데이터는 코드가 생성합니다.
