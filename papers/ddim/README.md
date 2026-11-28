# DDIM (arXiv:2010.02502, ICLR 2021 포스터) 분석 노트

*Denoising Diffusion Implicit Models*(Song, Meng, Ermon, arXiv v4 2022-10-05)의 식과 주장을 구현 모드로 분석한 노트입니다. **학습은 하지 않았습니다.** 신경망 대신 가우시안 혼합 데이터에서 닫힌 식으로 나오는 정확한 잡음 예측기 eps*로 보조정리 1, 식 12·16, 정리 1, ODE 연결(식 13-15)을 PyTorch(float64, CPU)로 구현하고 검산했습니다(pytest 24개). 표집 품질(슬라이스 W1, MMD, 20 시드), 이산화 오차, 인코딩·재구성, 일관성, 보간을 정확한 모델에서 쟀고 논문 표 1·2·3을 다시 읽었으며 공식 PyTorch 코드(`ermongroup/ddim`, MIT)를 정적으로 대조했습니다. 공식 코드는 실행하지 않았고 이 저장소에 넣지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스, 원문은 arXiv:2010.02502에서 받을 수 있습니다). 그래프는 모두 직접 그렸습니다.

## 구현 가능성 (요약)

- **논문 수치(CIFAR10 FID 등)의 재현은 이 자료의 범위 밖입니다.** DDPM 체크포인트와 FID 5만 장 생성이 필요합니다.
- **식과 표집기의 성질 검증은 CPU로 약 4분이면 됩니다**(PyTorch, float64, 메모리 약 1.1 GB).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(식 번호), 실험(표 번호), 주장별 근거 위치, 한계 |
| `analysis/02_prior_work.md` | 계보, 구성 요소의 출처, 가까운 대안, 자료 표 |
| `analysis/03_analysis.md` | 식 유도 검산, 정확한 eps* 실험(격자, 이산화 오차, 재구성, 보간), 논문 표 재계산 |
| `analysis/04_code_verification.md` | 공식 코드 대조(K1-K7, D1-D10)와 이후 문헌 확인(C1-C8) |
| `analysis/minimal_impl.py` | 정확한 eps* 최소 구현: GMM, 부분 수열, 식 12, 오일러 형태, ODE 기준 해, 인코딩, 지표 |
| `analysis/test_minimal_impl.py` | pytest 24개 |
| `analysis/minimal_impl_results.json` | 실험 결과 전체(시드, n, 신뢰구간, Holm) |
| `analysis/paper_tables.py`, `paper_tables_results.json` | 논문 표 1·2·3 재계산 |
| `analysis/plot_figures.py`, `figures/` | 직접 그린 그래프 3장 |

## 재실행

```bash
pip install torch numpy scipy matplotlib pytest
cd analysis
python3 -m pytest -q test_minimal_impl.py   # 24 passed
python3 paper_tables.py
python3 minimal_impl.py                     # 약 4분
python3 plot_figures.py
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 비독점 배포 라이선스(nonexclusive-distrib 1.0)이고 CC BY가 아니라서 논문 그림을 넣지 않았습니다.
- 공식 코드는 `github.com/ermongroup/ddim` (MIT)이며 이 폴더에 코드를 복사하지 않았고 실행하지 않았습니다.
- 후속 논문(PNDM, DPM-Solver, EDM, Improved DDPM, Analytic-DPM, 일관성 모델 등)의 내용은 arXiv 초록으로만 확인했습니다.
