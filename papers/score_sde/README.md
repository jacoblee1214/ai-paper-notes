# Score SDE (arXiv:2011.13456, ICLR 2021 구두) 분석 노트

*Score-Based Generative Modeling through Stochastic Differential Equations* (Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, arXiv v2 2021-02-10)를 **신경망 점수 대신 가우시안 혼합의 닫힌 식 점수**로 검증하고, 논문 표 1·2·3을 통계적으로 다시 읽고, 공식 PyTorch 코드를 정적으로 대조한 자료입니다. 학습, 가중치·데이터 다운로드, 설치, 공식 코드 실행은 하지 않았습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv:2011.13456에서 받을 수 있고, arXiv 라이선스는 CC BY가 아닙니다).

## 구현 가능성 (요약)

- **논문 수치(CIFAR-10 FID 2.20, 2.99 bits/dim)의 재현은 이 자료의 범위 밖입니다.** 학습된 NCSN++/DDPM++ 체크포인트와 GPU가 필요합니다.
- **식과 표집기의 성질 검증은 CPU로 약 8분이면 됩니다**(PyTorch, float64, 메모리 수백 MB).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(식 번호), 실험(표 번호), 주장별 근거 위치, 한계 |
| `analysis/02_prior_work.md` | 계보, 구성 요소의 출처, 가까운 대안, 자료 표 |
| `analysis/03_analysis.md` | 식 유도 검산, 정확한 점수 실험 결과, 표 재계산, flow matching 연결 |
| `analysis/04_claim_verification.md` | 이후 문헌의 확인·수정(C1–C7), 공식 코드 대조(K1–K8, D1–D5) |
| `analysis/minimal_impl.py` | 정확한 점수 최소 구현: VE/VP/sub-VP, 예측기, 교정기, 확률 흐름 ODE, 우도, 조건부 생성 |
| `analysis/test_minimal_impl.py` | pytest 40개 |
| `analysis/minimal_impl_results.json` | 실험 결과 전체(시드, n, 신뢰구간) |
| `analysis/paper_tables.py`, `paper_tables_results.json` | 논문 표 1·2·3의 Welch 검정, Holm 보정 |

## 재실행

```bash
pip install torch numpy scipy pytest
cd analysis
python3 -m pytest -q test_minimal_impl.py   # 40 passed
python3 paper_tables.py
python3 minimal_impl.py                     # 약 8분
```

## 출처와 라이선스

- 논문의 arXiv 라이선스는 비독점 배포 라이선스(nonexclusive-distrib 1.0)이고 CC BY가 아니라서 논문 그림을 넣지 않았습니다.
- 공식 코드는 `github.com/yang-song/score_sde_pytorch` (Apache-2.0)이며 이 폴더에 코드를 복사하지 않았고 실행하지 않았습니다.
- 후속 논문(Song 2021b, VDM, EDM, DPM-Solver, Flow Matching 등)의 내용은 arXiv 초록으로만 확인했습니다.
