# Toy Models of Superposition (arXiv:2209.10652) 분석 노트

Elhage et al. (Transformer Circuits, 2022)의 장난감 모델을 구현 모드로 분석한 노트입니다. 선형 모델이 상위 m개 특징만 담는다는 것과 2특징·1차원 단계도의 경계를 식으로 유도해 검산하고, PyTorch 최소 구현(CPU)과 pytest 21개로 확인했습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv 라이선스가 CC가 아니라서, 원문은 arXiv:2209.10652에서 받을 수 있습니다). 공식 노트북(`anthropics/toy-models-of-superposition`, MIT)은 정적으로 읽고 대조만 했고 이 폴더에 넣지 않았습니다.

## 검증 결과 (요약)

- **선형 모델**: 상위 m개 특징만 담고 희소도와 무관함을 Ky Fan 부등식으로 유도. n=20, m=5에서 학습 손실이 이론값과 0.33% 이내.
- **ReLU 모델**(n=20, m=5, 시드 5개): 밀집이면 5개, 밀도 0.01이면 평균 11.4개를 담고 손실은 선형의 0.18배.
- **2특징·1차원 단계도**: 세 해(한쪽만/한쪽만/대척점)의 손실과 경계를 닫힌 형식으로 유도(b=0이면 d < 4r/(2+5r) 또는 4/(5+2r)). 학습 해의 분류와 156칸 중 154칸(98.7%) 일치. 세 해는 근사라 27칸에서 학습 손실이 더 낮았음.
- **차원수**: 1/2 계단 선명(n=200, m=15), 평면(m=2)에서 정오각형(2/5). 3/4, 3/8 등 세밀한 계단은 절반 규모에서 재현 못 함.
- **이후 문헌**: SAE로 사후 분해가 실행되었으나 시드 의존, 참 특징 복원이 낮다는 보고 등으로 기대가 약해졌고, 선형성 가정은 다차원 특징으로 수정됨.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법, 주장과 근거 위치, 한계 |
| `analysis/02_prior_work.md` | 계보, 가까운 이론, 이후 문헌, 자료 표 |
| `analysis/03_analysis.md` | 식 유도 검산, 단계도 경계, 차원수, 토이 실험 결과 |
| `analysis/04_claim_verification.md` | 이후 문헌의 확인·수정 C1–C10, 공식 노트북 대조 D1–D6 |
| `analysis/minimal_impl.py` | 토이 모델, 학습, 측정(노름·간섭·차원수), 닫힌 형식, 2특징 단계도 (PyTorch) |
| `analysis/test_minimal_impl.py` | pytest 21개 |
| `analysis/experiments.py`, `results_{a,b,c,geom}.json` | 실험 스크립트와 결과(학습 설정 포함) |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 다섯 장 |

## 재실행

```bash
pip install torch numpy matplotlib pytest
python3 -m pytest analysis/test_minimal_impl.py -q        # 21 passed
cd analysis
python3 experiments.py a        # n=20, m=5 희소도 훑기 (CPU 약 7분)
python3 experiments.py b        # 2특징·1차원 단계도 (약 2분)
python3 experiments.py c        # n=200, m=15 차원수 (시드 3개, CPU 약 15분)
python3 experiments.py geom     # m=2, n=40 기하 (약 5분)
python3 plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2209.10652v1, arXiv 비독점 배포 라이선스(CC 아님). 게시본의 그래프는 모두 우리가 그렸고 논문 그림은 싣지 않았습니다.
- 공식 노트북은 MIT 라이선스이나 복사하지 않았고 실행하지 않았습니다. 이 폴더의 구현은 논문 설명을 보고 직접 짰습니다.
