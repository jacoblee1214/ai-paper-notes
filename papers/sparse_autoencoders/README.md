# Sparse Autoencoders Find Highly Interpretable Features in Language Models (arXiv:2309.08600) 분석 노트

Cunningham et al. (ICLR 2024 포스터)의 희소 오토인코더(SAE)를 구현 모드로 분석한 노트입니다. 식 (1)–(4)를 검산하고 L1 수축 편향을 유도해 학습 결과와 맞췄으며, PyTorch 최소 구현(CPU)과 pytest 13개로 (a) 정답을 아는 합성 superposition 데이터, (b) 2층 모델(induction_heads 분석의 산출물)의 활성에서 시험했습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(원문은 arXiv:2309.08600에서 받을 수 있습니다). 공식 코드(`HoagyC/sparse_coding`, LICENSE 파일 없음)는 정적으로 읽고 대조만 했고 이 폴더에 넣지 않았습니다.

## 검증 결과 (요약)

- **수축**: 직교 정답에서 최적 코드 c = max(a − α/2, 0)를 유도하고, 학습한 SAE의 평균 L0와 회귀 기울기가 이론과 0.05 이내로 일치(시드 3개).
- **정답 시험대**(정답 64개, 시드 3개): 정답/차원 = 2에서 α 0.1–0.3이면 복원율(최대 코사인 > 0.9) 0.98–1.00. α ≤ 0.03에서는 FVU 0.002–0.02인데도 복원율 0.02–0.39. 정답/차원 = 4에서는 어떤 (α, R)에서도 최대 0.46.
- **2층 모델 활성**(학습 97,920 벡터): FVU 0.17–0.23(L0 35–40/512)일 때 다음 토큰 손실 +0.37–0.45 nats, loss recovered 0.88–0.89. 시드 간 사전 일치는 원소의 34–36%만 최대 코사인 0.9 이상.
- **해석 대용 지표**(상위 20개 활성의 최빈 토큰 비율, 자동 해석 점수가 아님): SAE 0.46–0.56, 기저·무작위·PCA 0.26–0.36. 그러나 랜덤 초기화 모델의 SAE도 0.48.
- **induction 연결**: 반복 구간 선택 특징 34개가 층 1 head 직접 출력을 빼면 평균 55% 줄지만, 디코더 방향과 head 출력 방향의 코사인은 무작위 수준.
- **이후 문헌**: 시드 불안정, 랜덤 모델에서도 비슷한 해석 점수, 하류 과제에서 단순 기준선 열세 등으로 "SAE 특징 = 참 특징"이라는 읽기는 수정됨.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법, 평가, 주장과 근거 위치, 한계, 판 간 차이 |
| `analysis/02_prior_work.md` | 계보, 이후 문헌, 다른 분석과의 관계, 자료 표 |
| `analysis/03_analysis.md` | 식 검산, 정답 시험대, 2층 모델 SAE, 대용 지표, induction 연결 |
| `analysis/04_claim_verification.md` | 이후 문헌의 확인·수정 C1–C12, 공식 코드 대조 D1–D9 |
| `analysis/minimal_impl.py` | SAE, 학습, 지표, 합성 시험대, 모델 복사본과 훅, 대용 지표 (PyTorch) |
| `analysis/test_minimal_impl.py` | pytest 13개 |
| `analysis/experiments.py`, `results_{a,a2,b,c,d}.json` | 실험 스크립트와 결과(학습 설정 포함) |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 네 장 |

## 재실행

실험 b, c, d는 `induction_heads` 분석의 체크포인트와 activation(`artifacts/`, 132 MB, 이 노트에는 넣지 않음)이 필요합니다. 이 폴더와 같은 위치의 `../../induction_heads/artifacts/`를 읽으므로, `induction_heads` 분석의 README에 있는 재생성 명령으로 먼저 만드세요. 실험 a, a2와 테스트는 그것 없이 돕니다(테스트 하나는 artifacts가 없으면 건너뜀).

```bash
pip install torch numpy scipy matplotlib pytest
python3 -m pytest analysis/test_minimal_impl.py -q        # 13 passed
cd analysis
python3 experiments.py a        # 정답 시험대 alpha x R 격자 (CPU 약 12분)
python3 experiments.py a2       # 직교 정답 수축 검산 (약 1분)
python3 experiments.py b        # 2층 모델 SAE (약 13분)
python3 experiments.py c        # 해석 대용 지표, 랜덤 모델 대조 (약 6분)
python3 experiments.py d        # head 출력과 SAE 특징 (약 3분)
python3 plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2309.08600v3, CC BY 4.0. 게시본의 그래프는 모두 우리가 그렸고 논문 그림은 싣지 않았습니다.
- 공식 코드는 LICENSE 파일이 없어 복사하지 않았고 실행하지 않았습니다. 이 폴더의 구현은 논문 설명을 보고 직접 짰습니다.
