# Proving Test Set Contamination in Black Box Language Models (arXiv:2310.17623) 분석 노트

Oren et al. (ICLR 2024)의 교환성 기반 오염 검정(순열 검정, 분할 t 검정)을 구현 모드로 분석한 노트입니다. 식을 유도해 검산하고, numpy 최소 구현과 pytest 12개로 확인했으며, n-gram 토이 언어 모델에서 1종 오류율과 검정력을 쟀고, 논문 표 1·2·3·4를 다시 계산했습니다. 공식 코드(`tatsu-lab/test_set_contamination`, 라이선스 파일 없음)는 정적으로 읽고 대조만 했고 실행하지 않았으며 이 폴더에 넣지 않았습니다. 논문 원문 PDF와 논문 그림도 넣지 않았습니다 (원문은 arXiv:2310.17623에서 받을 수 있습니다). numpy를 쓴 이유: 검정의 통계적 성질을 보려고 같은 설정을 수백 번 반복해야 해서, 학습이 계수 세기뿐인 n-gram이 맞습니다.

## 검증 결과 (요약)

- 순열 검정 p̂ = (1 + #{순열 로그우도 ≥ 정준}) / (m+1)은 유한한 m에서 정확합니다(논문의 엄격한 부등호는 동점에서 비보수적). 분할 검정은 같은 계산량 n(1+m)에서 p 하한이 없지만, 이점의 원인은 샤딩이 아니라 정규 꼬리 외삽입니다.
- 토이 LM(문항 200, 4-gram): 오염 없음에서 1종 오류율은 α 이하, 정준 순서로 k번 넣으면 k = 2–4회부터 검출, **복사본마다 순서를 섞어 넣으면 16번 외웠어도 검출되지 않음**. α = 0.05에서는 순열 M=25가 샤드 t보다 강했고 α = 0.01에서는 샤드 t만 기각 가능했습니다.
- 표 4(MMLU 파일별 p)에서 음성 대조를 제외한 43개의 Fisher 합산이 표 2의 0.014, 0.011, 0.362를 재현합니다. 음성 대조 모델이 56개 파일 중 약 17%에서 p < 0.05라 비교환성이 흔합니다. 표 2의 p는 균등하지 않고 큰 쪽으로 치우쳐 있습니다.
- 공식 코드: 분할 검정만 구현, `np.std` ddof = 0(통계량 √(r/(r−1))배), `1 - cdf` 정밀도 한계 ~1e-16, CUDA 필수.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법, 실험, 주장과 근거 위치, v1과 v2 차이 |
| `analysis/02_prior_work.md` | 계보, 가까운 대안, 이후 문헌, 자료 표 |
| `analysis/03_analysis.md` | 식 유도 검산, 표 재계산, 토이 실험 결과 |
| `analysis/04_claim_verification.md` | 이후 문헌의 확인·반박 C1–C9, 공식 코드 대조 D1–D10 |
| `analysis/minimal_impl.py` | 순열 검정, 분할 검정, 정규 근사 대조 검정, n-gram 토이 LM (numpy, scipy) |
| `analysis/test_minimal_impl.py` | pytest 12개 (약 2분) |
| `analysis/toy_experiments.py`, `toy_results.json` | 1종 오류율, 검정력, 민감도 실험 (약 2–3분, 병렬) |
| `analysis/toy_variance_check.py`, `toy_variance_results.json` | 샤드 t의 분모 부풀림 확인 |
| `analysis/recompute.py`, `recompute_results.json` | 표 1·2·3·4 재계산 (논문 텍스트가 필요) |
| `analysis/repo_checks.py`, `repo_checks_results.json` | 공식 저장소 시험 파일 점검 (저장소 clone 필요) |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pytest
python3 -m pytest analysis/test_minimal_impl.py -q
python3 analysis/toy_experiments.py        # toy_results.json
python3 analysis/toy_variance_check.py
python3 analysis/plot_figures.py
```

`recompute.py`는 논문 텍스트(`pdftotext -layout paper.pdf paper/paper.txt`, 원문은 arXiv:2310.17623v2)를, `repo_checks.py`는 `git clone --depth 1 https://github.com/tatsu-lab/test_set_contamination repo`를 먼저 해야 합니다.

## 출처와 라이선스

- 논문: arXiv 2310.17623, CC BY 4.0 (abs 페이지 확인). 원문 PDF에 회사 저작권 고지는 없습니다. 게시본의 그래프는 모두 우리가 그렸고 논문 그림은 싣지 않았습니다.
- 공식 코드는 라이선스 파일이 없어 복사하지 않았고 실행하지 않았습니다. 시험 파일은 읽기만 했습니다.
