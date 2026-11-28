# Speculative Decoding (arXiv:2211.17192) 분석 노트

Leviathan et al.(ICML 2023, 구두)의 speculative sampling과 Algorithm 1을 구현 모드로 분석한 노트입니다. PyTorch로 짠 구현(pytest 24개)으로 출력 분포 보존, 식 (1)·Thm 3.8·Thm 3.11, 논문 표를 확인하고, 실제 Pythia-14M(초안)·70M·160M(목표)에서 수락률 α와 수락 길이를 쟀습니다. 논문 그림은 arXiv 라이선스(CC BY 4.0)에 따라 도식 Fig. 5 한 장만 `figures/paper_fig5.png`로 넣었고 나머지는 직접 그린 그래프입니다. 원문 PDF는 넣지 않았습니다. 공식 코드는 공개되어 있지 않은 것으로 보입니다.

## 결과 요약

- 합성 범주형 분포 8개 시나리오(표본 2,000,000)와 3토큰 결합분포 4개 γ(표본 600,000)가 정확한 분포와 구별되지 않았고(Holm 후 최소 p 0.15), 틀린 변형 3개(residual 없음, q 재표집, 항상 수락)는 TV 0.09-0.64로 확실히 걸렸습니다.
- 실제 Pythia 쌍에서 처음 2토큰의 결합분포(top-10 표준화, 표본 8,000, 프롬프트 4개)가 일치하고 T = 0에서는 20개 설정이 목표의 greedy 경로와 토큰 단위로 같았습니다. KV 캐시 1토큰 대 9토큰 호출의 float32 분포 차이는 TV 평균 1.2e-4입니다("분포가 같다"는 정확한 산술의 보장).
- 식 (1), Thm 3.8, Table 1은 재현됩니다. Table 4의 EXP는 반올림 범위에서 재현(EMP/EXP는 0.68-1.29)되지만 Table 5의 T = 1 열(2.5, 3.1, 3.6, 5배)은 Thm 3.8로 재현되지 않습니다(최적 γ에서도 2.33-4.79배).
- Pythia에서 α는 T = 1에서 0.61(70M)과 0.56(160M)입니다. 반복당 토큰 수는 식 (1)과 -11%에서 +4% 안으로 맞지만 i.i.d. 가정은 데이터가 기각합니다(위치별 평균 β 0.586에서 0.653로 증가). T = 0의 α는 반복 퇴화로 부풀려집니다.
- 이 CPU에서는 병렬 검증이 공짜가 아니어서(1토큰에서 2토큰 호출 시간이 4.5-6배) 검증 비용을 넣으면 모든 γ에서 느려집니다(0.4-0.7배, 모형 계산). 초안과 목표의 크기 비가 1 : 5-1 : 11이라 논문의 영역이 아닙니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_claim_verification.md` | 논문 분석, 계보, 식 검산과 실험 결과, 주장 검증 C1-C9 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | speculative sampling, Alg. 1, 논문 식과 pytest 24개 |
| `analysis/run_synthetic.py`, `synthetic_results.json` | 합성 분포 검정, 식·표 재계산 |
| `analysis/real_*.py`, `predict_speedup.py`, `numerics_check.py` 와 `*_results.json` | Pythia 측정 |
| `analysis/plot_figures.py`, `figures/` | 논문 Fig. 5와 우리가 그린 그래프 다섯 장 |

## 재실행

모델과 데이터는 로컬 캐시(`EleutherAI/pythia-14m|70m|160m`, 데이터셋 `NeelNanda/pile-10k`)를 씁니다.

```bash
pip install torch transformers numpy scipy pandas matplotlib pyarrow pytest
export HF_HUB_OFFLINE=1
cd analysis
python3 -m pytest test_minimal_impl.py -q
python3 run_synthetic.py
python3 real_alpha.py
python3 real_decode.py 60
python3 real_e2e.py
python3 real_timing.py && python3 predict_speedup.py
python3 numerics_check.py
python3 plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2211.17192v2, arXiv 라이선스 CC BY 4.0. 원문에 회사 저작권 고지가 없어 도식 한 장만 넣었습니다.
- 공식 코드는 공개되어 있지 않은 것으로 보여 보지 못했고, 이 폴더의 구현은 논문 설명으로 직접 짰습니다.
