# FlashAttention (arXiv:2205.14135) 분석 노트

Dao et al.(NeurIPS 2022)의 FlashAttention을 구현 모드로 분석한 노트입니다. 타일링 순전파(Alg. 1), 재계산 역전파(Alg. 4), 블록 희소(Alg. 5)와 HBM 읽기·쓰기를 줄마다 세는 시뮬레이터를 PyTorch로 짜서(CPU, pytest 24개) Theorem 2의 Θ(N²d²/M)를 78개 (N, d, M) 설정에서 확인했습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC가 아니라서). 공식 코드(`Dao-AILab/flash-attention`, BSD-3-Clause)는 읽기만 했고 이 폴더에 넣지 않았습니다.

## 결과 요약

- 계수기의 값이 닫힌 식(순전파 2Nd + T_c(3Nd + 4N), 역전파 4Nd + T_c(5Nd + 2N))과 78개 설정에서 정확히 같고, 주항 상수는 순전파 12, 역전파 20입니다(적합 12.3, 20.2).
- 상수까지 보면 M이 약 3d² 아래에서는 표준 attention보다 접근이 줄지 않습니다. 192KB SRAM에서 d = 128은 순전파 1.8배, d = 64는 6.1배입니다(논문 Alg. 1의 블록 규칙 기준이며 현재 공식 커널의 접근 수가 아님).
- 블록 희소의 접근 수는 2Nd + s·T_c·N(3d + 4)로 s에 정확히 선형입니다.
- GPT-2 small 실제 Q·K·V(헤드 720개)에서 float32 오차는 1e-5 이내로 naive와 같고, float16에서는 S를 fp16으로 저장하는 naive보다 약 2.5배 정확합니다(기준 구현에 따라 방향이 달라질 수 있음).
- 논문 표 재계산: Table 2·4 속도비 일치, Table 1의 ±가 SD인지 SE인지에 따라 유의성이 갈림(p = 0.0008 또는 0.22), Fig. 2의 66.6 GFLOPs는 우리가 센 값과 12배 어긋남.
- 이후 문헌: 최적성은 M ≥ d²에서만(arXiv:2402.07443), v1은 이론 최대 FLOPs/s의 25–40%(arXiv:2307.08691).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_claim_verification.md` | 논문 분석, 계보, 식 유도·실험 결과, 이후 문헌 C1-C8과 공식 코드 정적 대조 D1 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | 구현과 pytest 24개 |
| `analysis/run_experiments.py`, `results.json` | 접근 수 스케일링, 교차점, 블록 희소, 논문 표 재계산, 정확성 |
| `analysis/gpt2_flash_check.py`, `gpt2_flash_check_results.json` | GPT-2 small 실제 Q·K·V |
| `analysis/bench_cpu.py`, `bench_cpu_results.json` | CPU 메모리·시간 예비 측정 |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 네 장 |

## 재실행

```bash
pip install torch transformers numpy scipy matplotlib pyarrow pytest
python3 -m pytest analysis/test_minimal_impl.py -q
cd analysis
python3 run_experiments.py
# GPT-2 small (가중치 openai-community/gpt2, 텍스트 NeelNanda/pile-10k)
python3 gpt2_flash_check.py
python3 bench_cpu.py
python3 plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2205.14135v2, arXiv 비독점 배포 라이선스(CC 아님). 그래프는 모두 우리가 그렸습니다.
- 공식 코드는 BSD-3-Clause이나 복사하거나 실행하지 않았고, 이 폴더의 구현은 논문 설명으로 직접 짰습니다.
