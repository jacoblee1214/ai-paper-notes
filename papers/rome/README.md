# ROME (arXiv:2202.05262) 분석 노트

Meng et al.(NeurIPS 2022)의 인과 추적과 rank-one 편집을 구현 모드로 분석한 노트입니다. 갱신식을 제약 최소제곱에서 다시 유도해 KKT 풀이와 맞추고, 합성 사실을 외운 6층 transformer(CPU, 시드 3개, pytest 12개)에서 인과 추적과 편집을 시험했습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC가 아니라서). 공식 코드(`kmeng01/rome`, MIT)는 읽기만 했고 이 폴더에 넣지 않았습니다.

## 결과 요약

- 토이에서 인과 효과는 논문의 "주어 마지막 토큰 중간층 MLP"가 아니라 관계(마지막) 토큰의 0-1층에 몰렸고, 주어 위치 선계산 보조 손실을 줘도 같았습니다.
- 편집은 관계 토큰 키에서 모든 층 성공, 주어 마지막 토큰 키에서는 층이 깊을수록 실패하며 같은 주어의 다른 관계를 훼손했습니다.
- C를 키 차원(512)보다 적은 506개 토큰으로 추정하면 갱신이 90배 커지고 이웃·전역 정확도가 무너졌습니다.
- 사전학습 GPT-2 small(124M, 사실 237개)에서 인과 추적: late site(마지막 토큰의 attention·hidden)는 있고 주어 마지막 토큰의 중간층 MLP(early site)는 보이지 않았습니다. 사실 100 x 층 11개 ROME 편집에서 추적 효과는 편집 성공을 예측하지 못했습니다(전체 ρ -0.06, 추가 R² ≤ 0.02; 편집이 거의 모든 층에서 포화해 층 선택이 더 중요하다는 주장은 검정되지 않음). 한 가지 손상 규칙, 하이퍼파라미터는 GPT-2 XL 값, C는 pile-10k로 추정.
- 이후 문헌: 추적 위치는 편집 층을 예측하지 못하고(Hase et al.), 손상 규칙에 민감하며, 편집 뒤 다단계 추론과 반복 편집에서 실패합니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_claim_verification.md` | 논문 분석, 계보, 식 유도·토이 결과, 이후 문헌 C1-C7과 공식 코드 대조 D1-D8 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | 구현과 pytest 12개 |
| `analysis/run_experiments.py`, `results_*.json`, `plot_figures.py` | 실험, 결과(설정 포함), 그래프 |
| `analysis/gpt2_causal_trace.py`, `gpt2_localization_vs_edit.py`, `gpt2_*_results.json` | GPT-2 small 인과 추적과 추적 대 편집 상관 |
| `figures/` | 우리가 그린 그래프 열 장 |

## 재실행

```bash
pip install torch transformers numpy scipy matplotlib pyarrow pytest
python3 -m pytest analysis/test_minimal_impl.py -q
cd analysis
python3 run_experiments.py 0 1 2
python3 run_experiments.py aux:0 aux:1 aux:2
python3 plot_figures.py
# GPT-2 small (가중치 openai-community/gpt2, 텍스트 NeelNanda/pile-10k, 사실 파일은 `mkdir -p ../data && curl -L https://rome.baulab.info/data/dsets/known_1000.json -o ../data/known_1000.json`)
python3 gpt2_causal_trace.py
for k in 0 1 2 3; do python3 gpt2_localization_vs_edit.py --shard $k/4 & done; wait
python3 gpt2_localization_vs_edit.py --analyze_only
```

## 출처와 라이선스

- 논문: arXiv 2202.05262v5, arXiv 비독점 배포 라이선스(CC 아님). 그래프는 모두 우리가 그렸습니다.
- 공식 코드는 MIT이나 복사하거나 실행하지 않았고, 이 폴더의 구현은 논문 설명으로 직접 짰습니다.
