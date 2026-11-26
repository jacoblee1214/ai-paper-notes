# BLIP-2 (arXiv:2301.12597, ICML 2023) 분석 노트

*BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models* (Li, Li, Savarese, Hoi, Salesforce Research)를 식·마스크·파라미터 검산, 직접 짠 PyTorch 최소 구현(pytest 21개), 합성 장면 토이 실험, 공식 코드(LAVIS) 정적 대조, 이후 문헌 점검으로 분석한 자료입니다. 구현 모드라 논문 수치를 재현하지 않았고, 토이는 이해용입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv:2301.12597에서 받을 수 있습니다).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(Q-Former, 세 마스크, 2단계), 실험 위치, 한계, 논문 안의 모호함, 판 간 차이(v1-v3) |
| `analysis/02_prior_work.md` | 계보(ALBEF, BLIP, Frozen, Flamingo)와 이후 문헌(InstructBLIP, LLaVA, Honeybee, MM1 등), 자료 표 |
| `analysis/03_analysis.md` | 마스크·식·파라미터 검산, 논문 표 재계산, 최소 구현, 공식 코드 대조 D1-D12, 토이 결과 |
| `analysis/04_claim_verification.md` | 이후 문헌과 토이가 확인·수정·반박한 주장 C1-C9 |
| `analysis/minimal_impl.py` | Q-Former(질의 K개, 공유 self-attention, 질의 위치 교차 attention, 질의·텍스트 FFN 분리), 세 마스크, ITC/ITM/ITG 손실, 2단계 소프트 프롬프트, 대비용 MLP 연결기, 합성 장면 (PyTorch) |
| `analysis/test_minimal_impl.py` | pytest 21개 (CPU 수 초) |
| `analysis/run_experiments.py`, `toy_seed*.json`, `toy_results.json`, `decoder_check.json` | 토이 실험(질의 수, Q-Former 대 MLP, 1단계 유무, 5시드)과 요약 |
| `analysis/derivation_checks.py`, `derivation_results.json` | 마스크 행렬, 층별 정보 도달, 파라미터 산술, 표 재계산 |
| `analysis/plot_figures.py`, `figures/` | 그래프 두 장 |

## 재실행

```bash
pip install torch pytest numpy scipy matplotlib
python -m pytest analysis/test_minimal_impl.py -q          # 21 passed
python analysis/derivation_checks.py                        # derivation_results.json (수 초)
python analysis/run_experiments.py decoder                  # 고정 디코더 사전학습 (data/decoder.pt, 약 10분)
python analysis/run_experiments.py train --seed 0 --out analysis/toy_seed0.json   # 시드당 CPU 30-60분
python analysis/run_experiments.py summary                  # toy_results.json
python analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0으로 공개되어 있으나 이 저장소는 논문 그림을 싣지 않고 직접 그린 그래프만 씁니다.
- 공식 코드 `github.com/salesforce/LAVIS`(BSD-3-Clause)는 필요한 파일 5개만 정적으로 읽었고 이 저장소에는 넣지 않았습니다. 가중치와 데이터셋은 받지 않았고, 토이 데이터는 코드가 생성합니다.
