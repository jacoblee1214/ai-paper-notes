# CLIP (arXiv:2103.00020, ICML 2021) 분석 노트

*Learning Transferable Visual Models From Natural Language Supervision* (Radford 외, OpenAI)를 식 검산, 직접 짠 PyTorch 최소 구현(pytest 14개), 합성 쌍 토이 실험, 이후 문헌 점검으로 분석한 자료입니다. 구현 모드라 논문 수치를 재현하지 않았고, 토이는 이해용입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv:2103.00020에서 받을 수 있고, arXiv 라이선스는 CC BY가 아닙니다).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(그림 3 의사 코드), 실험 위치, 한계, 논문 안의 불일치 |
| `analysis/02_prior_work.md` | 계보(ConVIRT, VirTex 등)와 이후 문헌(OpenCLIP, SigLIP, MetaCLIP 등), 자료 표 |
| `analysis/03_analysis.md` | 식 유도 검산, 최소 구현 설명, 토이 결과, 공식 코드 대조 |
| `analysis/04_claim_verification.md` | 이후 문헌과 토이가 확인·수정·반박한 주장 C1–C9 |
| `analysis/minimal_impl.py` | 작은 ViT와 텍스트 Transformer, 대칭 InfoNCE, 학습되는 온도(상한 100), 제로샷 분류기와 프롬프트 앙상블, 합성 데이터 |
| `analysis/test_minimal_impl.py` | pytest 14개 (CPU 수 초) |
| `analysis/run_experiments.py`, `toy_*.json`, `gap_results.json` | 합성 쌍 토이 실험(설정 3개 x 5시드)과 모달리티 간격 |
| `analysis/derivation_checks.py`, `derivation_results.json` | 식 검산 수치 |
| `analysis/plot_figures.py`, `figures/` | 그래프 두 장 |

## 재실행

```bash
pip install torch pytest numpy matplotlib
python -m pytest analysis/test_minimal_impl.py -q          # 14 passed
python analysis/derivation_checks.py                        # derivation_results.json (수 초)
python analysis/run_experiments.py train --config baseline --out analysis/toy_baseline.json   # CPU, 시드당 수 분
python analysis/run_experiments.py summary                  # toy_results.json
python analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv의 기본 배포 라이선스(non-exclusive)라서 논문 그림을 싣지 않았습니다. 이 저장소의 그래프는 모두 직접 그린 것입니다.
- 공식 코드 `github.com/openai/CLIP`(MIT)과 `github.com/mlfoundations/open_clip`은 정적으로만 읽었고 이 저장소에는 넣지 않았습니다. 가중치와 데이터셋은 받지 않았고, 토이 데이터는 코드가 생성합니다.
