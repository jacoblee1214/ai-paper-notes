# LLaVA (arXiv:2304.08485, NeurIPS 2023 구두) 분석 노트

*Visual Instruction Tuning* (Liu, Li, Wu, Lee)를 식·산술 검산, 직접 짠 PyTorch 최소 구현(pytest 23개), 합성 장면 토이 실험, 공식 코드의 정적 대조, LLaVA-Bench 통계 점검, 이후 문헌 점검으로 분석한 자료입니다. 구현 모드라 논문 수치를 재현하지 않았고 토이는 이해용입니다. 논문 원문 PDF는 넣지 않았고(arXiv:2304.08485에서 받을 수 있음), 논문 그림은 구조 도식(그림 1) 한 장만 CC BY 4.0 출처 표기와 함께 실었습니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법(식 1–3), 실험 위치, 한계, 논문 안의 모호함, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 계보(Flamingo, BLIP-2, MiniGPT-4 등)와 이후 변화(LLaVA-1.5, Prismatic, Cambrian-1 등), 자료 표 |
| `analysis/03_analysis.md` | 식·산술 검산, 최소 구현 설명, 공식 코드 대조 D1–D12, 토이 결과, LLaVA-Bench·ScienceQA 통계 |
| `analysis/04_claim_verification.md` | 이후 문헌과 토이가 확인·수정·반박한 주장 C1–C13 |
| `analysis/minimal_impl.py` | 고정된 작은 ViT, 끝에서 두 번째 층 패치 특징, 선형·2층 MLP 투영기, 작은 인과 LM, 대화 템플릿, 답변 토큰 손실 마스크, 1·2단계 학습 루프 |
| `analysis/test_minimal_impl.py` | pytest 23개 (CPU 수 초) |
| `analysis/run_experiments.py`, `toy_runs/`, `toy_results.json` | 토이 실험(7설정 x 5시드)과 요약 |
| `analysis/derivation_checks.py`, `derivation_results.json` | 시각 토큰·투영기·비전 타워 파라미터, 문맥 예산, 계산량, KV 캐시 산술 |
| `analysis/bench_stats.py`, `bench_results.json` | LLaVA-Bench 문항 수, 표 재계산, 상대 점수 불확실성, ScienceQA 구간 |
| `analysis/plot_figures.py`, `crop_figures.py`, `figures/` | 그래프 두 장과 논문 그림 1 |

## 재실행

```bash
pip install torch pytest numpy matplotlib pymupdf
python -m pytest analysis/test_minimal_impl.py -q           # 23 passed
python analysis/derivation_checks.py                         # derivation_results.json (수 초)
python analysis/run_experiments.py train --seeds 0 1 2 3 4   # Apple MPS 기준 시드당 약 10-15분(CPU는 더 느림)
python analysis/run_experiments.py summary                   # toy_results.json
python analysis/plot_figures.py
```

`bench_stats.py`는 공식 저장소(`haotian-liu/LLaVA`)의 평가 질문·판정 파일과 HF `liuhaotian/llava-bench-in-the-wild`의 질문 파일을 필요로 합니다. 이 폴더에는 그 입력을 넣지 않았고 결과(`bench_results.json`)만 있습니다.

## 출처와 라이선스

- 논문은 arXiv CC BY 4.0입니다. 그림 1은 저자 도식이고 나머지 그래프는 모두 직접 그린 것입니다.
- 공식 코드 `github.com/haotian-liu/LLaVA`(Apache-2.0, 커밋 `c121f04`)는 정적으로만 읽었고 이 저장소에 넣지 않았습니다. 가중치와 데이터셋은 받지 않았고, 토이 데이터는 코드가 생성합니다.
