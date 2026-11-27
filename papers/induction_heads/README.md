# In-context Learning and Induction Heads (arXiv:2209.11895) 분석 노트

Olsson et al. (Anthropic, Transformer Circuits Thread 2022; arXiv 업로드 2022-09-24)을 구현 모드로 분석한 노트입니다. 정의(ICL 점수, prefix matching, copying)와 K-composition을 식으로 검산하고, 직접 짠 PyTorch 토이(pytest 10개)로 합성 반복 데이터와 자연 텍스트(파이썬 표준 라이브러리) 모델에서 단계 변화와 head ablation을 시험했습니다. 공식 코드와 가중치는 없고(내부 모델), 사전학습 가중치는 `gpt2_induction.py`에서만 공개 모델(GPT-2 small, Pythia-14m·70m, 추론만)로 썼습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (원문은 arXiv:2209.11895에서 받을 수 있습니다). 체크포인트와 activation도 넣지 않았고, 아래 명령으로 다시 만들 수 있습니다.

## 검증 결과 (요약)

- 1층 모델의 attention 비는 앞 토큰을 바꿔도 변하지 않습니다(pytest로 확인). prefix matching에는 K-composition(두 층) 또는 smeared key가 필요합니다.
- 합성 반복 데이터, 5시드: 2층 attention-only는 반복 손실이 평탄(약 2.75)에서 4.2 → 0.05–0.16으로 급락합니다. 손실 중간점 220–250 스텝, prefix 점수 중간점 260–300 스텝. 1층은 2.05에서 멈추고 smeared key 1층은 0.18까지 떨어집니다.
- 같은 층의 head가 모두 induction head라 하나씩 끄면 효과 +0.03–0.06, 함께 끄면 +4.4–4.7. 패턴 유지 ablation은 층 0(이전 토큰 head)의 기여를 거의 0으로 만듭니다.
- 자연 텍스트만으로 학습한 2층 MLP 토이(최대 약 1억 토큰)에서는 induction head가 생기지 않았고, 학습 창의 50%에 같은 구절을 두 번 넣으면 생기지만 자연 텍스트 ICL 점수는 오르지 않았습니다. 논문의 자연 텍스트 거시 주장은 재현하지 못했습니다.
- GPT-2 small(로컬 캐시, 추론만)에서 prefix 점수 상위 5개가 문헌의 5.1, 5.5, 6.9, 7.2, 7.10과 일치했고, 평균 ablation으로 5개/12개를 끄면 반복 복사 이득의 39%/69%가 사라지지만 ICL 점수(위치 50 − 500, 0.53)는 약 1/4–1/3만 줄었습니다(`analysis/03_analysis.md` §7).
- 논문의 ICL 점수 정의의 부호, prefix matching 문장 표현, copying 정규화가 모호합니다(`01` D1–D3).

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 정의, 여섯 논거, 논문 안의 불일치, 한계 |
| `analysis/02_prior_work.md` | 계보, 가까운 대안, 이후 문헌(초록 수준), 자료 표 |
| `analysis/03_analysis.md` | 정의·식 검산, 합성 토이, 자연 텍스트 토이, 지표 검증, 공개 모델(GPT-2) 측정 |
| `analysis/04_claim_verification.md` | 이후 문헌과 토이가 확인·수정한 주장 C1–C9 |
| `analysis/minimal_impl.py` | TinyLM, 측정 함수, ablation, 코퍼스 구성 (PyTorch) |
| `analysis/test_minimal_impl.py` | pytest 10개 (CPU 약 30–50초) |
| `analysis/run_experiments.py` | 합성, 자연 텍스트 실험과 activation 재생성 |
| `analysis/definition_checks.py`, `definition_checks_results.json` | 정의 기준선 |
| `analysis/synthetic_results.json`, `natural_results.json`, `natural_mix_results.json` | 실험 결과 |
| `analysis/gpt2_induction.py`, `gpt2_induction_results.json` | 공개 모델(GPT-2 small, Pythia-14m·70m)의 head 점수, ICL 점수, ablation(0·평균, 무작위 기준선, 부트스트랩 CI) |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프(토이 세 장, GPT-2 두 장) |

## 재실행

```bash
pip install torch numpy matplotlib pytest
python -m pytest analysis/test_minimal_impl.py -q
python analysis/definition_checks.py
python analysis/run_experiments.py synthetic            # CPU 약 10분
python analysis/run_experiments.py natural --seeds 0 1 2 --steps 4000 --dup 0.5 --out analysis/natural_mix_results.json
python analysis/run_experiments.py natural --seeds 0 1 2 --steps 4000 --dup 0 --no-artifacts --out analysis/natural_results.json
python analysis/plot_figures.py
# 공개 모델 측정: torch·transformers가 설치된 파이썬, 로컬 캐시의 openai-community/gpt2, EleutherAI/pythia-14m·70m,
# 데이터셋 NeelNanda/pile-10k (오프라인이면 HF_HUB_OFFLINE=1). MPS 약 10–15분, --quick은 작은 표본
python analysis/gpt2_induction.py
```

자연 텍스트 코퍼스는 실행하는 파이썬의 표준 라이브러리 소스와 `pydoc_data.topics`입니다. 결과 json에 코퍼스 sha256이 있고, 파이썬 버전이 다르면 코퍼스와 결과가 달라집니다(우리는 3.12.7).

## 출처와 라이선스

- 논문: arXiv 2209.11895, CC BY 4.0 (abs 페이지 확인). 원문 PDF에 저작권 고지는 없습니다. 게시본의 그래프는 모두 우리가 그렸고 논문 그림은 싣지 않았습니다.
