# AWQ (arXiv:2306.00978) 분석 노트

Lin et al.(MLSys 2024)의 AWQ를 구현 모드로 분석한 노트입니다. 활성 크기로 고른 입력 채널에 스케일을 곱해 보호하는 가중치 전용 양자화를 선형층 단위 PyTorch로 짜서(α 격자 탐색, 가중치 clipping, 그룹 INT4/INT3, CPU, pytest 20개) 식 2–3의 오차비 논증을 수치로 확인하고, GPT-2 small과 Pythia-160M에 W4·W3(그룹 128)을 적용해 pile-10k 보류 문서 64개로 RTN, AWQ, AWQ+clip, 채널 보호를 비교했습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC가 아니라서). 공식 코드(`mit-han-lab/llm-awq`, MIT)는 읽기만 했고 이 폴더에 넣지 않았습니다.

## 결과 요약

- 식 2의 오차비 1/s는 s ≤ 1.5에서 맞고, 그룹 최댓값이 바뀌어 Δ' ≈ Δ 가정이 깨지는 그룹이 s = 2에서 17–20%, s = 4에서 약 50%입니다. 토이의 출력 오차는 s ≈ 2–3에서 최소입니다.
- 실제 모델 두 개에서 "평균 Δ'/Δ × 1/s"가 논문 Table 2와 소수 둘째 자리까지 일치하고, 최적 s는 1.5–2, s = 4는 RTN보다 나쁩니다.
- 채널 보호 기준은 활성 > 가중치 노름 > 무작위(W3, 1% FP16, 모두 Holm p ≤ 0.003)입니다.
- AWQ(scale)는 W3에서 RTN 대비 perplexity를 12%(GPT-2), 37%(Pythia-160M) 낮추지만 W4의 GPT-2에서는 RTN과 구분되지 않습니다(비율 1.002, CI [0.989, 1.015]). clipping을 더하면 W3에서 추가 16%, 45% 낮아져 scale의 몫은 절반 이하였습니다.
- 논문 표 재계산: Table 2 산술 일치, Table 1에서 RTN보다 나쁜 칸 6개, Table 1과 3의 RTN 불일치(119.00 대 119.47), Table 4 INT4 g128에서 AWQ − GPTQ-R는 −0.01 ~ −0.05 PPL.
- Llama-2-7B W4 g128 가중치는 3.97 GB(압축비 3.4배), 24 GB 맥에서 컨텍스트 4096 KV 포함 약 5.7 GiB.
- 공식 코드와 논문의 차이 12개(`analysis/04_code_verification.md`): 블록 출력 손실, zero-point 비대칭, α 격자에 1.0 제외 등.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_code_verification.md` | 논문 분석, 계보, 식 유도·실험 결과, 공식 코드 정적 대조 D1–D12 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | 구현과 pytest 20개 |
| `analysis/saliency_toy.py` | 식 2–3 토이 |
| `analysis/run_experiments.py`, `analyze_results.py`, `calib_size.py`, `pos0_check.py` | 모델 실험, 부트스트랩·Holm 분석, 보정 크기와 위치 0 진단 |
| `analysis/paper_arithmetic.py` | 논문 표 재계산, 메모리·속도 산술 |
| `analysis/*_results.json` 등 | 위 스크립트의 결과 |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install torch transformers numpy scipy matplotlib pyarrow pytest
python3 -m pytest analysis/test_minimal_impl.py -q
cd analysis
python3 saliency_toy.py
# 모델 실험 (가중치 openai-community/gpt2, EleutherAI/pythia-160m, 텍스트 NeelNanda/pile-10k, 최대 RSS 약 3.8 GB)
python3 run_experiments.py gpt2 gpt2_results.json
python3 run_experiments.py pythia160m pythia160m_results.json
python3 calib_size.py gpt2 gpt2_calib_size.json
python3 calib_size.py pythia160m pythia160m_calib_size.json
python3 pos0_check.py gpt2 gpt2_pos0.json
python3 pos0_check.py pythia160m pythia160m_pos0.json
python3 analyze_results.py
python3 paper_arithmetic.py
python3 plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2306.00978v6, arXiv 비독점 배포 라이선스(CC 아님). 그래프는 모두 우리가 그렸습니다.
- 공식 코드는 MIT이나 복사하거나 실행하지 않았고, 이 폴더의 구현은 논문 설명으로 직접 짰습니다.
