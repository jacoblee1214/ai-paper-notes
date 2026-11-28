# SD3 (arXiv:2403.03206, ICML 2024 구두·최우수 논문상) 분석 노트

*Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (arXiv v1, 2024-03-05)의 통합 손실 식과 시간 표본 분포를 수치로 검산하고, 정확한 속도장을 가진 가우시안 혼합 토이와 멱법칙 스펙트럼 계산으로 해상도 이동 식(23)을 확인하고, 표 1·2·4·5를 다시 계산하고, 그림 8의 점을 읽어 스케일링과 상관을 재적합하고, 공식 추론 코드를 정적으로 대조한 자료입니다. **학습은 하지 않았습니다.** 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC가 아닙니다. 원문은 arXiv:2403.03206에서 받을 수 있습니다). 그래프는 모두 직접 그렸습니다.

## 구현 가능성 (요약)

- **논문 규모의 재현은 불가입니다.** 필터링한 내부 데이터와 8B 학습, 비공개 학습 코드가 필요합니다.
- **여기서 한 것은 학습 없는 검증입니다.** 닫힌 식, 정확한 속도장, 읽은 값의 재적합(CPU 수 분).
- **공식 코드는 추론 전용입니다.** `Stability-AI/sd3.5`(MIT)는 정적으로만 읽었고 이 저장소에는 넣지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 우리가 그린 그래프 다섯 장) |
| `analysis/01_paper_analysis.md` | 문제, 방법(식 번호), 실험 위치, 한계 |
| `analysis/02_prior_work.md` | 계보, 구성 요소 출처, 자료 표 |
| `analysis/03_analysis.md` | 식 검산, 시간 분포, 해석 토이, 해상도 이동, 표 재계산, 파라미터 수, 그림 8 재적합 |
| `analysis/04_code_verification.md` | 공식 추론 코드와 diffusers 학습 스크립트 정적 대조 D1–D16 |
| `analysis/loss_weighting.py` | 식 (10)–(22) 수치 검산과 환산 밀도 (numpy, scipy) |
| `analysis/gmm_toy.py` | 정확한 속도장 토이: 손실 바닥, 전이 구간, 오일러 오차 |
| `analysis/spectrum_shift.py` | 멱법칙 스펙트럼의 이동비 |
| `analysis/table_checks.py` | 표 1·2·4·5 재계산, 파라미터 수 |
| `analysis/fig8_digitize.py` | 그림 8의 래스터 패널에서 점 읽기 (poppler `pdfimages`, Pillow) |
| `analysis/scaling_refit.py` | 거듭제곱 재적합, 상관, 부트스트랩 |
| `analysis/code_checks.py` | 참조 코드의 시그마 스케줄 산술 |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 (matplotlib) |
| `analysis/*_results.json`, `fig8_points.json` | 결과 |
| `report/figures/` | 직접 그린 그래프 다섯 장 |

## 재실행

```bash
pip install numpy scipy matplotlib pillow
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2403.03206v1   # fig8_digitize.py가 읽음 (34 MB), poppler 필요
cd analysis
python3 loss_weighting.py && python3 gmm_toy.py && python3 spectrum_shift.py && python3 table_checks.py
python3 fig8_digitize.py ../paper/paper.pdf ../.scratch/f8
python3 scaling_refit.py && python3 code_checks.py && python3 plot_figures.py
```

난수는 `numpy.random.default_rng(0)`입니다. `fig8_points.json`은 이미 들어 있어 PDF 없이도 `scaling_refit.py`를 돌릴 수 있습니다.

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있습니다(CC 아님). 그래서 논문 그림과 원문 PDF는 이 저장소에 없습니다.
- 공식 코드 `github.com/Stability-AI/sd3.5`(MIT)와 `github.com/huggingface/diffusers`(Apache-2.0)는 정적으로 읽기만 했고 이 저장소에는 넣지 않았습니다. 가중치와 데이터는 받지 않았습니다.
