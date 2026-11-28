# MeanFlow (arXiv:2505.13447, NeurIPS 2025 구두) 분석 노트

*Mean Flows for One-step Generative Modeling* (arXiv v1, 2025-05-19)을 가우시안 혼합의 닫힌 식 속도장에서의 해석적 검산, 표 재계산, 공식 JAX 코드 정적 대조로 분석한 자료입니다. 모델 학습, 설치, 다운로드, 공식 코드 실행은 하지 않았습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2505.13447에서 받을 수 있습니다). 논문 그림은 도식인 그림 3과 차트인 그림 4만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 규모 재현은 불가입니다.** XL/2 1-NFE FID 3.43은 1.2M iter, 676M 파라미터, TPU 규모이고 공식 저장소의 설정 파일은 B/4 하나입니다.
- **해석적 검산은 CPU로 했습니다.** 항등식, JVP, 한 스텝 비교, 목표 연산자, CFG (PyTorch, 합쳐서 수 분).
- **공식 코드는** `Gsunshine/meanflow`(JAX, MIT)이고 이 저장소에는 넣지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 그림 3·4와 우리가 그린 그래프 네 장) |
| `analysis/01_paper_analysis.md` | 문제, 방법(식 번호), 실험(표 번호), 한계 |
| `analysis/02_prior_work.md` | 계보, 이후 연구, GitHub 표 |
| `analysis/03_analysis.md` | 식 유도 검산, 한 스텝 비교, 목표 연산자, (r,t) 표집, CFG, 표 재계산과 시드 잡음 가정 |
| `analysis/04_code_verification.md` | 공식 JAX 코드 대조: 일치 K1–K11, 차이 D1–D11 |
| `analysis/mf_exact.py` | 가우시안 혼합의 닫힌 식 속도장, 정확한 평균 속도 |
| `analysis/identity_check.py`, `onestep_check.py`, `target_fixedpoint.py`, `cfg_check.py` | 해석적 검산 (결과는 같은 이름의 `*_results.json`) |
| `analysis/paper_tables.py` | 표 재계산, 파라미터·FLOPs, 절제 대 잡음, FID 표본 잡음 토이 |
| `analysis/code_check.py` | 코드 마스크 논리의 numpy 시뮬레이션 |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 3, 4(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib torch
cd analysis
python3 identity_check.py && python3 onestep_check.py && python3 target_fixedpoint.py
python3 cfg_check.py && python3 paper_tables.py && python3 code_check.py && python3 plot_figures.py
```

모두 CPU이고 `target_fixedpoint.py`의 메모리가 가장 커서 약 1.1 GB입니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있고 원문 PDF에 저작권 고지가 없습니다(`pdftotext`로 확인).
- 논문 그림 3, 4는 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2505.13447").
- 공식 코드는 `github.com/Gsunshine/meanflow`(MIT)이고 정적으로만 읽었습니다. 가중치와 데이터는 받지 않았습니다.
