# Transfusion (arXiv:2408.11039, ICLR 2025 구두) 분석 노트

*Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model* (arXiv v1, 2024-08-20)을 마스크·손실 토이, 그림 5 점의 재적합(동등 FLOP 비율과 군집 부트스트랩), 계산량 산술, 표 재계산으로 분석한 자료입니다. 공식 코드가 없어서 주장 검증으로 했습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2408.11039에서 받을 수 있습니다). 논문 그림은 차트인 그림 5 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 규모 재현은 불가입니다.** 380M 이미지(Shutterstock, 비공개)와 0.5–2T 토큰 학습이 필요하고 가중치도 공개되지 않았습니다.
- **구조 검증은 CPU 토이로 했습니다.** 마스크, 두 손실, BOI 전환 규칙(PyTorch, 약 6분).
- **공식 코드는 없습니다.** 비공식 구현(lucidrains/transfusion-pytorch, MIT)은 참고용입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용, 논문 그림 5와 우리가 그린 그래프 네 장) |
| `analysis/01_paper_analysis.md` | 방법, 데이터·학습, 실험, 한계 |
| `analysis/02_prior_work.md` | 계보와 자료 표 |
| `analysis/03_analysis.md` | 마스크·손실 검산, 6ND·파라미터·이미지 수, 동등 FLOP 비율 재적합, 평가 표본과 절제 효과 크기 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C14, 비공식 구현 대조 U1–U8 |
| `analysis/paper_tables.py` | 논문 표 3–9를 옮겨 적은 데이터 |
| `analysis/figure_extract.py` | PDF 벡터에서 그림 5의 점과 추세선 읽기 (pymupdf) |
| `analysis/scaling_refit.py` | 거듭제곱 재적합, 동등 FLOP 비율, 세 가지 부트스트랩 (numpy) |
| `analysis/compute_check.py` | 파라미터, 6ND, 이미지 수, 생성 FLOPs |
| `analysis/ablation_check.py` | 표 3–8 재계산, 절제 효과 크기, Llama SE |
| `analysis/fid_bias_toy.py` | FID 표본 수 편향 토이 |
| `analysis/mask_loss_toy.py` | 마스크 검산과 두 손실 토이 (PyTorch, CPU, 약 6분) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 5(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf torch
curl -sL https://arxiv.org/pdf/2408.11039 -o paper/paper.pdf   # figure_extract.py가 읽음 (14 MB)
cd analysis
python3 figure_extract.py && python3 scaling_refit.py && python3 compute_check.py && python3 ablation_check.py
python3 fid_bias_toy.py && python3 mask_loss_toy.py && python3 plot_figures.py
```

`figure_extract.py`는 `../paper/paper.pdf`를 읽습니다. `mask_loss_toy.py`만 torch가 필요하고 CPU에서 약 6분 걸립니다.

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있고 원문 PDF에 저작권 고지가 없습니다(`pdftotext`로 확인).
- 논문 그림 5는 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2408.11039").
- 공식 코드는 없습니다. 비공식 구현은 `github.com/lucidrains/transfusion-pytorch`(MIT)이고 이 저장소에는 넣지 않았습니다.
- 가중치와 데이터는 받지 않았습니다.
