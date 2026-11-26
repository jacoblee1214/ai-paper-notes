# DINOv2 (arXiv:2304.07193) 분석 노트

*DINOv2: Learning Robust Visual Features without Supervision* (Oquab, Darcet, Moutakanni 외, Meta, TMLR 2024, ICLR 2025 포스터로 발표)을 구현 모드로 분석한 노트입니다. 목적 함수의 구성 요소(DINO 이미지 손실, iBOT 가린 패치 손실, Sinkhorn-Knopp 중심화, KoLeo 정규화, EMA 교사, 멀티 크롭)를 PyTorch로 직접 짜 pytest 29개로 확인했고, 합성 도형 토이로 중심화·날카롭게 하기·KoLeo가 붕괴를 막는지 보았습니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 라이선스가 CC가 아니고 저자 소속이 회사라서, 원문은 arXiv:2304.07193에서 받을 수 있습니다). 공식 코드(`facebookresearch/dinov2`, Apache-2.0)는 정적으로 읽고 대조만 했고 이 폴더에 넣지 않았습니다.

## 검증 결과 (요약)

- **표 재계산**: OpenCLIP-G 대비 DINOv2 ViT-g는 인스턴스·밀집 23/23, 이미지 수준 18/26에서 앞서고, 증류 ViT-L은 이미지 수준 14/26입니다. 표 1의 SK 행은 k-NN, 선형 모두 변화가 없습니다. 표 14 탄소 산술은 일치하고 "10배"는 OpenCLIP-G(12.3배)에만 맞습니다.
- **식 검산**: 교차 엔트로피 = 교사 엔트로피 + KL, 중심화·날카롭게 하기가 서로 반대 방향의 붕괴를 막는 원리와 교사 엔트로피 근사(log K − s²/2τ²), Sinkhorn-Knopp 반복 수와 열 합 오차, KoLeo = Kozachenko-Leonenko 엔트로피 추정량의 아핀 변환, EMA 닫힌 꼴.
- **토이**(합성 도형 6종, 시드 5개, 논문 재현 아님): 결과는 `analysis/03_analysis.md` §5와 블로그 글에 있습니다.
- **코드 대조**: 논문 공백(교사 온도, 마스크, 머리 구조)을 코드로 보충했고, KoLeo가 두 전역 크롭에 합산되는 차이와 표 17의 ViT-B 블록 수 오기, 증류·데이터 정제 코드 비공개를 확인했습니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` | 문제, 방법, 실험, 주장 S1–S10과 근거 위치, 한계, v1 대 v2 |
| `analysis/02_prior_work.md` | 구성 요소별 출처, 이후 계보(registers, DINOv3, Web-SSL 등), 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 식 유도 검산, 최소 구현 설명과 공식 코드와의 차이, 토이 결과 |
| `analysis/04_claim_verification.md` | 이후 문헌의 주장 검증 C1–C12, 공식 코드 대조 D1–D14 |
| `analysis/minimal_impl.py` | 작은 ViT, DINO 머리, 손실 3종, SK, KoLeo, EMA, 멀티 크롭, 합성 이미지 (PyTorch) |
| `analysis/test_minimal_impl.py` | pytest 29개 (약 3초, CPU) |
| `analysis/formula_checks.py`, `formula_results.json` | 표 재계산, 식 유도의 수치 검산 |
| `analysis/toy_experiment.py`, `toy_results.json` | 합성 도형 토이: 7개 조건 x 시드 5개 (MPS) |
| `analysis/plot_figures.py`, `figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install torch numpy scipy matplotlib pytest
python3 -m pytest analysis/test_minimal_impl.py -q        # 29 passed
cd analysis
python3 formula_checks.py       # 수 초
python3 toy_experiment.py       # 7개 조건 x 시드 5개 (MPS, 수십 분. CPU면 훨씬 오래 걸림)
python3 plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(CC가 아님)이므로 논문 그림은 넣지 않았고, 그래프는 모두 직접 그렸습니다.
- 공식 코드 `github.com/facebookresearch/dinov2`는 Apache-2.0이며 이 저장소에 넣지 않았고 실행하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다. 토이 데이터는 코드가 직접 만든 합성 이미지입니다.
