# Cambrian-1 (arXiv:2406.16860, NeurIPS 2024 구두) 분석 노트

*Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs* (arXiv v2, 2024-12-04)를 부록 표와 그림 재계산, 통계 검정(MDE), 벤치마크 군집 안정성, CV-Bench 주석 분석, SVA 산술과 토이, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2406.16860에서 받을 수 있습니다). 논문 그림은 차트인 그림 3과 도식인 그림 8만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문의 학습과 인코더 비교 실험(표 12–16)은 재현하지 않았고, 설정과 데이터 혼합이 공개되지 않아 재현할 수 없습니다.** 논문은 TPU-V4 128–512에서 학습합니다.
- **표·그림 재계산, 통계 검정, CV-Bench 주석 분석, SVA 산술은 CPU에서 수 초입니다.**
- 공식 코드는 실행하지 않았고 가중치도 받지 않았습니다. CV-Bench는 질문·답 주석(jsonl)만 사용했고 이미지는 받지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` | 메타·판본 차이, 문제, 방법, 실험 위치표, 한계 |
| `analysis/02_prior_work.md` | 계보, 이미지 없이 풀리는 벤치마크 비판, 후속, 자료 표 |
| `analysis/03_analysis.md` | 이미지 없음 비교, 군집 안정성, 인코더·결합·SVA 비교, CV-Bench, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조 (차이 D1–D10), 구현 가능성 |
| `analysis/parse_tables.py`, `analysis/tables.json` | 논문 부록 표 파싱과 전사 |
| `analysis/fig3_extract.py`, `analysis/fig3_values.json` | 그림 3 왼쪽 벡터 도형에서 값 복원 (pymupdf) |
| `analysis/cvbench_stats.py`, `analysis/cvbench_results.json` | CV-Bench 주석 분석 (주석 파일은 HF에서 받아 `data/`에 둠) |
| `analysis/recompute.py`, `analysis/recompute_results.json` | 재계산, 검정, 군집 부트스트랩 (numpy, scipy) |
| `analysis/sva_toy.py`, `analysis/sva_params_results.json` | SVA 구현 동치 토이와 파라미터 산술 (torch) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 두 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 네 장 (우리가 그림)과 논문 그림 두 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2406.16860v2 && pdftotext -layout paper/paper.pdf paper/paper.txt
mkdir -p data && for f in test_2d.jsonl test_3d.jsonl; do curl -L -o data/$f https://huggingface.co/datasets/nyu-visionx/CV-Bench/resolve/main/$f; done
python3 analysis/parse_tables.py
python3 analysis/fig3_extract.py
python3 -I analysis/cvbench_stats.py
python3 analysis/recompute.py
python3 -I analysis/plot_figures.py
python3 report/crop_figures.py
python analysis/sva_toy.py   # torch 필요
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv 라이선스 링크 확인). 원문 PDF에 회사 저작권 고지는 없습니다.
- 논문 그림 두 장 (그림 3 차트, 그림 8 도식)은 저작자 표기와 함께 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2406.16860"). CV-Bench 예시 이미지, 모델 출력 예시, 데이터셋 예시 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/cambrian-mllm/cambrian`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았습니다. CV-Bench 주석 파일도 넣지 않았고 위 명령으로 받을 수 있습니다.
- 가중치와 학습 데이터는 받지 않았습니다.
