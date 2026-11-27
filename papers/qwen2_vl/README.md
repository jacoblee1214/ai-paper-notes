# Qwen2-VL (arXiv:2409.12191, 기술 보고서) 분석 노트

*Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution* (arXiv v2, 2024-10-03)을 M-RoPE 식 검산, 토큰 수·파라미터·계산량 산술, 벤치마크 표의 표본 오차 검토, 공식 코드(transformers, 공식 저장소) 정적 대조와 위치 번호 함수의 직접 호출로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다 (arXiv:2409.12191에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **학습과 대규모 벤치마크 평가의 재현은 이 분석에서 하지 않았습니다.** GPU 규모입니다.
- **위치 번호, 토큰 수, 파라미터, 통계 산술은 CPU에서 수 초입니다.**
- 공식 코드는 위치 번호 함수(`get_rope_index_2`) 하나만 호출했고 학습·추론은 하지 않았으며 가중치도 받지 않았습니다. 구조 설정(`config.json`)과 전처리 설정(`preprocessor_config.json`)만 받았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, v1 대 v2 차이 |
| `analysis/02_prior_work.md` | 계보(RoPE, NaViT, Qwen-VL, Qwen2.5-VL, Qwen3-VL), 구성 요소의 출처, 공식 저장소 표 |
| `analysis/03_analysis.md` | M-RoPE 검산, 토큰 수·파라미터·계산량, 표 2·7·8과 그림 5의 MDE, 주장 판정 C1–C12 |
| `analysis/04_code_verification.md` | 논문 대 코드 대조 (차이 D1–D10) |
| `analysis/mrope_check.py`, `mrope_results.json` | 위치 번호, 회전, 상대 위치 검산 (PyTorch, 수 초) |
| `analysis/rope_index_check.py`, `rope_index_results.json` | `get_rope_index` 실측 (D6): 네 구현의 위치 번호 비교 (torch, transformers, 가중치 불필요) |
| `analysis/token_arith.py`, `token_results.json` | smart_resize, 토큰 수, 파라미터, FLOPs (표준 라이브러리) |
| `analysis/bench_stats.py`, `bench_results.json` | Wilson 구간과 MDE |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |
| `report/figures/` | 우리가 그린 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib torch
python3 analysis/mrope_check.py     # torch 필요
HF_HUB_OFFLINE=1 python3 analysis/rope_index_check.py   # torch, transformers 필요. 소스 경로는 --main/--v445/--official (기본값은 repo/ 아래, 공개본에는 없으므로 직접 받아 지정)
python3 analysis/token_arith.py
python3 analysis/bench_stats.py
python3 analysis/plot_figures.py
```

`bench_stats.py`의 문항 수 중 일부는 Hugging Face datasets-server의 `size` API로 확인했고(스크립트 머리말에 기재), 나머지는 가정으로 표시했습니다. `mrope_check.py`는 transformers의 `get_rope_index` 규칙을 직접 다시 구현한 것이고, `rope_index_check.py`가 실제 함수의 출력과 원소 단위로 대조합니다.

## 출처와 라이선스

- 논문은 arXiv에 비독점 배포 라이선스(nonexclusive-distrib/1.0)로 올라 있어 CC BY가 아닙니다. 그래서 논문 그림을 싣지 않았습니다. 원문 PDF 텍스트에는 회사 저작권 문구가 없습니다.
- 공식 코드는 `github.com/QwenLM/Qwen2-VL`(현재 `QwenLM/Qwen3-VL`로 이름이 바뀜, Apache-2.0)과 `huggingface/transformers`(Apache-2.0)에 있습니다. 이 저장소에는 코드를 넣지 않았습니다.
- 모델 가중치는 2B·7B가 Apache-2.0, 72B가 Qwen 라이선스입니다(Hugging Face 표기). 받지 않았습니다.
