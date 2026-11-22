# Large Language Models Cannot Self-Correct Reasoning Yet (arXiv:2310.01798) 분석 노트

Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet* (ICLR 2024)을 공개 표와 Figure 1 수치로 재계산하고, 비판 대상 선행 연구의 원문과 공개 코드를 읽어 주장을 검증한 노트입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2310.01798에서 받을 수 있습니다). 논문 그림은 싣지 않았고, 우리가 Figure 1 값에서 그린 그래프 한 장만 있습니다. 논문의 코드와 결과 파일은 공개되어 있지 않아 문항 단위 재계산은 하지 못했습니다.

## 검증 결과 (요약)

- 표 2–5, 7의 정확도 42개가 모두 문항 수의 정수 개수와 맞습니다.
- Figure 1의 변화 분포와 표 3·4가 8/8 설정에서 일치합니다. 이 분포에서 짝지은 분할표를 복원해 McNemar 검정을 했습니다.
- "라벨 없이는 모든 모델·벤치마크에서 하락"은 점추정으로만 맞습니다. 8개 설정 중 4개(GPT-3.5 CSQA, GPT-4 GSM8K, Llama-2 두 개)만 통계적으로 구분됩니다.
- 토론이 같은 응답 수의 자기 일관성보다 "유의하게" 못하다는 서술은 9응답에서만 성립합니다.
- 선행 연구 RCI는 v1에서 정답 라벨 사용을 밝히지 않았고, v2 부록, v3 본문 순으로 공개했습니다. Reflexion 공개 코드는 정답 신호를 프롬프트에 넣습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 설계, 실험, 주장과 근거 위치, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 계보, 논문 이후 문헌, 관련 저장소 |
| `analysis/03_analysis.md` | 재계산과 검정 |
| `analysis/04_claim_verification.md` | 주장별 판정 (C1–C9) |
| `analysis/recheck_tables.py` | 재계산 스크립트 (scipy) |
| `analysis/plot_figures.py` | 그래프 생성 (matplotlib) |
| `report/figures/own_paired_change.png` | Figure 1 복원 그래프 (우리가 그림) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recheck_tables.py   # recheck_tables_results.json
python3 analysis/plot_figures.py
```

입력은 스크립트 안에 옮겨 적은 논문 표와 Figure 1 값뿐이라 데이터를 따로 받을 필요가 없습니다.

## 출처와 라이선스

- 논문: arXiv 2310.01798, CC BY 4.0 (arXiv abs 페이지 확인). 원문 PDF에 회사 저작권 고지는 없습니다.
- 비판 대상 선행 연구의 코드(`noahshinn/reflexion`, `posgnu/rci-agent`, 둘 다 MIT)는 정적으로 읽었고 이 노트에 넣지 않았습니다.
