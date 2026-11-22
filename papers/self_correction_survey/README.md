# 자기 수정 비판 서베이 (arXiv:2406.01297, TACL 2024) 분석 노트

*When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs* (arXiv v3, 2024-12-03)가 근거로 든 원 논문들의 표와 공개 코드를 다시 확인한 자료입니다. 서베이에는 새 실험이 없어서, 서베이의 판정(오라클 사용, 약한 초기 프롬프트, 불공정 비교)이 원 논문의 조건과 수치에 맞는지를 확인했습니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2406.01297에서 받을 수 있습니다). 논문 그림은 개념 도식 두 장만 넣었고 출처를 적었습니다.

## 요약

- 서베이의 불공정 판정은 원 논문의 표와 공개 코드로 확인됩니다 (RCI 정답 레이블 정지, Reflexion 완전 일치 신호, Self-Refine 대화의 약한 초기 지시문).
- 같은 벤치마크 22개 셀에서 오라클 이득이 자기 피드백 이득보다 모두 큽니다. 문항 단위 자료가 없어 구간은 독립 표본 근사입니다.
- 서베이가 구분하지 못한 사례(Self-Debug TransCoder의 채점 테스트 노출, CRITIC TabMWP)와 원 논문의 내부 불일치(RCI 기준선 변경, CRITIC 표 2 대 10, Self-Refine 표 9)를 정리했습니다.
- 이후 문헌은 초록 수준으로만 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장 포함) |
| `analysis/01_paper_analysis.md` | 서베이의 구조, 분류 틀, 점검표, 판본 차이, 한계 |
| `analysis/02_prior_work.md` | 계보, 발표 이후 연구, 자료 표 |
| `analysis/03_analysis.md` | 원 논문 표 재계산과 대조 (V1–V10) |
| `analysis/04_claim_verification.md` | 서베이 결론 C1–C8의 확인·수정·반박 |
| `analysis/recheck_cases.py` | 원 논문 표의 값으로 이득, 구간, 검정을 계산 (표준 라이브러리만, 수 초) |
| `analysis/plot_figures.py` | 오라클 격차 그래프 (matplotlib) |
| `report/crop_figures.py` | 논문 그림 두 장을 PDF에서 자르기 (pymupdf) |
| `report/figures/` | 그래프 한 장 (우리가 그림)과 논문 그림 두 장 (© 저자, CC BY 4.0) |

## 재실행

```bash
pip install matplotlib pymupdf
python3 analysis/recheck_cases.py
python3 analysis/plot_figures.py
python3 report/crop_figures.py   # paper/paper.pdf (arXiv:2406.01297v3) 필요
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다 (arXiv abs 페이지 확인). 원문 PDF에 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장 (그림 2, 3)은 저자가 그린 개념 도식이고 출처를 적어 실었습니다 ("© 저자, CC BY 4.0, 출처 arXiv:2406.01297").
- 확인에 쓴 원 논문(RCI, Reflexion, Self-Refine, CRITIC, Self-Debug 등)은 받아서 표만 읽었고 이 저장소에 넣지 않았습니다. Self-Refine 공개 코드(Apache-2.0)는 지시문 한 파일을 읽기만 했습니다.
