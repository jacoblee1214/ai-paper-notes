# Where LLM Agents Fail and How They can Learn From Failures (AgentDebug, arXiv:2509.25370) 분석 노트

Zhu et al. (ICML 2026 워크숍 포스터)을 공개 표와 그림 값으로 재계산하고, 공식 코드(`ulab-uiuc/AgentDebug`)를 정적으로 읽어 논문과 대조한 노트입니다. 논문 원문 PDF는 넣지 않았습니다 (arXiv:2509.25370에서 받을 수 있습니다). 논문 그림은 싣지 않았고(CC BY-NC-SA 4.0), 우리가 그린 그래프 두 장만 있습니다. 공식 코드와 모델은 실행하지 않았습니다.

## 검증 결과 (요약)

- 표 1의 모든 셀이 환경별 궤적 수(100/50/50)의 정수 개수와 맞습니다. "Average"는 매크로 평균이고 풀링하면 AgentDebug의 step 42.5, all-correct 23.5입니다. 본문 Findings의 50.0·42.5는 표와 맞지 않습니다.
- 임계 단계 탐지의 우위는 all-correct와 GAIA·풀링 step에서 어떤 불일치쌍 배치에서도 유의하고, ALFWorld·WebShop의 step 차이는 최악 배치 p 0.45, 0.41로 표본 오차 안일 수 있습니다. 상수 추측 기준선은 15/18/26%입니다.
- 비공식 HF 미러(200궤적)가 논문 그림 11–13과 거의 일치합니다(WebShop 1건 차이). "대부분 6–15단계"는 데이터와 다르고(1–5단계 54.5%), 오류 유형 라벨 27건이 비어 있습니다.
- "최대 26% 상대 향상"은 Qwen3-8B의 +26 포인트(상대 +54%)이고 Self-Refine과 동률입니다. 같은 시도 수 Best-of-N이 강한 두 백본에서 이득의 75–77%를 설명합니다(하위 과제 n=100 가정).
- 공식 코드는 탐지 단계만 공개합니다. 2단계는 반사실 탐색이 아니라 단일 LLM 호출이고, 회복 루프·기준선·평가는 없습니다. κ=0.55는 한 값이고 재계산할 수 없습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 실험, 주장과 근거 위치 |
| `analysis/02_prior_work.md` | 계보, 이 저장소의 관련 분석과의 관계, 발표 이후 연구, 자료 표 |
| `analysis/03_analysis.md` | 재계산과 검정 |
| `analysis/04_code_verification.md` | 논문 대 공식 코드 대조 (D1–D9) |
| `analysis/recompute_tables.py` | 재계산 스크립트 (pandas, pyarrow, scipy) |
| `analysis/plot_figures.py` | 그래프 생성 (matplotlib) |
| `report/figures/own_*.png` | 우리가 그린 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib pandas pyarrow
# 비공식 미러 데이터(선택, 18 MB)를 data/ 에 받는다
mkdir -p data && for s in train validation test; do curl -sL "https://huggingface.co/datasets/davide221/agenterrorbench/resolve/main/data/$s-00000-of-00001.parquet" -o data/$s.parquet; done
python3 -I analysis/recompute_tables.py   # recompute_tables_results.json (data/ 가 없으면 미러 부분 건너뜀)
python3 analysis/plot_figures.py
```

표 1과 그림 5–7의 입력은 스크립트 안에 옮겨 적은 값입니다.

## 출처와 라이선스

- 논문: arXiv 2509.25370 v1, CC BY-NC-SA 4.0 (arXiv abs 페이지 확인). 원문 PDF에 회사 저작권 고지는 없습니다.
- 공식 코드: github.com/ulab-uiuc/AgentDebug (MIT). 이 폴더에는 넣지 않았습니다. 공식 데이터는 저장소 README의 Google Drive 링크에 있고, 재계산에는 제3자 HF 미러를 썼습니다(공식과 같은지 보증하지 않음).
