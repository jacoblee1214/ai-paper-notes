# AI Agents That Matter (arXiv:2407.01502) 분석 노트

Kapoor, Stroebl, Siegel, Nadgir, Narayanan, *AI Agents That Matter* (TMLR 2025)를 저자 공개 저장소(MIT)의 실행 결과로 다시 계산하고, 단순 기준선과 복잡한 에이전트의 "동률" 주장을 과제 단위 짝지은 비교와 동등 한계로 검정하고, 논문과 공식 코드를 정적으로 대조한 노트입니다. 논문 원문 PDF와 논문 그림은 싣지 않았습니다(arXiv:2407.01502v1에서 받을 수 있고, arXiv 라이선스가 CC BY가 아닙니다). 공식 코드는 실행하지 않았습니다.

## 검증 결과 (요약)

- 표 A1·A2·A3·A5의 값이 저자 공개 자료에서 모두 다시 나옵니다(정확도 0.063%p, 비용 0.005 USD 이내). 토큰 수 × 단가가 비용과 같고, LDB의 비용에는 시드 프로그램 생성비가 정확히 들어 있습니다.
- HumanEval에서 Warming(온도를 올리며 5회 재시도) 대 LDB (GPT-4)는 −0.12%p, 과제 단위 95% 구간 −2.3 ~ +2.0입니다. 동률은 ±2%p 이내에서만 성립하고 164문항의 최소 검출 차이는 약 3%p입니다. 변별하는 문항은 37개입니다.
- 디버깅의 순수 이득(LDB (GPT-4, GPT-3.5), 시드 대비 +1.3%p)이 단순 재시도(+2.3%p)보다 작습니다. 그 LDB 변형은 Warming보다 비용이 11% 낮습니다.
- 볼록 파레토 전선은 GPT-4 단가 0.06–3배와 2023-06판 모델 자료에서 변하지 않습니다. 비용 배수는 최대 1.7배 흔들립니다.
- HotPotQA 결합 최적화의 비용 절감(53–55%, 42–43%)은 확실하지만 정확도 유지는 ±6–9%p 이내이고 답 정확 일치로는 순서가 뒤집힙니다. 개발 정확도로 후보를 고르는 규칙은 Llama에서 더 큰 절감 후보를 놓쳤습니다.
- 본문과 표의 경미한 불일치: 손익분기 1,350 대 1,332, 홀드아웃 "계획 없음 7개"(표를 읽으면 5개), 파레토 정의(본문 대 부록).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 실험 설정, 주장과 근거 위치 |
| `analysis/02_prior_work.md` | 계보, 후속 문헌, 자료 표, 학회 |
| `analysis/03_analysis.md` | 재계산, 비용 산술, 동률 검정, 파레토, HotPotQA, NovelQA, 홀드아웃 |
| `analysis/04_code_verification.md` | 논문 대 공식 코드 정적 대조 |
| `analysis/recompute.py`, `recompute_results.json` | 재계산 스크립트와 결과 |
| `analysis/extract_pertask.py` | 저자 저장소 결과에서 과제별 통과 여부를 추출 (선택) |
| `analysis/plot_figures.py` | 그래프 생성 |
| `analysis/source_data/` | 저자 저장소(MIT)의 실행 집계 CSV 두 개, 우리가 추출한 과제별 결과 |
| `report/figures/` | 우리가 그린 그래프 두 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 -I analysis/recompute.py      # recompute_results.json (source_data만 필요)
python3 -I analysis/plot_figures.py
# source_data를 처음부터 만들려면(선택)
git clone --depth 1 https://github.com/benediktstroebl/agent-evals repo
python3 -I analysis/extract_pertask.py
```

## 출처와 라이선스

- 논문: arXiv 2407.01502 v1, arXiv 비독점 배포 라이선스(CC BY 아님). 원문 PDF에 회사 저작권 고지는 없습니다.
- 자료: 저자 저장소 `benediktstroebl/agent-evals`(MIT, 커밋 5ce7c0c)의 `plot_data_*.csv`와 결과 파일.
