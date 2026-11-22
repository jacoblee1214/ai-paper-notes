# Are "Solved Issues" in SWE-bench Really Solved Correctly? (arXiv:2503.15223, ICSE 2026) 분석 노트

공개 결과 파일(Zenodo 10.5281/zenodo.17074796)로 논문 표를 다시 계산하고, 외삽 수치에 구간을 붙이고, 공식 저장소를 정적으로 대조한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv 비독점 배포 라이선스).

## 재현 가능성

- 패치 실행, Docker, LLM 호출은 하지 않았습니다. 재계산은 공개 결과 파일만 씁니다.
- 결과 파일(약 40 MB 압축, 풀면 약 1.1 GB)은 이 폴더에 넣지 않았습니다. 받는 방법은 아래 재실행에 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트(게시용) |
| `analysis/01_paper_analysis.md` | 문제, 방법, RQ1–RQ4, 주장과 근거, 한계 |
| `analysis/02_prior_work.md` | 계보와 후속 연구, 공식 저장소 정보 |
| `analysis/03_analysis.md` | 표 재계산, 구간, 짝지은 검정, 외삽 구간 |
| `analysis/04_code_verification.md` | 공식 코드와 논문의 대조(D1–D11) |
| `analysis/*.py`, `analysis/*.json` | 재계산 스크립트와 결과 |

## 재실행

```bash
mkdir -p data/zenodo data/unz
curl -L -o data/zenodo/PatchDiff_0908.7z https://zenodo.org/records/17074796/files/PatchDiff_0908.7z
tar -xf data/zenodo/PatchDiff_0908.7z -C data/unz
python3 -I analysis/recompute_tables.py
python3 -I analysis/patch_size_analysis.py
python3 -I analysis/difftests_stats.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스입니다. 논문 그림은 넣지 않았고, 그래프는 직접 그렸습니다.
- 공식 저장소 ZJU-CTAG/PatchDiff에는 LICENSE가 없습니다. 코드를 넣지 않았고 실행하지 않았습니다.
- 결과 파일(Zenodo)은 CC BY 4.0입니다.
