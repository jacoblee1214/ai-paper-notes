# 02. 계보와 관계: Are "Solved Issues" in SWE-bench Really Solved Correctly?

## 1. 계보

| 연도 | 논문·자료 | 관계 |
|---|---|---|
| 2015 | Qi et al., Smith et al., Martinez et al. (자동 프로그램 수리의 overfitting 패치 연구; 이 논문 참고문헌 [46], [50], [40]) | 테스트를 통과하지만 틀린 패치가 흔하다는 선행. 논문 §6은 Martinez et al.이 Defects4J에서 plausible 패치 84개 중 11개만 정답이라고 보고했다고 소개합니다. 비공개 테스트로 거르면 잡힌다는 Smith et al. 제안도 인용 |
| 2018 | Xiong et al. (실행 동작 유사도로 오답 패치 검출; 참고문헌 [58]) | PatchDiff와 같은 발상(정답 패치와의 동작 차이). 차이는 PatchDiff가 LLM으로 테스트를 만들고 대상이 이슈 해결 벤치마크라는 점 |
| 2023-10 | SWE-bench (arXiv:2310.06770, ICLR 2024) | 평가 대상 벤치마크 |
| 2024-08 | SWE-bench Verified (OpenAI) | 이 논문의 대상 500문항. 테스트가 이슈와 무관한 문항을 사람이 거름 |
| 2024-10 | SWE-Bench+ (Aleithan et al., arXiv:2410.06992) | 같은 약한 테스트 문제를 **도구 하나, SWE-bench 전체**에서 수작업 분류. 이 논문 §6이 차이를 밝힘. 후속 판은 ACM 3rd International Conference on AI-Powered Software 논문집(DOI 10.1145/3805760.3814924, Crossref 2026-07-05)에 실림 |
| 2025-03 | **이 논문** (arXiv:2503.15223, ICSE 2026) | 도구 세 개, Verified, 자동 차별 테스트(PatchDiff) + 수작업 77건 |
| 2025-06 | UTBoost (arXiv:2506.09289, ACL 2025 main, `aclanthology.org/2025.acl-long.189`) | 테스트를 LLM으로 보강. 검색 결과 요약 기준: 잘못 통과한 패치 345개(Lite 176, Verified 169), 리더보드 순위 변동(Lite 18, Verified 11). 이 논문과 같은 결론을 다른 방법으로 얻음. 원문은 읽지 않았습니다 |
| 2026-02-23 | OpenAI, "Why we no longer evaluate SWE-bench Verified" (`openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/`, 본문 확인) | 모델이 일관되게 못 푼 138문항(500의 27.6%)을 검토해 59.4%에 테스트 또는 명세의 중대한 결함이 있었다고 보고(좁은 테스트 35.5%, 넓은 테스트 18.8%). 정답 패치를 재현하는 오염 증거도 제시. Verified 보고를 중단 |
| 2026-02 | SWE-ABS (arXiv:2603.00520) | 초록 기준: 상위 30개 에이전트의 "해결" 패치 5개 중 1개가 의미상 틀림. Verified 500문항 중 50.2%의 테스트를 강화해 이전 통과 패치의 19.71%를 거부, 1위 점수 78.80% → 62.20% |
| 2026-04 | Probe to Generate (arXiv:2604.01518) | 초록 기준: 인스턴스의 77%에서 기존 테스트를 통과하는 변형(틀린 동작)이 존재. 보강 후 상위 10개 에이전트 해결률 4.2–9.0% 하락 |

## 2. 가까운 대안

- **Aleithan et al. SWE-Bench+**: 가장 가까운 선행. 논문 §6 요약대로 한 도구·전체 SWE-bench 대상이고 수작업 분류가 중심입니다. 이 논문은 Verified에서 도구 세 개를 보고 패턴을 분류합니다.
- **UTBoost, SWE-ABS, Probe to Generate**: 모두 "테스트를 강화해 다시 채점"하는 방향입니다. 이 논문은 정답 패치와의 **차이**를 찾는 방향이라 비교 기준(정답 패치)이 필요하고, 그 대신 테스트가 정답을 안다는 장점이 있습니다. 수치(SWE-ABS 상위 에이전트 78.80 → 62.20, Probe 상위 10개 4.2–9.0% 하락, 이 논문 suspicious 제외 시 17.3%p 하락과 확실한 오답 기준 6.4%p)는 대상 에이전트, 해결률 정의, 방법이 달라 직접 비교할 수 없습니다.
- **LIBRO, Pynguin, CoverUp**: 이 논문이 직접 비교한 기존 테스트 생성기(표 5, 6). LIBRO는 이슈 설명만 써서 5.9%, Pynguin·CoverUp은 0개.
- **Mokav, Differential Prompting** (논문 §6): 함수 수준 프로그램의 차분 테스트. 저장소 수준에 맞춘 대상 함수 선택이 PatchDiff의 차이입니다.

## 4. 구성 요소 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 평가 하네스, 로그 파서, 환경 설정 상수 | SWE-bench 하네스 코드 (`swebench.py`, `swebench_constants.py`, `swebench_log_parsers.py`) | README가 출처를 밝힘 |
| 도구별 패치 | SWE-bench 실험 저장소의 verified 제출물(`all_preds.jsonl`, `results.json`) | CodeStory 2024-12-21, Learn-by-interact 2025-01-10, OpenHands 2024-10-29 |
| 코드 파싱 | tree-sitter 0.23.2, tree-sitter-python 0.23.4, unidiff 0.7.5 | README |
| LLM 호출 | litellm 1.58.2, 기본 모델 gpt-4o-mini-2024-07-18 | `src/llm.py` |
| 차별 테스트 아이디어 | 차분 테스트(Differential Testing) 문헌 | §6 |
| 수작업 분류의 "atomic semantic change" 정렬 | 이 논문이 정의 | §4.3 |

## 5. 공식 저장소와 자료 정보 (GitHub API·Zenodo API, 2026-11-22 확인)

| 항목 | 값 |
|---|---|
| 저장소 | https://github.com/ZJU-CTAG/PatchDiff |
| 확인한 커밋 | `0a8ed608f9f19396e1bb3f6f3dd88b0fb88b49d6` |
| 별 | 3 |
| 마지막 push | 2025-09-08 |
| 라이선스 | **없음**(GitHub API `license: null`, 최상위에 LICENSE 파일 없음) |
| 결과 파일 | Zenodo `10.5281/zenodo.17074796`, 라이선스 CC BY 4.0, 파일 `PatchDiff_0908.7z` 40.7 MB(풀면 약 1.1 GB, 대부분 차별 테스트 JSON). 1 GB 미만 다운로드이므로 받아서 표를 재계산함 |
| 재실행 요구 | Docker 이미지(문항당 약 3 GB), 디스크 200 GB 이상, 전체 캐시 약 420 GB, OpenAI API 키 (README) |

저장소는 `repo/`에 얕게 clone했고(gitignore) 실행하지 않았습니다. Zenodo 파일은 `data/`(gitignore)에 있습니다.

## 6. 학회

ICSE 2026. arXiv 메타데이터의 DOI `10.1145/3744916.3764576`을 Crossref에서 조회하면 "Proceedings of the 2026 IEEE/ACM 48th International Conference on Software Engineering", 행사 "ICSE '26", 2026-04-12 게재, 유형 proceedings-article입니다. `reading_list.md` 14절의 △(프로그램에만 있음)를 Crossref DOI로 확인해 ✓로 올립니다. 연구 트랙(research track) 여부는 DOI만으로 구분되지 않지만 ICSE 본 논문집 항목입니다.
