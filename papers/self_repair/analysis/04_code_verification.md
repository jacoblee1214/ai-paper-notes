# 04. 논문과 공식 코드 대조

공식 저장소 `github.com/theoxo/self-repair`를 얕게 clone해 **정적으로만** 읽었습니다(실행하지 않음). 대조한 커밋은 `fcda7a193a732d89fc1f610074c719682228fbc4`(2024-05-02)입니다. 저장소 상태: archived, LICENSE 파일 없음, 별 15.

범위: 논문 Sec 3(방법), Sec 4(설정), Appendix G(프롬프트)에 적힌 값과 코드의 기본값, 파이프라인 구조를 비교했습니다. 저장소 밖의 실행 스크립트(실제 실행 인자)는 없습니다.

## 1. 대조표

| 번호 | 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|---|
| D1 | APPS 실행 하네스 | 평가에 쓴 실행기 (언급 없음) | `src/apps/apps.py`의 `exec_sample`이 `NotImplementedError`를 던짐. "내부 프로젝트 의존으로 공개 불가"라고 적혀 있음. | 차이 (재현 불가 부분). APPS 생성·수리 파이프라인은 이 저장소만으로 실행되지 않습니다. 결과 파일(APPS 원자료)은 공개되어 있습니다. |
| D2 | 온도 | 0.8 (Sec 4) | `explain.py` 기본 0.8, `repair.py`의 `repair_code` 기본 0.8. 하지만 `repair.py` CLI 기본값은 0.5. | 부분 일치. 실제 실행 인자는 저장소에 없어서 0.8이 모든 실험에 쓰였는지 확인하지 못합니다. |
| D3 | 피드백 개수 n_f | 25 (Sec 3.2, Nf=25); 4.2 절은 10 | `explain.py`의 `explain_code(n=25)`. 원자료 APPS GPT-4·GPT-3.5 파일은 과제당 실패 프로그램마다 수리 25개, 혼합 파일은 10개. | 일치 (원자료 형태로 확인). |
| D4 | 수리 개수 n_r | 주 실험은 n_r = 1 (joint, Sec 3.2) | `repair_code(breadth=5)` 기본값. 원자료는 피드백마다 수리 1개 (25 = 25 피드백 x 1). | 기본값은 다르지만 실험 설정은 원자료로 맞습니다. 기본값을 실험에 썼다는 근거는 없습니다. |
| D5 | 피드백과 수리의 결합 | 주 실험은 joint (한 번 샘플) | APPS 경로는 설명 단계(`explain.py`)를 따로 하고, 그 설명을 수리 프롬프트에 넣습니다. HumanEval 쪽에는 `joint_explanation_*.txt` 프롬프트가 있습니다. | 부분 일치. joint 실험 구현은 HumanEval 쪽에만 보이고, APPS 쪽은 분리형입니다. 논문 Appendix G와 대조하려면 프롬프트 파일을 더 읽어야 합니다. |
| D6 | 샘플링 토큰 상한 | 명시 없음 | 설명 256, 수리 512 (`repair.py --max-tokens` 기본 512). | 논문에 없는 값입니다. 토큰 수 분석(Appendix A)에 영향을 줍니다. |
| D7 | top_p | 명시 없음 | `OAIModel.py` 기본 top_p = 0.95 | 논문에 없는 설정입니다. 재현 시 고정해야 합니다. |
| D8 | 복원추출 분석 | Np=50, Nf=25, Nt=1000, np ≤ 25 (자기 수리), np ≤ 50 (기준선) (Sec 3.2) | `paper/analysis/sample-and-estimate.py`: 기준선 `range(1, 51)`, 시드 2023, `--num-samples 1000`, `--with-replacement`. | 일치. 원자료 과제 300개, 기본 프로그램 50개, 수리 25개로 확인했습니다. |
| D9 | 인간 피드백 데이터 | 40프로그램 x 2 피드백 (Sec 4.3) | 저장소에 데이터 없음 (README: IRB). 부록 E 표만 있음. | 검산은 부록 E 표로만 할 수 있습니다 (`03` V1–V3). |
| D10 | HumanEval 수정본 | 논문 본문에서 수정본을 언급하는 부분을 찾지 못함 | README: 수정본(assertion 메시지 포함)은 gpg 암호화 파일로 제공, 원본 HumanEval로는 실행 안 됨. | 재현에 추가 준비가 필요합니다. 암호는 여기에 적지 않습니다. |
| D11 | 통과 기준 | 모든 단위 테스트 통과 | `check_pass(errors)`: 모든 테스트 결과가 "passed" | 일치. |
| D12 | 모델 스냅샷 | gpt-3.5-turbo-0301, gpt-4-0314 (각주 4) | `gpt3_5.py`, `gpt4.py`의 `MODEL_NAME`과 같음 | 일치. 스냅샷이 현재 제공되는지는 확인하지 않았습니다. |
| D13 | 분석 결과 파일 | 그림과 표 재현 가능 (README) | 결과는 git-lfs 포인터입니다. `git lfs pull`이 필요합니다. 이번에는 processed JSON 여러 개(APPS, HumanEval)와 APPS tarball을 받았습니다. | 저장소만 clone하면 그림 재현이 안 됩니다. |
| D14 | 라이선스 | 논문: CC BY 4.0 | 저장소에 LICENSE 없음 | 코드 재사용 조건이 불명확합니다. 코드를 게시본에 넣지 않았습니다. |

## 2. 판정 요약

- **재현 가능한 부분**: 분석 스크립트와 결과 원자료(APPS GPT-4, GPT-3.5, 혼합). 이번 검산(`03` V4–V11)은 이 원자료만으로 했습니다.
- **재현 불가 또는 어려운 부분**: APPS 실행 하네스(D1)는 공개되지 않았습니다. HumanEval은 수정본을 받아야 하고, 인간 실험 데이터는 공개되지 않았습니다.
- **실험 설정이 논문과 다를 수 있는 부분**: 온도 0.5/0.8(D2), 수리 개수 기본값(D4). 실행 인자는 저장소에 없어서 확인할 수 없습니다.
- **논문에 없는 설정**: top_p(D7), 토큰 상한(D6). 재현하려면 고정해야 합니다.

## 3. 재실행 방법 (참고)

```bash
git clone --depth 1 https://github.com/theoxo/self-repair.git repo
cd repo && git lfs pull        # 198 MB + 28 MB + 작은 파일들
cd paper/data/apps && tar -xjf apps-data.tar.bz2
```

`paper/data/apps/run-analysis.sh`는 APPS 전체 분석에 `APPS_DIR`(APPS 데이터셋 경로)을 요구합니다. 이번 검산에는 `APPS_DIR`이 필요 없었습니다. 과제 메타데이터(난이도)는 APPS 데이터셋에서 옵니다.
