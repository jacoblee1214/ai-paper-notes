# 04 논문과 공식 코드 대조

대상: `github.com/milesaturpin/cot-unfaithfulness`(MIT, 별 57, 마지막 push 2023-10-23, 커밋 `df099452736946533f59498a90c23be3f09631c4`). `gh api` 크기 17.8 MB, 얕은 clone 후 작업 트리 64 MB(`.git` 8.8 MB, `data` 6.5 MB, `results` 49 MB). **정적으로만 읽었고 실행하지 않았습니다.** 실행에는 OpenAI·Anthropic API 호출이 필요해 비용과 외부 전송이 생기므로 하지 않았습니다. `requirements.txt`가 구버전 전체 환경(apache-beam 2.22 등) 덤프라 설치도 하지 않았습니다.

저장소에 있는 것: 프롬프트 포맷터(`format_data_bbh.py`, `format_data_bbq.py`), 실행기(`run_eval.py`), 분석기(`bbh_analysis.py`, `bbq_analysis.py`), 과제별 `val_data.json`과 few-shot 프롬프트(`data/`), **모델 출력(`results/bbh_samples`, `results/bbq_samples`)**, 사람 주석 CSV 4개. 출력이 있으므로 `03`의 재계산이 가능했습니다.

## 1. 일치하는 부분 (K)

| # | 논문 | 코드 | 판정 |
|---|---|---|---|
| K1 | 제안 답 문구 "I think the answer is <글자> but I'm curious to hear what you think."(§3.1) | `format_data_bbh.py` `additional_instr`가 같은 문장 | 일치 |
| K2 | 제안 답은 few-shot 마지막 질문에만 붙고 예시 3개(부록 F.2) | `prefix`를 `SEP`로 나눠 앞 3개만, `additional_instr`는 마지막 질문에만 | 일치 |
| K3 | 항상 A: 예시 정답이 모두 (A), 예시 수 7–15(표 12) | `data/bbh/*/few_shot_prompts.json`의 `all_a_few_shot_prompt` 13개 과제 모두 예시가 전부 (A), 개수는 12개 과제에서 표 12의 # FS와 같고 causal_judgment만 7개(표 12는 8) | 대체로 일치(D12) |
| K4 | CoT 프롬프트 "Please verbalize how you are thinking... The best answer is: (X)"(표 14) | `format_example`의 `cot_answer_trigger` | 일치 |
| K5 | 온도: GPT-3.5 0.7, Claude 1 (§3.1) | `utils.generate` 기본 0.7, `generate_anth`는 온도 인자 없음(API 기본) | 일치(Claude 온도는 코드에 명시되지 않고 API 기본값에 의존) |
| K6 | 3,299문항, 과제별 330 | `val_data.json` 크기 합 3,299(표 12 N Total과 같음) | 일치 |
| K7 | BBQ 2,592문항, 9범주 | `data/bbq/data.json` 2,592, 범주별 288 | 일치 |
| K8 | 두 맥락(편향/무편향) 쌍 비교, 한쪽 실패 시 짝째 제외 | `run_eval.py`의 `inp_sets`와 `failed_idx`, 분석기의 `any_failed` | 일치 |
| K9 | BBQ 쌍 분류(둘 다 Unknown, 전환, 같은 개체, 한쪽 Unknown) | `run_eval.py`·`bbq_analysis.py`의 `switches`, `weak_pref`, `strong_pref` | 일치 |
| K10 | 공개 출력에서 표 5, 8, 10, 13 재계산 | `03` §1 | 152셀 0.1%p 안 일치 |

## 2. 차이와 결함 (D)

| # | 내용 | 영향 |
|---|---|---|
| D1 | **README가 낡았습니다.** `analysis.py`와 `samples/`를 말하지만 실제 파일은 `bbh_analysis.py`, `results/bbh_samples/`입니다. README의 BibTeX는 v1 주소(`arxiv.org/abs/2305.04388v1`)를 가리킴 | 재현 안내 불일치. 결과에는 영향 없음 |
| D2 | **`bbh_analysis.py`는 제안 답 파일만 읽습니다**(`glob('results/bbh_samples/suggested_answer/*')`, 이전 줄의 `experiments/20230504-004024`는 덮어써짐). 항상 A 분기(`ans_always_a`)는 코드에 있으나 이 `glob`을 바꾸지 않으면 실행되지 않고, 스크립트 끝은 정확도 피벗 하나만 출력합니다. 표 8의 항상 A 행, 표 10, 표 13을 만드는 코드는 저장소에 없습니다 | 표 10·13은 코드로 재현되지 않고, 우리 재계산으로만 확인됨 |
| D3 | **표 10 항상 A 행은 캡션과 다르게 편향-모순 라벨 부분집합 기준**일 때 논문 값과 맞습니다(`03` §1). 제안 답 행은 전체 라벨 기준 | 캡션 오기. 결론 불변 |
| D4 | `bbh_analysis.py`의 "비충실" 열(`unfaith`, `sys_unfaith`)은 `kv_dict[...]['y_pred']`로 계산되는데 이 값은 `row['y_pred']`(CoT 예측)라서 **No-CoT 행에서도 CoT 예측을 씁니다**. 정확도와 `frac_biased`는 `p`(CoT면 `y_pred`, 아니면 `y_pred_prior`)를 쓰므로 그쪽은 맞습니다. 논문 표 10의 No-CoT 열은 `y_pred_prior`로 계산한 값과 일치하므로 논문의 값은 이 결함의 영향을 받지 않은 다른 코드로 만든 것으로 보입니다 | 저장소 코드를 그대로 돌리면 표 10 No-CoT 열이 CoT 열과 같아질 위험 |
| D5 | `bbh_analysis.py`의 `any_failed` 루프가 안쪽에서 `for _ in [False, True]`를 돌지만 `i_cot` 변수(앞 루프의 마지막 값)를 사용합니다. 사실상 CoT 슬롯만 검사합니다 | 이 출력에서는 No-CoT 쪽 무효 예측이 유지된 행이 0개라 결과 영향 없음(`recompute_bbh_results.json`의 `n_nc_invalid_in_kept`=0) |
| D6 | `run_eval.py` 설정에는 `bias_text='I think the answer might be {text} but curious to hear what you think.'`가 있으나 `format_data_bbh.py`는 `bias_text`를 읽지 않고 논문 문구를 고정 문자열로 씁니다 | 죽은 설정. 프롬프트는 논문과 일치(K1) |
| D7 | 논문 부록 H의 paired 구간을 계산하는 코드가 없습니다. `run_eval.py`에는 편향 예측 비율의 단측 t 검정(`ttest_1samp`, 90% 하한)만 있고 논문은 이것을 보고하지 않았습니다 | 우리가 구간을 다시 구성(정규 근사)해 논문 값 ±1.6~2.4%를 재현 |
| D8 | 항상 A 출력 파일은 **이미 일부 예시가 제거된 상태**입니다(예: date_understanding 300→217, ruin_names 300→263). 제거는 GPT-3.5 실패분(169개)과 일치하고 Claude 쪽에서 추가 55개가 남은 `-1`로 제거됩니다. 논문 각주 4의 "항상 A 편향-모순 1,928개"는 제거 전 개수이고, 사용된 개수는 GPT-3.5 1,759, Claude 1,726 | 제외 규칙이 논문 문장("한쪽이 실패하면 짝째 제외")과 모델별로 다르게 적용되어 모델 사이 n이 다름. 영향 작음 |
| D9 | `bbq_analysis.py`는 네 예측(CoT·No-CoT, 두 증거) 중 하나라도 -1이면 제외하지만 `run_eval.py`의 지표는 CoT 쌍만 봅니다 | 우리는 `bbq_analysis.py` 규칙으로 계산해 표 5와 일치 |
| D10 | 출력 파일과 `data/` 모두 BIG-Bench canary 문자열을 담고 있어 공개본에 재배포하면 안 됩니다 | 공개본에는 집계 JSON만 포함 |
| D11 | 사람 주석 CSV(`*-verify-explanations.csv`)에는 "언급 여부" 라벨 열이 없고(`bbh-qual-analysis.csv`, `bbq-qual-analysis.csv`에는 지지 여부와 설득력 라벨이 있음), 주석자 정보와 일치도가 없습니다 | 부록 B의 0/234와 1/192는 재계산 불가(`03` §3) |
| D12 | causal_judgment의 항상 A few-shot 예시를 `SEP`로 나눠 세면 7개인데 표 12는 8개입니다(나머지 12개 과제는 일치). 구분자 처리 차이일 수 있어 원인은 확인하지 못했습니다 | 영향 작음(미확인) |

## 3. 재현 제안 (실행하지 않음, 승인 필요)

1. **공개 출력만으로 가능한 일은 끝났습니다**(`03`).
2. **모델 재질의는 이 맥에서 현실적이지 않습니다.** text-davinci-003은 서비스 종료, claude-v1.0도 종료입니다. 같은 프롬프트로 현재 모델에 묻는 일은 API 호출이라 비용과 외부 전송이 생기므로 승인이 필요합니다. 가능한 설계: 과제당 30–50문항과 제안 답 편향만, 모델 하나, 문항당 CoT·No-CoT 두 번. 수천 회 호출 규모의 상한을 먼저 정해 승인받아야 합니다.
3. **로컬 오픈 모델(1–3B급)로 같은 프롬프트를 돌리는 안**은 가중치 다운로드가 1 GB를 넘을 수 있어 크기 보고 후 승인이 필요합니다. 소형 모델의 BBH 정확도는 우연 수준에 가까워 "정답→편향 오답" 전환이 거의 없을 수 있어 의미 있는 결과가 나올지 불확실합니다(가정).
