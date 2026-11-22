# 04. 공식 코드·결과 대조

대상: GitHub `ASSERT-KTH/agentic-evals-lab`(커밋 e1036f6, 2026-03-20, clone 1.7 MB, `repo/`)와 HF 데이터셋 `ASSERT-KTH/agentic-evals-artifacts`(CC BY 4.0, 전체 약 48 GB 중 평가 결과 JSON 120개 5.8 MB만 `data/`에 받음). **코드는 실행하지 않고 정적으로만 읽었습니다.** 스캐폴드 본체 `nano-agent`(별도 저장소)와 R2E-Gym은 읽지 않았습니다. 데이터 쪽 대조는 `analysis/recompute_runs.py`, `analysis/empty_patches.py`로 했습니다.

## 1. 저장소에 있는 것과 없는 것

- **있음:** 추론·평가 실행 스크립트(`benchmarks/swe_bench/run_nano_eval.py`, `run_nano_eval_multiple.sh`, `run_harness_eval.sh`), 모델별 설정 YAML 6개(`benchmarks/configs/`), SLURM 작업(`swe_bench_nano_infer_job.sh`), 학습(GRPO·SFT) 코드. 저장소 소개는 "RL로 에이전트를 학습·평가하는 프레임워크"이고 이 논문의 동반 저장소입니다.
- **없음:** pass@k·pass^k 계산, Shapiro-Wilk, 검정력 표, 첫 토큰 분기 분석, 그림 생성 코드(`grep -i "pass@k|shapiro|divergence|power"`가 학습 문서와 KL 항에서만 걸림). R2E-Gym 실행 코드도 없고, `run_all_r2egym_evals.sh`가 이미 생성된 `*r2egym*.json` 예측을 평가하는 스크립트뿐입니다.
- 라이선스 파일이 없습니다(GitHub API `license: null`). 재사용 허가가 명시되지 않았습니다.

## 2. 논문과 일치하는 점

| # | 항목 | 논문 | 코드·데이터 |
|---|---|---|---|
| K1 | 반복 구조 | 설정당 10회 독립 실행 | `swe_bench_nano_infer_job.sh`가 SLURM 배열 ID를 run 번호로 쓰고(`run_${TASK_ID}`) 주석에 "variance measurement with multiple runs, --array=0-9"라고 적음. 데이터는 설정당 run_0–run_9 |
| K2 | 한 실행 = 500문항 | §2.3 N=500 | 평가 JSON의 `total_instances`가 120개 전부 500 |
| K3 | 토큰·도구 상한 | 65,536 토큰(Qwen3, DeepSWE), 256,000(Devstral), nano-agent 500 도구 호출 (표 4, 부록 C) | YAML의 `token_limit` 65536 / 256000, `tool_limit` 500 |
| K4 | 온도 0 비결정성의 원인 | 추론 엔진 병렬·배치(§2.4.1) | DeepSWE YAML이 `--enable-prefix-caching`, `--tensor-parallel-size 8`, 작업자 48개(`max_workers: 48`) 병렬 실행. 같은 프롬프트라도 배치 구성에 따라 수치가 달라지는 설정과 부합 |
| K5 | 평가 | SWE-bench 하네스 | `run_harness_eval.sh`가 `swebench.harness.run_evaluation`을 `--timeout 3600`으로 호출 |
| K6 | 온도 | DeepSWE 1.0/0.0, Devstral-2 0.2/0.0 | YAML 값 일치 |
| K7 | 수치 | 표 1 | 공개 평가 JSON에서 48개 값 전부 재현 (`03` §1) |

## 3. 차이와 미해결 (D1–D7)

| # | 내용 | 영향 | 상태 |
|---|---|---|---|
| D1 | **Qwen3-32B 온도.** 논문 표 4와 표 1은 권장 온도 0.6인데 `benchmarks/configs/nano_qwen3-32b.yaml`은 `temperature: 1.0`(top_p 0.95, top_k 20)입니다. 데이터셋 카드는 "`__temp0` 접미사가 없으면 temperature 0.6"이라 쓰고(이것도 DeepSWE=1.0, Devstral=0.2와 어긋남), R2E-Gym 결과 파일명은 `temp_0.6`입니다 | nano-agent/Qwen3-32B 행(16.4 ± 0.7)이 실제로 어떤 온도의 값인지 모름. 이 행은 표에서 최소 σ(0.7)를 준 행이라 표 2의 σ 범위(0.7)에도 걸림 | **미해결.** 궤적 파일(`detailed_predictions.jsonl`, 설정 파일 없음 포함 여부 불명)을 받아 확인해야 함. 인용할 때 nano-agent/Qwen3-32B의 온도는 "0.6(논문)/1.0(저장소 설정)"로 적을 것 |
| D2 | **분석 코드 미공개.** pass@k, 표 2, 정규성, 분기 분석을 만드는 코드가 저장소에 없음 | 표 3·2와 그림 1은 우리가 독립 재구현으로 재현했으나(`03`), 그림 2·3(분기)는 코드도 데이터도 확인 못함 | 재현은 공개 결과 JSON으로 대체 |
| D3 | **결과 파일 이름이 설정을 잘못 가리킴.** nano-agent DeepSWE 온도 1.0 폴더(`nano-agent-agentica-org_DeepSWE-Preview/run_N/`)의 평가 JSON 이름이 `deepswe_32b_agent_swebv_eval_temp_0.0_run_N.deepswe_32b__r2egym__run_N…json`(temp_0.0, r2egym)입니다. 폴더 평균(31.4)은 논문의 nano-agent T=1.0 행과 일치하므로 폴더 이름이 맞고 파일 이름이 틀린 것으로 보입니다 | 파일명만 보고 합치면 설정이 뒤섞임 | 폴더 이름과 평균으로 매핑해 사용 |
| D4 | **시간 제한.** YAML에 `time_limit: 600`(nano-agent)이 있으나 논문은 토큰·도구 호출 상한만 보고합니다(부록 C). 벽시계 시간 제한은 서버 부하에 따라 같은 궤적도 다르게 끊기게 해 또 다른 실행 간 분산 원인이 됩니다 | 분산 원인 분해에서 "샘플링 분기"와 "인프라 시간"이 섞여 있을 수 있음 | 단위(초)와 적용 대상은 외부 패키지 nano-agent를 읽어야 확정됨. 미확인 |
| D5 | **재개 규칙.** `run_nano_eval.py`는 기존 `preds.jsonl`이 있으면 읽고, **패치가 비었거나 오류가 있는 문항을 다시 돌립니다**(`inputs_to_run`). 한 번에 끝나지 않은 실행은 "문항마다 한 번 시도"가 아니라 일부 문항에 재시도가 들어갑니다 | 해당 실행의 pass@1이 약간 높아지고 분산 추정이 바뀜. 몇 개 실행이 재개되었는지 데이터로는 알 수 없음 | 논문 미언급. 영향 크기는 미확인 |
| D6 | **빈 패치가 해결률 차이의 큰 부분을 설명.** 평가 JSON의 `empty_patch_instances`: DeepSWE-preview는 온도 0에서 nano-agent 평균 148개(500문항의 30%), R2E-Gym 115개이고 권장 온도 1.0에서는 42.5개, 11.3개입니다. 패치를 낸 문항 중 해결률은 nano 34.3% 대 28.9%(차이 5.4 %p), R2E 35.2% 대 24.9%(10.3 %p)입니다 | 논문은 11.0 %p, 15.3 %p 차이를 "stochastic exploration이 문제 해결 능력을 높인다"(§2.4.1)로 해석하는데, 패치를 낸 문항만 보면 차이가 nano에서 절반(11.0 → 5.4), R2E에서 약 3분의 2(15.3 → 10.3)로 줄어, 차이의 상당 부분은 온도 0에서 패치 자체를 못 내는 문항이 늘어서입니다. 탐욕 디코딩이 반복 루프나 상한 소진에 빠졌을 가능성이 있으나 이 데이터로는 원인을 가릴 수 없음 | 추가 확인 필요(궤적 파일에서 종료 사유). 인용할 때 "온도 효과"를 능력 향상으로 일반화하지 말 것 |
| D7 | **하네스 재평가 없음.** 같은 패치를 하네스에 두 번 넣었을 때 판정이 같은지(불안정 테스트)를 따로 재지 않았습니다. 실행 간 분산에는 에이전트 우연과 판정 잡음이 섞여 있음 | σ의 상한으로 해석 | 논문 미언급 |

빈 패치와 해결 수의 실행 간 상관은 대부분 유의하지 않습니다(스피어만, 12개 중 nano-agent/Qwen3-32B T=0.6만 ρ=−0.72, p=0.02이고 12개를 보정 없이 본 것). 즉 "빈 패치가 많은 실행이 곧 낮은 해결률의 실행"이라는 단순한 설명은 실행 내 분산에는 맞지 않고, D6은 **설정 간** 차이에서만 두드러집니다(`empty_patches_results.json`).

## 4. 구현 가능성 판정

- **이 맥에서 논문 수치 재현(궤적 생성)은 불가능합니다.** 8×A100 vLLM 서버, 문항별 SWE-bench 컨테이너(Apptainer/Docker), Mistral API가 필요하고 논문 자체가 3,500 GPU시간입니다. 하지 않았고 제안하지도 않습니다.
- **가능한 것:** 공개된 평가 결과 JSON 재계산(완료, 5.8 MB), 궤적 파일 일부 받아 온도·종료 사유 확인(D1·D6. nano-agent 궤적은 실행당 약 55–90 MB라 6개 설정에서 하나씩이면 약 0.4 GB, R2E-Gym 궤적은 실행당 약 0.76 GB라 6개면 약 4.6 GB로 1 GB 초과. 이번에는 받지 않았고 승인 후 제안).
