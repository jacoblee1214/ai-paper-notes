# 02. 계보와 관련 자료

## 1. 계보 (논문이 스스로 쓰는 위치, §2와 §4.1 "Relation to Prior Work")

| 흐름 | 대표 문헌 (논문 번호) | MeanFlow와의 관계 |
|---|---|---|
| diffusion, score, probability-flow ODE | DDPM [19], score SDE [45], EDM [22] | 속도장 $v$가 정의되는 바탕. 반복 샘플링 |
| Flow Matching, rectified flow, stochastic interpolant | [28], [30], [1, 2] | 식 1–2, 선형 경로 $z_t=(1-t)x+t\epsilon$, 조건부 속도 $\epsilon-x$. $r=t$이면 MeanFlow = Flow Matching |
| 증류 기반 적은 스텝 | [39, 14, 41], score 증류 [32, 50, 53] | 선행 모델이 필요. MeanFlow는 "from scratch" |
| Consistency 계열 | CM [46], iCT [43], ECT [15], sCT [31], 일관성 흐름 매칭 [49] | 경로를 데이터 쪽에 고정($r\equiv0$), 시간 변수 하나. 네트워크 출력 성질을 제약. MeanFlow는 이 위에서 $(r,t)$ 두 변수로 일반화한 장을 학습 |
| 두 시간 변수 모델 | Flow Map Matching [3], Consistency Trajectory Models [23], Shortcut [13], IMM [52] | Flow map은 변위(displacement), MeanFlow는 변위/시간 간격. Shortcut과 IMM은 별도 자기 일관성 제약을 추가 |
| 구성 요소 출처 | 로짓 정규 시간 표집 [11, SD3], DiT 구조 [34], 적응 손실 가중 [15], CFG [18], VAE 잠재 [37] | 대부분 기존 부품 |

핵심 차이(논문 주장): 항등식(식 6)은 $v$와 $u$ 두 **장 사이**의 관계이고 네트워크와 무관하다. 그래서 목표가 네트워크에 독립이라고 말합니다. 우리의 검산(`03` §1)은 이 항등식과 JVP 형태가 맞음을 해석적 장에서 확인했습니다.

## 2. 이후 연구 (원문 미확인, 검색 결과와 공식 README의 제목 수준)

- **AlphaFlow** (arXiv:2510.20771, ICLR 2026 포스터, iclr.cc와 proceedings.iclr.cc 검색 결과 확인): MeanFlow 목적을 궤적 flow matching과 궤적 일관성으로 나누고 두 항의 그래디언트가 음의 상관이라고 분석, 둘을 잇는 커리큘럼을 제안. 검색 결과의 초록은 XL/2+에서 1-step 2.58, 2-step 2.15를 보고한다고 함. 이 분석에서 본문을 읽지 않음. MeanFlow의 "커리큘럼이 필요 없다"는 주장에 대한 직접적인 후속 반론 후보입니다.
- **Decoupled MeanFlow** (arXiv:2510.24474, `reading_list.md`에 ICLR 2026 △): 학습된 flow 모델을 flow map으로 바꿔 1–2 스텝 생성.
- **iMF**(improved MeanFlow, arXiv:2512.02012)와 **pMF**(pixel MeanFlow, arXiv:2601.22158): 공식 저장소 README의 갱신 로그에서만 확인(원문 미확인).
- **On Variance Reduction in Learning Mean Flows** (arXiv:2605.09235): 검색 결과 제목만 확인(원문 미확인).
- **The FID Lottery** (arXiv:2606.20536): SiT ImageNet 256 수백 개를 학습 시드·샘플링 시드로 나눠 FID를 무작위 변수로 본 연구. 검색 결과 요약에서 학습 시드 변화가 샘플링 시드 변화의 약 3.2배 크고 FID 변동계수가 약 1–2%라고 함(원문 미확인, `03` §5에서 가정으로만 사용).

## 3. 이 저장소와 이웃 저장소의 관련 분석

- `../sd3_rectified_flow/` (같은 묶음에서 작성 중): 로짓 정규 시간 표집과 rectified flow. MeanFlow의 $(r,t)$ 표집이 이 표집기를 빌려 쓰며(`03` §4), 이 분석에서 $r=t$ 표본의 $t$ 분포가 "두 번 뽑은 값의 최대"가 되는 효과를 계산했습니다.
- `../ddpm/`, `../score_sde/`, `../ddim/`, `../classifier_free_guidance/` (같은 묶음에서 작성 중): 속도장·확률 흐름 ODE·CFG의 출처.
- `../transfusion/`: 같은 축(생성)의 이웃. Transfusion은 diffusion 손실로 이미지를 생성하므로, 샘플링 스텝 수가 병목일 때 MeanFlow가 대안이 되는지는 비교한 적이 없습니다.

## 4. GitHub와 자료 표 (`gh api`, 2026-11-28 조회)

| 저장소 | 성격 | 라이선스 | 별 | 마지막 push | 크기 | 비고 |
|---|---|---|---|---|---|---|
| `Gsunshine/meanflow` | **공식**, JAX, TPU에서 작성·검증 | MIT | 628 | 2026-03-13 | 2.5 MB | `repo/`에 얕게 clone (커밋 `d70cb55`), 정적으로만 읽음. 가중치는 README의 B/4 체크포인트(Google Drive)만 |
| `Gsunshine/py-meanflow` | 공식 저자의 PyTorch, CIFAR-10 | NOASSERTION | 370 | 2025-07-30 | 2.8 MB | 읽지 않음 |
| `haidog-yaqub/MeanFlow` | 비공식 PyTorch, iMF 포함 | MIT | 1,221 | 2026-10-28 | 23.8 MB | 읽지 않음 |
| `zhuyu-cs/MeanFlow` | 비공식 PyTorch, ImageNet·CIFAR-10 | MIT | 522 | 2026-04-20 | 31.5 MB | 읽지 않음 |
| `Lyy-iiis/imeanflow`, `Lyy-iiis/pMF` | iMF, pMF의 JAX·PyTorch | README 링크만 | - | - | - | 조회하지 않음 |

공식 저장소 README의 "expected FID 11.4"는 제공된 MF-B/4 체크포인트의 평가 값입니다. 논문에는 해당 수치가 없고, 이 체크포인트를 어떤 설정으로 학습했는지도 적혀 있지 않습니다(저장소의 `run_b4.yml`은 240 에폭, 기본 ω=1.0, κ=0.5로 유효 안내 2.0이지만, 체크포인트가 이 설정인지는 확인할 수 없음).
