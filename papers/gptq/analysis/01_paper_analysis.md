# 01. 논문 분석: GPTQ (arXiv:2210.17323)

*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (Frantar, Ashkboos, Hoefler, Alistarh). 분석 판은 arXiv v2(2023-03-22, ICLR 2023 camera-ready), 분석일 2026-11-28. 모드는 구현 모드(식 유도 검산, 최소 구현, 소형 모델 측정). 학회판 제목은 "OPTQ: Accurate Quantization for Generative Pre-trained Transformers"(아래 §6).

## 1. 문제

- 수십억 파라미터 생성 모델은 FP16로 저장해도 GPT3-175B가 326 GB(1024 배수 기준)라 단일 GPU에 안 올라갑니다(1절). 재학습이 필요한 압축은 이 규모에서 비싸므로, **재학습 없이 한 번에(one-shot) 3–4비트로 줄이는** 후처리 양자화(PTQ)가 목표입니다.
- 당시 이 규모에서 쓰이던 것은 가까운 값으로 반올림(RTN)뿐이었고(ZeroQuant, LLM.int8(), nuQmm; 2절), 8비트에서는 되지만 더 낮은 비트에서는 무너집니다. AdaRound, BRECQ, OBQ처럼 정확한 PTQ는 1억 파라미터급까지만 몇 GPU 시간에 돌았습니다(2절).
- 이 논문의 질문: **OBQ 수준의 정확도를 1000배 이상 싸게 얻어 175B에 쓸 수 있는가.**

## 2. 방법

### 2.1 층별 목적함수와 OBQ (3절)

층 ℓ의 가중치 W, 작은 보정 입력 X에 대해 **식 (1)**: argmin_Ŵ ‖WX − ŴX‖²₂. 양자화 격자는 미리 고정하고 가중치는 자유롭게 움직입니다. 식 (1)은 W의 **행별 제곱 오차의 합**이라 OBQ는 행을 독립으로 풉니다. 한 행에서 남은 가중치 집합을 F라 하면 Hessian은 H_F = 2 X_F X_Fᵀ이고, 탐욕 선택과 보상 갱신은 **식 (2)**:

- w_q = argmin_{w_q} (quant(w_q) − w_q)² / [H_F⁻¹]_qq,  δ_F = −(w_q − quant(w_q)) / [H_F⁻¹]_qq · (H_F⁻¹)_{:,q}

한 가중치를 고정한 뒤 H_F⁻¹은 **식 (3)**(가우스 소거 한 단계)로 갱신합니다: H⁻¹_{−q} = (H⁻¹ − H⁻¹_{:,q}H⁻¹_{q,:}/[H⁻¹]_qq)_{−p}. 이 방식의 비용은 d_row × d_col 행렬에서 O(d_row · d_col³)입니다. ResNet-50(25M)은 GPU 한 장에서 약 1시간입니다.

### 2.2 GPTQ의 세 가지 변경 (4절)

| 단계 | 내용 | 효과 |
|---|---|---|
| Step 1 임의 순서 | OBQ의 행별 탐욕 순서 대신 **모든 행이 같은 열 순서**로 양자화. 큰 층에서는 탐욕 순서의 이득이 작다는 관찰 | 모든 행이 같은 H_F⁻¹을 공유 → 식 (3)을 행·가중치마다가 아니라 **열마다 한 번**(d_col번). 비용 O(d_row·d_col³) → O(max{d_row·d_col², d_col³}) |
| Step 2 지연 배치 갱신 | B = 128열씩 묶어 블록 안에서만 갱신, 블록이 끝나면 나머지 열을 **식 (4), (5)**의 다중 가중치 형태로 한 번에 갱신 | 연산량은 같고 메모리 접근을 줄임(GPU 활용도). 논문 표현으로 큰 모델에서 "한 자릿수(order of magnitude)" 속도 향상 |
| Step 3 Cholesky 재구성 | 필요한 정보는 H_F⁻¹의 q행(대각부터)뿐이라 **Cholesky(H⁻¹)ᵀ**를 미리 한 번 계산. 평균 대각의 1%(λ)를 H 대각에 더하는 dampening 병용 | 식 (5)의 반복 적용이 쌓는 수치 오차(H⁻¹이 비양정치가 되어 갱신이 엉뚱한 방향으로 가는 현상, 수십억 파라미터에서 거의 항상 발생) 제거 |

**알고리즘 1**은 위 세 가지를 합친 의사코드이며, 격자는 **행별 비대칭 min-max**(5절 Setup, RTN과 같은 격자)입니다. 그룹 양자화(g개 연속 가중치마다 격자)는 "격자 선택과 독립"이라며 결합합니다(5절 Additional Tricks; 격자 매개변수는 그 시점의 갱신된 가중치로 잡음).

### 2.3 실행 구조 (5절 Setup)

- 보정 데이터: C4에서 무작위 2048토큰 구간 128개. 과제별 데이터를 쓰지 않으므로 "zero-shot".
- 트랜스포머 블록(층 6개) 하나씩 GPU에 올려 Hessian을 모으고 양자화한 뒤, **이미 양자화된 블록으로 다시 통과시켜 다음 블록의 입력**을 만듭니다(부분 양자화된 모델의 실제 입력 사용). 논문은 "눈에 띄는 개선이 거의 공짜"라고 씁니다.
- 임베딩과 출력층은 FP16 유지(5절 Practical Speedups). 모델 전체를 A100 80GB 한 장에서 양자화.

## 3. 실험과 결과 위치

| 표/그림 | 내용 | 핵심 수치 |
|---|---|---|
| 표 1 | ResNet18/50 4·3비트 PTQ 비교(AdaRound, AdaQuant, BRECQ, OBQ, GPTQ) | 4비트는 동급, 3비트는 OBQ·BRECQ보다 약간 낮음(RN18 67.88 대 68.69) |
| 표 8(부록 A.1) | OBQ와 직접 비교, BERT-base·OPT-125M | OPT-125M 4비트 OBQ 32.52, GPTQ 31.12. 3비트 69.32 대 53.85 |
| 표 2 | 전체 양자화 시간(A100 한 장) | OPT 13B 20.9분, 30B 44.9분, 66B 1.6시간, 175B 4.2시간. BLOOM 176B 3.8시간 |
| 표 3, 4, 그림 1 | OPT·BLOOM 전 계열 WikiText2 perplexity 3·4비트 | OPT-175B 4비트: RTN 10.54, GPTQ 8.37(FP16 8.34). 3비트 RTN 7.3e3, GPTQ 8.68 |
| 표 9–12 | PTB, C4 perplexity | 같은 경향 |
| 표 5 | OPT-175B, BLOOM-176B 요약(Wiki2, PTB, C4, LAMBADA)과 그룹(g1024, g128) | 3비트 g128: 8.45 / 12.37 / 10.36 |
| 그림 3, 부록 A.4 | LAMBADA·ARC·PIQA zero-shot | 3비트 RTN은 붕괴, GPTQ는 유지 |
| 표 6, 부록 A.2.2 | OPT-175B 3비트 단일 GPU 생성 지연 | A100 230→71 ms(3.24×), A6000 589→130 ms(4.53×), 3비트 메모리 약 63 GB + KV 약 9 GB |
| 표 7, 그림 4 | 2비트·삼진(그룹 8까지) | OPT-175B 2비트 g128 9.58, g32 8.94, 삼진 g8 9.20 |

## 4. 주장의 구조

1. GPTQ는 OBQ와 정확도가 비슷하되(표 1, 8) 훨씬 빠릅니다(표 2). 
2. 큰 모델일수록 양자화가 쉽습니다(표 3, 4; OPT-66B 예외는 죽은 유닛이라는 각주 2).
3. 3비트에서 RTN은 붕괴하고 GPTQ는 유지(표 3–5).
4. 그룹 격자와 잘 맞고 2비트 근처까지 내려갑니다(표 7).
5. 3비트 커널이 메모리 대역폭 이득으로 생성 속도를 높입니다(표 6). 연산량 감소는 아님.

## 5. 한계 (논문이 인정한 것과 우리가 본 것)

- **논문이 인정(6절):** 속도 이득은 메모리 이동 감소에서만 나오고 연산은 줄지 않음(FP16×INT4 혼합 정밀도 연산을 지원하는 하드웨어가 없음), 활성 양자화는 다루지 않음, 생성 과제와 perplexity 중심.
- **평가 편중:** 평가는 perplexity와 소수 zero-shot 과제입니다. 3비트 GPTQ가 이후 문헌에서 긴 문맥·추론·코드에서 어떻게 달라지는지는 이 논문의 범위 밖입니다(`02` 참조, 본문 미확인).
- **작은 모델은 더 취약:** 표 3에서 OPT-125M 3비트 GPTQ가 53.85(FP16 27.65의 1.95배)입니다. 같은 3비트가 175B에서는 +0.34입니다. 우리 소형 측정에서도 모델 크기와 구조에 따라 격차가 크게 달랐습니다(`03` §5).
- **원본 논문에 없는 것:** act-order(열을 활성 크기 내림차순으로)와 true-sequential(블록 안 층 순차)은 camera-ready 이후 공식 저장소에 추가된 휴리스틱입니다(공식 README). 이 분석에서 "GPTQ"라고 하면 논문 알고리즘 1(고정 열 순서)이고, act-order는 별도로 표시합니다.

## 6. 학회

- 공식 목록(iclr.cc virtual JSON `iclr-2023-orals-posters.json`)에서 이름 **"OPTQ: Accurate Quantization for Generative Pre-trained Transformers"**, 유형 Poster, decision "Accept: poster", 저자 4명 일치, OpenReview `forum?id=tcbBPnfwxS`로 확인했습니다(2026-11-28). arXiv 제목(GPTQ ...)과 학회 등재 제목(OPTQ ...)이 다릅니다(arXiv v2 PDF 머리글은 "Published as a conference paper at ICLR 2023"이고 제목은 GPTQ입니다. 개명 경위는 확인하지 않았습니다). `reading_list`의 "ICLR 2023 ○"는 **포스터**로 확정됩니다(구두 발표 아님).
- OpenReview API는 직접 조회하지 않았습니다(iclr.cc 목록으로 충분).
