# 02. 계보, 가까운 대안, 이후 문헌, 관련 분석

범위: 이 글이 무엇을 이어받았는지(§1), 가까운 이론과의 관계(§2), 이후 무엇을 바꿨는지(§3), 이 저장소의 다른 분석과의 관계(§4), 자료 표(§5). 이 글이 직접 인용한 문헌의 서술은 논문의 §Related Work와 참고문헌 번호에 근거하고, 이후 문헌은 arXiv 초록(2026-11-24 arXiv API로 받은 것)이나 해당 게시 페이지에서 읽은 것만 적었습니다. 본문을 읽지 않은 것은 "초록 기준"으로 표시합니다.


## 1. 계보

### 1.1 다의 뉴런과 "특징은 방향"이라는 관점

- 이 논문의 출발점은 Circuits 연구에서 본 다의 뉴런입니다. "Zoom In"(Olah et al. 2020, 논문 [1])과 "Softmax Linear Units"(Elhage et al. 2022, [2])가 superposition을 가설로 이미 언급했습니다(§Introduction, "A number of previous interpretability papers have speculated about it [1, 2]"). 저자들은 "특징 superposition이 신경망에서 명확히 시연된 적은 없다"고 적었습니다(§Definitions and Motivation).
- **새로운 점**은 가설을 장난감 모델에서 만들어 보이고, 언제 일어나는지(희소도·중요도의 상도표)와 어떤 기하로 일어나는지를 처음으로 정리한 것입니다. 개념 자체는 새롭지 않다고 저자들이 직접 말합니다.

### 1.2 압축 센싱, 분산 표현, 희소 코딩

논문 §Related Work(Compressed Sensing, Theories of Neural Coding and Representation)에 따름.

| 갈래 | 논문의 서술 | 관계 |
|---|---|---|
| 압축 센싱(Donoho 2006 [3], Candès-Tao 2005 [25]) | 희소 벡터는 저차원 사영에서 복원 가능. 하한 m = Ω(k log(n/k)). | 장난감 모델은 압축 센싱의 "더 약한 복원기"(한 번의 선형변환 + ReLU)라 하한이 superposition 양의 **상한**이 된다고 부록 정리 1에서 주장. 차이: 임베딩을 학습하고, 중요도가 있고, 희소도의 정의가 "비영 성분 수"가 아니라 "0일 확률" |
| 분산·밀집·집단 코드(Thorpe 1989 [4], Bengio et al. 2013 [5]) | 표현은 하나의 단위가 아니라 단위들의 패턴 | superposition은 이를 "더 많은 특징 개수"로 정량화 |
| 희소 코딩(Olshausen & Field 1997 [24]) | V1이 과완비 기저를 쓴다 | 데이터의 희소성의 근거 인용, 그리고 §Strategic Picture에서 "사후에 과완비 기저를 찾는다"의 선행으로 나옴 |
| Johnson-Lindenstrauss(§The Superposition Hypothesis) | 고차원에 exp(n)개의 거의 직교 벡터 | "특징은 거의 직교 방향"이라는 직관의 수학적 근거 |
| Cheung et al. 2019 "Superposition of many models into one" [7] | 파라미터에 여러 독립 모델을 담기("모델 수준 superposition") | 이름은 같고 문제는 다름(특징 수준 vs 모델 수준) |
| 풀림(disentanglement; Bengio 2013 [5] 이래) | 기저가 요인에 대응하게 | 저자는 superposition이 풀림이 어려운 이유일 수 있다고 추측(§Related Work) |

### 1.3 이론적 도구

- 선형 모델의 동역학: Saxe, McClelland, Ganguli 2014 [26], 2019 [31]. 이 논문은 선형 모델이 PCA와 같다는 사실의 직관으로 인용.
- Thomson 문제(구 위의 점 정렬): 간섭 항이 Thomson 문제와 닮았다는 비유(§Mathematical Understanding). 나온 정다면체·반프리즘 해와 차원수의 대응은 §Why These Geometric Structures?.
- 학습 동역학의 이산성: Olsson 2022(유도 머리 상전이, [27], 이 저장소 `induction_heads/`가 다룸), Nanda & Lieberum([28]), 그로킹([29]).

### 1.4 이 글의 위치

| 질문 | 이 글 이전 | 이 글 |
|---|---|---|
| 다의 뉴런은 왜 있나 | 가설(superposition) | 장난감 모델에서 재현, 희소도에 따라 나타남 |
| 얼마나 많은 특징이 겹치나 | 압축 센싱 상한 | 희소도·중요도의 함수로 측정, D_i의 계단 |
| 풀려면 | 말만 있음 | 세 갈래(§Three Ways Out): 모델 쪽, 사후 과완비 기저, 혼합 |

## 2. 가까운 대안과 이론

- **계산 능력 쪽**: 이 글의 §Computation in Superposition은 한 층에서 abs(x)를 근사하는 실험뿐입니다. 이후 Hänni et al.(arXiv:2408.05451, 초록 기준)은 입력 특징이 superposition일 때도 쌍별 AND를 Õ(m^{2/3}) 뉴런으로 계산하는 구성을 수학적으로 냅니다(`04` C7).
- **용량 쪽**: Scherlis et al.(arXiv:2210.01892, 초록 기준)은 같은 시기에 같은 현상을 "특징 용량(feature capacity; 특징이 차지하는 분수 차원)" 관점에서 다룹니다. 그들의 최적 용량 배분은 "가장 중요한 특징은 단의로, 덜 중요한 특징은 다의로, 가장 덜 중요한 것은 무시"입니다. 이 글의 상도표(안 배움/겹침/전담)와 같은 구조를 장난감 모델에서 독립적으로 얻은 결과로 읽히지만 두 글의 모델이 같은지는 확인하지 않았습니다.
- **다의성의 다른 원인**: Lecomte et al.(arXiv:2312.03096, 초록 기준)은 뉴런이 충분해도 정규화·잡음·무작위 초기화 때문에 다의성이 "우연히" 생긴다고 보입니다. superposition이 유일한 원인이 아니라는 지적입니다.

## 3. 이후 문헌이 바꾼 것

1. **사후 분해가 실제로 시도됨**: 이 글 §Three Ways Out의 "2번 과완비 기저 찾기"가 곧 희소 오토인코더(SAE)입니다. Cunningham et al.(arXiv:2309.08600), Bricken et al. "Towards Monosemanticity"(Transformer Circuits, 2023-10, 1층 트랜스포머의 512 뉴런 층을 4,000개 넘는 특징으로 분해), Templeton et al. "Scaling Monosemanticity"(2024-05, Claude 3 Sonnet 중간층, SAE 1M/4M/34M 특징)가 이어집니다. 이 저장소의 31번 [`sparse_autoencoders/`](../../sparse_autoencoders/)(예정)가 다룹니다(`04` C5).
2. **SAE의 한계 보고**: 같은 seed에서 다른 특징 집합(Paulo & Belrose, arXiv:2501.16615: 131K 잠재 변수 SAE에서 시드 간 공유 약 30%), 특징 흡수(Chanin et al., 2409.14507), 정규 단위가 아님(Leask et al., 2502.04878), 하류 작업에서 기준선을 못 이김(Kantamneni et al., 2502.16681), 무작위 기준선과 비슷함(Heap et al., 2501.17727; Korznikov et al., 2602.14111). 이는 "특징이 방향이고 SAE가 그것을 찾는다"는 이 글의 전략 서술을 수정합니다(`04` C5, C6).
3. **선형성 가정의 수정**: Engels et al.(arXiv:2405.14860)은 GPT-2·Mistral 7B에서 요일·월을 나타내는 원형(2차원) 특징을 찾고 계산에 쓰임을 보였습니다. 이 글은 §Features as Directions와 부록 "What about Multidimensional Features?"에서 다차원 특징을 논의했지만 장난감 실험은 1차원 특징만 다룹니다(`04` C4). Park et al.(arXiv:2311.03658)는 선형 표현의 의미를 반사실로 정식화합니다.
4. **규모와의 연결**: Liu, Liu, Gore(arXiv:2505.10465)가 이 글의 장난감 모델에 weight decay로 superposition 정도를 조절해 손실이 모델 차원에 반비례하는 "강한 superposition" 영역을 보이고, 공개 LLM이 이 영역에 있다고 주장합니다. 이 글의 §Open Questions "규모를 키우면 superposition이 사라지나"에 대한 한 답입니다(`04` C8).
5. **후속 장난감 연구**: Henighan et al. "Superposition, Memorization, and Double Descent"(Transformer Circuits, 2023-01, 이 글의 ReLU 출력 모델을 유한 데이터로 학습해 데이터셋 크기에 따라 "데이터점이 superposition에 저장되는 과적합 영역"과 "특징이 superposition에 저장되는 일반화 영역"과 이중 하강이 나타난다고 보고; 검색 결과 요약 기준).

## 4. 이 저장소의 다른 분석과의 관계

| 폴더 | 관계 |
|---|---|
| [`spike_sparse_sink/`](../../spike_sparse_sink/) (28번) | massive activation은 **소수의 채널이 큰 값을 독점**하는 현상이라 "특징이 뉴런/채널 기저에 정렬(privileged basis)"의 극단 사례로 볼 수 있다는 것이 우리의 해석입니다(이 글의 주장이 아님). 반대로 superposition은 특징이 기저와 무관한 방향에 겹쳐 담기는 경우입니다. 두 현상이 같은 층에서 어떻게 공존하는지는 측정한 적이 없습니다 |
| [`attention_sink_emerges/`](../../attention_sink_emerges/) (27번) | 어텐션 싱크의 "남는 확률 질량을 버리는 자리" 해석과 달리 이 글은 표현 용량에 대한 이론입니다. 직접 겹치는 주장은 없습니다. 싱크가 한 방향(첫 토큰의 큰 노름)에 몰리는 현상을 이 글의 "특징 이득-간섭" 틀로 쓰려면 먼저 그 방향이 특징인지부터 정해야 합니다 |
| [`induction_heads/`](../../induction_heads/) (29번) | 이 글의 §Superposition and Learning Dynamics가 유도 머리의 상전이([27])를 "학습 중 이산적 점프"의 선례로 인용합니다 |
| [`sparse_autoencoders/`](../../sparse_autoencoders/) (31번, 예정) | 이 글이 제기한 "superposition을 풀어야 한다"에 대한 구체적 수단. 이 글의 장난감 모델은 SAE의 **정답이 있는 시험대**로 쓸 수 있음 |

## 5. 자료 표

| 자료 | 라이선스 | 규모 | 마지막 갱신 | 비고 |
|---|---|---|---|---|
| arXiv:2209.10652 (v1) | arXiv 비독점 배포 라이선스(CC 아님) | PDF 4.9 MB | 2022-09-21 | 원문은 transformer-circuits.pub/2022/toy_model/index.html |
| `anthropics/toy-models-of-superposition` (GitHub) | MIT | 별 159, 파일: `toy_models.ipynb`(노트북 1개), README, LICENSE | 마지막 push 2022-09-14 | HEAD `562710e079704b84a132b640db134d4cebe22466`. 정적으로 읽음, 실행 안 함 (`04` D 절) |
| 논문이 가리키는 단계도 이론 노트북 | 확인하지 못함 | | | 본문에 링크 문구만 있음. 이 저장소에서는 구적법으로 직접 계산(`03` §6) |
