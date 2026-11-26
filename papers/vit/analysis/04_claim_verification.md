# 04. 주장 검증: 이후 문헌이 확인·수정·반박한 것 (C1–C9)과 공식 코드 대조 (D1–D8)

구현 모드의 04입니다. 이후 논문의 **초록**(arXiv API)과 본문 일부(DeiT, AugReg, Better plain ViT, Registers는 PDF를 내려받아 `pdftotext`로 읽음)를 근거로 했고, 모든 수치에 출처 논문을 붙였습니다. 이후 논문의 실험은 우리가 재현하지 않았고, 인용은 그 논문의 보고입니다. 웹 검색은 쓰지 않았습니다(arXiv API와 `curl`로 PDF 확인).

판정 기호: **확인**(이후 연구가 같은 방향), **수정**(조건이나 크기가 달라짐), **반박**(반대 결과), **미확인**(근거를 찾지 못함).

## 1. 요약표

| # | 원전 주장 (위치) | 판정 | 한 줄 이유 |
|---|---|---|---|
| C1 | 데이터가 적으면 ViT는 ResNet보다 몇 %p 낮다 (§1, §4.3, 그림 3·4) | **수정** | 증강·증류·레시피가 격차의 큰 몫. 토이에서도 증강이 구조보다 큼 |
| C2 | 대규모 사전학습이 귀납 편향을 이긴다 (§1, 그림 3·4) | **수정** | 증강+연산으로 데이터 약 10–25배를 대체(AugReg). 단 아주 작은 데이터에는 해당 안 됨 |
| C3 | 같은 성능에 연산 약 2–4배 적다 (§4.4, 그림 5) | **조건부 확인** | 표 6로 재계산하면 평균 2.8–3.0배, ImageNet만은 1.8–2.3배. 강한 ResNet 레시피에서는 격차 축소 |
| C4 | 1D·2D·상대 위치 임베딩 차이가 작다 (§3.1, 부록 D.4, 표 8) | **조건부 확인** | 짧은 학습(90 에폭)에서 2D sincos가 학습 1D보다 +1.5%p, 길게 하면 +0.4%p |
| C5 | [CLS]와 GAP는 비슷하다 (부록 D.3, 그림 9) | **수정** | ImageNet-1k 처음부터 학습에서는 GAP가 +1.5–1.8%p |
| C6 | 낮은 층에서도 일부 head가 전역 정보를 쓴다 (§4.5, 그림 7) | **확인** | Raghu et al. 2021 |
| C7 | (원전이 다루지 않음) 큰 ViT에서 고노름 토큰이 생긴다 | **새 문제** | Darcet et al. (ICLR 2024): 약 2% 토큰, 노름 약 10배, register로 해소 |
| C8 | 마스크 패치 예측 자기지도는 지도에 4% 뒤진다 (§4.6) | **수정** | MAE: ViT-H 87.8%(ImageNet-1K만) |
| C9 | 선형 패치 사영(식 1)으로 충분하다 (§3.1) | **수정** | 합성곱 stem이 최적화 안정성과 +1–2%p (Xiao et al.) |

## 2. 주장별 검증

### C1. "데이터가 적으면 ViT는 ResNet보다 못하다" — 수정

- **원전.** ImageNet만으로 사전학습하면 ViT-L이 ViT-B보다 낮고(그림 3), JFT 9M 부분집합에서 ViT-B/32가 ResNet50보다 훨씬 낮다(그림 4). 해석은 "합성곱 귀납 편향이 작은 데이터에서 유용"(§4.3). ViT-B/16의 ImageNet→ImageNet은 77.91%(표 5), 이때 규제는 weight decay, dropout, label smoothing 세 가지만 조정(§4.3).
- **DeiT** (Touvron et al., arXiv:2012.12877). 증강·정규화와 증류 토큰으로 ImageNet-1k만으로 ViT-B가 **증류 없이 81.8%(224), 83.1%(384)**, 증류 포함 최대 85.2%. 본문에 "ViT-B/16 [원전] 384에서 77.9"와 같은 표에서 대비하고, 증류 모델이 ImageNet-21k·JFT 사전학습 ViT-B(84.15%)를 넘는다고 적는다. 컨브넷 교사가 트랜스포머 교사보다 낫고 이를 "증류로 귀납 편향을 물려받는다"고 해석한다(§5.2).
- **Better plain ViT baselines** (Beyer et al., arXiv:2205.01580). ViT-S/16을 ImageNet-1k에서 표준 증강(RandAugment, Mixup) 등으로 **90 에폭 76.5%, 300 에폭 80.0%**(표 1). 같은 노트의 표 1에서 "원래 레시피"는 66.8/67.2/67.1, RandAugment+Mixup을 빼면 73.6/73.7/73.7이라 증강이 가장 큰 항목(−2.9 ~ −6.3%p)이다. 학습 시간은 TPUv3-8에서 90 에폭 약 6.5시간.
- **같은 조건의 ResNet도 올랐다.** ResNet strikes back(Wightman et al., arXiv:2110.00476): ResNet-50이 같은 224 해상도에서 **80.4%**(초록). 즉 "ViT는 못한다"의 비교 기준(ResNet 레시피)도 움직였고, 증강을 맞춘 비교에서는 격차가 레시피에 가려진다.
- **우리 토이** (`03` §5). 합성 도형 분류(0.1M 파라미터, 시드 5개)에서 증강 없이는 모든 학습 세트 크기에서 CNN이 ViT보다 앞서고(n=600에서 격차 46.5 ± 1.5%p, n=20000에서 10.7 ± 0.8%p) ViT가 앞서는 교차는 이 범위에서 없었다. **이동 증강**을 넣으면 ViT가 n=200–2000에서 +24~+35%p 올라 CNN과의 격차가 n=600에서 46.5 → 13.9%p로 줄지만, 증강이 있어도 격차는 약 10%p 남는다. ViT가 큰 n에서 학습이 덜 끝났고 하이퍼파라미터를 모델별로 조정하지 않아 구조 효과와 레시피 효과가 분리되지 않는다(`03` §5).
- **판정: 수정.** 원전의 관찰(규제를 거의 주지 않은 레시피에서는 ViT가 작은 데이터에 불리)은 맞지만, "작은 데이터에서 못한다"는 구조의 한계가 아니라 레시피의 한계에 가깝다는 쪽으로 이후 문헌이 옮겼다. 구조 효과를 인용하려면 "같은 증강·규제로 맞춘 비교"여야 한다.

### C2. "대규모 사전학습이 귀납 편향을 이긴다" — 수정

- **원전.** "large scale training trumps inductive bias"(§1), JFT 9M→300M에서 ViT가 ResNet을 역전(그림 4).
- **AugReg** (Steiner et al., arXiv:2106.10270). 증강과 규제(AugReg)를 쓴 ImageNet-1k 모델이 규제 없는 ImageNet-21k(10배 데이터)와 **대략 같고**, 연산도 늘린 ImageNet-21k+AugReg 모델이 규제 없는 JFT-300M(25배 데이터)과 **같거나 앞선다**(본문 4.1, 초록). 단서: 같은 논문은 "이 결과가 임의로 작은 데이터에는 성립하지 않는다"고 적고, ImageNet-21k에서 학습 예산을 고정하면 거의 모든 AugReg가 오히려 해롭고 연산도 함께 늘릴 때만 도움이 된다(그림 4 캡션).
- **Scaling Vision Transformers** (Zhai et al., arXiv:2106.04560, CVPR 2022). 20억 파라미터 ViT가 ImageNet 90.45%(초록): 원전의 "포화하지 않는다"(S8)와 같은 방향.
- **ConvNeXt** (Liu et al., arXiv:2201.03545, CVPR 2022). ResNet을 ViT 설계 쪽으로 "현대화"한 순수 합성곱이 ImageNet 87.8%로 Swin과 겨룬다. 이 논문의 문제의식은 "Transformer가 우월하다는 증거의 상당 부분이 구조가 아니라 레시피"라는 것.
- **판정: 수정.** "큰 데이터가 필요하다"는 "큰 데이터 또는 많은 연산과 증강"으로 바뀌었다. 구조 우월성이라는 읽기는 이후 문헌이 약화시켰다.

### C3. "같은 성능에 연산이 약 2–4배 적다" — 조건부 확인

- **우리 재계산** (`03` §3.2). 표 6 숫자로 ResNet(BiT) 효율 경계를 보간해 같은 정확도 비를 구하면 ImageNet 단독 1.8–2.3배(B/32, B/16, L/32, L/16), 5개 평균 2.8–3.0배(B/32, B/16)이고 L/32는 약 6배, 더 큰 모델은 ResNet 범위 밖이다. exaFLOPs 산식은 표 6의 ViT 행과 0.25% 안에서 맞는다(B/32만 +2.0%).
- **이후 문헌.** ResNet-50이 같은 모델 크기로 80.4%(ResNet strikes back), ConvNeXt가 Swin과 같은 FLOPs-정확도 곡선(ConvNeXt 초록), Better plain ViT는 ViT-S/16이 고전적 ResNet50 기준선과 비슷하다고(초록의 "similar to the classic ResNet50 baseline")만 적는다. 어느 것도 2–4배 격차를 재확인하지 않는다.
- **연산 척도의 한계.** 부록 D.5와 `03` §2: 224에서 attention이 3–4%라 FLOPs는 거의 선형 토큰 수이고, 원전이 말하듯 실제 시간은 FLOPs로 잘 예측되지 않는다(그림 12는 ViT와 비슷한 ResNet의 속도가 "comparable").
- **패치 크기가 연산 손잡이**: 패치 절반이면 토큰 4배, 연산 약 4배(`03` §2). FlexiViT(Beyer et al., arXiv:2212.08013)는 학습 때 패치 크기를 무작위화해 **가중치 하나로** 여러 패치 크기를 쓰게 하여 이 손잡이를 배포 시점에 옮겼다.
- **판정: 조건부 확인.** JFT 7–14 에폭의 BiT식 ResNet 기준으로는 우리 재계산이 2–3배 근처이지만(ImageNet 단독은 하한 근처), 레시피를 맞춘 ResNet과의 격차는 이후 문헌에서 확인되지 않는다. 인용 시 "JFT 사전학습, 해당 레시피에서"를 붙여야 한다.

### C4. "1D, 2D, 상대 위치 임베딩의 차이는 작다" — 조건부 확인

- **원전.** 표 8(ViT-B/16, ImageNet 5-shot): 없음 0.61382, 1D 0.64206, 2D 0.64001, 상대 0.64032(입력에서만 더함). 위치 정보가 없을 때만 −2.8%p이고 나머지 사이는 0.002 이내. 이유를 "패치 수준 입력이라 14×14에서 2D 관계를 쉽게 배운다"고 추측한다.
- **Better plain ViT** (표 1, ViT-S/16, ImageNet-1k). 2D sin-cos를 학습 1D로 바꾸면 90/150/300 에폭에서 **−1.5/−0.5/−0.4%p**: 짧게 학습하면 차이가 보이고 길게 하면 줄어든다. 이 노트는 시드나 구간을 적지 않아 효과가 노이즈 위인지 판별할 수 없다.
- **고해상도·가변 해상도에서의 역할.** 위치 임베딩 보간(§3.2)은 `03` §4.2처럼 임베딩이 부드러워야 손실이 작다. NaViT(arXiv:2307.06304)는 원 해상도·종횡비 입력을 쓰고, 같은 저장소의 `../qwen2_vl/`은 ViT에 2D-RoPE를 쓴다. 그 분석에 따르면 naive dynamic resolution(원 해상도 가변 토큰 수) 구성의 일부다.
- **판정: 조건부 확인.** "구분이 작다"는 5-shot, 7 에폭, 점추정의 결과이고, 학습이 짧은 조건에서는 2D sincos가 앞선다는 보고가 있다. 해상도가 변하는 조건은 직접 비교를 찾지 못했고, NaViT와 Qwen2-VL이 2D 구조를 위치 표현에 넣은 설계가 간접 증거일 뿐이다.

### C5. "[CLS]와 GAP는 비슷하다" — 수정

- **원전** (부록 D.3, 그림 9). JFT 7 에폭, 선형 5-shot에서 두 방식 모두 비슷하게 잘 된다. 단 "처음의 GAP 시도가 매우 나빴던 것은 [CLS]나 GAP 탓이 아니라 학습률이 달라야 했기 때문"(lr 8e-4 대 3e-4를 따로 맞춤). v2에서 추가된 부록이다.
- **Better plain ViT** (표 1, ViT-S/16, ImageNet-1k). GAP를 [CLS]로 바꾸면 **−1.5/−1.6/−1.8%p**(90/150/300 에폭). 이 노트의 개선 목록에 "GAP로 교체"가 들어 있다. MLP 헤드로 되돌려도 개선 값과 +0.2/+0.1/−0.2%p로 사실상 무차이.
- **Do Vision Transformers See Like CNNs?** (Raghu et al., arXiv:2108.08810). 분류 방식(CLS 대 GAP)이 공간 정보 보존에 눈에 띄는 영향(초록).
- **판정: 수정.** 원전의 "비슷하다"는 JFT 5-shot, 학습률을 따로 맞춘 조건이고, ImageNet-1k 처음부터 학습에서는 GAP가 약 1.5–1.8%p 앞선다는 보고가 있다. 공식 저장소도 GAP 설정(ViT-S/16, B/16 `*-gap-norep`)을 따로 제공한다(`D6`).

### C6. "낮은 층에서도 일부 head가 이미지 전체를 본다" — 확인

- **원전.** 평균 attention 거리(그림 7 오른쪽, 부록 D.7 그림 11: 128장 평균): 일부 head가 가장 낮은 층에서도 이미지 대부분에 attention하고, 다른 head는 국소적이며, 하이브리드에서는 국소 head가 줄어든다.
- **Raghu et al.** (arXiv:2108.08810). self-attention이 "전역 정보의 이른 집적"을 가능하게 하고 ViT의 잔차 연결이 낮은 층 특징을 높은 층으로 강하게 전달한다고 확인(초록). 또 CNN과 달리 ViT 표현은 층 사이에서 더 균일.
- **Cordonnier et al.** (arXiv:1911.03584). 충분한 수의 head를 가진 다층 self-attention이 합성곱을 표현할 수 있고 실제로 픽셀 격자 패턴에 attention한다(초록). 국소 head의 존재와 같은 방향.
- **판정: 확인.** 우리가 직접 확인하지는 않았다(가중치 필요).

### C7. 고노름 토큰과 register (원전이 다루지 않은 문제)

- **Vision Transformers Need Registers** (Darcet et al., arXiv:2309.16588, ICLR 2024, 본문 "Published as a conference paper at ICLR 2024"). 지도(DeiT-III), 텍스트 지도(OpenCLIP), 자기지도(DINOv2)로 학습한 ViT의 특징 지도에서, 정보량이 낮은 배경 패치에 **출력 노름이 약 10배인 토큰이 전체의 약 2%**(DINOv2 ViT-g에서 노름 150 초과 2.37%) 나타난다. 모델 층의 중간쯤(40층 중 약 15층)부터 생기고, 학습의 3분의 1 이후에 나타나며, 모델 크기는 ViT-L 이상에서만(DINOv2 6종 중 큰 3개). 이 토큰은 위치와 원래 픽셀 정보가 적고 이미지 전체에 대한 선형 분류 정확도는 높아서 "전역 정보를 모으는 데 재활용된다"고 해석한다. 입력에 추가 토큰(register)을 주면 고노름 토큰이 사라지고 밀집 예측이 좋아지며 attention 지도가 부드러워진다. DINO(v1)는 예외.
- **Massive Activations in LLMs** (Sun et al., arXiv:2402.17762, COLM 2024). LLM의 소수 활성이 100,000배 클 수 있고 상수에 가까운 bias처럼 작동하며 attention이 그 토큰에 집중된다. ViT에서도 연구했다고 초록에 적었다(구체 수치는 읽지 못함).
- **Vision Transformers Don't Need Trained Registers** (Jiang et al., arXiv:2506.08010, NeurIPS 2025 스포트라이트). CLIP, DINOv2에서 소수 "register neuron"이 고노름 활성을 이상치 토큰에 집중시킨다는 것을 보이고, 이 활성을 추가 토큰으로 옮겨 **재학습 없이** 같은 효과를 낸다(초록).
- **이 저장소의 sink 분석과의 관계** (두 폴더 README를 읽기만 함). `../attention_sink_emerges/`는 LLM의 첫 토큰 attention sink가 key bias처럼 작동하고, softmax 정규화를 없애면(sigmoid attention) 생기지 않으며 충분한 데이터 없이는 생기지 않는다고 정리한다. `../spike_sparse_sink/`는 사전학습 LLM 7개에서 massive activation(spike)을 초기 FFN 블록이 주입하고 후반 블록이 상쇄하며, 둘의 공출현이 pre-norm이 이어 주는 구조적 우연이라고 주장한다. register 문제와 공통점: (1) 소수 토큰에 큰 노름과 attention이 몰린다, (2) 모델이 충분히 크고 오래 학습해야 나타난다(Darcet: ViT-L 이상, 학습 1/3 이후), (3) 추가 토큰 또는 정규화 변경으로 사라진다(register 대 sigmoid attention). 차이: LLM의 sink는 **위치 0(첫 토큰)에 고정**이고 ViT의 고노름 토큰은 **내용이 낮은 배경 패치에 위치가 입력마다 달라진다**. ViT가 원전처럼 pre-norm(식 2, 3)이므로 spike–sink 메커니즘이 같은 원인일 가능성은 **가설**이고, 이 저장소에서 검증하지 않았다.
- **원전과의 관계(우리의 해석).** 원전은 전역 정보를 모으는 자리가 [CLS] 하나(식 4)이고 논문의 attention 지도(그림 6)는 출력 토큰에서 입력으로 본 것이다. 고노름 토큰의 존재는 전역 정보를 담는 보조 토큰 수가 부족할 때 모델이 패치 토큰을 임시 저장소로 재활용한다는 Darcet의 해석과 맞지만, 원전 구조가 원인이라는 직접 증거는 없다(Darcet 본문이 DINO v1은 생기지 않는다고 하므로 구조 단독 원인이 아님).
- **판정: 새 문제.** 원전의 attention 거리·attention 지도 해석(§4.5)을 큰 모델에서 읽을 때는 이상치 토큰을 점검해야 한다.

### C8. "마스크 패치 예측 자기지도는 지도에 4% 뒤진다" — 수정

- **원전** (§4.6, 부록 B.1.2): ViT-B/16이 79.9%, 처음부터 학습보다 +2%, 지도 사전학습보다 −4%.
- **BEiT** (Bao et al., arXiv:2106.08254)는 패치를 이산 토큰으로 바꿔 예측한다. **MAE** (He et al., arXiv:2111.06377)는 패치 75%를 가리고 보이는 패치만 인코더에 넣어 학습을 3배 이상 빠르게 하며, vanilla ViT-Huge가 ImageNet-1K만으로 **87.8%**(초록). 하류 과제 전이가 지도 사전학습보다 좋다고 보고한다(초록). 모델 크기(원전 ViT-B 79.9% 대 ViT-H 87.8%)가 달라 두 수치를 직접 빼면 안 된다.
- **판정: 수정.** 원전이 예고한 방향("future work")이 맞았고 수치는 크게 달라졌다.

### C9. "선형 패치 사영으로 충분하다" — 수정

- **원전.** 식 1의 선형 사영이 최소 수정 원칙.
- **Early Convolutions Help Transformers See Better** (Xiao et al., arXiv:2106.14881, NeurIPS 2021). 16×16 stride-16 패치 사영이 최적화 민감성(옵티마이저, 하이퍼파라미터, 학습 길이)의 원인이라고 보고, 3×3 stride-2 합성곱 몇 개로 바꾸면 안정성이 높아지고 ImageNet-1k 정확도가 약 1–2%p 오르며 FLOPs와 속도는 유지된다(초록).
- **판정: 수정.** 단 이후 대부분의 VLM 인코더(SigLIP 등)는 선형 패치 사영을 그대로 쓴다. 안정성 문제는 레시피로도 완화된다(Better plain ViT, AugReg).

## 3. 공식 코드 대조 (`google-research/vision_transformer`, 커밋 `64801f1`, Apache-2.0, 정적으로만 읽음)

핵심 파일은 `vit_jax/models_vit.py`(308줄), `vit_jax/checkpoint.py`의 `interpolate_posembed`, `vit_jax/configs/{models,common}.py`. **이 저장소는 미세조정·추론 코드이고 JFT 사전학습 코드는 없다**(README가 사전학습은 `big_vision`이라고 적음). 가중치는 받지 않았고 실행하지 않았다.

| # | 항목 | 논문 | 공식 코드 | 우리 최소 구현 | 판정 |
|---|---|---|---|---|---|
| D1 | 패치 임베딩 | 선형 사영 E (식 1) | `nn.Conv(features=hidden, kernel=stride=patch)` (코드 주석 "merge s2d+emb into a single conv; it's the same") | conv와 patchify+linear 둘 다, 테스트로 같음 확인 | 일치 |
| D2 | [CLS]와 위치 임베딩 | 학습 벡터, 학습 1D | `cls`는 **0 초기화**, 위치 임베딩은 N(0, 0.02²)("from BERT"), `AddPositionEmbs` 뒤 dropout | 같은 초기화 | 일치(논문에 초기화 값 없음) |
| D3 | 인코더 블록 | pre-norm, MSA, GELU MLP | `Encoder1DBlock`: LN → `MultiHeadDotProductAttention` → dropout → 잔차, LN → `MlpBlock` → 잔차. MLP 가중치 Xavier uniform, 편향 N(0, 10⁻⁶) | torch 기본 초기화(Kaiming uniform) | 구조 일치, 초기화 다름 |
| D4 | 마지막 LN | y = LN(z_L^0) (식 4) | `encoder_norm`을 **모든 토큰**에 적용한 뒤 `x[:, 0]` | 같음(LN은 토큰별이라 동치) | 일치 |
| D5 | 풀링과 헤드 | [CLS] + 사전학습 은닉 1층 MLP, 미세조정은 0 초기화 선형층 | `classifier`: `token`(기본)/`gap`/`unpooled`/`token_unpooled`, `representation_size`로 `pre_logits`(Dense+**tanh**) 선택, 헤드는 항상 **0 초기화**. `configs/*.py`에서 `representation_size`가 None이 아닌 설정은 grep으로 찾지 못함(모두 None) | 단일 선형 헤드, 0 초기화 | 논문 부록 D.3의 tanh MLP는 설정으로 선택 |
| D6 | [CLS] 대 GAP | 부록 D.3 | `get_s16_gap_norep_config` 등 GAP 설정 별도 제공 | `pool="gap"` | 있음 |
| D7 | 위치 임베딩 보간 | "원래 이미지 위치에 따라 2D 보간"(§3.2) | `interpolate_posembed`: 격자를 `sqrt`로 복원하고 `scipy.ndimage.zoom(order=1)`(bilinear, 양 끝점 정렬)로 보간, [CLS] 위치는 유지. **정사각 격자만 지원**(assert) | `F.interpolate(bilinear, align_corners=True)`, 직사각 허용. 테스트가 `scipy.zoom`과 10⁻⁵ 안에서 일치 | 일치(논문에는 보간 종류가 없음, 코드에서 bilinear 확인) |
| D8 | 미세조정 하이퍼파라미터 | SGD momentum 0.9, 배치 512, grad clip 1, cosine, 해상도 384, ImageNet 20k 스텝, CIFAR 10k 스텝(표 4, 부록 B.1.1) | `common.py`: 배치 512, `grad_norm_clip=1.0`, cosine, `warmup_steps=500`, crop 384, `base_lr=0.03`(논문은 {0.001…0.06} 격자), `train.py`의 momentum 0.9, 스텝 수는 `DATASET_PRESETS`(imagenet 20k, cifar 10k) | | 대부분 일치. 논문의 학습률 격자 탐색은 코드 기본값 하나 |

추가 관찰: 공개 ViT 설정(`models.py`)의 드롭아웃은 `attention_dropout_rate=0`, `dropout_rate=0`이고 테스트 설정만 0.1이다. 논문 표 3의 사전학습 설정(ImageNet 300 에폭의 dropout 0.1, weight decay 0.3 등)은 이 저장소에 없다.

## 4. 인용 시 주의 (요약)

1. "ViT는 데이터가 적으면 못한다"를 인용할 때는 DeiT, AugReg, Better plain ViT를 함께 적는다(C1, C2).
2. "ViT는 연산이 2–4배 적다"는 JFT 사전학습, 7–14 에폭, 해당 ResNet 레시피의 결과다(C3).
3. "위치 임베딩은 중요하지 않다", "[CLS]와 GAP는 같다"는 각각 5-shot 7 에폭 조건에서만 맞고 ImageNet-1k 처음부터 학습에서는 차이가 보고되었다(C4, C5).
4. 큰 ViT 특징·attention 지도를 해석할 때는 고노름 토큰을 점검한다(C7).
