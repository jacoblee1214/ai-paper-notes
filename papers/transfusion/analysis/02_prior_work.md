# 02. 계보와 관련 자료

분류: 7축 멀티모달 표현·결합(결합 시점: 이산 토큰 early fusion 대 연속 패치 early fusion, 한 모델에서의 이해와 생성). 확인일 2026-11-26. 웹 검색은 쓰지 않았고 논문 본문, arXiv API(제목·날짜), `gh api`로 확인했습니다. 이 분석에서 원문을 읽지 않은 후속 논문의 설명은 제목과 일반적으로 알려진 요지만 적었고 "(원문 미확인)"을 붙였습니다. 학회 표기는 확인한 것만 적고 나머지는 "arXiv"로 둡니다.

## 1. 계보

| 줄기 | 논문 | Transfusion이 가져온 것 / 바꾼 것 |
|---|---|---|
| 확산 | DDPM(arXiv:2006.11239, Ho 외), 개선 DDPM·코사인 일정(Nichol & Dhariwal, 논문 참고문헌), 잠재 확산 LDM(2112.10752, Rombach 외) | ε 예측 손실(식 3)과 코사인 노이즈 일정, VAE 잠재 공간(8×8 픽셀 → 8차원)을 그대로 씀. 바꾼 것: 확산 모델을 따로 두지 않고 언어 모델과 같은 Transformer에서 학습 |
| 확산 + Transformer | DiT(2212.09748), U-ViT(2209.12152) | 확산 모델의 백본으로 Transformer를 쓰는 흐름. Transfusion은 DiT류의 별도 조건 주입(adaLN) 없이 timestep 임베딩을 패치 벡터에 더하고(선형 변형) 텍스트와 같은 시퀀스에서 처리. U-Net 변형은 입출력 쪽에만 U-Net 블록을 둠(각주 7) |
| 이산 토큰 early fusion | DALL-E(2102.12092, Ramesh 외), Parti(논문 참고문헌), CM3Leon(논문 참고문헌 Yu 외 2023), **Chameleon**(2405.09818) | Transfusion의 직접 대조군. 같은 VAE 구조에 양자화 층과 코드북 손실만 바꿔 데이터·계산을 통제(§4.1, 부록 A). Chameleon 레시피의 안정성 수정(QK 정규화, post-norm, denominator loss, 낮은 lr)이 효율 비용이라는 것이 대조의 해석에 들어감(§4.2, 표 4) |
| 연속 입력만 | Fuyu(논문 참고문헌), 선형 패치 입력 VLM | 이미지 패치를 입력으로만 쓰는 모델. Transfusion은 출력(생성)도 연속으로 함 |
| 모델 + 확산 접목 | GILL, DreamLLM(2309.11499), LLaVA-Plus(도구) | 사전학습 모델을 투영층으로 잇는 방식. Transfusion은 한 아키텍처로 끝까지 학습 |
| 이산 이미지 생성 변형 | Show-o(2408.12528, 원문 미확인: 한 Transformer에서 텍스트 자기회귀 + 이미지 마스크 토큰 이산 확산으로 알려짐), Emu3(2409.18869, 원문 미확인: 모든 모달리티를 다음 토큰 예측으로 통일) | 연속 패치 대신 이산 토큰에서 결합. Transfusion(2024-08-20)과 Show-o(2024-08-22)는 이틀 차이로 동시기 |
| 이해·생성 분리 인코더 | Janus(2410.13848, 원문 미확인: 이해용과 생성용 비전 인코더를 분리) | Transfusion은 하나의 표현(U-Net 또는 선형)으로 이해(캡션)와 생성을 함께 함. 두 목적이 한 표현에서 충돌한다는 문제의식이 분리의 동기로 알려져 있음 |
| 자기회귀 + 확산 헤드 | MAR(2406.11838, 원문 미확인: 연속 토큰을 자기회귀로 예측하되 토큰별 확산 손실 헤드) | 연속 이미지 표현을 언어 모델식으로 다루는 다른 방법. Transfusion은 이미지 안 양방향 + 이미지 전체를 함께 denoise, MAR는 토큰 단위 확산 |
| 모달리티별 파라미터 | **Mixture-of-Transformers**(2411.04996, TMLR 2025, 공식 저장소 설명 확인): Lili Yu, Chunting Zhou 등 같은 그룹. 저장소 README는 "Transfusion 설정에서 7B MoT가 밀집 모델과 같은 품질을 FLOPs 3분의 1로 낸다"고 적음(저자 주장, 이 분석에서 검증 안 함) | Transfusion의 가중치 공유(모든 파라미터 공유)를 모달리티별 Q/K/V/O·FFN·LayerNorm 분리로 바꾼 후속. 어텐션은 전역 |
| | LMFusion(2412.15188, 원문 미확인: 사전학습 언어 모델에 이미지용 모달리티별 모듈을 붙이고 텍스트 가중치는 고정) | Transfusion 학습을 사전학습 LLM 위에서 하는 방향 |
| 후속 통합 모델 | BAGEL(2505.14683, 원문 미확인: 이해·생성 전문가를 둔 MoT 계열 통합 사전학습 모델로 알려짐) | 이 저장소에서는 진행 순서에 없음 |
| 확산 → 흐름 | Flow matching(2210.02747, Lipman 외) | 논문이 "미래 과제"로 둔 대체(§3 식 4 다음). 비공식 구현(`lucidrains/transfusion-pytorch`)은 flow matching으로 대체해 구현(`04`) |

## 2. 이 저장소의 다른 분석과의 관계

- **`paligemma/`(36):** PaliGemma의 prefix-LM 마스크(이미지·접두 양방향, 접미 인과)와 Transfusion의 마스크(전체 인과 + 이미지 블록 안 양방향)는 같은 계열의 "블록 안은 양방향, 블록 사이는 인과" 마스크입니다. big_vision의 `make_attn_mask`(`cumsum[j] <= cumsum[i]`)에서 ar 플래그만 바꾸면 Transfusion 마스크가 나온다는 것을 `03` §1에서 확인했습니다. 차이는 (a) PaliGemma는 이미지가 맨 앞에 한 번이고 Transfusion은 시퀀스 중간에 여러 번 나온다는 점, (b) PaliGemma는 이미지가 입력일 뿐 출력(생성)이 아니라는 점, (c) 손실이 접미에만 걸리는 대 모든 텍스트 위치와 이미지 위치에 각각 다른 손실이 걸린다는 점입니다.
- **`prismatic_vlms/`(35), `qwen2_vl/`(37):** 이해 쪽 VLM 설계(연결기, 해상도, 위치 부호). Transfusion은 이미지를 입력으로 넣는 방식이 VAE 잠재 패치(선형 또는 U-Net)로 다르고, 사전학습 비전 인코더(CLIP·SigLIP)를 쓰지 않습니다. 이 때문에 캡션 CIDEr(표 7: 7B 27.2–33.7)는 SigLIP 기반 VLM의 캡션 점수와 직접 비교할 수 없습니다(데이터·평가 설정이 다름). 내용은 확인하지 않고 링크만 겁니다.
- **`cambrian1/`(39), `eyes_wide_shut/`(40), `vit/`(41), `clip/`(42), `siglip/`(43), `dinov2/`(44), `blip2/`(45), `llava/`(46):** 같은 묶음. 링크만 걸었습니다.
- **스케일링 재적합:** 이 저장소에는 거듭제곱 법칙 재적합 분석이 아직 없습니다. 이 분석의 `analysis/scaling_refit.py`(PDF 벡터에서 점 읽기 + 군집 부트스트랩)가 처음입니다. 통계 일반론은 `statistical_precipice/`(2108.13264)의 부트스트랩 접근과 닿아 있으나 방법이 다릅니다.

## 3. 자료 표 (확인 2026-11-26)

| 자료 | 주소 | 라이선스 | 비고 |
|---|---|---|---|
| 논문 | arxiv.org/abs/2408.11039 | CC BY 4.0 | v1(2024-08-20) 하나. 분야 cs.AI, cs.CV. ICLR 2025 구두 |
| 공식 코드 | 없음 | — | 논문에 링크 없음. `facebookresearch/transfusion` 404 |
| Meta MoT 저장소 | github.com/facebookresearch/Mixture-of-Transformers | BSD-3-Clause | 별 316, 마지막 push 2025-09-13, 커밋 `2ba51c0`, 작업 트리 약 1 MB. MoT 튜토리얼과 기본 모듈 3개(`src/`). Transfusion 학습 코드 아님 |
| **비공식** 구현 | github.com/lucidrains/transfusion-pytorch | MIT | 별 1,412, 마지막 push 2026-09-04, 커밋 `a0a0949`, 작업 트리 71 MB(`repo/transfusion-pytorch/`, 정적으로만 읽음). 논문과 다른 점은 `04` |
| 데이터 | Shutterstock 380M(라이선스, 비공개), 공개 이미지 220M(사람 제외), CC12M | — | 논문 §4.1. Shutterstock·혼합 비율은 공개되지 않음 |
| 가중치 | 없음 | — | 논문과 저장소 어디에도 공개 가중치 언급 없음 |
