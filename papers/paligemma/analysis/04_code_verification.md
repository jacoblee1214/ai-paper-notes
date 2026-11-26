# 04. 논문 ↔ 공식 코드 대조

대상: `google-research/big_vision`(Apache-2.0), 커밋 `0127fb6`(2025-05-19, 확인일 2026-11-26), `repo/`에 `--depth 1` clone(작업 트리 14 MB, gitignore). **실행, import, 설치, 가중치 다운로드를 하지 않았습니다.** 읽은 파일: `models/proj/paligemma/paligemma.py`, `gemma_bv.py`, `models/ppp/gemma.py`, `models/vit.py`, `trainers/proj/paligemma/train.py`·`run.py`, `pp/proj/paligemma/ops.py`, `configs/proj/paligemma/README.md`와 `transfers/*.py` 35개. π0 쪽은 openpi의 `pi0.py`, `tokenizer.py` 두 파일을 `gh api`로 읽었습니다.

## 1. 일치하는 것

| 항목 | 논문 | 코드 | 판정 |
|---|---|---|---|
| 마스크 식 | 이미지·접두 양방향, 접미 인과, PAD도 인과 영역(§3.1, 그림 2) | `make_attn_mask`: `cumsum[j] <= cumsum[i]` AND 유효 마스크(`paligemma.py` 27–51). 이미지 `mask_ar`=0을 앞에 채움(123–127), 접두·SEP 0, 접미 1, 패딩 1(`transfers/common.py` 36–41) | **일치** (`mask_toy.py` T1: 서로 다른 4개 길이 구성에서 논문 규칙으로 만든 마스크와 정확히 같음, 학습·평가 모두) |
| 배열 | `[img…, BOS, 접두…, SEP, 접미…, EOS, PAD…]`, SEP = `\n`을 따로 토큰화 | `tok(prefix, bos=yes)`, `tok(suffix, eos=yes)`, `tok(septok, text='\n')`, `masked_concat([prefix, septok, suffix])` | 일치 |
| 손실 | 접미에만 다음 토큰 손실(§5.2) | `mask_loss=[0,0,1]`, 입력 `text[:-1]`, 목표 `text[1:]`, `mask_loss[1:]`(`train.py` 341–355). SEP 위치의 출력이 첫 접미 토큰을 예측, EOS까지 감독, 패딩과 접두는 감독 안 함(`mask_toy.py` T5) | 일치 |
| 연결기 | 영초기화 선형층(§3.1) | ViT `pool_type='none'`, `num_classes=llm.embdim`, `head_zeroinit=True`(`vit.py` 199, 271) | 일치 |
| 구조 | SigLIP So400m/14 + Gemma-2B | `variant='So400m/14'`, `gemma_2b`(폭 2048, 18층, MQA, head_dim 256), 어휘 256000+1024+128 | 일치(`03` §1의 파라미터 산술이 이 값 사용) |
| 이미지 전처리 | 정사각형 리사이즈(§4, 부록 C) | `resize(res)|value_range(-1,1)` | 일치 |
| KV 캐시 | (논문 무기재) | `prefill_cache`는 접두를 블록 마스크로 한 번에 처리하고 `extend_cache`는 사용 중인 모든 칸을 본다 | 설계가 성립함을 토이로 확인(`mask_toy.py` T4: prefill + extend와 전체 순전파의 접미 은닉 상태 차이 2×10⁻¹⁵) |

## 2. 차이와 빈칸 (D1–Dn)

**D1. 사전학습 설정과 혼합이 공개되지 않음.** `configs/proj/paligemma/`에는 README, 노트북, `transfers/`만 있습니다. Stage1·2 설정, 과제 혼합 비율, 일부 비공개 데이터셋(논문 §3.2.6도 인정)이 없어서 §3.2의 사전학습과 §5의 절제(그림 4–10, 17)는 코드로 재현할 수 없습니다.

**D2. "이미지에 인과 마스크" 절제는 설정으로 표현되지 않음.** `embed_image_and_text`가 이미지 토큰의 `mask_ar`를 0으로 고정합니다(`paligemma.py` 126). 그림 5의 이미지 AR 절제는 코드 수정이 필요하고 그 수정본은 공개되지 않았습니다. 접두 AR 절제는 `mask_ar` 배열만 바꾸면 되므로 표현 가능합니다.

**D3. 패딩 처리가 학습과 평가에서 다름.** 학습은 `input_mask`를 주지 않아 전부 True라서 패딩이 입력이고 인과 영역에 있습니다(논문 서술과 같음). 평가·생성은 `mask_input`으로 패딩을 가립니다(`common.py` 58–64). 논문은 이 구분을 쓰지 않습니다. π0는 패딩을 가리는 쪽(평가 방식)을 학습에도 씁니다.

**D4. 손실 정규화(논문 무기재).** 예제마다 `-sum(logp·mask)/max(sum(mask),1)` 후 배치 평균입니다(`train.py` 349–353). 토큰 평균이 아니라서 긴 출력(탐지·분할·OCR)의 토큰당 가중이 1/길이로 줄어듭니다. 짧은 3토큰과 긴 30토큰 예제로 손으로 계산하면 2.0과 1.0일 때 예제 평균 1.5, 토큰 평균 1.09입니다(`mask_toy.py` T5).

**D5. 부록 J와 전이 설정: 432칸 전부 일치.** `hparam_check.py`가 31개 과제의 `sweep_best()`를 ast로 읽어(실행 없이) 에폭, 배치, 학습률, 가중치 감쇠, 드롭아웃, 라벨 스무딩, ViT 동결을 논문 표 62행과 비교했습니다. 불일치 0, 해석 불가 2칸(COCO-35L 에폭: 코드는 `total_examples = 5 × 566,435`이고 566,435 = 5캡션 × 113,287장이라 25 에폭과 같음). 부록 J의 "코드에 있다"는 말은 사실입니다.

**D6. 896px 시도와 비보고(논문 무기재).** 설정 주석에 "896 not better than 448"(Screen2Words, AI2D, RSVQA-hr, A-OKVQA-DA, OKVQA, GQA, ChartQA, SciCap, WidgetCap)이 있습니다. 표 1의 896 열이 비어 있는 이유로 보이지만 논문은 "896을 해 보니 이득이 없었다"를 쓰지 않았습니다. 896 이득이 일반적이라는 인상을 줄 수 있습니다. 또 ChartQA 주석은 "TODO: Update once latest numbers are in and have only 1 setup", GQA 주석은 "freeze_vit=False for all resolutions"를 시험할 필요를 적어 일부 설정은 충분히 탐색되지 않았음을 보여 줍니다.

**D7. 평가 분할과 하이퍼파라미터 선택(§3.5, 논문 §4와 비교).** 설정은 `minitrain`(train[:5%]), `minival`(train[-5%:]), `eval`(val), `test`를 정의하고 sweep은 `final_split=False`(train[:-5%])입니다. 주석은 minival을 "To tune hparams", 검증을 "final publishable scores"라고 부릅니다. 논문은 "검증 분할로, 없으면 minival로" 선택했다고 씁니다. 두 서술이 같은 절차인지는 코드만으로 알 수 없습니다.

**D8. 위치와 스케일(논문 무기재).** RoPE 위치는 이미지 토큰을 포함해 0부터 연속입니다(학습: `arange(seq_len)`, 디코딩: `cumsum(input_mask)-1`). 텍스트 임베딩은 √폭(약 45.25)을 곱하고 이미지 임베딩(영초기화 선형층 출력)은 곱하지 않습니다(`gemma.py` 242, `paligemma.py` 123의 연결). 논문은 "RoPE 보간 불필요"(§7)와 영초기화만 씁니다.

**D9. 위치 임베딩 확장(논문 무기재).** ViT는 높은 해상도로 올릴 때 위치 임베딩 격자를 쌍선형(`scipy.ndimage.zoom(order=1)`)으로 보간해 불러옵니다(`vit.py` 306–322). Stage2에서 이를 썼는지, 새로 학습했는지는 논문에 없습니다.

**D10. MMVP, Objaverse 평가 코드는 없음.** 저장소에 MMVP·Objaverse 평가기와 데이터셋 설정이 없고 README에 점수만 있습니다(MMVP 47.33/45.33은 71/150, 68/150과 일치). POPE 평가기와 CountBenchQA 데이터셋 정의는 있습니다.

**D11. README 결과와 표 1의 8칸 불일치.** `03` §6: COCO-35L avg34, XM3600 avg35, OCR-VQA.

**D12. 가중치 라이선스.** 코드는 Apache-2.0이지만 가중치는 Gemma 약관에 수동 동의가 필요합니다(README). "공개 모델"의 재사용 조건은 코드 라이선스와 다릅니다.

## 3. 구현 가능성 (이 맥: Apple M4 Pro, 약 24 GiB, MPS)

- **추론:** 3B 모델 float32는 11.7 GB, bf16이면 약 5.8 GB라 메모리는 맞습니다. 다만 공식 체크포인트와 코드는 JAX이고 Gemma 약관 동의와 다운로드(수 GB)가 필요합니다. 승인이 필요합니다.
- **마스크·캐시 구조 재구현:** 이미 numpy 토이로 확인했습니다(`mask_toy.py`).
- **전이(미세조정):** 전체 파라미터 학습은 float32 상태(가중치 + Adam 2배)로 약 35 GB 이상이라 24 GiB 맥에서 불가(산술 추정). 공식 노트북은 T4 한 장의 "가벼운 미세조정"을 말하지만 CUDA를 전제로 합니다.
- **사전학습, Stage2, 절제 재현:** TPUv5e-256에서 3일이 든 학습이라 불가이고 설정도 없습니다(D1).
