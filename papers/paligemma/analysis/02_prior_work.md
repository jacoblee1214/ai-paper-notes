# 02. 계보와 관련 자료

분류: 7축 멀티모달 표현·결합(연결기, 결합 시점, 설계 비교). 확인일 2026-11-26. 웹 검색은 쓰지 않았고 논문 본문, 공식 README, arXiv API, `gh api`로 확인했습니다. 학회가 아닌 항목은 "arXiv"로 적었고 기억에 기대는 학회 표기는 "(기억, 미확인)"을 붙였습니다.

## 1. 계보

| 줄기 | 논문 | PaliGemma가 가져온 것 / 바꾼 것 |
|---|---|---|
| PaLI 계열 | PaLI(arXiv:2209.06794, ICLR 2023, 기억·미확인), PaLI-X(2305.18565), PaLI-3(2310.09199), PaLM-E(논문 §1) | 단계 구성(단일 모달 → 다중 모달 → 해상도 증가 → 전이)과 "무한" 스케줄을 가져옴(§3.2). 인코더-디코더 mT5/UL2 대신 디코더 Gemma, SplitCap 대신 평범한 캡션(§3.2.5). PaLI-3가 5B로 55B를 따라잡았다는 흐름을 3B 미만으로 이어 감(§1) |
| 비전 인코더 | SigLIP(arXiv:2303.15343, ICCV 2023), "shape-optimized" So400m(Alabdulmohsin 외, NeurIPS 2023, 논문 참고문헌 [5]) | SigLIP-So400m/14를 그대로 쓰되 동결하지 않음. CapPa·LocCa 계열 근거(§3.2.2) → `siglip/`(진행 순서 43) |
| 언어 모델 | Gemma-2B v1.0(논문 [82]) | 어휘 256,000에 위치 1024, 분할 128 토큰 추가(§5.3) |
| 마스크 | prefix-LM: T5(JMLR 2020, 기억)의 prefix LM, UniLM(NeurIPS 2019, 기억)의 seq2seq 마스크 | 인코더 없이 디코더 하나에서 입력 부분만 양방향. 논문은 이 두 문헌을 인용하지 않았고 "이미지 토큰이 접두를 미리 본다"는 점을 동기로 씀(§3.1, §5.2). 이 계보 연결은 우리가 붙인 것입니다 |
| 연결기 비교 | LLaVA(MLP) → `llava/`(46), BLIP-2 Q-Former → `blip2/`(45), Idefics2 resampler(논문 부록 A) | 선형이 MLP와 같거나 약간 낫다는 절제(§5.5). Prismatic도 같은 질문을 다룸 → `prismatic_vlms/`(35) |
| 해상도 | 윈도우·타일(InternLM-XComposer-4KHD 등), 동적 해상도 → `qwen2_vl/`(37), FlexiViT, NaViT | 해상도별 체크포인트 3개 + Stage2(§5.7). 윈도우는 열세 |
| 인코더 없는 VLM | Fuyu, EVE, Chameleon → `transfusion/`(38) | 인코더 제거 절제(§5.6, 그림 8) |
| 설계 비교 연구 | Prismatic(2402.07865), Cambrian-1(2406.16860), MM1, Idefics2 | 같은 질문(무엇이 중요한가)에 PaliGemma는 전이 관점으로 답함 → `prismatic_vlms/`, `cambrian1/`, `eyes_wide_shut/`(MMVP 벤치마크의 출처, 40) |
| 후속 | PaliGemma 2(arXiv:2412.03555, 공식 README), SigLIP 2(공식 코드 체크포인트 목록에 있음) | PaliGemma 2는 Gemma 2 {2B, 9B, 27B}에 같은 224/448/896 체크포인트(README). 이 저장소에서 분석하지 않음 |

## 2. 이 저장소의 다른 분석과의 관계

- **`eval_error_bars/`(2411.00640), `little_power/`(2010.06595):** 이 논문은 표준편차를 "재실행(훈련 무작위성)"으로만 보고하고 시험 세트 표본 오차는 다루지 않습니다. 두 분석의 짝지은 SE, MDE(≈2.80·√(불일치율/n)), 군집 SE를 `03` §3에서 썼습니다. 문항별 결과가 없어서 짝지은 검정은 불일치율 시나리오로만 했습니다.
- **`prismatic_vlms/`(35), `qwen2_vl/`(37), `transfusion/`(38), `cambrian1/`(39), `eyes_wide_shut/`(40), `vit/`(41), `clip/`(42), `siglip/`(43), `dinov2/`(44), `blip2/`(45), `llava/`(46):** 같은 묶음. 링크만 걸었고 내용은 확인하지 않았습니다.

## 3. 자료 표 (확인 2026-11-26)

| 자료 | 주소 | 라이선스 | 비고 |
|---|---|---|---|
| 공식 코드 | github.com/google-research/big_vision | Apache-2.0 | 별 3,580, 마지막 push 2025-05-19, 커밋 `0127fb6`, 작업 트리 14 MB. PaliGemma 모델·학습기·전이 설정 35개, 사전학습 설정 없음 |
| 가중치 | Kaggle, Hugging Face (README 링크) | Gemma 이용 약관, 수동 동의 필요 | 받지 않음. float32 11GB(README), bf16·f16 변형 있음 |
| openpi (π0 구현) | github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 약 14.8k, `pi0.py`·`tokenizer.py` 두 파일만 `gh api`로 읽음 |
| 논문 | arxiv.org/abs/2407.07726 | CC BY 4.0 표기, 그러나 PDF에 © Google DeepMind | 그림 게시 불가 |
