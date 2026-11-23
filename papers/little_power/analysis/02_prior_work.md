# 02. 계보와 관련 연구: With Little Power Comes Great Responsibility

## 1. 계보: 이 논문이 기대는 것

| 구성 요소 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| 검정력 개념, 80% 관행 | Cohen (1962), *The statistical power of abnormal-social psychological research*, J. Abnormal and Social Psychology 65(3) | §1 "검정력 부족 = 80% 미만" |
| 소표본의 부작용 (효과 과대, 낮은 양성 예측값) | Button et al. (2013) *Power failure*, Nature Reviews Neuroscience; Ioannidis et al. (2017) | §1: 다른 분야의 재현 위기와 연결 |
| Type-M, Type-S 오류 | Gelman & Carlin (2014), Perspectives on Psychological Science 9(6) | 그림 1, 부록 B: 유의한 결과가 효과를 부풀리고 부호가 뒤집힘 |
| NLP의 유의성 검정 지침 | Dror et al. (2018) *The Hitchhiker's Guide to Testing Statistical Significance in NLP* (ACL, aclanthology.org/P18-1128); Berg-Kirkpatrick et al. (2012) *An Empirical Investigation of Statistical Significance in NLP* (EMNLP-CoNLL, D12-1091); Koehn (2004) 번역 유의성 검정 (W04-3250); Dietterich (1998) | McNemar(분류), 무작위화 검정(BLEU)을 검정으로 고름 (§3.1, §4). 이 논문의 위치: 이 문헌들은 "어떤 검정을 쓰나"를 다뤘고, 이 논문은 "그 검정이 효과를 잡을 만큼 데이터가 있나"를 더함 |
| McNemar 검정력·표본 크기 | Duffy (1984), Connett et al. (1987), Lachenbruch (1992) *On the sample size for studies based upon McNemar's test* (Statistics in Medicine 11), Fagerland et al. (2013) | 불일치 쌍이 검정력을 결정한다는 식(부록 E). 저자 R 코드의 `power_mcnemar_test`가 Connett·Duffy 정규식이고 표 9의 "Lachenbruch min/mid/max"가 이것 (`03` §2) |
| 혼합 효과 모형(평가자·문항 무작위 효과) | Barr et al. (2013) "keep it maximal", Westfall et al. (2014), Gelman & Hill (2006), van der Lee et al. (2019) 사람 평가 모범 사례 | §5.2, 부록 H.3: 식 5–10 |
| 보고 관행 | Dodge et al. (2019) *Show Your Work* (EMNLP, arXiv:1909.03004), Schwartz et al. (2019) Green AI | 각주 3: 이 논문의 주장은 "훈련된 모델 인스턴스"에 대한 것이고, 계산 예산에 대한 주장은 따로 |

**이 논문이 새로 한 것.** (1) 짝지은 정확도 비교의 검정력을 NLP 데이터에 맞춰 **불일치율 Pa를 회귀로 예측**한 것(표 5·6), (2) BLEU처럼 **분해되지 않는 코퍼스 지표**에 대해 "교환 효과 δ_i"의 Delta–Laplace 혼합으로 검정력을 계산하는 생성 과정(§4, 부록 G), (3) EMNLP 2019 사람 평가 41편의 메타분석(§5.1), (4) GLUE·SQuAD의 현행 MDE 표와 공개 노트북. 개별 통계 식(McNemar 검정력, 혼합 모형)은 의학·심리학의 기존 식입니다. 우리가 `03`에서 확인한 대로, 식 자체는 고전이고 기여는 **NLP 모수 추정과 현황 측정**입니다.

## 2. 이후 문헌이 바꾼 것

각 항목은 요약 위치만 적었고, 본문은 `04`의 C로 검증합니다.

| 문헌 | 한 일 | 이 논문과의 관계 |
|---|---|---|
| Bowman & Dahl (2021) *What Will it Take to Fix Benchmarking in NLU?* (NAACL, arXiv:2104.02145) | 벤치마크가 갖춰야 할 기준 네 가지에 **크기(작은 개선을 구별할 수 있는 시험 세트)**를 포함 | 이 논문의 "MRPC·SST-2는 너무 작다"를 기준 수준으로 올림 (abstract 기준 확인, `04` C1) |
| Madaan et al. (2024) *Quantifying Variance in Evaluation Benchmarks* (arXiv:2406.10229) | 평가 벤치마크의 **시드 분산**, 학습 중 단조성 등을 측정 | 이 논문은 시험 세트 표본 분산만 다룸. 학습 시드 분산을 더하는 쪽 (`04` C5) |
| Miller (2024) *Adding Error Bars to Evals* (arXiv:2411.00640, 이 저장소 21번 예정) | 문항을 모집단의 표본으로 보고, 두 모델 비교를 **문항 쌍체 분석**으로, 평가 실험 계획(검정력) 공식 제시 | 이 논문의 짝지은 검정력을 LLM 벤치마크 언어로 옮긴 것 (abstract 기준) |
| Bowyer et al. (2025) *Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints* (arXiv:2503.01747, ICML 2025 Spotlight Position) | 문항이 수백 개 미만이면 CLT 기반 오차 막대가 크게 작게 나온다고 보이고 베이즈·빈도 대안 제시 | 이 논문의 정규 근사 해석식도 작은 n에서 어긋남 (`03` §2의 WNLI) (`04` C3) |
| Zhuang et al. (2026) *Pre-Registering the Detectable Effect: A Paired-MDE Budget for 4-bit Quantization Benchmarks* (arXiv:2605.28873, 소형 노트, 심사 정보 없음) | Miettinen (1968)의 짝지은 이진 표본 크기 식으로 δ* ≤ (z₁₋α/₂ + z₁₋β)√(ρ_d/m)를 사용 | 우리가 `03` §1에서 유도한 MDE와 같은 식 (`04` C2) |

## 3. 이 저장소의 관련 분석

이 논문의 방법이 앞 분석들의 어느 줄과 이어지는지입니다.

| 분석 | 이어지는 곳 |
|---|---|
| `agentic_eval_randomness/` (On Randomness in Agentic Evals) | 같은 질문(에이전트 평가의 검정력)을 **반복 실행 수 K**로 푼 것. 그 분석의 쌍체 MDE는 K=1에서 4.1–6.1 %p(500문항)였고, 이 논문의 해석식으로 500문항·불일치율 12–20%를 넣으면 4.3–5.6 %p로 같은 범위입니다(`03` §8). 이 논문의 불일치율 d가 그쪽에서는 "같은 설정을 두 번 돌렸을 때 달라지는 문항 비율"이기도 합니다. 반복 실행(K↑)은 d의 실행 잡음 몫만 줄이고 문항 모집단 몫은 남는다는 그쪽 결론은, 이 논문(시험 세트 표본만 봄)에 빠진 항을 보여 줍니다 |
| `llm_nondeterminism/` (Non-Determinism of "Deterministic" LLM Settings) | temperature 0에서도 같은 문항의 결과가 실행마다 달라지므로, 한 모델의 "정확도"에는 실행 간 불일치가 이미 들어 있습니다. 두 모델의 짝지은 불일치율 d의 하한이 0이 아니라는 뜻이라 d를 가정할 때 이 값이 바닥이 됩니다 |
| `self_debug/` (15번), `swe_agent/` (16번), `agentdebug/` (18번) | 문항별 결과가 공개되지 않아 **주변 합(정답 수)만으로 McNemar p의 가능 범위**를 냈습니다(불일치쌍을 가장 적게/많게 배치). 이것이 이 논문 표 9의 **Lachenbruch min/max와 같은 계산**입니다. `03` §7에서 swe_agent의 54 대 33(300문항)을 다시 계산해 p ≈ 1e-6과 0.031을 얻었고(그쪽 분석과 일치), 불일치율 범위 [0.07, 0.29]에서 검정력은 0.997에서 0.616까지 움직입니다. 즉 "p_max가 0.05 미만이면 어떤 배치에서도 유의"를 검정력 언어로 바꾸면 "불일치가 최대여도 검정력이 충분한가"입니다. 이 논문의 Pa 회귀(GLUE는 불일치 약 6%)는 이 범위의 가운데를 주는 방법이지만, 에이전트 벤치마크에 쓸 회귀는 아직 없습니다 |
| `swe_bench/` (17번) | n = 500(Verified), 300(Lite), 2294(전체)가 MDE를 정합니다. `03` §8 표 |
| `statistical_precipice/` (19번) | 여러 과제·시드의 **집계 지표**(IQM, 층화 부트스트랩)의 불확실성. 이 논문은 한 비교의 검정력이고, 다과제 집계의 검정력은 각주 8이 미룬 부분입니다. 링크만 |
| `eval_error_bars/` (21번, 예정) | 이 논문의 짝지은 McNemar를 군집 표준오차·쌍체 차이로 일반화. 링크만 |

## 4. 자료 표

| 자료 | 주소 | 라이선스 | 크기·별·마지막 push | 쓴 방법 |
|---|---|---|---|---|
| 논문 | arxiv.org/abs/2010.06595 (v1) | arXiv 비독점 배포 라이선스(arxiv.org/licenses/nonexclusive-distrib/1.0). ACL Anthology 판은 CC BY 4.0 | 1.3 MB PDF | `paper/` |
| 공식 저장소 | github.com/dallascard/NLP-power-analysis | Apache-2.0 | 14.7 MB(`gh api` 크기), 별 41, 마지막 push 2020-10-15. HEAD `a3f63b5e…` | `repo/`에 얕게 clone, **정적으로만** 읽음. 실행 안 함 |
| 저장소 속 공개 데이터 | `data/GLUE`(주석 CSV), `data/SQuAD2`(리더보드 쌍 결과), `data/human_eval`(메타분석, lme4 결과 CSV) | 저장소와 같음 | 작은 CSV들. `data/mt/wmt18and19.en-de.zip`(15 MB)은 받아졌으나 열지 않음 | `recompute.py`가 읽음 |
| 비공개 | GLUE 시험 예측, SQuAD 시험 쌍별 결과 | 저자도 공유 불가라고 명시 | | 표 5의 회귀 계수는 저자 노트북에 하드코딩된 값을 씀 |
