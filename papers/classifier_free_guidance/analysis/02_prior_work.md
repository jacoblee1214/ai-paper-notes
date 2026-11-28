# 02. 계보와 관련 분석, 자료 표

## 1. 계보

```
Sohl-Dickstein 2015 ─ Song & Ermon 2019 (NCSN) ─ DDPM 2020 ─ Score SDE 2020/21 ─ DDIM 2020/21
                                                      │
 BigGAN 2019 (truncation) ─┐                          ▼
 Glow 2018 (낮은 온도)  ───┤        Dhariwal & Nichol 2021 (분류기 안내, ADM-G)
                           ▼                          │
                     Classifier-Free Guidance 2021 워크숍 / 2022 arXiv  ← 이 분석
                (분류기를 암시적 분류기 p(z|c)/p(z)로 대체, 한 모델에서 조건·무조건 겸용)
                                  │
   ┌──────────────┬───────────────┼────────────────┬──────────────────┬────────────────┐
   ▼              ▼               ▼                ▼                  ▼                ▼
 GLIDE 2021    Imagen·DALL-E 2   Latent Diffusion  분석: "기울인 분포   개선: 재조정,      효율: 안내 증류,
 (병행 CFG)    (텍스트 안내,     / Stable Diffusion  를 표집하지 않는다"  구간 안내,          1-NFE 안내
               guidance scale)   (guidance_scale)   (Chidambaram, B&N,   스케줄러, CADS,      (MeanFlow)
                                                    Wu 2024)            자기 안내(2023–24)
                                  │
                                  ▼
                 flow matching / rectified flow(SD3) / MeanFlow: 속도장에 같은 외삽
```

(오른쪽 갈래의 논문 이름, 연도, 한 줄 주장은 arXiv API 초록으로 확인했고 본문은 열람하지 않았습니다. `04` 참고.)

## 2. 구성 요소별 출처 (논문의 인용)

| 구성 요소 | 출처 |
|---|---|
| 연속 시간 확산, 로그 SNR λ 매개변수화 | Song 외 2021b, Kingma 외 2021(VDM). §2는 "continuous time"으로 학습한다고 씀 |
| 분산 보간 (σ̃²)^{1−v}(σ²)^v, 상수 v | Nichol & Dhariwal 2021. "학습한 v 대신 상수 v가 효과적"은 이 논문의 선택(§2) |
| 코사인형 p(λ) = hyperbolic secant 분포(구간 제한) | Nichol & Dhariwal의 이산 코사인 스케줄에서 영감(§2) |
| 분류기 안내 | Dhariwal & Nichol 2021 (arXiv:2105.05233), §3.1 |
| 낮은 온도·truncation의 동기 | Brock 외 2019(BigGAN), Kingma & Dhariwal 2018(Glow) |
| 비보존 점수장 | Salimans & Ho 2021, §5 |
| 암시적 분류기가 좋지 않을 수 있다는 단서 | Grandvalet & Bengio 2004, Grünwald & Langford 2007, §3.2 |
| 하이퍼파라미터·구조 | 분류기 안내 모델(ADM)과 같음, §4 |
| 기준선 | ADM, ADM-G, CDM(Ho 외 2021), BigGAN-deep, LOGAN, §4 |
| 새로움(논문의 주장) | **무조건 모델에서 음의 점수 항으로 안내**("to our knowledge not yet explored", §5), 분류기 없는 안내 |
| 병행 연구 | GLIDE(arXiv:2112.10741)도 같은 시기에 같은 기법을 텍스트-이미지에 사용(공개일 2021-12-20) |

## 3. 이후 논문이 바꾼 것 (자세한 판정은 `04`)

| 방향 | 논문 | 바꾼 것 (초록 수준) |
|---|---|---|
| 해석 | Chidambaram 외 2024(2409.13074) | 안내가 의도한 기울임을 표집하지 못함. 큰 w는 지지집합 경계, 점수 오차가 있으면 밖으로 |
| 해석 | Bradley & Nakkiran 2024(2408.09000) | 어떤 표집기도 γ 거듭제곱 분포를 만들지 못함. CFG = 예측자-교정자 |
| 해석 | Wu 외 2024(2403.01639) | 가우시안 혼합에서 확신도 증가와 엔트로피 감소를 증명 |
| 개선 | Kynkäänniemi 외 2024(2404.07724) | 안내를 중간 잡음 구간에만 적용. ImageNet-512 FID 1.81 → 1.40 |
| 개선 | Karras 외 2024(2406.02507) | 작고 덜 학습된 자기 자신으로 안내. 변동은 유지, FID 1.01(64×64), 1.25(512×512) |
| 개선 | Sadat 외 2023(2310.17347) | 조건 신호에 잡음을 섞어 큰 w의 다양성을 보존(CADS) |
| 개선 | Wang 외 2024(2404.13040) | 단조 증가 안내 스케줄러 |
| 개선 | Lin 외 2023(2305.08891) | 과노출 방지용 CFG 재조정(다른 결함 교정과 함께) |
| 개선 | Chung 외 2024(2406.08070, CFG++) | 다양체 제약 CFG(제목만 확인, 내용 미확인) |
| 효율 | Meng 외 2023(2210.03142) | 안내 모델 증류(제목만 확인, 내용 미확인) |

## 4. 이 저장소의 다른 분석과의 관계

- `ddpm/`: ε-예측, 정확한 denoiser(혼합 닫힌 식). 이 분석의 `minimal_impl.py`는 같은 기법(혼합의 닫힌 점수)을 클래스 조건부로 확장한 것입니다. 알고리즘 2의 조상 표집과 식 (3), (4)의 분산 보간 v는 DDPM의 σ² 논의(`ddpm/` `03` §3)와 이어집니다.
- `score_sde/`: 확률 흐름 ODE와 VP SDE. 이 분석의 DDIM = 확률 흐름 ODE의 오일러 이산화(`03` §1.5)는 거기서 다룬 표집기 비교의 안내 버전입니다.
- `ddim/`: 결정적 표집. 이 분석 `03` §1.5의 "DDIM은 확률 흐름 ODE의 오일러 단계" 확인(λ 매개화)은 그쪽에서 ᾱ 매개화로 확인한 스텝별 일치(1e-13)와 같은 내용입니다.
- `sd3_rectified_flow/`: 텍스트 조건에 안내를 쓰는 대표 응용. 코드는 `neg + (pos − neg) · cond_scale`(γ 규약)이고 이 논문의 안내가 속도장에서도 같다는 것은 `03` §1.3, `04` C10입니다.
- `meanflow/`: 1-NFE에서 CFG를 목표 장에 넣는 식(13–21)과 𝔼_c = 사후 가중 지적. **안내의 흐름 매칭판 분석은 거기서 하고 여기서는 되풀이하지 않았습니다**(`meanflow/analysis/03_analysis.md` §5). 안내를 일부 구간에만 두는 토이 결과(클래스 평균 거리 1.30 → 0.81)가 이 분석의 E4와 같은 방향입니다.

## 5. 자료 표

| 자료 | 내용 | 라이선스 | 비고 |
|---|---|---|---|
| (저자 공식 코드) | 없음 | – | `hojonathanho`, `TimSalimans` 저장소 목록에 없음(`gh api`, 2026-11-28) |
| arXiv 2207.12598 | 유일한 판 v1, 2022-07-26 | arXiv 비독점 배포(CC 아님) | PDF 3.8 MB |
| OpenReview `qw8AKxfYbI` | NeurIPS 2021 워크숍 단편 | | 직접 조회는 챌린지 인증으로 막힘. 논문 각주와 arXiv 코멘트, neurips.cc 가상 행사 목록, mlanthology로 확인(웹 검색 1회) |
| `openai/glide-text2im` | CFG 포함 병행 연구 구현 | MIT | 별 3,683, 마지막 push 2024-03-08 |
| `openai/guided-diffusion` | 분류기 안내 구현 | MIT | 별 7,448, 마지막 push 2024-07-02 |
| `jcwang-gh/classifier-free-diffusion-guidance-Pytorch` | 제3자 구현 | MIT | 별 212, 마지막 push 2023-03-13 |
| `huggingface/diffusers` | `guidance_scale` 파이프라인 | Apache-2.0 | 별 34,986, 마지막 push 2026-11-28 |
| 이후 문헌 | 위 3절의 논문들 | 라이선스 미확인 | 그림 재사용 판단에는 쓰지 않음 |
