# PagedAttention / vLLM (arXiv:2309.06180) 분석 노트

Kwon et al.(SOSP 2023)의 PagedAttention을 구현 모드로 분석한 노트입니다. 블록 표 기반 KV 캐시 관리자(논리 -> 물리 블록, 자유 목록, 참조 횟수, 블록 단위 copy-on-write, 스와프, 재계산·스와프 선점 서버)와 쪽 단위 어텐션 커널 세 가지를 PyTorch로 짜서(pytest 19개) 연속 예약(Orca Max, Pow2, Oracle, buddy 포함)과 쪽 단위의 메모리 구성을 시뮬레이션하고, 병렬 샘플링과 빔 서치의 공유 절약을 재고, GPT-2 실제 K/V에서 쪽 단위 어텐션이 밀집 어텐션과 수치로 같음을 확인했습니다. 논문 그림은 넣지 않았고(ACM 고지가 있는 PDF) 그래프는 모두 직접 그린 것입니다. 원문 PDF와 공식 코드(`vllm-project/vllm`, Apache-2.0)는 넣지 않았습니다.

## 결과 요약

- 그림 2의 Orca 20.4-38.2% 대 vLLM 96.3%는 길이 평균만 맞춘 시뮬레이션에서 재현됩니다(우리 Max 20.3%, Oracle 37.9%, vLLM 96.5% [96.4, 96.6]). Max의 외부 단편화 8.9%는 7 x 2048 뒤의 자투리(1,392 / 15,728 = 8.85%)입니다. Oracle의 36%는 buddy 올림에서 옵니다.
- 쪽 안 낭비는 요청당 (B-1)/2 슬롯으로 이론과 일치합니다. Alpaca 평균 길이에서 B=128은 동시 처리 수가 B=16의 55%입니다.
- 병렬 샘플링 공유 절약은 논문(그림 15)의 약 절반이라 재현하지 못했고, 빔 서치는 후보의 부모 분포 가정에 지배됩니다.
- OPT-13B 800 KB/토큰은 800 KiB(819,200 B)이고 표 1의 GB는 GiB로 읽어야 슬롯 수가 맞습니다.
- 쪽 단위 어텐션은 GPT-2 실제 K/V에서 밀집 어텐션과 최대 절대 오차 5.2e-6 이내입니다.
- 현재 공식 코드(V1)에서 논문의 스와핑 선점과 시퀀스 그룹 CoW는 코어에서 찾지 못했습니다.
- 처리량 2-4배와 커널 지연은 GPU가 필요해 재지 않았습니다. 길이 분포는 평균만 맞춘 로그정규(분산 가정)입니다.

| 경로 | 내용 |
|---|---|
| `analysis/01_paper_analysis.md` ... `04_claim_verification.md` | 논문 분석, 계보, 검산과 토이 실험, 주장 검증 C1-C8와 공식 코드 정적 대조 D1-D8 |
| `analysis/minimal_impl.py`, `test_minimal_impl.py` | `BlockAllocator`, `BlockManager`, `PagedKVCache`, `PagedServer`와 pytest 19개 |
| `analysis/fragmentation_sim.py` | buddy 포함 연속 예약 대 쪽 단위 시뮬레이션 |
| `analysis/sharing_sim.py` | 병렬 샘플링·빔 서치 공유 절약 |
| `analysis/gpt2_paged_check.py` | GPT-2 실제 K/V에서 쪽 단위 대 밀집 어텐션 |
| `analysis/paper_tables_recalc.py` | 표 1, 그림 1, 2, 11, 13 산술 재계산과 선점 비용 토이 |
| `analysis/plot_figures.py`, `figures/` | 직접 그린 그래프 3장 |

## 재실행

모델과 데이터는 로컬 캐시(`openai-community/gpt2`, 데이터셋 `NeelNanda/pile-10k`)를 씁니다. 시뮬레이션은 모델 없이 돕니다.

```bash
pip install torch transformers numpy scipy pandas matplotlib pyarrow huggingface_hub pytest
export HF_HUB_OFFLINE=1
cd analysis
python3 -m pytest test_minimal_impl.py -q
python3 paper_tables_recalc.py
python3 fragmentation_sim.py      # 프로세스 2개, 수 분, 메모리 1 GB 미만
python3 sharing_sim.py
python3 gpt2_paged_check.py
python3 plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv 2309.06180v1, arXiv 라이선스 CC BY 4.0. 원문 PDF에 ACM SOSP '23 허가·저작권 고지가 있어 논문 그림은 넣지 않았습니다.
- 공식 코드는 Apache-2.0이며 읽기만 했고 복사하거나 실행하지 않았습니다. 이 폴더의 구현은 논문 설명으로 직접 짰습니다.
