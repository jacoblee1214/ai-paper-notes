"""토이: 같은 행렬곱을 배치 크기만 바꿔 돌리면 비트가 달라지는가, 그리고 그 차이가 greedy argmax를 뒤집는가.

논문은 원인을 "입력 버퍼 공유(input buffer packing)"로 추측만 했다(§Abstract, §8.1). 이 토이는 호스팅 API를 재현하지 않고,
'배치 크기에 따라 커널이 다른 누적 순서를 고르면 같은 입력의 출력 비트가 달라진다'는 메커니즘 하나만 CPU/MPS에서 확인한다.
  1) 행 0의 로짓을 배치 1 / 배치 B(나머지 행은 다른 사용자의 입력)로 계산해 비트 일치 비율과 최대 차이를 잰다.
  2) 실제 LM의 상위 2개 로짓 간격(gap) 분포는 모른다. 간격이 Exponential(평균 g)이라고 가정하고 flip 확률 ≈ P(gap < δ)를 적는다.
실행: <torch가 있는 python> -I analysis/batch_invariance_toy.py
출력: analysis/batch_invariance_toy_results.json
"""
import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent / "batch_invariance_toy_results.json"
torch.manual_seed(0)
K, V = 2048, 4096  # 은닉 차원, 어휘 크기(축소)
devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
res = []
for dev in devices:
    for dn, dt in dtypes.items():
        W = (torch.randn(K, V) / K ** 0.5).to(dev, dt)
        h0 = torch.randn(1, K).to(dev, dt)
        solo = (h0 @ W).float().cpu()
        for B in (2, 7, 33, 128):
            eq_bits, maxd, relmed = [], [], []
            for trial in range(8):
                others = torch.randn(B - 1, K).to(dev, dt)
                batch = torch.cat([h0, others], 0)
                out = (batch @ W)[:1].float().cpu()
                eq_bits.append((out == solo).float().mean().item())
                d = (out - solo).abs()
                maxd.append(d.max().item())
                relmed.append((d / solo.abs().clamp_min(1e-6)).median().item())
            res.append(dict(device=dev, dtype=dn, batch=B, frac_bitwise_equal=sum(eq_bits) / len(eq_bits),
                            max_abs_diff=max(maxd), median_rel_diff=sum(relmed) / len(relmed)))
            print(res[-1], flush=True)
# 간격 가정에 따른 flip 확률 (로짓 표준편차 ~1 규모에서 δ = 최대 절대 차이)
flip = []
for r in res:
    if r["batch"] == 128:
        for g in (0.1, 1.0):
            flip.append(dict(device=r["device"], dtype=r["dtype"], gap_mean=g, delta=r["max_abs_diff"],
                             p_flip_per_token=1 - torch.exp(torch.tensor(-r["max_abs_diff"] / g)).item()))
OUT.write_text(json.dumps(dict(runs=res, flip_estimates=flip, torch=torch.__version__), indent=1))
print(*flip, sep="\n")
