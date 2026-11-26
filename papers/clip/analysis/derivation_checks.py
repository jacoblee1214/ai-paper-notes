"""식 검산 수치 (학습 없음, 수 초). 결과는 derivation_results.json.

1. 무작위 임베딩의 손실 = log N + s^2/(2d) 근사
2. 정렬된 직교 임베딩의 닫힌 식 log(1 + (N-1) e^{-s}), N = 32768 포함
3. InfoNCE 상호정보 하한 log N - L (이변량 정규, 최적 비판자)
4. 기울기 dL/dt의 식 유도와 autograd의 일치
5. 배치 안에 속성을 맞바꾼 쌍(twin)이 들어올 확률 (토이 데이터)
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as m  # noqa: E402

out = {}
g = torch.Generator().manual_seed(0)

# 1. 무작위 임베딩. 로짓 s*cos의 분산 s^2/d 이므로 E[logsumexp] ~ log N + s^2/(2d) (로짓이 정규일 때 로그정규 모멘트)
rows = []
for N in (16, 256, 4096):
    for d in (64, 512):
        for s in (1.0, 1 / 0.07):
            I = F.normalize(torch.randn(N, d, generator=g, dtype=torch.float64), dim=-1)
            T = F.normalize(torch.randn(N, d, generator=g, dtype=torch.float64), dim=-1)
            L = m.clip_loss(I, T, s).item()
            rows.append(dict(N=N, d=d, scale=round(s, 3), loss=L, log_N=math.log(N), approx=math.log(N) + s * s / (2 * d)))
out["random_embedding_loss"] = rows

# 2. 정렬된 직교 임베딩
rows = []
for N in (8, 32768):
    for s in (1.0, 1 / 0.07, 30.0, 100.0):
        rows.append(dict(N=N, scale=round(s, 3), loss=math.log1p((N - 1) * math.exp(-s))))
out["aligned_orthogonal_closed_form"] = rows
# 정답이 아닌 쌍이 코사인 c를 가질 때: log(1 + (N-1) exp(-s(1-c))). 배율 상한 100에서 N = 32768
out["aligned_with_negative_cos_c_N32768_s100"] = {str(c): math.log1p(32767 * math.exp(-100 * (1 - c))) for c in (0.0, 0.5, 0.8, 0.9, 0.95)}

# 3. 상호정보 하한
rows = []
for N in (16, 128):
    for rho in (0.5, 0.9, 0.99, 0.999, 0.9999999):
        B = 2000 if N == 16 else 400
        x = torch.randn(B, N, generator=g, dtype=torch.float64)
        y = rho * x + math.sqrt(1 - rho**2) * torch.randn(B, N, generator=g, dtype=torch.float64)
        sc = -0.5 * math.log(1 - rho**2) - (y[:, None, :] - rho * x[:, :, None]) ** 2 / (2 * (1 - rho**2)) + y[:, None, :] ** 2 / 2
        L = F.cross_entropy(sc.reshape(B * N, N), torch.arange(N).repeat(B)).item()
        rows.append(dict(N=N, rho=rho, I_true=-0.5 * math.log(1 - rho**2), log_N=math.log(N), bound=math.log(N) - L))
out["mi_bound_gaussian"] = rows

# 4. 기울기
I = F.normalize(torch.randn(12, 5, generator=g, dtype=torch.float64), dim=-1)
T = F.normalize(torch.randn(12, 5, generator=g, dtype=torch.float64), dim=-1)
t = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
m.clip_loss(I, T, t.exp()).backward()
s, C = math.exp(1.5), I @ T.T
def d(Cm):
    P = torch.softmax(s * Cm, dim=1)
    return s * ((P * Cm).sum(1) - torch.diagonal(Cm)).mean()
out["grad_wrt_log_scale"] = dict(autograd=t.grad.item(), formula=((d(C) + d(C.T)) / 2).item())

# 5. 배치 안에 "맞바꾼 쌍(twin)"이 들어올 확률. 두 물체 장면 i (c1 != c2, s1 != s2)의 twin은 물체가
#    ((c1, s2), (c2, s1))인 다른 장면이다. twin이 배치에 있어야만 표준 손실이 결합(어느 색이 어느 모양인가)을 직접 구별하게 한다.
def twin_rate(batch: int, trials: int = 300) -> dict:
    gen = torch.Generator().manual_seed(123)
    hit = tot = 0
    for _ in range(trials):
        two = torch.rand(batch, generator=gen) < 0.5
        c1, s1 = m.sample_seen_combos(batch, gen)
        c2, s2 = m.sample_seen_combos(batch, gen)
        objs = [((int(a), int(b)), (int(c), int(d))) for a, b, c, d, t in zip(c1, s1, c2, s2, two) if t]
        have = set(objs)
        for (a, b), (c, d) in objs:
            if a != c and b != d:
                tot += 1
                hit += ((a, d), (c, b)) in have
    return {"batch": batch, "two_object_eligible_samples": tot, "twin_in_batch_rate": hit / tot}


out["swap_twin_in_batch"] = [twin_rate(32), twin_rate(128)]

Path(__file__).with_name("derivation_results.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k)
    for r in (v if isinstance(v, list) else [v]):
        print("  ", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()} if isinstance(r, dict) else r)
