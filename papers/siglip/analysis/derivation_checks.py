"""03_analysis.md §1의 식 검산 수치 (학습 없음, 수 초). 결과는 derivation_results.json.
  python analysis/derivation_checks.py
"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as M  # noqa: E402

sig = lambda x: 1 / (1 + math.exp(-x))
out = {}

# 1. 초기 손실과 편향 기울기 (모든 로짓 ~ b, 양성 포함, c=0): B와 b에 따라
rows = []
for B in (256, 4096, 32768, 1_048_576):
    for name, b in (("b=0", 0.0), ("b=-10", -10.0), ("b=-ln(B-1)", M.bias_balance_point(B))):
        L = M.loss_const_cos(B, 10.0, b, 0.0, 0.0)
        gb = M.bias_grad_const_cos(B, 10.0, b, 0.0, 0.0)
        pos_w = sig(-b)                    # 양성 쌍 하나의 dL/dl 크기
        neg_w = sig(b)                     # 음성 쌍 하나
        rows.append({"B": B, "b_init": name, "b": round(b, 3), "init_loss_per_row": L, "dL_db_per_row": gb,
                     "pos_pair_weight": pos_w, "neg_pair_weight": neg_w, "neg_total_over_pos": (B - 1) * neg_w / pos_w})
out["init"] = rows

# 2. 최적에서 양성·음성 로짓 분리 (직교 임베딩)
opt = []
for B in (256, 1024, 32768, 1_048_576):
    for t in (10.0, 20.0, 100.0):
        o = M.optimal_orthogonal(B, t)
        opt.append({"B": B, "t": t, **o, "gap_pos_minus_neg": o["logit_pos"] - o["logit_neg"],
                    "softmax_loss_same_embeddings": M.softmax_loss_const_cos(B, t)})
out["optimal_logits"] = opt

# 3. 메모리와 통신 산술 (float32 한 원소 4바이트)
mem = []
for B, D in ((32768, 8), (32768, 256), (1_048_576, 32), (1_048_576, 256), (1_048_576, 1024)):
    m = M.memory_elements(B, D)
    mem.append({"B": B, "D": D, **m, "naive_GiB": m["naive"] * 4 / 2**30, "clip_local_GiB": m["clip_local"] * 4 / 2**30,
                "sigmoid_chunk_GiB": m["sigmoid_chunk"] * 4 / 2**30})
out["memory"] = mem

# 4. 양성:음성 비와 예제 대 쌍 (고정 본 예제 수 N일 때 쌍 수 = N * B)
N = 9e9
out["pairs"] = [{"B": B, "pos_per_step": B, "neg_per_step": B * (B - 1), "neg_to_pos": B - 1,
                 "pairs_seen_for_9B_examples": N * B} for B in (512, 4096, 16384, 32768, 307_000)]

# 5. sigmoid 행 단위 기울기 합 대 softmax (초기, 모든 로짓 같다고 가정): 한 행에서 음성이 받는 총 기울기 가중
out["row_negative_weight_total"] = {
    "softmax_any_B": "1 - p_ii (<= 1), B와 무관하게 1 이하",
    "sigmoid_B=32768_b=0": (32768 - 1) * sig(0.0), "sigmoid_B=32768_b=-10": (32768 - 1) * sig(-10.0)}

# 6. 표 4 읽기: t0=log1 (t=1)에서 b=0이면 로짓 범위 [-1,1]이라 모든 쌍 확률 ~0.5
out["table4_init_loss_B8192"] = {
    "b=0,t=10": M.loss_const_cos(8192, 10.0, 0.0, 0.0, 0.0), "b=0,t=1": M.loss_const_cos(8192, 1.0, 0.0, 0.0, 0.0),
    "b=-10,t=10": M.loss_const_cos(8192, 10.0, -10.0, 0.0, 0.0), "b=-10,t=1": M.loss_const_cos(8192, 1.0, -10.0, 0.0, 0.0)}

# 7. v1 의사 코드(-dot * t + b) 와 v4 의 차이: 같은 입력에서 정렬 쌍의 손실 방향
torch.manual_seed(0)
n, d = 8, 32
x = F.normalize(torch.randn(n, d, dtype=torch.float64), dim=-1)
y_aligned = x.clone()
y_random = F.normalize(torch.randn(n, d, dtype=torch.float64), dim=-1)
v1 = lambda a, c, t, b: -F.logsigmoid((2 * torch.eye(n, dtype=torch.float64) - 1) * (-(a @ c.T) * t + b)).sum() / n
out["v1_pseudocode_sign"] = {"v4_loss_aligned": float(M.sigmoid_loss(x, y_aligned, 10.0, -10.0)), "v4_loss_random": float(M.sigmoid_loss(x, y_random, 10.0, -10.0)),
                             "v1_loss_aligned": float(v1(x, y_aligned, 10.0, -10.0)), "v1_loss_random": float(v1(x, y_random, 10.0, -10.0))}
Path(__file__).with_name("derivation_results.json").write_text(json.dumps(out, indent=1))
for r in out["init"]:
    if r["B"] in (256, 32768): print(r)
print(out["table4_init_loss_B8192"]); print(out["v1_pseudocode_sign"])
for o in out["optimal_logits"]:
    if o["B"] in (1024, 32768) and o["t"] in (10.0, 20.0): print({k: round(v, 3) for k, v in o.items()})
for m in out["memory"]: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
