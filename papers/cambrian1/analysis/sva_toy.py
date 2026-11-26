"""SVA(Spatial Vision Aggregator) 산술 검산과 작은 토이 (직접 짠 코드, 공식 코드는 실행하지 않음).

1. 식 (1)의 '질의 (i,j)가 각 인코더 특징 맵의 m_k x m_k 부분 영역만 본다'가 재배열(view/permute)로 구현한 국소 어텐션과
   같은지, 전체 키에 마스크를 씌운 어텐션과 수치적으로 비교한다 (CPU, 작은 크기).
2. 공식 코드(`cambrian/model/vision_sampler.py`, `cambrian_arch.py`)를 정적으로 읽고 만든 모듈 구조를 같은 모양으로 짜서,
   파라미터 수 공식이 실제 nn.Module 개수와 일치하는지 작은 크기에서 확인한 뒤 실제 설정(스크립트의 8B/13B/34B)에 적용한다.
3. 토큰 수와 키 수(비용) 비교.
결과: analysis/sva_params_results.json
실행: torch가 있는 Python 환경에서 python analysis/sva_toy.py   (torch만 필요, 수 초)
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)

# ---------------------------------------------------------------- 1. 국소 어텐션 = 마스크 어텐션
L, C, H = 4, 32, 4
ms = [1, 1, 1, 4]  # 인코더별 배율 m_k (스크립트: 토큰 길이 [576,576,576,9216], L=24 -> m=[1,1,1,4])
B = 2
feats = [torch.randn(B, (m * L) ** 2, C) for m in ms]
Wq = nn.Linear(C, C, bias=False)
Wk = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in ms])
Wv = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in ms])
x = torch.randn(1, C)
Q = Wq(x).expand(B, L * L, C)  # 모든 격자 위치가 같은 잠재 x에서 시작(위치 정보는 부분 영역으로만 들어옴)


def heads(t):
    b, n, c = t.shape
    return t.view(b, n, H, c // H).transpose(1, 2)


# (a) 재배열 방식 (공식 코드 rearrange_vision_tower_features_*와 같은 view/permute)
keys_loc, vals_loc = [], []
for f, m, wk, wv in zip(feats, ms, Wk, Wv):
    g = f.view(B, L, m, L, m, C).permute(0, 1, 3, 2, 4, 5).contiguous()  # B, L, L, m, m, C
    g = g.view(B * L * L, m * m, C)
    keys_loc.append(wk(g)); vals_loc.append(wv(g))
k_loc = torch.cat(keys_loc, 1); v_loc = torch.cat(vals_loc, 1)   # (B*L*L, sum m^2, C)
q_loc = Q.reshape(B * L * L, 1, C)
out_loc = F.scaled_dot_product_attention(heads(q_loc), heads(k_loc), heads(v_loc)).transpose(1, 2).reshape(B, L * L, C)

# (b) 전체 키 + 마스크 (질의 (i,j)는 자기 부분 영역의 키만 허용)
K_all = torch.cat([wk(f) for f, wk in zip(feats, Wk)], 1)   # B, sum (mL)^2, C
V_all = torch.cat([wv(f) for f, wv in zip(feats, Wv)], 1)
mask = torch.zeros(L * L, K_all.shape[1], dtype=torch.bool)
off = 0
for m in ms:
    for i in range(L):
        for j in range(L):
            for a in range(m):
                for b_ in range(m):
                    row, col = m * i + a, m * j + b_
                    mask[i * L + j, off + row * (m * L) + col] = True
    off += (m * L) ** 2
out_mask = F.scaled_dot_product_attention(heads(Q), heads(K_all), heads(V_all), attn_mask=mask.view(1, 1, L * L, -1)).transpose(1, 2).reshape(B, L * L, C)
max_abs_diff = float((out_loc - out_mask).abs().max().detach())
assert max_abs_diff < 1e-5, max_abs_diff

# ---------------------------------------------------------------- 2. 파라미터 수 공식 대 nn.Module
def layer_params_formula(q, C, N, kv_sizes):
    """VisionCrossAttentionLayer (joint) 한 층. q: 질의 차원, C: vision_hidden_size, N: 인코더 수."""
    ctx = C * C                          # proj_context (bias 없음)
    pin = (q + C) * C                    # proj_in
    pout = C * C + C * q                 # proj_out = MLP(C, C, q) (bias 없음)
    norm = 2 * C
    attn = (2 * C + C * C) + 2 * N * (2 * C + C * C) + C * C   # q_proj(LN+Linear), k/v_proj x N, o_proj
    pos = sum(s * s * C for s in kv_sizes if s > 1)
    return ctx + pin + pout + norm + attn + pos


class MLPb(nn.Module):  # bias 없는 MLP (vision_sampler.MLP)
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_h, bias=False); self.l2 = nn.Linear(d_h, d_out, bias=False)


class MultiKV(nn.Module):
    def __init__(self, C, N):
        super().__init__()
        self.q = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C, bias=False))
        self.k = nn.ModuleList([nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C, bias=False)) for _ in range(N)])
        self.v = nn.ModuleList([nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C, bias=False)) for _ in range(N)])
        self.o = nn.Linear(C, C, bias=False)


class JointLayer(nn.Module):
    def __init__(self, q, C, N, kv_sizes):
        super().__init__()
        self.proj_context = nn.Linear(C, C, bias=False)
        self.proj_in = nn.Linear(q + C, C, bias=False)
        self.proj_out = MLPb(C, C, q)
        self.norm = nn.LayerNorm(C)
        self.attn = MultiKV(C, N)
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(s * s, C)) for s in kv_sizes if s > 1])


def count(m):
    return sum(p.numel() for p in m.parameters())


checks = []
for (q, C, N, sizes) in [(48, 32, 4, [1, 1, 1, 4]), (64, 32, 2, [1, 2]), (32, 32, 3, [1, 1, 1])]:
    checks.append({"q": q, "C": C, "N": N, "sizes": sizes, "module": count(JointLayer(q, C, N, sizes)),
                   "formula": layer_params_formula(q, C, N, sizes)})
assert all(c["module"] == c["formula"] for c in checks)


def aux_proj(h, C):  # Linear(h,C), GELU, Linear(C,C), LayerNorm(C)  (bias 있음)
    return (h * C + C) + (C * C + C) + 2 * C


def final_proj(C, G, Hd):  # Linear(C*G, H), GELU, Linear(H,H) (bias 있음)
    return (C * G * Hd + Hd) + (Hd * Hd + Hd)


def sva_total(Hd, n_in_llm, C=1024, N=4, hs=(1152, 1024, 1536, 5760), depth=3, G=1, sizes=(1, 1, 1, 4)):
    connector = depth * layer_params_formula(C, C, N, sizes)
    in_llm = n_in_llm * layer_params_formula(Hd, C, N, sizes)
    aux = sum(aux_proj(h, C) for h in hs)
    fin = final_proj(C, G, Hd)
    misc = G * C + Hd  # vision_query, image_newline
    tot = connector + in_llm + aux + fin + misc
    return {"connector_D3": connector, "in_LLM_layers": in_llm, "per_encoder_aux_mlp": aux, "final_mlp": fin, "misc": misc, "total": tot}


cfg = {  # scripts/cambrian/finetune_cambrian_{8,13,34}b.sh
    "8B (Llama-3-8B, H=4096, 32 layers)": dict(Hd=4096, n_in_llm=10, stride=3, llm_layers=32, llm_params=8.03e9),
    "13B (Vicuna-1.5-13B, H=5120, 40 layers)": dict(Hd=5120, n_in_llm=10, stride=4, llm_layers=40, llm_params=13.0e9),
    "34B (Hermes-2-Yi-34B, H=7168, 60 layers)": dict(Hd=7168, n_in_llm=9, stride=7, llm_layers=60, llm_params=34.4e9),
}
res_cfg = {}
for k, v in cfg.items():
    t = sva_total(v["Hd"], v["n_in_llm"])
    idx = [i * v["stride"] for i in range(v["n_in_llm"])]
    t["in_llm_layer_indices"] = idx
    t["indices_within_llm"] = max(idx) < v["llm_layers"]
    t["fraction_of_llm_params_pct"] = round(100 * t["total"] / v["llm_params"], 2)
    res_cfg[k] = t
# 논문 식 (1)만 읽었을 때의 최소 파라미터(W_Q, W_K^k, W_V^k 만, C=1024, N=4): (1+2N) C^2
minimal_eq1_per_layer = (1 + 2 * 4) * 1024 ** 2
code_layer_connector = layer_params_formula(1024, 1024, 4, [1, 1, 1, 4])

# ---------------------------------------------------------------- 3. 토큰·키 수
Lq = 24
keys_per_query_spatial = sum(m * m for m in [1, 1, 1, 4])
keys_dense = sum((m * Lq) ** 2 for m in [1, 1, 1, 4])
tokens = {
    "query_grid": Lq * Lq, "llm_positions_with_newline": Lq * (Lq + 1),
    "keys_per_query_spatial": keys_per_query_spatial, "keys_per_query_dense": keys_dense,
    "dense_over_spatial": keys_dense / keys_per_query_spatial,
    "qk_pairs_spatial": Lq * Lq * keys_per_query_spatial, "qk_pairs_dense": Lq * Lq * keys_dense,
    "llava_next_tokens": 2880, "reduction_vs_llava_next": round(2880 / (Lq * Lq), 2),
}

out = {"local_vs_masked_attention_max_abs_diff": max_abs_diff, "param_formula_vs_module_small_checks": checks,
       "per_layer_params": {"connector_layer_C1024": code_layer_connector,
                            "in_LLM_layer_H4096_C1024": layer_params_formula(4096, 1024, 4, [1, 1, 1, 4]),
                            "eq1_minimal_(1+2N)C^2": minimal_eq1_per_layer,
                            "connector_layer_over_eq1_minimal": round(code_layer_connector / minimal_eq1_per_layer, 2)},
       "totals": res_cfg, "tokens": tokens,
       "assumptions": "vision_hidden_size C=1024 (스크립트), 인코더 hidden 1152/1024/1536/5760은 코드와 HF 설정에서 읽은 값 (DINOv2는 스크립트가 giant: 1536), 헤드 16, D=3, G=1. 공식 코드는 실행하지 않았고 정적 대조만 함."}
(ROOT / "analysis" / "sva_params_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("max abs diff local vs masked attention:", max_abs_diff)
print(json.dumps(out["per_layer_params"], indent=1))
for k, v in res_cfg.items():
    print(k, {kk: vv for kk, vv in v.items() if kk in ("connector_D3", "in_LLM_layers", "total", "fraction_of_llm_params_pct", "indices_within_llm")})
print(tokens)
