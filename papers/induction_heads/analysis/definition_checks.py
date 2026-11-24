"""정의·기준선 검산 (학습 없음, 수 초): 03_analysis.md §1이 쓰는 숫자를 definition_checks_results.json로 저장한다.

  - prefix-matching 점수의 이론값: 이상적 head 1, 균등 causal attention의 우연 수준, 이전 토큰 head 0
  - copying 고유값 점수의 우연 수준: 무작위 OV 행렬의 sum(Re λ)/sum|λ| 분포
  - 합성 과제의 손실 기준선: 첫 구간 ln(V-1), '문맥에 나온 토큰 집합 균등' 예측기의 반복 구간 손실(1층 모델이 할 수 있는 수준)
  - 모델 크기(파라미터 수)
실행: python analysis/definition_checks.py  (torch가 설치된 python)
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_impl import Cfg, TinyLM, induction_target_mask, ov_copy_eig_score, prefix_matching_score, repeat_batch

HERE = Path(__file__).resolve().parent


def uniform_causal(T):
    p = torch.tril(torch.ones(T, T))
    return p / p.sum(-1, keepdim=True)


def main():
    out = {}
    # 1) prefix-matching 점수
    for L, n in ((25, 4), (12, 4), (8, 4)):
        T = 1 + L * n
        M = induction_target_mask(L, n)
        ideal = M / M.sum(-1, keepdim=True).clamp_min(1)
        ideal[: 1 + L] = uniform_causal(T)[: 1 + L]
        out[f"prefix_L{L}_n{n}"] = {"ideal": prefix_matching_score(ideal, L, n).item(),
                                    "uniform_causal": prefix_matching_score(uniform_causal(T), L, n).item(),
                                    "n_dest": L * (n - 1), "targets_per_dest": (M.sum(-1)[1 + L:]).tolist()[:: max(1, L)]}
    # 2) 무작위 OV의 copying 고유값 점수 (d x d 가우시안, E = U = I)
    g = torch.Generator().manual_seed(0)
    for d in (64, 128):
        vals = []
        for _ in range(200):
            O = torch.randn(d, d, generator=g) / math.sqrt(d)
            vals.append(ov_copy_eig_score(torch.eye(d), O, torch.eye(d)))
        t = torch.tensor(vals)
        out[f"copy_eig_random_d{d}"] = {"mean": t.mean().item(), "std": t.std().item(), "min": t.min().item(), "max": t.max().item()}
    # 3) 합성 과제의 손실 기준선 (SYN 설정: V=64, T=64, L in 6..20)
    V, T, Lmin, Lmax = 64, 64, 6, 20
    gen = torch.Generator().manual_seed(0)
    toks, L = repeat_batch(gen, 20000, T, V, Lmin, Lmax)
    first = math.log(V - 1)
    # 반복 구간(토큰 위치 >= 1+L)에서 '지금까지 본 서로 다른 토큰 수'의 자연로그 평균 = 문맥 토큰 집합 균등 예측기 손실(상한 근사)
    losses = []
    for b in range(2000):
        l = int(L[b])
        for tpos in range(1 + l, T):
            seen = len(set(toks[b, 1:tpos].tolist()))
            losses.append(math.log(seen))
    out["synthetic_baselines"] = {"first_period_ln(V-1)": first, "seen_set_uniform_repeat_loss": sum(losses) / len(losses),
                                  "oracle_induction_repeat_loss": 0.0, "uniform_vocab_ln_V": math.log(V)}
    # 4) 모델 크기
    sizes = {}
    for name, cfg in {"syn_1L": Cfg(vocab=64, d=128, n_layer=1, n_head=4, ctx=64), "syn_2L": Cfg(vocab=64, d=128, n_layer=2, n_head=4, ctx=64),
                      "nat_2L_mlp": Cfg(vocab=2048, d=128, n_layer=2, n_head=4, d_mlp=512, ctx=256)}.items():
        sizes[name] = sum(p.numel() for p in TinyLM(cfg).parameters())
    out["n_params"] = sizes
    (HERE / "definition_checks_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
