"""층 하나에서 GPTQ의 성질을 재는 실험(합성 + GPT-2 small 실제 활성).
A 합성 상관 활성에서 RTN / 탐욕 OBQ / GPTQ의 층 오차  B dampening  C act-order  D GPT-2 실제 층(보정 대 held-out)
E 그룹 격자를 블록 시작 값으로 잡는 변형(공식 저장소 방식)의 차이  F 순차 OBQ 대 GPTQ 시간
오차 = mean_n ||(W - Ŵ) x_n||² (식 1을 토큰 수로 나눈 것). 실제 층은 ||W x||²로 나눈 상대 오차도 적는다.
실행: python3 layer_checks.py   (결과 layer_checks_results.json)
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

from minimal_impl import gptq, hessian, layer_error, obq_fixed_order, obq_greedy, rtn

HERE = Path(__file__).resolve().parent
RES = {}


def make(d_row, d_col, n, seed, spread=3.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_row, d_col, generator=g, dtype=dtype)
    A = torch.randn(d_col, d_col, generator=g, dtype=dtype) / d_col ** 0.5 + torch.eye(d_col, dtype=dtype)
    scale = torch.logspace(-spread / 2, spread / 2, d_col, dtype=dtype)[torch.randperm(d_col, generator=g)]
    X = (torch.randn(n, d_col, generator=g, dtype=dtype) @ A) * scale
    return W, X


def boot(v, n=2000, seed=0):
    r = np.random.RandomState(seed)
    v = np.asarray(v)
    m = [v[r.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# A. 합성: 상관 있는 활성, 시드 10개
def exp_A():
    out = {}
    for d_col, d_row, bits in [(32, 16, 3), (64, 16, 3), (64, 16, 4)]:
        rt, gp, ob = [], [], []
        for s in range(10):
            W, X = make(d_row, d_col, 8 * d_col, s, spread=1.5)
            H = hessian(X)
            e0 = layer_error(W, rtn(W, bits), X)
            rt.append(e0)
            gp.append(layer_error(W, gptq(W, H, bits, block=16)[0], X) / e0)
            ob.append(layer_error(W, obq_greedy(W, H, bits), X) / e0)
        out[f"d{d_col}_b{bits}"] = {"rtn_abs": boot(rt), "gptq_over_rtn": boot(gp), "greedy_over_rtn": boot(ob),
                                    "gptq_over_greedy": boot(np.array(gp) / np.array(ob)),
                                    "gptq_wins_over_rtn": int(sum(g < 1 for g in gp))}
    RES["A_synthetic"] = out


# B. dampening: n < d_col이라 H가 특이. 학습 활성 대 새 활성(held-out)에서 오차
def exp_B():
    out = {}
    for dtype in (torch.float64, torch.float32):
        rows = {}
        rtn_te = []
        for damp in [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            tr, te, fail = [], [], 0
            for s in range(10):
                W, Xfull = make(32, 96, 64 + 2000, s, spread=1.5, dtype=dtype)  # 같은 분포에서 학습 64개 + 새 2000개
                X, Xte = Xfull[:64], Xfull[64:]
                try:
                    Q, _ = gptq(W, hessian(X), 3, block=32, damp=damp)
                    if not torch.isfinite(Q).all():
                        raise RuntimeError
                    tr.append(layer_error(W, Q, X))
                    te.append(layer_error(W, Q, Xte))
                except Exception:
                    fail += 1
                if damp == 0.0:
                    rtn_te.append(layer_error(W, rtn(W, 3), Xte))
            rows[str(damp)] = {"fail": fail, "train_err": float(np.mean(tr)) if tr else None,
                               "heldout_err": float(np.mean(te)) if te else None}
        rows["rtn_heldout_err"] = float(np.mean(rtn_te))
        out[str(dtype).split(".")[1]] = rows
    RES["B_dampening_underdetermined"] = out


# C. act-order: 열 크기 편차가 큰 합성 활성
def exp_C():
    out = {}
    for spread in (1.0, 3.0):
        base, act = [], []
        for s in range(10):
            W, X = make(32, 128, 1024, s, spread=spread)
            H = hessian(X)
            e0 = layer_error(W, rtn(W, 3), X)
            base.append(layer_error(W, gptq(W, H, 3, block=32)[0], X) / e0)
            act.append(layer_error(W, gptq(W, H, 3, block=32, act_order=True)[0], X) / e0)
        out[f"spread10^{spread}"] = {"gptq_over_rtn": boot(base), "gptq_act_over_rtn": boot(act),
                                     "act_better_in": int(sum(a < b for a, b in zip(act, base)))}
    RES["C_act_order_synthetic"] = out


@torch.no_grad()
def collect(model, ids, want_blocks, bs=4):
    from lm_quant import blocks_of, capture_first_block_inputs, linears_of, run_block
    inps, kw = capture_first_block_inputs(model, ids, bs)
    out = {}
    for bi, blk in enumerate(blocks_of(model)):
        if bi > max(want_blocks):
            break
        lins = linears_of(blk)
        acc = {n: [torch.zeros(m.weight.shape[0] if type(m).__name__ == "Conv1D" else m.weight.shape[1],
                               m.weight.shape[0] if type(m).__name__ == "Conv1D" else m.weight.shape[1]), 0]
               for n, m in lins}
        hs = []
        for n, m in lins:
            def f(_m, inp, _o, n=n):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                acc[n][0] += x.T @ x
                acc[n][1] += x.shape[0]
            hs.append(m.register_forward_hook(f))
        outs = run_block(blk, inps, kw)
        for h in hs:
            h.remove()
        inps = outs
        if bi in want_blocks:
            for n, (S, c) in acc.items():
                out[(bi, n)] = 2.0 / c * S
    return out


# D, E. GPT-2 실제 층
def exp_DE():
    from lm_quant import get_W, linears_of, blocks_of, load_docs, load_model, make_sets
    docs = load_docs()
    model, tok = load_model("openai-community/gpt2")
    calib, ev, _ = make_sets(tok, docs, 512, 64, 150, 0)
    want = [0, 5, 11]
    Hc = collect(model, calib, want)
    import gc
    gc.collect()
    Hh = collect(model, ev[:64], want)  # held-out 문서의 활성
    gc.collect()
    layers = {(bi, n): m for bi in want for n, m in linears_of(blocks_of(model)[bi])}
    D = {}
    for (bi, n), Hcal in Hc.items():
        W = get_W(layers[(bi, n)])
        Hho = Hh[(bi, n)]
        norm_c = float(0.5 * torch.trace(W @ Hcal @ W.T))
        norm_h = float(0.5 * torch.trace(W @ Hho @ W.T))
        for bits in (4, 3):
            row = {}
            for tag, Q in [("rtn", rtn(W, bits)), ("gptq", gptq(W, Hcal, bits)[0]),
                           ("gptq_act", gptq(W, Hcal, bits, act_order=True)[0])]:
                Dl = W - Q
                row[tag] = {"calib_rel": float(0.5 * torch.trace(Dl @ Hcal @ Dl.T)) / norm_c,
                            "heldout_rel": float(0.5 * torch.trace(Dl @ Hho @ Dl.T)) / norm_h}
            D[f"b{bi}.{n}/b{bits}"] = row
    RES["D_real_gpt2_layers"] = D
    # E. 그룹 g=32, 블록 128: 현재 값 대 블록 시작 값
    E = {}
    for (bi, n) in [(5, "mlp.c_fc"), (5, "attn.c_attn"), (11, "mlp.c_proj")]:
        W = get_W(layers[(bi, n)])
        H = Hc[(bi, n)]
        for g in (32, 64, 128):
            a = gptq(W, H, 3, block=128, group=g)[0]
            b = gptq(W, H, 3, block=128, group=g, stale_group_params=True)[0]
            ea = float(0.5 * torch.trace((W - a) @ H @ (W - a).T)) / float(0.5 * torch.trace(W @ H @ W.T))
            eb = float(0.5 * torch.trace((W - b) @ H @ (W - b).T)) / float(0.5 * torch.trace(W @ H @ W.T))
            E[f"b{bi}.{n}/g{g}"] = {"current": ea, "block_start": eb}
    RES["E_group_param_variant"] = E


# F. 시간
def exp_F():
    out = {}
    for d in (16, 32, 64, 128):
        W, X = make(4, d, 4 * d, 0, spread=1.0)
        H = hessian(X)
        t = {}
        for name, fn in [("gptq", lambda: gptq(W, H, 3, block=32)), ("obq_fixed_naive", lambda: obq_fixed_order(W, H, 3, range(d))),
                         ("obq_greedy", lambda: obq_greedy(W, H, 3))]:
            t0 = time.time()
            fn()
            t[name] = time.time() - t0
        out[f"d_col={d}"] = t
    RES["F_time_d_row4"] = out


if __name__ == "__main__":
    torch.manual_seed(0)
    for f in (exp_A, exp_B, exp_C, exp_DE, exp_F):
        t = time.time()
        f()
        print(f.__name__, f"{time.time() - t:.0f}s", flush=True)
    (HERE / "layer_checks_results.json").write_text(json.dumps(RES, indent=1))
