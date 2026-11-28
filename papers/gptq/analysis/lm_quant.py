"""언어 모델 전체에 층별 순차 GPTQ / RTN을 적용하고 문서별 NLL을 재는 도구(GPT-2 small, Pythia).

변형: 논문 5절 Setup과 같은 '트랜스포머 블록 단위 순차' 양자화다.
  블록 i에 대해 (1) 이미 양자화된 앞 블록들의 출력을 입력으로 받아 (2) 블록 안 선형층 4개의 Hessian을 한 번의 통과로 모으고
  (FP 가중치 상태의 블록) (3) 4개를 양자화한 뒤 (4) 양자화된 블록으로 다시 통과시켜 다음 블록의 입력을 만든다.
  블록 안에서 층끼리의 순차(true-sequential)는 하지 않는다. 임베딩, 위치 임베딩, LayerNorm, lm_head는 FP32로 둔다(논문과 같음).
양자화는 모의(fake) 양자화다: 가중치를 격자 값으로 바꿔 FP32로 저장한다(비트 패킹이나 커널은 없음).
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from minimal_impl import gptq, hessian, rtn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
PILE_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"


def load_docs():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("NeelNanda/pile-10k", PILE_FILE, repo_type="dataset", local_files_only=True)
    return pd.read_parquet(p)["text"].tolist()


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    model.config.use_cache = False
    return model, tok


def make_sets(tok, docs, T, n_calib, n_eval, calib_seed, split=5000, doc_seed=0):
    """보정용: 앞쪽 문서(섞은 순서의 처음 split개)에서 길이 T 구간 n_calib개(문서당 하나, 무작위 시작).
    평가용: 뒤쪽 문서에서 길이 ≥ T인 문서의 첫 T토큰 n_eval개. 두 집합은 문서가 겹치지 않는다."""
    perm = np.random.RandomState(doc_seed).permutation(len(docs))
    rng = np.random.RandomState(calib_seed)
    calib, order = [], list(rng.permutation(perm[:split]))
    for d in order:
        ids = tok(docs[d], return_tensors="pt", truncation=False).input_ids[0]
        if len(ids) >= T:
            s = rng.randint(0, len(ids) - T + 1)
            calib.append(ids[s:s + T])
        if len(calib) == n_calib:
            break
    ev, ev_docs = [], []
    for d in perm[split:]:
        ids = tok(docs[d], return_tensors="pt", truncation=True, max_length=T).input_ids[0]
        if len(ids) == T:
            ev.append(ids)
            ev_docs.append(int(d))
        if len(ev) == n_eval:
            break
    return torch.stack(calib), torch.stack(ev), ev_docs


def blocks_of(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    return model.gpt_neox.layers


def linears_of(block):
    out = []
    for name, m in block.named_modules():
        if isinstance(m, nn.Linear) or type(m).__name__ == "Conv1D":
            out.append((name, m))
    return out


def get_W(m):
    W = m.weight.data
    return (W.t() if type(m).__name__ == "Conv1D" else W).float().clone()


def set_W(m, Q):
    m.weight.data.copy_(Q.t() if type(m).__name__ == "Conv1D" else Q)


@torch.no_grad()
def capture_first_block_inputs(model, ids, bs):
    """블록 0의 입력(임베딩 출력)과 블록 호출에 쓰이는 kwargs를 잡는다."""
    blocks = blocks_of(model)
    store = {"inps": [], "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args, **kwargs):
            h = args[0] if args else kwargs.pop("hidden_states")
            store["inps"].append(h.detach().clone())
            if store["kwargs"] is None:
                store["kwargs"] = kwargs
            raise StopIteration

    orig = blocks[0]
    blocks[0] = Catcher(orig)
    try:
        for i in range(0, len(ids), bs):
            try:
                model(input_ids=ids[i:i + bs], use_cache=False)
            except StopIteration:
                pass
    finally:
        blocks[0] = orig
    return store["inps"], store["kwargs"]


def run_block(block, inps, kwargs):
    outs = []
    for h in inps:
        o = block(h, **kwargs)
        outs.append((o[0] if isinstance(o, tuple) else o).detach())
    return outs


@torch.no_grad()
def quantize_model(model, calib_ids, method, bits, group=None, damp=0.01, act_order=False, bs=8, block=128, log=None):
    """method: 'rtn' | 'gptq'."""
    blocks = blocks_of(model)
    if method == "rtn":
        for blk in blocks:
            for _, m in linears_of(blk):
                set_W(m, rtn(get_W(m), bits, group))
        return {}
    inps, kwargs = capture_first_block_inputs(model, calib_ids, bs)
    stats = {}
    for bi, blk in enumerate(blocks):
        lins = linears_of(blk)
        Hs = {n: [torch.zeros(_in_features(m), _in_features(m)), 0] for n, m in lins}
        hooks = []

        def mk(n):
            def f(_m, inp, _out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                Hs[n][0] += x.T @ x
                Hs[n][1] += x.shape[0]
            return f

        for n, m in lins:
            hooks.append(m.register_forward_hook(mk(n)))
        run_block(blk, inps, kwargs)
        for h in hooks:
            h.remove()
        for n, m in lins:
            H = 2.0 / Hs[n][1] * Hs[n][0]
            W = get_W(m)
            Q, pred = gptq(W, H, bits, block=block, damp=damp, group=group, act_order=act_order)
            set_W(m, Q)
            stats[f"{bi}.{n}"] = {"pred_loss": pred}
        if log:
            log(f"block {bi} done")
        inps = run_block(blk, inps, kwargs)
    return stats


def _in_features(m):
    return m.weight.shape[0] if type(m).__name__ == "Conv1D" else m.weight.shape[1]


@torch.no_grad()
def doc_nll(model, ids, bs=4):
    """문서(길이 T 고정)마다 다음 토큰 NLL 평균(nats)."""
    out = []
    for i in range(0, len(ids), bs):
        x = ids[i:i + bs]
        logits = model(input_ids=x, use_cache=False).logits[:, :-1].float()
        nll = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                                          reduction="none").reshape(x.shape[0], -1)
        out.append(nll.mean(1))
    return torch.cat(out).numpy()
