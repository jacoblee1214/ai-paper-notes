"""GPT-2 small / Pythia-160M 에 W4·W3 (그룹 128) 양자화를 적용해 pile-10k 보류 문서의 perplexity를 잰다.

방법: fp32, RTN, AWQ(scale), AWQ(scale+clip), 채널 보호(FP16 1% 등: 활성/가중치 노름/무작위), s 고정 스윕(Table 2형).
인자: model(gpt2|pythia160m) out.json [--quick]
모든 가중치 양자화는 시뮬레이션(양자화 후 dequantize한 fp32 가중치를 넣음). 입력 쪽 diag(s)^-1은
가중치에 되돌려 합쳐 둔다(수학적으로 같은 함수, 04 D5).

보정 데이터: pile-10k train을 시드 0으로 섞어 앞에서부터 이어 붙인(문서 사이 EOS) 토큰열을 512 토큰 N_CALIB개로 자름.
평가 데이터: 보정에 쓴 문서 다음 문서들 중 토큰 128 이상인 첫 N_EVAL개, 문서마다 앞 512 토큰.
"""
import glob
import json
import os
import resource
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minimal_impl import (act_scale, auto_clip, awq_search, layer_loss, make_scales,
                          mixed_precision_quantize, quantize_group, salient_channels, scaled_quantize)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODELS = {"gpt2": "openai-community/gpt2", "pythia160m": "EleutherAI/pythia-160m"}
N_CALIB, SEQ, N_EVAL, EVAL_LEN, GS = 32, 512, 64, 512, 128
N_SEARCH = 2048            # AWQ 탐색에 쓰는 보정 토큰 수(전체 N_CALIB*SEQ개에서 무작위 부분집합). 평균 |x|는 전체로 계산
SEED = 0


def load_docs():
    p = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots/*/data/*.parquet"))[0]
    texts = pq.read_table(p).column("text").to_pylist()
    order = np.random.RandomState(SEED).permutation(len(texts))
    return texts, order


def build_data(tok):
    texts, order = load_docs()
    eos = tok.eos_token_id
    ids, i = [], 0
    while len(ids) < N_CALIB * SEQ:
        ids += tok(texts[order[i]])["input_ids"] + [eos]
        i += 1
    calib = torch.tensor(ids[: N_CALIB * SEQ]).view(N_CALIB, SEQ)
    n_calib_docs = i
    docs = []
    while len(docs) < N_EVAL:
        t = tok(texts[order[i]])["input_ids"][:EVAL_LEN]
        i += 1
        if len(t) >= 128:
            docs.append(t)
    return calib, docs, dict(n_calib_docs=int(n_calib_docs), n_docs_scanned_for_eval=int(i - n_calib_docs))


def linears(model, name):
    """[(layer_idx, role, module, is_conv1d)] 블록 안의 큰 선형 층 4개씩."""
    out = []
    if name == "gpt2":
        for li, b in enumerate(model.transformer.h):
            out += [(li, "qkv", b.attn.c_attn, True), (li, "o", b.attn.c_proj, True),
                    (li, "fc1", b.mlp.c_fc, True), (li, "fc2", b.mlp.c_proj, True)]
    else:
        for li, b in enumerate(model.gpt_neox.layers):
            out += [(li, "qkv", b.attention.query_key_value, False), (li, "o", b.attention.dense, False),
                    (li, "fc1", b.mlp.dense_h_to_4h, False), (li, "fc2", b.mlp.dense_4h_to_h, False)]
    return out


def get_w(mod, conv):
    return (mod.weight.data.T if conv else mod.weight.data).clone()


def set_w(mod, conv, w):
    mod.weight.data.copy_(w.T if conv else w)


@torch.no_grad()
def collect_stats(model, lins, calib, skip_first=False):
    """FP 모델을 한 번 돌려 각 선형 층 입력의 채널별 평균 |x| (전체 토큰)와 무작위 N_SEARCH행을 모은다."""
    sums = {id(m): torch.zeros(m.weight.shape[0 if conv else 1], dtype=torch.float64) for _, _, m, conv in lins}
    n_tok = {"n": 0}
    rows = {id(m): [] for _, _, m, _ in lins}
    g = torch.Generator().manual_seed(SEED)
    per_batch = N_SEARCH // (N_CALIB // 4)   # 배치 4개 문서씩
    sel = {}
    hooks = []

    def mk(m):
        def hook(_, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            sums[id(m)] += x.abs().sum(0).double()
            if "idx" not in sel:
                perm = torch.randperm(x.shape[0], generator=g)
                if skip_first:                      # 각 문장의 첫 토큰(위치 0) 행을 탐색 행에서 뺀다
                    perm = perm[perm % SEQ != 0]
                sel["idx"] = perm[:per_batch]
            rows[id(m)].append(x[sel["idx"]].clone())
        return hook
    for _, _, m, _ in lins:
        hooks.append(m.register_forward_pre_hook(mk(m)))
    for b in range(0, N_CALIB, 4):
        sel.clear()
        model.base_model(calib[b:b + 4])
        n_tok["n"] += 4 * SEQ
    for h in hooks:
        h.remove()
    stat = {id(m): (sums[id(m)] / n_tok["n"]).float() for _, _, m, _ in lins}
    return stat, {k: torch.cat(v) for k, v in rows.items()}


@torch.no_grad()
def eval_docs(model, docs, bs=4):
    """문서별 (NLL 합, 토큰 수). 오른쪽 패딩은 인과 모델에서 앞 위치에 영향이 없다."""
    nll, cnt = [], []
    for b in range(0, len(docs), bs):
        chunk = docs[b:b + bs]
        L = max(len(d) for d in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        for i, d in enumerate(chunk):
            ids[i, :len(d)] = torch.tensor(d)
            att[i, :len(d)] = 1
        h = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
        head = model.get_output_embeddings()
        tok_nll = torch.zeros(len(chunk), L - 1)
        for i, d in enumerate(chunk):          # 문서마다 로짓을 만들어 메모리를 L x V로 제한
            lg = head(h[i, :len(d) - 1]).float()
            tok_nll[i, :len(d) - 1] = torch.nn.functional.cross_entropy(lg, ids[i, 1:len(d)], reduction="none")
            del lg
        nll += tok_nll.sum(1).tolist()
        cnt += (att[:, 1:].sum(1)).tolist()
    return nll, cnt


def main():
    name, outp = sys.argv[1], sys.argv[2]
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.float32).eval()
    calib, docs, meta = build_data(tok)
    lins = linears(model, name)
    orig = {id(m): get_w(m, conv) for _, _, m, conv in lins}
    t0 = time.time()
    stat, rows = collect_stats(model, lins, calib)
    print(f"stats {time.time()-t0:.0f}s", flush=True)
    res = dict(model=name, n_calib=N_CALIB, seq=SEQ, n_search_tokens=N_SEARCH, n_eval_docs=len(docs),
               eval_len=EVAL_LEN, group_size=GS, seed=SEED, meta=meta,
               n_params_quantized=int(sum(w.numel() for w in orig.values())), methods={}, awq_info={}, table2={})

    def run_method(tag, fn):
        """fn(w, x_mean, x_rows, mod_info) -> 양자화된 w. 모든 선형 층에 적용해 평가 후 원복."""
        t = time.time()
        for li, role, m, conv in lins:
            set_w(m, conv, fn(orig[id(m)], stat[id(m)], rows[id(m)], (li, role)))
        nll, cnt = eval_docs(model, docs)
        for _, _, m, conv in lins:
            set_w(m, conv, orig[id(m)])
        res["methods"][tag] = dict(nll=nll, cnt=cnt)
        print(f"{tag}: ppl {np.exp(sum(nll)/sum(cnt)):.3f} ({time.time()-t:.0f}s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB)", flush=True)

    t = time.time()
    nll, cnt = eval_docs(model, docs)
    res["methods"]["fp32"] = dict(nll=nll, cnt=cnt)
    print(f"fp32 ppl {np.exp(sum(nll)/sum(cnt)):.3f} ({time.time()-t:.0f}s)", flush=True)

    for nb in (4, 3):
        run_method(f"rtn_w{nb}", lambda w, xm, xr, mi: quantize_group(w, nb, GS))
        for clip in (False, True):
            info = {}

            def awq_fn(w, xm, xr, mi, nb=nb, clip=clip, info=info):
                r = awq_search(w, xr, nb, GS, x_stat=xm.view(1, -1))  # 평균 |x|는 전체 보정 토큰, 손실은 부분집합
                s = r["scales"]
                cm = None
                if clip:
                    cm = auto_clip(w * s.view(1, -1), xr / s.view(1, -1), nb, GS)
                wq = scaled_quantize(w, s, nb, GS, clip_max=cm)
                info[f"{mi[0]}.{mi[1]}"] = dict(alpha=r["alpha"], loss_rtn=r["loss_rtn"], loss_awq=r["loss"],
                                                loss_final=layer_loss(w, wq, xr))
                return wq
            run_method(f"awq{'_clip' if clip else ''}_w{nb}", awq_fn)
            res["awq_info"][f"{'clip' if clip else 'scale'}_w{nb}"] = info
    # Table 1형: FP16로 두는 채널 비율과 선택 기준 (W3)
    nb = 3
    for kind, fracs in (("act", (0.001, 0.01, 0.03)), ("weight", (0.01,)), ("random", (0.01,))):
        for fr in fracs:
            gens = {}

            def fn(w, xm, xr, mi, kind=kind, fr=fr, gens=gens):
                g = gens.setdefault("g", torch.Generator().manual_seed(SEED))
                idx = salient_channels(kind, w, xm, fr, g)
                return mixed_precision_quantize(w, idx, nb, GS)
            run_method(f"fp16_{kind}_{fr}_w{nb}", fn)
    # W4에서도 1% 활성 기준
    run_method("fp16_act_0.01_w4", lambda w, xm, xr, mi: mixed_precision_quantize(w, salient_channels("act", w, xm, 0.01), 4, GS))
    # Table 2형: 활성 상위 1% 채널만 s배 (W3), Delta 변화 통계
    for s_val in (1.0, 1.25, 1.5, 2.0, 4.0):
        acc = dict(groups=0, changed=0, ratio_sum=0.0)

        def fn(w, xm, xr, mi, s_val=s_val, acc=acc):
            idx = salient_channels("act", w, xm, 0.01)
            s = torch.ones(w.shape[1])
            s[idx] = s_val
            ws = w * s.view(1, -1)
            wg, wsg = w.reshape(w.shape[0], -1, GS), ws.reshape(w.shape[0], -1, GS)
            d0 = wg.amax(-1) - wg.amin(-1)
            d1 = wsg.amax(-1) - wsg.amin(-1)
            has = torch.zeros(w.shape[1] // GS, dtype=torch.bool)
            has[(idx // GS).unique()] = True          # 스케일한 채널이 든 그룹만 센다
            acc["groups"] += int(has.sum() * w.shape[0])
            acc["changed"] += int((d1[:, has] != d0[:, has]).sum())
            acc["ratio_sum"] += float((d1[:, has] / d0[:, has]).sum())
            return scaled_quantize(w, s, nb, GS)
        run_method(f"s_sweep_{s_val}_w{nb}", fn)
        res["table2"][str(s_val)] = dict(p_changed=acc["changed"] / acc["groups"], mean_ratio=acc["ratio_sum"] / acc["groups"],
                                         err_ratio=acc["ratio_sum"] / acc["groups"] / s_val, n_groups=acc["groups"])
    json.dump(res, open(outp, "w"))
    print("done", f"{time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
