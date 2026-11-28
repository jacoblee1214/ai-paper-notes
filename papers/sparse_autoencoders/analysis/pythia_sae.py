"""R2: Pythia-70M 잔차 흐름에 SAE를 학습해 재구성을 끼웠을 때 perplexity와 loss recovered를 잰다 (논문 §6.2의 25 -> 40 확인).

  python pythia_sae.py --train    # 격자 (층 1-4 x alpha 4개, R=4) + 층 2 보조 격자 + 랜덤 초기화 모델 SAE -> pythia_sae_results.json
  python pythia_sae.py quick      # 아주 짧은 시험 실행(결과 파일을 쓰지 않음)
  python pythia_sae.py            # 아무것도 학습하지 않고 안내만 출력 (학습은 메모리를 많이 쓰므로 --train 을 명시해야 함)

주의: 학습 집합 활성(층당 153만 x 512 float32 약 3 GB)과 평가 활성, 모델, MPS 버퍼가 겹쳐 실행 중 메모리를 크게 쓴다(관측 상한은 03 §10 참고).

필요: torch, transformers, numpy, pyarrow 가 설치된 파이썬, 로컬 HF 캐시의 EleutherAI/pythia-70m 과 NeelNanda/pile-10k (HF_HUB_OFFLINE=1 권장).
SAE 정의·학습 규칙(식 (4), 디코더 행 정규화, Adam 1e-3, 활성 정규화)과 loss recovered 정의는 minimal_impl.py 를 그대로 쓴다.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import minimal_impl as M

HERE = Path(__file__).resolve().parent
MODEL = "EleutherAI/pythia-70m"
DATA = "NeelNanda/pile-10k"
CTX = 256
N_EVAL_SEQ = 400            # 문서 순서 기준 마지막 400개 창은 평가 전용(학습에 쓰지 않음)
MAX_TRAIN_SEQ = 6000        # 학습 창 상한 (활성 벡터 약 153만 개)
LAYERS = [1, 2, 3, 4]
ALPHAS = [0.01, 0.03, 0.1, 0.3]   # minimal_impl 규약: 평균 ||x||^2 = 1 로 정규화한 활성에서 제곱 오차 합 + alpha * L1
RATIO = 4
STEPS = 12000
BATCH = 1024
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def log(*a):
    print(*a, flush=True)


def load_tokens():
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    root = Path(snapshot_download(DATA, repo_type="dataset"))
    texts = pq.read_table(next(root.rglob("*.parquet")), columns=["text"]).column("text").to_pylist()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = []
    for t in texts:
        ids.extend(tok(t)["input_ids"] + [tok.eos_token_id])      # 문서 사이에 EOS, 하나의 스트림으로 이어 붙임
    n = len(ids) // CTX
    seqs = torch.tensor(ids[:n * CTX]).view(n, CTX)
    return seqs[:-N_EVAL_SEQ][:MAX_TRAIN_SEQ], seqs[-N_EVAL_SEQ:], n


def load_model(random_init=False):
    from transformers import AutoConfig, AutoModelForCausalLM
    if random_init:
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL))
    else:
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    return m.to(dev).eval()


def hook_layer(model, layer, fn=None, store=None):
    """gpt_neox.layers[layer] 의 출력(= 블록 layer 의 resid_post, 최종 LayerNorm 이전)을 읽거나(store 리스트) 바꾼다(fn)."""
    def h(mod, args, out):
        x = out[0] if isinstance(out, tuple) else out
        if store is not None:
            store.append(x[:, 1:].detach())
        if fn is not None:
            y = x.clone()
            y[:, 1:] = fn(x[:, 1:])                      # 위치 0(문서 첫 토큰, 어텐션 싱크)은 건드리지 않음
            return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
    return model.gpt_neox.layers[layer].register_forward_hook(h)


@torch.no_grad()
def collect(model, seqs, layer, bs=64):
    st = []
    hd = hook_layer(model, layer, store=st)
    for i in range(0, len(seqs), bs):
        model(seqs[i:i + bs].to(dev))
        st[-1] = st[-1].reshape(-1, st[-1].shape[-1]).cpu()
    hd.remove()
    return torch.cat(st)


@torch.no_grad()
def nll(model, seqs, layer=None, fn=None, bs=64):
    """평균 다음 토큰 교차 엔트로피(nats) (위치 1..T-1을 맞히는 항, 즉 창당 255개)."""
    hd = hook_layer(model, layer, fn=fn) if fn is not None else None
    tot, cnt = 0.0, 0
    for i in range(0, len(seqs), bs):
        t = seqs[i:i + bs].to(dev)
        lg = model(t).logits
        tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), t[:, 1:].reshape(-1), reduction="sum"))
        cnt += t[:, 1:].numel()
    if hd is not None:
        hd.remove()
    return tot / cnt


def train(Xtr, alpha, ratio, steps, seed=0):
    """minimal_impl.SAE + sae_loss 로 학습. train_sae 와 같은 규칙(별도 디코더, Adam 1e-3, 매 step 뒤 디코더 행 정규화); 데이터가 MPS 에 있어 루프만 다시 씀."""
    d = Xtr.shape[1]
    sae = M.SAE(d, ratio * d, tied=False, seed=seed).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    for _ in range(steps):
        x = Xtr[torch.randint(0, Xtr.shape[0], (BATCH,), generator=g).to(dev)]
        xh, c = sae(x)
        loss, _, _ = M.sae_loss(x, xh, c, alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.renorm_()
    return sae


def run_layer(model, tr_seqs, ev_seqs, layer, alphas, ratio, steps, base=None, full=True):
    t0 = time.time()
    Xtr_raw, Xev_raw = collect(model, tr_seqs, layer), collect(model, ev_seqs, layer)
    scale = float((Xtr_raw ** 2).sum(-1).mean().sqrt())
    Xtr, Xev = (Xtr_raw / scale).to(dev), Xev_raw / scale
    mean_vec = Xtr_raw.mean(0).to(dev)
    if base is None:
        L_clean = nll(model, ev_seqs)
        L_zero = nll(model, ev_seqs, layer, lambda x: torch.zeros_like(x))
        L_mean = nll(model, ev_seqs, layer, lambda x: mean_vec.expand_as(x))
        base = dict(L_clean=L_clean, L_zero=L_zero, L_mean=L_mean)
    log(f"layer {layer}: scale {scale:.2f}, train {len(Xtr)}, eval {len(Xev)}, base {base} ({time.time()-t0:.0f}s)")
    out = []
    for a in alphas:
        s = train(Xtr, a, ratio, steps)
        s_cpu = M.SAE(s.d, s.n_feat)
        s_cpu.load_state_dict({k: v.cpu() for k, v in s.state_dict().items()})
        ev = M.evaluate(s_cpu, Xev)
        r = dict(layer=layer, alpha=a, ratio=ratio, scale=scale, **ev)
        if full:
            L_rec = nll(model, ev_seqs, layer, lambda x, s=s: _recon(s, scale, x))
            r.update(L_rec=L_rec, **base,
                     ppl_clean=math.exp(base["L_clean"]), ppl_rec=math.exp(L_rec), ppl_zero=math.exp(base["L_zero"]), ppl_mean=math.exp(base["L_mean"]),
                     loss_recovered_zero=M.loss_recovered(base["L_clean"], L_rec, base["L_zero"]),
                     loss_recovered_mean=M.loss_recovered(base["L_clean"], L_rec, base["L_mean"]),
                     loss_recovered_strict=M.loss_recovered(base["L_clean"], L_rec, min(base["L_zero"], base["L_mean"])))
        log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), f"({time.time()-t0:.0f}s)")
        out.append(r)
    return out, base, scale


def _recon(sae, scale, x):
    with torch.no_grad():
        xh, _ = sae(x.reshape(-1, x.shape[-1]) / scale)
    return (xh * scale).reshape(x.shape)


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    if not quick and "--train" not in sys.argv[1:]:
        print("학습하지 않음. 결과는 pythia_sae_results.json 에 있다. 다시 학습하려면 --train (메모리 주의, 문서 참고).")
        return
    torch.manual_seed(0)
    tr, ev, n_all = load_tokens()
    log(f"창 {n_all}개(길이 {CTX}) 중 학습 {len(tr)}개, 평가 {len(ev)}개 = {ev.numel()} 토큰 (활성 벡터는 창당 255개)")
    model = load_model()
    res = dict(model=MODEL, data=DATA, ctx=CTX, n_train_seq=len(tr), n_eval_seq=len(ev), eval_tokens=int(ev.numel()),
               train_vectors=len(tr) * (CTX - 1), eval_vectors=len(ev) * (CTX - 1), steps=STEPS, batch=BATCH, hook="gpt_neox.layers[l] 출력 = resid_post(l), 0부터 센 층", main=[], extra_ratio=[], random_init=[])
    if quick:
        r, _, _ = run_layer(model, tr[:200], ev[:50], 2, [0.1], RATIO, 300)
        return
    path = HERE / "pythia_sae_results.json"
    if path.exists():                                   # 중단된 실행을 이어서 한다: 이미 끝난 층은 건너뜀
        old = json.load(open(path))
        for k in ("main", "extra_ratio", "random_init"):
            res[k] = old.get(k, [])
    done = {x["layer"] for x in res["main"] if sum(y["layer"] == x["layer"] for y in res["main"]) == len(ALPHAS)}
    bases = {x["layer"]: {k: x[k] for k in ("L_clean", "L_zero", "L_mean")} for x in res["main"]}
    for l in LAYERS:
        if l in done:
            continue
        res["main"] = [x for x in res["main"] if x["layer"] != l]
        r, bases[l], _ = run_layer(model, tr, ev, l, ALPHAS, RATIO, STEPS)
        res["main"] += r
        json.dump(res, open(path, "w"), indent=1)
    # 층 2 보조: 사전 크기 R=1, 8 (alpha = 0.03)
    if not res["extra_ratio"]:
        for R in (1, 8):
            r, _, _ = run_layer(model, tr, ev, 2, [0.03], R, STEPS, base=bases[2])
            res["extra_ratio"] += r
        json.dump(res, open(path, "w"), indent=1)
    # 랜덤 초기화 모델의 층 2 (같은 데이터, 같은 alpha 두 개; 이 모델의 정상 손실을 기준으로 loss recovered도 계산하지만 참고용)
    if not res["random_init"]:
        rm = load_model(random_init=True)
        r, b, _ = run_layer(rm, tr, ev, 2, [0.03, 0.1], RATIO, STEPS, full=True)
        res["random_init"] = r
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
