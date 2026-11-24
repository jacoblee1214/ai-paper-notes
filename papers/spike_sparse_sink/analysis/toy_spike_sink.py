"""Spike(거대 activation)와 sink가 첫 토큰에 함께 생기는지 보는 CPU/MPS 토이.

- 데이터: 이 저장소 다른 폴더의 paper/paper.txt(영문 논문 텍스트 약 3 MB, 연속 공백은 1칸으로)를 바이트 단위(어휘 256)로. 길이 T=64의 무작위 창, BOS 없음: 위치 0은 아무 바이트.
- 모델: pre-norm RMSNorm + SwiGLU + RoPE, 4층, d=128, 4 head(dhead 32). 변형: prenorm, sandwich(블록 출력에 RMSNorm 추가).
- 지표: (a) spike 비 = 위치 0의 최대 |activation| / 나머지 위치 전체 |activation|의 중앙값 (층별, 잔차 흐름)
        (b) sink ratio = 논문 식 (27)–(28): 헤드별로 첫 T/2 위치 중 평균 받은 attention 최댓값 > eps 인 헤드 비율
        (c) 개입: 위치 0의 spike 채널 상위 m개를 매 층 출력 뒤 0으로 만들기, 같은 수의 무작위 채널 0으로 만들기(대조)
논문 수치와 비교하지 않고 방향만 본다. 실행: TOY_LR=6e-3 python toy_spike_sink.py 4000 0,1,2  (torch 필요, TOY_TEXT로 영문 텍스트 파일 지정 가능)
"""
import json, math, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
V, T, D, L, H = 256, 64, 128, 4, 4
DH = D // H
FF = 344


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-5):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        return s.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps)


def rope(x, base=10000.0):
    B, Hh, Tt, Dh = x.shape
    pos = torch.arange(Tt, device=x.device, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, Dh, 2, device=x.device, dtype=torch.float32) / Dh))
    f = torch.outer(pos, inv); cos, sin = f.cos(), f.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, sandwich):
        super().__init__()
        s.n1, s.n2 = RMSNorm(D), RMSNorm(D)
        s.q, s.k, s.v, s.o = [nn.Linear(D, D, bias=False) for _ in range(4)]
        s.q, s.k, s.v, s.o = s.q, s.k, s.v, s.o
        s.gate, s.up = nn.Linear(D, FF, bias=False), nn.Linear(D, FF, bias=False)
        s.down = nn.Linear(FF, D, bias=False)
        s.sandwich = sandwich
        if sandwich:
            s.p1, s.p2 = RMSNorm(D), RMSNorm(D)

    def attn(s, x, want_map):
        B, Tt, _ = x.shape
        sp = lambda t: t.view(B, Tt, H, DH).transpose(1, 2)
        q, k, v = rope(sp(s.q(x))), rope(sp(s.k(x))), sp(s.v(x))
        if want_map:
            sc = q @ k.transpose(-1, -2) / math.sqrt(DH)
            sc = sc.masked_fill(torch.triu(torch.ones(Tt, Tt, dtype=torch.bool, device=x.device), 1), float("-inf"))
            A = sc.softmax(-1); o = A @ v
        else:
            A = None; o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(o.transpose(1, 2).reshape(B, Tt, D)), A

    def forward(s, x, want_map=False):
        a, A = s.attn(s.n1(x), want_map)
        if s.sandwich: a = s.p1(a)
        h = x + a
        f = s.down(F.silu(s.gate(s.n2(h))) * s.up(s.n2(h)))
        if s.sandwich: f = s.p2(f)
        return h + f, A


class LM(nn.Module):
    def __init__(s, sandwich):
        super().__init__()
        s.emb = nn.Embedding(V, D)
        s.blocks = nn.ModuleList([Block(sandwich) for _ in range(L)])
        s.norm = RMSNorm(D); s.head = nn.Linear(D, V, bias=False)
        for m in s.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(s.emb.weight, std=1.0)

    def forward(s, idx, record=False, ablate=None):
        x = s.emb(idx); hs, maps = [x], []
        for b in s.blocks:
            x, A = b(x, want_map=record)
            if ablate is not None:  # 위치 0의 지정 채널을 0으로
                x = x.clone(); x[:, 0, ablate] = 0.0
            if record: hs.append(x); maps.append(A)
        out = s.head(s.norm(x))
        return (out, hs, maps) if record else out


def load_text():
    import glob, re, os
    if os.environ.get("TOY_TEXT"):  # 영문 텍스트 파일 하나(수 MB 권장)를 직접 지정
        return torch.tensor(list(open(os.environ["TOY_TEXT"], "rb").read()), dtype=torch.uint8)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    me = os.path.abspath(os.path.join(root, "spike_sparse_sink"))
    parts = []
    for f in sorted(glob.glob(os.path.join(root, "*", "paper", "paper.txt"))):
        if os.path.abspath(os.path.dirname(os.path.dirname(f))) == me: continue
        parts.append(re.sub(r"[ \t]+", " ", open(f, encoding="utf-8", errors="ignore").read()))
    b = "\n".join(parts).encode("utf-8", errors="ignore")
    return torch.tensor(list(b), dtype=torch.uint8)

_TEXT = None
def make_data(seed, n):
    global _TEXT
    if _TEXT is None: _TEXT = load_text()
    cut = int(len(_TEXT) * 0.9)
    g = torch.Generator().manual_seed(1000 + seed)
    def win(lo, hi, m):
        st = torch.randint(lo, hi - T, (m,), generator=g)
        return torch.stack([_TEXT[s:s + T] for s in st.tolist()]).long()
    return _TEXT[:cut].long(), win(cut, len(_TEXT), n)


def sink_ratio(maps, eps):
    # maps: list of [B,H,T,T]; 식 (27): alpha_k = mean_t A[t,k]; (28): 첫 T/2 위치 최댓값 > eps 인 헤드 비율, 시퀀스 평균
    r = []
    for A in maps:
        alpha = A.mean(2)                        # [B,H,T]
        r.append((alpha[:, :, : T // 2].max(-1).values > eps).float().mean(0))  # [H]
    return torch.stack(r).mean().item()


@torch.no_grad()
def measure(model, X, ablate=None):
    model.eval()
    out, hs, maps = model(X, record=True, ablate=ablate)
    loss = F.cross_entropy(out[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1)).item()
    res = {"loss": loss}
    ratios = []
    for l, h in enumerate(hs[1:], 1):
        a = h.abs()
        rest = a[:, 1:].reshape(-1)
        med = rest.median().item()
        ratios.append({"layer": l, "pos0_max": a[:, 0].max(-1).values.mean().item(), "median_rest": med,
                       "ratio": a[:, 0].max(-1).values.mean().item() / med,
                       "rest_max": a[:, 1:].max(-1).values.max(-1).values.mean().item()})
    res["layers"] = ratios
    res["sink"] = {str(e): sink_ratio(maps, e) for e in (0.1, 0.2, 0.3, 0.5)}
    # 첫 토큰이 받는 평균 attention (모든 층·헤드 평균)
    res["alpha0_mean"] = torch.stack([A.mean(2)[:, :, 0].mean() for A in maps]).mean().item()
    return res, hs


import os
LR = float(os.environ.get("TOY_LR", "2e-3"))
VARS = os.environ.get("TOY_VARIANTS", "prenorm,sandwich").split(",")


def train(variant, seed, steps, bs=128, lr=LR):
    torch.manual_seed(seed)
    tr, ev = make_data(seed, 2000)
    tr, ev = tr.to(DEV), ev.to(DEV)
    P = None; ar = torch.arange(T, device=DEV)
    model = LM(variant == "sandwich").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = lambda s: min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr * sched(s)
        st = torch.randint(0, tr.shape[0] - T, (bs,), device=DEV)
        idx = tr[st[:, None] + ar[None, :]]
        out = model(idx)
        loss = F.cross_entropy(out[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model, ev, P


def run(variant, seed, steps):
    t0 = time.time()
    model, ev, P = train(variant, seed, steps)
    base, hs = measure(model, ev)
    # spike 채널: 중간층(층 2) 위치 0 |activation| 평균 상위 m개
    mid = 2
    prof = hs[mid][:, 0].abs().mean(0)
    top = prof.topk(3).indices
    g = torch.Generator().manual_seed(seed)
    rnd = torch.randperm(D, generator=g)[:3].to(DEV)
    ab, _ = measure(model, ev, ablate=top)
    rc, _ = measure(model, ev, ablate=rnd)
    return {"variant": variant, "seed": seed, "steps": steps, "base": base, "top_channels": top.tolist(),
            "ablate_top3": ab, "ablate_random3": rc, "random_channels": rnd.tolist(), "sec": time.time() - t0}


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    out = sys.argv[3] if len(sys.argv) > 3 else "toy_spike_sink_results.json"
    allr = []
    for v in VARS:
        for sd in seeds:
            r = run(v, sd, steps); allr.append(r)
            b = r["base"]
            print(v, sd, "loss %.3f" % b["loss"], "a0 %.3f" % b["alpha0_mean"],
                  "ratio(L2) %.1f" % b["layers"][1]["ratio"], "sink0.3 %.3f" % b["sink"]["0.3"],
                  "| zero top3: loss %.3f sink %.3f" % (r["ablate_top3"]["loss"], r["ablate_top3"]["sink"]["0.3"]),
                  "| zero rnd3: loss %.3f" % r["ablate_random3"]["loss"], "%.0fs" % r["sec"], flush=True)
    json.dump({"device": DEV, "config": dict(V=V, T=T, D=D, L=L, H=H, FF=FF), "runs": allr}, open(out, "w"), indent=1)
