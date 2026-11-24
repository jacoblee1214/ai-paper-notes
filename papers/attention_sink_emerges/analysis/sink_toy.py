"""직접 짠 작은 decoder-only transformer 토이: softmax 대 정규화 없는 sigmoid attention의 첫 토큰 sink.

논문 4–5, 7.4절의 질문(학습 단계에 따라 언제 sink가 생기나, softmax 정규화를 없애면 사라지나)을
수 분 안에 도는 크기로 방향만 본다. 논문 수치와 직접 비교하지 않는다.

- 구조: pre-norm, RMSNorm, RoPE, SwiGLU (논문 기본 설정과 같은 종류), 층 L, 폭 d, 헤드 H
- attention 변형 (논문 표 6 행과 같은 이름)
    softmax          : 표준
    sigmoid_nonorm   : sigmoid(qk/sqrt(dh)) 를 정규화 없이 value에 곱함 (논문의 핵심 비교군)
    sigmoid_norm     : sigmoid 점수를 행 합으로 나눔 (논문 표 6의 대조군, sink가 다시 나온다고 보고)
- sink 지표: 논문 식 그대로 alpha_1 = (1/T) sum_i A[i,1], 헤드가 alpha_1 > eps 이면 sink, 모든 헤드·층 평균.
  정규화 없는 sigmoid는 논문처럼 proxy 점수 A_ij = s_ij / sum_j' s_ij' 로 잰다.
- 데이터: 고정된 무작위 희소 bigram 마르코프 사슬(전역 "언어")에서 뽑은 길이 T 시퀀스.
  확률 p_copy 로 뒷절반이 앞절반의 복사(induction이 쓸모 있는 구간)가 된다.
  손실은 논문처럼 첫 토큰 예측을 제외하고 t>=2 위치 다음 토큰에만 건다(식 1 아래 "p(x1) is ignored").
실행: python3 -I analysis/sink_toy.py  (torch 필요, MPS 또는 CPU)
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EPS_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def rope_cache(T, dh, device):
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = torch.outer(torch.arange(T, device=device).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,dh
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Block(nn.Module):
    def __init__(self, d, H, ff, kind):
        super().__init__()
        self.H, self.dh, self.kind = H, d // H, kind
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin, want_attn=False):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        if self.kind == "softmax":
            A = s.masked_fill(~mask, float("-inf")).softmax(-1)
            P = A  # 지표에 쓰는 점수
        else:
            sig = torch.sigmoid(s) * mask
            if self.kind == "sigmoid_norm":
                A = sig / (sig.sum(-1, keepdim=True) + 1e-8)
                P = A
            else:  # sigmoid_nonorm: 출력에는 정규화 없는 sig, 지표에는 proxy
                A = sig
                P = sig / (sig.sum(-1, keepdim=True) + 1e-8)
        y = (A @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(y)
        h = self.n2(x)
        x = x + self.w3(F.silu(self.w1(h)) * self.w2(h))
        return x, (P.detach() if want_attn else None)


class LM(nn.Module):
    def __init__(self, V, d, L, H, ff, kind):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, H, ff, kind) for _ in range(L)])
        self.nf = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.H, self.dh = H, d // H
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, want_attn=False):
        cos, sin = rope_cache(idx.shape[1], self.dh, idx.device)
        x = self.emb(idx)
        attns = []
        for b in self.blocks:
            x, A = b(x, cos, sin, want_attn)
            attns.append(A)
        return self.head(self.nf(x)), attns


def load_text(max_bytes=40_000_000):
    """파이썬 표준 라이브러리 소스(.py)를 바이트로 이어 붙인 코퍼스(PSF 라이선스, 설치된 파일만 읽음).
    파일 이름 순서로 이어 붙이고 마지막 5%를 검증용으로 쓴다. 다운로드 없음."""
    import sysconfig
    root = Path(sysconfig.get_paths()["stdlib"])
    buf, n = [], 0
    for f in sorted(root.glob("*.py")) + sorted(root.glob("*/*.py")):
        if "site-packages" in str(f) or "test" in f.parts[-2:-1]:
            continue
        b = f.read_bytes()
        buf.append(np.frombuffer(b, dtype=np.uint8))
        n += len(b)
        if n >= max_bytes:
            break
    arr = np.concatenate(buf)
    cut = int(len(arr) * 0.95)
    return arr[:cut], arr[cut:]


def sample_text(arr, B, T, rng):
    """무작위 위치의 길이 T 조각. 첫 토큰은 문서(파일) 중간의 아무 바이트다(논문의 packing과 같은 성격)."""
    off = rng.randint(0, len(arr) - T, B)
    return arr[off[:, None] + np.arange(T)[None, :]].astype(np.int64)


def make_language(V, branch, seed):
    """고정된 희소 bigram 전이: 각 토큰에서 branch개의 후속 토큰만 가능."""
    rng = np.random.RandomState(seed)
    nxt = np.stack([rng.choice(V, branch, replace=False) for _ in range(V)])
    pr = rng.dirichlet(np.ones(branch), size=V)
    return nxt, pr


def sample_batch(nxt, pr, B, T, p_copy, rng):
    V, br = nxt.shape
    out = np.zeros((B, T), dtype=np.int64)
    out[:, 0] = rng.randint(0, V, B)
    u = rng.rand(B, T)
    cum = pr.cumsum(1)
    for t in range(1, T):
        j = (u[:, t:t + 1] > cum[out[:, t - 1]]).sum(1).clip(max=br - 1)
        out[:, t] = nxt[out[:, t - 1], j]
    half = T // 2
    cp = rng.rand(B) < p_copy
    out[cp, half:] = out[cp, :half]
    return out


@torch.no_grad()
def measure(model, x, T_eval):
    """x: (N, T). 반환: eps별 Sink_1 (%), alpha_1 의 헤드별 평균, 평균 loss."""
    model.eval()
    logits, attns = model(x[:, :T_eval], want_attn=True)
    loss = F.cross_entropy(logits[:, 1:-1].reshape(-1, logits.shape[-1]), x[:, 2:T_eval].reshape(-1)).item()
    ar = torch.arange(T_eval, 0, -1, device=x.device).float()  # 열 k 의 분모 T-k+1
    al = []
    for A in attns:  # N,H,T,T
        al.append((A.sum(-2) / ar)[..., 0])  # alpha_1: N,H
    al = torch.stack(al, 1)  # N,L,H
    res = {f"{e}": (al > e).float().mean().item() * 100 for e in EPS_LIST}
    model.train()
    return res, al.mean(0).cpu().numpy(), loss


def run(kind, seed, cfg, dev):
    torch.manual_seed(seed)
    rng = np.random.RandomState(1000 + seed)
    if cfg["data"] == "text":
        tr, va = load_text()
        draw = lambda B, r: sample_text(tr, B, cfg["T"], r)
        ev = torch.tensor(sample_text(va, 100, cfg["T"], np.random.RandomState(999)), device=dev)
    else:
        nxt, pr = make_language(cfg["V"], cfg["branch"], seed=7)  # 언어는 모든 실행에서 같다
        draw = lambda B, r: sample_batch(nxt, pr, B, cfg["T"], cfg["p_copy"], r)
        ev = torch.tensor(sample_batch(nxt, pr, 100, cfg["T"], cfg["p_copy"], np.random.RandomState(999)), device=dev)
    model = LM(cfg["V"], cfg["d"], cfg["L"], cfg["H"], cfg["ff"], kind).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["wd"])
    hist = []
    t0 = time.time()
    for step in range(cfg["steps"] + 1):
        if step % cfg["log_every"] == 0:
            res, al, loss = measure(model, ev, cfg["T"])
            hist.append({"step": step, "sink": res, "alpha1_mean": float(al.mean()), "val_loss": loss})
        lr = cfg["lr"] * min(1.0, step / cfg["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.tensor(draw(cfg["B"], rng), device=dev)
        logits, _ = model(x)
        loss = F.cross_entropy(logits[:, 1:-1].reshape(-1, cfg["V"]), x[:, 2:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    res, al, loss = measure(model, ev, cfg["T"])
    return {"kind": kind, "seed": seed, "hist": hist, "final_alpha1_per_layer_head": al.tolist(),
            "final": {"sink": res, "val_loss": loss}, "seconds": time.time() - t0}


if __name__ == "__main__":
    cfg = dict(data="text", V=256, branch=4, T=64, d=128, L=4, H=4, ff=256, B=64, steps=2000, warmup=100,
               lr=2e-3, wd=0.1, p_copy=0.5, log_every=100)
    seeds = [0, 1, 2]
    kinds = ["softmax", "sigmoid_nonorm", "sigmoid_norm"]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        cfg["steps"], seeds, kinds = 600, [0], ["softmax", "sigmoid_nonorm"]
        if len(sys.argv) > 2:  # 탐색용: JSON으로 cfg 일부 덮어쓰기
            cfg.update(json.loads(sys.argv[2]))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {"cfg": cfg, "device": dev, "torch": torch.__version__, "runs": []}
    for kind in kinds:
        for s in seeds:
            r = run(kind, s, cfg, dev)
            out["runs"].append(r)
            print(f"{kind:15s} seed {s} loss {r['final']['val_loss']:.3f} sink@0.3 {r['final']['sink']['0.3']:.1f}% "
                  f"({r['seconds']:.0f}s)", flush=True)
    name = "sink_toy_quick.json" if len(sys.argv) > 1 else "sink_toy_results.json"
    (ROOT / "analysis" / name if len(sys.argv) == 1 else ROOT / ".scratch" / name).write_text(json.dumps(out))
