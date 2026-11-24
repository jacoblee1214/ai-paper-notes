"""ROME(Meng et al., 2022)의 인과 추적과 rank-one 편집을 작은 합성 사실 모델에서 구현한 토이.

구성
  World      : 합성 사실(주어 = 이름 토큰 2개, 관계 4개 + 범주 관계 1개, 관계 템플릿 2종, 앞 문맥 토큰)
  GPT        : 4-6층 transformer(MLP 포함). 층별 hook으로 상태를 읽고 바꾼다.
  trace_*    : 인과 추적. 깨끗한 실행 / 주어 임베딩에 잡음(또는 다른 주어로 교체)을 넣은 실행 /
               잡음 실행에서 한 상태만 깨끗한 값으로 되돌린 실행. IE, AIE.
  rank_one_update, key_second_moment, compute_v_star, rome_edit : 논문 식 (2)-(4)
  evaluate_edit : efficacy / paraphrase / specificity(이웃) / 같은 주어의 다른 관계 / 전체 정확도

CPU에서 수 분 안에 도는 크기다. 사전학습 가중치나 외부 데이터를 쓰지 않는다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------
# 합성 사실 세계
# ----------------------------------------------------------------------------------------
@dataclass
class World:
    """주어 S명(이름 = 성 토큰 + 이름 토큰), 관계 R개 + 범주 관계 1개(IS_A, ROME의 'X is a' KL 항 대응).

    토큰 id: [이름 F][성 G][채움 NF][관계 템플릿 A: NR개][템플릿 B: 2*NR개][목적어][PAD]
    프롬프트: 채움 토큰 plen개(0-3) + 주어 2토큰 + 관계 템플릿(A는 1토큰, B는 2토큰).
    """
    seed: int = 0
    n_subj: int = 400
    n_first: int = 40
    n_last: int = 40
    n_fill: int = 20
    n_rel: int = 4          # 일반 관계 수. 관계 인덱스 n_rel 은 IS_A(범주)
    obj_per_rel: int = 30
    n_cat: int = 10

    def __post_init__(self):
        rng = np.random.RandomState(self.seed)
        F_, G_ = self.n_first, self.n_last
        pairs = rng.choice(F_ * G_, size=self.n_subj, replace=False)
        self.subj = np.stack([pairs // G_, F_ + pairs % G_], axis=1)  # (S,2) 토큰 id
        self.NR = self.n_rel + 1
        self.fill0 = F_ + G_
        self.relA0 = self.fill0 + self.n_fill
        self.relB0 = self.relA0 + self.NR
        self.obj0 = self.relB0 + 2 * self.NR
        self.n_obj_tok = self.n_rel * self.obj_per_rel + self.n_cat
        self.PAD = self.obj0 + self.n_obj_tok
        self.vocab = self.PAD + 1
        self.max_len = 3 + 2 + 2
        # 사실 표: obj[s, r] = 토큰 id
        self.obj = np.zeros((self.n_subj, self.NR), dtype=np.int64)
        for r in range(self.n_rel):
            self.obj[:, r] = self.obj0 + r * self.obj_per_rel + rng.randint(0, self.obj_per_rel, self.n_subj)
        self.obj[:, self.n_rel] = self.obj0 + self.n_rel * self.obj_per_rel + rng.randint(0, self.n_cat, self.n_subj)

    # 관계 r 의 목적어 토큰 범위 [lo, hi)
    def obj_range(self, r: int):
        if r < self.n_rel:
            lo = self.obj0 + r * self.obj_per_rel
            return lo, lo + self.obj_per_rel
        lo = self.obj0 + self.n_rel * self.obj_per_rel
        return lo, lo + self.n_cat

    def build(self, s, r, tmpl, plen, fills):
        """벡터화된 프롬프트 생성. 반환: tokens (B,max_len), last(마지막 토큰 위치), pos_first, pos_last(주어)."""
        s, r, tmpl, plen = map(np.asarray, (s, r, tmpl, plen))
        B = len(s)
        T = self.max_len
        tok = np.full((B, T), self.PAD, dtype=np.int64)
        j = np.arange(T)[None, :]
        pl = plen[:, None]
        fills = np.asarray(fills)
        tok = np.where(j < pl, fills[:, np.minimum(np.arange(T), 2)], tok)
        first = self.subj[s, 0][:, None]
        last = self.subj[s, 1][:, None]
        tok = np.where(j == pl, first, tok)
        tok = np.where(j == pl + 1, last, tok)
        relA = (self.relA0 + r)[:, None]
        relB1 = (self.relB0 + 2 * r)[:, None]
        relB2 = (self.relB0 + 2 * r + 1)[:, None]
        t = np.asarray(tmpl)[:, None]
        tok = np.where((j == pl + 2) & (t == 0), relA, tok)
        tok = np.where((j == pl + 2) & (t == 1), relB1, tok)
        tok = np.where((j == pl + 3) & (t == 1), relB2, tok)
        length = plen + 2 + 1 + tmpl
        return (torch.from_numpy(tok), torch.from_numpy(length - 1),
                torch.from_numpy(plen), torch.from_numpy(plen + 1))

    def random_fills(self, rng, B):
        return self.fill0 + rng.randint(0, self.n_fill, size=(B, 3))

    def sample_batch(self, rng, B):
        s = rng.randint(0, self.n_subj, B)
        r = rng.randint(0, self.NR, B)
        tmpl = rng.randint(0, 2, B)
        plen = rng.randint(0, 4, B)
        tok, last, _, _ = self.build(s, r, tmpl, plen, self.random_fills(rng, B))
        return tok, last, torch.from_numpy(self.obj[s, r])

    def sample_aux(self, rng, B):
        """보조 과제(변형 B): '주어 다음에 그 주어의 속성 토큰 하나가 온다'. 주어 마지막 토큰 위치에서 그 주어의 모든 속성
        (관계 4개 + 범주)에 대한 균등 분포를 맞춘다. 실제 LM이 관계를 모르는 채 주어 뒤를 예측해야 하는 압력의 대용."""
        s = rng.randint(0, self.n_subj, B)
        plen = rng.randint(0, 4, B)
        tok, _, _, pl = self.build(s, np.zeros(B, int), np.zeros(B, int), plen, self.random_fills(rng, B))
        target = torch.zeros(B, self.vocab)
        target.scatter_(1, torch.from_numpy(self.obj[s]), 1.0 / self.NR)
        return tok, pl, target


# ----------------------------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)      # W_fc : 키를 만드는 쪽
        self.proj = nn.Linear(4 * d, d)    # W_proj : ROME이 고치는 행렬. 가중치 모양 (d, 4d)

    def attn(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        sh = lambda t: t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class GPT(nn.Module):
    """pre-LN decoder. hooks = {사이트 이름: fn(x) -> x}. 사이트: 'embed', f'attn{l}', f'key{l}', f'mlp{l}', f'resid{l}'."""

    def __init__(self, vocab, d=128, n_layer=6, n_head=4, max_len=8):
        super().__init__()
        self.n_layer, self.d = n_layer, d
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, last, hooks=None):
        """idx (B,T), last (B,) 각 행에서 예측 위치. 반환 logits (B,V)."""
        hooks = hooks or {}
        hk = lambda name, x: hooks[name](x) if name in hooks else x
        h = hk("embed", self.tok(idx))
        h = h + self.pos(torch.arange(idx.shape[1], device=idx.device))[None]
        for l, b in enumerate(self.blocks):
            h = h + hk(f"attn{l}", b.attn(b.ln1(h)))
            key = hk(f"key{l}", F.gelu(b.fc(b.ln2(h))))
            h = h + hk(f"mlp{l}", b.proj(key))
            h = hk(f"resid{l}", h)
        h = self.ln_f(h)
        return self.head(h[torch.arange(idx.shape[0]), last])


def train_model(world: World, seed=0, d=128, n_layer=6, n_head=4, steps=4000, bs=1024, lr=3e-3, wd=0.01, log=None,
                aux_bs=0, aux_w=1.0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(1000 + seed)
    model = GPT(world.vocab, d, n_layer, n_head, world.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for it in range(steps):
        tok, last, y = world.sample_batch(rng, bs)
        loss = F.cross_entropy(model(tok, last), y)
        if aux_bs:
            atok, alast, atgt = world.sample_aux(rng, aux_bs)
            loss = loss - aux_w * (atgt * F.log_softmax(model(atok, alast), -1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and (it % 500 == 0 or it == steps - 1):
            log(f"  step {it} loss {loss.item():.4f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def probs_for(model, tok, last, hooks=None):
    return F.softmax(model(tok, last, hooks), dim=-1)


@torch.no_grad()
def fact_accuracy(model, world: World, n=None, tmpl=0, plen=2, seed=0):
    """모든(또는 n개 무작위) (주어, 관계) 사실의 top-1 정확도."""
    rng = np.random.RandomState(seed)
    s, r = np.meshgrid(np.arange(world.n_subj), np.arange(world.NR), indexing="ij")
    s, r = s.ravel(), r.ravel()
    if n is not None:
        idx = rng.choice(len(s), n, replace=False)
        s, r = s[idx], r[idx]
    B = len(s)
    tok, last, _, _ = world.build(s, r, np.full(B, tmpl), np.full(B, plen), world.random_fills(rng, B))
    p = probs_for(model, tok, last)
    return (p.argmax(-1).numpy() == world.obj[s, r]).mean()


# ----------------------------------------------------------------------------------------
# 인과 추적 (Section 2, 부록 B)
# ----------------------------------------------------------------------------------------
def embedding_std(model, world: World):
    """주어에 쓰이는 이름·성 토큰 임베딩 전체 원소의 표준편차(공식 코드 collect_embedding_std와 같은 정의)."""
    ids = torch.from_numpy(np.unique(world.subj))
    return model.tok(ids).detach().std().item()


def _window(l, w, L):
    lo = l - w // 2  # w=1:[l], w=3:[l-1,l,l+1]. 가장자리는 잘라 낸다(논문 부록 B.2와 같음)
    return [x for x in range(lo, lo + w) if 0 <= x < L]


@torch.no_grad()
def causal_trace(model, world: World, s, r, noise_mult=3.0, n_draws=10, plen=2, mode="GN",
                 windows=(1, 3), seed=0, sever=True, eps=None):
    """각 사실 (s_i, r_i)에 대해 층 x 위치 IE 격자를 구한다. 모든 복원은 '잡음 실행'에 깨끗한 값을 되돌려 넣는다.

    mode='GN'  : 주어 두 토큰 임베딩에 N(0, (noise_mult*sigma)^2) 잡음을 더한다(논문). noise_mult=0이면 잡음 없음.
    mode='STR' : 주어를 다른 주어로 바꾼다(Zhang & Nanda 2023의 symmetric token replacement). 잡음 없음.
    반환 dict:
      p_clean, p_corrupt (N,), L, T, pos_labels
      ie['resid'] (N,L,T)             : 잔차 상태 h_i^(l) 복원
      ie[f'mlp_w{w}'], ie[f'attn_w{w}'] (N,L,T) : 층 [l-w//2, ...] 창의 MLP(또는 attn) 출력 복원
      ie['resid_sever_mlp'], ie['resid_sever_attn'] (N,L,T) : 복원한 위치 i의 MLP(또는 attn) 출력을 잡음 실행 값으로 얼린 채 h_i^(l) 복원
    p_*: 정답 목적어 확률.
    """
    s, r = np.asarray(s), np.asarray(r)
    N = len(s)
    L = model.n_layer
    rng = np.random.RandomState(seed)
    tmpl = np.zeros(N, dtype=np.int64)
    fills = world.random_fills(rng, N)
    plens = np.full(N, plen)
    tok, last, pf, pl = world.build(s, r, tmpl, plens, fills)
    T = int(last[0]) + 1
    y = torch.from_numpy(world.obj[s, r])
    ar = torch.arange(N)

    # 깨끗한 실행: 상태 기록
    clean = {}

    def rec(name):
        def fn(x):
            clean[name] = x.detach().clone()
            return x
        return fn

    hooks = {}
    for l in range(L):
        for kind in ("resid", "mlp", "attn"):
            hooks[f"{kind}{l}"] = rec(f"{kind}{l}")
    p_clean = F.softmax(model(tok, last, hooks), -1)[ar, y]

    # 잡음 실행의 입력: n_draws 개씩 복제
    rep = lambda t: t.repeat_interleave(n_draws, dim=0)
    tokc, lastc, yc = rep(tok), rep(last), rep(y)
    pfc, plc = rep(pf), rep(pl)
    M = N * n_draws
    arM = torch.arange(M)
    emb_hook = None
    if mode == "GN":
        sigma = embedding_std(model, world)
        g = torch.Generator().manual_seed(seed + 77)
        noise = torch.randn(M, 2, model.d, generator=g) * (noise_mult * sigma) if eps is None else eps

        def emb_hook(x):
            # 주어 위치 두 곳에 잡음을 더한다
            x = x.clone()
            x[arM, pfc] += noise[:, 0]
            x[arM, plc] += noise[:, 1]
            return x
    elif mode == "STR":
        s_alt = np.empty(M, dtype=np.int64)
        for i in range(N):
            cand = rng.choice(np.delete(np.arange(world.n_subj), s[i]), n_draws, replace=False)
            s_alt[i * n_draws:(i + 1) * n_draws] = cand
        tokc, _, _, _ = world.build(s_alt, np.repeat(r, n_draws), np.repeat(tmpl, n_draws),
                                    np.repeat(plens, n_draws), np.repeat(fills, n_draws, axis=0))
    else:
        raise ValueError(mode)

    base_hooks = {"embed": emb_hook} if emb_hook else {}
    crep = {k: rep(v) for k, v in clean.items()}

    # 잡음 실행(기준선): MLP/attn 출력도 기록
    corr = {}

    def recc(name):
        def fn(x):
            corr[name] = x.detach().clone()
            return x
        return fn

    hk = dict(base_hooks)
    for l in range(L):
        hk[f"mlp{l}"] = recc(f"mlp{l}")
        hk[f"attn{l}"] = recc(f"attn{l}")
    p_corr = F.softmax(model(tokc, lastc, hk), -1)[arM, yc]

    def restored(hooks_extra):
        h = dict(base_hooks)
        h.update(hooks_extra)
        return F.softmax(model(tokc, lastc, h), -1)[arM, yc]

    def set_pos(src, i):
        def fn(x):
            x = x.clone()
            x[:, i] = src[:, i]
            return x
        return fn

    def combine(*fns_by_site):
        out = {}
        for d_ in fns_by_site:
            for k, f in d_.items():
                if k in out:
                    f0, f1 = out[k], f
                    out[k] = (lambda a, b: (lambda x: b(a(x))))(f0, f1)
                else:
                    out[k] = f
        return out

    kinds = ["resid"] + [f"{k}_w{w}" for k in ("mlp", "attn") for w in windows]
    if sever:
        kinds += ["resid_sever_mlp", "resid_sever_attn"]
    ie = {k: np.zeros((N, L, T)) for k in kinds}
    pc = p_corr.view(N, n_draws).mean(1).numpy()
    for i in range(T):
        for l in range(L):
            def avg(p):
                return p.view(N, n_draws).mean(1).numpy()
            ie["resid"][:, l, i] = avg(restored({f"resid{l}": set_pos(crep[f"resid{l}"], i)})) - pc
            for w in windows:
                for kind in ("mlp", "attn"):
                    hs = {f"{kind}{x}": set_pos(crep[f"{kind}{x}"], i) for x in _window(l, w, L)}
                    ie[f"{kind}_w{w}"][:, l, i] = avg(restored(hs)) - pc
            if sever:
                for kind in ("mlp", "attn"):
                    frozen = {f"{kind}{x}": set_pos(corr[f"{kind}{x}"], i) for x in range(L)}
                    hs = combine(frozen, {f"resid{l}": set_pos(crep[f"resid{l}"], i)})
                    ie[f"resid_sever_{kind}"][:, l, i] = avg(restored(hs)) - pc
    labels = (["fill"] * plen) + ["subj_first", "subj_last", "relation"]
    return dict(p_clean=p_clean.numpy(), p_corrupt=pc, L=L, T=T, pos_labels=labels, ie=ie)


def trace_restore_prob(model, world, s, r, layer, pos, noise_mult=3.0, n_draws=10, plen=2, seed=0):
    """(테스트용) 상태 h_pos^(layer) 하나를 되돌렸을 때의 P*, clean[o]. causal_trace의 'resid' 격자와 같은 값."""
    out = causal_trace(model, world, s, r, noise_mult=noise_mult, n_draws=n_draws, plen=plen, windows=(1,),
                       seed=seed, sever=False)
    return out["ie"]["resid"][:, layer, pos] + out["p_corrupt"], out


def summarize_trace(res):
    """AIE 격자와 ATE."""
    ie = {k: v.mean(0) for k, v in res["ie"].items()}
    return dict(ate=float((res["p_clean"] - res["p_corrupt"]).mean()), p_clean=float(res["p_clean"].mean()),
                p_corrupt=float(res["p_corrupt"].mean()), aie={k: v.tolist() for k, v in ie.items()},
                pos_labels=res["pos_labels"])


# ----------------------------------------------------------------------------------------
# ROME (Section 3, 부록 A)
# ----------------------------------------------------------------------------------------
def rank_one_update(W, b, k_star, v_star, C=None):
    """식 (2): W_hat = W + Lambda (C^-1 k*)^T,  Lambda = (v* - W k* - b) / ((C^-1 k*)^T k*).
    W (d_out,d_in), b (d_out,) 또는 None, C (d_in,d_in) 또는 None(= 단위행렬). 반환 delta (d_out,d_in).
    float64로 계산한다."""
    W, k, v = W.double(), k_star.double(), v_star.double()
    bb = 0.0 if b is None else b.double()
    u = k if C is None else torch.linalg.solve(C.double(), k)
    lam = (v - W @ k - bb) / (u @ k)
    return torch.outer(lam, u)


@torch.no_grad()
def key_second_moment(model, world: World, n_prompts_list=(100, 1000, 20000), chunk=500, seed=123):
    """각 층 W_proj 입력(키)의 비중심 2차 모멘트 C = E[k k^T]를 학습 분포의 프롬프트에서 모든 토큰 위치로 추정한다.
    (논문: 위키백과 100,000개 토큰 위치. 여기서는 합성 프롬프트의 PAD 아닌 모든 위치.)
    반환 {n_prompts: [C_l 층별 (4d,4d) float64]}, 토큰 수."""
    rng = np.random.RandomState(seed)
    L = model.n_layer
    D = 4 * model.d
    acc = [torch.zeros(D, D, dtype=torch.float64) for _ in range(L)]
    n_tok = 0
    done = 0
    out, toks = {}, {}
    for target in sorted(n_prompts_list):
        while done < target:
            B = min(chunk, target - done)
            tok, last, _ = world.sample_batch(rng, B)
            valid = torch.arange(tok.shape[1])[None, :] <= last[:, None]
            keys = {}
            hooks = {f"key{l}": (lambda l: lambda x: keys.__setitem__(l, x) or x)(l) for l in range(L)}
            model(tok, last, hooks)
            for l in range(L):
                kv = keys[l][valid].double()
                acc[l] += kv.T @ kv
            n_tok += int(valid.sum())
            done += B
        out[target] = [a / n_tok for a in acc]
        toks[target] = n_tok
    return out, toks


def make_contexts(world: World, rng, n_ctx=20):
    """k*, v* 계산에 쓰는 앞 문맥: 첫째는 문맥 없음, 나머지는 길이 1-3의 무작위 채움 토큰 (논문의 무작위 접두 문장 대응)."""
    plen = np.concatenate([[0], rng.randint(1, 4, n_ctx)])
    return plen, world.random_fills(rng, len(plen))


def _lookup_batch(world, s, r, plen, fills, tmpl=0, pos="last"):
    n = len(plen)
    tok, last, pf, pl = world.build(np.full(n, s), np.full(n, r), np.full(n, tmpl), plen, fills)
    lk = {"last": pl, "first": pf, "final": last}[pos]  # 주어 마지막 토큰 / 주어 첫 토큰 / 프롬프트 마지막(관계) 토큰
    return tok, last, lk


@torch.no_grad()
def compute_k_star(model, world, layer, s, r, plen, fills, pos="last"):
    tok, last, lk = _lookup_batch(world, s, r, plen, fills, pos=pos)
    got = {}
    model(tok, last, {f"key{layer}": lambda x: got.__setitem__("k", x) or x})
    k = got["k"][torch.arange(len(plen)), lk]
    return k.mean(0), k[0]  # 문맥 평균(논문 식 3), 문맥 없는 첫 프롬프트의 키(공식 코드가 나눗셈에 쓰는 값)


def compute_v_star(model, world, layer, s, r, o_new, plen, fills, pos="last", steps=60, lr=0.1,
                   kl_factor=0.0625, wd=0.5, clamp=4.0, early_stop=0.05):
    """식 (4): 층 layer의 MLP 출력에 더하는 벡터 delta를 최적화. 손실 = NLL(o*) + kl_factor*KL(new||orig on 'X is_a') + wd*|delta|/|v|^2.
    (kl_factor, wd, clamp 형태는 공식 코드 hparams/ROME/gpt2-xl.json과 compute_v.py를 따랐다. 논문 부록 E.5의 값과 다름: 04 D2)
    반환 v* = v_init + delta, v_init."""
    tok, last, lk = _lookup_batch(world, s, r, plen, fills, pos=pos)
    n = len(plen)
    ar = torch.arange(n)
    # 'X is_a' (본질 보존) 프롬프트
    ktok, klast, klk = _lookup_batch(world, s, world.n_rel, plen[:1], fills[:1], pos=pos)
    with torch.no_grad():
        got = {}
        model(tok, last, {f"mlp{layer}": lambda x: got.__setitem__("v", x) or x})
        v_init = got["v"][0, lk[0]].clone()
        kl0 = F.log_softmax(model(ktok, klast), -1)
    delta = torch.zeros(model.d, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    lo, hi = world.obj_range(r)
    for it in range(steps):
        def add_at(lkx, nn_):
            def fn(x):
                m = torch.zeros(nn_, x.shape[1], 1)
                m[torch.arange(nn_), lkx] = 1
                return x + m * delta
            return fn
        with torch.enable_grad():
            logp = F.log_softmax(model(tok, last, {f"mlp{layer}": add_at(lk, n)}), -1)
            nll = -logp[ar, o_new].mean()
            klp = F.log_softmax(model(ktok, klast, {f"mlp{layer}": add_at(klk, 1)}), -1)
            kl = (klp.exp() * (klp - kl0)).sum(-1).mean()
            reg = wd * delta.norm() / v_init.norm() ** 2
            loss = nll + kl_factor * kl + reg
            if loss.item() < early_stop or it == steps - 1:
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mx = clamp * v_init.norm()
            if delta.norm() > mx:
                delta.mul_(mx / delta.norm())
    return (v_init + delta).detach(), v_init, dict(loss=float(loss.detach()), nll=float(nll.detach()), kl=float(kl.detach()), steps=it + 1)


def rome_edit(model, world, layer, s, r, o_new, C, rng, pos="last", n_ctx=20, key_mode="mean", **vkw):
    """한 사실을 고친다. 모델은 바꾸지 않고 (delta W, 정보)를 돌려준다.
    key_mode='mean': 논문 식 (3)처럼 k*(문맥 평균)로 u=C^-1 k* 와 Lambda의 분모를 모두 계산.
    key_mode='plain': 공식 코드처럼 u는 문맥 평균 키로, Lambda의 분모는 문맥 없는 첫 프롬프트의 키로 계산(비대칭)."""
    plen, fills = make_contexts(world, rng, n_ctx)
    k_mean, k_plain = compute_k_star(model, world, layer, s, r, plen, fills, pos=pos)
    v_star, v_init, info = compute_v_star(model, world, layer, s, r, o_new, plen, fills, pos=pos, **vkw)
    blk = model.blocks[layer].proj
    W, b = blk.weight.detach(), blk.bias.detach()
    if key_mode == "mean":
        dW = rank_one_update(W, b, k_mean, v_star, C)
    else:
        u = k_mean.double() if C is None else torch.linalg.solve(C.double(), k_mean.double())
        # 공식 코드: right = (target - cur_output) / (cur_input . u), cur_input = 문맥 없는 프롬프트의 키, cur_output도 그 프롬프트의 출력
        tok, last, lk = _lookup_batch(world, s, r, plen[:1], fills[:1], pos=pos)
        got = {}
        model(tok, last, {f"mlp{layer}": lambda x: got.__setitem__("o", x) or x})
        cur_out = got["o"][0, lk[0]].double()
        dW = torch.outer((v_star.double() - cur_out) / (k_plain.double() @ u), u)
    info.update(k_norm=float(k_mean.norm()), dW_fro=float(dW.norm()), v_norm=float(v_init.norm()))
    return dW.float(), info


class Edited:
    """with Edited(model, layer, dW): ... 블록 안에서만 가중치를 고친다."""

    def __init__(self, model, layer, dW):
        self.w = model.blocks[layer].proj.weight
        self.dW = dW

    def __enter__(self):
        self.old = self.w.data.clone()
        self.w.data += self.dW
        return self

    def __exit__(self, *a):
        self.w.data.copy_(self.old)


# ----------------------------------------------------------------------------------------
# 편집 평가 (논문 3.3: efficacy / paraphrase / specificity)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_edit(model, world: World, s, r, o_old, o_new, seed=0, n_ctx_eval=20, n_nb=20, n_glob=200):
    """모델을 평가한다. seed가 같으면 같은 프롬프트(문맥, 이웃, 전역 사실)를 쓰므로 편집 전후를 같은 조건에서 비교할 수 있다.
    반환 dict:
      es / pm   : 편집 문장(템플릿 A, 새 무작위 문맥)에서 P[o*] > P[o_c] 비율과 P[o*] 평균 (논문 ES, 확률 크기)
      ps / pp   : 같은 사실의 다른 표현(템플릿 B, 무작위 문맥)에서 같은 지표 (논문 PS)
      nb_near   : 주어와 이름 또는 성 토큰을 공유하는 이웃 주어(같은 관계)의 top-1 정확도(원래 정답 유지)
      ns_near   : 논문 NS 정의 P[o_c(n)] > P[o*]
      nb_prob   : 이웃의 정답 확률 평균 (편집 전과 빼서 '확률 마진 손상'으로 본다)
      nb_rand   : 무작위 다른 주어의 정확도
      same_subj : 같은 주어의 다른 관계(템플릿 A) 정확도(IS_A 포함)
      glob_acc, glob_prob : 편집한 사실 외 무작위 사실 n_glob개의 정확도와 정답 확률 평균(전역 손상)
    """
    rng = np.random.RandomState(seed)
    out = {}
    plen = rng.randint(0, 4, n_ctx_eval)
    fills = world.random_fills(rng, n_ctx_eval)
    for name, tmpl in (("e", 0), ("p", 1)):
        tok, last, _ = _lookup_batch(world, s, r, plen, fills, tmpl=tmpl)
        p = probs_for(model, tok, last)
        pn, po = p[:, o_new], p[:, o_old]
        out["es" if name == "e" else "ps"] = float((pn > po).float().mean())
        out["pm" if name == "e" else "pp"] = float(pn.mean())
    near = np.where(((world.subj[:, 0] == world.subj[s, 0]) | (world.subj[:, 1] == world.subj[s, 1])))[0]
    near = near[near != s]
    near = rng.choice(near, min(n_nb, len(near)), replace=False)
    rand = rng.choice(np.delete(np.arange(world.n_subj), s), n_nb, replace=False)
    for name, grp in (("near", near), ("rand", rand)):
        B = len(grp)
        tok, last, _, _ = world.build(grp, np.full(B, r), np.zeros(B, int), np.full(B, 2), world.random_fills(rng, B))
        p = probs_for(model, tok, last)
        oc = torch.from_numpy(world.obj[grp, r])
        out["nb_" + name] = float((p.argmax(-1) == oc).float().mean())
        if name == "near":
            out["ns_near"] = float((p[torch.arange(B), oc] > p[:, o_new]).float().mean())
            out["nb_prob"] = float(p[torch.arange(B), oc].mean())
    rels = [x for x in range(world.NR) if x != r]
    B = len(rels)
    tok, last, _, _ = world.build(np.full(B, s), np.array(rels), np.zeros(B, int), np.full(B, 2), world.random_fills(rng, B))
    out["same_subj"] = float((probs_for(model, tok, last).argmax(-1).numpy() == world.obj[s, rels]).mean())
    gs = rng.randint(0, world.n_subj, n_glob)
    gr = rng.randint(0, world.NR, n_glob)
    keep = ~((gs == s) & (gr == r))
    gs, gr = gs[keep], gr[keep]
    B = len(gs)
    tok, last, _, _ = world.build(gs, gr, np.zeros(B, int), np.full(B, 2), world.random_fills(rng, B))
    p = probs_for(model, tok, last)
    oc = torch.from_numpy(world.obj[gs, gr])
    out["glob_acc"] = float((p.argmax(-1) == oc).float().mean())
    out["glob_prob"] = float(p[torch.arange(B), oc].mean())
    return out


def harmonic(*xs):
    xs = [max(x, 1e-9) for x in xs]
    return len(xs) / sum(1 / x for x in xs)
