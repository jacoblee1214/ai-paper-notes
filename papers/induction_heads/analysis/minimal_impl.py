"""In-context Learning and Induction Heads (Olsson et al. 2022)의 최소 구현.

논문 결과를 재현하는 코드가 아니라 정의와 측정을 직접 짜서 이해하려는 토이입니다.
  - TinyLM: attention-only 또는 MLP가 있는 작은 decoder-only transformer (pre-LN, 학습형 절대 위치 임베딩).
    논문의 smeared-key 변형(k_j <- s(a) k_j + (1-s(a)) k_{j-1})을 옵션으로 둡니다.
  - 측정: prefix-matching 점수, copying 점수(OV 고유값 / 직접 logit 경로), 이전 토큰 점수,
    K-composition 흔적(논문 부록 (4)), ICL 점수, head ablation(전체 / 패턴 유지).
  - 데이터: 반복 토큰 시퀀스(무작위 토큰 열을 되풀이), 파이썬 표준 라이브러리 소스+pydoc 문서 코퍼스.

CPU 또는 MPS에서 돈다. 사용법은 README.md, 실험 실행은 run_experiments.py.
"""
import hashlib
import math
import os
import re
import sysconfig
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS = 0


# --------------------------------------------------------------------------- 모델
@dataclass
class Cfg:
    vocab: int = 64
    d: int = 64
    n_layer: int = 2
    n_head: int = 2
    d_mlp: int = 0          # 0이면 attention-only
    ctx: int = 64
    smear: bool = False     # 논문 Argument 2의 smeared key

    def dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.h, self.dh = cfg.n_head, cfg.d // cfg.n_head
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Parameter(torch.randn(cfg.n_head, self.dh, cfg.d) * 0.02)   # head별 W_O
        self.alpha = nn.Parameter(torch.zeros(cfg.n_head)) if cfg.smear else None

    def forward(self, x, ablate_head=None, pattern=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if pattern is None:
            if self.alpha is not None:       # k_j <- s k_j + (1-s) k_{j-1}, 첫 토큰은 그대로
                s = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
                k = s * k + (1 - s) * torch.cat([k[:, :, :1], k[:, :, :-1]], 2)
            sc = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            pattern = sc.masked_fill(~mask, float("-inf")).softmax(-1)
        head_out = torch.einsum("bhtd,hdm->bhtm", pattern @ v, self.o)     # (B,H,T,d)
        if ablate_head is not None:
            head_out = head_out.clone()
            head_out[:, ablate_head] = 0
        return head_out.sum(1), pattern, head_out


class Block(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.d), Attention(cfg)
        self.mlp = None
        if cfg.d_mlp:
            self.ln2 = nn.LayerNorm(cfg.d)
            self.mlp = nn.Sequential(nn.Linear(cfg.d, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d))


class TinyLM(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.ctx, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lnf = nn.LayerNorm(cfg.d)
        self.unembed = nn.Linear(cfg.d, cfg.vocab, bias=False)
        for m in (self.tok, self.pos):
            nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward(self, tokens, ablate=None, freeze=None, cache=False):
        """ablate=(층, 헤드) 또는 그 튜플의 리스트: 그 헤드(들)의 출력을 0으로. freeze=패턴 리스트: 모든 층의 attention 패턴을 고정
        (논문의 pattern-preserving ablation: 깨끗한 실행의 패턴을 캐시해 두고 ablate 실행에 강제)."""
        B, T = tokens.shape
        pairs = [] if ablate is None else ([ablate] if isinstance(ablate[0], int) else list(ablate))
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))
        c = {"patterns": [], "head_out": [], "resid_post": [], "mlp_out": []}
        for l, b in enumerate(self.blocks):
            ah = [h for (ll, h) in pairs if ll == l] or None
            a, p, ho = b.attn(b.ln1(x), ah, None if freeze is None else freeze[l])
            x = x + a
            if b.mlp is not None:
                m = b.mlp(b.ln2(x))
                x = x + m
                c["mlp_out"].append(m)
            if cache:
                c["patterns"].append(p)
                c["head_out"].append(ho)
                c["resid_post"].append(x)
        logits = self.unembed(self.lnf(x))
        return (logits, c) if cache else logits


def per_pos_loss(logits, tokens):
    """(B, T-1): 위치 t의 값은 토큰 t+1을 맞히는 손실."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1),
                           reduction="none").view(tokens.shape[0], -1)


# --------------------------------------------------------------------------- 합성 데이터: 반복 토큰 열
def repeat_batch(gen, B, T, vocab, Lmin, Lmax, lo=1, hi=None):
    """BOS + (길이 L 무작위 토큰 열을 순환 반복해 T까지). L은 시퀀스마다 Lmin..Lmax에서 뽑는다.
    L이 고정이면 위치 임베딩만으로 'L칸 앞'을 보는 지름길이 생기므로 가변으로 둔다."""
    hi = vocab if hi is None else hi
    L = torch.randint(Lmin, Lmax + 1, (B,), generator=gen)
    base = torch.randint(lo, hi, (B, T), generator=gen)
    idx = (torch.arange(T - 1)[None, :] % L[:, None])
    toks = torch.gather(base, 1, idx)
    return torch.cat([torch.full((B, 1), BOS), toks], 1), L


def fixed_repeat_batch(gen, B, L, n_rep, lo, hi):
    """논문 evaluator의 입력: BOS + 길이 L 무작위 열을 n_rep번 반복."""
    base = torch.randint(lo, hi, (B, L), generator=gen)
    return torch.cat([torch.full((B, 1), BOS), base.repeat(1, n_rep)], 1)


def repeat_losses(logits, tokens, L):
    """반복 구간(토큰 위치 >= 1+L)과 첫 구간(1..L)의 평균 손실. L은 (B,) 또는 정수."""
    loss = per_pos_loss(logits, tokens)            # 대상 토큰 위치 = 인덱스+1
    tpos = torch.arange(1, tokens.shape[1], device=tokens.device)[None, :]
    L = torch.as_tensor(L, device=tokens.device).reshape(-1, 1) if not isinstance(L, int) else L
    rep = (tpos >= 1 + L).float().expand_as(loss)
    first = (tpos < 1 + L).float().expand_as(loss)
    return (loss * rep).sum() / rep.sum(), (loss * first).sum() / first.sum()


# --------------------------------------------------------------------------- 측정
def induction_target_mask(L, n_rep):
    """(T,T) 0/1: 목적지 i(2번째 반복 이후)가 '이전 반복에서 현재 토큰 다음 위치' j=i-kL+1에 두는 칸.
    i-kL >= 1 (이전 사본의 같은 토큰이 BOS가 아님)인 k=1,2,...를 모두 포함한다."""
    T = 1 + L * n_rep
    M = torch.zeros(T, T)
    for i in range(1 + L, T):
        k = 1
        while i - k * L >= 1:
            M[i, i - k * L + 1] = 1
            k += 1
    return M


def prefix_matching_score(pattern, L, n_rep):
    """pattern (..., T, T) -> (...). 2번째 반복 이후의 각 목적지에서, 이전 반복의 '현재 토큰 바로 다음 토큰'
    위치들에 둔 attention 질량의 평균. 논문 부록(3)은 '평균'이라고만 적었고 합/평균 정규화는 불명확해서
    위치당 질량(합)을 쓴다. 이상적 induction head는 1, 무작위 패턴은 작다."""
    M = induction_target_mask(L, n_rep).to(pattern)
    dest = torch.arange(1 + L, 1 + L * n_rep)
    return (pattern * M).sum(-1)[..., dest].mean(-1)


def previous_token_score(pattern):
    """모든 목적지 i>=1에서 i-1에 둔 attention의 평균. pattern (..., T, T)."""
    T = pattern.shape[-1]
    i = torch.arange(1, T)
    return pattern[..., i, i - 1].mean(-1)


def ov_copy_eig_score(E, O, U):
    """OV 회로 E·O·U (토큰 -> 토큰)의 고유값 중 양의 비율: sum(Re λ)/sum(|λ|), -1..1.
    E (V,d) 임베딩, O (d,d) 잔차 좌표의 W_V W_O, U (d,V) 언임베딩. 0이 아닌 고유값은 d x d 곱 O·U·E와 같다."""
    lam = torch.linalg.eigvals((O @ U @ E).detach().cpu().double())   # MPS에 eigvals 없음
    return (lam.real.sum() / lam.abs().sum().clamp_min(1e-12)).item()


def head_ov(model, l, h):
    """층 l 헤드 h의 잔차 좌표 OV (ln1 gamma 포함, 정규화 크기와 beta는 무시)."""
    b = model.blocks[l]
    d = model.cfg.d
    Wv = b.attn.qkv.weight.view(3, model.cfg.n_head, d // model.cfg.n_head, d)[2, h]   # (dh, d)  v = x @ Wv.T
    return torch.diag(b.ln1.weight) @ Wv.T @ b.attn.o[h]


def copy_scores_eig(model):
    """(층, 헤드) 별 OV 고유값 copying 점수."""
    E = model.tok.weight
    U = model.unembed.weight.T * model.lnf.weight[:, None]          # (d,V), ln_f gamma 포함
    return torch.tensor([[ov_copy_eig_score(E, head_ov(model, l, h), U) for h in range(model.cfg.n_head)]
                         for l in range(model.cfg.n_layer)])


def copy_scores_logit(model, cache, tokens, L, n_rep, K=None):
    """직접 logit 경로 copying 점수(논문 evaluator를 따르되 정규화는 우리 정의).
    각 목적지(2번째 반복 이후)에서 헤드 출력의 logit 기여를 '어휘 평균 빼고 ReLU' 한 뒤,
    그 헤드가 가장 크게 attend한 토큰의 logit 몫 / 표본에 나온 토큰들(K개)의 몫 합.
    (몫 - 1/K) / (1 - 1/K)로 맞춰 우연 = 0, 완전한 복사 = 1. 반환 (층, 헤드)."""
    uniq = len(set(tokens[0, 1:].tolist())) if K is None else K
    W = model.unembed.weight * model.lnf.weight[None, :]                    # (V,d)
    dest = torch.arange(1 + L, 1 + L * n_rep)
    out = torch.zeros(model.cfg.n_layer, model.cfg.n_head)
    onehot_ids = tokens[:, 1:]                                              # (B,L*n_rep) 표본 토큰
    for l in range(model.cfg.n_layer):
        ho = cache["head_out"][l][:, :, dest]                                # (B,H,D,d)
        lg = ho @ W.T                                                        # (B,H,D,V)
        lg = F.relu(lg - lg.mean(-1, keepdim=True))
        att_src = cache["patterns"][l][:, :, dest].argmax(-1)                # (B,H,D)
        att_tok = torch.gather(tokens[:, None, :].expand(-1, model.cfg.n_head, -1), 2, att_src)   # (B,H,D)
        num = torch.gather(lg, 3, att_tok[..., None]).squeeze(-1)
        # 분모: 시퀀스 b의 표본 토큰 집합에 대한 합 (중복 없이)
        den = torch.zeros_like(num)
        for b in range(tokens.shape[0]):
            ids = torch.unique(onehot_ids[b])
            den[b] = lg[b][:, :, ids].sum(-1)
        share = (num / den.clamp_min(1e-9))
        valid = den > 1e-9
        s = (share - 1 / uniq) / (1 - 1 / uniq)
        out[l] = torch.stack([s[:, h][valid[:, h]].mean() if valid[:, h].any() else torch.tensor(float("nan"))
                              for h in range(model.cfg.n_head)])
    return out


def kcomp_trace(model, l, h, prev_w):
    """논문 부록 (4): 층 l 헤드 h의 QK 회로 중, 앞 층(0) 헤드들이 '이전 토큰'에 attend하는 정도(prev_w)로 가중한
    순수 K-composition 항의 대각합. 양수이면 '현재 토큰 = 키 위치의 앞 토큰'을 선호(same-matching).
    row-vector 규약: score(e_q, e_p) = e_q B e_p^T, B = D Wq Wk^T D O_g^T, 대각합 = tr(B E^T E)."""
    d, H, dh = model.cfg.d, model.cfg.n_head, model.cfg.d // model.cfg.n_head
    b = model.blocks[l]
    W = b.attn.qkv.weight.view(3, H, dh, d)
    D = torch.diag(b.ln1.weight)
    Bq = D @ W[0, h].T @ W[1, h] @ D                                         # (d,d)
    E = model.tok.weight
    G = E.T @ E
    tot = 0.0
    for g in range(H):
        Og = head_ov(model, 0, g)
        tot += prev_w[g].item() * torch.trace(Bq @ Og.T @ G).item()
    return tot


@torch.no_grad()
def head_table(model, gen, device, L=25, n_rep=4, lo=1, hi=None, B=32, natural_tokens=None):
    """모든 헤드의 prefix-matching, copying(고유값, logit), 이전 토큰 점수. 반복 무작위 시퀀스 평가."""
    hi = model.cfg.vocab if hi is None else hi
    toks = fixed_repeat_batch(gen, B, L, n_rep, lo, hi).to(device)
    _, c = model(toks, cache=True)
    pm = torch.stack([prefix_matching_score(p, L, n_rep).mean(0) for p in c["patterns"]]).cpu()
    cl = copy_scores_logit(model, c, toks, L, n_rep).cpu()
    ce = copy_scores_eig(model).detach().cpu()
    if natural_tokens is None:
        pt = torch.stack([previous_token_score(p).mean(0) for p in c["patterns"]]).cpu()
    else:
        _, cn = model(natural_tokens.to(device), cache=True)
        pt = torch.stack([previous_token_score(p).mean(0) for p in cn["patterns"]]).cpu()
    return {"prefix": pm, "copy_logit": cl, "copy_eig": ce, "prev_token": pt}


@torch.no_grad()
def ablation_effects(model, toks, metric_fn, device, groups=None):
    """모든 헤드를 하나씩 끄고 metric_fn(logits, toks)의 변화를 잰다. 전체 ablation과 패턴 유지 ablation 둘 다.
    반환 dict: base, full[층][헤드], preserve[층][헤드] (ablate 값 - 기준값).
    groups={이름: [(층, 헤드), ...]}이면 그 묶음을 한꺼번에 끈 효과도 groups[이름] = {full, preserve}에 넣는다
    (같은 일을 하는 head가 여럿이면 하나씩 끄는 ablation은 중요도를 가린다)."""
    toks = toks.to(device)
    lg, c = model(toks, cache=True)
    base = float(metric_fn(lg, toks))
    frozen = [p.detach() for p in c["patterns"]]
    full, keep = [], []
    for l in range(model.cfg.n_layer):
        fr, kr = [], []
        for h in range(model.cfg.n_head):
            fr.append(float(metric_fn(model(toks, ablate=(l, h)), toks)) - base)
            kr.append(float(metric_fn(model(toks, ablate=(l, h), freeze=frozen), toks)) - base)
        full.append(fr)
        keep.append(kr)
    out = {"base": base, "full": full, "preserve": keep}
    if groups:
        out["groups"] = {name: {"full": float(metric_fn(model(toks, ablate=prs), toks)) - base,
                                "preserve": float(metric_fn(model(toks, ablate=prs, freeze=frozen), toks)) - base}
                         for name, prs in groups.items()}
    return out


# --------------------------------------------------------------------------- 학습 루프
def train(model, batch_fn, steps, lr, eval_fn=None, eval_every=50, warmup=100, wd=0.01, device="cpu"):
    """batch_fn(step) -> tokens (B,T). eval_fn(model, step) -> dict, eval_every 마다 기록. 학습률은 warmup 후 일정."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    hist = []
    for step in range(steps + 1):
        if eval_fn is not None and step % eval_every == 0:
            model.eval()
            hist.append({"step": step, **eval_fn(model, step)})
            model.train()
        if step == steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        tokens = batch_fn(step).to(device)
        loss = per_pos_loss(model(tokens), tokens).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return hist


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------- 자연어 코퍼스 (다운로드 없음)
def _stdlib_sources():
    std = sysconfig.get_paths()["stdlib"]
    skip = {"site-packages", "test", "tests", "__pycache__", "idlelib", "lib-dynload", "turtledemo", "ensurepip", "lib2to3"}
    out = []
    for root, dirs, fs in os.walk(std):
        dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("config-"))
        for f in sorted(fs):
            if f.endswith(".py"):
                out.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
    return out


def build_corpus(vocab=2048, chunk=1000, val_every=20):
    """파이썬 표준 라이브러리 소스(.py, 테스트·idlelib 제외) + pydoc_data.topics 문서 영어 산문.
    토큰: 정규식 \\w+ | 문장부호 | 개행(공백은 버림). 어휘: 학습 청크의 상위 vocab-2개 + UNK(id 1) (id 0 = BOS).
    1000토큰 청크로 잘라 20번째마다 검증용으로 뺀다. 같은 파이썬 설치에서는 결정적이다.
    반환 dict: train, val (int16 배열), itos (list), sha256(코퍼스 텍스트), n_tokens."""
    import pydoc_data.topics as topics
    text = "\n".join(_stdlib_sources() + [topics.topics[k] for k in sorted(topics.topics)])
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    toks = re.findall(r"\w+|[^\w\s]|\n", text)
    chunks = [toks[i:i + chunk] for i in range(0, len(toks), chunk)]
    tr = [c for i, c in enumerate(chunks) if i % val_every != val_every - 1]
    va = [c for i, c in enumerate(chunks) if i % val_every == val_every - 1]
    cnt = Counter(t for c in tr for t in c)
    itos = ["<bos>", "<unk>"] + [t for t, _ in cnt.most_common(vocab - 2)]
    stoi = {t: i for i, t in enumerate(itos)}
    enc = lambda cs: np.array([stoi.get(t, 1) for c in cs for t in c], dtype=np.int16)
    return {"train": enc(tr), "val": enc(va), "itos": itos, "sha256": sha, "n_tokens": len(toks)}


def val_windows(val, ctx):
    """검증 스트림을 겹치지 않는 ctx 길이 창으로 자르고 각 창의 첫 토큰을 BOS로 바꾼다."""
    n = len(val) // ctx
    w = torch.from_numpy(val[: n * ctx].astype(np.int64)).view(n, ctx).clone()
    w[:, 0] = BOS
    return w


def train_batch(train_arr, gen, B, ctx, dup=0.0):
    """학습 스트림에서 무작위 창 B개. 첫 토큰은 BOS. dup > 0이면 그 비율의 창은
    [BOS, p, p, x] (p = 창 앞쪽 (ctx-2)/2 토큰의 자연 텍스트 구간)로 바꾼다: 같은 구절이 두 번 나오는 문서를 흉내 낸 증강."""
    starts = torch.randint(0, len(train_arr) - ctx, (B,), generator=gen)
    idx = starts[:, None] + torch.arange(ctx)[None, :]
    w = torch.from_numpy(train_arr.astype(np.int64))[idx]
    w[:, 0] = BOS
    if dup > 0:
        h = (ctx - 2) // 2
        p = w[:, 1:1 + h]
        d = torch.cat([w[:, :1], p, p, w[:, -1:]], 1)
        pick = (torch.rand(B, generator=gen) < dup)[:, None]
        w = torch.where(pick, d, w)
    return w


def dup_windows(windows):
    """검증 창을 [BOS, p, p, x]로 바꾼 반복 구절 세트 (p = 창 앞쪽 (ctx-2)/2 토큰). 학습의 dup 증강과 같은 모양."""
    ctx = windows.shape[1]
    h = (ctx - 2) // 2
    p = windows[:, 1:1 + h]
    return torch.cat([windows[:, :1], p, p, windows[:, -1:]], 1), h


def dup_losses(model, windows, device, bs=64):
    """반복 구절 세트에서 (첫 사본 평균 손실, 둘째 사본 평균 손실). 둘째 사본이 훨씬 낮으면 문맥 복사가 된다."""
    d, h = dup_windows(windows)
    a = b = 0.0
    with torch.no_grad():
        for i in range(0, len(d), bs):
            t = d[i:i + bs].to(device)
            l = per_pos_loss(model(t), t)                # 대상 토큰 위치 1..T-1 -> 인덱스 0..T-2
            a += l[:, :h].sum().item()                   # 토큰 위치 1..h (첫 사본)
            b += l[:, h:2 * h].sum().item()              # 토큰 위치 h+1..2h (둘째 사본)
    n = len(d) * h
    return a / n, b / n


@torch.no_grad()
def val_position_loss(model, windows, device, bs=64):
    """검증 창들의 위치별 평균 손실 (ctx-1,)."""
    s, n = 0, 0
    for i in range(0, len(windows), bs):
        t = windows[i:i + bs].to(device)
        s = s + per_pos_loss(model(t), t).sum(0)
        n += t.shape[0]
    return (s / n).cpu()


def icl_score(pos_loss, early=(22, 28), late=(244, 250)):
    """ICL 점수 = (이른 위치 평균 손실) - (늦은 위치 평균 손실). 클수록 문맥에서 많이 배운다.
    논문은 '500번째 토큰 손실 - 50번째 토큰 손실'(음수가 됨)이라고 적었지만 보고하는 값(약 0.4 nats)은 부호를 뒤집은 양수다.
    우리는 양수 쪽을 쓴다. pos_loss[t]는 토큰 t+1의 손실이라 인덱스를 토큰 위치로 맞춘다(early, late는 토큰 위치)."""
    return float(pos_loss[early[0] - 1:early[1] - 1].mean() - pos_loss[late[0] - 1:late[1] - 1].mean())
