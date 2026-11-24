"""Sparse Autoencoders Find Highly Interpretable Features in Language Models (Cunningham et al., 2023) 최소 구현.

논문 결과를 재현하는 코드가 아니라, SAE의 정의와 측정을 직접 짜서 이해하려는 토이다(공식 저장소 코드는 복사하지 않았고 정적으로 읽기만 했다).

구성
  1. SAE: 식 (1)–(4). c = ReLU(Mx+b), x_hat = M^T c (묶은 가중치) 또는 별도 디코더, 디코더 행 정규화, 손실 ||x-x_hat||^2 + alpha ||c||_1
  2. 지표: 설명 못 한 분산 비율(FVU), 평균 L0, 죽은 특징 비율, MMCS와 정답 복원율, 수축(shrinkage) 기울기와 그 이론값
  3. 정답 있는 시험대: 30번(toy_superposition)의 generate_batch로 만든 희소 특징을 무작위 단위 방향에 심어 SAE에 먹임
  4. 29번 모델(induction_heads) 로드와 SAE 재구성 끼우기(loss recovered)
  5. 해석 가능성의 대용 지표(상위 activation 토큰의 집중도)와 기저·PCA·무작위 방향 기준선
  6. 29번 A1: 층 1 head 출력 방향과 SAE 특징의 연결

출처 주석
  - generate_batch: ../toy_superposition/analysis/minimal_impl.py (30번)의 같은 이름 함수를 복사(인스턴스 축은 1로 고정).
  - Cfg, Attention, Block, TinyLM: ../induction_heads/analysis/minimal_impl.py (29번)를 복사해 훅(hooks) 인자를 더하고 smear 옵션을 뺌.
    그 폴더가 바뀌어도 이 분석이 깨지지 않게 하려는 복사다. state_dict 키는 29번 체크포인트와 같다.
CPU에서 돈다.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 1. SAE
# --------------------------------------------------------------------------


class SAE(nn.Module):
    """c = ReLU(E x + b),  x_hat = c @ D,  D의 행(사전 원소)은 단위 노름.

    tied=True : 논문 식 (1)–(2). 인코더 E = M, 디코더 D = M (M은 행 정규화). 파라미터는 M과 b뿐.
    tied=False: 논문 부록 C.3 식 (5)–(6). 인코더 E는 자유, 디코더 D는 행 정규화.
    행 정규화는 매 optimizer step 뒤에 D의 행을 단위 노름으로 되돌려 지킨다(논문 각주 3: ||c||_1을 키우는 대신 D를 줄이는 꼼수를 막음).
    """

    def __init__(self, d: int, n_feat: int, tied: bool = False, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        D0 = F.normalize(torch.randn(n_feat, d, generator=g), dim=1)
        self.d, self.n_feat, self.tied = d, n_feat, tied
        self.W_dec = nn.Parameter(D0.clone())
        self.W_enc = None if tied else nn.Parameter(D0.clone())
        self.b = nn.Parameter(torch.zeros(n_feat))

    def encoder_matrix(self) -> torch.Tensor:
        return self.W_dec if self.tied else self.W_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.encoder_matrix().T + self.b)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.W_dec

    def forward(self, x: torch.Tensor):
        c = self.encode(x)
        return self.decode(c), c

    def dictionary(self) -> torch.Tensor:
        """단위 노름 사전 [n_feat, d] (디코더 행)."""
        return F.normalize(self.W_dec.detach(), dim=1)

    @torch.no_grad()
    def renorm_(self):
        self.W_dec.copy_(F.normalize(self.W_dec, dim=1))


def sae_loss(x, x_hat, c, alpha: float):
    """논문 식 (4): L = ||x - x_hat||_2^2 + alpha ||c||_1  (표본별 합, 배치 평균)."""
    rec = ((x - x_hat) ** 2).sum(-1).mean()
    l1 = c.abs().sum(-1).mean()
    return rec + alpha * l1, rec, l1


def train_sae(data, d: int, n_feat: int, alpha: float, *, tied: bool = False, steps: int = 2000, batch: int = 512,
              lr: float = 1e-3, seed: int = 0) -> SAE:
    """data: 텐서 [N, d] (에폭 없이 무작위 복원추출 미니배치) 또는 callable(batch, gen) -> [batch, d] (무한 데이터).
    Adam, lr 1e-3(논문 부록 B), 배치 512, 매 step 뒤 디코더 행 정규화."""
    sae = SAE(d, n_feat, tied=tied, seed=seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1000 + seed)
    for _ in range(steps):
        if callable(data):
            x = data(batch, gen)
        else:
            x = data[torch.randint(0, data.shape[0], (batch,), generator=gen)]
        x_hat, c = sae(x)
        loss, _, _ = sae_loss(x, x_hat, c, alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.renorm_()
    return sae


# --------------------------------------------------------------------------
# 2. 지표
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(sae: SAE, X: torch.Tensor, bs: int = 8192) -> dict:
    """FVU = sum||x-x_hat||^2 / sum||x-mean||^2, 평균 L0 = 0이 아닌 c의 개수, 죽은 특징 = X 전체에서 한 번도 안 켜진 특징."""
    mu = X.mean(0, keepdim=True)
    sse = sst = l1 = l0 = 0.0
    ever = torch.zeros(sae.n_feat, dtype=torch.bool)
    for i in range(0, X.shape[0], bs):
        x = X[i:i + bs]
        x_hat, c = sae(x)
        sse += float(((x - x_hat) ** 2).sum())
        sst += float(((x - mu) ** 2).sum())
        l1 += float(c.sum())
        l0 += float((c > 0).sum())
        ever |= (c > 0).any(0)
    n = X.shape[0]
    return {"fvu": sse / sst, "mse_per_vec": sse / n, "l0": l0 / n, "l1": l1 / n, "dead_frac": float(1 - ever.float().mean())}


def mmcs(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A의 각 행에 대해 B의 행들과의 최대 코사인(부호 있음)을 구한 값 [len(A)].
    mmcs(G, D).mean() = 정답 특징 G가 사전 D에서 얼마나 잘 찾아졌나(정답 기준 재현율).
    mmcs(D, G).mean() = 사전 원소가 어떤 정답과 닮았나(공식 repo의 mcs_duplicates 방향).
    논문 본문에는 이 지표가 없다(Sharkey et al. 2023 중간 보고의 지표)."""
    return (F.normalize(A, dim=1) @ F.normalize(B, dim=1).T).max(dim=1).values


def recovered_frac(G: torch.Tensor, D: torch.Tensor, thresh: float = 0.9) -> float:
    """정답 특징 중 최대 코사인 > thresh 인 비율."""
    return float((mmcs(G, D) > thresh).float().mean())


# --------------------------------------------------------------------------
# 3. 정답 있는 시험대
# --------------------------------------------------------------------------


def generate_batch(batch: int, density: float, n: int, gen: torch.Generator) -> torch.Tensor:
    """30번 toy_superposition/analysis/minimal_impl.py의 generate_batch(인스턴스 1개).
    특징 n개는 각각 확률 density(=1-S)로 U[0,1], 아니면 0. 반환 [batch, n]."""
    u = torch.rand(batch, n, generator=gen)
    keep = torch.rand(batch, n, generator=gen) < density
    return u * keep


def make_truth(n: int, d: int, seed: int = 0, orthogonal: bool = False) -> torch.Tensor:
    """정답 특징 방향 [n, d], 단위 노름. orthogonal=True면 n <= d인 직교 기저(회전된 단위 행렬)."""
    g = torch.Generator().manual_seed(seed)
    if orthogonal:
        assert n <= d
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        return q[:n].contiguous()
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def synthetic_sampler(G: torch.Tensor, density: float):
    """x = sum_i a_i g_i, a는 30번의 희소 특징. callable(batch, gen) -> [batch, d]."""
    n = G.shape[0]

    def f(batch: int, gen: torch.Generator):
        return generate_batch(batch, density, n, gen) @ G
    return f


def shrinkage_slope(sae: SAE, G: torch.Tensor, density: float, thresh: float = 0.9, n_samples: int = 20000, seed: int = 123):
    """정답 특징 i와 가장 닮은 사전 원소 j(코사인 > thresh)에 대해 c_j를 a_i에 회귀한 기울기 sum(c_j a_i)/sum(a_i^2)의 평균.
    L1 수축이 없고 완벽히 복원하면 1. 짝을 못 찾은 특징은 뺀다. (평균, 짝 수)를 반환."""
    gen = torch.Generator().manual_seed(seed)
    a = generate_batch(n_samples, density, G.shape[0], gen)
    with torch.no_grad():
        c = sae.encode(a @ G)
        cs = F.normalize(G, dim=1) @ sae.dictionary().T
        best, j = cs.max(dim=1)
        ok = best > thresh
        if ok.sum() == 0:
            return float("nan"), 0
        num = (c[:, j[ok]] * a[:, ok]).sum(0)
        den = (a[:, ok] ** 2).sum(0)
        return float((num / den).mean()), int(ok.sum())


def shrinkage_theory(alpha: float) -> float:
    """간섭 없는 한 특징(a~U[0,1], 단위 방향)의 최적 코드 c = max(a - alpha/2, 0) (식 (4)에서 d/dc[(a-c)^2 + alpha c] = 0).
    c를 a에 회귀한 기울기 E[c a]/E[a^2] = 3 * int_{t}^{1} (a - t) a da, t = min(alpha/2, 1)."""
    t = min(alpha / 2, 1.0)
    return 3.0 * ((1 - t ** 3) / 3 - t * (1 - t ** 2) / 2)


# --------------------------------------------------------------------------
# 4. 29번 모델 로드와 재구성 끼우기
# --------------------------------------------------------------------------


@dataclass
class Cfg:
    vocab: int = 2048
    d: int = 128
    n_layer: int = 2
    n_head: int = 4
    d_mlp: int = 512
    ctx: int = 256
    smear: bool = False


class Attention(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.h, self.dh = cfg.n_head, cfg.d // cfg.n_head
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Parameter(torch.randn(cfg.n_head, self.dh, cfg.d) * 0.02)   # head별 W_O

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        sc = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        pattern = sc.masked_fill(~mask, float("-inf")).softmax(-1)
        head_out = torch.einsum("bhtd,hdm->bhtm", pattern @ v, self.o)     # (B,H,T,d)
        return head_out.sum(1), pattern, head_out


class Block(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.d), Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.mlp = nn.Sequential(nn.Linear(cfg.d, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d))


class TinyLM(nn.Module):
    """29번의 TinyLM(pre-LN, 학습형 절대 위치 임베딩, 2층, MLP 있음). hooks로 활성을 바꿔 끼울 수 있다."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.ctx, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lnf = nn.LayerNorm(cfg.d)
        self.unembed = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, tokens, hooks=None, cache=False):
        """hooks: {"resid_post_{l}" 또는 "mlp_out_{l}": fn(tensor)->tensor}. 해당 지점의 값을 fn의 반환값으로 바꿔 계속 진행.
        resid_post_l = 층 l의 attention과 MLP를 더한 직후 잔차(ln_f 이전), mlp_out_l = 층 l MLP가 잔차에 더하는 값(29번 artifacts와 같은 정의)."""
        hooks = hooks or {}
        B, T = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))
        c = {"patterns": [], "head_out": [], "resid_post": [], "mlp_out": []}
        for l, b in enumerate(self.blocks):
            a, p, ho = b.attn(b.ln1(x))
            x = x + a
            m = b.mlp(b.ln2(x))
            if f"mlp_out_{l}" in hooks:
                m = hooks[f"mlp_out_{l}"](m)
            x = x + m
            if f"resid_post_{l}" in hooks:
                x = hooks[f"resid_post_{l}"](x)
            if cache:
                c["patterns"].append(p)
                c["head_out"].append(ho)
                c["resid_post"].append(x)
                c["mlp_out"].append(m)
        logits = self.unembed(self.lnf(x))
        return (logits, c) if cache else logits


def load_model(art_dir) -> TinyLM:
    """29번 artifacts/model_seed0.pt를 읽는다(읽기만)."""
    ck = torch.load(Path(art_dir) / "model_seed0.pt", map_location="cpu")
    cfg = Cfg(**ck["cfg"])
    assert not cfg.smear
    model = TinyLM(cfg)
    model.load_state_dict(ck["state"])
    return model.eval()


@torch.no_grad()
def lm_loss(model: TinyLM, tokens: torch.Tensor, hooks=None, bs: int = 64) -> float:
    """다음 토큰 교차 엔트로피(nats)의 평균. 위치 0(BOS)에서 토큰 1을 맞히는 항 포함."""
    tot, cnt = 0.0, 0
    for i in range(0, tokens.shape[0], bs):
        t = tokens[i:i + bs]
        lg = model(t, hooks=hooks)
        tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), t[:, 1:].reshape(-1), reduction="sum"))
        cnt += t[:, 1:].numel()
    return tot / cnt


def splice_hook(fn):
    """위치 0(BOS)은 그대로 두고 위치 1 이후의 활성 [B, T-1, d]에만 fn을 적용하는 훅.
    SAE는 BOS 벡터(항상 같음)로 학습하지 않으므로 BOS는 건드리지 않는다."""
    def h(x):
        y = x.clone()
        y[:, 1:] = fn(x[:, 1:])
        return y
    return h


def sae_recon_fn(sae: SAE, scale: float):
    """정규화한 활성(x/scale)에서 학습한 SAE의 재구성을 원래 스케일로 되돌려 끼우는 함수."""
    def fn(x):
        shp = x.shape
        with torch.no_grad():
            xh, _ = sae(x.reshape(-1, shp[-1]) / scale)
        return (xh * scale).reshape(shp)
    return fn


def loss_recovered(L_clean: float, L_rec: float, L_abl: float) -> float:
    """1 - (L_rec - L_clean)/(L_abl - L_clean). 재구성이 완벽하면 1, 활성을 지운(제거) 것만큼 나쁘면 0."""
    return 1.0 - (L_rec - L_clean) / (L_abl - L_clean)


# --------------------------------------------------------------------------
# 5. 해석 가능성의 대용 지표 (API 없음, 논문의 자동 해석 점수를 대신하지 않는다)
# --------------------------------------------------------------------------


def top_token_stats(acts: torch.Tensor, tok: torch.Tensor, k: int = 20):
    """acts [N, F] (0 이상), tok [N] 각 행의 현재 토큰 id. 특징마다 상위 k개 activation 위치의 토큰을 보아
    top1_share = 가장 많은 토큰의 비율, entropy = 토큰 분포의 엔트로피(비트). 양의 activation이 k개 미만인 특징은 NaN.
    반환: top1_share [F], entropy [F], top_tokens(리스트: 특징별 [(토큰 id, 개수)...])."""
    N, Fn = acts.shape
    vals, idx = torch.topk(acts, k, dim=0)
    t = tok[idx].numpy()                 # [k, F]
    ok = (vals > 0).all(0).numpy()
    share = np.full(Fn, np.nan)
    ent = np.full(Fn, np.nan)
    tops = [None] * Fn
    for f in np.nonzero(ok)[0]:
        u, cnt = np.unique(t[:, f], return_counts=True)
        p = cnt / k
        share[f] = p.max()
        ent[f] = float(-(p * np.log2(p)).sum())
        o = np.argsort(-cnt)
        tops[f] = [(int(u[i]), int(cnt[i])) for i in o[:5]]
    return share, ent, tops


def baseline_acts(X: torch.Tensor, kind: str, n: int | None = None, seed: int = 0) -> torch.Tensor:
    """같은 활성 X [N, d]에서 SAE 대신 쓰는 기준선 특징의 activation [N, F] (모두 ReLU로 0 이상).
    basis : 잔차 기저 (ReLU(x_i), F=d).             논문: 기저에서 음수를 0으로.
    random: 무작위 단위 방향 n개 (ReLU(v.x)).         논문: 음수를 0으로.
    pca   : 중심화한 X의 주성분 d개 x 부호 2 (ReLU(+-v.(x-mean)), F=2d). 주성분은 양끝이 모두 후보이므로 부호를 둘 다 둔다(논문과 다른 점)."""
    d = X.shape[1]
    if kind == "basis":
        return torch.relu(X)
    if kind == "random":
        g = torch.Generator().manual_seed(seed)
        V = F.normalize(torch.randn(n or d, d, generator=g), dim=1)
        return torch.relu(X @ V.T)
    if kind == "pca":
        Xc = X - X.mean(0, keepdim=True)
        _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
        P = Xc @ Vt.T
        return torch.relu(torch.cat([P, -P], 1))
    raise ValueError(kind)


def summarize_stats(share: np.ndarray, ent: np.ndarray, max_feat: int = 128, seed: int = 0) -> dict:
    """NaN이 아닌 특징 중 최대 max_feat개를 무작위로 골라(논문은 150개) 평균과 중앙값을 낸다."""
    ok = np.nonzero(~np.isnan(share))[0]
    rng = np.random.default_rng(seed)
    if len(ok) > max_feat:
        ok = rng.choice(ok, max_feat, replace=False)
    if len(ok) == 0:
        return {"n_scored": 0}
    return {"n_scored": int(len(ok)), "top1_share_mean": float(share[ok].mean()), "top1_share_median": float(np.median(share[ok])),
            "entropy_mean": float(ent[ok].mean()), "entropy_median": float(np.median(ent[ok]))}


# --------------------------------------------------------------------------
# 6. 29번 A1: 반복 구간과 head 출력 방향
# --------------------------------------------------------------------------


def make_repeat_tokens(tokens: torch.Tensor, L: int = 64) -> torch.Tensor:
    """자연 텍스트 창의 앞 L개 토큰(위치 1..L)을 BOS 뒤에 되풀이해 길이를 ctx까지 채운 반복 시퀀스.
    복사본 번호 = (위치-1)//L. 0번이 첫 등장, 1번 이후가 반복."""
    B, T = tokens.shape
    seg = tokens[:, 1:1 + L]
    reps = (T - 1 + L - 1) // L
    body = seg.repeat(1, reps)[:, :T - 1]
    return torch.cat([tokens[:, :1], body], 1)


@torch.no_grad()
def head_dependence(model: TinyLM, sae: SAE, scale: float, toks: torch.Tensor, L: int, layer: int = 1, bs: int = 64):
    """resid_post_{layer}에 놓은 SAE에서, 반복 시퀀스 toks의 위치별로
      c        : 원래 특징 activation  [N, F]
      c_noattn : 층 `layer`의 모든 head 출력을 뺀 잔차 x - sum_h head_out_h 에서의 activation (head는 잔차에 더해지기만 하므로 이 뺄셈이 head ablation의 직접 효과)
      copy     : 복사본 번호 [N]
      mean_dir : head별 반복 구간(복사본 >= 1) 평균 출력 방향(단위) [H, d]
    를 모아 반환한다. 위치 0은 뺀다."""
    cs, cn, cp, hd = [], [], [], []
    T = toks.shape[1]
    copy = ((torch.arange(T) - 1).clamp(min=0) // L)
    for i in range(0, toks.shape[0], bs):
        _, cache = model(toks[i:i + bs], cache=True)
        x = cache["resid_post"][layer]
        ho = cache["head_out"][layer]                      # [B,H,T,d]
        xs = x[:, 1:]
        cs.append(sae.encode((xs / scale).reshape(-1, xs.shape[-1])))
        cn.append(sae.encode(((x - ho.sum(1))[:, 1:] / scale).reshape(-1, xs.shape[-1])))
        cp.append(copy[1:].repeat(x.shape[0]))
        rep = (copy >= 1)
        hd.append(ho[:, :, rep].sum((0, 2)))               # [H,d] 합
    mean_dir = F.normalize(torch.stack(hd).sum(0), dim=1)
    return torch.cat(cs), torch.cat(cn), torch.cat(cp), mean_dir
