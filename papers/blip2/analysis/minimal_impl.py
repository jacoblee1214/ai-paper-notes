"""BLIP-2(arXiv:2301.12597)의 핵심을 작게 구현한 토이. 논문 재현이 아니라 이해용이다.

구성 (논문 절과 그림)
  - Q-Former (3.1절, 그림 2): 학습되는 질의 K개, 질의와 텍스트가 공유하는 self-attention,
    질의 위치에만 걸리는 교차 attention(격층), 질의용 FFN과 텍스트용 FFN을 따로 둔 층
  - 세 self-attention 마스크 (3.2절, 그림 2 오른쪽): ITC 단일 모달 / ITG 다중 모달 인과 / ITM 양방향
  - 세 손실 (3.2절): ITC(질의별 유사도의 최댓값), ITG(다음 토큰 예측), ITM(질의별 로짓의 평균, 유사도 기반 어려운 음성)
  - 2단계 (3.3절, 그림 3): 질의 출력 -> 선형 투영 -> 고정된 디코더 앞에 소프트 프롬프트로 붙임
  - 대비용 연결기: 모든 패치 토큰을 MLP로 넘기는 투영기(LLaVA 방식), 2x2 평균 풀링 뒤 MLP
  - 직접 만든 합성 장면(8x8 칸, 색 물체 1-4개), 고정된 무작위 "이미지 인코더", 텍스트로 사전학습한 고정 디코더

공식 코드 대조(정적): salesforce/LAVIS lavis/models/blip2_models/{blip2_qformer,Qformer,blip2_opt,blip2}.py
(커밋 05cb670). 03의 D1-D8에 차이를 적었다. 데이터와 가중치는 받지 않고 모두 코드로 생성한다.
이 폴더 밖 코드는 복사하지 않았다. 구조(데이터 생성 -> 모델 -> 손실 -> 시험 가능한 순수 함수)는
저장소의 clip/analysis/minimal_impl.py를 따랐다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================ 어휘와 합성 장면

COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
NCOL = len(COLORS)
RGB = torch.tensor([[1.0, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 1.0],
                    [1.0, 0.9, 0.1], [1.0, 0.1, 1.0], [0.1, 1.0, 1.0]])
G = 8                      # 칸 수 한 변 (패치 64개)
MAX_OBJ = 4
WORDS = ["a", "object", "objects", "at", "and", "what", "color", "where", "is", "the", "how", "many",
         "there", "are", "?", "yes", "no"]
ROWS = [f"r{i}" for i in range(G)]
COLS = [f"c{i}" for i in range(G)]
NUMS = [f"n{i}" for i in range(MAX_OBJ + 1)]
VOCAB = ["<pad>", "<cls>", "<dec>"] + WORDS + COLORS + ROWS + COLS + NUMS
TOK = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
PAD, CLS, DEC = TOK["<pad>"], TOK["<cls>"], TOK["<dec>"]
ROW0, COL0, NUM0, COLOR0 = TOK["r0"], TOK["c0"], TOK["n0"], TOK["red"]

CAP_LEN = 1 + 4 * MAX_OBJ + (MAX_OBJ - 1)   # <cls> + (색 at 행 열) x 4 + "and" x 3 = 20
Q_LEN = 6
QTYPES = ["count", "exist", "where", "row_is"]


def sample_scenes(n: int, gen: torch.Generator, g: int = G, max_obj: int = MAX_OBJ):
    """물체 1..max_obj개. 색은 서로 다르고 칸도 서로 다르다. cells[n, max_obj] (-1 = 없음)."""
    nobj = torch.randint(1, max_obj + 1, (n,), generator=gen)
    cell_perm = torch.argsort(torch.rand(n, g * g, generator=gen), dim=1)[:, :max_obj]
    col_perm = torch.argsort(torch.rand(n, NCOL, generator=gen), dim=1)[:, :max_obj]
    valid = torch.arange(max_obj)[None] < nobj[:, None]
    return {"nobj": nobj, "cells": torch.where(valid, cell_perm, -1), "colors": torch.where(valid, col_perm, -1), "g": g}


def render_patches(sc) -> torch.Tensor:
    """장면 -> 패치 원료 [n, g*g, 4] = (rgb x 점유, 점유). 이미지 인코더가 보는 입력이다."""
    n, g = sc["nobj"].shape[0], sc["g"]
    out = torch.zeros(n, g * g, 4)
    for j in range(sc["cells"].shape[1]):
        ok = sc["cells"][:, j] >= 0
        idx = torch.nonzero(ok).squeeze(1)
        cell, col = sc["cells"][idx, j], sc["colors"][idx, j]
        out[idx, cell, :3] = RGB[col]
        out[idx, cell, 3] = 1.0
    return out


def caption_ids(sc, gen: torch.Generator) -> torch.Tensor:
    """"<cls> 색 at 행 열 and 색 at 행 열 ..." 물체 순서는 무작위. [n, CAP_LEN], 오른쪽 PAD."""
    n, g = sc["nobj"].shape[0], sc["g"]
    perm = torch.argsort(torch.rand(n, MAX_OBJ, generator=gen), dim=1).tolist()
    cells, colors, nobj = sc["cells"].tolist(), sc["colors"].tolist(), sc["nobj"].tolist()
    rows = []
    for b in range(n):
        toks = [CLS]
        for j in [j for j in perm[b] if j < nobj[b]]:
            if len(toks) > 1:
                toks.append(TOK["and"])
            c = cells[b][j]
            toks += [COLOR0 + colors[b][j], TOK["at"], ROW0 + c // g, COL0 + c % g]
        rows.append(toks + [PAD] * (CAP_LEN - len(toks)))
    return torch.tensor(rows, dtype=torch.long)


def make_qa(sc, qtype: str, gen: torch.Generator):
    """질문 [n, 6]과 답 [n, 2] (두 번째 칸은 PAD일 수 있음)."""
    n, g = sc["nobj"].shape[0], sc["g"]
    q = torch.zeros(n, Q_LEN, dtype=torch.long)
    a = torch.full((n, 2), PAD, dtype=torch.long)
    pick = (torch.rand(n, generator=gen) * sc["nobj"]).long()          # 존재하는 물체 하나
    ar = torch.arange(n)
    pcell, pcol = sc["cells"][ar, pick], sc["colors"][ar, pick]
    if qtype == "count":
        q[:] = torch.tensor([TOK[w] for w in ["how", "many", "objects", "are", "there", "?"]])
        a[:, 0] = NUM0 + sc["nobj"]
    elif qtype == "where":
        q[:, 0], q[:, 1], q[:, 2] = TOK["where"], TOK["is"], TOK["the"]
        q[:, 3], q[:, 4], q[:, 5] = COLOR0 + pcol, TOK["object"], TOK["?"]
        a[:, 0], a[:, 1] = ROW0 + pcell // g, COL0 + pcell % g
    elif qtype == "exist":
        ask = torch.randint(0, NCOL, (n,), generator=gen)
        use_present = torch.rand(n, generator=gen) < 0.5
        ask = torch.where(use_present, pcol, ask)
        present = (sc["colors"] == ask[:, None]).any(1)
        q[:, 0], q[:, 1], q[:, 2] = TOK["is"], TOK["there"], TOK["a"]
        q[:, 3], q[:, 4], q[:, 5] = COLOR0 + ask, TOK["object"], TOK["?"]
        a[:, 0] = torch.where(present, TOK["yes"], TOK["no"])
    elif qtype == "row_is":
        prow = pcell // g
        other = (prow + torch.randint(1, g, (n,), generator=gen)) % g          # 정답 행이 아닌 행
        same = torch.rand(n, generator=gen) < 0.5
        ask = torch.where(same, prow, other)
        q[:, 0], q[:, 1], q[:, 2] = TOK["is"], TOK["the"], COLOR0 + pcol
        q[:, 3], q[:, 4], q[:, 5] = TOK["at"], ROW0 + ask, TOK["?"]
        a[:, 0] = torch.where(ask == prow, TOK["yes"], TOK["no"])
    else:
        raise ValueError(qtype)
    return q, a


def mixed_qa(sc, gen: torch.Generator, offset: int = 0):
    """장면마다 질문 유형을 번갈아 배정한다(유형당 n/4개, offset으로 순환). 반환: 질문, 답, 유형 번호."""
    n = sc["nobj"].shape[0]
    t = (torch.arange(n) + offset) % len(QTYPES)
    q = torch.zeros(n, Q_LEN, dtype=torch.long)
    a = torch.zeros(n, 2, dtype=torch.long)
    for i, name in enumerate(QTYPES):
        qq, aa = make_qa(sc, name, gen)
        q[t == i], a[t == i] = qq[t == i], aa[t == i]
    return q, a, t


# ================================================================ 고정된 "이미지 인코더"

class FrozenImageEncoder(nn.Module):
    """무작위로 초기화한 뒤 고정하는 패치 인코더: MLP(rgb, 점유) + 칸 위치 임베딩.
    정보는 보존하지만 의미 있게 정렬되어 있지 않다. 논문의 고정된 ViT 자리다(모든 매개변수 requires_grad=False)."""

    def __init__(self, d_img: int = 32, hidden: int = 64, max_g: int = 12, seed: int = 1234):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(torch.randn(4, hidden, generator=g) / 2.0, requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(hidden, d_img, generator=g) / math.sqrt(hidden), requires_grad=False)
        self.pos = nn.Parameter(torch.randn(max_g, max_g, d_img, generator=g) * 0.5, requires_grad=False)
        self.d_img = d_img

    def forward(self, raw: torch.Tensor, g: int) -> torch.Tensor:
        x = torch.tanh(raw @ self.w1) @ self.w2
        return x + self.pos[:g, :g].reshape(g * g, -1)


# ================================================================ Q-Former

@dataclass
class QFConfig:
    d: int = 64
    heads: int = 4
    layers: int = 4
    ffn: int = 128
    k_queries: int = 8
    d_img: int = 32
    vocab: int = V
    max_text: int = CAP_LEN
    cross_freq: int = 2        # 층 번호 % cross_freq == 0 인 층에만 교차 attention (LAVIS: layer_num % 2 == 0)
    embed_dim: int = 32        # ITC 투영 차원 (LAVIS는 256)


def build_mask(mode: str, K: int, T: int, text_pad: torch.Tensor | None = None) -> torch.Tensor:
    """self-attention 허용 행렬 [B 또는 1, 1, K+T, K+T] (True = 볼 수 있음). 행 = 보는 쪽, 열 = 보이는 쪽.
    순서는 [질의 K개; 텍스트 T개]. text_pad[B, T]가 True인 텍스트 열은 항상 가린다.
      itc: 질의는 질의끼리, 텍스트는 텍스트끼리(양방향). 서로 못 봄.
      itg: 질의는 질의끼리. 텍스트는 모든 질의 + 앞쪽 텍스트(자기 자신 포함).
      itm: 모두가 모두를 봄.
    """
    L = K + T
    m = torch.zeros(L, L, dtype=torch.bool)
    m[:K, :K] = True
    if mode == "itc":
        m[K:, K:] = True
    elif mode == "itg":
        m[K:, :K] = True
        m[K:, K:] = torch.tril(torch.ones(T, T, dtype=torch.bool))
    elif mode == "itm":
        m[:] = True
    elif mode != "query":
        raise ValueError(mode)
    m = m[None, None]
    if text_pad is not None:
        keep = torch.ones(text_pad.shape[0], L, dtype=torch.bool)
        keep[:, K:] = ~text_pad
        m = m & keep[:, None, None, :]
    return m


class MHA(nn.Module):
    def __init__(self, d, heads, d_kv=None):
        super().__init__()
        self.h, d_kv = heads, d_kv or d
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d_kv, d), nn.Linear(d_kv, d), nn.Linear(d, d)

    def forward(self, x, kv=None, mask=None):
        kv = x if kv is None else kv
        B, L, d = x.shape
        sp = lambda t: t.view(B, -1, self.h, d // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(sp(self.q(x)), sp(self.k(kv)), sp(self.v(kv)), attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, L, d))


def ffn(d, f):
    return nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))


class QFormerBlock(nn.Module):
    """self-attention(질의·텍스트 공유) -> [질의 위치만] 교차 attention(고정 이미지 특징) -> FFN.
    FFN은 질의용과 텍스트용이 따로다(LAVIS Qformer.py BertLayer의 intermediate_query/output_query)."""

    def __init__(self, cfg: QFConfig, has_cross: bool):
        super().__init__()
        d = cfg.d
        self.ln1, self.attn = nn.LayerNorm(d), MHA(d, cfg.heads)
        self.has_cross = has_cross
        if has_cross:
            self.lnx, self.cross = nn.LayerNorm(d), MHA(d, cfg.heads, d_kv=cfg.d_img)
        self.ln2q, self.ffq = nn.LayerNorm(d), ffn(d, cfg.ffn)
        self.ln2t, self.fft = nn.LayerNorm(d), ffn(d, cfg.ffn)

    def forward(self, h, mask, img, K):
        h = h + self.attn(self.ln1(h), mask=mask)
        q, t = h[:, :K], h[:, K:]
        if K:
            if self.has_cross:
                q = q + self.cross(self.lnx(q), kv=img)
            q = q + self.ffq(self.ln2q(q))
        if t.shape[1]:
            t = t + self.fft(self.ln2t(t))
        return torch.cat([q, t], dim=1)


class QFormer(nn.Module):
    def __init__(self, cfg: QFConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.query_tokens = nn.Parameter(torch.randn(cfg.k_queries, d) * 0.02)
        self.tok = nn.Embedding(cfg.vocab, d)
        self.pos = nn.Embedding(cfg.max_text, d)
        self.emb_ln = nn.LayerNorm(d)
        self.ln_img = nn.LayerNorm(cfg.d_img)     # LAVIS의 ln_vision: 고정 인코더 출력 뒤의 학습되는 LayerNorm
        self.blocks = nn.ModuleList([QFormerBlock(cfg, i % cfg.cross_freq == 0) for i in range(cfg.layers)])
        self.out_ln = nn.LayerNorm(d)
        self.vision_proj, self.text_proj = nn.Linear(d, cfg.embed_dim), nn.Linear(d, cfg.embed_dim)
        self.itm_head = nn.Linear(d, 2)
        self.lm_head = nn.Linear(d, cfg.vocab)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def cross_layers(self):
        return [i for i, b in enumerate(self.blocks) if b.has_cross]

    def forward(self, img_feats, text_ids=None, mode="query", return_layers=False):
        """img_feats [B, P, d_img] (P는 아무 값). mode: query(텍스트 없음) / itc / itg / itm / text(질의 없음).
        반환 (q_out [B, K, d] 또는 None, t_out [B, T, d] 또는 None)."""
        B = img_feats.shape[0]
        K = 0 if mode == "text" else self.cfg.k_queries
        parts = []
        if K:
            parts.append(self.query_tokens[None].expand(B, -1, -1))
        T, text_pad = 0, None
        if mode != "query":
            T = text_ids.shape[1]
            text_pad = text_ids == PAD
            parts.append(self.tok(text_ids) + self.pos(torch.arange(T))[None])   # 텍스트 위치는 0부터
        h = self.emb_ln(torch.cat(parts, dim=1))
        mask = build_mask({"text": "itc"}.get(mode, mode), K, T, text_pad)
        img = self.ln_img(img_feats)
        layers = []
        for blk in self.blocks:
            h = blk(h, mask, img, K)
            layers.append(h)
        h = self.out_ln(h)
        q_out = h[:, :K] if K else None
        t_out = h[:, K:] if T else None
        return (q_out, t_out, layers) if return_layers else (q_out, t_out)


# ================================================================ 1단계 손실

def itc_similarity(q_out: torch.Tensor, t_cls: torch.Tensor, vision_proj: nn.Module, text_proj: nn.Module):
    """sim_i2t[i, j] = max_k <z_ik, t_j> (정규화된 투영 뒤). 논문 3.2절: 질의별 유사도 중 최댓값.
    sim_q2t [B, B, K]도 함께 돌려준다."""
    zq = F.normalize(vision_proj(q_out), dim=-1)           # [B, K, E]
    t = F.normalize(text_proj(t_cls), dim=-1)              # [B, E]
    sim_q2t = torch.einsum("ike,je->ijk", zq, t)
    return sim_q2t.max(-1).values, sim_q2t


def itc_loss(sim_i2t: torch.Tensor, temp: torch.Tensor, label_smoothing: float = 0.1):
    """대칭 교차 엔트로피. sim_t2i = sim_i2t.T (같은 점곱의 질의별 최댓값이라 같은 행렬의 전치)."""
    logits = sim_i2t / temp
    y = torch.arange(logits.shape[0])
    return 0.5 * (F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                  + F.cross_entropy(logits.t(), y, label_smoothing=label_smoothing))


def sample_hard_negatives(sim_i2t: torch.Tensor, temp: float, gen: torch.Generator | None = None):
    """BLIP·BLIP-2의 어려운 음성: 유사도 softmax 분포에서 (대각선 제외) 한 개를 뽑는다. 가장 비슷한 것을 고르는 argmax가 아니다.
    반환: 이미지마다 음성 텍스트 인덱스 neg_t[B], 텍스트마다 음성 이미지 인덱스 neg_i[B]."""
    B = sim_i2t.shape[0]
    with torch.no_grad():
        eye = torch.eye(B, dtype=torch.bool)
        w_i2t = F.softmax((sim_i2t / temp).masked_fill(eye, -10000.0), dim=1)
        w_t2i = F.softmax((sim_i2t.t() / temp).masked_fill(eye, -10000.0), dim=1)
        neg_t = torch.multinomial(w_i2t, 1, generator=gen).squeeze(1)
        neg_i = torch.multinomial(w_t2i, 1, generator=gen).squeeze(1)
    return neg_t, neg_i


def stage1_losses(model: QFormer, img_feats: torch.Tensor, cap: torch.Tensor, gen: torch.Generator | None = None):
    """ITC + ITM + ITG. 반환 dict(loss_itc, loss_itm, loss_itg)."""
    B = cap.shape[0]
    # ITC: 단일 모달 마스크, 한 번의 순전파로 질의와 텍스트를 함께 (서로 못 보므로 따로 돌린 것과 같다)
    q_out, t_out = model(img_feats, cap, "itc")
    sim_i2t, _ = itc_similarity(q_out, t_out[:, 0], model.vision_proj, model.text_proj)
    temp = model.temp.clamp(min=0.01)            # LAVIS에는 clamp가 없다 (03 D4)
    loss_itc = itc_loss(sim_i2t, temp)

    # ITM: 양성 B개 + 음성 2B개 (LAVIS 순서: 텍스트 [t, t, t_neg], 이미지 [img, img_neg, img])
    neg_t, neg_i = sample_hard_negatives(sim_i2t, float(temp.detach()), gen)
    text_all = torch.cat([cap, cap, cap[neg_t]])
    img_all = torch.cat([img_feats, img_feats[neg_i], img_feats])
    q_itm, _ = model(img_all, text_all, "itm")
    logits = model.itm_head(q_itm).mean(dim=1)   # 질의별 2-클래스 로짓의 평균
    labels = torch.cat([torch.ones(B, dtype=torch.long), torch.zeros(2 * B, dtype=torch.long)])
    loss_itm = F.cross_entropy(logits, labels)

    # ITG: 첫 토큰 <cls> -> <dec>, 다중 모달 인과 마스크
    dec = cap.clone()
    dec[:, 0] = DEC
    _, t_g = model(img_feats, dec, "itg")
    pred = model.lm_head(t_g[:, :-1])
    tgt = dec[:, 1:].masked_fill(dec[:, 1:] == PAD, -100)
    loss_itg = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), ignore_index=-100)
    return {"loss_itc": loss_itc, "loss_itm": loss_itm, "loss_itg": loss_itg}


# ================================================================ 고정된 디코더 ("LLM")

def rope(x: torch.Tensor) -> torch.Tensor:
    """x [B, H, L, hd]에 회전 위치 인코딩. 위치가 상대적이라 앞에 소프트 프롬프트가 몇 개 붙어도 같은 디코더를 쓸 수 있다."""
    B, H, L, hd = x.shape
    half = hd // 2
    freq = 1.0 / (10000 ** (torch.arange(half) / half))
    ang = torch.arange(L)[:, None] * freq[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class DecBlock(nn.Module):
    def __init__(self, d, heads, f):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = ffn(d, f)

    def forward(self, x, mask):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=mask)
        x = x + self.o(y.transpose(1, 2).reshape(B, L, d))
        return x + self.ff(self.ln2(x))


class TinyDecoder(nn.Module):
    """작은 인과 Transformer 언어 모델(RoPE). 텍스트(캡션 + 질문 + 답)로만 사전학습한 뒤 고정한다."""

    def __init__(self, d: int = 96, heads: int = 4, layers: int = 3, f: int = 192):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([DecBlock(d, heads, f) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)

    def forward(self, text_ids, prefix=None, key_pad=None):
        """text_ids [B, T], prefix [B, Kp, d] (소프트 프롬프트, 없을 수 있음) -> 로짓 [B, Kp+T, V]."""
        x = self.tok(text_ids)
        kp = 0
        if prefix is not None:
            kp = prefix.shape[1]
            x = torch.cat([prefix, x], dim=1)
        L = x.shape[1]
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool))[None, None]
        if key_pad is not None:
            keep = torch.ones(x.shape[0], L, dtype=torch.bool)
            keep[:, kp:] = ~key_pad
            mask = mask & keep[:, None, None, :]
            mask = mask | torch.eye(L, dtype=torch.bool)[None, None]    # 패딩 행도 자기 자신은 봄(NaN 방지)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


def freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    return module.eval()


@torch.no_grad()
def oracle_caption_acc(dec: TinyDecoder, gen: torch.Generator, n: int = 500):
    """이미지 대신 정답 캡션을 글자로 주었을 때의 정확 일치율(유형별 전체, 물체 수별). 연결기가 도달할 수 있는 상한의 어림이다."""
    out = {}
    for name in QTYPES:
        sc = sample_scenes(n, gen)
        cap = caption_ids(sc, gen)[:, 1:]
        q, a = make_qa(sc, name, gen)
        ids = torch.cat([cap, q], 1)
        a0 = dec(ids, key_pad=ids == PAD)[:, -1].argmax(-1)
        ids2 = torch.cat([ids, a0[:, None]], 1)
        a1 = dec(ids2, key_pad=ids2 == PAD)[:, -1].argmax(-1)
        ok = ((a == PAD) | (torch.stack([a0, a1], 1) == a)).all(1).float()
        out[name] = {"all": round(ok.mean().item(), 3), "by_nobj": [round(ok[sc["nobj"] == k].mean().item(), 3) for k in range(1, MAX_OBJ + 1)]}
    return out


def pretrain_decoder(dec: TinyDecoder, gen: torch.Generator, steps: int = 6000, batch: int = 128, lr: float = 2e-3,
                     check_every: int = 500, target: float = 0.97, tail: int = 600):
    """텍스트만으로 사전학습: [캡션(장면 설명) + 질문 + 답] x (질문 3개). 이미지는 보지 않는다. 캡션을 읽고 답하는 법을 배운다.
    lr은 일정하게 두다가 모든 유형의 오라클 정확도가 target을 넘으면(찾기 학습은 갑자기 일어나는 단계 전이라 시점이 불규칙하다)
    tail 스텝 동안 선형으로 0까지 줄이고 멈춘다. steps는 상한이다."""
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=0.01)
    done_at = None
    chk = torch.Generator().manual_seed(4242)
    for s in range(steps):
        if done_at is None and s > 0 and s % check_every == 0:
            dec.eval()
            acc = oracle_caption_acc(dec, chk)
            dec.train()
            if min(v["all"] for v in acc.values()) >= target:
                done_at = s
        if done_at is not None:
            frac = 1.0 - (s - done_at) / tail
            if frac <= 0:
                break
            for g_ in opt.param_groups:
                g_["lr"] = lr * frac
        else:
            for g_ in opt.param_groups:
                g_["lr"] = lr
        sc = sample_scenes(batch, gen)
        cap = caption_ids(sc, gen)
        cap[:, 0] = PAD                       # 소프트 프롬프트 자리에는 <cls>가 없다
        off = int(torch.randint(0, 4, (1,), generator=gen))
        qa = [torch.cat(mixed_qa(sc, gen, off + i)[:2], dim=1) for i in range(3)]   # 한 캡션에 질문 3개(유형이 다름)
        ids = torch.cat([cap] + qa, dim=1)
        pad = ids == PAD
        logits = dec(ids[:, :-1], key_pad=pad[:, :-1])
        tgt = ids[:, 1:].clone()
        tgt[tgt == PAD] = -100
        ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100, reduction="none").view_as(tgt)
        w = torch.ones_like(ce)
        w[:, CAP_LEN - 1 :] = 5.0                  # 질문·답 구간에 가중치 5, 캡션의 언어 모델링도 함께 학습
        loss = (ce * w * (tgt != -100)).sum() / ((tgt != -100) * w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    dec.stopped_at = done_at if done_at is not None else steps
    return freeze(dec)


# ================================================================ 연결기와 2단계 VLM

class QFormerConnector(nn.Module):
    """Q-Former 질의 출력 K개 -> 선형 투영 -> 소프트 프롬프트 (논문 3.3절, 그림 3의 Fully Connected)."""

    def __init__(self, qf: QFormer, d_lm: int):
        super().__init__()
        self.qf, self.proj = qf, nn.Linear(qf.cfg.d, d_lm)

    def forward(self, feats, g=None):
        q_out, _ = self.qf(feats, None, "query")
        return self.proj(q_out)


class MLPConnector(nn.Module):
    """LLaVA-1.5식: 모든 패치 토큰을 2층 MLP로 투영해 그대로 넘긴다(토큰 수 = 패치 수)."""

    def __init__(self, d_img: int, d_lm: int, pool: int = 1, hidden: int | None = None):
        super().__init__()
        self.pool = pool
        h = hidden or d_lm
        self.mlp = nn.Sequential(nn.Linear(d_img, h), nn.GELU(), nn.Linear(h, d_lm))

    def forward(self, feats, g=G):
        if self.pool > 1:        # 2x2 평균 풀링: 토큰 수를 1/4로 (공간 구조는 유지하되 칸을 합침)
            B, P, d = feats.shape
            x = feats.view(B, g, g, d).permute(0, 3, 1, 2)
            feats = F.avg_pool2d(x, self.pool).flatten(2).transpose(1, 2)
        return self.mlp(feats)


class VLM(nn.Module):
    """고정 이미지 인코더 + 학습되는 연결기 + 고정 디코더. 학습되는 것은 연결기뿐이다."""

    def __init__(self, enc: FrozenImageEncoder, connector: nn.Module, dec: TinyDecoder):
        super().__init__()
        self.enc, self.connector, self.dec = enc, connector, dec

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        self.dec.eval()
        return self

    def prefix(self, raw, g=G):
        with torch.no_grad():
            feats = self.enc(raw, g)
        return self.connector(feats, g)

    def loss(self, raw, q, a, g=G):
        pre = self.prefix(raw, g)
        text = torch.cat([q, a], dim=1)                                  # [B, 8]
        logits = self.dec(text[:, :-1], prefix=pre)
        kp = pre.shape[1]
        pred = logits[:, kp + Q_LEN - 1 :]                               # 답 위치 예측 [B, 2, V]
        tgt = a.masked_fill(a == PAD, -100)
        return F.cross_entropy(pred.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def answer(self, raw, q, g=G):
        """탐욕 디코딩으로 답 토큰 2개 [B, 2]."""
        pre = self.prefix(raw, g)
        kp = pre.shape[1]
        a0 = self.dec(q, prefix=pre)[:, kp + Q_LEN - 1].argmax(-1)
        a1 = self.dec(torch.cat([q, a0[:, None]], 1), prefix=pre)[:, kp + Q_LEN].argmax(-1)
        return torch.stack([a0, a1], 1)


def qa_accuracy(vlm: VLM, gen, per_type: int = 400):
    """유형별 정확 일치율 (답 토큰이 모두 맞아야 함)."""
    out = {}
    for name in QTYPES:
        scn = sample_scenes(per_type, gen)
        q, a = make_qa(scn, name, gen)
        pred = vlm.answer(render_patches(scn), q)
        ok = (a == PAD) | (pred == a)
        out[name] = ok.all(1).float().mean().item()
    return out


def train_connector(vlm: VLM, gen: torch.Generator, steps: int = 2500, batch: int = 128, lr: float = 1e-3):
    params = [p for p in vlm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    vlm.train()
    losses = []
    for s in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = lr * (min(1.0, (s + 1) / 100)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        sc = sample_scenes(batch, gen)
        q, a, _ = mixed_qa(sc, gen)
        loss = vlm.loss(render_patches(sc), q, a)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def train_stage1(qf: QFormer, enc: FrozenImageEncoder, gen: torch.Generator, steps: int = 1500, batch: int = 128,
                 lr: float = 1e-3, log_every: int = 0):
    opt = torch.optim.AdamW(qf.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    for s in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = lr * (min(1.0, (s + 1) / 100)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        sc = sample_scenes(batch, gen)
        cap = caption_ids(sc, gen)
        with torch.no_grad():
            feats = enc(render_patches(sc), G)
        ls = stage1_losses(qf, feats, cap, gen)
        loss = ls["loss_itc"] + ls["loss_itm"] + ls["loss_itg"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qf.parameters(), 1.0)
        opt.step()
        if log_every and (s % log_every == 0 or s == steps - 1):
            hist.append({"step": s, **{k: round(v.item(), 4) for k, v in ls.items()}})
    return hist


# ================================================================ 파라미터 수 산술 (논문 188M 검산용)

def qformer_param_count(V=30523, P=512, d=768, f=3072, L=12, dv=1408, freq=2, K=32, E=256,
                        query_ffn=True, stage=1):
    """LAVIS Qformer.py(BertLMHeadModel + 층별 교차 attention + 질의용 FFN) 구조의 매개변수 수.
    stage=1: 어휘 임베딩, 위치 임베딩, LM 헤드, ITC 투영, ITM 헤드 포함. stage=2: 텍스트 쪽(어휘·위치 임베딩, 텍스트 FFN,
    LM 헤드, ITC·ITM 헤드)을 제거하고 질의 경로만 남긴 뒤 LLM 입력 차원 E로 가는 선형 투영(out)을 더한다.
    BERT 구조: 후-LN, 선형 층마다 편향. V=30522+[DEC]."""
    attn = 4 * (d * d + d) + 2 * d                       # q, k, v, o + 출력 LayerNorm
    ffn_ = 2 * d * f + f + d + 2 * d                     # intermediate + output(+LayerNorm)
    cross = (d * d + d) + 2 * (dv * d + d) + (d * d + d) + 2 * d
    n_cross = sum(1 for l in range(L) if l % freq == 0)
    layers = L * (attn + ffn_ + (ffn_ if query_ffn else 0)) + n_cross * cross
    if stage == 1:
        emb = V * d + P * d + 2 * d
        head = d * d + d + 2 * d + V                      # BertLMHead: transform + 묶인 가중치의 편향
        extra = K * d + 2 * (d * E + E) + d * 2 + 2 + 1      # 질의, ITC 투영 2개, ITM 헤드, 온도
        return emb + layers + head + extra
    out = E                                               # stage 2에서는 E가 LLM 은닉 차원
    layers = L * (attn + (ffn_ if query_ffn else 0)) + n_cross * cross
    return layers + 2 * d + K * d + d * out + out         # 임베딩 LayerNorm만 남음


def toy_param_count(cfg: QFConfig) -> int:
    """이 파일의 QFormer 모듈 구조(전-LN, 편향 포함)에 대한 닫힌 식. 실제 모듈 매개변수 수와 일치해야 한다(시험)."""
    d, f = cfg.d, cfg.ffn
    ln = 2 * d
    lin = lambda i, o: i * o + o
    n = cfg.k_queries * d + cfg.vocab * d + cfg.max_text * d + ln + 2 * cfg.d_img
    for l in range(cfg.layers):
        n += ln + 4 * lin(d, d)                                                   # ln1 + self-attn
        if l % cfg.cross_freq == 0:
            n += ln + lin(d, d) + 2 * lin(cfg.d_img, d) + lin(d, d)             # lnx + cross-attn
        n += 2 * (ln + lin(d, f) + lin(f, d))                                      # 질의 FFN, 텍스트 FFN
    n += ln + 2 * lin(d, cfg.embed_dim) + lin(d, 2) + lin(d, cfg.vocab) + 1        # out_ln, 투영 2개, itm, lm, temp
    return n
