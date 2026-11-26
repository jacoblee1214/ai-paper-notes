"""합성 도형 토이의 데이터와 작은 이미지-텍스트 모델. 논문 재현이 아니라 이해용이다.

출처: 이 저장소의 clip/analysis/minimal_impl.py (42번 CLIP 분석)에서 어휘·데이터 생성·ViT·텍스트 Transformer 부분을
그대로 복사했다(clip/ 파일은 수정하지 않았다). 바뀐 점은 TinyDual의 로짓 매개변수뿐이다:
  - logit_scale(t' = log t)의 초기값과 상한을 인자로 받고, 선택적으로 학습되는 편향 b를 둔다 (SigLIP 논문 3.2절).
  - 복사한 코드의 CLIP 전용 부분(hard negative, 결합 평가, 모달리티 간격)은 쓰지 않아서 뺐다.
이 파일 아래의 손실과 학습 루프는 siglip/analysis/toy_train.py에 있다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- 어휘와 데이터

COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "white"]
SHAPES = ["square", "circle", "triangle", "cross", "diamond", "ring", "hbar", "vbar"]
NC, NS = len(COLORS), len(SHAPES)
WORDS = ["a", "photo", "of", "drawing", "and"]
# EOT가 가장 큰 id여야 한다: 공식 코드가 text.argmax(-1)로 EOT 위치를 찾기 때문이다 (model.py encode_text)
PAD, SOT = 0, 1
VOCAB = ["<pad>", "<sot>"] + WORDS + COLORS + SHAPES + ["<eot>"]
TOK = {w: i for i, w in enumerate(VOCAB)}
EOT = TOK["<eot>"]
CONTEXT = 12  # SOT + 최대 10 토큰 + EOT

# 템플릿: 앞에 붙는 문맥 토큰. 학습 분포에서 T0(맨 이름)은 드물게 둔다(논문 3.1.4절의 "단어 하나짜리 텍스트는 드물다"를 흉내)
TEMPLATES = {"bare": [], "photo": ["a", "photo", "of"], "drawing": ["a", "drawing", "of"]}
TEMPLATE_PROBS = {"bare": 0.05, "photo": 0.5, "drawing": 0.45}

# 학습에서 빼는 (색, 모양) 조합. 8 x 8 = 64 조합 중 대각선 8개. 이름은 텍스트에 나오고 이미지는 학습에서 본 적 없다.
HELD_OUT = [(i, i) for i in range(8)]

CANVAS = 16
OBJ = 7
_SQ = torch.zeros(OBJ, OBJ)
_SQ[1:6, 1:6] = 1
_yy, _xx = torch.meshgrid(torch.arange(OBJ), torch.arange(OBJ), indexing="ij")
_CIRCLE = (((_yy - 3) ** 2 + (_xx - 3) ** 2) <= 9).float()
_TRI = ((_xx - 3).abs() <= (_yy.float() * 0.55)).float()
_TRI[0] = 0
_CROSS = ((_yy == 3) | (_xx == 3)).float()
_DIAMOND = ((_yy - 3).abs() + (_xx - 3).abs() <= 3).float()
_RING = _SQ.clone()
_RING[2:5, 2:5] = 0
_HBAR = torch.zeros(OBJ, OBJ)
_HBAR[2:5, :] = 1
_VBAR = _HBAR.T.clone()
MASKS = torch.stack([_SQ, _CIRCLE, _TRI, _CROSS, _DIAMOND, _RING, _HBAR, _VBAR])  # [8, 7, 7]
RGB = torch.tensor([[1.0, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 1.0], [1.0, 0.9, 0.1],
                    [1.0, 0.1, 1.0], [0.1, 1.0, 1.0], [1.0, 0.5, 0.0], [1.0, 1.0, 1.0]])


def held_out_mask(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m = torch.zeros_like(c, dtype=torch.bool)
    for hc, hs in HELD_OUT:
        m |= (c == hc) & (s == hs)
    return m


def sample_seen_combos(n: int, gen: torch.Generator, allow_held_out: bool = False):
    """(색, 모양)을 n개 뽑는다. 기본은 학습에서 빼는 조합을 제외한다."""
    c = torch.randint(0, NC, (n,), generator=gen)
    s = torch.randint(0, NS, (n,), generator=gen)
    if not allow_held_out:
        bad = held_out_mask(c, s)
        while bad.any():
            k = int(bad.sum())
            c[bad] = torch.randint(0, NC, (k,), generator=gen)
            s[bad] = torch.randint(0, NS, (k,), generator=gen)
            bad = held_out_mask(c, s)
    return c, s


def render(objs: list[tuple[torch.Tensor, ...]], gen: torch.Generator, noise: float = 0.05) -> torch.Tensor:
    """objs = [(color[B], shape[B], cx[B], cy[B]), ...]. 겹치지 않는 물체를 16x16 RGB에 그린다."""
    B = objs[0][0].shape[0]
    canvas = torch.zeros(B, 3, CANVAS * CANVAS)
    ar = torch.arange(OBJ)
    for color, shape, cx, cy in objs:
        y0 = (cy - 3)[:, None, None] + ar[None, :, None]
        x0 = (cx - 3)[:, None, None] + ar[None, None, :]
        flat = (y0 * CANVAS + x0).reshape(B, -1)  # [B, 49]
        mask = MASKS[shape].reshape(B, -1)
        bright = 0.75 + 0.25 * torch.rand(B, 1, 1, generator=gen)
        val = RGB[color][:, :, None] * mask[:, None, :] * bright
        canvas.scatter_(2, flat[:, None, :].expand(B, 3, flat.shape[1]), val)
    img = canvas.reshape(B, 3, CANVAS, CANVAS)
    return (img + noise * torch.randn(img.shape, generator=gen)).clamp(0, 1)


def _tokens(words: list[list[str]]) -> torch.Tensor:
    out = torch.zeros(len(words), CONTEXT, dtype=torch.long)
    for i, w in enumerate(words):
        ids = [SOT] + [TOK[x] for x in w] + [EOT]
        out[i, : len(ids)] = torch.tensor(ids)
    return out


def caption_words(template: str, objs: list[tuple[int, int]]) -> list[str]:
    body: list[str] = []
    for k, (c, s) in enumerate(objs):
        if k:
            body.append("and")
        body += [COLORS[c], SHAPES[s]]
    return TEMPLATES[template] + body


def make_batch(n: int, gen: torch.Generator, p_two: float = 0.5, hard_neg: bool = False):
    """학습 배치. 한 장면에 물체 1개(위치 자유) 또는 2개(왼쪽, 오른쪽). 캡션은 왼쪽 물체부터 쓴다.
    hard_neg=True이면 두 물체의 모양을 맞바꾼 캡션(같은 단어 가방)도 함께 돌려준다.
    반환: img[n,3,H,W], text[n,L], (swap_text[m,L], swap_idx[m]) 또는 None"""
    two = torch.rand(n, generator=gen) < p_two
    c1, s1 = sample_seen_combos(n, gen)
    c2, s2 = sample_seen_combos(n, gen)
    # 한 물체 장면: 가운데 영역 어디든
    cx1 = torch.where(two, torch.randint(3, 5, (n,), generator=gen), torch.randint(4, 12, (n,), generator=gen))
    cy1 = torch.randint(3, 13, (n,), generator=gen)
    cx2 = torch.randint(11, 13, (n,), generator=gen)
    cy2 = torch.randint(3, 13, (n,), generator=gen)
    img1 = render([(c1, s1, cx1, cy1)], gen)
    img2 = render([(c1, s1, cx1, cy1), (c2, s2, cx2, cy2)], gen)
    img = torch.where(two[:, None, None, None], img2, img1)
    tmpl = list(TEMPLATE_PROBS)
    pick = torch.multinomial(torch.tensor(list(TEMPLATE_PROBS.values())), n, replacement=True, generator=gen)
    words, swap_words, swap_idx = [], [], []
    for i in range(n):
        t = tmpl[int(pick[i])]
        if bool(two[i]):
            o = [(int(c1[i]), int(s1[i])), (int(c2[i]), int(s2[i]))]
            words.append(caption_words(t, o))
            if hard_neg and o[0][0] != o[1][0] and o[0][1] != o[1][1]:
                swap_words.append(caption_words(t, [(o[0][0], o[1][1]), (o[1][0], o[0][1])]))
                swap_idx.append(i)
        else:
            words.append(caption_words(t, [(int(c1[i]), int(s1[i]))]))
    swap = (_tokens(swap_words), torch.tensor(swap_idx)) if hard_neg and swap_words else None
    return img, _tokens(words), swap


def make_single_eval(n: int, gen: torch.Generator, held_out: bool):
    """제로샷 평가용 한 물체 이미지와 클래스 번호(색 * NS + 모양)."""
    if held_out:
        k = torch.randint(0, len(HELD_OUT), (n,), generator=gen)
        c, s = torch.tensor([HELD_OUT[i][0] for i in k]), torch.tensor([HELD_OUT[i][1] for i in k])
    else:
        c, s = sample_seen_combos(n, gen)
    cx = torch.randint(4, 12, (n,), generator=gen)
    cy = torch.randint(3, 13, (n,), generator=gen)
    return render([(c, s, cx, cy)], gen), c * NS + s


def make_two_eval(n: int, gen: torch.Generator):
    """결합 평가: 본 조합만 쓰고 c1 != c2, s1 != s2 인 두 물체 장면. 반환 img, (c1,s1,c2,s2)."""
    out = []
    while sum(x[0].shape[0] for x in out) < n:
        c1, s1 = sample_seen_combos(n, gen)
        c2, s2 = sample_seen_combos(n, gen)
        ok = (c1 != c2) & (s1 != s2)
        out.append((c1[ok], s1[ok], c2[ok], s2[ok]))
    c1, s1, c2, s2 = [torch.cat([x[k] for x in out])[:n] for k in range(4)]
    n = c1.shape[0]
    cx1 = torch.randint(3, 5, (n,), generator=gen)
    cx2 = torch.randint(11, 13, (n,), generator=gen)
    cy1 = torch.randint(3, 13, (n,), generator=gen)
    cy2 = torch.randint(3, 13, (n,), generator=gen)
    return render([(c1, s1, cx1, cy1), (c2, s2, cx2, cy2)], gen), (c1, s1, c2, s2)



# ---------------------------------------------------------------- 모델

LOGIT_SCALE_INIT = math.log(1 / 0.07)  # 논문 2.5절: 0.07에 해당하는 값에서 시작
LOGIT_SCALE_MAX = math.log(100.0)  # 논문 2.5절: 로짓 배율이 100을 넘지 않게 자른다 (log 도메인에서 ln 100 = 4.6052)


@dataclass
class Cfg:
    width: int = 64
    layers: int = 2
    heads: int = 4
    embed: int = 32
    patch: int = 4


class Block(nn.Module):
    """pre-LN Transformer 블록."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class ImageEncoder(nn.Module):
    """ViT: 패치 임베딩 + CLS + 위치 임베딩, CLS를 LayerNorm 후 선형 투영 (공식 VisionTransformer와 같은 순서)."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        n = (CANVAS // cfg.patch) ** 2
        self.patchify = nn.Conv2d(3, cfg.width, cfg.patch, cfg.patch, bias=False)
        self.cls = nn.Parameter(torch.randn(cfg.width) * cfg.width**-0.5)
        self.pos = nn.Parameter(torch.randn(n + 1, cfg.width) * cfg.width**-0.5)
        self.ln_pre = nn.LayerNorm(cfg.width)
        self.blocks = nn.ModuleList([Block(cfg.width, cfg.heads) for _ in range(cfg.layers)])
        self.ln_post = nn.LayerNorm(cfg.width)
        self.proj = nn.Parameter(torch.randn(cfg.width, cfg.embed) * cfg.width**-0.5)

    def forward(self, img):
        x = self.patchify(img).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], 1, -1), x], 1) + self.pos
        x = self.ln_pre(x)
        for b in self.blocks:
            x = b(x)
        return self.ln_post(x[:, 0]) @ self.proj


class TextEncoder(nn.Module):
    """causal Transformer. EOT 위치(토큰 id 최댓값)의 활성을 ln_final 후 선형 투영한다 (공식 encode_text)."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), cfg.width)
        self.pos = nn.Parameter(torch.randn(CONTEXT, cfg.width) * 0.01)
        self.blocks = nn.ModuleList([Block(cfg.width, cfg.heads) for _ in range(cfg.layers)])
        self.ln_final = nn.LayerNorm(cfg.width)
        self.proj = nn.Parameter(torch.randn(cfg.width, cfg.embed) * cfg.width**-0.5)
        self.register_buffer("mask", torch.triu(torch.full((CONTEXT, CONTEXT), float("-inf")), 1), persistent=False)

    def forward(self, text):
        x = self.tok(text) + self.pos
        for b in self.blocks:
            x = b(x, self.mask)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.proj


class TinyDual(nn.Module):
    """두 타워 + 학습되는 log 온도 t'와 (선택) 편향 b. 임베딩은 L2 정규화한다.
    softmax(CLIP): t' 초기 log(1/0.07), 상한 ln 100, 편향 없음.   sigmoid(SigLIP): t' 초기 log 10, b 초기 -10, 상한 없음 (논문 3.2절)."""

    def __init__(self, cfg: Cfg | None = None, t_init: float = LOGIT_SCALE_INIT, b_init: float | None = None,
                 t_max: float | None = LOGIT_SCALE_MAX):
        super().__init__()
        cfg = cfg or Cfg()
        self.visual, self.text = ImageEncoder(cfg), TextEncoder(cfg)
        self.logit_scale = nn.Parameter(torch.tensor(float(t_init)))  # t' (log 매개변수화)
        self.logit_bias = nn.Parameter(torch.tensor(float(b_init))) if b_init is not None else None
        self.t_max = t_max

    def encode_image(self, img):
        return F.normalize(self.visual(img), dim=-1)

    def encode_text(self, text):
        return F.normalize(self.text(text), dim=-1)

    @torch.no_grad()
    def clamp_logit_scale(self):
        """옵티마이저 스텝 뒤에 호출한다. t_max가 None이면 자르지 않는다."""
        if self.t_max is not None:
            self.logit_scale.clamp_(0, self.t_max)

    def forward(self, img, text):
        return self.encode_image(img), self.encode_text(text)
