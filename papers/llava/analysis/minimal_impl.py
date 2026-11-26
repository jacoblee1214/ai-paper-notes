"""LLaVA (Liu, Li, Wu, Lee; NeurIPS 2023, arXiv:2304.08485) 최소 구현.

논문 결과를 재현하려는 코드가 아니라 구조와 학습 절차를 이해하려고 짠 토이다. 식·절 번호는 논문 v2 기준.
  식 1  H_v = W · Z_v, Z_v = g(X_v)                     -> build_projector (linear), LlavaToy.encode_images
  식 2  첫 턴의 [X_q, X_v] 또는 [X_v, X_q]               -> build_instruct (이 토이는 공식 코드처럼 이미지 먼저)
  식 3  p(X_a | X_v, X_instruct) = Π p(x_i | ...)        -> LlavaToy.forward (답변 토큰과 종료 토큰에만 손실)
  §4.1  CLIP ViT-L/14의 마지막 층 앞 격자 특징             -> TinyViT.hidden_states, select_features(select_layer=-2)
  §4.2  1단계: 투영만 학습(특징 정렬), 2단계: 투영 + LM    -> set_stage, train_stage
  LLaVA-1.5(arXiv:2310.03744)의 2층 MLP 투영기            -> build_projector("mlp2x_gelu")

공식 코드(haotian-liu/LLaVA, 커밋 c121f04, 정적으로만 읽음)와 맞춘 점
  - 이미지 자리표시 토큰 IMAGE_TOKEN_INDEX = -200 과 IGNORE_INDEX = -100 (llava/constants.py)
  - 시각 토큰을 자리표시 위치에 끼우고 그 라벨을 IGNORE 로 채움 (llava_arch.prepare_inputs_labels_for_multimodal)
  - hidden_states[select_layer] 에서 CLS 를 떼어 패치 토큰만 씀 (clip_encoder.feature_select)
  - 1단계 템플릿은 "plain" (이미지 + 캡션 + 구분자, 질문 없음), 2단계는 v1 계열 (USER/ASSISTANT 표지, 답변과 </s> 만 손실)
  - 비전 인코더는 requires_grad False + no_grad, 1단계는 투영기만 requires_grad (tune_mm_mlp_adapter)
토이와 실물의 차이: 비전 인코더는 직접 사전학습한 4층 ViT(CLS 전역 목적), LM 은 3층 디코더(고정 사인 위치, GELU MLP),
이미지는 20x20 합성 장면, 어휘는 약 50개. 사전학습 데이터와 가중치는 받지 않고 전부 코드로 생성한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100          # llava/constants.py
IMAGE_TOKEN_INDEX = -200     # llava/constants.py

# ================================================================ 해석식 (실물 규모 산술)


def num_visual_tokens(image_size: int, patch: int, include_cls: bool = False) -> int:
    """격자 패치 수 (image/patch)^2. LLaVA 는 CLS 를 떼므로 기본은 CLS 제외 (feature_select 의 [:, 1:])."""
    if image_size % patch:
        raise ValueError(f"image {image_size} not divisible by patch {patch}")
    return (image_size // patch) ** 2 + (1 if include_cls else 0)


def projector_params(kind: str, d_vis: int, d_lm: int) -> int:
    """linear: d_vis*d_lm + d_lm. mlp{k}x_gelu: 첫 층은 d_vis -> d_lm, 나머지 k-1 층은 d_lm -> d_lm (builder.py)."""
    if kind == "linear":
        return d_vis * d_lm + d_lm
    if kind.startswith("mlp") and kind.endswith("x_gelu"):
        depth = int(kind[3:-6])
        return (d_vis * d_lm + d_lm) + (depth - 1) * (d_lm * d_lm + d_lm)
    raise ValueError(kind)


def clip_vit_params(width: int, layers: int, mlp: int, patch: int, image: int, in_ch: int = 3) -> int:
    """HF CLIPVisionModel 의 파라미터 수(투영 head 제외). 합성곱 편향 없음, 사전 LN, 사후 LN, q/k/v/out 편향 있음."""
    n_pos = (image // patch) ** 2 + 1
    emb = width + in_ch * patch * patch * width + n_pos * width
    ln = 2 * width
    per_layer = 2 * ln + 4 * (width * width + width) + (width * mlp + mlp) + (mlp * width + width)
    return emb + ln + layers * per_layer + ln


def llama_params(hidden: int, layers: int, ffn: int, vocab: int = 32000) -> int:
    """Llama/Vicuna: 입력 임베딩 + lm_head(비공유) + 층 x (4 h^2 + 3 h ffn + 2 RMSNorm) + 최종 RMSNorm."""
    per_layer = 4 * hidden * hidden + 3 * hidden * ffn + 2 * hidden
    return 2 * vocab * hidden + layers * per_layer + hidden


def context_budget(max_len: int, n_visual: int) -> int:
    """시각 토큰을 뺀 텍스트 토큰 예산. 공식 코드는 합친 뒤 max_len 으로 뒤를 자른다(llava_arch)."""
    return max_len - n_visual


# ================================================================ 합성 장면, 질문, 어휘

G = 5                    # 칸 수 한 변. 패치 크기 4px, 이미지 20px -> 패치 25개 (G=8 은 학습이 느려 예비 실험 뒤 줄였다)
PATCH = 4
IMG = G * PATCH
MAX_OBJ = 4
COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
NCOL = len(COLORS)
RGB = torch.tensor([[1.0, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 1.0],
                    [1.0, 0.9, 0.1], [1.0, 0.1, 1.0], [0.1, 1.0, 1.0]])
SPECIAL = ["<pad>", "<bos>", "<eos>", "<sys>", "<user>", "<asst>"]
WORDS = ["a", "object", "objects", "at", "and", "what", "color", "where", "is", "the", "how", "many",
         "there", "are", "?", "yes", "no"]
ROWS = [f"r{i}" for i in range(G)]
COLS = [f"c{i}" for i in range(G)]
NUMS = [f"n{i}" for i in range(MAX_OBJ + 1)]
VOCAB = SPECIAL + WORDS + COLORS + ROWS + COLS + NUMS
TOK = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
PAD, BOS, EOS, SYS, USER, ASST = (TOK[w] for w in SPECIAL)
ROW0, COL0, NUM0, COLOR0 = TOK["r0"], TOK["c0"], TOK["n0"], TOK["red"]
Q_LEN = 6
QTYPES = ["count", "color_at", "where", "exist"]
SPATIAL_QTYPES = ["count", "color_at", "where"]   # 위치·개수가 필요한 질문 (exist 는 색 유무만)


@dataclass
class Scenes:
    nobj: torch.Tensor      # [n]
    cells: torch.Tensor     # [n, MAX_OBJ] 칸 번호(행 * G + 열), 물체 없음 = -1
    colors: torch.Tensor    # [n, MAX_OBJ] 색 번호, 물체 없음 = -1


def sample_scenes(n: int, gen: torch.Generator) -> Scenes:
    """물체 1..4개, 색은 서로 다르고 칸도 서로 다르다."""
    nobj = torch.randint(1, MAX_OBJ + 1, (n,), generator=gen)
    cells = torch.argsort(torch.rand(n, G * G, generator=gen), dim=1)[:, :MAX_OBJ]
    colors = torch.argsort(torch.rand(n, NCOL, generator=gen), dim=1)[:, :MAX_OBJ]
    valid = torch.arange(MAX_OBJ)[None] < nobj[:, None]
    return Scenes(nobj, torch.where(valid, cells, -1), torch.where(valid, colors, -1))


def render(sc: Scenes, gen: torch.Generator, noise: float = 0.05) -> torch.Tensor:
    """장면 -> 이미지 [n, 3, 20, 20]. 배경은 회색 0.5, 물체는 4x4 픽셀 색 사각형, 가우스 잡음을 더한다."""
    n = sc.nobj.shape[0]
    grid = torch.full((n, G * G, 3), 0.5)
    for j in range(MAX_OBJ):
        idx = torch.nonzero(sc.cells[:, j] >= 0).squeeze(1)
        grid[idx, sc.cells[idx, j]] = RGB[sc.colors[idx, j]]
    img = grid.reshape(n, G, G, 3).permute(0, 3, 1, 2)
    img = img.repeat_interleave(PATCH, dim=2).repeat_interleave(PATCH, dim=3)
    return img + noise * torch.randn(img.shape, generator=gen)


def sorted_objects(sc: Scenes):
    """칸 번호 오름차순으로 정렬한 (칸, 색) 목록. 캡션과 같은 순서."""
    out = []
    cells, colors = sc.cells.tolist(), sc.colors.tolist()
    for b in range(len(cells)):
        objs = sorted((c, k) for c, k in zip(cells[b], colors[b]) if c >= 0)
        out.append(objs)
    return out


def caption_tokens(sc: Scenes) -> list:
    """"a red object at r2 c3 and a blue object at r5 c1" (칸 번호 순). 2단계 이전의 정렬 데이터와 LM 사전학습에 쓴다."""
    caps = []
    for objs in sorted_objects(sc):
        toks = []
        for i, (c, k) in enumerate(objs):
            if i:
                toks.append(TOK["and"])
            toks += [TOK["a"], COLOR0 + k, TOK["object"], TOK["at"], ROW0 + c // G, COL0 + c % G]
        caps.append(toks)
    return caps


def make_qa(sc: Scenes, qtype: str, gen: torch.Generator):
    """질문 [n, 6]과 답 [n, 2] (두 번째 칸은 PAD 일 수 있음). 질문 정의는 45번 BLIP-2 토이와 같은 형태다."""
    n = sc.nobj.shape[0]
    q = torch.zeros(n, Q_LEN, dtype=torch.long)
    a = torch.full((n, 2), PAD, dtype=torch.long)
    pick = (torch.rand(n, generator=gen) * sc.nobj).long()
    ar = torch.arange(n)
    pcell, pcol = sc.cells[ar, pick], sc.colors[ar, pick]
    if qtype == "count":
        q[:] = torch.tensor([TOK[w] for w in ["how", "many", "objects", "are", "there", "?"]])
        a[:, 0] = NUM0 + sc.nobj
    elif qtype == "color_at":
        q[:, 0], q[:, 1], q[:, 2] = TOK["what"], TOK["color"], TOK["at"]
        q[:, 3], q[:, 4], q[:, 5] = ROW0 + pcell // G, COL0 + pcell % G, TOK["?"]
        a[:, 0] = COLOR0 + pcol
    elif qtype == "where":
        q[:, 0], q[:, 1], q[:, 2] = TOK["where"], TOK["is"], TOK["the"]
        q[:, 3], q[:, 4], q[:, 5] = COLOR0 + pcol, TOK["object"], TOK["?"]
        a[:, 0], a[:, 1] = ROW0 + pcell // G, COL0 + pcell % G
    elif qtype == "exist":
        ask = torch.randint(0, NCOL, (n,), generator=gen)
        ask = torch.where(torch.rand(n, generator=gen) < 0.5, pcol, ask)
        present = (sc.colors == ask[:, None]).any(1)
        q[:, 0], q[:, 1], q[:, 2] = TOK["is"], TOK["there"], TOK["a"]
        q[:, 3], q[:, 4], q[:, 5] = COLOR0 + ask, TOK["object"], TOK["?"]
        a[:, 0] = torch.where(present, TOK["yes"], TOK["no"])
    else:
        raise ValueError(qtype)
    return q, a


def mixed_qa(sc: Scenes, gen: torch.Generator):
    """장면마다 질문 유형을 번갈아 배정한다. 반환: 질문, 답, 유형 번호 [n]."""
    n = sc.nobj.shape[0]
    t = torch.arange(n) % len(QTYPES)
    q = torch.zeros(n, Q_LEN, dtype=torch.long)
    a = torch.zeros(n, 2, dtype=torch.long)
    for i, name in enumerate(QTYPES):
        qq, aa = make_qa(sc, name, gen)
        q[t == i], a[t == i] = qq[t == i], aa[t == i]
    return q, a, t


def answer_with_eos(a: torch.Tensor) -> torch.Tensor:
    """[n, 2] 답 -> [n, 3] (답 토큰들, EOS, PAD). 모델은 어디서 멈출지도 배운다(Table 2 설명)."""
    n = a.shape[0]
    out = torch.full((n, 3), PAD, dtype=torch.long)
    length = (a != PAD).sum(1)
    out[:, :2] = a
    out[torch.arange(n), length] = EOS
    return out


# ================================================================ 대화 템플릿과 손실 마스크

def build_instruct(q: torch.Tensor, a: torch.Tensor):
    """2단계(v1 계열) 템플릿: <bos> <sys> <user> <image> 질문 <asst> 답 <eos>. 손실은 답과 <eos> 에만(공식 코드의 마스킹).
    ids 에는 이미지 자리표시 IMAGE_TOKEN_INDEX 가 한 개 들어 있다."""
    n = q.shape[0]
    a3 = answer_with_eos(a)
    head = torch.tensor([BOS, SYS, USER, IMAGE_TOKEN_INDEX]).expand(n, 4)
    ids = torch.cat([head, q, torch.full((n, 1), ASST), a3], dim=1)
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[:, -3:] = torch.where(a3 == PAD, torch.full_like(a3, IGNORE_INDEX), a3)
    return ids, labels


def pad_rows(rows: list, pad_value: int) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [pad_value] * (width - len(r)) for r in rows], dtype=torch.long)


def build_plain(caps: list):
    """1단계 "plain" 템플릿: <bos> <image> 캡션 <eos>. 질문과 시스템 메시지가 없다(공식 preprocess_plain).
    논문 §4.2 는 '간단한 설명을 요청하는 질문을 무작위로 뽑는다'고 쓰지만 코드는 질문을 버리고 이미지 토큰만 둔다."""
    ids = pad_rows([[BOS, IMAGE_TOKEN_INDEX] + c + [EOS] for c in caps], PAD)
    labels = pad_rows([[IGNORE_INDEX, IGNORE_INDEX] + c + [EOS] for c in caps], IGNORE_INDEX)
    return ids, labels


def build_text_pretrain(sc: Scenes, gen: torch.Generator):
    """LM 사전학습(텍스트만): 절반은 캡션 단독 `<bos> 캡션 <eos>`, 절반은 장면 설명을 글로 읽고 답하는 질의응답.
    실물에서 Vicuna 가 이미 텍스트만으로 학습된 상태를 흉내 낸다. 모든 토큰에 손실을 건다."""
    caps = caption_tokens(sc)
    q, a, _ = mixed_qa(sc, gen)
    a3 = answer_with_eos(a).tolist()
    rows = []
    for i, cap in enumerate(caps):
        if i % 2 == 0:
            rows.append([BOS] + cap + [EOS])
        else:
            ans = [t for t in a3[i] if t != PAD]
            rows.append([BOS, SYS, USER] + cap + q[i].tolist() + [ASST] + ans)
    ids = pad_rows(rows, PAD)
    labels = torch.where(ids == PAD, torch.full_like(ids, IGNORE_INDEX), ids)
    return ids, labels


# ================================================================ 모델

class Block(nn.Module):
    """pre-LN 블록. 비전 인코더(양방향)와 LM(인과) 가 공유한다."""

    def __init__(self, dim, heads, mlp, causal):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, mlp), nn.Linear(mlp, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x + self.out(o.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def sincos_2d(grid: int, dim: int, n_tokens: int) -> torch.Tensor:
    """[1, n_tokens, dim]: 첫 토큰(CLS)은 0, 나머지는 행 사인파(앞 절반 차원)와 열 사인파(뒤 절반 차원). 래스터 순서."""
    half = dim // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, 2) / half))
    ang = torch.arange(grid)[:, None] * freqs[None]                        # [grid, half/2]
    one = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)               # [grid, half]
    rows = one[:, None, :].expand(grid, grid, half)
    cols = one[None, :, :].expand(grid, grid, half)
    pe = torch.cat([rows, cols], dim=-1).reshape(grid * grid, dim)
    out = torch.zeros(1, n_tokens, dim)
    out[0, 1:] = pe
    return out


class TinyViT(nn.Module):
    """CLIP 비전 타워와 같은 구조: 합성곱 패치 임베딩(편향 없음), CLS, 위치 임베딩, 사전 LN, 블록들, 사후 LN.
    hidden_states 는 HF 와 같이 [임베딩 직후(사전 LN 뒤), 블록1 출력, ..., 블록N 출력] 이고 사후 LN 은 CLS 에만 건다."""

    def __init__(self, img=IMG, patch=PATCH, dim=64, depth=4, heads=4, mlp=256, in_ch=3):
        super().__init__()
        self.dim, self.patch, self.depth = dim, patch, depth
        n = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, patch, bias=False)
        self.cls = nn.Parameter(torch.randn(dim) * 0.02)
        # 위치 임베딩은 2D 사인 초기화로 둔다(학습 가능). 학습된 ViT 의 위치 임베딩은 위치를 읽을 수 있을 만큼 크지만,
        # 이 토이의 전역 목적은 위치를 쓰지 않아 무작위 작은 초기화(0.02)는 끝까지 작게 남아 위치 정보가 사라지기 때문이다.
        self.pos = nn.Parameter(sincos_2d(img // patch, dim, n + 1))
        self.pre_ln = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp, causal=False) for _ in range(depth)])
        self.post_ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NCOL + MAX_OBJ + 1)   # 사전학습 전용(전역 목적). 타워의 일부가 아니다.

    def hidden_states(self, x):
        z = self.patch_embed(x).flatten(2).transpose(1, 2)
        z = torch.cat([self.cls.expand(z.shape[0], 1, -1), z], dim=1) + self.pos
        hs = [self.pre_ln(z)]
        for blk in self.blocks:
            hs.append(blk(hs[-1]))
        return hs

    def tower_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("head."))

    def global_logits(self, x):
        """CLS 만 쓰는 전역 목적(CLIP 처럼 이미지 전체를 한 벡터로 요약): 색 존재 6 + 물체 수 5."""
        return self.head(self.post_ln(self.hidden_states(x)[-1][:, 0]))


def select_features(hs: list, select_layer: int = -2, select_feature: str = "patch"):
    """공식 CLIPVisionTower.feature_select: hidden_states[select_layer], 'patch' 면 CLS 를 뗀다."""
    f = hs[select_layer]
    if select_feature == "patch":
        return f[:, 1:]
    if select_feature == "cls_patch":
        return f
    raise ValueError(select_feature)


def build_projector(kind: str, d_vis: int, d_lm: int) -> nn.Module:
    """공식 builder.py: 'linear' 또는 'mlp{k}x_gelu' (Linear, (GELU, Linear) x (k-1))."""
    if kind == "linear":
        return nn.Linear(d_vis, d_lm)
    if kind.startswith("mlp") and kind.endswith("x_gelu"):
        depth = int(kind[3:-6])
        mods = [nn.Linear(d_vis, d_lm)]
        for _ in range(1, depth):
            mods += [nn.GELU(), nn.Linear(d_lm, d_lm)]
        return nn.Sequential(*mods)
    raise ValueError(kind)


class TinyLM(nn.Module):
    """인과 Transformer 디코더. 입력은 임베딩이라 시각 토큰을 그대로 끼울 수 있다(inputs_embeds)."""

    def __init__(self, vocab=V, dim=96, depth=3, heads=4, mlp=384, max_len=160):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        # 고정된 사인 위치 인코딩: 사전학습(텍스트 45토큰 이하)에서 본 적 없는 위치(시각 토큰이 놓이는 곳)도 정의되어 있다.
        # 실물 LLM 의 RoPE 가 어떤 위치에서도 정의되는 점을 흉내 낸 것이다(학습되는 위치 임베딩은 1단계에서 얼어 있어 미학습 위치가 무작위로 남는다).
        pe = torch.zeros(max_len, dim)
        ang = torch.arange(max_len)[:, None] / (10000 ** (torch.arange(0, dim, 2)[None] / dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
        self.register_buffer("pos", pe[None], persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp, causal=True) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, embeds, labels=None):
        x = embeds + self.pos[:, : embeds.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln(x))
        if labels is None:
            return logits, None
        # 다음 토큰 예측: 위치 t 의 로짓이 t+1 의 라벨을 맞힌다. IGNORE 위치는 손실에서 빠진다.
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                               ignore_index=IGNORE_INDEX)
        return logits, loss

    def text_loss(self, ids, labels):
        return self(self.embed(ids), labels)[1]


def merge_visual_tokens(ids, labels, embed, vis, max_len=None):
    """공식 prepare_inputs_labels_for_multimodal 의 단일 이미지 버전.
    ids 의 IMAGE_TOKEN_INDEX 자리에 시각 임베딩(vis [B, T, D])을 끼우고, 그 위치의 라벨은 IGNORE 로 채운다.
    행마다 길이가 같으므로(우리 템플릿은 오른쪽 PAD) 그대로 쌓는다. max_len 이 있으면 공식 코드처럼 뒤를 자른다."""
    rows_e, rows_l = [], []
    emb_all = embed(ids.clamp(min=0))
    for b in range(ids.shape[0]):
        pos = torch.nonzero(ids[b] == IMAGE_TOKEN_INDEX).squeeze(1)
        assert pos.numel() == 1, "토이는 이미지 하나만 지원한다"
        p = int(pos[0])
        t = vis.shape[1]
        rows_e.append(torch.cat([emb_all[b, :p], vis[b], emb_all[b, p + 1:]], dim=0))
        rows_l.append(torch.cat([labels[b, :p], torch.full((t,), IGNORE_INDEX, dtype=labels.dtype, device=labels.device), labels[b, p + 1:]]))
    emb, lab = torch.stack(rows_e), torch.stack(rows_l)
    if max_len is not None:
        emb, lab = emb[:, :max_len], lab[:, :max_len]
    return emb, lab


class LlavaToy(nn.Module):
    """비전 인코더(동결) -> 선택한 층의 패치 특징 -> 투영기 -> 텍스트 임베딩 사이에 삽입 -> LM."""

    def __init__(self, vision: TinyViT, lm: TinyLM, projector: str = "linear", select_layer: int = -2):
        super().__init__()
        self.vision, self.lm = vision, lm
        self.select_layer = select_layer
        self.projector_kind = projector
        self.projector = build_projector(projector, vision.dim, lm.dim)
        self.vision.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    @torch.no_grad()
    def vision_features(self, images):
        return select_features(self.vision.hidden_states(images), self.select_layer)

    def encode_images(self, images):
        return self.projector(self.vision_features(images))        # 식 1: H_v = W Z_v

    def logits_and_loss(self, images, ids, labels, max_len=None):
        emb, lab = merge_visual_tokens(ids, labels, self.lm.embed, self.encode_images(images), max_len)
        logits, loss = self.lm(emb, lab)
        return logits, lab, loss

    def forward(self, images, ids, labels, max_len=None):
        return self.logits_and_loss(images, ids, labels, max_len)[2]


def set_stage(model: LlavaToy, stage: int):
    """stage 1: 투영기만(--tune_mm_mlp_adapter True 가 나머지를 얼림). stage 2: 투영기 + LM, 비전 인코더는 계속 동결."""
    model.requires_grad_(False)
    for p in model.projector.parameters():
        p.requires_grad = True
    if stage == 2:
        for p in model.lm.parameters():
            p.requires_grad = True
    elif stage != 1:
        raise ValueError(stage)
    return [p for p in model.parameters() if p.requires_grad]


# ================================================================ 사전학습과 학습 루프

def cosine_lr(step: int, total: int, base: float, warmup: float = 0.03) -> float:
    w = max(1, int(total * warmup))
    if step < w:
        return base * (step + 1) / w
    return base * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def run_steps(params, steps, lr, loss_fn, clip=1.0):
    """AdamW(weight_decay 0) + 코사인 + 3% 워밍업 + 기울기 클립 1.0. 공식 학습 인자를 따른다. 마지막 50스텝 평균 손실을 돌려준다."""
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    tail = []
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(s, steps, lr)
        loss = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if s >= steps - 50:
            tail.append(float(loss.detach()))
    return sum(tail) / max(1, len(tail))


def pretrain_vision(vision: TinyViT, gen, steps=500, batch=128, lr=2e-3):
    dev = next(vision.parameters()).device
    """CLIP 을 흉내 낸 전역 목적: CLS 가 색 존재(6, BCE)와 물체 수(CE)만 예측한다. 위치는 목적에 없다.
    이 목적에서는 마지막 블록의 패치 출력이 손실에 닿지 않는다(CLS 는 마지막 블록의 입력 패치만 본다)."""
    vision.train()
    params = list(vision.parameters())

    def loss_fn():
        sc = sample_scenes(batch, gen)
        img = render(sc, gen)
        present = torch.zeros(batch, NCOL)
        for j in range(MAX_OBJ):
            idx = torch.nonzero(sc.colors[:, j] >= 0).squeeze(1)
            present[idx, sc.colors[idx, j]] = 1.0
        out = vision.global_logits(img.to(dev))
        return (F.binary_cross_entropy_with_logits(out[:, :NCOL], present.to(dev))
                + F.cross_entropy(out[:, NCOL:], sc.nobj.to(dev)))

    final = run_steps(params, steps, lr, loss_fn)
    vision.eval()
    vision.requires_grad_(False)
    return final


def pretrain_lm(lm: TinyLM, gen, steps=1200, batch=128, lr=1e-3):
    lm.train()
    dev = next(lm.parameters()).device

    def loss_fn():
        ids, labels = build_text_pretrain(sample_scenes(batch, gen), gen)
        return lm.text_loss(ids.to(dev), labels.to(dev))

    return run_steps(list(lm.parameters()), steps, lr, loss_fn)


def stage1_batch(gen, batch):
    sc = sample_scenes(batch, gen)
    ids, labels = build_plain(caption_tokens(sc))
    return render(sc, gen), ids, labels


def stage2_batch(gen, batch):
    sc = sample_scenes(batch, gen)
    q, a, _ = mixed_qa(sc, gen)
    ids, labels = build_instruct(q, a)
    return render(sc, gen), ids, labels


def train_stage(model: LlavaToy, stage: int, gen, steps: int, batch: int, lr: float):
    model.train()
    params = set_stage(model, stage)
    make = stage1_batch if stage == 1 else stage2_batch
    dev = next(model.parameters()).device
    return run_steps(params, steps, lr, lambda: model(*[t.to(dev) for t in make(gen, batch)]))


@torch.no_grad()
def evaluate(model: LlavaToy, n_per_type: int = 500, seed: int = 999):
    """질문 유형별 정확도(답 토큰과 <eos> 가 모두 맞아야 정답). 교사 강제의 argmax 가 모두 맞는 것은 탐욕 생성이 맞는 것과 같다."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name in QTYPES:
        sc = sample_scenes(n_per_type, gen)
        q, a = make_qa(sc, name, gen)
        ids, labels = build_instruct(q, a)
        dev = next(model.parameters()).device
        logits, lab, _ = model.logits_and_loss(render(sc, gen).to(dev), ids.to(dev), labels.to(dev))
        pred = logits[:, :-1].argmax(-1)
        tgt = lab[:, 1:]
        mask = tgt != IGNORE_INDEX
        ok = ((pred == tgt) | ~mask).all(dim=1)
        out[name] = float(ok.float().mean())
    out["spatial"] = sum(out[k] for k in SPATIAL_QTYPES) / len(SPATIAL_QTYPES)
    out["macro"] = sum(out[k] for k in QTYPES) / len(QTYPES)
    return out


@torch.no_grad()
def caption_eval(model: LlavaToy, n: int = 500, seed: int = 998):
    """1단계 진단: 캡션 토큰 정확도(교사 강제). 내용 토큰(색, 행, 열)만 따로 센다."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    sc = sample_scenes(n, gen)
    ids, labels = build_plain(caption_tokens(sc))
    dev = next(model.parameters()).device
    logits, lab, _ = model.logits_and_loss(render(sc, gen).to(dev), ids.to(dev), labels.to(dev))
    pred, tgt = logits[:, :-1].argmax(-1), lab[:, 1:]
    mask = tgt != IGNORE_INDEX
    content = mask & (((tgt >= COLOR0) & (tgt < COLOR0 + NCOL)) | ((tgt >= ROW0) & (tgt < ROW0 + G)) |
                      ((tgt >= COL0) & (tgt < COL0 + G)))
    return {"token_acc": float(((pred == tgt) & mask).sum() / mask.sum()),
            "content_acc": float(((pred == tgt) & content).sum() / content.sum())}


# 단계별 설정 이름 -> (투영기, 층, 단계 구성). run_experiments.py 가 쓴다.
def build_model(vision: TinyViT, lm: TinyLM, projector: str, select_layer: int, seed: int) -> LlavaToy:
    """사전학습된 인코더와 LM 을 복사해 붙인다(arm 마다 같은 출발점). 투영기 초기화는 seed 로 고정한다."""
    import copy
    torch.manual_seed(10_000 + seed)
    return LlavaToy(copy.deepcopy(vision), copy.deepcopy(lm), projector, select_layer)
