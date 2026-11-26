"""ViT (Dosovitskiy et al., ICLR 2021, arXiv:2010.11929) 최소 구현.

논문 결과를 재현하려는 코드가 아니라 구조와 식을 이해하려고 짠 토이다. 식 번호는 논문 기준.
  식 1  z0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos         -> PatchEmbed, ViT.cls, ViT.pos_embed
  식 2  z'_l = MSA(LN(z_{l-1})) + z_{l-1}                      -> Block
  식 3  z_l  = MLP(LN(z'_l)) + z'_l                             -> Block
  식 4  y = LN(z_L^0)                                           -> ViT.norm 뒤 cls(또는 GAP)
  부록 A 식 5-8 (qkv, softmax(qk^T/sqrt(Dh)), 헤드 연결, U_msa) -> MSA
  §3.2  고해상도 미세조정에서 위치 임베딩 2D 보간                    -> resize_pos_embed
  표 1  ViT-B/L/H 의 층 수, D, MLP 크기, 헤드 수                  -> CONFIGS

해석식(파라미터 수, 연산량)도 여기에 둔다: count_params_formula, macs_formula, pretrain_exaflops.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 표 1 (층 수, 은닉 크기 D, MLP 크기, 헤드 수). 패치 크기는 모델 이름의 /P.
CONFIGS = {
    "B": dict(depth=12, dim=768, mlp=3072, heads=12),
    "L": dict(depth=24, dim=1024, mlp=4096, heads=16),
    "H": dict(depth=32, dim=1280, mlp=5120, heads=16),
}


def num_patches(img_h, img_w, patch):
    """N = HW / P^2 (§3.1). H, W 가 P 로 나누어떨어져야 한다."""
    if img_h % patch or img_w % patch:
        raise ValueError(f"image {img_h}x{img_w} not divisible by patch {patch}")
    return (img_h // patch) * (img_w // patch)


def patchify(x, patch):
    """(B, C, H, W) -> (B, N, C*P*P). 패치는 래스터 순서, 각 패치는 (C, P, P) 순으로 편다.
    이 순서가 nn.Conv2d 가중치 (D, C, P, P) 를 (D, C*P*P) 로 편 것과 같아서 conv 와 선형 사영이 일치한다."""
    b, c, h, w = x.shape
    num_patches(h, w, patch)
    x = x.reshape(b, c, h // patch, patch, w // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // patch) * (w // patch), c * patch * patch)


class PatchEmbed(nn.Module):
    """식 1의 x_p E. 구현은 stride=P 인 conv 이고, patchify + Linear 와 같다(`forward_linear`)."""

    def __init__(self, patch, in_ch, dim):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)

    def forward_linear(self, x):
        w = self.proj.weight.reshape(self.proj.out_channels, -1)  # (D, C*P*P)
        return patchify(x, self.patch) @ w.t() + self.proj.bias


class MSA(nn.Module):
    """부록 A 식 5-8. D_h = D/k. qkv 는 한 번에 D -> 3D 로 사영한다."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, z, return_attn=False):
        b, n, d = z.shape
        q, k, v = self.qkv(z).reshape(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)  # 식 6, (B, h, N, N)
        o = (a @ v).transpose(1, 2).reshape(b, n, d)  # 식 7 + 헤드 연결
        o = self.out(o)  # 식 8, U_msa
        return (o, a) if return_attn else o


class Block(nn.Module):
    """pre-norm 인코더 블록 (식 2, 3)."""

    def __init__(self, dim, mlp, heads):
        super().__init__()
        self.ln1, self.msa = nn.LayerNorm(dim), MSA(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, z):
        z = z + self.msa(self.ln1(z))
        return z + self.mlp(self.ln2(z))


class ViT(nn.Module):
    """pool='cls' 는 논문 기본(식 4의 z_L^0), 'gap' 은 부록 D.3 의 평균 풀링.
    use_pos=False 면 위치 임베딩을 더하지 않는다(부록 D.4 의 "No Pos. Emb." = 패치 주머니).
    head 는 미세조정 때처럼 0 으로 초기화된 단일 선형층이다(§3.2). 사전학습의 은닉 1층 MLP 헤드는 넣지 않았다."""

    def __init__(self, img_size=224, patch=16, in_ch=3, num_classes=1000, depth=12, dim=768, mlp=3072,
                 heads=12, pool="cls", use_pos=True):
        super().__init__()
        assert pool in ("cls", "gap")
        self.pool, self.use_pos, self.patch = pool, use_pos, patch
        self.embed = PatchEmbed(patch, in_ch, dim)
        n = num_patches(img_size, img_size, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        n_tok = n + (1 if pool == "cls" else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tok, dim) * 0.02) if use_pos else None
        self.blocks = nn.ModuleList([Block(dim, mlp, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def tokens(self, x):
        z = self.embed(x)
        if self.cls_token is not None:
            z = torch.cat([self.cls_token.expand(z.shape[0], -1, -1), z], dim=1)
        if self.pos_embed is not None:
            z = z + self.pos_embed
        return z

    def features(self, x):
        z = self.tokens(x)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)  # LN 은 토큰별이라 cls 만 정규화한 식 4 와 같다
        return z[:, 0] if self.pool == "cls" else z.mean(dim=1)

    def forward(self, x):
        return self.head(self.features(x))


def make_vit(name, img_size=224, num_classes=1000, **kw):
    """name 예: 'B/16', 'L/16', 'H/14'."""
    size, p = name.split("/")
    return ViT(img_size=img_size, patch=int(p), num_classes=num_classes, **CONFIGS[size], **kw)


def resize_pos_embed(pos, old_grid, new_grid, has_cls=True):
    """§3.2: 사전학습한 위치 임베딩을 원래 이미지 위치에 따라 2D 보간한다.
    pos: (1, [1+]old_h*old_w, D). 격자는 (h, w). [CLS] 위치는 그대로 둔다.
    공식 코드(scipy.ndimage.zoom order=1)와 같게 양 끝점을 끝점에 맞춘다(= align_corners=True)."""
    old_h, old_w = old_grid
    new_h, new_w = new_grid
    cls, grid = (pos[:, :1], pos[:, 1:]) if has_cls else (pos[:, :0], pos)
    assert grid.shape[1] == old_h * old_w
    d = grid.shape[-1]
    grid = grid.reshape(1, old_h, old_w, d).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_h, new_w), mode="bilinear", align_corners=True)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, d)
    return torch.cat([cls, grid], dim=1)


# ---------------------------------------------------------------- 해석식

def count_params_formula(depth, dim, mlp, patch, img_size, num_classes, in_ch=3, pool="cls"):
    """파라미터 수를 항별로 센다 (편향 포함).
    패치 임베딩 P²C·D + D, [CLS] D, 위치 (N+1)·D,
    층당: LN 2D x2, qkv 3D²+3D, U_msa D²+D, MLP 2DM + M + D,
    마지막 LN 2D, 헤드 D·K + K."""
    n = (img_size // patch) ** 2
    n_tok = n + (1 if pool == "cls" else 0)
    embed = patch * patch * in_ch * dim + dim
    cls = dim if pool == "cls" else 0
    pos = n_tok * dim
    per_layer = 2 * (2 * dim) + (3 * dim * dim + 3 * dim) + (dim * dim + dim) + (2 * dim * mlp + mlp + dim)
    return embed + cls + pos + depth * per_layer + 2 * dim + dim * num_classes + num_classes


def macs_formula(depth, dim, mlp, patch, img_size, num_classes, in_ch=3, pool="cls", img_w=None):
    """순전파 곱셈-덧셈(MAC) 수. 토큰 수 T = N(+1).
    패치 임베딩 N·P²C·D, 층당 qkv 3T D², 사영 T D², MLP 2T D M, attention 행렬곱 2 T² D (QK^T, AV), 헤드 D·K.
    softmax, LN, GELU, 덧셈은 세지 않는다."""
    gh = img_size // patch
    gw = (img_w or img_size) // patch
    n = gh * gw
    t = n + (1 if pool == "cls" else 0)
    embed = n * patch * patch * in_ch * dim
    lin = t * (3 * dim * dim + dim * dim + 2 * dim * mlp)
    attn = 2 * t * t * dim
    return embed + depth * (lin + attn) + dim * num_classes


def attn_share(depth, dim, mlp, patch, img_size):
    """층 안에서 attention 행렬곱(2T²D)이 나머지 선형층(T(4D²+2DM))의 몇 배인가."""
    t = (img_size // patch) ** 2 + 1
    return (2 * t * t * dim) / (t * (4 * dim * dim + 2 * dim * mlp))


def pretrain_exaflops(macs, images_seen):
    """학습 연산량(exaFLOPs) = 3(순전파+역전파) x 2(MAC->FLOP) x MAC x 본 이미지 수 / 1e18.
    논문 표 6의 ViT 행과 맞는지 확인하려고 둔 어림식이다(부록 D.5 는 계산 방법을 적지 않음)."""
    return 6.0 * macs * images_seen / 1e18
