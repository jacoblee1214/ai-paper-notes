"""DINOv2 (Oquab et al., TMLR 2024, arXiv:2304.07193) 최소 구현.

논문 결과를 재현하려는 코드가 아니라 목적 함수의 구성 요소를 이해하려고 짠 토이다(작은 ViT, 합성 이미지, 프로토타입 64개).
논문 §4의 요소와 이 파일의 대응:
  이미지 수준 DINO 손실   L_DINO = -sum p_t log p_s     -> dino_cls_loss, DINOHead
  패치 수준 iBOT 손실     L_iBOT = -sum_i p_ti log p_si  -> ibot_patch_loss, TinyViT(mask=...)
  머리 분리(untying)       DINO 머리와 iBOT 머리를 따로   -> DINOv2Toy.student["dino_head"], ["ibot_head"]
  Sinkhorn-Knopp 중심화    3회 반복                      -> sinkhorn_knopp
  소프트맥스 중심화(DINO)  이동 평균 중심                  -> DINOCenter
  KoLeo 정규화            -sum log d_{n,i} / n           -> koleo_loss (+ 검산용 kl_entropy)
  EMA 교사                m 일정(논문 0.994 -> 1)         -> ema_update, cosine_schedule
  멀티 크롭               전역 2개 + 지역 여러 개          -> random_resized_crop, DINOv2Toy.loss

ViT 부분(PatchEmbed, MSA, Block)은 ../../vit/analysis/minimal_impl.py(ViT 최소 구현, 같은 저장소 41번)에서 복사해 줄였다.
그 파일은 수정하지 않았다. 달라진 점: 마스크 토큰, 크기가 다른 입력을 위한 위치 임베딩 보간, 패치 토큰 출력.

공식 코드(facebookresearch/dinov2)와의 차이는 04_claim_verification.md 의 D1-D.. 에 있다. 요약:
LayerScale, stochastic depth, SwiGLU, xFormers, FSDP, 시퀀스 패킹, 블록 마스킹(여기는 무작위 패치 마스킹),
고해상도 적응, 128k 프로토타입, 가중치 정규화 머리의 g(여기는 1 고정 코사인 머리)를 넣지 않았다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- ViT 백본 (vit/ 41번에서 복사, 출처 위 주석)

class PatchEmbed(nn.Module):
    def __init__(self, patch, in_ch, dim):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class MSA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, z):
        b, n, d = z.shape
        q, k, v = self.qkv(z).reshape(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), dim=-1)
        return self.out((a @ v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, mlp, heads):
        super().__init__()
        self.ln1, self.msa = nn.LayerNorm(dim), MSA(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, z):
        z = z + self.msa(self.ln1(z))
        return z + self.mlp(self.ln2(z))


class TinyViT(nn.Module):
    """[CLS] + 패치 토큰을 돌려주는 작은 ViT. mask(B, N) bool 이 True 인 패치는 임베딩 단계에서 mask_token 으로 바꾼다
    (공식 코드 prepare_tokens_with_masks 와 같은 위치: 위치 임베딩을 더하기 전).
    입력 크기가 학습 격자와 다르면(지역 크롭) 위치 임베딩을 bilinear 로 보간한다(공식은 bicubic + offset 0.1, D 참고)."""

    def __init__(self, img_size=32, patch=4, in_ch=3, dim=64, depth=3, heads=4, mlp=128):
        super().__init__()
        self.patch, self.grid = patch, img_size // patch
        self.embed = PatchEmbed(patch, in_ch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.grid ** 2, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, mlp, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def pos(self, gh, gw):
        if gh == gw == self.grid:
            return self.pos_embed
        p = self.pos_embed[:, 1:].reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=(gh, gw), mode="bilinear", align_corners=False)
        return torch.cat([self.pos_embed[:, :1], p.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)], dim=1)

    def forward(self, x, mask=None):
        gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        z = self.embed(x)
        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.mask_token.to(z.dtype), z)
        z = torch.cat([self.cls_token.expand(z.shape[0], -1, -1), z], dim=1) + self.pos(gh, gw)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        return z[:, 0], z[:, 1:]  # (B, D), (B, N, D)


# ---------------------------------------------------------------- 머리와 손실

class DINOHead(nn.Module):
    """공식 DINOHead: MLP -> L2 정규화 -> 가중치 정규화 선형층(프로토타입). 여기서는 g=1 로 고정한 코사인 머리다:
    logits_k = <normalize(h), normalize(w_k)>  (범위 [-1, 1], 온도로 나눠서 쓴다)."""

    def __init__(self, in_dim, n_proto, hidden=128, bottleneck=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.Parameter(torch.randn(n_proto, bottleneck) * 0.02)

    def forward(self, x):
        h = F.normalize(self.mlp(x), dim=-1)
        return h @ F.normalize(self.last, dim=-1).t()


@torch.no_grad()
def sinkhorn_knopp(logits, temp, n_iter=3):
    """공식 sinkhorn_knopp_teacher(단일 기기). 입력 (B, K) 교사 로짓, 출력 (B, K) 각 행 합 1.
    Q = exp(z/T) 를 K x B 로 두고 [행 합 = 1/K, 열 합 = 1/B] 를 번갈아 맞춘다(n_iter 번). 마지막에 Q*B 로 열 합을 1 로 만든다."""
    z = logits.float()
    q = torch.exp((z - z.max()) / temp).t()  # (K, B). 전체 최댓값을 빼는 것은 Q 전체에 상수를 곱하는 것이라 결과가 같다
    k, b = q.shape
    q = q / q.sum()
    for _ in range(n_iter):
        q = q / q.sum(dim=1, keepdim=True) / k  # 각 프로토타입의 총 질량 = 1/K
        q = q / q.sum(dim=0, keepdim=True) / b  # 각 샘플의 총 질량 = 1/B
    return (q * b).t()


class DINOCenter(nn.Module):
    """DINO 의 중심화: 교사 로짓에서 c 를 빼고 낮은 온도로 softmax. c 는 배치 평균의 이동 평균(momentum 0.9).
    공식 코드는 이번 스텝의 softmax 에는 이전 스텝까지의 c 를 쓰고 배치 평균은 다음 호출 때 반영한다(= 여기서 update 를 나중에 부르는 것과 같음)."""

    def __init__(self, dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(1, dim))

    @torch.no_grad()
    def probs(self, logits, temp):
        return F.softmax((logits - self.center) / temp, dim=-1)

    @torch.no_grad()
    def update(self, logits):
        self.center = self.center * self.momentum + logits.mean(dim=0, keepdim=True) * (1 - self.momentum)


def teacher_probs(logits, temp, mode, center=None, n_iter=3):
    """mode: 'center'(DINO), 'sk'(DINOv2), 'none'(중심화 없음, 소거 실험용)."""
    if mode == "center":
        return center.probs(logits, temp)
    if mode == "sk":
        return sinkhorn_knopp(logits, temp, n_iter)
    if mode == "none":
        return F.softmax(logits / temp, dim=-1)
    raise ValueError(mode)


def dino_cls_loss(student_global, student_local, teacher_p, student_temp=0.1):
    """이미지 수준 손실. student_global: (2, B, K) 학생 전역 크롭 로짓, student_local: (L, B, K), teacher_p: (2, B, K) 교사 확률.
    교사 전역 크롭 j 는 같은 크롭 j 를 뺀 모든 학생 크롭과 짝짓는다: 전역 2x1 쌍 + 지역 2L 쌍 = 2 + 2L 항의 평균
    (공식 코드의 n_global_crops_loss_terms + n_local_crops_loss_terms 로 나누는 것과 같다). 교차 엔트로피 -sum p_t log p_s."""
    ng = student_global.shape[0]
    terms = []
    for j in range(ng):
        t = teacher_p[j].detach()
        for i in range(ng):
            if i != j:
                terms.append(-(t * F.log_softmax(student_global[i] / student_temp, dim=-1)).sum(-1).mean())
        for s in student_local:
            terms.append(-(t * F.log_softmax(s / student_temp, dim=-1)).sum(-1).mean())
    return sum(terms) / len(terms)


def ibot_patch_loss(student_logits, teacher_p, mask, student_temp=0.1):
    """패치 수준 손실. student_logits, teacher_p: (B, N, K), mask: (B, N) bool(학생에서 가린 패치).
    가린 패치에서만 교차 엔트로피를 계산하고 이미지마다 가린 개수로 나눈 뒤 B 로 평균한다(공식 forward_masked 의 masks_weight 와 같다)."""
    ce = -(teacher_p.detach() * F.log_softmax(student_logits / student_temp, dim=-1)).sum(-1)  # (B, N)
    m = mask.float()
    per_img = (ce * m).sum(1) / m.sum(1).clamp(min=1.0)
    return per_img.sum() / mask.shape[0]


def koleo_loss(x, eps=1e-8):
    """KoLeo: L2 정규화한 x (n, d) 에서 -mean_i log(min_{j!=i} ||x_i - x_j||). 공식 KoLeoLoss 와 같다."""
    x = F.normalize(x, dim=-1, eps=eps)
    dots = x @ x.t()
    dots = dots.masked_fill(torch.eye(x.shape[0], dtype=torch.bool, device=x.device), -2.0)
    nn_idx = dots.argmax(dim=1)
    d = F.pairwise_distance(x, x[nn_idx], p=2, eps=eps)
    return -torch.log(d + eps).mean()


def kl_entropy(x):
    """Kozachenko-Leonenko 미분 엔트로피 추정량(k=1, 단위 nats): H = psi(n) - psi(1) + log V_d + (d/n) sum log rho_i.
    rho_i 는 최근접 이웃 거리, V_d = pi^{d/2} / Gamma(d/2+1) 은 d 차원 단위 공의 부피, psi(1) = -gamma.
    koleo_loss = -mean log rho_i 이므로 H_hat = d * (-koleo) + (psi(n) + gamma + log V_d) 이다."""
    n, d = x.shape
    dist = torch.cdist(x.double(), x.double(), compute_mode="donot_use_mm_for_euclid_dist")  # 행렬곱 방식은 가까운 점에서 0 이 나옴
    dist.fill_diagonal_(float("inf"))
    rho = dist.min(dim=1).values
    log_vd = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)
    euler_gamma = 0.5772156649015329
    return float(torch.special.digamma(torch.tensor(float(n))) + euler_gamma + log_vd + d * torch.log(rho).mean())


# ---------------------------------------------------------------- 교사-학생, EMA, 일정

@torch.no_grad()
def ema_update(teacher, student, m):
    """theta_t <- m * theta_t + (1 - m) * theta_s. m=1 이면 교사 그대로, m=0 이면 학생 복사."""
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def cosine_schedule(base, final, it, total):
    """공식 CosineScheduler(워밍업 없음): final + 0.5 (base - final)(1 + cos(pi it / total))."""
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * min(it, total) / total))


def teacher_temp_schedule(it, warmup_iters, start=0.04, end=0.07):
    """공식: 교사 온도를 start 에서 end 로 선형 워밍업한 뒤 end 유지(DINOv2 기본 0.04 -> 0.07)."""
    return end if it >= warmup_iters else start + (end - start) * it / warmup_iters


class DINOv2Toy(nn.Module):
    """학생(백본 + DINO 머리 + iBOT 머리)과 같은 구조의 EMA 교사. 교사는 requires_grad=False."""

    def __init__(self, n_proto=64, dim=64, depth=3, heads=4, patch=4, img_size=32, head_hidden=128, bottleneck=32):
        super().__init__()

        def make():
            return nn.ModuleDict(dict(
                backbone=TinyViT(img_size, patch, 3, dim, depth, heads, 2 * dim),
                dino_head=DINOHead(dim, n_proto, head_hidden, bottleneck),
                ibot_head=DINOHead(dim, n_proto, head_hidden, bottleneck)))  # 머리 분리(untying)

        self.student, self.teacher = make(), make()
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.dino_center = DINOCenter(n_proto)
        self.ibot_center = DINOCenter(n_proto)
        self.n_proto = n_proto

    def update_teacher(self, m):
        ema_update(self.teacher, self.student, m)

    def loss(self, global_crops, local_crops, masks, teacher_temp, centering="center", koleo_weight=0.1,
             ibot_weight=1.0, student_temp=0.1, sk_iters=3):
        """global_crops (2, B, 3, H, W), local_crops (L, B, 3, h, w), masks (2, B, N) bool(학생 전역 크롭에서 가릴 패치).
        돌려주는 것: (총 손실, 진단 dict)."""
        ng, b = global_crops.shape[:2]
        flat_g = global_crops.flatten(0, 1)
        flat_mask = masks.flatten(0, 1)

        with torch.no_grad():  # 교사: 가리지 않은 전역 크롭만 본다. 기울기 없음
            t_cls, t_patch = self.teacher["backbone"](flat_g)
            t_cls_logits = self.teacher["dino_head"](t_cls)  # (2B, K)
            p_cls = teacher_probs(t_cls_logits, teacher_temp, centering, self.dino_center, sk_iters)
            idx = flat_mask.nonzero(as_tuple=True)
            t_patch_logits = self.teacher["ibot_head"](t_patch[idx])  # (M, K) 가린 위치만
            p_patch = teacher_probs(t_patch_logits, teacher_temp, centering, self.ibot_center, sk_iters) \
                if t_patch_logits.shape[0] > 0 else t_patch_logits
            if centering == "center":  # 이번 스텝의 softmax 뒤에 중심을 갱신(공식과 같은 순서)
                self.dino_center.update(t_cls_logits)
                if t_patch_logits.shape[0] > 0:
                    self.ibot_center.update(t_patch_logits)
        p_cls = p_cls.view(ng, b, -1)

        s_cls_g, s_patch = self.student["backbone"](flat_g, mask=flat_mask)
        s_logits_g = self.student["dino_head"](s_cls_g).view(ng, b, -1)
        if local_crops.shape[0] > 0:  # 지역 크롭은 크기가 같으므로 한 번에 통과시킨다
            s_cls_l, _ = self.student["backbone"](local_crops.flatten(0, 1))
            s_logits_l = self.student["dino_head"](s_cls_l).view(local_crops.shape[0], b, -1)
        else:
            s_logits_l = global_crops.new_zeros(0, b, self.n_proto)

        # 교사 전역 크롭 j 는 학생의 다른 크롭과 짝을 짓는다(dino_cls_loss 안에서 i != j 로 처리)
        l_dino = dino_cls_loss(s_logits_g, s_logits_l, p_cls, student_temp)

        # iBOT: 학생 가린 패치 로짓을 (2B, N, K) 로 되돌려 같은 함수를 쓴다
        l_ibot = global_crops.new_zeros(())
        if ibot_weight > 0 and idx[0].numel() > 0:
            s_patch_logits = self.student["ibot_head"](s_patch[idx])
            full_s = s_patch_logits.new_zeros(flat_mask.shape[0], flat_mask.shape[1], self.n_proto)
            full_t = s_patch_logits.new_zeros(flat_mask.shape[0], flat_mask.shape[1], self.n_proto)
            full_s = full_s.index_put(idx, s_patch_logits)
            full_t = full_t.index_put(idx, p_patch)
            l_ibot = ibot_patch_loss(full_s, full_t, flat_mask, student_temp)

        l_koleo = global_crops.new_zeros(())
        if koleo_weight > 0:  # 같은 이미지의 두 전역 크롭 사이는 비교하지 않는다(각 크롭 묶음 안에서만)
            l_koleo = sum(koleo_loss(c) for c in s_cls_g.view(ng, b, -1)) / ng

        total = l_dino + ibot_weight * l_ibot + koleo_weight * l_koleo
        with torch.no_grad():
            ent = -(p_cls * torch.log(p_cls.clamp_min(1e-12))).sum(-1)  # (2, B) 표본별 엔트로피
            marg = p_cls.reshape(-1, p_cls.shape[-1]).mean(0)
            marg_ent = -(marg * torch.log(marg.clamp_min(1e-12))).sum()
        diag = dict(dino=float(l_dino.detach()), ibot=float(l_ibot.detach()), koleo=float(l_koleo.detach()),
                    teacher_ent=float(ent.mean()), marginal_ent=float(marg_ent))
        return total, diag


# ---------------------------------------------------------------- 합성 이미지와 멀티 크롭

SHAPES = ["circle", "square", "diamond", "cross", "ring", "triangle"]


def make_shapes(n, size=32, gen=None, device="cpu"):
    """도형(6종) 라벨, 무작위 색·크기·위치, 회색 잡음 배경의 합성 이미지 (n, 3, size, size), 라벨 (n,)."""
    g = dict(generator=gen, device=device)
    label = torch.randint(0, len(SHAPES), (n,), **g)
    r = 6 + 5 * torch.rand(n, **g)
    cx = r + 1 + (size - 2 - 2 * r) * torch.rand(n, **g)
    cy = r + 1 + (size - 2 - 2 * r) * torch.rand(n, **g)
    color = 0.3 + 0.7 * torch.rand(n, 3, **g)
    ys, xs = torch.meshgrid(torch.arange(size, device=device).float() + 0.5,
                            torch.arange(size, device=device).float() + 0.5, indexing="ij")
    dx = xs[None] - cx[:, None, None]
    dy = ys[None] - cy[:, None, None]
    rr = r[:, None, None]
    dist = (dx ** 2 + dy ** 2).sqrt()
    masks = torch.stack([
        dist <= rr,
        torch.maximum(dx.abs(), dy.abs()) <= 0.85 * rr,
        dx.abs() + dy.abs() <= 1.15 * rr,
        ((dx.abs() <= 0.3 * rr) & (dy.abs() <= rr)) | ((dy.abs() <= 0.3 * rr) & (dx.abs() <= rr)),
        (dist <= rr) & (dist >= 0.55 * rr),
        (dy >= -rr) & (dy <= rr) & (dx.abs() <= 0.5 * (dy + rr)),
    ], dim=1)
    m = masks[torch.arange(n, device=device), label].float()[:, None]
    bg = 0.1 + 0.05 * torch.randn(n, 3, size, size, **g)
    img = bg * (1 - m) + color[:, :, None, None] * m
    return img.clamp(0, 1), label


def random_resized_crop(x, out_size, scale, gen=None):
    """정사각 영역을 면적 비율 scale=(lo, hi) 에서 뽑아 out_size 로 bilinear 리샘플(무작위 좌우 반전 포함, 종횡비 고정).
    scale=(1, 1) 이고 반전이 없으면 항등(테스트)."""
    b = x.shape[0]
    dev = x.device
    area = scale[0] + (scale[1] - scale[0]) * torch.rand(b, generator=gen, device=dev)
    w = area.sqrt()
    tx = (1 - w) * (2 * torch.rand(b, generator=gen, device=dev) - 1)
    ty = (1 - w) * (2 * torch.rand(b, generator=gen, device=dev) - 1)
    flip = (torch.rand(b, generator=gen, device=dev) < 0.5).float() * 2 - 1
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 1, 1] = w
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, (b, x.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def color_augment(x, gen=None):
    """채널 순서를 무작위로 섞고(색 불변성을 강제), 밝기 0.6-1.4 배, 약한 잡음. 크롭마다 독립."""
    b = x.shape[0]
    perm = torch.rand(b, 3, generator=gen, device=x.device).argsort(dim=1)
    x = torch.gather(x, 1, perm[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3]))
    gain = 0.6 + 0.8 * torch.rand(b, 1, 1, 1, generator=gen, device=x.device)
    return (x * gain + 0.02 * torch.randn(x.shape, generator=gen, device=x.device)).clamp(0, 1)


def random_patch_masks(n_img, n_patch, prob=0.5, ratio=(0.1, 0.5), gen=None, device="cpu"):
    """iBOT 마스크: 이미지의 prob 비율에만 마스크를 만들고 가리는 비율은 ratio 구간에서 균등 추출(공식은 블록 마스킹, 여기는 무작위 패치)."""
    do = torch.rand(n_img, generator=gen, device=device) < prob
    frac = ratio[0] + (ratio[1] - ratio[0]) * torch.rand(n_img, generator=gen, device=device)
    k = (frac * n_patch).long() * do.long()
    order = torch.rand(n_img, n_patch, generator=gen, device=device).argsort(dim=1).argsort(dim=1)
    return order < k[:, None]
