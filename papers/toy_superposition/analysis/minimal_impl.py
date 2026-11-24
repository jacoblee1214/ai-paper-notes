"""Toy Models of Superposition (Elhage et al., 2022) 최소 구현.

논문의 토이 모델을 PyTorch로 직접 짠 것이다(공식 노트북 코드는 복사하지 않았고 정적으로 읽기만 했다).

모델   x' = ReLU(W^T W x + b)   (relu=True)  /  x' = W^T W x + b  (relu=False, 선형 모델)
        W: [m, n] (n개 특징, m < n 은닉 차원), 특징 i의 방향은 열 W_i
데이터 x_i = 0 (확률 S), 아니면 U[0,1];  손실 L = E_x sum_i I_i (x_i - x'_i)^2
여러 모델(인스턴스)을 한 번에 학습해 희소도·중요도를 훑는다.

구성
  1. 데이터·모델·학습: init_params, generate_batch, forward, train
  2. 측정: feature_norms, interference, dimensionality, count_represented
  3. 닫힌 형식: 선형 모델 최적 손실, 1-희소 항, n=2·m=1 세 가지 해의 손실(구적법)
  4. 2특징·1차원 단계도의 모집단 손실 학습
"""
from __future__ import annotations

import math

import numpy as np
import torch

# --------------------------------------------------------------------------
# 1. 데이터, 모델, 학습
# --------------------------------------------------------------------------


def importance_curve(n: int, base: float) -> torch.Tensor:
    """I_i = base**i  (i = 0..n-1). 논문 §Basic Results: 0.7**i (n=20), 0.9**i (n=80)."""
    return base ** torch.arange(n, dtype=torch.float32)


def init_params(n_inst: int, n: int, m: int, gen: torch.Generator, device="cpu"):
    """W: [inst, m, n] (xavier normal과 같은 표준편차 sqrt(2/(m+n))), b: [inst, n] (0)."""
    std = math.sqrt(2.0 / (m + n))
    W = (torch.randn(n_inst, m, n, generator=gen) * std).to(device).requires_grad_()
    b = torch.zeros(n_inst, n, device=device).requires_grad_()
    return W, b


def generate_batch(batch: int, density: torch.Tensor, n: int, gen: torch.Generator, device="cpu"):
    """x: [batch, inst, n]. 특징은 확률 density(=1-S)로 U[0,1], 아니면 0. density: [inst]."""
    inst = density.shape[0]
    u = torch.rand(batch, inst, n, generator=gen)
    keep = torch.rand(batch, inst, n, generator=gen) < density.view(1, inst, 1)
    return (u * keep).to(device)


def forward(W: torch.Tensor, b: torch.Tensor, x: torch.Tensor, relu: bool = True) -> torch.Tensor:
    h = torch.einsum("bin,imn->bim", x, W)
    out = torch.einsum("bim,imn->bin", h, W) + b
    return torch.relu(out) if relu else out


def train(
    n: int,
    m: int,
    density,
    importance: torch.Tensor,
    *,
    relu: bool = True,
    steps: int = 10_000,
    batch: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
    cosine: bool = False,
    device: str = "cpu",
):
    """density(=1-S) 길이 n_inst 텐서마다 독립 모델 하나. 반환: dict(W, b, loss(마지막 200스텝 평균), config).

    옵티마이저 AdamW(기본 weight decay 0.01), lr 일정(cosine=True면 코사인 감쇠), 배치 1024.
    손실은 인스턴스별 sum_i I_i (x_i - x'_i)^2 의 배치 평균이고 인스턴스 합을 역전파한다.
    """
    density = torch.as_tensor(density, dtype=torch.float32)
    n_inst = density.shape[0]
    gen = torch.Generator().manual_seed(seed)
    W, b = init_params(n_inst, n, m, gen, device)
    imp = importance.to(device).view(1, 1, n)
    opt = torch.optim.AdamW([W, b], lr=lr)
    tail = []
    for step in range(steps):
        if cosine:
            for g in opt.param_groups:
                g["lr"] = lr * math.cos(0.5 * math.pi * step / max(steps - 1, 1))
        x = generate_batch(batch, density, n, gen, device)
        out = forward(W, b, x, relu)
        per_inst = (imp * (x - out) ** 2).sum(-1).mean(0)  # [inst]
        opt.zero_grad(set_to_none=True)
        per_inst.sum().backward()
        opt.step()
        if step >= steps - 200:
            tail.append(per_inst.detach().cpu())
    loss = torch.stack(tail).mean(0)
    return dict(
        W=W.detach().cpu(),
        b=b.detach().cpu(),
        loss=loss,
        config=dict(
            n=n, m=m, steps=steps, batch=batch, lr=lr, seed=seed, optimizer="AdamW(wd=0.01)",
            lr_schedule="cosine" if cosine else "constant", relu=relu,
            density=[float(d) for d in density], init="normal std sqrt(2/(m+n)), b=0",
        ),
    )


# --------------------------------------------------------------------------
# 2. 측정
# --------------------------------------------------------------------------


def feature_norms(W: torch.Tensor) -> torch.Tensor:
    """||W_i||. W: [..., m, n] -> [..., n]."""
    return W.norm(dim=-2)


def interference(W: torch.Tensor) -> torch.Tensor:
    """sum_{j != i} (W^_i . W_j)^2. W: [..., m, n] -> [..., n]. 안 배운 특징(||W_i||~0)은 0."""
    norms = W.norm(dim=-2, keepdim=True)
    unit = W / norms.clamp_min(1e-6)
    dots = torch.einsum("...mi,...mj->...ij", unit, W)
    sq = dots ** 2
    diag = torch.diagonal(sq, dim1=-2, dim2=-1)
    return sq.sum(-1) - diag


def dimensionality(W: torch.Tensor) -> torch.Tensor:
    """D_i = ||W_i||^2 / sum_j (W^_i . W_j)^2  (논문 §Feature Dimensionality; j=i 포함).

    ||W_i||^4 / sum_j (W_i . W_j)^2 와 같다. W의 전체 크기에 불변이고, 안 배운 특징은 0.
    """
    norms = W.norm(dim=-2)
    unit = W / norms.unsqueeze(-2).clamp_min(1e-6)
    denom = (torch.einsum("...mi,...mj->...ij", unit, W) ** 2).sum(-1)
    return torch.where(norms > 1e-3, norms ** 2 / denom.clamp_min(1e-12), torch.zeros_like(norms))


def count_represented(W: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """||W_i||^2 > thresh 인 특징 수. (표현된 특징은 ||W_i||^2 ~ 1, 아닌 특징은 ~ 0)"""
    return (feature_norms(W) ** 2 > thresh).sum(-1)


def dims_per_feature(W: torch.Tensor) -> torch.Tensor:
    """D* = m / ||W||_F^2 (논문 §Uniform Superposition)."""
    return W.shape[-2] / (W ** 2).sum(dim=(-2, -1))


# --------------------------------------------------------------------------
# 3. 닫힌 형식
# --------------------------------------------------------------------------


def uniform_stats(density: float):
    """x_i = 0 (확률 S=1-density), 아니면 U[0,1]: 평균 mu, 분산 sigma2."""
    mu = density / 2
    return mu, density / 3 - mu ** 2


def linear_optimal_loss(importance: np.ndarray, m: int, density: float) -> float:
    """선형 모델 x' = W^T W x + b 의 최적 손실 = sigma^2 * (중요도 하위 n-m개의 합).

    유도(03 §1): b를 최적화하면 잔차는 (I-M)(x-mu), 공분산 sigma^2 I 이므로
    L = sigma^2 tr(D (I-M)^2),  M = W^T W (rank <= m). 고유분해로 L/sigma^2 = tr D - sum_k (2 l_k - l_k^2) u_k^T D u_k
    >= tr D - (상위 m개 I의 합). 등호는 M = 상위 m개 특징의 좌표 투영.
    """
    _, s2 = uniform_stats(density)
    imp = np.sort(np.asarray(importance, dtype=float))[::-1]
    return float(s2 * imp[m:].sum())


def relu_integral_sq(c: float, b: float) -> float:
    """int_0^1 ReLU(c x + b)^2 dx 의 닫힌 형식. 1-희소 입력에서 특징 j가 받는 간섭 손실의 핵심 항."""
    if c == 0:
        return max(b, 0.0) ** 2
    lo, hi = 0.0, 1.0
    if c > 0:
        lo = max(0.0, -b / c)
    else:
        hi = min(1.0, -b / c)
    if hi <= lo:
        return 0.0
    f = lambda x: c * x + b
    return (f(hi) ** 3 - f(lo) ** 3) / (3 * c)


def one_sparse_loss(W: np.ndarray, b: np.ndarray, importance: np.ndarray) -> float:
    """L_1 = E_i[ 입력이 특징 i 하나만 켜졌을 때 손실 ] * n = sum_i (특징 i 항 + 간섭 항), x_i ~ U[0,1].

    특징 i 항: I_i int (x - ReLU(a_i x + b_i))^2 dx,  a_i = ||W_i||^2 (구적법으로 계산)
    간섭 항 : sum_{j != i} I_j int ReLU((W_j.W_i) x + b_j)^2 dx   (닫힌 형식)
    반환은 i에 대해 합한 값(i당 평균에 n을 곱한 것).
    """
    M = W.T @ W
    n = M.shape[0]
    xs = (np.arange(4000) + 0.5) / 4000
    total = 0.0
    for i in range(n):
        own = importance[i] * np.mean((xs - np.maximum(M[i, i] * xs + b[i], 0.0)) ** 2)
        inter = sum(importance[j] * relu_integral_sq(M[j, i], b[j]) for j in range(n) if j != i)
        total += own + inter
    return total


# n=2, m=1: 세 가지 해. W = [w1, w2] (1x2). 모집단 손실은 구적법(중점)으로 계산한다.
_G1 = 600   # 1차원 격자
_G2 = 120   # 2차원 격자(각 축)


def _grids(device="cpu", dtype=torch.float64, sizes=None):
    n1, n2 = sizes if sizes is not None else (_G1, _G2)
    g1 = ((torch.arange(n1, device=device, dtype=dtype) + 0.5) / n1)
    g2 = ((torch.arange(n2, device=device, dtype=dtype) + 0.5) / n2)
    xx, yy = torch.meshgrid(g2, g2, indexing="ij")
    return g1, xx.reshape(-1), yy.reshape(-1)


def population_loss_2feat(w1, w2, b1, b2, imp2, density, grids=None):
    """n=2, m=1 모델의 정확한(구적법) 모집단 손실. 중요도 (1, imp2), 모든 입력은 확률 density로 켜짐.

    텐서 인자는 같은 모양(배치)이어야 한다. 반환: 같은 모양.
    4가지 희소 패턴의 혼합: (1-S)^2 L_both + S(1-S)(L_only1 + L_only2) + S^2 L_none.
    """
    g1, xx, yy = grids if grids is not None else _grids(w1.device, w1.dtype)
    d = density
    s = 1 - d
    m11, m12, m22 = w1 * w1, w1 * w2, w2 * w2
    sh = (-1,) + (1,) * w1.dim()

    def c(t):
        return t.unsqueeze(0)

    # 둘 다 켜짐
    X, Y = xx.view(sh), yy.view(sh)
    o1 = torch.relu(c(m11) * X + c(m12) * Y + c(b1))
    o2 = torch.relu(c(m12) * X + c(m22) * Y + c(b2))
    l_both = ((X - o1) ** 2 + imp2 * (Y - o2) ** 2).mean(0)
    # 특징 1만 켜짐
    X = g1.view(sh)
    o1 = torch.relu(c(m11) * X + c(b1))
    o2 = torch.relu(c(m12) * X + c(b2))
    l_only1 = ((X - o1) ** 2 + imp2 * o2 ** 2).mean(0)
    # 특징 2만 켜짐
    o1 = torch.relu(c(m12) * X + c(b1))
    o2 = torch.relu(c(m22) * X + c(b2))
    l_only2 = (o1 ** 2 + imp2 * (X - o2) ** 2).mean(0)
    l_none = torch.relu(b1) ** 2 + imp2 * torch.relu(b2) ** 2
    return d * d * l_both + d * s * (l_only1 + l_only2) + s * s * l_none


def _minimize_b(w, imp2, density, grids, steps=300):
    """고정한 W=(w1,w2)에서 b=(b1,b2)를 Adam으로 최소화(여러 초기값 중 최소). 반환 (loss, b1, b2)."""
    w1 = torch.full_like(density, float(w[0]))
    w2 = torch.full_like(density, float(w[1]))
    best = None
    for init in ((0.0, 0.0), (0.2, 0.2), (-0.2, -0.2), (0.4, 0.0), (0.0, 0.4)):
        b = torch.zeros((2,) + tuple(density.shape), dtype=density.dtype)
        b[0], b[1] = init
        b.requires_grad_()
        opt = torch.optim.Adam([b], lr=0.02)
        for _ in range(steps):
            loss = population_loss_2feat(w1, w2, b[0], b[1], imp2, density, grids)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        with torch.no_grad():
            loss = population_loss_2feat(w1, w2, b[0], b[1], imp2, density, grids)
        if best is None:
            best = (loss.clone(), b[0].detach().clone(), b[1].detach().clone())
        else:
            better = loss < best[0]
            best = (torch.where(better, loss, best[0]), torch.where(better, b[0].detach(), best[1]),
                    torch.where(better, b[1].detach(), best[2]))
    return best


def three_config_losses(imp2: torch.Tensor, density: torch.Tensor, optimize_bias: bool = True, quad=None):
    """'토이의 토이': W=[1,0](특징 2 버림), [0,1](특징 1 버림), [1,-1](대척점 superposition)의 손실.

    imp2, density: 같은 모양 텐서(float64). optimize_bias=False면 b=0(대척점) / A,B는 닫힌 형식.
    반환 dict(A=, B=, C=) (각각 같은 모양 손실).
    """
    sigma2 = density / 3 - (density / 2) ** 2
    A = imp2 * sigma2          # 특징 1 전담: 특징 2는 상수 E[x2]로 출력
    B = 1.0 * sigma2           # 특징 2 전담
    if optimize_bias:
        grids = _grids(dtype=density.dtype, sizes=quad)
        C = _minimize_b((1.0, -1.0), imp2, density, grids)[0]
    else:
        C = density ** 2 * (1 + imp2) / 6   # b = 0 닫힌 형식(E[min(x1,x2)^2] = 1/6)
    return dict(A=A, B=B, C=C)


def b0_boundary_density(imp2: float) -> float:
    """b=0 근사의 상전이 경계 density* (이보다 density가 낮으면 대척점이 최선).

    A: r sigma^2,  B: sigma^2,  C: d^2 (1+r)/6, sigma^2 = d/3 - d^2/4.
    r <= 1: C < A  <=>  d < 4 r / (2 + 5 r);   r >= 1: C < B  <=>  d < 4 / (5 + 2 r).
    """
    r = imp2
    return 4 * r / (2 + 5 * r) if r <= 1 else 4 / (5 + 2 * r)


# --------------------------------------------------------------------------
# 4. 2특징·1차원 단계도: 모집단 손실을 직접 최소화
# --------------------------------------------------------------------------


def learn_2feat_phase(imp2_grid, density_grid, n_init: int = 8, steps: int = 1500, lr: float = 0.02, seed: int = 0,
                      quad=(256, 64)):
    """각 (imp2, density) 칸에서 (w1,w2,b1,b2)를 n_init개 초기값으로 Adam 최소화해 최소 손실 해를 고른다.

    반환: dict(w=[R,D,2], b=[R,D,2], loss=[R,D]) (numpy). 논문은 10개 학습 후 최악 하나를 버리고 평균 — 우리는 최소 손실을 쓴다.
    """
    gen = torch.Generator().manual_seed(seed)
    R, D = len(imp2_grid), len(density_grid)
    r = torch.tensor(imp2_grid, dtype=torch.float64).view(R, 1, 1).expand(R, D, n_init).contiguous()
    d = torch.tensor(density_grid, dtype=torch.float64).view(1, D, 1).expand(R, D, n_init).contiguous()
    w = (torch.randn(2, R, D, n_init, generator=gen, dtype=torch.float64) * 0.7).requires_grad_()
    b = torch.zeros(2, R, D, n_init, dtype=torch.float64).requires_grad_()
    grids = _grids(dtype=torch.float64, sizes=quad)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        loss = population_loss_2feat(w[0], w[1], b[0], b[1], r, d, grids)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        loss = population_loss_2feat(w[0], w[1], b[0], b[1], r, d, grids)
        idx = loss.argmin(-1, keepdim=True)
        pick = lambda t: t.gather(-1, idx).squeeze(-1)
        return dict(
            w=np.stack([pick(w[0]).numpy(), pick(w[1]).numpy()], -1),
            b=np.stack([pick(b[0]).numpy(), pick(b[1]).numpy()], -1),
            loss=pick(loss).numpy(),
            config=dict(n_init=n_init, steps=steps, lr=lr, seed=seed, optimizer="Adam", quadrature=list(quad)),
        )


def classify_2feat(w: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """w: [..., 2] (1x2 행렬 W의 두 성분). ||W_i||^2 = w_i^2 > thresh 로 표현 여부 판정.
    0: 특징 1만, 1: 둘 다(superposition), 2: 특징 2만, 3: 둘 다 안 배움."""
    a = w[..., 0] ** 2 > thresh
    c = w[..., 1] ** 2 > thresh
    out = np.full(a.shape, 3)
    out[a & ~c] = 0
    out[a & c] = 1
    out[~a & c] = 2
    return out


def classify_theory(A, B, C) -> np.ndarray:
    """세 해의 손실 중 최소: 0: A(특징 1만), 1: C(둘 다 겹침), 2: B(특징 2만)."""
    stack = np.stack([A, C, B], -1)
    return stack.argmin(-1)
