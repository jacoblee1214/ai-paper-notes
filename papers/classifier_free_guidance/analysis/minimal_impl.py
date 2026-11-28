"""Classifier-free guidance(Ho & Salimans, arXiv:2207.12598)의 최소 구현.

학습은 하지 않는다. 신경망 대신 2차원 클래스 조건부 가우시안 혼합에서 닫힌 식으로 나오는
정확한 조건부 점수 p(z|c)와 무조건 점수 p(z)를 쓴다. 그래서 eps_c, eps_u가 모든 로그 SNR(lambda)에서
오차 없이 주어지고, CFG 식 (6)과 샘플러(알고리즘 2)의 성질만 남는다.

표기는 논문과 같다: z = alpha*x + sigma*eps, alpha^2 = sigmoid(lambda), eps* = -sigma * grad log p(z).
"""
import math

import torch

torch.set_default_dtype(torch.float64)
TWO_PI = 2.0 * math.pi


def alpha_sigma(lam):
    lam = torch.as_tensor(lam, dtype=torch.float64)
    a2 = torch.sigmoid(lam)
    return torch.sqrt(a2), torch.sqrt(1.0 - a2)


class ToyGMM:
    """클래스 c의 데이터 p(x|c) = sum_m wm[c,m] N(mu[c,m], std^2 I), 클래스 사전 p(c) = pi[c] (2차원)."""

    def __init__(self, mu, wm, pi, std):
        self.mu = torch.as_tensor(mu)          # (C, M, 2)
        self.wm = torch.as_tensor(wm)          # (C, M), 클래스 안에서 합 1
        self.pi = torch.as_tensor(pi)          # (C,)
        self.s2 = float(std) ** 2
        self.C, self.M = self.mu.shape[:2]

    def _comps(self, c):
        if c is None:
            return self.mu.reshape(-1, 2), torch.log(self.pi[:, None] * self.wm).reshape(-1)
        return self.mu[c], torch.log(self.wm[c])

    def log_p(self, z, alpha, sigma, c=None):
        """log p(z|c) (c=None이면 log p(z)). z = alpha x + sigma eps 의 주변 밀도."""
        mu, lw = self._comps(c)
        V = alpha ** 2 * self.s2 + sigma ** 2
        d2 = ((z[:, None, :] - alpha * mu[None]) ** 2).sum(-1)
        return torch.logsumexp(lw - d2 / (2 * V) - torch.log(TWO_PI * V), dim=1)

    def score(self, z, alpha, sigma, c=None):
        """grad_z log p(z|c)의 닫힌 식: sum_m r_m (alpha mu_m - z)/V."""
        mu, lw = self._comps(c)
        V = alpha ** 2 * self.s2 + sigma ** 2
        d2 = ((z[:, None, :] - alpha * mu[None]) ** 2).sum(-1)
        r = torch.softmax(lw - d2 / (2 * V), dim=1)
        return (r @ (alpha * mu) - z) / V

    def eps_c(self, z, lam, c):
        a, s = alpha_sigma(lam)
        return -s * self.score(z, a, s, c)

    def eps_u(self, z, lam):
        a, s = alpha_sigma(lam)
        return -s * self.score(z, a, s, None)

    def log_post(self, z, alpha, sigma):
        """log p(c|z) (C 열). 베이즈: log pi_c + log p(z|c) - log p(z)."""
        lp = torch.stack([self.log_p(z, alpha, sigma, c) for c in range(self.C)], dim=1)
        return torch.log(self.pi)[None] + lp - self.log_p(z, alpha, sigma)[:, None]

    def sample(self, c, n, gen):
        k = torch.multinomial(self.wm[c], n, replacement=True, generator=gen)
        return self.mu[c][k] + math.sqrt(self.s2) * torch.randn(n, 2, generator=gen)


def eps_guided(model, z, lam, c, w):
    """식 (6): (1+w) eps(z,c) - w eps(z). w는 논문 규약이고 gamma = 1+w 이면 eps_u + gamma (eps_c - eps_u)."""
    ec = model.eps_c(z, lam, c)
    if w == 0:
        return ec
    return (1.0 + w) * ec - w * model.eps_u(z, lam)


def lam_schedule(T, lmin=-20.0, lmax=20.0):
    """논문 2절의 lambda = -2 log tan(a u + b). u를 균등 간격으로 1에서 0까지 두어 lambda를 증가 순서로 만든다."""
    b = math.atan(math.exp(-lmax / 2))
    a = math.atan(math.exp(-lmin / 2)) - b
    u = torch.linspace(1.0, 0.0, T)
    return -2.0 * torch.log(torch.tan(a * u + b))


def _x_hat(z, a, s, eps):
    return (z - s * eps) / a


def sample_ancestral(eps_fn, n, T, gen, v=0.3, lmin=-20.0, lmax=20.0, z0=None):
    """알고리즘 2: 식 (3), (4)의 조상 표집. eps_fn(z, lam) -> 안내된 잡음 예측."""
    lams = lam_schedule(T, lmin, lmax)
    z = torch.randn(n, 2, generator=gen) if z0 is None else z0
    for t in range(T):
        lam = lams[t]
        a, s = alpha_sigma(lam)
        xt = _x_hat(z, a, s, eps_fn(z, lam))
        if t == T - 1:
            return xt
        lam2 = lams[t + 1]
        a2, s2 = alpha_sigma(lam2)
        e = torch.exp(lam - lam2)
        mean = e * (a2 / a) * z + (1 - e) * a2 * xt
        var_tilde = (1 - e) * s2 ** 2          # sigma~^2_{lam'|lam}
        var_fwd = (1 - e) * s ** 2             # sigma^2_{lam|lam'}
        var = var_tilde ** (1 - v) * var_fwd ** v
        z = mean + torch.sqrt(var) * torch.randn(n, 2, generator=gen)
    return z


def sample_ddim(eps_fn, n, T, gen, lmin=-20.0, lmax=20.0, z0=None):
    """결정적(eta=0) DDIM: z' = alpha' x~ + sigma' eps~. 연속 극한은 확률 흐름 ODE."""
    lams = lam_schedule(T, lmin, lmax)
    z = torch.randn(n, 2, generator=gen) if z0 is None else z0
    for t in range(T):
        lam = lams[t]
        a, s = alpha_sigma(lam)
        eps = eps_fn(z, lam)
        xt = _x_hat(z, a, s, eps)
        if t == T - 1:
            return xt
        a2, s2 = alpha_sigma(lams[t + 1])
        z = a2 * xt + s2 * eps
    return z


def guided_fn(model, c, w, lam_band=None):
    """lam_band=(lo, hi)이면 lo <= lambda <= hi에서만 안내한다(구간 안내, 이후 문헌). None이면 모든 lambda."""
    def f(z, lam):
        ww = w
        if lam_band is not None and not (lam_band[0] <= float(lam) <= lam_band[1]):
            ww = 0
        return eps_guided(model, z, lam, c, ww)
    return f


# ---------------------------------------------------------------- 정확한 기울인 목표 p(x|c) p(c|x)^w

class TiltedGrid:
    """t=0에서 p~(x|c) ∝ p(x|c)^(1+w) p(x)^(-w) = p(x|c) (p(c|x)/pi_c)^w 를 격자에서 정규화한다."""

    def __init__(self, model, c, w, R=7.0, G=280):
        self.R, self.G = R, G
        h = 2 * R / G
        ax = -R + h * (torch.arange(G) + 0.5)
        X, Y = torch.meshgrid(ax, ax, indexing="ij")
        self.pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
        self.h = h
        one, zero = torch.tensor(1.0), torch.tensor(0.0)
        logf = (1 + w) * model.log_p(self.pts, one, zero, c) - w * model.log_p(self.pts, one, zero, None)
        self.p = torch.softmax(logf, dim=0)    # 격자 칸 확률

    def moments(self):
        m = (self.p[:, None] * self.pts).sum(0)
        d = self.pts - m
        cov = (self.p[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
        return m, cov

    def sample(self, n, gen):
        k = torch.multinomial(self.p, n, replacement=True, generator=gen)
        return self.pts[k] + self.h * (torch.rand(n, 2, generator=gen) - 0.5)


def oracle_tilted_eps_fn(grid, chunk=500):
    """일관된(consistent) 기울임: 기울인 목표 p~_0를 순방향 확산시킨 주변 q_lambda의 정확한 점수로 만든 eps.
    CFG의 식 (6)과 달리 모든 lambda에서 확산된 분포의 점수다. 격자 합으로 계산한다."""
    pts, logq0 = grid.pts, torch.log(grid.p + 1e-300)

    def f(z, lam):
        a, s = alpha_sigma(lam)
        out = []
        for i in range(0, z.shape[0], chunk):
            zc = z[i:i + chunk]
            d2 = (zc ** 2).sum(1, keepdim=True) - 2 * a * (zc @ pts.T) + a ** 2 * (pts ** 2).sum(1)[None]
            r = torch.softmax(logq0[None] - d2 / (2 * s ** 2), dim=1)
            score = (r @ (a * pts) - zc) / s ** 2
            out.append(-s * score)
        return torch.cat(out)
    return f


# ---------------------------------------------------------------- 지표

def sliced_w1(x, y, dirs=64, seed=123):
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(2, dirs, generator=g)
    th = th / th.norm(dim=0, keepdim=True)
    px, _ = torch.sort(x @ th, dim=0)
    py, _ = torch.sort(y @ th, dim=0)
    return (px - py).abs().mean().item()


def knn_precision_recall(real, fake, k=3):
    """Kynkaanniemi 외 2019: 각 점의 k번째 이웃 반경으로 다양체를 근사한다."""
    def radii(a):
        d = torch.cdist(a, a)
        return torch.kthvalue(d, k + 1, dim=1).values
    rr, rf = radii(real), radii(fake)
    d = torch.cdist(fake, real)
    precision = (d <= rr[None, :]).any(1).double().mean().item()
    d2 = torch.cdist(real, fake)
    recall = (d2 <= rf[None, :]).any(1).double().mean().item()
    return precision, recall


def frechet_gauss(x, y):
    """2차원 점들의 가우시안 적합 사이 Frechet 거리(FID의 2차원 모사. 혼합이라 거친 지표다)."""
    mx, my = x.mean(0), y.mean(0)
    cx, cy = torch.cov(x.T), torch.cov(y.T)
    ev, U = torch.linalg.eigh(cx)
    sx = U @ torch.diag(torch.sqrt(ev.clamp_min(0))) @ U.T
    mid = sx @ cy @ sx
    ev2 = torch.linalg.eigvalsh((mid + mid.T) / 2).clamp_min(0)
    return ((mx - my) ** 2).sum().item() + torch.trace(cx).item() + torch.trace(cy).item() - 2 * torch.sqrt(ev2).sum().item()


def inception_score_toy(model, x_by_class):
    """IS의 2차원 모사: exp E_x KL(p(y|x) || p(y)). p(y|x)는 정확한 베이즈 사후, p(y)는 생성 표본 평균."""
    xs = torch.cat(x_by_class)
    one, zero = torch.tensor(1.0), torch.tensor(0.0)
    post = torch.exp(model.log_post(xs, one, zero))
    py = post.mean(0)
    kl = (post * (torch.log(post + 1e-300) - torch.log(py + 1e-300))).sum(1).mean()
    return math.exp(kl.item())


def default_toy():
    """3 클래스, 클래스마다 2개 성분(2봉), 사전 (0.5, 0.3, 0.2). 이웃 클래스끼리 가장자리가 겹친다."""
    R, off, C = 2.0, 0.5, 3
    mu = torch.zeros(C, 2, 2)
    for k in range(C):
        th = 2 * math.pi * k / C
        for j, sgn in enumerate((-1, 1)):
            mu[k, j] = R * torch.tensor([math.cos(th + sgn * off), math.sin(th + sgn * off)])
    wm = torch.tensor([[0.5, 0.5], [0.7, 0.3], [0.5, 0.5]])
    return ToyGMM(mu, wm, torch.tensor([0.5, 0.3, 0.2]), std=0.8)


# ---------------------------------------------------------------- 가우시안 쌍의 닫힌 식

class GaussPair:
    """조건부 N(mc, vc I), 무조건 N(mu, vu I) (2차원). 확산해도 가우시안이라 eps_c, eps_u와 ODE가 닫힌 식이다.
    두 분포는 서로 일관된 혼합일 필요가 없다. CFG 식 (6)이 두 점수에만 의존하기 때문이다."""

    def __init__(self, mc, vc, mu, vu):
        self.mc, self.vc = torch.as_tensor(mc, dtype=torch.float64), float(vc)
        self.mu, self.vu = torch.as_tensor(mu, dtype=torch.float64), float(vu)

    def eps_c(self, z, lam, c=None):
        a, s = alpha_sigma(lam)
        return s * (z - a * self.mc) / (a ** 2 * self.vc + s ** 2)

    def eps_u(self, z, lam):
        a, s = alpha_sigma(lam)
        return s * (z - a * self.mu) / (a ** 2 * self.vu + s ** 2)

    def tilted_moments(self, w):
        """t=0의 p(x|c)^(1+w) p(x)^(-w): 정밀도 (1+w)/vc - w/vu 가 양수일 때 가우시안."""
        prec = (1 + w) / self.vc - w / self.vu
        if prec <= 0:
            return None
        mean = ((1 + w) * self.mc / self.vc - w * self.mu / self.vu) / prec
        return mean, 1.0 / prec

    def ode_moments(self, w, lmin=-20.0, lmax=20.0):
        """안내된 확률 흐름 ODE dz/dlam = (sigma^2/2) z - (sigma/2) eps~ 의 끝점(x~ = z/alpha at lmax) 평균·분산.
        선형 ODE라 z의 평균 m과 분산 P가 닫힌 상미분방정식을 따른다. z(lmin) ~ N(0, I)."""
        from scipy.integrate import solve_ivp

        mc, mu_ = self.mc, self.mu

        def rhs(lam, y):
            a2 = 1 / (1 + math.exp(-lam)); s2 = 1 - a2; a = math.sqrt(a2)
            Vc, Vu = a2 * self.vc + s2, a2 * self.vu + s2
            coef = 0.5 * s2 * (1 - (1 + w) / Vc + w / Vu)          # d z/d lam = coef z + b
            b = 0.5 * s2 * ((1 + w) * a * mc / Vc - w * a * mu_ / Vu)
            return [coef * y[0] + float(b[0]), coef * y[1] + float(b[1]), 2 * coef * y[2]]
        sol = solve_ivp(rhs, (lmin, lmax), [0.0, 0.0, 1.0], method="DOP853", rtol=1e-11, atol=1e-13)
        a = 1 / math.sqrt(1 + math.exp(-lmax))
        m = torch.tensor(sol.y[:2, -1]) / a
        return m, sol.y[2, -1] / a ** 2


# ---------------------------------------------------------------- 안내된 점수는 확산 분포의 점수인가

def fp_residual(model, z, lam, c, w):
    """안내된 장 s~ = (1+w) grad log p_lam(z|c) - w grad log p_lam(z) 가 어떤 분포족 q_lam의 점수이고 그 족이
    순방향 VP 확산(포커-플랑크)을 따르려면, log q~ = (1+w) log p_c - w log p (정규화 상수 제외)가
    d_lam log q~ + (sigma^2/2) (d + z.g + lap + |g|^2) = (z에 무관한 상수)를 만족해야 한다.
    z에 따른 이 값의 표준편차(잔차)를 돌려준다. w=0이면 0이어야 한다."""
    z = z.clone().requires_grad_(True)
    lam = float(lam)
    a, s = alpha_sigma(lam)

    def logq(zz, l):
        aa, ss = alpha_sigma(l)
        return (1 + w) * model.log_p(zz, aa, ss, c) - w * model.log_p(zz, aa, ss, None)
    lq = logq(z, lam)
    g = torch.autograd.grad(lq.sum(), z, create_graph=True)[0]
    h = 1e-5   # lambda 미분은 중심차분(점마다 따로 필요해서). 오차 ~1e-10
    dlam = (logq(z.detach(), lam + h) - logq(z.detach(), lam - h)) / (2 * h)
    lap = sum(torch.autograd.grad(g[:, i].sum(), z, create_graph=True)[0][:, i] for i in range(2))
    R = dlam + (s ** 2 / 2) * (2 + (z * g).sum(1) + lap + (g ** 2).sum(1))
    return R.detach().std().item(), R.detach()
