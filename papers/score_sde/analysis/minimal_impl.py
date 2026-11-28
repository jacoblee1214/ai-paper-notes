"""Score SDE (Song et al., ICLR 2021) 최소 구현: 학습 없이 가우시안 혼합의 닫힌 식 점수로 확인한다.

학습한 신경망 대신 데이터가 가우시안 혼합이면 모든 시각 t의 p_t와 점수가 닫힌 식이라는 점을 쓴다.
  p_0t(x(t)|x(0)) = N(s(t) x(0), sigma(t)^2 I)   (논문 식 29-31, 33-34)
  p_t = sum_k w_k N(s mu_k, s^2 Sigma_k + sigma^2 I)
그래서 SDE 해법(예측기·교정기), 확률 흐름 ODE, 정확한 우도, 조건부 생성을 점수 오차 없이 검사할 수 있다.

구성
  GMM, PerturbedGMM       데이터 분포와 p_t의 점수·헤시안·후사 평균
  VE, VP, SubVP, LinearPath  섭동 핵 (s, sigma)와 거기서 유도한 f, g^2
  run_sampler             예측기(역확산, 조상, 오일러-마루야마, 확률 흐름) + 랑주뱅 교정기 (알고리즘 1-5)
  rk4_pf_sample, encode   확률 흐름 ODE (RK4 고정 스텝)
  log_density             순간 변수 변환 (식 39) 정확한 대각합과 허친슨 추정
  실험 함수와 main()      결과를 minimal_impl_results.json에 저장

실행: python minimal_impl.py   (CPU, 수 분, 메모리 수백 MB)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

# ---------------------------------------------------------------- 데이터 분포


class GMM:
    """d차원 가우시안 혼합 sum_k w_k N(mu_k, Sigma_k)."""

    def __init__(self, w, mu, cov):
        self.w = torch.as_tensor(w, dtype=torch.float64)
        self.mu = torch.as_tensor(mu, dtype=torch.float64)
        self.cov = torch.as_tensor(cov, dtype=torch.float64)
        self.K, self.d = self.mu.shape

    def sample(self, n, g, return_k=False):
        k = torch.multinomial(self.w, n, replacement=True, generator=g)
        L = torch.linalg.cholesky(self.cov)
        z = torch.randn(n, self.d, generator=g)
        x = self.mu[k] + torch.einsum("nij,nj->ni", L[k], z)
        return (x, k) if return_k else x

    def logpdf(self, x):
        return PerturbedGMM(self, VP()).logp(x, 0.0, exact_t0=True)


def default_gmm():
    """모드 세 개, 가중치 0.5/0.3/0.2, 회전된 비등방 공분산 (2차원)."""
    def rot(a, s1, s2):
        c, s = math.cos(a), math.sin(a)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
        return R @ torch.diag(torch.tensor([s1 ** 2, s2 ** 2], dtype=torch.float64)) @ R.T
    mu = [[-2.5, -1.0], [2.5, -1.5], [0.0, 3.0]]
    cov = torch.stack([rot(0.5, 0.6, 0.3), rot(-0.8, 0.5, 0.35), rot(1.2, 0.7, 0.25)])
    return GMM([0.5, 0.3, 0.2], mu, cov)


# ---------------------------------------------------------------- SDE (섭동 핵에서 유도)


class KernelSDE:
    """p_0t(x|x0) = N(s(t) x0, sigma(t)^2 I)인 아핀 SDE  dx = f(t) x dt + g(t) dw.

    섭동 핵에서 f = s'/s, g^2 = d(sigma^2)/dt - 2 f sigma^2 로 유도한다(03절 1.2).
    """
    T = 1.0

    def f_coef(self, t):
        return self.ds(t) / self.s(t)

    def g2_from_kernel(self, t):
        sg = self.sig(t)
        return 2 * sg * self.dsig(t) - 2 * self.f_coef(t) * sg ** 2

    def g2(self, t):
        return self.g2_from_kernel(t)

    prior_std = 1.0


class VE(KernelSDE):
    """식 9, 30, 31. sigma(t) = smin (smax/smin)^t."""
    name = "VE"

    def __init__(self, smin=0.01, smax=10.0):
        self.smin, self.smax = smin, smax
        self.lr = math.log(smax / smin)
        self.prior_std = smax

    def s(self, t): return 1.0
    def ds(self, t): return 0.0
    def sig(self, t): return self.smin * math.exp(self.lr * t)
    def dsig(self, t): return self.sig(t) * self.lr
    def f_coef(self, t): return 0.0
    def g2(self, t): return 2 * self.lr * self.sig(t) ** 2


class VP(KernelSDE):
    """식 11, 32, 33. beta(t) = bmin + t (bmax - bmin)."""
    name = "VP"

    def __init__(self, bmin=0.1, bmax=20.0):
        self.bmin, self.bmax = bmin, bmax

    def beta(self, t): return self.bmin + t * (self.bmax - self.bmin)
    def B(self, t): return self.bmin * t + 0.5 * (self.bmax - self.bmin) * t * t
    def s(self, t): return math.exp(-0.5 * self.B(t))
    def ds(self, t): return -0.5 * self.beta(t) * self.s(t)
    def sig(self, t): return math.sqrt(1.0 - math.exp(-self.B(t)))
    def dsig(self, t): return self.beta(t) * math.exp(-self.B(t)) / (2 * self.sig(t))
    def g2(self, t): return self.beta(t)


class SubVP(VP):
    """식 12, 27, 34. 같은 beta, 표준편차 1 - exp(-B)."""
    name = "subVP"

    def sig(self, t): return 1.0 - math.exp(-self.B(t))
    def dsig(self, t): return self.beta(t) * math.exp(-self.B(t))
    def g2(self, t): return self.beta(t) * (1.0 - math.exp(-2 * self.B(t)))


class LinearPath(KernelSDE):
    """정류 흐름/flow matching의 직선 경로 x_t = (1-t) x0 + t eps (s=1-t, sigma=t). f와 g^2는 t=1에서 발산한다."""
    name = "linear"

    def s(self, t): return 1.0 - t
    def ds(self, t): return -1.0
    def sig(self, t): return t
    def dsig(self, t): return 1.0


# ---------------------------------------------------------------- p_t의 닫힌 식


class PerturbedGMM:
    """p_t(x) = sum_k w_k N(x; s mu_k, s^2 Sigma_k + sigma^2 I). t는 스칼라."""

    def __init__(self, gmm, sde):
        self.gmm, self.sde = gmm, sde

    def _parts(self, x, t, exact_t0=False):
        s = 1.0 if exact_t0 else self.sde.s(t)
        var = 0.0 if exact_t0 else self.sde.sig(t) ** 2
        g = self.gmm
        eye = torch.eye(g.d)
        C = s * s * g.cov + var * eye          # [K,d,d]
        Ci = torch.linalg.inv(C)
        m = s * g.mu                           # [K,d]
        diff = x[:, None, :] - m[None]         # [n,K,d]
        Cid = torch.einsum("kij,nkj->nki", Ci, diff)
        maha = (diff * Cid).sum(-1)
        logdet = torch.linalg.slogdet(C)[1]
        logN = -0.5 * (maha + logdet[None] + g.d * math.log(2 * math.pi))
        lj = torch.log(g.w)[None] + logN
        return lj, Cid, Ci, s, var

    def logp(self, x, t, exact_t0=False):
        lj = self._parts(x, t, exact_t0)[0]
        return torch.logsumexp(lj, dim=1)

    def score(self, x, t):
        lj, Cid, *_ = self._parts(x, t)
        r = torch.softmax(lj, dim=1)
        return -(r[..., None] * Cid).sum(1)

    def score_and_hess(self, x, t):
        lj, Cid, Ci, *_ = self._parts(x, t)
        r = torch.softmax(lj, dim=1)
        gk = -Cid                                   # grad log N_k = -C^-1 (x-m_k)
        S = (r[..., None] * gk).sum(1)
        H = torch.einsum("nk,nki,nkj->nij", r, gk, gk) - torch.einsum("ni,nj->nij", S, S) \
            - torch.einsum("nk,kij->nij", r, Ci)
        return S, H

    def hess_trace(self, x, t):
        return torch.diagonal(self.score_and_hess(x, t)[1], dim1=1, dim2=2).sum(-1)

    def x0_mean(self, x, t):
        """E[x(0) | x(t)=x]: 성분별 사후 평균의 책임도 가중합 (트위디 식과는 독립인 닫힌 식)."""
        lj, Cid, Ci, s, var = self._parts(x, t)
        r = torch.softmax(lj, dim=1)
        g = self.gmm
        # x0 | x, k  ~ mu_k + s Sigma_k C_k^-1 (x - s mu_k)
        post = g.mu[None] + s * torch.einsum("kij,nkj->nki", g.cov, Cid)
        return (r[..., None] * post).sum(1)

    def sample_t(self, n, t, g):
        x0 = self.gmm.sample(n, g)
        return self.sde.s(t) * x0 + self.sde.sig(t) * torch.randn(n, self.gmm.d, generator=g)

    def pf_velocity(self, x, t):
        """확률 흐름 ODE의 속도 f x - g^2/2 * score (식 13). s>0, sigma>0 인 구간."""
        return self.sde.f_coef(t) * x - 0.5 * self.sde.g2(t) * self.score(x, t)

    def fm_velocity(self, x, t):
        """flow matching 형태의 속도 s' E[x0|x] + sigma' E[eps|x], E[eps|x] = -sigma * score. s=0이어도 정의된다."""
        S = self.score(x, t)
        return self.sde.ds(t) * self.x0_mean(x, t) + self.sde.dsig(t) * (-self.sde.sig(t) * S)


# ---------------------------------------------------------------- 표집기 (알고리즘 1-5, 식 4, 43, 46)


def _norm_mean(a):
    return a.norm(dim=-1).mean()


def predictor_step(pg, x, t, h, kind, g, pf=False):
    """t에서 t-h로 한 스텝. kind: euler | reverse_diffusion | ancestral. pf이면 확률 흐름(잡음 없음, 점수 항 1/2)."""
    sde = pg.sde
    S = pg.score(x, t)
    c = 0.5 if pf else 1.0
    z = torch.zeros_like(x) if pf else torch.randn(x.shape, generator=g)
    name = sde.name
    if kind == "ancestral" and not pf:
        if name == "VE":
            s2, s2p = sde.sig(t) ** 2, sde.sig(t - h) ** 2
            std = math.sqrt(s2p * (s2 - s2p) / s2)
            return x + (s2 - s2p) * S + std * z
        if name == "VP":
            beta = sde.beta(t) * h                   # 식 4, DDPM의 beta_i = beta(t_i)/N
            return (x + beta * S) / math.sqrt(1 - beta) + math.sqrt(beta) * z
    if kind == "reverse_diffusion" and name == "VE":
        G2 = sde.sig(t) ** 2 - sde.sig(t - h) ** 2   # SMLD 이산화 (식 20), 식 46
        return x + c * G2 * S + math.sqrt(G2) * z
    if kind == "reverse_diffusion" and name == "VP":
        beta = sde.beta(t) * h                       # DDPM 이산화 (식 22), 식 46
        return x + (1 - math.sqrt(1 - beta)) * x + c * beta * S + math.sqrt(beta) * z
    # 오일러-마루야마 (subVP의 역확산 이산화도 이것)
    drift = sde.f_coef(t) * x - (c * sde.g2(t)) * S
    gz = 0.0 if pf else math.sqrt(sde.g2(t) * h)
    return x - h * drift + gz * z


def corrector_step(pg, x, t, snr, g, h_pred):
    """랑주뱅 교정기, 알고리즘 4-5의 신호 대 잡음비 규칙: eps = 2 alpha (r |z| / |score|)^2 (배치 평균 노름)."""
    S = pg.score(x, t)
    z = torch.randn(x.shape, generator=g)
    alpha = 1.0 if pg.sde.name == "VE" else 1.0 - pg.sde.beta(t) * h_pred
    step = 2 * alpha * (snr * _norm_mean(z) / _norm_mean(S)) ** 2
    return x + step * S + math.sqrt(2 * step) * z


def run_sampler(pg, n, N, g, predictor="reverse_diffusion", pf=False, snr=0.0, n_corr=0, eps=None,
                c_only=False, denoise=True, project=None):
    """PC 표집기. 점수 호출 횟수(NFE) = (0 if c_only else N) + N * n_corr.

    c_only이면 예측기 없이 N개 수준마다 교정기만 n_corr번(어닐링 랑주뱅). project(x, t, g)는 조건부 생성용 훅.
    """
    sde = pg.sde
    if eps is None:
        eps = 1e-5 if sde.name == "VE" else 1e-3     # 논문 부록 C와 코드의 sampling_eps
    ts = torch.linspace(sde.T, eps, N + 1).tolist()
    h = (sde.T - eps) / N
    x = sde.prior_std * torch.randn(n, pg.gmm.d, generator=g)
    if project is not None:
        x = project(x, ts[0], g)
    for i in range(N):
        t, tn = ts[i], ts[i + 1]
        if not c_only:
            x = predictor_step(pg, x, t, h, predictor, g, pf=pf)
            if project is not None:
                x = project(x, tn, g)
        for _ in range(n_corr):
            x = corrector_step(pg, x, tn, snr, g, h)
            if project is not None:
                x = project(x, tn, g)
    if denoise:
        x = pg.x0_mean(x, eps)
    return x


# ---------------------------------------------------------------- 확률 흐름 ODE


def rk4_integrate(fun, x, t0, t1, N):
    """dx/dt = fun(x,t)를 t0에서 t1까지 (t1<t0도 가능) 고정 스텝 RK4."""
    h = (t1 - t0) / N
    t = t0
    for _ in range(N):
        k1 = fun(x, t)
        k2 = fun(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = fun(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = fun(x + h * k3, t + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return x


def default_eps(sde):
    return 1e-5 if sde.name == "VE" else 1e-3


def rk4_pf_sample(pg, z, N=200, eps=None):
    """z ~ prior (t=T) 에서 t=eps까지 확률 흐름 ODE를 푼다."""
    eps = default_eps(pg.sde) if eps is None else eps
    return rk4_integrate(pg.pf_velocity, z, pg.sde.T, eps, N)


def encode(pg, x, N=200, eps=None):
    """데이터(t=eps)에서 잠재 코드(t=T)로 확률 흐름 ODE를 따라 간다."""
    eps = default_eps(pg.sde) if eps is None else eps
    return rk4_integrate(pg.pf_velocity, x, eps, pg.sde.T, N)


# ---------------------------------------------------------------- 정확한 우도 (식 39-40)


def prior_logp(sde, z):
    d = z.shape[-1]
    sd = sde.prior_std
    return -0.5 * d * math.log(2 * math.pi * sd * sd) - (z ** 2).sum(-1) / (2 * sd * sd)


def div_autograd_exact(pg, x, t):
    """대각합을 자동미분으로 정확히 계산 (2차원이면 vjp 두 번)."""
    x = x.detach().requires_grad_(True)
    v = pg.pf_velocity(x, t)
    tr = 0.0
    for i in range(x.shape[1]):
        gi = torch.autograd.grad(v[:, i].sum(), x, retain_graph=True)[0]
        tr = tr + gi[:, i]
    return tr.detach()


def div_hutchinson(pg, x, t, e):
    """Skilling-Hutchinson: e^T (dv/dx) e, 하나의 vjp."""
    x = x.detach().requires_grad_(True)
    v = pg.pf_velocity(x, t)
    ge = torch.autograd.grad((v * e).sum(), x)[0]
    return (ge * e).sum(-1).detach()


def div_analytic(pg, x, t):
    d = x.shape[1]
    return d * pg.sde.f_coef(t) - 0.5 * pg.sde.g2(t) * pg.hess_trace(x, t)


def log_density(pg, x0, mode="analytic", probes=None, N=200, eps=None, prior="gauss"):
    """log p_eps(x0) = log p_T(x(T)) + int_eps^T div v dt.
    mode: analytic | autograd | hutchinson(probes=e, 점마다 고정). prior: gauss | true(p_T의 정확한 밀도).
    반환: (log p, x(T), M) 여기서 M = int J dt 는 mode=analytic일 때만(허친슨 분산의 이론값용)."""
    sde = pg.sde
    eps = default_eps(sde) if eps is None else eps
    n, d = x0.shape

    def rhs(state, t):
        x, D, M = state[:, :d], state[:, d], state[:, d + 1:]
        v = pg.pf_velocity(x, t)
        if mode == "analytic":
            S, H = pg.score_and_hess(x, t)
            div = d * sde.f_coef(t) - 0.5 * sde.g2(t) * torch.diagonal(H, dim1=1, dim2=2).sum(-1)
            J = sde.f_coef(t) * torch.eye(d)[None] - 0.5 * sde.g2(t) * H
            return torch.cat([v, div[:, None], J.reshape(n, d * d)], 1)
        with torch.enable_grad():
            div = div_autograd_exact(pg, x, t) if mode == "autograd" else div_hutchinson(pg, x, t, probes)
        return torch.cat([v, div[:, None], torch.zeros(n, d * d)], 1)

    state = torch.cat([x0, torch.zeros(n, 1 + d * d)], 1)
    out = rk4_integrate(rhs, state, eps, sde.T, N)
    xT, D, M = out[:, :d], out[:, d], out[:, d + 1:].reshape(n, d, d)
    if prior == "true":
        lp_T = PerturbedGMM(pg.gmm, sde).logp(xT, sde.T)
    else:
        lp_T = prior_logp(sde, xT)
    return lp_T + D, xT, M


# ---------------------------------------------------------------- 지표


class SlicedW1:
    """고정 기준 표본(n_ref)과 무작위 방향 L개로 슬라이스 바서슈타인-1 거리 (n개 표본, 분위수 맞춤)."""

    def __init__(self, gmm, n_ref=20000, L=128, seed=12345):
        g = torch.Generator().manual_seed(seed)
        dirs = torch.randn(L, gmm.d, generator=g)
        self.dirs = dirs / dirs.norm(dim=1, keepdim=True)
        self.ref = (gmm.sample(n_ref, g) @ self.dirs.T).numpy()
        self.cache = {}

    def __call__(self, x):
        n = x.shape[0]
        if n not in self.cache:
            self.cache[n] = np.quantile(self.ref, (np.arange(n) + 0.5) / n, axis=0)
        p = np.sort((x @ self.dirs.T).numpy(), axis=0)
        return float(np.abs(p - self.cache[n]).mean())


def mode_tv(gmm, x):
    """성분 책임도 argmax로 센 모드 비율과 참 가중치 사이의 전체 변동 거리."""
    pg = PerturbedGMM(gmm, VP())
    lj = pg._parts(x, 0.0, exact_t0=True)[0]
    k = lj.argmax(1)
    p = torch.bincount(k, minlength=gmm.K).double() / x.shape[0]
    return float(0.5 * (p - gmm.w).abs().sum())


def boot_ci(v, B=4000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def boot_diff_ci(a, b, B=4000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (B, len(a)))].mean(1) - b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def holm(pvals):
    """Holm-Bonferroni 보정 p값 (입력 순서 유지)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------- 실험


def exp_pc_comparison(gmm, sw, n=4000, seeds=12, tune_seeds=3, budgets=(2000, 100)):
    """표 1 모사: 같은 점수 호출 횟수(NFE)에서 P-only, C-only, PC 비교. 점수는 정확하다.
    budgets 2000은 논문 표 1의 NFE, 100은 이산화 오차가 표본 잡음 위로 드러나는 작은 예산이다."""
    out = {}
    grid = [0.03, 0.06, 0.1, 0.16, 0.25, 0.4]
    floor = [sw(gmm.sample(n, torch.Generator().manual_seed(7000 + s))) for s in range(seeds)]
    floor_tv = [mode_tv(gmm, gmm.sample(n, torch.Generator().manual_seed(7000 + s))) for s in range(seeds)]
    out["floor_sw1"] = {"values": floor, "mean_ci": boot_ci(floor)}
    out["floor_tv"] = {"mean_ci": boot_ci(floor_tv)}
    allp = []
    for B in budgets:
        h = B // 2
        for sde in (VE(), VP()):
            pg = PerturbedGMM(gmm, sde)

            def run(cfg, seed):
                g = torch.Generator().manual_seed(seed)
                x = run_sampler(pg, n, g=g, **cfg)
                return sw(x), mode_tv(gmm, x)

            tuned = {}
            for key, base in (("C", dict(N=h, n_corr=2, c_only=True)),
                              ("PC", dict(N=h, n_corr=1, predictor="reverse_diffusion"))):
                scores = [float(np.mean([run({**base, "snr": r}, 1000 + s)[0] for s in range(tune_seeds)])) for r in grid]
                tuned[key] = {"grid": grid, "sw1": scores, "best_r": grid[int(np.argmin(scores))]}
            rC, rPC = tuned["C"]["best_r"], tuned["PC"]["best_r"]
            cfgs = {
                f"P{h}_reverse": dict(N=h),
                f"P{B}_reverse": dict(N=B),
                f"P{h}_ancestral": dict(N=h, predictor="ancestral"),
                f"P{B}_ancestral": dict(N=B, predictor="ancestral"),
                f"P{h}_pf": dict(N=h, pf=True),
                f"P{B}_pf": dict(N=B, pf=True),
                f"C{B}": dict(N=h, n_corr=2, c_only=True, snr=rC),
                f"PC{h}_reverse": dict(N=h, n_corr=1, snr=rPC),
                f"PC{h}_ancestral": dict(N=h, n_corr=1, snr=rPC, predictor="ancestral"),
                f"PC{h}_pf": dict(N=h, n_corr=1, snr=rPC, pf=True),
            }
            res = {}
            for name, cfg in cfgs.items():
                vals = [run(cfg, s) for s in range(seeds)]
                sw_v = [v[0] for v in vals]
                tv_v = [v[1] for v in vals]
                res[name] = {"sw1": sw_v, "sw1_mean_ci": boot_ci(sw_v), "tv": tv_v, "tv_mean_ci": boot_ci(tv_v)}
            comps = [(f"PC{h}_reverse", f"P{B}_reverse"), (f"PC{h}_reverse", f"C{B}"), (f"PC{h}_reverse", f"P{h}_reverse"),
                     (f"P{B}_reverse", f"P{h}_reverse"), (f"PC{h}_pf", f"P{B}_pf"), (f"PC{h}_reverse", f"P{B}_ancestral")]
            rows = []
            for a, b in comps:
                t = stats.ttest_ind(res[a]["sw1"], res[b]["sw1"], equal_var=False)
                rows.append({"a": a, "b": b, "diff_sw1": boot_diff_ci(res[a]["sw1"], res[b]["sw1"]), "p_welch": float(t.pvalue)})
                allp.append((B, sde.name, len(rows) - 1))
            out[f"{sde.name}_B{B}"] = {"tuning": tuned, "samplers": res, "comparisons": rows}
    adj = holm([out[f"{sn}_B{B}"]["comparisons"][i]["p_welch"] for B, sn, i in allp])
    for (B, sn, i), a in zip(allp, adj):
        out[f"{sn}_B{B}"]["comparisons"][i]["p_holm_all"] = a
    out["n"], out["seeds"] = n, seeds
    return out


def _mom_stats(a, ref):
    """평균은 표준오차 단위 z, 공분산은 sqrt(대각 곱)으로 정규화한 최대 오차."""
    n1, n2 = a.shape[0], ref.shape[0]
    se = torch.sqrt(a.var(0) / n1 + ref.var(0) / n2)
    z = ((a.mean(0) - ref.mean(0)).abs() / se).max()
    C, R = torch.cov(a.T), torch.cov(ref.T)
    dd = torch.sqrt(torch.diag(R)[:, None] * torch.diag(R)[None])
    return {"mean": a.mean(0).tolist(), "cov": C.flatten().tolist(), "max_mean_z": float(z),
            "max_abs_mean_err": float((a.mean(0) - ref.mean(0)).abs().max()),
            "max_norm_cov_err": float(((C - R).abs() / dd).max())}


def exp_marginals_and_encoding(gmm, sw):
    """확률 흐름 ODE와 역방향 SDE가 같은 중간 주변분포를 내는지, 인코딩이 결정적·가역적인지."""
    out = {}
    n, t_mid = 20000, 0.4
    for sde in (VE(), VP(), SubVP()):
        pg = PerturbedGMM(gmm, sde)
        g = torch.Generator().manual_seed(1)
        truth = pg.sample_t(n, t_mid, g)
        entry = {"t": t_mid, "n": n, "truth_mean": truth.mean(0).tolist(), "truth_cov": torch.cov(truth.T).flatten().tolist()}
        for start in ("true_pT", "gaussian_prior"):
            if start == "true_pT":
                z = pg.sample_t(n, sde.T, g)
            else:
                z = sde.prior_std * torch.randn(n, 2, generator=g)
            pf = rk4_integrate(pg.pf_velocity, z, sde.T, t_mid, 400)
            x = z.clone()
            N = 2000
            h = (sde.T - t_mid) / N
            t = sde.T
            for _ in range(N):
                x = predictor_step(pg, x, t, h, "euler", g)
                t -= h
            entry[start] = {"pf_ode": _mom_stats(pf, truth), "reverse_sde": _mom_stats(x, truth),
                            "sw1_pf": sw_generic(pf, truth, sw), "sw1_sde": sw_generic(x, truth, sw)}
        # 인코딩
        g2 = torch.Generator().manual_seed(2)
        x0 = gmm.sample(500, g2)
        code = encode(pg, x0, N=400)
        back = rk4_pf_sample(pg, code, N=400)
        code_b = encode(pg, x0, N=400)
        code_c = encode(pg, x0, N=100)
        pT = pg.sample_t(20000, sde.T, g2)
        entry["encoding"] = {
            "deterministic_max_diff": float((code - code_b).abs().max()),
            "roundtrip_max_abs_err_N400": float((back - x0).abs().max()),
            "code_N100_vs_N400_max": float((code_c - code).abs().max()),
            "code_over_prior_std_cov": torch.cov((code / sde.prior_std).T).flatten().tolist(),
            "pT_over_prior_std_cov": torch.cov((pT / sde.prior_std).T).flatten().tolist(),
            "code_mean": code.mean(0).tolist(), "pT_mean": pT.mean(0).tolist()}
        out[sde.name] = entry
    return out


def sw_generic(a, b, sw):
    return _sw_two(sw, a, b)


def exp_likelihood(gmm, P=100, R=64, N=200):
    """식 39: PF-ODE 순간 변수 변환 로그 우도 대 닫힌 식 로그 밀도. 정확한 대각합(2차원)과 허친슨(R개 프로브)."""
    out = {}
    g = torch.Generator().manual_seed(3)
    x0 = gmm.sample(P, g)
    for sde in (VE(), VP(), SubVP()):
        pg = PerturbedGMM(gmm, sde)
        eps = default_eps(sde)
        true_eps = pg.logp(x0, eps)
        true_0 = gmm.logpdf(x0)
        lp_an, xT, M = log_density(pg, x0, "analytic", N=N)
        lp_ag, _, _ = log_density(pg, x0, "autograd", N=N)
        lp_true_prior, _, _ = log_density(pg, x0, "analytic", N=N, prior="true")
        lp_fine, _, _ = log_density(pg, x0, "analytic", N=4 * N)
        # 허친슨: 점마다 R개 프로브를 배치로
        res = {}
        for kind in ("rademacher", "gaussian"):
            xr = x0.repeat_interleave(R, 0)
            if kind == "rademacher":
                e = (torch.randint(0, 2, xr.shape, generator=g).double() * 2 - 1)
            else:
                e = torch.randn(xr.shape, generator=g)
            lp_h, _, _ = log_density(pg, xr, "hutchinson", probes=e, N=N)
            lp_h = lp_h.reshape(P, R)
            err = lp_h - lp_an[:, None]
            res[kind] = {"bias_mean": float(err.mean()), "std_over_probes_mean": float(err.std(1).mean()),
                         "rmse": float(err.pow(2).mean().sqrt())}
            Msym = 0.5 * (M + M.transpose(1, 2))
            if kind == "rademacher":
                pred = 4 * Msym[:, 0, 1] ** 2
            else:
                pred = 2 * (Msym ** 2).sum((1, 2))
            res[kind]["pred_var_mean"] = float(pred.mean())
            res[kind]["emp_var_mean"] = float(err.var(1).mean())
            # 점별 이론 분산 대 경험 분산의 상관
            res[kind]["corr_pred_emp"] = float(np.corrcoef(pred.numpy(), err.var(1).numpy())[0, 1])
        out[sde.name] = {
            "P": P, "R": R, "N_rk4": N, "eps": eps,
            "err_vs_true_logp_eps_gauss_prior": _stats(lp_an - true_eps),
            "err_vs_true_logp_eps_true_prior": _stats(lp_true_prior - true_eps),
            "err_autograd_vs_analytic_max": float((lp_ag - lp_an).abs().max()),
            "err_rk4_N_vs_4N_max": float((lp_an - lp_fine).abs().max()),
            "bias_eps_truncation_logp_eps_minus_logp0": _stats(true_eps - true_0),
            "hutchinson": res}
    return out


def _stats(v):
    return {"mean": float(v.mean()), "max_abs": float(v.abs().max()), "std": float(v.std())}


def exp_controllable(gmm, sw, n=3000, seeds=10):
    out = {}
    sde = VP()
    pg = PerturbedGMM(gmm, sde)
    # (a) 식 49의 항등식: 성분 k의 점수 = 혼합 점수 + grad log p_t(y=k|x)  (분류기는 베이즈 규칙으로 닫힌 식)
    g = torch.Generator().manual_seed(5)
    t = 0.3
    x = pg.sample_t(200, t, g)
    errs = []
    for k in range(gmm.K):
        xx = x.clone().requires_grad_(True)
        lj = pg._parts(xx, t)[0]
        logpyx = lj[:, k] - torch.logsumexp(lj, 1)            # log p_t(y=k|x)
        g_cls = torch.autograd.grad(logpyx.sum(), xx)[0]
        s_mix = pg.score(x, t)
        xx2 = x.clone().requires_grad_(True)
        lk = pg._parts(xx2, t)[0][:, k] - math.log(float(gmm.w[k]))   # log N_k (성분 k의 p_t(x|y=k))
        s_k = torch.autograd.grad(lk.sum(), xx2)[0]
        errs.append(float((s_mix + g_cls - s_k).abs().max()))
    out["identity_eq49_max_abs_err"] = max(errs)
    # (b) 클래스 조건부 표집: 점수 + 분류기 기울기로 역방향 SDE (PC 아님, 오일러-마루야마 N=1000)
    class_res = {}
    for k in range(gmm.K):
        vals_c, vals_u = [], []
        for s in range(seeds):
            g = torch.Generator().manual_seed(100 + s)
            ref = _component_samples(gmm, k, n, torch.Generator().manual_seed(500 + s))
            xs = _cond_class_sample(pg, k, n, 1000, g)
            vals_c.append(_sw_two(sw, xs, ref))
            xu = run_sampler(pg, n, 1000, g, predictor="euler")
            vals_u.append(_sw_two(sw, xu, ref))
        class_res[k] = {"conditional_sw1": boot_ci(vals_c), "unconditional_sw1": boot_ci(vals_u)}
    out["class_conditional"] = class_res
    # (c) 인페인팅: x_1 = c 를 관측, x_2를 복원 (공식 코드 get_pc_inpainter 방식: 매 스텝 알려진 좌표를 섭동한 관측값으로 덮어씀)
    inp = {}
    for c in (-2.5, 0.0, 2.5):
        W1s, W1e, means = [], [], []
        q = _exact_cond_x2_quantiles(gmm, c, n)
        for s in range(seeds):
            g = torch.Generator().manual_seed(300 + s)
            xs = _inpaint(pg, c, n, 1000, g)[:, 1]
            W1s.append(float((torch.sort(xs)[0] - q).abs().mean()))
            means.append(float(xs.mean()))
            xe = _cond_exact_sample(pg, gmm, c, n, 1000, torch.Generator().manual_seed(400 + s))
            W1e.append(float((torch.sort(xe)[0] - q).abs().mean()))
        mean2, var2 = _exact_cond_x2_moments(gmm, c)
        inp[str(c)] = {"w1_replacement_vs_exact_conditional": boot_ci(W1s),
                       "w1_exact_conditional_score_vs_exact_conditional": boot_ci(W1e),
                       "replacement_mean_x2": boot_ci(means),
                       "exact_mean_x2": mean2, "exact_sd_x2": math.sqrt(var2)}
    out["inpainting"] = inp
    return out


def _sw_two(sw, a, b):
    """두 표본 사이의 슬라이스 W1 (같은 방향)."""
    pa = np.sort((a @ sw.dirs.T).numpy(), axis=0)
    pb = np.sort((b @ sw.dirs.T).numpy(), axis=0)
    m = min(len(pa), len(pb))
    pa = pa[np.linspace(0, len(pa) - 1, m).astype(int)]
    pb = pb[np.linspace(0, len(pb) - 1, m).astype(int)]
    return float(np.abs(pa - pb).mean())


def _component_samples(gmm, k, n, g):
    L = torch.linalg.cholesky(gmm.cov[k])
    return gmm.mu[k] + torch.randn(n, gmm.d, generator=g) @ L.T


def _cond_class_sample(pg, k, n, N, g, eps=1e-3):
    sde = pg.sde
    ts = torch.linspace(sde.T, eps, N + 1).tolist()
    h = (sde.T - eps) / N
    x = torch.randn(n, 2, generator=g)
    for i in range(N):
        t = ts[i]
        xx = x.clone().requires_grad_(True)
        with torch.enable_grad():
            lj = pg._parts(xx, t)[0]
            logpyx = lj[:, k] - torch.logsumexp(lj, 1)
            gc = torch.autograd.grad(logpyx.sum(), xx)[0]
        S = pg.score(x, t) + gc
        x = x - h * (sde.f_coef(t) * x - sde.g2(t) * S) + math.sqrt(sde.g2(t) * h) * torch.randn(x.shape, generator=g)
    tl = ts[-1]
    xx = x.clone().requires_grad_(True)
    with torch.enable_grad():
        lj = pg._parts(xx, tl)[0]
        gc = torch.autograd.grad((lj[:, k] - torch.logsumexp(lj, 1)).sum(), xx)[0]
    S = pg.score(x, tl) + gc
    return (x + sde.sig(tl) ** 2 * S) / sde.s(tl)


def _cond_exact_sample(pg, gmm, c, n, N, g, eps=1e-3):
    """대조군: x_1(0)=c 를 정확히 조건으로 한 1차원 p_t(x_2 | x_1(0)=c) 혼합의 점수로 역방향 SDE."""
    sde = pg.sde
    w, m, v = (torch.tensor(a) for a in _cond_params(gmm, c))
    ts = torch.linspace(sde.T, eps, N + 1).tolist()
    h = (sde.T - eps) / N

    def score(x, t):
        s_, sg2 = sde.s(t), sde.sig(t) ** 2
        var = s_ * s_ * v + sg2
        mu = s_ * m
        lj = torch.log(w)[None] - 0.5 * ((x[:, None] - mu[None]) ** 2 / var[None] + torch.log(var)[None])
        r = torch.softmax(lj, 1)
        return -(r * (x[:, None] - mu[None]) / var[None]).sum(1)
    x = torch.randn(n, generator=g)
    for i in range(N):
        t = ts[i]
        x = x - h * (sde.f_coef(t) * x - sde.g2(t) * score(x, t)) + math.sqrt(sde.g2(t) * h) * torch.randn(n, generator=g)
    return (x + sde.sig(ts[-1]) ** 2 * score(x, ts[-1])) / sde.s(ts[-1])


def _inpaint(pg, c, n, N, g, eps=1e-3):
    sde = pg.sde

    def project(x, t, gen):
        x = x.clone()
        x[:, 0] = sde.s(t) * c + sde.sig(t) * torch.randn(x.shape[0], generator=gen)
        return x
    x = run_sampler(pg, n, N, g, predictor="reverse_diffusion", eps=eps, denoise=False, project=project)
    x[:, 0] = c
    # 마지막 잡음 제거는 알려지지 않은 좌표만
    full = pg.x0_mean(x, eps)
    full[:, 0] = c
    return full


def _cond_params(gmm, c):
    """x_1 = c 일 때 x_2의 조건부 혼합: 성분별 (가중치, 평균, 분산)."""
    w, m, v = [], [], []
    for k in range(gmm.K):
        mu, S = gmm.mu[k], gmm.cov[k]
        a = math.exp(-0.5 * (c - mu[0]) ** 2 / S[0, 0]) / math.sqrt(2 * math.pi * S[0, 0])
        w.append(float(gmm.w[k]) * float(a))
        m.append(float(mu[1] + S[1, 0] / S[0, 0] * (c - mu[0])))
        v.append(float(S[1, 1] - S[1, 0] ** 2 / S[0, 0]))
    w = np.array(w) / sum(w)
    return w, np.array(m), np.array(v)


def _exact_cond_x2_moments(gmm, c):
    w, m, v = _cond_params(gmm, c)
    mean = float((w * m).sum())
    return mean, float((w * (v + m ** 2)).sum() - mean ** 2)


def _exact_cond_x2_quantiles(gmm, c, n):
    w, m, v = _cond_params(gmm, c)
    grid = np.linspace(m.min() - 8 * np.sqrt(v.max()), m.max() + 8 * np.sqrt(v.max()), 200001)
    cdf = sum(wk * stats.norm.cdf(grid, mk, math.sqrt(vk)) for wk, mk, vk in zip(w, m, v))
    return torch.tensor(np.interp((np.arange(n) + 0.5) / n, cdf, grid))


def exp_flow_link(gmm, sw, n=4000, seeds=5):
    """정확한 점수의 확률 흐름 ODE를 오일러로 NFE개 스텝 푼 오차. 경로(s, sigma)가 곧 이산화 오차를 정한다."""
    out = {}
    nfes = [5, 10, 20, 50, 100, 200]
    for sde in (VE(), VP(), SubVP(), LinearPath()):
        pg = PerturbedGMM(gmm, sde)
        eps = 1e-5 if sde.name == "VE" else (1e-3)
        row = {}
        for nfe in nfes:
            vals = []
            for s in range(seeds):
                g = torch.Generator().manual_seed(40 + s)
                x = sde.prior_std * torch.randn(n, 2, generator=g)
                h = (sde.T - eps) / nfe
                t = sde.T
                for _ in range(nfe):
                    v = pg.fm_velocity(x, t) if sde.name == "linear" else pg.pf_velocity(x, t)
                    x = x - h * v
                    t -= h
                vals.append(sw(pg.x0_mean(x, eps)))
            row[str(nfe)] = boot_ci(vals)
        out[sde.name] = row
    return out


def exp_sde_facts():
    """식 26, 28: 분산 상한. 이산 DDPM/SMLD와 연속 핵의 차이 (그림 5의 수치판)."""
    out = {}
    sde = SubVP()
    vp = VP()
    ts = np.linspace(0.01, 1, 100)
    gap = []
    for t in ts:
        B = vp.B(t)
        v_vp0 = 1 - math.exp(-B)             # Sigma(0)=0
        v_sub0 = (1 - math.exp(-B)) ** 2
        gap.append(v_vp0 - v_sub0)
    out["subvp_leq_vp_min_gap_data_var0"] = float(min(gap))
    # 데이터 분산 sigma0^2 = 0.25, 4, 100에서 식 26, 28
    rows = {}
    for v0 in (0.25, 4.0, 100.0):
        diffs = []
        for t in ts:
            E = math.exp(-vp.B(t))
            diffs.append((1 + E * (v0 - 1)) - (1 + E * E + E * (v0 - 2)))
        rows[str(v0)] = {"min_vp_minus_subvp": float(min(diffs)), "max": float(max(diffs))}
    out["variance_gap_eq26_minus_eq28"] = rows
    disc = {}
    for N in (100, 1000, 10000):
        b = np.linspace(0.1 / N, 20.0 / N, N)                  # 식 22의 beta_i (산술 수열)
        a = np.cumprod(1 - b)
        tt = (np.arange(1, N + 1)) / N
        Bt = 0.1 * tt + 0.5 * 19.9 * tt ** 2
        mean_err = np.abs(np.sqrt(a) - np.exp(-0.5 * Bt)).max()
        var_err = np.abs((1 - a) - (1 - np.exp(-Bt))).max()
        disc[str(N)] = {"mean_coef_max_err": float(mean_err), "var_max_err": float(var_err)}
    out["ddpm_discrete_vs_vp_kernel"] = disc
    return out


def exp_corrector_bias(n=40000, steps=150, rs=(0.05, 0.16, 0.4)):
    """N(0,v I) 표적에 정확한 점수로 랑주뱅 교정기를 오래 돌렸을 때의 정상 분산. 배치 노름 규칙의 고정점 예측: u = v (1 + r^2)."""
    one = GMM([1.0], [[0.0, 0.0]], torch.eye(2)[None])
    pg = PerturbedGMM(one, VE())
    t = 0.5
    v = 1.0 + pg.sde.sig(t) ** 2
    out = {"v": v, "n": n, "steps": steps}
    for r in rs:
        g = torch.Generator().manual_seed(21)
        x = pg.sample_t(n, t, g)
        for _ in range(steps):
            x = corrector_step(pg, x, t, r, g, 1e-3)
        out[str(r)] = {"var_ratio": float(x.var(0).mean() / v), "pred_1_plus_r2": 1 + r * r, "pred_naive_1_over_1_minus_r2": 1 / (1 - r * r)}
    return out


def main():
    t0 = time.time()
    gmm = default_gmm()
    sw = SlicedW1(gmm)
    here = Path(__file__).resolve().parent
    res = {"sde_facts": exp_sde_facts()}
    print("sde_facts", round(time.time() - t0, 1), flush=True)
    res["marginals_encoding"] = exp_marginals_and_encoding(gmm, sw)
    print("marginals", round(time.time() - t0, 1), flush=True)
    res["likelihood"] = exp_likelihood(gmm)
    print("likelihood", round(time.time() - t0, 1), flush=True)
    res["controllable"] = exp_controllable(gmm, sw)
    print("controllable", round(time.time() - t0, 1), flush=True)
    res["flow_link"] = exp_flow_link(gmm, sw)
    print("flow_link", round(time.time() - t0, 1), flush=True)
    res["corrector_bias"] = exp_corrector_bias()
    res["pc_comparison"] = exp_pc_comparison(gmm, sw)
    print("pc", round(time.time() - t0, 1), flush=True)
    with open(here / "minimal_impl_results.json", "w") as f:
        json.dump(res, f, indent=1)
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
