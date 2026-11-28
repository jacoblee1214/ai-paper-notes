"""DDPM(Ho et al. 2020, arXiv:2006.11239)의 최소 구현. 학습은 하지 않는다.

신경망 대신 가우시안 혼합 데이터의 정확한(해석적) 잡음 예측기 eps*를 쓴다.
    q(x_t | x_0) = N(sqrt(abar) x_0, (1-abar) I)  (식 4)
    eps*(x_t, t) = E[eps | x_t] = (x_t - sqrt(abar) E[x_0 | x_t]) / sqrt(1-abar)
데이터가 혼합 sum_k w_k N(m_k, S_k)이면 x_t | k ~ N(sqrt(abar) m_k, abar S_k + (1-abar) I)이므로
E[x_0 | x_t]가 닫힌 식이다. 그래서 식 (4)-(14), 알고리즘 1·2를 모델 오차 없이 검산할 수 있다.

포함: 스케줄(선형, 건너뛰기), 순방향 표본, 사후 q(x_{t-1}|x_t,x_0)(식 6-7), 손실 L_simple(식 14, 알고리즘 1의 평가),
조상 표본기(알고리즘 2, sigma_t^2 = beta_t 또는 beta~_t), 변분 하한 항 L_T, L_{t-1}, L_0(식 5), 슬라이스 W1.
모든 계산은 float64, CPU. 인덱스는 배열 0부터이고 논문의 t = 배열 인덱스 + 1이다.
"""
import math

import torch

DT = torch.float64
LOG2PI = math.log(2.0 * math.pi)


class Schedule:
    """beta_1..beta_T와 파생량. tmap은 (건너뛴 스케줄일 때) 원래 스케줄의 t(1부터)."""

    def __init__(self, betas, tmap=None):
        betas = torch.as_tensor(betas, dtype=DT)
        self.T = betas.numel()
        self.betas = betas
        self.alphas = 1.0 - betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.abar_prev = torch.cat([torch.ones(1, dtype=DT), self.abar[:-1]])
        self.beta_tilde = betas * (1.0 - self.abar_prev) / (1.0 - self.abar)  # 식 (7), t=1에서는 0
        self.tmap = tmap

    @classmethod
    def linear(cls, T=1000, b0=1e-4, b1=0.02):
        return cls(torch.linspace(b0, b1, T, dtype=DT))

    def respace(self, steps):
        """원래 t 중 steps(오름차순, 1부터)만 쓰는 스케줄. abar는 그대로이고 beta'_i = 1 - abar_i/abar_{i-1}."""
        idx = torch.as_tensor(steps, dtype=torch.long) - 1
        ab = self.abar[idx]
        prev = torch.cat([torch.ones(1, dtype=DT), ab[:-1]])
        return Schedule(1.0 - ab / prev, tmap=list(steps))

    def sigma2(self, mode):
        """reverse 분산 sigma_t^2. 'beta' 또는 'beta_tilde'. beta_tilde 모드의 t=1은 beta~_2를 쓴다(공식 코드의 관례, beta~_1=0이라서)."""
        if mode == "beta":
            return self.betas.clone()
        if mode == "beta_tilde":
            s = self.beta_tilde.clone()
            s[0] = s[1] if self.T > 1 else self.betas[0]
            return s
        raise ValueError(mode)


def q_sample(x0, abar, eps):
    """식 (4): x_t = sqrt(abar) x_0 + sqrt(1-abar) eps. abar는 스칼라 또는 (B,)."""
    a = torch.as_tensor(abar, dtype=DT).reshape(-1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * eps


def posterior_mean_var(sched, i, x0, xt):
    """식 (6)-(7): q(x_{t-1} | x_t, x_0)의 평균 mu~_t와 분산 beta~_t. i는 0부터(t=i+1)."""
    ab, abp, b, a = sched.abar[i], sched.abar_prev[i], sched.betas[i], sched.alphas[i]
    mean = abp.sqrt() * b / (1 - ab) * x0 + a.sqrt() * (1 - abp) / (1 - ab) * xt
    return mean, sched.beta_tilde[i]


class GMM:
    """sum_k w_k N(mu_k, cov_k). 정확한 E[x_0|x_t]와 eps*를 준다."""

    def __init__(self, w, mu, cov):
        self.w = torch.as_tensor(w, dtype=DT)
        self.mu = torch.as_tensor(mu, dtype=DT)
        self.cov = torch.as_tensor(cov, dtype=DT)
        self.K, self.D = self.mu.shape
        self.chol = torch.linalg.cholesky(self.cov)
        self.eye = torch.eye(self.D, dtype=DT)

    # ----- 기본 -----
    def sample(self, n, gen):
        comp = torch.multinomial(self.w, n, replacement=True, generator=gen)
        z = torch.randn(n, self.D, generator=gen, dtype=DT)
        return self.mu[comp] + torch.einsum("nij,nj->ni", self.chol[comp], z)

    def log_prob(self, x):
        d = x[:, None, :] - self.mu[None]
        sol = torch.linalg.solve(self.cov[None], d.unsqueeze(-1)).squeeze(-1)
        _, logdet = torch.linalg.slogdet(self.cov)
        lp = -0.5 * (d * sol).sum(-1) - 0.5 * logdet[None] - 0.5 * self.D * LOG2PI
        return torch.logsumexp(lp + self.w.log()[None], dim=-1)

    def moments(self):
        m = (self.w[:, None] * self.mu).sum(0)
        dm = self.mu - m
        c = (self.w[:, None, None] * (self.cov + dm[:, :, None] * dm[:, None, :])).sum(0)
        return m, c

    def at(self, abar):
        """x_t의 주변분포(abar는 스칼라)도 혼합이다."""
        a = float(abar)
        return GMM(self.w, math.sqrt(a) * self.mu, a * self.cov + (1 - a) * self.eye)

    def entropy_quad(self, n_grid=None, half_width=None):
        """미분 엔트로피 -E log q(x_0) (nat)를 격자 구적으로 구한다. D = 1, 2."""
        sd = self.cov.diagonal(dim1=-2, dim2=-1).sqrt().max().item()
        hw = half_width or (self.mu.abs().max().item() + 8 * sd)
        if self.D == 1:
            n = n_grid or 400001
            g = torch.linspace(-hw, hw, n, dtype=DT)
            dx = (g[1] - g[0]).item()
            lp = self.log_prob(g[:, None])
            return float(-(lp.exp() * lp).sum() * dx)
        if self.D == 2:
            n = n_grid or 1601
            g = torch.linspace(-hw, hw, n, dtype=DT)
            dx = (g[1] - g[0]).item()
            tot = 0.0
            for r in range(0, n, 100):
                gx = g[r:r + 100]
                pts = torch.stack(torch.meshgrid(gx, g, indexing="ij"), -1).reshape(-1, 2)
                lp = self.log_prob(pts)
                tot += float((lp.exp() * lp).sum())
            return -tot * dx * dx
        raise NotImplementedError

    # ----- 정확한 사후 -----
    def _posterior(self, xt, abar):
        a = torch.as_tensor(abar, dtype=DT).reshape(-1, 1, 1, 1)  # (A,1,1,1), A = 1 또는 B
        sa = a.sqrt()
        C = a * self.cov[None] + (1 - a) * self.eye  # (A,K,D,D)
        d = xt[:, None, :] - sa[:, :, :, 0] * self.mu[None]  # (B,K,D)
        Cd = torch.linalg.solve(C, d.unsqueeze(-1)).squeeze(-1)
        _, logdet = torch.linalg.slogdet(C)
        logn = -0.5 * (d * Cd).sum(-1) - 0.5 * logdet - 0.5 * self.D * LOG2PI
        r = torch.softmax(self.w.log()[None] + logn, dim=-1)  # (B,K)
        Ek = self.mu[None] + sa[:, :, :, 0] * torch.einsum("kij,bkj->bki", self.cov, Cd)
        return a, sa, C, r, Ek

    def x0_hat(self, xt, abar):
        _, _, _, r, Ek = self._posterior(xt, abar)
        return (r[..., None] * Ek).sum(1)

    def eps_star(self, xt, abar):
        """최적 잡음 예측기 E[eps | x_t] = (x_t - sqrt(abar) E[x_0|x_t]) / sqrt(1-abar)."""
        a = torch.as_tensor(abar, dtype=DT).reshape(-1, 1)
        return (xt - a.sqrt() * self.x0_hat(xt, abar)) / (1 - a).sqrt()

    def cond_var_trace(self, xt, abar):
        """tr Var(x_0 | x_t) (표본마다)."""
        a, _, C, r, Ek = self._posterior(xt, abar)
        S = self.cov
        SCS = torch.einsum("kij,akjl->akil", S, torch.linalg.solve(C, S[None].expand_as(C)))
        tr_post = S.diagonal(dim1=-2, dim2=-1).sum(-1)[None] - a[:, :, 0, 0] * SCS.diagonal(dim1=-2, dim2=-1).sum(-1)
        xh = (r[..., None] * Ek).sum(1)
        spread = ((Ek - xh[:, None, :]) ** 2).sum(-1)
        return (r * (tr_post + spread)).sum(1)


def standardized_mixture(w, mu, cov):
    """혼합 전체가 평균 0, 좌표별 분산 1이 되도록 선형 변환한다(논문 §3.2의 '좌표별 단위 분산' 조건)."""
    g = GMM(w, mu, cov)
    m, c = g.moments()
    s = c.diagonal().sqrt()
    mu2 = (g.mu - m) / s
    cov2 = g.cov / (s[:, None] * s[None, :])
    return GMM(w, mu2, cov2)


def toy_mixture_2d():
    """2차원 비등방 가우시안 4개의 혼합(좌표별 분산 1로 표준화)."""
    w = [0.4, 0.3, 0.2, 0.1]
    mu = [[-2.0, -1.0], [2.0, 0.0], [0.0, 2.5], [-1.0, 3.0]]
    cov = [[[0.30, 0.12], [0.12, 0.20]], [[0.15, -0.05], [-0.05, 0.40]],
           [[0.25, 0.0], [0.0, 0.10]], [[0.05, 0.0], [0.0, 0.05]]]
    return standardized_mixture(w, mu, cov)


def toy_mixture_2d_tight():
    """거의 점질량인 군집 5개(표준편차 0.04)의 혼합. 좌표별 분산 1로 표준화. '데이터가 거의 결정적'인 극단에 가깝다."""
    w = [0.3, 0.25, 0.2, 0.15, 0.1]
    mu = [[-2.0, -1.0], [2.0, 0.0], [0.0, 2.5], [-1.0, 3.0], [1.5, -2.0]]
    cov = [[[0.0016, 0.0], [0.0, 0.0016]]] * 5
    return standardized_mixture(w, mu, cov)


def toy_mixture_1d():
    return standardized_mixture([0.3, 0.7], [[-1.6], [0.7]], [[[0.16]], [[0.81]]])


# ----- 알고리즘 1: 손실 평가(학습 없음) -----
@torch.no_grad()
def l_simple_samples(eps_fn, data, sched, n, gen, chunk=50000):
    """식 (14)의 표본: ||eps - eps_theta(x_t, t)||^2 (좌표 합). t ~ U{1..T}. (표본별 손실, 표본별 t 인덱스)를 돌려준다."""
    losses, idxs = [], []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        x0 = data.sample(m, gen)
        idx = torch.randint(0, sched.T, (m,), generator=gen)
        eps = torch.randn(m, data.D, generator=gen, dtype=DT)
        ab = sched.abar[idx]
        xt = q_sample(x0, ab, eps)
        losses.append(((eps - eps_fn(xt, ab)) ** 2).sum(-1))
        idxs.append(idx)
    return torch.cat(losses), torch.cat(idxs)


# ----- 알고리즘 2: 조상 표본 -----
@torch.no_grad()
def sample(eps_fn, sched, n, D, sigma_mode, gen):
    """x_T ~ N(0,I), x_{t-1} = (x_t - beta_t/sqrt(1-abar_t) eps(x_t,t))/sqrt(alpha_t) + sigma_t z.
    z는 매 단계 같은 순서로 뽑아(t=1에서는 버림) 같은 시드면 sigma_mode가 달라도 난수가 같다."""
    s2 = sched.sigma2(sigma_mode)
    x = torch.randn(n, D, generator=gen, dtype=DT)
    for i in reversed(range(sched.T)):
        ab, a, b = sched.abar[i], sched.alphas[i], sched.betas[i]
        e = eps_fn(x, ab)
        mean = (x - b / (1 - ab).sqrt() * e) / a.sqrt()
        z = torch.randn(n, D, generator=gen, dtype=DT)
        x = mean + (s2[i].sqrt() * z if i > 0 else 0.0)
    return x


def gaussian_sampler_variances(s2_data, sched, sigma_mode):
    """1차원 N(0, s2_data) 데이터에서 정확한 eps*로 알고리즘 2를 돌릴 때 각 t의 표본 분산 V_t (t = 0..T, MC 없이 정확).
    eps* = sqrt(1-abar) x_t / v_t, v_t = abar s2 + 1 - abar (순방향 주변 분산), 한 단계는 x -> (1 - beta/v) x / sqrt(alpha) + sigma z.
    V_T = 1(사전분포)에서 V_{t-1} = c_t^2 V_t + sigma_t^2 (t=1은 잡음 없음). 정확한 사슬이면 V_t = v_t."""
    s2 = sched.sigma2(sigma_mode)
    V = [0.0] * (sched.T + 1)
    V[sched.T] = 1.0
    for i in reversed(range(sched.T)):
        v = sched.abar[i] * s2_data + 1 - sched.abar[i]
        c = (1 - sched.betas[i] / v) / sched.alphas[i].sqrt()
        V[i] = float(c * c * V[i + 1] + (s2[i] if i > 0 else 0.0))
    return V


def gaussian_sampler_variance(s2_data, sched, sigma_mode):
    """최종(t=0) 분산 V_0."""
    return gaussian_sampler_variances(s2_data, sched, sigma_mode)[0]


def forward_marginal_variance(s2_data, sched):
    """순방향 주변 분산 v_t (t = 0..T), x_0 ~ N(0, s2_data)."""
    return [s2_data] + [float(a * s2_data + 1 - a) for a in sched.abar]


# ----- 변분 하한(식 5) -----
def kl_iso(mu1, v1, mu2, v2):
    """KL(N(mu1, v1 I) || N(mu2, v2 I)), 좌표 합. v1, v2는 스칼라."""
    D = mu1.shape[-1]
    return 0.5 * D * (v1 / v2 - 1 + math.log(v2 / v1)) + ((mu1 - mu2) ** 2).sum(-1) / (2 * v2)


@torch.no_grad()
def vlb_terms(data, sched, sigma_mode, n, gen):
    """eps*를 쓴 reverse 과정의 L_T, L_{t-1}(t>=2), L_0 (nat/표본). 항마다 독립 표본 n개의 MC.
    mu_theta는 식 (11), 사후 평균은 식 (7)로 계산해 그대로 KL을 구한다. L_0은 연속 가우시안 밀도(이산화 없음)."""
    T = sched.T
    s2 = sched.sigma2(sigma_mode)
    L = torch.zeros(T, dtype=DT)
    SE = torch.zeros(T, dtype=DT)
    for i in range(T):
        x0 = data.sample(n, gen)
        eps = torch.randn(n, data.D, generator=gen, dtype=DT)
        ab, a, b = sched.abar[i], sched.alphas[i], sched.betas[i]
        xt = q_sample(x0, ab, eps)
        e = data.eps_star(xt, ab)
        mu_th = (xt - b / (1 - ab).sqrt() * e) / a.sqrt()  # 식 (11)
        if i == 0:
            term = 0.5 * data.D * (LOG2PI + math.log(float(s2[0]))) + ((x0 - mu_th) ** 2).sum(-1) / (2 * s2[0])
        else:
            mu_t, bt = posterior_mean_var(sched, i, x0, xt)
            term = kl_iso(mu_t, float(bt), mu_th, float(s2[i]))
        L[i], SE[i] = term.mean(), term.std() / math.sqrt(n)
    _, c = data.moments()
    Ex2 = float(torch.trace(c) + (data.moments()[0] ** 2).sum())
    abT = float(sched.abar[-1])
    L_T = 0.5 * (data.D * (1 - abT) - data.D - data.D * math.log(1 - abT) + abT * Ex2)
    return {"L_T": L_T, "L_0": float(L[0]), "L_mid": L[1:], "se_0": float(SE[0]), "se_mid": SE[1:]}


@torch.no_grad()
def marginal_kl_terms(data, sched, sigma_mode, n, gen):
    """식 (16)/(26) 형태: L - H(x_0) = KL(q(x_T)||p(x_T)) + sum_t E KL(q(x_{t-1}|x_t) || p_theta(x_{t-1}|x_t)).
    q(x_{t-1}|x_t)는 주변분포(혼합)로 계산한다: log q(x_{t-1}|x_t) = log q_{t-1}(x_{t-1}) + log N(x_t; sqrt(alpha) x_{t-1}, beta I) - log q_t(x_t).
    항마다 독립 표본 n개의 MC(nat). 각 항은 KL이라 기댓값이 0 이상이다. 식 (5)의 합보다 분산이 훨씬 작다."""
    D, T = data.D, sched.T
    s2 = sched.sigma2(sigma_mode)
    terms = torch.zeros(T, dtype=DT)
    se = torch.zeros(T, dtype=DT)
    for i in range(T):
        ab, abp, a, b = (float(v[i]) for v in (sched.abar, sched.abar_prev, sched.alphas, sched.betas))
        x0 = data.sample(n, gen)
        xprev = math.sqrt(abp) * x0 + math.sqrt(1 - abp) * torch.randn(n, D, generator=gen, dtype=DT)
        xt = math.sqrt(a) * xprev + math.sqrt(b) * torch.randn(n, D, generator=gen, dtype=DT)
        q_prev = data if i == 0 else data.at(abp)
        log_trans = -0.5 * D * math.log(2 * math.pi * b) - ((xt - math.sqrt(a) * xprev) ** 2).sum(-1) / (2 * b)
        logq = q_prev.log_prob(xprev) + log_trans - data.at(ab).log_prob(xt)
        e = data.eps_star(xt, sched.abar[i])
        mu = (xt - b / math.sqrt(1 - ab) * e) / math.sqrt(a)
        v = float(s2[i])
        logp = -0.5 * D * math.log(2 * math.pi * v) - ((xprev - mu) ** 2).sum(-1) / (2 * v)
        kl = logq - logp
        terms[i], se[i] = kl.mean(), kl.std() / math.sqrt(n)
    abT = float(sched.abar[-1])
    x0 = data.sample(n, gen)
    xT = math.sqrt(abT) * x0 + math.sqrt(1 - abT) * torch.randn(n, D, generator=gen, dtype=DT)
    klT = data.at(abT).log_prob(xT) + 0.5 * D * LOG2PI + 0.5 * (xT ** 2).sum(-1)
    return {"KL_T": float(klT.mean()), "se_T": float(klT.std() / math.sqrt(n)), "terms": terms, "se": se}


# ----- 표본 비교 -----
def random_directions(D, P, gen):
    z = torch.randn(D, P, generator=gen, dtype=DT)
    return z / z.norm(dim=0, keepdim=True)


def sliced_w1(x, y, proj):
    """슬라이스 1-Wasserstein(같은 표본 수). 방향 평균."""
    px = (x @ proj).sort(0).values
    py = (y @ proj).sort(0).values
    return float((px - py).abs().mean())


def bootstrap_ci(values, B=10000, seed=0, alpha=0.05):
    v = torch.as_tensor(values, dtype=DT)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, v.numel(), (B, v.numel()), generator=g)
    m = v[idx].mean(1)
    return float(v.mean()), float(m.quantile(alpha / 2)), float(m.quantile(1 - alpha / 2))


if __name__ == "__main__":
    # 작은 시연: 선형 스케줄, 2차원 혼합, eps*로 알고리즘 2
    g = torch.Generator().manual_seed(0)
    sch = Schedule.linear()
    data = toy_mixture_2d()
    print("abar_T = %.3e" % float(sch.abar[-1]))
    x = sample(data.eps_star, sch.respace(list(range(10, 1001, 10))), 5000, 2, "beta_tilde", g)
    ref = data.sample(5000, g)
    print("mean(model) =", x.mean(0).tolist(), " cov(model) =", torch.cov(x.T).tolist())
    print("mean(data)  =", ref.mean(0).tolist(), " cov(data)  =", torch.cov(ref.T).tolist())
    loss, _ = l_simple_samples(data.eps_star, data, sch, 20000, g)
    print("L_simple at eps* (n=20000): %.4f +- %.4f" % (loss.mean(), loss.std() / 20000 ** 0.5))
