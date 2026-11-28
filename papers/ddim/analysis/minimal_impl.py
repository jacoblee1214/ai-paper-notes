"""DDIM 최소 구현 (정확한 epsilon* 사용, 학습 없음).

데이터가 가우시안 혼합이면 모든 노이즈 수준에서 E[x0 | x_t]가 닫힌 식이라
    eps*(x_t, t) = (x_t - sqrt(abar_t) E[x0 | x_t]) / sqrt(1 - abar_t)
를 그대로 쓸 수 있다. 신경망 오차가 없으므로 "스텝 수를 줄이면 생기는 오차"(이산화 오차)만 따로 볼 수 있다.

표기: 논문(Song, Meng, Ermon 2021)의 alpha_t는 DDPM의 abar_t(누적곱)이다. 코드에서는 ab[t], ab[0]=1.
시간 색인은 논문처럼 1..T이다.

실행: python3 minimal_impl.py   (CPU, float64, 메모리 수백 MB 이하, 수 분)
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

DT = torch.float64
torch.set_num_threads(2)


# ----------------------------------------------------------------------------
# 스케줄, 데이터, 정확한 epsilon*
# ----------------------------------------------------------------------------
def make_alpha_bar(T=1000, beta_1=1e-4, beta_T=0.02):
    """DDPM 선형 스케줄. ab[0]=1, ab[t]=prod_{s<=t}(1-beta_s)."""
    betas = torch.linspace(beta_1, beta_T, T, dtype=DT)
    ab = torch.cumprod(1.0 - betas, 0)
    return torch.cat([torch.ones(1, dtype=DT), ab])


class GMM:
    """등방 가우시안 혼합 sum_k w_k N(mu_k, s_k^2 I)."""

    def __init__(self, w, mu, s):
        self.w = torch.as_tensor(w, dtype=DT)
        self.mu = torch.as_tensor(mu, dtype=DT)
        self.s = torch.as_tensor(s, dtype=DT)
        self.d = self.mu.shape[1]

    def sample(self, n, gen):
        comp = torch.multinomial(self.w, n, replacement=True, generator=gen)
        return self.mu[comp] + self.s[comp, None] * torch.randn(n, self.d, generator=gen, dtype=DT)

    def _resp_and_mean(self, x, a):
        sa = math.sqrt(a)
        v = a * self.s ** 2 + (1.0 - a)                        # (K,) 섭동 후 성분 분산
        diff = x[:, None, :] - sa * self.mu[None]              # (n,K,d)
        logp = torch.log(self.w)[None] - 0.5 * self.d * torch.log(v)[None] - 0.5 * (diff ** 2).sum(-1) / v[None]
        r = torch.softmax(logp, dim=1)
        m = self.mu[None] + (sa * self.s ** 2 / v)[None, :, None] * diff
        return r, m

    def x0_hat(self, x, a):
        """E[x0 | x_t], a = abar_t."""
        r, m = self._resp_and_mean(x, a)
        return (r[..., None] * m).sum(1)

    def eps(self, x, a):
        """정확한 epsilon*(x_t). a=1이면 0(극한값)."""
        if a >= 1.0:
            return torch.zeros_like(x)
        return (x - math.sqrt(a) * self.x0_hat(x, a)) / math.sqrt(1.0 - a)

    def log_pdf_t(self, x, a):
        """섭동 분포 q(x_t) 의 로그 밀도 (autograd 검산용)."""
        sa = math.sqrt(a)
        v = a * self.s ** 2 + (1.0 - a)
        diff = x[:, None, :] - sa * self.mu[None]
        lp = torch.log(self.w)[None] - 0.5 * self.d * torch.log(2 * math.pi * v)[None] - 0.5 * (diff ** 2).sum(-1) / v[None]
        return torch.logsumexp(lp, dim=1)

    def label(self, x):
        """가장 그럴듯한 성분 번호 (t=0 사후 확률 기준)."""
        diff = x[:, None, :] - self.mu[None]
        lp = torch.log(self.w)[None] - self.d * torch.log(self.s)[None] - 0.5 * (diff ** 2).sum(-1) / self.s[None] ** 2
        return lp.argmax(1)


def default_gmm():
    """2차원, 4개 봉우리(가중치 0.4/0.3/0.2/0.1), 봉우리 표준편차 0.1, 봉우리 간 거리 1.4."""
    mu = [[-0.7, -0.7], [0.7, -0.7], [0.7, 0.7], [-0.7, 0.7]]
    return GMM([0.4, 0.3, 0.2, 0.1], mu, [0.1] * 4)


# ----------------------------------------------------------------------------
# 부분 수열 tau, sigma
# ----------------------------------------------------------------------------
def make_taus(T, S, kind="linear"):
    """논문 부록 D.2: linear tau_i=floor(c i), quadratic tau_i=floor(c i^2), tau_S가 T 근처가 되게 c를 잡는다.
    1 미만은 1로 올리고 중복은 지운다(이차 격자는 S가 크면 앞쪽이 겹친다). 반환은 1..T의 증가 수열."""
    i = np.arange(1, S + 1, dtype=np.float64)
    if kind == "linear":
        t = np.floor(i * T / S)
    elif kind == "quad":
        t = np.floor(i ** 2 * T / S ** 2)
    else:
        raise ValueError(kind)
    return [int(v) for v in np.unique(np.maximum(t, 1).astype(int))]


def make_taus_official(T, S, kind="uniform"):
    """공식 저장소(runners/diffusion.py)의 격자를 논문 색인(1..T)으로 옮긴 것(0 기준 색인 + 1).
    uniform: range(0, T, T//S). quad: int(linspace(0, sqrt(0.8 T), S)^2). 중복은 지우지 않는다."""
    if kind == "uniform":
        seq = list(range(0, T, T // S))
    elif kind == "quad":
        seq = [int(v) for v in (np.linspace(0, math.sqrt(T * 0.8), S) ** 2)]
    else:
        raise ValueError(kind)
    return [v + 1 for v in seq]


def sigma_eta(a, a_prev, eta):
    """식 16: sigma_{tau_i}(eta) = eta sqrt((1-a_prev)/(1-a)) sqrt(1-a/a_prev)."""
    if a_prev >= 1.0:
        return 0.0
    return eta * math.sqrt((1.0 - a_prev) / (1.0 - a)) * math.sqrt(max(1.0 - a / a_prev, 0.0))


def sigma_hat(a, a_prev):
    """부록 D.3: sigma_hat = sqrt(1 - a/a_prev) (잡음 계수만 바뀌고 방향 항은 eta=1 그대로)."""
    return math.sqrt(max(1.0 - a / a_prev, 0.0))


# ----------------------------------------------------------------------------
# 표집기 (식 12)
# ----------------------------------------------------------------------------
def ddim_step(x, e, a, a_prev, sigma, noise_std=None, gen=None):
    """식 12 한 스텝. e=eps(x_t). noise_std가 None이면 sigma를 잡음 크기로 쓴다."""
    x0 = (x - math.sqrt(1.0 - a) * e) / math.sqrt(a)
    c = math.sqrt(max(1.0 - a_prev - sigma ** 2, 0.0))
    out = math.sqrt(a_prev) * x0 + c * e
    ns = sigma if noise_std is None else noise_std
    if ns > 0.0:
        out = out + ns * torch.randn(x.shape, generator=gen, dtype=DT)
    return out


def sample(eps_fn, x_T, taus, ab, eta=0.0, gen=None, hat=False, mask_last_hat=False, record=False):
    """역방향으로 taus를 따라 x_T에서 x_0까지. eta: 식 16. hat=True면 sigma_hat 변형.
    mask_last_hat=True면 hat의 마지막 스텝 잡음을 지운다(공식 코드 ddpm_steps의 mask)."""
    x = x_T
    traj = [x]
    prev = [0] + list(taus[:-1])
    for t, tp in zip(reversed(list(taus)), reversed(prev)):
        a, ap = float(ab[t]), float(ab[tp])
        e = eps_fn(x, a)
        sig = sigma_eta(a, ap, 1.0 if hat else eta)
        ns = None
        if hat:
            ns = 0.0 if (mask_last_hat and tp == 0) else sigma_hat(a, ap)
        x = ddim_step(x, e, a, ap, sig, ns, gen)
        if record:
            traj.append(x)
    return (x, traj) if record else x


def sample_euler_bar(eps_fn, x_T, taus, ab, variant="ddim"):
    """eta=0 표집을 재매개화 변수 xbar = x/sqrt(abar), sbar = sqrt((1-abar)/abar)의 오일러 스텝으로 직접 적는다.
    variant='ddim': dxbar = eps dsbar (식 13-14).  variant='pf': dxbar = eps d(sbar^2)/(2 sbar) (식 15)."""
    sb = torch.sqrt((1.0 - ab) / ab)
    xb = x_T / math.sqrt(float(ab[taus[-1]]))
    prev = [0] + list(taus[:-1])
    for t, tp in zip(reversed(list(taus)), reversed(prev)):
        a = float(ab[t])
        e = eps_fn(xb * math.sqrt(a), a)
        if variant == "ddim":
            xb = xb + float(sb[tp] - sb[t]) * e
        elif variant == "pf":
            xb = xb + 0.5 * float(sb[tp] ** 2 - sb[t] ** 2) / float(sb[t]) * e
        else:
            raise ValueError(variant)
    return xb  # ab[0]=1 이므로 x0 = xbar


def ode_reference(gmm, x_T, ab, n_steps=2000, s_min=1e-4):
    """식 14의 ODE dxbar/dsbar = eps*(xbar/sqrt(1+sbar^2))를 sbar(T)에서 0까지 RK4로 푼 기준 해.
    격자는 sbar에 대해 등비수열(n_steps 구간), 마지막 [s_min, 0] 구간은 오일러 한 스텝."""
    T = len(ab) - 1
    s_T = math.sqrt((1.0 - float(ab[T])) / float(ab[T]))
    grid = np.geomspace(s_T, s_min, n_steps + 1)
    xb = x_T / math.sqrt(float(ab[T]))

    def f(xb_, s):
        a = 1.0 / (1.0 + s * s)
        return gmm.eps(xb_ * math.sqrt(a), a)

    for k in range(n_steps):
        s0, s1 = grid[k], grid[k + 1]
        h = s1 - s0
        k1 = f(xb, s0)
        k2 = f(xb + 0.5 * h * k1, s0 + 0.5 * h)
        k3 = f(xb + 0.5 * h * k2, s0 + 0.5 * h)
        k4 = f(xb + h * k3, s1)
        xb = xb + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xb + (0.0 - s_min) * f(xb, s_min)


def encode(eps_fn, x0, taus, ab):
    """eta=0 의 역방향(x_0 -> x_T): 각 스텝에서 현재(잡음이 더 적은) 점의 eps로 식 12를 거꾸로 적용(오일러 전진)."""
    x = x0
    prev = 0
    for t in taus:
        a, ap = float(ab[t]), float(ab[prev])
        e = eps_fn(x, ap)
        x0h = (x - math.sqrt(1.0 - ap) * e) / math.sqrt(ap)
        x = math.sqrt(a) * x0h + math.sqrt(1.0 - a) * e
        prev = t
    return x


# ----------------------------------------------------------------------------
# 정리 1(J_sigma = L_gamma + C)의 가중치
# ----------------------------------------------------------------------------
def q_sigma_coefs(ab, t, sigma):
    """q_sigma(x_{t-1}|x_t, x0)의 평균 = sqrt(ab_{t-1}) x0 + c (x_t - sqrt(ab_t) x0)/sqrt(1-ab_t), c=sqrt(1-ab_{t-1}-sigma^2).
    x0에 대한 계수 A, x_t에 대한 계수 B를 반환한다 (평균 = A x0 + B x_t)."""
    a, ap = float(ab[t]), float(ab[t - 1])
    c = math.sqrt(1.0 - ap - sigma ** 2)
    return math.sqrt(ap) - c * math.sqrt(a) / math.sqrt(1.0 - a), c / math.sqrt(1.0 - a)


def theorem1_gamma(ab, t, sigma):
    """KL(q_sigma(.|x_t,x0) || q_sigma(.|x_t,f)) = gamma_t ||eps - eps_theta||^2 의 gamma_t (t>1).
    두 가우시안의 공분산이 sigma^2 I로 같으므로 KL = ||A (x0 - f)||^2 / (2 sigma^2), x0 - f = sqrt((1-ab_t)/ab_t)(eps_theta - eps)."""
    a = float(ab[t])
    A, _ = q_sigma_coefs(ab, t, sigma)
    return A ** 2 * (1.0 - a) / a / (2.0 * sigma ** 2)


def paper_gamma(ab, t, sigma, d=1):
    """논문 부록 식 32, 35의 가중치 1/(2 d sigma_t^2 abar_t)."""
    return 1.0 / (2.0 * d * sigma ** 2 * float(ab[t]))


def ho_gamma(ab, t):
    """DDPM(Ho 2020 식 12)의 가중치 beta_t^2 / (2 sigma_t^2 alpha_t (1-abar_t)), sigma_t^2 = beta_tilde_t."""
    a, ap = float(ab[t]), float(ab[t - 1])
    alpha_t = a / ap
    beta = 1.0 - alpha_t
    var = beta * (1.0 - ap) / (1.0 - a)
    return beta ** 2 / (2.0 * var * alpha_t * (1.0 - a))


# ----------------------------------------------------------------------------
# 지표와 통계
# ----------------------------------------------------------------------------
_DIRS = {}


def _dirs(d, n_dir=128):
    key = (d, n_dir)
    if key not in _DIRS:
        g = torch.Generator().manual_seed(12345)
        v = torch.randn(n_dir, d, generator=g, dtype=DT)
        _DIRS[key] = v / v.norm(dim=1, keepdim=True)
    return _DIRS[key]


def sliced_w1(x, y, n_dir=128):
    """슬라이스 1-바서슈타인(고정 방향 128개, 표본 수 같음)."""
    dd = _dirs(x.shape[1], n_dir)
    px, py = (x @ dd.T).sort(0).values, (y @ dd.T).sort(0).values
    return float((px - py).abs().mean())


def mmd2(x, y, bws=(0.1, 0.3, 1.0)):
    """RBF 커널 MMD^2의 불편 추정(대역폭 세 개의 합)."""
    n, m = x.shape[0], y.shape[0]
    tot = 0.0
    dxx, dyy, dxy = torch.cdist(x, x) ** 2, torch.cdist(y, y) ** 2, torch.cdist(x, y) ** 2
    for h in bws:
        kxx, kyy, kxy = torch.exp(-dxx / (2 * h * h)), torch.exp(-dyy / (2 * h * h)), torch.exp(-dxy / (2 * h * h))
        tot += float((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) - 2 * kxy.mean())
    return tot


def bootstrap_ci(vals, B=10000, seed=0, alpha=0.05):
    """평균의 백분위 부트스트랩 신뢰구간. 반환 (평균, 하한, 상한)."""
    v = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni. 반환: 보정 p 목록과 기각 여부."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj.tolist(), (adj < alpha).tolist()


def paired_test(diff):
    """짝지은 차이의 평균, 부트스트랩 95% 구간, 대응표본 t 검정 p."""
    from scipy import stats
    mean, lo, hi = bootstrap_ci(diff)
    p = float(stats.ttest_1samp(np.asarray(diff), 0.0).pvalue)
    return {"mean": mean, "lo": lo, "hi": hi, "p": p}


# ----------------------------------------------------------------------------
# 실험
# ----------------------------------------------------------------------------
SEEDS = 20
N = 3000


def _gen(seed):
    return torch.Generator().manual_seed(int(seed))


def exp_sweep(gmm, ab, T=1000):
    """S x eta 격자(표 1의 토이판). 정확한 eps*, 선형 부분 수열, 20 시드 x 3000 표본."""
    Ss = [10, 20, 50, 100, 1000]
    etas = {"0.0": 0.0, "0.2": 0.2, "0.5": 0.5, "1.0": 1.0, "hat": "hat", "hat_masked": "hat_masked"}
    eps_fn = gmm.eps
    res = {"S": Ss, "n": N, "seeds": SEEDS, "metrics": {}}
    floor_sw, floor_mmd = [], []
    cell = {(S, k): {"sw": [], "mmd": []} for S in Ss for k in etas}
    for r in range(SEEDS):
        ref = gmm.sample(N, _gen(2000 + r))
        data = gmm.sample(N, _gen(4000 + r))
        floor_sw.append(sliced_w1(data, ref))
        floor_mmd.append(mmd2(data, ref))
        xT = torch.randn(N, gmm.d, generator=_gen(1000 + r), dtype=DT)
        for S in Ss:
            taus = make_taus(T, S, "linear")
            for ki, (k, eta) in enumerate(etas.items()):
                g = _gen(10 ** 6 + 10000 * r + 100 * S + ki)
                if eta == "hat":
                    x0 = sample(eps_fn, xT, taus, ab, gen=g, hat=True)
                elif eta == "hat_masked":
                    x0 = sample(eps_fn, xT, taus, ab, gen=g, hat=True, mask_last_hat=True)
                else:
                    x0 = sample(eps_fn, xT, taus, ab, eta=eta, gen=g)
                cell[(S, k)]["sw"].append(sliced_w1(x0, ref))
                cell[(S, k)]["mmd"].append(mmd2(x0, ref))
    res["floor"] = {"sw1": bootstrap_ci(floor_sw), "mmd2": bootstrap_ci(floor_mmd)}
    for metric in ("sw", "mmd"):
        tab = {}
        for S in Ss:
            tab[str(S)] = {k: bootstrap_ci(cell[(S, k)][metric]) for k in etas}
        res["metrics"][metric] = tab
    # eta=0 대비 짝지은 비교 + Holm (SW1 전체 20개 비교를 한 묶음으로)
    comp = []
    for S in Ss:
        for k in etas:
            if k == "0.0":
                continue
            d = np.array(cell[(S, k)]["sw"]) - np.array(cell[(S, "0.0")]["sw"])
            t = paired_test(d)
            t.update({"S": S, "eta": k})
            comp.append(t)
    adj, rej = holm([c["p"] for c in comp])
    for c, a_, r_ in zip(comp, adj, rej):
        c["p_holm"], c["significant"] = a_, r_
    res["paired_vs_eta0_sw1"] = comp
    comp2 = []
    for S in Ss:
        for k in etas:
            if k == "0.0":
                continue
            d = np.array(cell[(S, k)]["mmd"]) - np.array(cell[(S, "0.0")]["mmd"])
            t = paired_test(d)
            t.update({"S": S, "eta": k})
            comp2.append(t)
    adj, rej = holm([c["p"] for c in comp2])
    for c, a_, r_ in zip(comp2, adj, rej):
        c["p_holm"], c["significant"] = a_, r_
    res["paired_vs_eta0_mmd2"] = comp2
    # eta=0 의 순위: 각 S에서 평균 SW1이 가장 작은 설정
    res["best_per_S_sw1"] = {str(S): min(etas, key=lambda k: np.mean(cell[(S, k)]["sw"])) for S in Ss}
    return res


def exp_grids(gmm, ab, T=1000):
    """부분 수열 종류: 논문 선형/이차 vs 공식 코드의 uniform/quad(마지막 0.8T)."""
    out = {}
    eps_fn = gmm.eps
    for S in [10, 20, 50, 100]:
        for name, taus in [("linear", make_taus(T, S, "linear")), ("quad", make_taus(T, S, "quad")),
                           ("official_uniform", make_taus_official(T, S, "uniform")),
                           ("official_quad", make_taus_official(T, S, "quad"))]:
            for eta in (0.0, 1.0, "hat", "hat_masked"):
                sw = []
                for r in range(SEEDS):
                    ref = gmm.sample(N, _gen(2000 + r))
                    xT = torch.randn(N, gmm.d, generator=_gen(1000 + r), dtype=DT)
                    g = _gen(5 * 10 ** 6 + 977 * r + S)
                    if eta in ("hat", "hat_masked"):
                        x0 = sample(eps_fn, xT, taus, ab, gen=g, hat=True, mask_last_hat=(eta == "hat_masked"))
                    else:
                        x0 = sample(eps_fn, xT, taus, ab, eta=eta, gen=g)
                    sw.append(sliced_w1(x0, ref))
                out[f"S{S}_{name}_eta{eta}"] = {"nfe": len(taus), "tau_max": max(taus), "abar_start": float(ab[max(taus)]),
                                                "tau_min": min(taus), "sw1": bootstrap_ci(sw)}
    return out


def exp_ode_error(gmm, ab, T=1000):
    """eta=0 표집이 연속 ODE의 정확한 흐름에서 얼마나 벗어나나(같은 x_T). DDIM(Delta sbar 오일러) 대 식 15(PF 오일러)."""
    eps_fn = gmm.eps
    xT = torch.randn(N, gmm.d, generator=_gen(777), dtype=DT)
    ref = ode_reference(gmm, xT, ab)
    ref_lab = gmm.label(ref)
    out = {"ode_reference_steps": 2000}
    for S in [5, 10, 20, 50, 100, 200, 500, 1000]:
        taus = make_taus(T, S, "linear")
        a = sample(eps_fn, xT, taus, ab, eta=0.0)
        b = sample_euler_bar(eps_fn, xT, taus, ab, "pf")
        c = sample_euler_bar(eps_fn, xT, taus, ab, "ddim")
        out[str(S)] = {
            "ddim_rms_to_ode": float(((a - ref) ** 2).sum(1).mean().sqrt()),
            "ddim_median_err_to_ode": float((a - ref).norm(dim=1).median()),
            "ddim_p90_err_to_ode": float(torch.quantile((a - ref).norm(dim=1), 0.9)),
            "pf_eq15_rms_to_ode": float(((b - ref) ** 2).sum(1).mean().sqrt()),
            "ddim_vs_euler_bar_maxabs": float((a - c).abs().max()),
            "ddim_label_agree_ode": float((gmm.label(a) == ref_lab).double().mean()),
            "pf_label_agree_ode": float((gmm.label(b) == ref_lab).double().mean()),
        }
    xs = np.log([5, 10, 20, 50, 100, 200, 500, 1000])
    ys = np.log([out[str(S)]["ddim_rms_to_ode"] for S in [5, 10, 20, 50, 100, 200, 500, 1000]])
    out["loglog_slope_ddim_S10_to_1000"] = float(np.polyfit(xs[1:], ys[1:], 1)[0])
    ym = np.log([out[str(S)]["ddim_median_err_to_ode"] for S in [5, 10, 20, 50, 100, 200, 500, 1000]])
    out["loglog_slope_ddim_median_S10_to_1000"] = float(np.polyfit(xs[1:], ym[1:], 1)[0])
    out["loglog_slope_ddim_median_S10_to_200"] = float(np.polyfit(xs[1:6], ym[1:6], 1)[0])
    # 같은 x_T, 다른 S의 일관성(논문 5.2): S=1000 결과 대비
    x1000 = sample(eps_fn, xT, make_taus(T, 1000), ab, eta=0.0)
    cons = {}
    for S in [10, 20, 50, 100]:
        x0 = sample(eps_fn, xT, make_taus(T, S), ab, eta=0.0)
        cons[str(S)] = {"rms_to_S1000": float(((x0 - x1000) ** 2).sum(1).mean().sqrt()),
                        "label_agree_S1000": float((gmm.label(x0) == gmm.label(x1000)).double().mean())}
        # 같은 x_T에 eta=1 이면 잡음이 달라 서로 다른 결과: 같은 S에서 시드만 바꾼 두 번의 라벨 일치
        y1 = sample(eps_fn, xT, make_taus(T, S), ab, eta=1.0, gen=_gen(31))
        y2 = sample(eps_fn, xT, make_taus(T, S), ab, eta=1.0, gen=_gen(32))
        cons[str(S)]["eta1_label_agree_two_noise_seeds"] = float((gmm.label(y1) == gmm.label(y2)).double().mean())
        y1000 = sample(eps_fn, xT, make_taus(T, 1000), ab, eta=1.0, gen=_gen(33))
        cons[str(S)]["eta1_label_agree_S1000"] = float((gmm.label(y1) == gmm.label(y1000)).double().mean())
    out["consistency"] = cons
    # S=1000 의 오차가 S=500 보다 훨씬 작은 이유 점검: S=1000 격자에서 tau=1 만 빼면?
    t_no1 = list(range(2, T + 1))
    xa = sample(eps_fn, xT, t_no1, ab, eta=0.0)
    out["S999_without_tau1_rms_to_ode"] = float(((xa - ref) ** 2).sum(1).mean().sqrt())
    out["ode_ref_label_agree_S1000"] = float((gmm.label(x1000) == ref_lab).double().mean())
    return out


def exp_reconstruction(gmm, ab, T=1000):
    """표 2의 토이판: x_0 -> x_T(eta=0 인코딩) -> x_0(디코딩), 같은 S로. 오차는 성분당 평균제곱오차를 데이터 분산으로 나눈 값."""
    eps_fn = gmm.eps
    x0 = gmm.sample(N, _gen(888))
    var = float(x0.var(0).mean())
    out = {"data_var_per_dim": var, "n": N}
    Ss = [10, 20, 50, 100, 200, 500, 1000]
    mses, latent_sw = [], []
    g = torch.randn(N, gmm.d, generator=_gen(889), dtype=DT)
    for S in Ss:
        taus = make_taus(T, S)
        xT = encode(eps_fn, x0, taus, ab)
        rec = sample(eps_fn, xT, taus, ab, eta=0.0)
        mse = float(((rec - x0) ** 2).mean())
        mses.append(mse / var)
        latent_sw.append(sliced_w1(xT, g))
        med = float((rec - x0).norm(dim=1).median())
        out[str(S)] = {"rel_mse": mse / var, "median_roundtrip_err": med, "latent_sw1_to_N01": latent_sw[-1],
                       "latent_mean_norm2": float((xT ** 2).sum(1).mean())}
    meds = [out[str(S)]["median_roundtrip_err"] for S in Ss]
    out["loglog_slope_median_err_S10_to_1000"] = float(np.polyfit(np.log(Ss), np.log(meds), 1)[0])
    out["loglog_slope_rel_mse"] = float(np.polyfit(np.log(Ss), np.log(mses), 1)[0])
    out["loglog_slope_rel_mse_S10_to_200"] = float(np.polyfit(np.log(Ss[:5]), np.log(mses[:5]), 1)[0])
    # 비교: 두 번 독립 N(0,I) 표본 사이 SW1 바닥
    floor = [sliced_w1(torch.randn(N, gmm.d, generator=_gen(900 + r), dtype=DT), torch.randn(N, gmm.d, generator=_gen(950 + r), dtype=DT)) for r in range(10)]
    out["latent_sw1_floor"] = bootstrap_ci(floor)
    # eta=1(DDPM)로는 같은 x_T 재구성이 안 된다: 잡음 두 번의 결과 차이
    xT = encode(eps_fn, x0, make_taus(T, 100), ab)
    r1 = sample(eps_fn, xT, make_taus(T, 100), ab, eta=1.0, gen=_gen(891))
    out["eta1_S100_rel_mse_decode_of_encoded"] = float(((r1 - x0) ** 2).mean()) / var
    return out


def slerp(z0, z1, alpha):
    cos = (z0 * z1).sum(-1, keepdim=True) / (z0.norm(dim=-1, keepdim=True) * z1.norm(dim=-1, keepdim=True))
    th = torch.acos(cos.clamp(-1 + 1e-12, 1 - 1e-12))
    return torch.sin((1 - alpha) * th) / torch.sin(th) * z0 + torch.sin(alpha * th) / torch.sin(th) * z1


def exp_interpolation(gmm, ab, T=1000, npairs=2000, na=21):
    """x_T 보간(논문 5.3, 부록 D.5의 slerp). 경로를 따라 디코딩한 점의 봉우리 번호가 몇 번 바뀌나."""
    eps_fn = gmm.eps
    z0 = torch.randn(npairs, gmm.d, generator=_gen(1), dtype=DT)
    z1 = torch.randn(npairs, gmm.d, generator=_gen(2), dtype=DT)
    alphas = torch.linspace(0, 1, na, dtype=DT)

    def labels_along(zpath, eta, S, seed):
        flat = zpath.reshape(-1, gmm.d)
        x0 = sample(eps_fn, flat, make_taus(T, S), ab, eta=eta, gen=_gen(seed))
        return gmm.label(x0).reshape(npairs, na)

    def stats(lab):
        sw = (lab[:, 1:] != lab[:, :-1]).sum(1).double()
        distinct = torch.tensor([len(set(row.tolist())) for row in lab], dtype=DT)
        no_revisit = (sw == distinct - 1).double()
        ends_diff = (lab[:, 0] != lab[:, -1])
        return {"mean_switches": bootstrap_ci(sw.numpy(), B=2000),
                "frac_no_revisit": bootstrap_ci(no_revisit.numpy(), B=2000),
                "frac_endpoints_differ": float(ends_diff.double().mean()),
                "frac_exactly_one_switch_given_ends_differ": float((sw[ends_diff] == 1).double().mean())}

    slerp_path = torch.stack([slerp(z0, z1, a) for a in alphas], 1)       # (npairs, na, d)
    lerp_path = torch.stack([(1 - a) * z0 + a * z1 for a in alphas], 1)
    out = {"npairs": npairs, "n_alpha": na, "S": 50}
    out["ddim_slerp"] = stats(labels_along(slerp_path, 0.0, 50, 0))
    out["ddim_lerp"] = stats(labels_along(lerp_path, 0.0, 50, 0))
    out["ddpm_eta1_slerp_fixed_xT_path"] = stats(labels_along(slerp_path, 1.0, 50, 7))
    iid = torch.randn(npairs, na, gmm.d, generator=_gen(3), dtype=DT)
    out["iid_xT_baseline"] = stats(labels_along(iid, 0.0, 50, 0))
    for S in (10, 20, 100):
        out[f"ddim_slerp_S{S}"] = stats(labels_along(slerp_path, 0.0, S, 0))
    return out


def exp_derivations(gmm, ab, T=1000):
    """식 유도의 수치 검산: Lemma 1, eta=1 = DDPM, 정리 1의 가중치, sigma_hat의 초과 분산, 격자 통계."""
    out = {}
    rng = np.random.default_rng(0)
    dm, dv = 0.0, 0.0
    for t in range(2, T + 1):
        sig = rng.uniform(0.05, 1.0) * math.sqrt(1 - float(ab[t - 1]))
        A, B = q_sigma_coefs(ab, t, sig)
        dm = max(dm, abs(A + B * math.sqrt(float(ab[t])) - math.sqrt(float(ab[t - 1]))))
        dv = max(dv, abs(B ** 2 * (1 - float(ab[t])) + sig ** 2 - (1 - float(ab[t - 1]))))
    out["lemma1_max_dev_mean_coef"], out["lemma1_max_dev_variance"] = dm, dv
    e_var, e_a, e_b = 0.0, 0.0, 0.0
    for t in range(2, T + 1):
        a, ap = float(ab[t]), float(ab[t - 1])
        beta = 1 - a / ap
        sig = sigma_eta(a, ap, 1.0)
        e_var = max(e_var, abs(sig ** 2 - beta * (1 - ap) / (1 - a)))
        A, B = q_sigma_coefs(ab, t, sig)
        e_a = max(e_a, abs(A - math.sqrt(ap) * beta / (1 - a)))
        e_b = max(e_b, abs(B - math.sqrt(a / ap) * (1 - ap) / (1 - a)))
    out["eta1_vs_ddpm_max_err"] = {"variance": e_var, "x0_coef": e_a, "xt_coef": e_b}
    g = {}
    for t in (2, 10, 100, 500, 1000):
        a, ap = float(ab[t]), float(ab[t - 1])
        sig1 = sigma_eta(a, ap, 1.0)
        row = {"gamma_true_eta1": theorem1_gamma(ab, t, sig1), "gamma_ho": ho_gamma(ab, t),
               "gamma_paper_d1": paper_gamma(ab, t, sig1, 1)}
        row["paper_over_true"] = row["gamma_paper_d1"] / row["gamma_true_eta1"]
        row["gamma_ratio_eta0.2_over_eta1"] = theorem1_gamma(ab, t, 0.2 * sig1) / row["gamma_true_eta1"]
        g[str(t)] = row
    out["theorem1_gamma"] = g
    # sigma_hat 의 초과 분산: x_{tau_{i-1}} 의 잡음 분산이 1-abar 를 넘는 양 = sigma_hat^2 - sigma_eta(1)^2
    ex = {}
    for name, taus in [("linear_S10", make_taus(T, 10)), ("official_uniform_S10", make_taus_official(T, 10, "uniform")),
                       ("linear_S100", make_taus(T, 100))]:
        prev = [0] + taus[:-1]
        rows = [(t, tp, sigma_hat(float(ab[t]), float(ab[tp])) ** 2 - sigma_eta(float(ab[t]), float(ab[tp]), 1.0) ** 2)
                for t, tp in zip(taus, prev)]
        ex[name] = {"max_excess_var": max(r[2] for r in rows),
                    "at_step_to": max(rows, key=lambda r: r[2])[1], "from": max(rows, key=lambda r: r[2])[0],
                    "sigma_hat_largest": math.sqrt(max(sigma_hat(float(ab[t]), float(ab[tp])) ** 2 for t, tp in zip(taus, prev)))}
    out["sigma_hat_excess_variance"] = ex
    gs = {}
    for S in (10, 20, 50, 100):
        gs[str(S)] = {"linear_tau": make_taus(T, S, "linear")[:3] + ["..."] + make_taus(T, S, "linear")[-1:],
                      "quad_unique": len(make_taus(T, S, "quad")),
                      "official_uniform_first_last": [make_taus_official(T, S, "uniform")[0], make_taus_official(T, S, "uniform")[-1]],
                      "official_quad_unique": len(set(make_taus_official(T, S, "quad"))),
                      "official_quad_last": make_taus_official(T, S, "quad")[-1],
                      "abar_official_uniform_start": float(ab[make_taus_official(T, S, "uniform")[-1]]),
                      "abar_official_quad_start": float(ab[make_taus_official(T, S, "quad")[-1]])}
    out["grids_stats"] = gs
    return out


def main(out_path=None, only=None):
    t0 = time.time()
    gmm, ab = default_gmm(), make_alpha_bar()
    p = Path(out_path) if out_path else Path(__file__).with_name("minimal_impl_results.json")
    res = {"setup": {"T": 1000, "beta": [1e-4, 0.02], "dim": 2, "weights": gmm.w.tolist(), "means": gmm.mu.tolist(),
                     "std": gmm.s.tolist(), "n_per_seed": N, "seeds": SEEDS,
                     "abar_T": float(ab[-1]), "abar_1": float(ab[1])}}
    if only:
        res = json.loads(p.read_text())
    for name, fn in [("sweep", exp_sweep), ("grids", exp_grids), ("ode_error", exp_ode_error),
                     ("reconstruction", exp_reconstruction), ("interpolation", exp_interpolation),
                     ("derivations", exp_derivations)]:
        if only and name not in only:
            continue
        ts = time.time()
        res[name] = fn(gmm, ab)
        print(f"{name}: {time.time() - ts:.1f}s", flush=True)
    if not only:
        res["runtime_s"] = time.time() - t0
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("saved", p.name, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    # 사용: python3 minimal_impl.py [결과.json] [실험이름,...]   (이름을 주면 그 실험만 다시 돌려 JSON에 덮어쓴다)
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2].split(",") if len(sys.argv) > 2 else None)
