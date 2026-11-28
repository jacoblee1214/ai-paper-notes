"""MeanFlow 해석 검산용 공통 모듈: 가우시안 혼합 데이터의 닫힌 형태 속도장과 정확한 평균 속도.

선형(rectified-flow) 경로 z_t = (1-t) x + t eps, x ~ sum_k pi_k N(m_k, s^2 I), eps ~ N(0, I).
모델 학습은 하지 않는다. 모든 계산은 float64, CPU.
"""
import math
import numpy as np
import torch
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)


class GMM:
    def __init__(self, means, s, pi=None):
        self.m = torch.as_tensor(means, dtype=torch.float64)  # (K, d)
        self.K, self.d = self.m.shape
        self.s = float(s)
        self.pi = torch.full((self.K,), 1.0 / self.K) if pi is None else torch.as_tensor(pi, dtype=torch.float64)

    # ---- 경로 통계 -------------------------------------------------
    def sig2(self, t):
        return (1 - t) ** 2 * self.s ** 2 + t ** 2

    def comp_vel(self, z, t):
        """성분별 주변 속도 v_k(z,t) (B,K,d)와 로그 책임도 (B,K). t: (B,)"""
        t = t.reshape(-1, 1, 1)
        sig2 = self.sig2(t)
        diff = z[:, None, :] - (1 - t) * self.m[None]
        coef = (t - (1 - t) * self.s ** 2) / sig2
        vk = coef * diff - self.m[None]
        logw = torch.log(self.pi)[None] - 0.5 * self.d * torch.log(sig2[:, :, 0]) \
            - 0.5 * (diff ** 2).sum(-1) / sig2[:, :, 0]
        return vk, logw

    def v(self, z, t):
        """비조건(전체 혼합) 주변 속도 v(z,t) = E[eps - x | z_t = z]."""
        vk, logw = self.comp_vel(z, t)
        w = torch.softmax(logw, dim=1)
        return (w[:, :, None] * vk).sum(1)

    def v_cls(self, z, t, c):
        """클래스 조건 속도 v(z,t|c) (성분 c가 한 클래스)."""
        vk, _ = self.comp_vel(z, t)
        return vk[:, c, :]

    def v_cfg(self, z, t, c, omega):
        return omega * self.v_cls(z, t, c) + (1 - omega) * self.v(z, t)

    # ---- 표본 -------------------------------------------------------
    def sample_data(self, n, gen, comp=None):
        if comp is None:
            comp = torch.multinomial(self.pi, n, replacement=True, generator=gen)
        x = self.m[comp] + self.s * torch.randn(n, self.d, generator=gen)
        return x, comp

    def sample_zt(self, n, t, gen, comp=None):
        x, comp = self.sample_data(n, gen, comp)
        eps = torch.randn(n, self.d, generator=gen)
        tt = t.reshape(-1, 1) if torch.is_tensor(t) else t
        return (1 - tt) * x + tt * eps, x, eps, comp


# ---- 정확한 평균 속도 -------------------------------------------------
def u_rk4(field, z, r, t, N=64):
    """u(z,r,t) = (1/(t-r)) int_r^t v dtau 를 RK4(N 스텝)로 계산. r=t여도 0/0 없이 v를 돌려준다.
    field(z, tau) -> (B,d), tau: (B,). r, t: (B,)."""
    h = ((t - r) / N)
    hh = h.reshape(-1, 1)
    zc, tau, acc = z, t, 0.0
    for _ in range(N):
        k1 = field(zc, tau)
        k2 = field(zc - 0.5 * hh * k1, tau - 0.5 * h)
        k3 = field(zc - 0.5 * hh * k2, tau - 0.5 * h)
        k4 = field(zc - hh * k3, tau - h)
        phi = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        acc = acc + phi
        zc = zc - hh * phi
        tau = tau - h
    return acc / N


def flow_dop(field, z, t0, t1, rtol=1e-13, atol=1e-13):
    """z(t0)=z 에서 ODE dz/dtau = field(z,tau)를 t0 -> t1 으로 DOP853 적분 (스칼라 시각)."""
    B, d = z.shape

    def rhs(tau, y):
        zz = torch.from_numpy(y.reshape(B, d))
        return field(zz, torch.full((B,), float(tau))).numpy().reshape(-1)

    sol = solve_ivp(rhs, (t0, t1), z.numpy().reshape(-1), method="DOP853", rtol=rtol, atol=atol)
    return torch.from_numpy(sol.y[:, -1].reshape(B, d)), sol.nfev


def u_dop(field, z, r, t, rtol=1e-13, atol=1e-13):
    """u = (z_t - z_r)/(t-r), z_r 은 DOP853으로 t -> r 적분 (r != t)."""
    zr, nfev = flow_dop(field, z, t, r, rtol, atol)
    return (z - zr) / (t - r), zr, nfev


def full(B, val):
    return torch.full((B,), float(val))
