"""MeanFlow의 CFG(식 13–21)를 클래스 조건 가우시안 혼합에서 검산한다. 클래스 = 혼합 성분. 학습 없음."""
import json
from pathlib import Path
import torch
from torch.func import jvp
from mf_exact import GMM, u_rk4, flow_dop, full

OUT = Path(__file__).with_name("cfg_results.json")
gen = torch.Generator().manual_seed(5)
ang = torch.arange(3) * 2 * torch.pi / 3
gm = GMM(1.5 * torch.stack([ang.cos(), ang.sin()], 1), s=0.6)   # 클래스끼리 겹치는 혼합
rms = lambda a: float(a.pow(2).mean().sqrt())
res = {}
c = 0

# 1. 각주 3: E_c[v_cfg(z,t|c)] = v(z,t)  (클래스 사전확률 균등)
z = torch.randn(256, 2, generator=gen) * 2
t = full(256, 0.5)
vk, logw = gm.comp_vel(z, t)
post = torch.softmax(logw, 1)                       # p(c | z_t)
for om in [1.5, 3.0]:
    vc_all = torch.stack([gm.v_cfg(z, t, k, om) for k in range(3)], 1)   # (B,3,d)
    avg_post = (post[:, :, None] * vc_all).sum(1)
    avg_prior = (gm.pi[None, :, None] * vc_all).sum(1)
    res[f"footnote3_omega={om}"] = {"max|E_{c~p(c|z)} v_cfg - v|": float((avg_post - gm.v(z, t)).abs().max()),
                                    "max|E_{c~prior} v_cfg - v| (논문 식 14의 표기를 사전확률로 읽을 때)": float((avg_prior - gm.v(z, t)).abs().max())}

# 2. 식 19: ṽ = ω vt + (1-ω) u(z,t,t) 의 z,c 조건부 기댓값 = v_cfg(z,t|c)   (u(z,t,t)=v 정확)
t1 = 0.6
n, M = 6, 20000
zq, *_ = gm.sample_zt(n, torch.tensor(t1), gen, comp=torch.full((n,), c))
sig2 = gm.sig2(torch.tensor(t1))
a = (1 - t1) * gm.s ** 2 / sig2
mean = gm.m[c] + a * (zq - (1 - t1) * gm.m[c])
var = gm.s ** 2 * t1 ** 2 / sig2
x = mean[:, None, :] + var ** 0.5 * torch.randn(n, M, 2, generator=gen)  # p(x | z, 클래스 c)
eps = (zq[:, None, :] - (1 - t1) * x) / t1
vt = (eps - x).mean(1)
for om in [1.5, 3.0]:
    v_t_target = om * vt + (1 - om) * gm.v(zq, full(n, t1))
    res[f"eq19_target_expectation_omega={om}"] = {
        "max|E[ṽ|z,c]-v_cfg|": float((v_t_target - gm.v_cfg(zq, full(n, t1), c, om)).abs().max()),
        "mc_std_of_mean_approx": float((eps - x).std(1).max() / M ** 0.5), "M": M, "seed": 5}

# 3. 식 15: 안내 장 v_cfg(.|c)의 평균 속도 u_cfg 가 항등식을 만족, 1-NFE = 다스텝
B = 128
r, tq = 0.1, 0.8
zt, *_ = gm.sample_zt(B, torch.tensor(tq), gen, comp=torch.full((B,), c))
for om in [1.0, 2.0, 4.0]:
    fld = lambda z_, t_, om=om: gm.v_cfg(z_, t_, c, om)
    f = lambda z_, r_, t_, fld=fld: u_rk4(fld, z_, r_, t_, N=128)
    rr, tt = full(B, r), full(B, tq)
    v = fld(zt, tt)
    u, jv = jvp(f, (zt, rr, tt), (v, torch.zeros_like(rr), torch.ones_like(tt)))
    res[f"eq15_identity_omega={om}"] = {"max_resid": float((jv - (v - u) / (tq - r)).abs().max())}

# 4. 식 20·21: 목표 혼합. 고정점 v_c^cfg = ω v_c + κ v_c^cfg + (1-ω-κ) v 의 해가 ω'=ω/(1-κ) 의 CFG
z = torch.randn(256, 2, generator=gen) * 2
t = full(256, 0.4)
rows = {}
for om_eff in [2.0, 2.5]:
    for kappa in [0.0, 0.5, 0.9]:
        om = (1 - kappa) * om_eff
        vc, v0 = gm.v_cls(z, t, c), gm.v(z, t)
        # v^cfg 에 대한 고정점 (선형): (1-κ) v^cfg = ω vc + (1-ω-κ) v0
        vcfg = (om * vc + (1 - om - kappa) * v0) / (1 - kappa)
        rows[f"w'={om_eff},kappa={kappa}"] = {"omega": om, "max|v^cfg - CFG(w')|": float((vcfg - gm.v_cfg(z, t, c, om_eff)).abs().max())}
res["eq20_21_effective_scale"] = rows

# 5. 안내 세기가 표본에 미치는 영향 (클래스 0, 다스텝 정확 ODE) + t 구간 제한 [0, t_end]
n = 4000
e = torch.randn(n, 2, generator=gen)


def sample_guided(om, t_end=1.0):
    fld = lambda z_, t_: gm.v_cfg(z_, t_, c, om)
    if t_end < 1.0:
        z1, _ = flow_dop(gm.v_cls_c if False else (lambda z_, t_: gm.v_cls(z_, t_, c)), e, 1.0, t_end, 1e-10, 1e-10)  # 구간 밖은 ω=1
        z0, _ = flow_dop(fld, z1, t_end, 0.0, 1e-10, 1e-10)
        return z0
    return flow_dop(fld, e, 1.0, 0.0, 1e-10, 1e-10)[0]


def stats(x):
    d = torch.cdist(x, gm.m)
    nearest = d.argmin(1)
    return {"frac_nearest_is_class0": float((nearest == c).float().mean()),
            "mean_dist_to_class0_mean": float((x - gm.m[c]).norm(dim=1).mean()),
            "per_axis_std": float(x.std(0).mean())}


samp = {}
for om in [1.0, 1.5, 2.0, 3.0, 5.0]:
    samp[f"omega={om}"] = stats(sample_guided(om))
    if om > 1:
        samp[f"omega={om}, guided only t<=0.75"] = stats(sample_guided(om, 0.75))
res["guided_samples_class0"] = samp
res["guided_samples_class0"]["_data_class0"] = stats(gm.sample_data(n, gen, comp=torch.full((n,), c))[0])
res["guided_samples_class0"]["_n"] = n

OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
