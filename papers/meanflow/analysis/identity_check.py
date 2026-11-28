"""MeanFlow 항등식(식 6)·JVP(식 8)·반군 성질·경계 조건을 해석적 가우시안 혼합에서 검산한다. 학습 없음."""
import json
from pathlib import Path
import torch
from torch.func import jvp
from mf_exact import GMM, u_rk4, u_dop, flow_dop, full

OUT = Path(__file__).with_name("identity_results.json")
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
res = {}

ang = torch.arange(3) * 2 * torch.pi / 3
gm = GMM(3.0 * torch.stack([ang.cos(), ang.sin()], 1), s=0.5)
rms = lambda a: float(a.pow(2).mean().sqrt())
mx = lambda a: float(a.abs().max())

# 0. 적분기 검증: 단일 가우시안은 유량 맵이 닫힌 형태
g1 = GMM([[1.5, -0.5]], s=0.5)
B = 64
t0, r0 = 0.9, 0.1
z, *_ = g1.sample_zt(B, torch.tensor(t0), gen)
zr_closed = (1 - r0) * g1.m[0] + (g1.sig2(torch.tensor(r0)) / g1.sig2(torch.tensor(t0))).sqrt() * (z - (1 - t0) * g1.m[0])
u_closed = (z - zr_closed) / (t0 - r0)
u4 = u_rk4(g1.v, z, full(B, r0), full(B, t0), N=64)
ud, _, nfev = u_dop(g1.v, z, r0, t0)
res["solver_vs_closed_form"] = {"rk4_N64_maxerr": mx(u4 - u_closed), "dop853_maxerr": mx(ud - u_closed), "dop853_nfev": nfev}

# 1. 항등식 잔차: RK4 스텝 수에 따른 수렴 (r, t) 쌍들
pairs = [(0.0, 1.0), (0.2, 0.7), (0.5, 0.9), (0.1, 0.3)]
B = 128
conv = {}
for (r, t) in pairs:
    zt, *_ = gm.sample_zt(B, torch.tensor(t), gen)
    rr, tt = full(B, r), full(B, t)
    row = {}
    for N in [2, 4, 8, 16, 32, 64, 128]:
        f = lambda z_, r_, t_: u_rk4(gm.v, z_, r_, t_, N=N)
        v = gm.v(zt, tt)
        u, dudt = jvp(f, (zt, rr, tt), (v, torch.zeros_like(rr), torch.ones_like(tt)))
        lhs = dudt
        rhs = (v - u) / (t - r)
        row[N] = {"max_resid": mx(lhs - rhs), "rms_resid": rms(lhs - rhs)}
    conv[f"r={r},t={t}"] = row
res["identity_residual_vs_rk4_steps"] = conv

# 1b. 독립 확인: DOP853 궤적 위의 중심 유한차분 (RK4 JVP와 무관)
B = 24
fd = {}
for (r, t) in [(0.2, 0.7), (0.1, 0.5)]:
    zt, *_ = gm.sample_zt(B, torch.tensor(t), gen)
    dl = 1e-4
    zp, _ = flow_dop(gm.v, zt, t, t + dl)
    zm, _ = flow_dop(gm.v, zt, t, t - dl)
    up, _, _ = u_dop(gm.v, zp, r, t + dl)
    um, _, _ = u_dop(gm.v, zm, r, t - dl)
    ut, _, _ = u_dop(gm.v, zt, r, t)
    v = gm.v(zt, full(B, t))
    dudt_fd = (up - um) / (2 * dl)
    rhs = (v - ut) / (t - r)
    fd[f"r={r},t={t}"] = {"max_resid": mx(dudt_fd - rhs), "typical_|dudt|": rms(rhs)}
res["identity_vs_dop853_central_difference"] = fd

# 2. 식 8: JVP = v dz u + dt u  (자동 미분 야코비안으로 직접 전개)
B = 16
r, t = 0.2, 0.8
zt, *_ = gm.sample_zt(B, torch.tensor(t), gen)
rr, tt = full(B, r), full(B, t)
f = lambda z_, r_, t_: u_rk4(gm.v, z_, r_, t_, N=32)
v = gm.v(zt, tt)
_, jv = jvp(f, (zt, rr, tt), (v, torch.zeros_like(rr), torch.ones_like(tt)))
# 표본마다 독립이므로 합의 야코비안의 대각 블록을 쓴다
Jz = torch.autograd.functional.jacobian(lambda z_: f(z_, rr, tt).sum(0), zt)  # (d, B, d) -> [out, b, in]
dtu = torch.autograd.functional.jacobian(lambda t_: f(zt, rr, t_).sum(0), tt)  # (d, B)
manual = torch.einsum("obi,bi->bo", Jz, v) + dtu.T
res["eq8_jvp_equals_explicit_jacobian"] = {"max_diff": mx(jv - manual)}

# 3. 반군(일관성) 성질
B = 128
r, s, t = 0.1, 0.45, 0.9
zt, *_ = gm.sample_zt(B, torch.tensor(t), gen)
u_rt, _, _ = u_dop(gm.v, zt, r, t)
u_st, zs, _ = u_dop(gm.v, zt, s, t)           # z_s 는 같은 궤적 위의 점
u_rs, _, _ = u_dop(gm.v, zs, r, s)
lhs = (t - r) * u_rt
rhs = (s - r) * u_rs + (t - s) * u_st
res["semigroup_consistency"] = {"max_resid": mx(lhs - rhs), "typical_|(t-r)u|": rms(lhs)}

# 4. 경계: r = t 이면 u = v
B = 64
t = 0.37
zt, *_ = gm.sample_zt(B, torch.tensor(t), gen)
u_b = u_rk4(gm.v, zt, full(B, t), full(B, t), N=8)
res["boundary_r_eq_t"] = {"max_|u-v|": mx(u_b - gm.v(zt, full(B, t)))}

# 5. dr u 항등식: d/dr u = (u - v(z_r, r)) / (t - r)  (z_t 고정)
B = 128
r, t = 0.15, 0.85
zt, *_ = gm.sample_zt(B, torch.tensor(t), gen)
rr, tt = full(B, r), full(B, t)
f = lambda z_, r_, t_: u_rk4(gm.v, z_, r_, t_, N=128)
z0 = torch.zeros_like(zt)
u, dru = jvp(f, (zt, rr, tt), (z0, torch.ones_like(rr), torch.zeros_like(tt)))
zr = zt - (t - r) * u
rhs = (u - gm.v(zr, rr)) / (t - r)
res["dr_identity"] = {"max_resid": mx(dru - rhs), "typical_|dru|": rms(rhs)}

# 6. 잘못된 접선(표 1b 대응): 목표 u_tgt = v - (t-r) JVP 와 정확한 u 의 차이
B = 512
r, t = 0.1, 0.9
zt, *_ = gm.sample_zt(B, torch.tensor(t), gen)
rr, tt = full(B, r), full(B, t)
f = lambda z_, r_, t_: u_rk4(gm.v, z_, r_, t_, N=64)
v = gm.v(zt, tt)
tang = {"(v,0,1)": (1, 0, 1), "(v,0,0)": (1, 0, 0), "(v,1,0)": (1, 1, 0), "(v,1,1)": (1, 1, 1)}
tv = {}
for name, (a, b, c) in tang.items():
    u, jv = jvp(f, (zt, rr, tt), (a * v, b * torch.ones_like(rr), c * torch.ones_like(tt)))
    tgt = v - (t - r) * jv
    tv[name] = {"rms_target_error": rms(tgt - u), "rms_u": rms(u)}
res["tangent_variants_target_error"] = tv

OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1)[:6000])
