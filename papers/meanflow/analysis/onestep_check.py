"""정확한 u로 한 스텝 샘플링 z0 = z1 - u(z1,0,1) 이 다스텝 ODE 해와 같은지,
v로 한 스텝 Euler(한 스텝 flow matching)는 궤적이 휜 정도에 따라 얼마나 틀리는지. 학습 없음."""
import json
from pathlib import Path
import torch
from mf_exact import GMM, u_rk4, u_dop, flow_dop, full

OUT = Path(__file__).with_name("onestep_results.json")
gen = torch.Generator().manual_seed(1)
n = 2000


def sliced_w2(a, b, gen, L=200):
    th = torch.randn(L, a.shape[1], generator=gen)
    th = th / th.norm(dim=1, keepdim=True)
    pa, pb = (a @ th.T).sort(0).values, (b @ th.T).sort(0).values
    return float((pa - pb).pow(2).mean().sqrt())


def euler(field, e, N):
    z = e
    for i in range(N):
        t = 1 - i / N
        z = z - (1.0 / N) * field(z, full(len(z), t))
    return z


def straightness(field, e, steps=200):
    """chord/path 와 rectified-flow식 직진도 S = mean_t |v(z_t,t) - (e - x_end)|^2 (세밀한 Euler 경로)."""
    z, path, dev = e, torch.zeros(len(e)), []
    chord = None
    zs = [e]
    for i in range(steps):
        t = 1 - i / steps
        zn = z - (1 / steps) * field(z, full(len(z), t))
        path += (zn - z).norm(dim=1)
        z = zn
        zs.append(z)
    disp = e - z
    for i in range(steps):
        t = 1 - i / steps
        dev.append((field(zs[i], full(len(e), t)) - disp).pow(2).sum(1))
    return float((e - z).norm(dim=1).div(path).mean()), float(torch.stack(dev).mean())


res = {}
for R in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]:
    ang = torch.arange(3) * 2 * torch.pi / 3
    gm = GMM(R * torch.stack([ang.cos(), ang.sin()], 1), s=0.5)
    e = torch.randn(n, 2, generator=gen)
    x_ref, nf = flow_dop(gm.v, e, 1.0, 0.0)
    u01, _, _ = u_dop(gm.v, e, 0.0, 1.0)
    x_mf = e - u01                                   # 정확한 u 한 스텝
    x_mf4 = e - u_rk4(gm.v, e, full(n, 0.0), full(n, 1.0), N=64)
    data, _ = gm.sample_data(n, gen)
    row = {
        "mean_sep_R": R,
        "chord_over_path": (st := straightness(gm.v, e))[0],
        "S_velocity_deviation": st[1],
        "exact_u_dop_1step_vs_ode_maxerr(동어반복)": float((x_mf - x_ref).abs().max()),
        "rk4_u_N64_1step_vs_ode_maxerr": float((x_mf4 - x_ref).abs().max()),
        "ode_vs_data_sw2": sliced_w2(x_ref, data, gen),
        "data_vs_data_sw2": sliced_w2(data, gm.sample_data(n, gen)[0], gen),
        "data_std_per_axis": float(data.std(0).mean()),
    }
    for N in [1, 2, 4, 8, 16, 64]:
        xe = euler(gm.v, e, N)
        row[f"euler{N}_pointwise_rms"] = float((xe - x_ref).pow(2).sum(1).mean().sqrt())
        row[f"euler{N}_sw2_vs_ode"] = sliced_w2(xe, x_ref, gen)
    xe1 = euler(gm.v, e, 1)
    row["euler1_output_std"] = float(xe1.std(0).mean())
    row["euler1_output_minus_data_mean_max"] = float((xe1 - gm.m.mul(gm.pi[:, None]).sum(0)).abs().max())
    res[f"R={R}"] = row
    print(R, {k: round(v, 4) for k, v in row.items() if k.startswith(("chord","S_", "euler1_p", "euler1_sw", "euler8_sw", "rk4"))})
OUT.write_text(json.dumps(res, indent=1))
