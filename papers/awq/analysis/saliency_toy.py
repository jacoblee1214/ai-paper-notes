"""논문 §3.2 식 2의 오차비(Delta'/Delta * 1/s)와 'Delta 불변' 가정이 깨지는 지점을 토이로 수치 확인한다.

그룹(128 입력 채널) 하나에 1개 채널(1/128 = 0.8%, 논문의 1%)을 s배 한다. 가중치 분포 3가지,
양자화기 2가지(식 1의 대칭 absmax, 공식 코드의 zero-point 비대칭), 4비트·3비트.
측정(그룹 수 n_groups, 시드 고정):
  p_changed   Delta' != Delta 인 그룹 비율
  mean_ratio  평균 Delta'/Delta
  err_ratio   (평균 Delta'/Delta) / s   (논문 Table 2 마지막 줄에 해당)
  sal_rel     s배 한 채널의 상대 오차 |Q(ws)/s - w| 평균 / 원래 |Q(w) - w| 평균 (이론 1/s)
  nonsal_rel  나머지 채널의 오차 평균 / 원래 평균 (Delta'/Delta 만큼 커지면 1보다 큼)
  out_mse     활성 크기 x_sal(그 채널)과 1(나머지)로 입력을 만든 출력 오차 MSE / s=1일 때 MSE
"""
import json
import sys

import torch

from minimal_impl import quantize_group, quantize_sym

N_GROUPS = 20000
GS = 128
SCALES = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0]


def draw(kind, g):
    if kind == "gauss":
        return torch.randn(N_GROUPS, GS, generator=g)
    if kind == "student3":      # 꼬리가 무거움(LLM 가중치의 이상치 흉내)
        z = torch.randn(N_GROUPS, GS, generator=g)
        chi = torch.distributions.Chi2(torch.tensor(3.0)).sample((N_GROUPS, GS))
        return z / (chi / 3).sqrt()
    if kind == "laplace":
        u = torch.rand(N_GROUPS, GS, generator=g) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs())
    raise ValueError(kind)


def deltas(w, n_bit, quant):
    """그룹별 Delta: 대칭이면 absmax/2^(n-1), 비대칭이면 (max-min)/(2^n-1)."""
    if quant == "sym":
        return w.abs().amax(-1) / 2 ** (n_bit - 1)
    return (w.amax(-1) - w.amin(-1)) / (2 ** n_bit - 1)


def run(kind, quant, n_bit, x_sal=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = draw(kind, g)
    qf = (lambda a: quantize_sym(a, n_bit, GS)[0]) if quant == "sym" else (lambda a: quantize_group(a, n_bit, GS))
    d0 = deltas(w, n_bit, quant)
    q0 = qf(w)
    e0_sal = (q0[:, 0] - w[:, 0]).abs().mean()
    e0_non = (q0[:, 1:] - w[:, 1:]).abs().mean()
    xv = torch.ones(GS)
    xv[0] = x_sal
    mse0 = (((q0 - w) * xv) .sum(-1) ** 2).mean()
    rows = []
    for s in SCALES:
        ws = w.clone()
        ws[:, 0] *= s
        d1 = deltas(ws, n_bit, quant)
        q1 = qf(ws)
        q1[:, 0] /= s                               # Q(w s) / s
        e_sal = (q1[:, 0] - w[:, 0]).abs().mean()
        e_non = (q1[:, 1:] - w[:, 1:]).abs().mean()
        mse = (((q1 - w) * xv).sum(-1) ** 2).mean()
        r = d1 / d0
        rows.append(dict(s=s, p_changed=float((d1 != d0).float().mean()), mean_ratio=float(r.mean()),
                         err_ratio=float(r.mean() / s), sal_rel=float(e_sal / e0_sal),
                         nonsal_rel=float(e_non / e0_non), out_mse=float(mse / mse0)))
    return rows


if __name__ == "__main__":
    out = {}
    for kind in ("gauss", "student3", "laplace"):
        for quant in ("sym", "asym"):
            for n_bit in (4, 3):
                out[f"{kind}/{quant}/W{n_bit}"] = run(kind, quant, n_bit)
    json.dump(dict(n_groups=N_GROUPS, group_size=GS, x_sal=10.0, seed=0, results=out),
              open("saliency_toy_results.json", "w"), indent=1)
    for k in ("gauss/sym/W4", "gauss/asym/W4", "student3/asym/W3"):
        print(k)
        for r in out[k]:
            print("  s=%-5g p_chg=%.3f  D'/D=%.3f  D'/D/s=%.3f  sal=%.3f  non=%.3f  out_mse=%.3f" % (
                r["s"], r["p_changed"], r["mean_ratio"], r["err_ratio"], r["sal_rel"], r["nonsal_rel"], r["out_mse"]))
