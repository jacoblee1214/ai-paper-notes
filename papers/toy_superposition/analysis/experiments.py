"""Toy Models of Superposition 최소 구현 실험. 사용: python experiments.py {a,b,c,geom} [--n400]

a    : n=20, m=5, I_i=0.7^i. 희소도(밀도 1-S)를 훑으며 ReLU 모델과 선형 모델 학습. 시드 5개.
b    : 2특징·1차원 단계도. (상대 중요도 x 밀도) 격자, 모집단 손실 최소화 vs 세 해의 이론.
c    : 균일 중요도, n=200, m=15 (논문 n=400, m=30의 절반). 밀도 훑기, 차원수 D_i. 시드 3개. --n400이면 n=400, m=30 한 시드.
geom : m=2, n=40, 균일 중요도: 평면 위 다각형(2/k)과 대척점 쌍.
결과는 results_<exp>.json. 학습 설정은 json의 config에 남긴다.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import minimal_impl as mi

HERE = Path(__file__).parent
torch.set_num_threads(6)
FRACS = {"1": 1.0, "3/4": 0.75, "2/3": 2 / 3, "1/2": 0.5, "2/5": 0.4, "3/8": 0.375, "1/3": 1 / 3, "0": 0.0}


def frac_hist(D, rep, tol=0.02):
    """표현된 특징의 D_i를 가장 가까운 분수에 배정(차이 < tol). 어디에도 안 들면 '기타'."""
    D = D[rep]
    names = [k for k in FRACS if k != "0"]
    vals = np.array([FRACS[k] for k in names])
    out = {k: 0 for k in names}
    other = 0
    for v in D:
        j = int(np.abs(vals - v).argmin())
        if abs(vals[j] - v) < tol:
            out[names[j]] += 1
        else:
            other += 1
    out["기타"] = other
    return out


def exp_a():
    n, m = 20, 5
    imp = mi.importance_curve(n, 0.7)
    dens = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01])
    steps, seeds = 20000, [0, 1, 2, 3, 4]
    res = dict(density=dens.tolist(), seeds=seeds, relu={}, linear={})
    for relu in (True, False):
        key = "relu" if relu else "linear"
        cnt, loss, norms2, inter, dimsf = [], [], [], [], []
        for sd in seeds:
            t = time.time()
            r = mi.train(n, m, dens, imp, relu=relu, steps=steps, seed=sd)
            W = r["W"]
            cnt.append(mi.count_represented(W).tolist())
            loss.append(r["loss"].tolist())
            norms2.append((mi.feature_norms(W) ** 2).tolist())
            inter.append(mi.interference(W).tolist())
            dimsf.append(mi.dims_per_feature(W).tolist())
            if sd == 0:
                res[f"{key}_config"] = r["config"]
                res[f"{key}_W_seed0"] = W.tolist()
                res[f"{key}_b_seed0"] = r["b"].tolist()
            print(key, "seed", sd, f"{time.time()-t:.0f}s", cnt[-1], flush=True)
        res[key] = dict(count=cnt, loss=loss, norm2=norms2, interference=inter, D_star=dimsf)
    # 선형 모델의 이론 최적 손실 (밀도별)
    res["linear_theory_loss"] = [mi.linear_optimal_loss(imp.numpy(), m, float(d)) for d in dens]
    json.dump(res, open(HERE / "results_a.json", "w"))


def exp_b():
    R = np.logspace(-1, 1, 13)       # 상대 중요도 0.1 .. 10 (r=1이 격자에 있음: 퇴화)
    R = np.array([0.1, 0.16, 0.25, 0.4, 0.63, 0.8, 1.25, 1.6, 2.5, 4.0, 6.3, 10.0])
    Dn = np.logspace(0, -2, 13)      # 밀도 1 .. 0.01
    t = time.time()
    learned = mi.learn_2feat_phase(R, Dn, n_init=6, steps=1200)
    print("learned", time.time() - t, flush=True)
    r_t = torch.tensor(R, dtype=torch.float64).view(-1, 1).expand(len(R), len(Dn)).contiguous()
    d_t = torch.tensor(Dn, dtype=torch.float64).view(1, -1).expand(len(R), len(Dn)).contiguous()
    th = mi.three_config_losses(r_t, d_t, True, quad=(256, 64))
    th0 = mi.three_config_losses(r_t, d_t, False)
    A, B, C = (th[k].numpy() for k in "ABC")
    A0, B0, C0 = (th0[k].numpy() for k in "ABC")
    cls_learn = mi.classify_2feat(learned["w"])
    cls_th = mi.classify_theory(A, B, C)
    cls_th0 = mi.classify_theory(A0, B0, C0)
    # 학습 분류는 0:특징1만 1:둘다 2:특징2만 3:안배움. 이론은 0,1,2 동일 의미
    res = dict(
        R=R.tolist(), density=Dn.tolist(), learn_config=learned["config"],
        learn_w=learned["w"].tolist(), learn_b=learned["b"].tolist(), learn_loss=learned["loss"].tolist(),
        theory_loss=dict(A=A.tolist(), B=B.tolist(), C=C.tolist()),
        cls_learn=cls_learn.tolist(), cls_theory=cls_th.tolist(), cls_theory_b0=cls_th0.tolist(),
        agree_theory=float((cls_learn == cls_th).mean()), agree_theory_b0=float((cls_learn == cls_th0).mean()),
        # 학습 손실이 세 해의 최소 손실보다 얼마나 낮은가(세 해가 실제 최적에 얼마나 가까운가)
        gap_learn_vs_theory=(np.minimum(np.minimum(A, B), C) - learned["loss"]).tolist(),
    )
    # 경계(밀도) 정밀 탐색: 고정 r에서 세 해 이론의 A/B -> C 전환
    fine_d = np.linspace(0.05, 0.95, 46)
    fb = {}
    for r in (0.25, 0.5, 2.0, 4.0, 1.0):
        rt = torch.full((len(fine_d),), r, dtype=torch.float64)
        dt = torch.tensor(fine_d, dtype=torch.float64)
        o = mi.three_config_losses(rt, dt, True, quad=(256, 64))
        a, b_, c = (o[k].numpy() for k in "ABC")
        lone = np.minimum(a, b_)
        sup = c < lone  # 대척점이 이김
        # 가장 큰 밀도에서 sup이 True인 값(경계 바로 아래)
        idx = np.where(sup)[0]
        fb[str(r)] = dict(
            boundary_opt_bias=float(fine_d[idx.max()]) if len(idx) else None,
            boundary_b0_closed_form=mi.b0_boundary_density(r),
        )
        print("boundary r", r, fb[str(r)], flush=True)
    res["boundary"] = fb
    json.dump(res, open(HERE / "results_b.json", "w"))


def run_uniform(n, m, dens, steps, seeds, tag, keepW=True):
    imp = torch.ones(n)
    out = dict(density=list(map(float, dens)), n=n, m=m, seeds=seeds, D_star=[], count=[], loss=[], frac=[], configs=None)
    for sd in seeds:
        t = time.time()
        r = mi.train(n, m, dens, imp, steps=steps, seed=sd)
        W = r["W"]
        D = mi.dimensionality(W).numpy()
        rep = (mi.feature_norms(W) ** 2 > 0.5).numpy()
        out["D_star"].append(mi.dims_per_feature(W).tolist())
        out["count"].append(mi.count_represented(W).tolist())
        out["loss"].append(r["loss"].tolist())
        out["frac"].append([frac_hist(D[i], rep[i]) for i in range(len(dens))])
        if sd == seeds[0]:
            out["config"] = r["config"]
            out["D_seed0"] = D.tolist()
            out["norm2_seed0"] = (mi.feature_norms(W) ** 2).tolist()
            if keepW:
                out["W_seed0"] = W.tolist()
        print(tag, "seed", sd, f"{time.time()-t:.0f}s", out["count"][-1], flush=True)
    return out


def exp_c(n400=False):
    dens = np.logspace(0, -2, 13)
    if n400:
        out = run_uniform(400, 30, dens, 8000, [0], "n400", keepW=False)
        json.dump(out, open(HERE / "results_c400.json", "w"))
    else:
        out = run_uniform(200, 15, dens, 10000, [0, 1, 2], "n200", keepW=False)
        json.dump(out, open(HERE / "results_c.json", "w"))


def exp_geom():
    dens = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01])
    out = run_uniform(40, 2, dens, 15000, [0, 1, 2], "m2")
    json.dump(out, open(HERE / "results_geom.json", "w"))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "a":
        exp_a()
    elif which == "b":
        exp_b()
    elif which == "c":
        exp_c("--n400" in sys.argv)
    elif which == "geom":
        exp_geom()
