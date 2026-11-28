"""CFG 토이 실험(학습 없음, 정확한 점수). 결과는 toy_results.json.

E1  w 스윕: 충실도-다양성(정밀도/재현율, 분산 수축, 모드 비율, 지지집합 밖 비율, FD·IS 모사)과
    정확한 기울인 목표 p(x|c) p(c|x)^w 와의 불일치(sliced W1). DDIM과 조상 표집.
E2  단계 수 T 의존(w=2): 불일치가 T -> 무한대에서 사라지는가.
E3  일관된(consistent) 기울임 점수(오라클)와 CFG 비교: 불일치가 표집기가 아니라 안내 장에서 오는가.
E4  구간 안내(이후 문헌, Kynkaanniemi 외 2024의 아이디어를 토이에 옮긴 것): 어느 lambda 구간에서 안내하느냐.

실행: python toy_experiments.py [E1 E2 ...]   (인자 없으면 전부. CPU, 메모리 ~0.5 GB, 수 분)
"""
import json
import math
import sys
import time

import numpy as np
import torch
from scipy import stats

import minimal_impl as mi

torch.set_num_threads(2)
M = mi.default_toy()
SEEDS = list(range(5))
N_CLS = 10000          # 클래스당 표본 수
N_PR = 2000            # 정밀도/재현율용 부분표본
WS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
OUT = {}
CHI2_999 = 13.8155     # 자유도 2의 99.9% 분위수: 참 표본의 0.1%가 "지지집합 밖"으로 센다


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(d):
    return {k: boot_ci(v) for k, v in d.items()}


def out_of_support(x):
    d2 = ((x[:, None, :] - M.mu.reshape(1, -1, 2)) ** 2).sum(-1) / M.s2
    return (d2.min(1).values > CHI2_999).double().mean().item()


def minor_mass(x, c):
    """클래스 c의 두 성분 중 가중이 작은 쪽(성분 인덱스 argmin wm)에 가장 가까운 표본의 비율."""
    d = ((x[:, None, :] - M.mu[c][None]) ** 2).sum(-1)
    k = int(torch.argmin(M.wm[c]))
    return (d.argmin(1) == k).double().mean().item()


def eval_samples(xs, c, gen_ref):
    """xs: 클래스 c의 생성 표본(N_CLS,2). 클래스 지표를 돌려준다."""
    real = M.sample(c, N_PR, gen_ref)
    p, r = mi.knn_precision_recall(real, xs[:N_PR])
    return {
        "mean_x": xs.mean(0).tolist(),
        "var": xs.var(0).mean().item(),
        "oos": out_of_support(xs),
        "minor": minor_mass(xs, c),
        "prec": p, "rec": r,
        "max_r99": torch.quantile(xs.norm(dim=1), 0.99).item(),
    }


def pooled(xs_by_class, seed):
    """prior 비율로 합친 표본의 FD(2차원 가우시안 적합)와 IS 모사."""
    counts = [int(round(float(p) * N_CLS)) for p in M.pi]
    gen = torch.Generator().manual_seed(9000 + seed)
    real = torch.cat([M.sample(c, counts[c], gen) for c in range(M.C)])
    fake_by = [xs_by_class[c][:counts[c]] for c in range(M.C)]
    one, zero = torch.tensor(1.0), torch.tensor(0.0)
    fake = torch.cat(fake_by)
    mean_pc = torch.cat([torch.exp(M.log_post(fake_by[c], one, zero)[:, c]) for c in range(M.C)]).mean().item()
    return {"fd": mi.frechet_gauss(fake, real), "is": mi.inception_score_toy(M, fake_by), "mean_pc": mean_pc}


def run_sampler(name, c, w, T, seed, v=0.3, band=None, n=N_CLS):
    gen = torch.Generator().manual_seed(seed * 100 + c)    # w, 표집기마다 같은 z_1 (공통 난수)
    fn = mi.guided_fn(M, c, w, band)
    if name == "ddim":
        return mi.sample_ddim(fn, n, T, gen)
    return mi.sample_ancestral(fn, n, T, gen, v=v)


def target_stats(w):
    """정확한 기울인 목표의 같은 지표와 표본 잡음 바닥(독립 두 표본 사이 sW1)."""
    per, floors = [], []
    tgs = []
    for c in range(M.C):
        tg = mi.TiltedGrid(M, c, w)
        tgs.append(tg)
        gen = torch.Generator().manual_seed(777 + c)
        xs = tg.sample(N_CLS, gen)
        a, b = tg.sample(5000, gen), tg.sample(5000, gen)
        floors.append(mi.sliced_w1(a, b))
        per.append(eval_samples(xs, c, torch.Generator().manual_seed(555 + c)))
        per[-1]["_xs"] = xs
    pl = pooled([p.pop("_xs") for p in per], 99)
    return {"per_class": per, "pooled": pl, "floor_sw1": float(np.mean(floors))}, tgs


def e1():
    res = {"settings": {"seeds": SEEDS, "n_per_class": N_CLS, "T": 256, "v_ancestral": 0.3, "ws": WS}}
    tg_cache = {}
    for w in WS:
        res.setdefault("target", {})[str(w)], tg_cache[w] = target_stats(w)
    for name in ("ddim", "ancestral"):
        res[name] = {}
        for w in WS:
            acc = {}
            for sd in SEEDS:
                xs, per = [], []
                for c in range(M.C):
                    x = run_sampler(name, c, w, 256, sd)
                    xs.append(x)
                    m = eval_samples(x, c, torch.Generator().manual_seed(sd * 100 + 50 + c))
                    ref = tg_cache[w][c].sample(5000, torch.Generator().manual_seed(sd * 100 + 80 + c))
                    m["sw1"] = mi.sliced_w1(x[:5000], ref)
                    per.append(m)
                pl = pooled(xs, sd)
                row = {
                    "var": np.mean([p["var"] for p in per]), "oos": np.mean([p["oos"] for p in per]),
                    "prec": np.mean([p["prec"] for p in per]), "rec": np.mean([p["rec"] for p in per]),
                    "sw1": np.mean([p["sw1"] for p in per]),
                    "minor_c1": per[1]["minor"], "minor_c0": per[0]["minor"], "minor_c2": per[2]["minor"],
                    "mean_norm": np.mean([np.linalg.norm(p["mean_x"]) for p in per]),
                    "fd": pl["fd"], "is": pl["is"], "mean_pc": pl["mean_pc"],
                    "max_r99": np.mean([p["max_r99"] for p in per]),
                }
                for k, val in row.items():
                    acc.setdefault(k, []).append(float(val))
            res[name][str(w)] = {"raw": acc, "summary": summarize(acc)}
            s = res[name][str(w)]["summary"]
            print(f"E1 {name:9s} w={w:4.2f} var={s['var'][0]:.3f} oos={s['oos'][0]:.4f} prec={s['prec'][0]:.3f} "
                  f"rec={s['rec'][0]:.3f} sw1={s['sw1'][0]:.3f} minor_c1={s['minor_c1'][0]:.3f} "
                  f"fd={s['fd'][0]:.3f} is={s['is'][0]:.3f}", flush=True)
    # 표집기 간 쌍 비교(sW1, 시드 쌍), Holm 보정
    ps = []
    for w in WS:
        a = res["ddim"][str(w)]["raw"]["sw1"]
        b = res["ancestral"][str(w)]["raw"]["sw1"]
        ps.append(float(stats.ttest_rel(a, b).pvalue))
    order = np.argsort(ps)
    holm = [None] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        holm[i] = run
    res["ddim_vs_ancestral_sw1"] = {str(w): {"p_raw": p, "p_holm": h} for w, p, h in zip(WS, ps, holm)}
    # 단조성: 시드별 분산 수축이 w에서 단조인가(공통 난수)
    for name in ("ddim", "ancestral"):
        mono = []
        for k in range(len(SEEDS)):
            v = [res[name][str(w)]["raw"]["var"][k] for w in WS]
            mono.append(all(x > y for x, y in zip(v, v[1:])))
        res[f"{name}_var_monotone_all_seeds"] = bool(all(mono))
    OUT["E1"] = res


def e2():
    w = 2.0
    Ts = [4, 8, 16, 32, 64, 128, 256, 1024]
    tgs = [mi.TiltedGrid(M, c, w) for c in range(M.C)]
    tstat, _ = target_stats(w)
    res = {"w": w, "Ts": Ts, "target_var": float(np.mean([p["var"] for p in tstat["per_class"]])),
           "floor_sw1": tstat["floor_sw1"]}
    for name in ("ddim", "ancestral"):
        res[name] = {}
        for T in Ts:
            acc = {"sw1": [], "var": []}
            for sd in SEEDS:
                s_, v_ = [], []
                for c in range(M.C):
                    x = run_sampler(name, c, w, T, sd)
                    ref = tgs[c].sample(5000, torch.Generator().manual_seed(sd * 100 + 80 + c))
                    s_.append(mi.sliced_w1(x[:5000], ref))
                    v_.append(x.var(0).mean().item())
                acc["sw1"].append(float(np.mean(s_)))
                acc["var"].append(float(np.mean(v_)))
            res[name][str(T)] = {"raw": acc, "summary": summarize(acc)}
            print(f"E2 {name:9s} T={T:5d} sw1={res[name][str(T)]['summary']['sw1'][0]:.3f} "
                  f"var={res[name][str(T)]['summary']['var'][0]:.3f} (target var {res['target_var']:.3f})", flush=True)
    OUT["E2"] = res


def e3():
    res = {"seeds": [0, 1], "n": 2000, "grid": "R=6, G=80"}
    for w in (2.0, 4.0):
        tgs = [mi.TiltedGrid(M, c, w, R=6.0, G=80) for c in range(M.C)]
        floor = []
        for c in range(M.C):
            g = torch.Generator().manual_seed(31 + c)
            floor.append(mi.sliced_w1(tgs[c].sample(2000, g), tgs[c].sample(2000, g)))
        for T in (64, 128):
            acc = {"cfg": [], "oracle": [], "cfg_var": [], "oracle_var": [], "target_var": []}
            for sd in res["seeds"]:
                a, b, av, bv, tv = [], [], [], [], []
                for c in range(M.C):
                    gen = torch.Generator().manual_seed(sd * 100 + c)
                    z0 = torch.randn(2000, 2, generator=gen)
                    ref = tgs[c].sample(2000, torch.Generator().manual_seed(sd * 100 + 80 + c))
                    x1 = mi.sample_ddim(mi.guided_fn(M, c, w), 2000, T, None, z0=z0)
                    x2 = mi.sample_ddim(mi.oracle_tilted_eps_fn(tgs[c]), 2000, T, None, z0=z0)
                    a.append(mi.sliced_w1(x1, ref)); b.append(mi.sliced_w1(x2, ref))
                    av.append(x1.var(0).mean().item()); bv.append(x2.var(0).mean().item()); tv.append(ref.var(0).mean().item())
                for k, val in zip(("cfg", "oracle", "cfg_var", "oracle_var", "target_var"), (a, b, av, bv, tv)):
                    acc[k].append(float(np.mean(val)))
            res[f"w{w}_T{T}"] = {"raw": acc, "summary": summarize(acc), "floor_sw1": float(np.mean(floor))}
            s = res[f"w{w}_T{T}"]["summary"]
            print(f"E3 w={w} T={T} sw1 cfg={s['cfg'][0]:.3f} oracle={s['oracle'][0]:.3f} floor={np.mean(floor):.3f} "
                  f"var cfg={s['cfg_var'][0]:.3f} oracle={s['oracle_var'][0]:.3f} target={s['target_var'][0]:.3f}", flush=True)
    OUT["E3"] = res


def e4():
    lam_lo, lam_hi = -3.38, 2.55   # EDM sigma = exp(-lambda/2) 로 (0.28, 5.42] 에 해당
    bands = {"full": None, "mid_only": (lam_lo, lam_hi), "off_high_noise": (lam_lo, 99.0), "off_low_noise": (-99.0, lam_hi)}
    res = {"bands": {k: v for k, v in bands.items()}, "T": 256}
    for w in (2.0, 4.0):
        tgs = [mi.TiltedGrid(M, c, w) for c in range(M.C)]
        for bn, band in bands.items():
            acc = {}
            for sd in SEEDS:
                per = []
                for c in range(M.C):
                    x = run_sampler("ddim", c, w, 256, sd, band=band)
                    m = eval_samples(x, c, torch.Generator().manual_seed(sd * 100 + 50 + c))
                    ref = tgs[c].sample(5000, torch.Generator().manual_seed(sd * 100 + 80 + c))
                    m["sw1"] = mi.sliced_w1(x[:5000], ref)
                    per.append(m)
                row = {k: float(np.mean([p[k] for p in per])) for k in ("var", "oos", "prec", "rec", "sw1")}
                row["minor_c1"] = per[1]["minor"]
                for k, val in row.items():
                    acc.setdefault(k, []).append(val)
            res[f"w{w}_{bn}"] = {"raw": acc, "summary": summarize(acc)}
            s = res[f"w{w}_{bn}"]["summary"]
            print(f"E4 w={w} {bn:15s} var={s['var'][0]:.3f} oos={s['oos'][0]:.4f} prec={s['prec'][0]:.3f} "
                  f"rec={s['rec'][0]:.3f} sw1={s['sw1'][0]:.3f} minor_c1={s['minor_c1'][0]:.3f}", flush=True)
    OUT["E4"] = res


if __name__ == "__main__":
    which = sys.argv[1:] or ["E1", "E2", "E3", "E4"]
    t0 = time.time()
    for k in which:
        {"E1": e1, "E2": e2, "E3": e3, "E4": e4}[k]()
        print(f"{k} done {time.time() - t0:.0f}s", flush=True)
    path = "toy_results.json" if len(which) == 4 else f"toy_results_{'_'.join(which)}.json"
    with open(path, "w") as f:
        json.dump(OUT, f, indent=1)
