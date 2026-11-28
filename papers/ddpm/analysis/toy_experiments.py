"""정확한 잡음 예측기 eps*로 DDPM을 돌려 보는 토이 실험. 학습 없음, CPU, float64.

E1. 표본기 비교: sigma_t^2 = beta_t vs beta~_t, 단계 수 10..1000 (2차원 혼합, 슬라이스 W1과 공분산 오차, 시드 8개, 짝 비교와 홀름 보정)
E2. 가우시안 데이터에서 표본기의 정확한 분산 (MC 없음)
E3. L_simple의 줄일 수 없는 바닥: 알고리즘 1을 eps*로 평가한 MC vs 해석값(가우시안 닫힌 식, 1차원 혼합 구적)
E4. 변분 하한 L = L_T + sum L_{t-1} + L_0 을 eps*로 계산하고 참 엔트로피와 비교, 단계 수에 따른 간격
E5. 식 (16): 같은 L을 'KL(q(x_{t-1}|x_t) || p) 의 합 + H(x_0)'로 계산해 식 (5)와 일치하는지 (1차원 혼합)
결과는 toy_results.json.
"""
import json
import math
from pathlib import Path

import torch
from scipy import stats

from minimal_impl import (DT, GMM, Schedule, bootstrap_ci, gaussian_sampler_variance, l_simple_samples, marginal_kl_terms,
                          random_directions, sample, sliced_w1, toy_mixture_1d, toy_mixture_2d, toy_mixture_2d_tight, vlb_terms)

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)
R = {}
STEPS = [10, 20, 50, 100, 250, 1000]
DATASETS = {"broad": toy_mixture_2d, "tight": toy_mixture_2d_tight}


def steps_for(T, S):
    return sorted({int(round(v)) for v in torch.linspace(1, T, S).tolist()})


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


# ---------- E1 ----------
def e1_sampler(name, n=10000, seeds=8, P=64):
    data = DATASETS[name]()
    full = Schedule.linear()
    m_true, c_true = data.moments()
    proj = random_directions(2, P, torch.Generator().manual_seed(7))
    out = {"n_per_seed": n, "seeds": seeds, "P_directions": P, "steps": {}}
    null_sw = []
    for sd in range(seeds):
        g1 = torch.Generator().manual_seed(1000 + sd)
        g2 = torch.Generator().manual_seed(2000 + sd)
        null_sw.append(sliced_w1(data.sample(n, g1), data.sample(n, g2), proj))
    out["null_sw_data_vs_data"] = dict(zip(("mean", "lo", "hi"), bootstrap_ci(null_sw)))
    raw = {}
    for S in STEPS:
        sch = full.respace(steps_for(1000, S))
        raw[S] = {}
        for mode in ("beta", "beta_tilde"):
            sws, cov_err, mean_err = [], [], []
            for sd in range(seeds):
                g = torch.Generator().manual_seed(sd)  # 두 모드가 같은 x_T와 z를 쓴다
                x = sample(data.eps_star, sch, n, 2, mode, g)
                ref = data.sample(n, torch.Generator().manual_seed(1000 + sd))
                sws.append(sliced_w1(x, ref, proj))
                cov_err.append(float((torch.cov(x.T) - c_true).norm()))
                mean_err.append(float((x.mean(0) - m_true).norm()))
            raw[S][mode] = {"sw": sws, "cov_err": cov_err, "mean_err": mean_err}
        row = {}
        for mode in ("beta", "beta_tilde"):
            row[mode] = {k: dict(zip(("mean", "lo", "hi"), bootstrap_ci(raw[S][mode][k]))) for k in ("sw", "cov_err", "mean_err")}
        out["steps"][S] = row
        print("E1", name, "steps", S, "done", flush=True)
    # 짝 비교 (beta - beta_tilde), 시드 8개, 짝 t 검정, 단계 수 6개에 대한 홀름
    for metric in ("sw", "cov_err"):
        ps, diffs = [], []
        for S in STEPS:
            d = torch.tensor(raw[S]["beta"][metric]) - torch.tensor(raw[S]["beta_tilde"][metric])
            diffs.append(dict(zip(("mean", "lo", "hi"), bootstrap_ci(d.tolist()))))
            ps.append(float(stats.ttest_rel(raw[S]["beta"][metric], raw[S]["beta_tilde"][metric]).pvalue))
        adj = holm(ps)
        out["paired_" + metric] = {str(S): {"diff_beta_minus_tilde": diffs[i], "p": ps[i], "p_holm": adj[i]} for i, S in enumerate(STEPS)}
    out["raw"] = {str(S): raw[S] for S in STEPS}
    R.setdefault("E1_sampler", {})[name] = out


# ---------- E2 ----------
def e2_gaussian_exact():
    full = Schedule.linear()
    rows = {}
    for S in STEPS:
        sch = full.respace(steps_for(1000, S))
        rows[S] = {}
        for s2 in (1.0, 0.25, 1e-2):
            rows[S][str(s2)] = {m: gaussian_sampler_variance(s2, sch, m) / s2 - 1 for m in ("beta", "beta_tilde")}
    R["E2_gaussian_exact_rel_var_error"] = rows


# ---------- E3 ----------
def quad_mse_1d(data, sched, idx, half=12.0, ng=200001):
    """1차원 데이터에서 t별 eps* 의 줄일 수 없는 MSE = abar/(1-abar) * E Var(x0|x_t) 를 x_t 구적으로 구한다."""
    x = torch.linspace(-half, half, ng, dtype=DT)[:, None]
    dx = float(x[1] - x[0])
    out = []
    for i in idx:
        ab = sched.abar[i]
        qt = data.at(float(ab))
        w = qt.log_prob(x).exp()
        cv = data.cond_var_trace(x, ab)
        out.append(float((w * cv).sum() * dx * ab / (1 - ab)))
    return out


def e3_floor():
    sch = Schedule.linear()
    out = {}
    # (a) N(0,1): 정확한 최소값 abar_t (eps* = sqrt(1-abar) x_t). 시드 20개 x 5e5로 z 점수의 분포까지 본다
    gauss = GMM([1.0], [[0.0]], [[[1.0]]])
    n, seeds = 500000, 20
    analytic = float(sch.abar.mean())
    zs, means = [], []
    for sd in range(seeds):
        loss, _ = l_simple_samples(gauss.eps_star, gauss, sch, n, torch.Generator().manual_seed(sd))
        means.append(float(loss.mean()))
        zs.append(float((loss.mean() - analytic) / (loss.std() / math.sqrt(n))))
    pooled = sum(means) / seeds
    sd_means = float(torch.tensor(means).std())
    out["gauss_unit"] = {"n_per_seed": n, "seeds": seeds, "analytic_mean_abar": analytic, "pooled_mc_mean": pooled,
                         "pooled_se": sd_means / math.sqrt(seeds), "z_mean": sum(zs) / seeds,
                         "z_sd": float(torch.tensor(zs).std()), "z_min": min(zs), "z_max": max(zs)}
    # (b) 1차원 혼합: MC(알고리즘 1) vs 구적. 시드 10개 x 4e5
    data = toy_mixture_1d()
    qm = quad_mse_1d(data, sch, list(range(sch.T)))
    analytic = sum(qm) / sch.T
    n, seeds = 400000, 10
    means, zs = [], []
    binsum = [[0.0, 0.0, 0] for _ in range(10)]
    for sd in range(seeds):
        loss, idx = l_simple_samples(data.eps_star, data, sch, n, torch.Generator().manual_seed(100 + sd))
        means.append(float(loss.mean()))
        zs.append(float((loss.mean() - analytic) / (loss.std() / math.sqrt(n))))
        for b in range(10):
            m = (idx >= b * 100) & (idx < (b + 1) * 100)
            binsum[b][0] += float(loss[m].sum())
            binsum[b][1] += float((loss[m] ** 2).sum())
            binsum[b][2] += int(m.sum())
    out["mixture_1d"] = {"n_per_seed": n, "seeds": seeds, "quadrature_mean": analytic,
                         "pooled_mc_mean": sum(means) / seeds, "pooled_se": float(torch.tensor(means).std()) / math.sqrt(seeds),
                         "z_mean": sum(zs) / seeds, "z_sd": float(torch.tensor(zs).std()), "z_min": min(zs), "z_max": max(zs)}
    bins = []
    for b in range(10):
        sm, sq, cnt = binsum[b]
        mc = sm / cnt
        sb = math.sqrt((sq / cnt - mc * mc) / cnt)
        qa = sum(qm[b * 100:(b + 1) * 100]) / 100
        bins.append({"t_range": [b * 100 + 1, (b + 1) * 100], "mc": mc, "se": sb, "quad": qa, "z": (mc - qa) / sb})
    out["mixture_1d_bins"] = bins
    out["mixture_1d_mse_by_t"] = {str(t): qm[t - 1] for t in (1, 2, 5, 10, 50, 100, 250, 500, 750, 1000)}
    # (c) 2차원 혼합: MC(알고리즘 1) vs 조건부 분산 닫힌 식(Rao-Blackwell)
    data2 = toy_mixture_2d()
    g = torch.Generator().manual_seed(13)
    n2 = 400000
    loss2, _ = l_simple_samples(data2.eps_star, data2, sch, n2, g)
    g = torch.Generator().manual_seed(14)
    rb = []
    for i in range(sch.T):
        x0 = data2.sample(2000, g)
        xt = torch.sqrt(sch.abar[i]) * x0 + torch.sqrt(1 - sch.abar[i]) * torch.randn(2000, 2, generator=g, dtype=DT)
        rb.append(float(data2.cond_var_trace(xt, sch.abar[i]).mean() * sch.abar[i] / (1 - sch.abar[i])))
    rb_mean = sum(rb) / sch.T
    out["mixture_2d"] = {"n": n2, "seed": 13, "mc_mean": float(loss2.mean()), "mc_se": float(loss2.std() / math.sqrt(n2)),
                         "rao_blackwell_mean": rb_mean, "rb_n_per_t": 2000, "rb_seed": 14,
                         "z_vs_rb": float((loss2.mean() - rb_mean) / (loss2.std() / math.sqrt(n2))),
                         "floor_by_t_rb": {str(t): rb[t - 1] for t in (1, 2, 5, 10, 50, 100, 250, 500, 750, 1000)}}
    # (d) 섭동: eps*를 (1+c) 배하면 손실이 c^2 E||eps*||^2 만큼 늘어난다 (직교성). 1차원 혼합, 같은 난수(CRN)
    g = torch.Generator().manual_seed(15)
    nn_ = 400000
    x0 = data.sample(nn_, g)
    ii = torch.randint(0, sch.T, (nn_,), generator=g)
    eps = torch.randn(nn_, 1, generator=g, dtype=DT)
    ab = sch.abar[ii]
    xt = torch.sqrt(ab)[:, None] * x0 + torch.sqrt(1 - ab)[:, None] * eps
    es = data.eps_star(xt, ab)
    base = ((eps - es) ** 2).sum(-1)
    pert = {}
    for c in (-0.2, -0.05, 0.05, 0.2):
        d = ((eps - (1 + c) * es) ** 2).sum(-1) - base
        pred = c * c * float((es ** 2).sum(-1).mean())
        pert[str(c)] = {"increase_mc": float(d.mean()), "se": float(d.std() / math.sqrt(nn_)), "predicted_c2_E_eps*2": pred}
    out["perturbation_1d"] = {"n": nn_, "seed": 15, "by_c": pert, "E_eps_star_sq": float((es ** 2).sum(-1).mean())}
    R["E3_floor"] = out


# ---------- E4 ----------
def e4_vlb(name, n5=10000, n16=20000):
    data = DATASETS[name]()
    H = data.entropy_quad()
    full = Schedule.linear()
    out = {"H_nat": H, "H_bits_per_dim": H / (data.D * LN2), "n_eq5_per_term": n5, "n_eq16_per_term": n16, "seeds": [21, 22], "by_steps": {}}
    by_t = {}
    for S in STEPS:
        sch = full.respace(steps_for(1000, S))
        for mode in ("beta", "beta_tilde"):
            r = vlb_terms(data, sch, mode, n5, torch.Generator().manual_seed(21))
            mid = float(r["L_mid"].sum())
            se5 = math.sqrt(float((r["se_mid"] ** 2).sum()) + r["se_0"] ** 2)
            total = r["L_T"] + mid + r["L_0"]
            m = marginal_kl_terms(data, sch, mode, n16, torch.Generator().manual_seed(22))
            gap16 = m["KL_T"] + float(m["terms"].sum())
            se16 = math.sqrt(float((m["se"] ** 2).sum()) + m["se_T"] ** 2)
            out["by_steps"].setdefault(str(S), {})[mode] = {
                "L_T": r["L_T"], "sum_L_t_minus_1": mid, "L_0": r["L_0"], "total_eq5_nat": total, "se_eq5_nat": se5,
                "gap_eq5_nat": total - H, "gap_eq16_nat": gap16, "se_eq16_nat": se16,
                "gap_eq16_bits_per_dim": gap16 / (data.D * LN2), "total_eq16_nat": H + gap16,
                "n_negative_marginal_kl_terms": int((m["terms"] < 0).sum())}
            if S == 1000:
                by_t[mode] = {"L_mid": [float(v) for v in r["L_mid"]], "marginal_kl": [float(v) for v in m["terms"]]}
        print("E4", name, "steps", S, "done", flush=True)
    out["T1000_by_t"] = by_t
    R.setdefault("E4_vlb", {})[name] = out


# ---------- E5 ----------
def e5_eq16(S=100, n=100000):
    data = toy_mixture_1d()
    sch = Schedule.linear().respace(steps_for(1000, S))
    H = data.entropy_quad()
    m = marginal_kl_terms(data, sch, "beta", n, torch.Generator().manual_seed(31))
    r = vlb_terms(data, sch, "beta", n, torch.Generator().manual_seed(32))
    total5 = r["L_T"] + float(r["L_mid"].sum()) + r["L_0"]
    se5 = math.sqrt(float((r["se_mid"] ** 2).sum()) + r["se_0"] ** 2)
    gap16 = m["KL_T"] + float(m["terms"].sum())
    R["E5_eq16"] = {"S": S, "n": n, "seeds": [31, 32], "H_nat": H, "KL_qT_N01": m["KL_T"],
                    "sum_marginal_KL": float(m["terms"].sum()), "total_eq16": H + gap16,
                    "se_eq16": math.sqrt(float((m["se"] ** 2).sum()) + m["se_T"] ** 2),
                    "total_eq5": total5, "se_eq5": se5, "min_marginal_KL_term": float(m["terms"].min()),
                    "n_terms": S, "n_negative_terms": int((m["terms"] < 0).sum()),
                    "most_negative_over_its_se": float((m["terms"] / m["se"]).min()), "gap_nat_eq16": gap16}


def main():
    e2_gaussian_exact()
    e3_floor()
    print("E3 done", flush=True)
    e5_eq16()
    print("E5 done", flush=True)
    for name in DATASETS:
        e1_sampler(name)
        e4_vlb(name)
    (HERE / "toy_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
