"""합성 범주형 분포에서 (1) speculative sampling의 출력 분포 정확성, (2) Algorithm 1의 다중 토큰 정확성,
(3) 논문 식·표 재계산, (4) beta가 문맥에 따라 달라질 때 식 (1)의 어긋남을 확인한다. 결과: synthetic_results.json
실행: python3 run_synthetic.py   (CPU, 약 2분)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import minimal_impl as M

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).with_name("synthetic_results.json")


def dirichlet(V, conc, gen):
    x = torch._standard_gamma(torch.full((V,), conc), generator=gen)
    return x / x.sum()


def chi2(counts, probs):
    n = counts.sum(); exp = probs * n
    big = exp >= 5
    c = np.append(counts[big], counts[~big].sum()); e = np.append(exp[big], exp[~big].sum())
    if e[-1] < 5:
        c, e = c[:-1], e[:-1]
    stat = ((c - e) ** 2 / e).sum()
    if len(c) < 2:   # 분포가 한 점(one-hot)이면 모든 표본이 그 점이어야 한다
        return float(stat), 0, 1.0 if stat == 0 else 0.0
    return float(stat), int(len(c) - 1), float(stats.chi2.sf(stat, len(c) - 1))


def holm(pvals):
    """Holm-Bonferroni 보정 p값 (입력 순서 유지)."""
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def null_tv(probs, N, reps, rng):
    """분포 probs에서 N개를 뽑았을 때 경험 분포와의 TV의 평균·97.5% 분위수 (귀무 분포)."""
    t = [tv(rng.multinomial(N, probs) / N, probs) for _ in range(reps)]
    return float(np.mean(t)), float(np.quantile(t, 0.975))


# ---------------------------------------------------------------- 1. 토큰 하나
def scenarios(V, g):
    p, q = dirichlet(V, 0.5, g), dirichlet(V, 0.5, g)
    close = p + 0.05 * dirichlet(V, 1.0, g); close = close / close.sum()
    qz = q.clone(); qz[:3] = 0; qz = qz / qz.sum()
    return {
        "random(Dir 0.5)": (p, q),
        "close(q=p+0.05 noise)": (p, close),
        "q uniform": (p, torch.full((V,), 1 / V)),
        "sharp p (T=0.2)": (torch.softmax(torch.log(p) / 0.2, 0), q),
        "q has zeros where p>0": (p, qz),
        "disjoint support": (torch.tensor([.5, .3, .2, 0, 0, 0, 0, 0]), torch.tensor([0, 0, 0, .1, .2, .3, .2, .2])),
        "greedy (T=0: one-hot p, q)": (torch.eye(V)[int(p.argmax())], torch.eye(V)[int(q.argmax())]),
        "greedy p, soft q": (torch.eye(V)[int(p.argmax())], q),
    }


def single_token(N=2_000_000, chunk=500_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    V = 8
    rows, pvals = [], []
    for si, (name, (p, q)) in enumerate(scenarios(V, g).items()):
        gen = torch.Generator().manual_seed(1000 + si)
        counts = np.zeros(V, dtype=np.int64); acc_sum = 0
        for _ in range(N // chunk):
            x, acc = M.speculative_sampling(p.expand(chunk, -1), q.expand(chunk, -1), gen)
            counts += torch.bincount(x, minlength=V).numpy(); acc_sum += int(acc.sum())
        emp = counts / N
        stat, dof, pv = chi2(counts, p.numpy())
        alpha = float(M.alpha_from_dists(p, q))
        nm, nq = null_tv(p.numpy(), N, 200, rng)
        rows.append(dict(scenario=name, N=N, alpha=alpha, alpha_hat=acc_sum / N,
                         alpha_se=math.sqrt(max(alpha * (1 - alpha), 1e-12) / N), chi2=stat, dof=dof, p=pv,
                         tv_emp_vs_p=tv(emp, p.numpy()), tv_null_mean=nm, tv_null_q975=nq,
                         max_abs_dev=float(np.abs(emp - p.numpy()).max())))
        pvals.append(pv)
    for r, a in zip(rows, holm(pvals)):
        r["p_holm"] = a
    # 틀린 변형 (검정의 검출력): 같은 첫 시나리오 (random)
    p, q = scenarios(V, torch.Generator().manual_seed(seed))["random(Dir 0.5)"]
    wrong = {}
    variants = {"reject->resample from p (no residual)": dict(residual_fn=lambda a, b: a),
                "reject->resample from q": dict(residual_fn=lambda a, b: b),
                "always accept (lenience 0)": dict(lenience=0.0)}
    for k, (name, kw) in enumerate(variants.items()):
        gen = torch.Generator().manual_seed(2000 + k)
        counts = np.zeros(V, dtype=np.int64)
        for _ in range(N // chunk):
            x, _ = M.speculative_sampling(p.expand(chunk, -1), q.expand(chunk, -1), gen, **kw)
            counts += torch.bincount(x, minlength=V).numpy()
        stat, dof, pv = chi2(counts, p.numpy())
        wrong[name] = dict(chi2=stat, p=pv, tv_emp_vs_p=tv(counts / N, p.numpy()))
    return rows, wrong


# ---------------------------------------------------------------- 2. 다중 토큰
def random_markov(V, seed, conc):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([dirichlet(V, conc, g) for _ in range(V)])


def multi_token(N=600_000, V=5, K=3):
    P, Q = random_markov(V, 11, 0.5), random_markov(V, 12, 0.5)
    draft = lambda ids: Q[ids[:, -1]]
    target = lambda ids, k: P[ids[:, -k:]]
    start = 1
    exact = torch.zeros(V ** K)
    for cell in range(V ** K):
        d = [(cell // V ** (K - 1 - j)) % V for j in range(K)]
        pr = P[start, d[0]]
        for j in range(1, K):
            pr = pr * P[d[j - 1], d[j]]
        exact[cell] = pr
    rows, pvals = [], []
    for gamma in (0, 1, 3, 5):
        gen = torch.Generator().manual_seed(300 + gamma)
        t0 = time.time()
        out = M.sample_first_k(draft, target, torch.full((N, 1), start, dtype=torch.long), K, gamma, gen)
        cell = torch.zeros(N, dtype=torch.long)
        for j in range(K):
            cell = cell * V + out[:, j]
        counts = torch.bincount(cell, minlength=V ** K).numpy()
        stat, dof, pv = chi2(counts, exact.numpy())
        rows.append(dict(gamma=gamma, N=N, cells=V ** K, chi2=stat, dof=dof, p=pv,
                         tv_emp_vs_p=tv(counts / N, exact.numpy()), seconds=round(time.time() - t0, 1)))
        pvals.append(pv)
    for r, a in zip(rows, holm(pvals)):
        r["p_holm"] = a
    return rows


# ---------------------------------------------------------------- 3. 식·표 재계산
def formula_tables():
    out = {}
    # Table 1 (c = c_hat = 0)
    t1 = [(0.6, 2, 1.53, 1.96), (0.7, 3, 1.58, 2.53), (0.8, 2, 1.23, 2.44), (0.8, 5, 1.63, 3.69),
          (0.9, 2, 1.11, 2.71), (0.9, 10, 1.60, 6.86)]
    out["table1"] = [dict(alpha=a, gamma=g, ops_paper=o, speed_paper=s,
                          ops_ours=round(M.ops_factor(a, g, 0), 3), speed_ours=round(M.walltime_factor(a, g, 0), 3),
                          ops_match=abs(M.ops_factor(a, g, 0) - o) <= 0.005, speed_match=abs(M.walltime_factor(a, g, 0) - s) <= 0.005)
                     for a, g, o, s in t1]
    # Table 2 (alpha, gamma, speed EMP) 와 Table 4 (alpha, c, EXP, EMP), Table 3 alpha 교차
    t4 = [("EnDe", "T5-small", 0, 7, .75, .02, 3.2, 3.4), ("EnDe", "T5-base", 0, 7, .8, .04, 3.3, 2.8),
          ("EnDe", "T5-large", 0, 7, .82, .11, 2.5, 1.7), ("EnDe", "T5-small", 1, 7, .62, .02, 2.3, 2.6),
          ("EnDe", "T5-base", 1, 5, .68, .04, 2.4, 2.4), ("EnDe", "T5-large", 1, 3, .71, .11, 2.0, 1.4),
          ("CNNDM", "T5-small", 0, 5, .65, .02, 2.4, 3.1), ("CNNDM", "T5-base", 0, 5, .73, .04, 2.6, 3.0),
          ("CNNDM", "T5-large", 0, 3, .74, .11, 2.0, 2.2), ("CNNDM", "T5-small", 1, 5, .53, .02, 1.9, 2.3),
          ("CNNDM", "T5-base", 1, 3, .55, .04, 1.8, 2.2), ("CNNDM", "T5-large", 1, 3, .56, .11, 1.6, 1.7)]
    t2_speed = [3.4, 2.8, 1.7, 2.6, 2.4, 1.4, 3.1, 3.0, 2.2, 2.3, 2.2, 1.7]       # Table 2 속도 열
    t3_alpha = [.75, .8, .82, .62, .68, .71, .65, .73, .74, .53, .55, .56]          # Table 3 T5-XXL 행의 alpha (small/base/large)
    rows = []
    for (task, mq, T, g, a, c, exp, emp), s2, a3 in zip(t4, t2_speed, t3_alpha):
        ours = M.walltime_factor(a, g, c)
        og, ov = M.optimal_gamma(a, c)
        lo = min(M.walltime_factor(a + da, g, c + dc) for da in (-.005, .005) for dc in (-.005, .005))
        hi = max(M.walltime_factor(a + da, g, c + dc) for da in (-.005, .005) for dc in (-.005, .005))
        rows.append(dict(task=task, mq=mq, T=T, gamma=g, alpha=a, c=c, exp_paper=exp, exp_ours=round(ours, 3),
                         exp_match=abs(ours - exp) <= 0.05 + 1e-9,
                         exp_range_if_inputs_rounded=[round(lo, 3), round(hi, 3)],
                         exp_match_within_rounding=(lo - 0.05 - 1e-9 <= exp <= hi + 0.05 + 1e-9), emp=emp, emp_over_exp=round(emp / exp, 3),
                         table2_speed_equals_emp=(s2 == emp), table3_alpha_equals=(a3 == a),
                         optimal_gamma=og, optimal_factor=round(ov, 3), gain_vs_paper_gamma=round(ov / ours, 3)))
    out["table4"] = rows
    r = np.array([x["emp_over_exp"] for x in rows])
    out["table4_summary"] = dict(n=len(rows), exp_matches=sum(x["exp_match"] for x in rows),
                                 exp_matches_within_rounding=sum(x["exp_match_within_rounding"] for x in rows), emp_over_exp_min=float(r.min()),
                                 emp_over_exp_max=float(r.max()), emp_over_exp_median=float(np.median(r)),
                                 emp_gt_exp=int((r > 1).sum()), mean_abs_log_ratio=float(np.abs(np.log(r)).mean()))
    # 3.6절: 이중그램 alpha = 0.2, gamma = 3 -> 1.25X (c = 0)
    out["bigram_claim"] = dict(alpha=0.2, gamma=3, ours=round(M.walltime_factor(0.2, 3, 0), 4), paper=1.25)
    # 부록 A.5 lenience: Table 5 alpha, c = 0.015 -> 개선 2.5, 3.1, 3.6, 5 (T=1, gamma 불명) ; T=0 은 gamma=8 -> 3.3 3.3 3.9 4.9
    a5 = [(1, .62, 2.5), (.5, .71, 3.1), (.3, .76, 3.6), (.1, .84, 5.0)]
    out["table5_T1"] = [dict(l=l, alpha=a, paper=f, best_gamma=M.optimal_gamma(a, 0.015)[0],
                             ours_at_best_gamma=round(M.optimal_gamma(a, 0.015)[1], 3),
                             ours_c0_best_gamma=round(M.optimal_gamma(a, 0.0)[1], 3),
                             ours_by_gamma={g: round(M.walltime_factor(a, g, 0.015), 2) for g in (3, 5, 7, 8, 10)}) for l, a, f in a5]
    t0 = [(1, .75, 3.3), (.5, .75, 3.3), (.3, .8, 3.9), (.1, .87, 4.9)]
    out["lenience_T0_gamma8"] = [dict(l=l, alpha=a, paper=f, ours=round(M.walltime_factor(a, 8, 0.015), 3)) for l, a, f in t0]
    # 3.5절 최적 gamma 격자 (Fig. 3 재계산)
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    out["optimal_gamma_grid"] = {f"c={c}": {f"alpha={a}": M.optimal_gamma(a, c)[0] for a in alphas} for c in (0.01, 0.02, 0.05, 0.1)}
    out["optimal_factor_grid"] = {f"c={c}": {f"alpha={a}": round(M.optimal_gamma(a, c)[1], 3) for a in alphas} for c in (0.01, 0.02, 0.05, 0.1)}
    # Cor 3.9: 개선 가능 <=> alpha > c, 최소 개선 (1+alpha)/(1+c)
    viol = 0; grid = 0
    for a in np.linspace(0.05, 0.95, 19):
        for c in np.linspace(0.0, 1.0, 21):
            grid += 1
            best = M.optimal_gamma(float(a), float(c), 64)[1]
            if (best > 1 + 1e-12) != (a > c) and abs(a - c) > 1e-9:
                viol += 1
            if a > c and best < (1 + a) / (1 + c) - 1e-9:
                viol += 1
    out["cor_3_9"] = dict(grid_points=grid, violations=viol)
    # c = 0에서 gamma 무한대 극한 1/(1-alpha); 3.5절 오라클 gamma 상한 주장은 비교 대상이 모호해 식만 기록
    out["limit_c0"] = {f"alpha={a}": dict(gamma10=round(M.walltime_factor(a, 10, 0), 3), gamma_inf=round(1 / (1 - a), 3)) for a in alphas}
    # Fig.4: ops 배수 vs 속도 배수 (c=c_hat=0)
    out["fig4_grid"] = {f"gamma={g}": {f"alpha={a}": (round(M.walltime_factor(a, g, 0), 2), round(M.ops_factor(a, g, 0), 2)) for a in [0.5, 0.7, 0.9]} for g in (1, 3, 5, 7, 10)}
    return out


# ---------------------------------------------------------------- 4. beta가 문맥마다 다를 때
def heterogeneous_beta(N=1_000_000, V=6, gamma=5):
    """직전 토큰마다 p, q가 다른 마르코프 모델에서 한 번의 Algorithm 1: n의 분포를 식 (1)의 capped geometric와 비교.
    비교 기준 alpha는 시작 위치의 E[beta]. 문맥에 따라 beta가 크게 다르면 식 (1)이 어긋나는지 본다."""
    rows = []
    for conc, label in ((50.0, "homogeneous (Dir 50)"), (0.5, "mixed (Dir 0.5)"), (0.1, "very peaked (Dir 0.1)")):
        P, Q = random_markov(V, 41, conc), random_markov(V, 42, conc)
        draft = lambda ids: Q[ids[:, -1]]
        target = lambda ids, k: P[ids[:, -k:]]
        # 정지 분포로 시작 토큰을 뽑는다 (P의 정지 분포)
        w, v = torch.linalg.eig(P.T)
        pi = v[:, w.real.argmax()].real.abs(); pi = pi / pi.sum()
        gen = torch.Generator().manual_seed(500)
        start = torch.multinomial(pi, N, replacement=True, generator=gen)[:, None]
        res = M.speculative_step(draft, target, start, gamma, gen)
        alpha_stat = float((pi * M.accept_prob(P, Q)).sum())          # E[beta] under stationary start
        counts = torch.bincount(res["n"], minlength=gamma + 1).numpy()
        emp = counts / N; geo = np.array(M.capped_geometric_pmf(alpha_stat, gamma))
        rows.append(dict(setting=label, N=N, gamma=gamma, alpha_mean_beta=alpha_stat,
                         beta_std_over_states=float(torch.sqrt((pi * (M.accept_prob(P, Q) - alpha_stat) ** 2).sum())),
                         tokens_per_iter_emp=float((res["n"] + 1).double().mean()),
                         tokens_per_iter_eq1=M.expected_tokens(alpha_stat, gamma),
                         tv_n_hist_vs_geometric=tv(emp, geo), n_hist=emp.tolist(), n_geo=geo.tolist()))
    return rows


def main():
    t0 = time.time()
    res = {}
    res["single_token"], res["single_token_wrong_variants"] = single_token()
    print("single token done", round(time.time() - t0, 1), flush=True)
    res["multi_token"] = multi_token()
    print("multi token done", round(time.time() - t0, 1), flush=True)
    res["formulas"] = formula_tables()
    res["heterogeneous_beta"] = heterogeneous_beta()
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for r in res["single_token"]:
        print(f"{r['scenario']:30s} alpha {r['alpha']:.4f} hat {r['alpha_hat']:.4f} p {r['p']:.3f} holm {r['p_holm']:.3f} TV {r['tv_emp_vs_p']:.5f} (null {r['tv_null_mean']:.5f})")
    for k, v in res["single_token_wrong_variants"].items():
        print("WRONG", k, f"p={v['p']:.2e} TV={v['tv_emp_vs_p']:.4f}")
    for r in res["multi_token"]:
        print("multi", r)
    print(res["formulas"]["table4_summary"], res["formulas"]["cor_3_9"])
    for r in res["heterogeneous_beta"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("n_hist", "n_geo")})


if __name__ == "__main__":
    main()
