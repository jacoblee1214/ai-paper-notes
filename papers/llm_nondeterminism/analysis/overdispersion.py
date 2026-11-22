"""실행 간 정확도 변동이 문항 수만으로 예상되는 변동보다 큰가(과분산)를 본다.

조건(모델 x 과제 x shot x 배치) 114개, 각 10회 실행, 문항x실행 정답 행렬 M (recompute_results.json).

귀무가설
  A. "같은 정확도를 가진 독립 이항 실행": 실행마다 정답 수 ~ Binomial(n, p̂). 분산 n p̂ (1-p̂).
     문항이 매번 같으므로 현실과 맞지 않는 느슨한 기준이다(문항 난이도가 고정이라 분산이 더 작아야 정상).
  B. "문항별 정답률은 고정, 문항 사이 뒤집힘은 독립": 문항 q의 정답률 q̂_q(10회 비율)를 고정하고 각 실행에서 독립 베르누이.
     분산 Σ q̂(1-q̂) N/(N-1) (표본 q̂ 편향 보정). 순열 검정(문항 안에서 실행 순서를 섞음)으로 p값을 얻는다.
     관측 분산이 B보다 크면 한 실행 안의 여러 문항이 함께 움직인다(실행 단위 효과).
통계량: 실행별 정답 수의 표본 분산(ddof=1).

또 같은 조건을 두 번 돌린 34쌍(2024-12 배치, 2025-02 배치)은 배치 간 평균 정확도 차이를 실행 단위 순열 검정으로 본다.

실행: python3 -I analysis/overdispersion.py   (먼저 recompute_runs.py)
출력: analysis/overdispersion_results.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
OUT = ROOT / "analysis" / "overdispersion_results.json"
rng = np.random.default_rng(20241122)
P_PERM = 4000


def perm_var_stats(M, P=P_PERM):
    """문항(행) 안에서 실행 순서를 독립적으로 섞어 열 합의 분산과 범위 분포를 만든다."""
    nq, N = M.shape
    vs = np.empty(P)
    rg = np.empty(P)
    for i in range(P):
        idx = np.argsort(rng.random((nq, N)), axis=1)
        S = np.take_along_axis(M, idx, axis=1).sum(0)
        vs[i] = S.var(ddof=1)
        rg[i] = S.max() - S.min()
    return vs, rg


def bh(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj


rows = []
for r in RES:
    M = np.array(r["matrix"])
    nq, N = M.shape
    c = M.sum(0)
    p_hat = c.mean() / nq
    v_obs = c.var(ddof=1)
    v_a = nq * p_hat * (1 - p_hat)
    q = M.mean(1)
    v_b = float((q * (1 - q)).sum() * N / (N - 1))
    chi2_a = (N - 1) * v_obs / v_a if v_a > 0 else np.nan
    p_a = float(stats.chi2.sf(chi2_a, N - 1)) if v_a > 0 else np.nan
    mixed = int(((M.sum(1) > 0) & (M.sum(1) < N)).sum())
    row = dict(model=r["model"], task=r["task"], shots=str(r["shots"]), date=r["date"], n=nq, N=N,
               acc_mean=round(100 * p_hat, 2), range_pp=round(100 * (c.max() - c.min()) / nq, 2),
               sd_pp=round(100 * c.std(ddof=1) / nq, 3), sd_binom_pp=round(100 * np.sqrt(v_a) / nq, 3),
               ratio_A=round(v_obs / v_a, 4) if v_a > 0 else None, p_A=p_a,
               ratio_B=round(v_obs / v_b, 3) if v_b > 0 else None, mixed_questions=mixed,
               best_minus_worst_pp=round(100 * (r["best_count"] - r["worst_count"]) / nq, 1),
               tarr=round(100 * r["TACr"] / nq, 1), tara=round(100 * r["TACa"] / nq, 1))
    if mixed >= 1 and v_obs > 0:
        vs, rg = perm_var_stats(M)
        row["p_perm_var"] = float((1 + (vs >= v_obs - 1e-12).sum()) / (1 + P_PERM))
        row["p_perm_low"] = float((1 + (vs <= v_obs + 1e-12).sum()) / (1 + P_PERM))
        row["perm_range_mean_pp"] = round(100 * rg.mean() / nq, 2)
        # null A 아래에서 10개 독립 이항 추출의 기대 범위
        sim = rng.binomial(nq, p_hat, size=(20000, N))
        row["binom_range_mean_pp"] = round(100 * (sim.max(1) - sim.min(1)).mean() / nq, 2)
    else:
        row["p_perm_var"] = None
        row["p_perm_low"] = None
        row["perm_range_mean_pp"] = 0.0
        row["binom_range_mean_pp"] = None
    rows.append(row)

# 순열 검정(과분산 방향) BH
tested = [x for x in rows if x["p_perm_var"] is not None]
adj = bh([x["p_perm_var"] for x in tested])
for x, a in zip(tested, adj):
    x["p_perm_var_bh"] = float(a)

summary = dict(
    n_conditions=len(rows), n_tested=len(tested),
    A_ratio_median=float(np.median([x["ratio_A"] for x in rows if x["ratio_A"] is not None])),
    A_ratio_max=float(max(x["ratio_A"] for x in rows if x["ratio_A"] is not None)),
    A_overdispersed_p05=int(sum(1 for x in rows if x["p_A"] is not None and x["p_A"] < 0.05)),
    B_ratio_median=float(np.median([x["ratio_B"] for x in rows if x["ratio_B"] is not None])),
    B_over_p05=int(sum(1 for x in tested if x["p_perm_var"] < 0.05)),
    B_over_bh05=int(sum(1 for x in tested if x["p_perm_var_bh"] < 0.05)),
    B_expected_false_positive_p05=round(0.05 * len(tested), 1),
    B_under_p05=int(sum(1 for x in tested if x["p_perm_low"] < 0.05)),
    max_range_pp=max(x["range_pp"] for x in rows),
    max_range_cond=[(x["model"], x["task"], x["shots"], x["date"][:10], x["range_pp"]) for x in sorted(rows, key=lambda y: -y["range_pp"])[:5]],
    n_range_gt_binom_mean=int(sum(1 for x in rows if x["binom_range_mean_pp"] is not None and x["range_pp"] > x["binom_range_mean_pp"])),
    n_with_mixed=len(tested),
)

# 모델별 요약
by_model = {}
for m in sorted({x["model"] for x in rows}):
    sub = [x for x in tested if x["model"] == m]
    by_model[m] = dict(n=len(sub), over_p05=int(sum(1 for x in sub if x["p_perm_var"] < 0.05)),
                       median_ratio_B=float(np.median([x["ratio_B"] for x in sub])))
summary["by_model"] = by_model

# 배치 간 비교
pairs = []
keyf = lambda r: (r["model"], r["task"], str(r["shots"]))
groups = {}
for r in RES:
    groups.setdefault(keyf(r), []).append(r)
for k, g in groups.items():
    if len(g) < 2:
        continue
    g = sorted(g, key=lambda r: r["date"])
    a, b = g[0], g[-1]
    if a["N"] != 10 or b["N"] != 10:  # llama3-8b 전문회계 few의 5회 배치는 쌍 비교에서 뺀다
        g10 = [x for x in g if x["N"] == 10]
        if len(g10) < 2:
            continue
        a, b = g10[0], g10[-1]
    ca, cb = np.array(a["correct_count_per_run"]), np.array(b["correct_count_per_run"])
    na = a["n_questions"]
    diff = (cb.mean() - ca.mean()) / na * 100
    allc = np.concatenate([ca, cb])
    obs = abs(cb.mean() - ca.mean())
    cnt = 0
    PP = 20000
    for _ in range(PP):
        rng.shuffle(allc)
        if abs(allc[:10].mean() - allc[10:].mean()) >= obs - 1e-12:
            cnt += 1
    p_perm = (1 + cnt) / (1 + PP)
    # 문항 단위 대응 비교: 문항별 정답률 차이의 부호 뒤집기 검정
    qa, qb = np.array(a["matrix"]).mean(1), np.array(b["matrix"]).mean(1)
    d = qb - qa
    obs_q = abs(d.sum())
    flips = rng.choice([-1, 1], size=(20000, len(d)))
    p_q = float((1 + (np.abs((flips * d).sum(1)) >= obs_q - 1e-12).sum()) / 20001)
    pairs.append(dict(model=k[0], task=k[1], shots=k[2], date_a=a["date"][:10], date_b=b["date"][:10], top_p_a=a["top_p_k"], top_p_b=b["top_p_k"], n=na,
                      acc_a=round(100 * ca.mean() / na, 2), acc_b=round(100 * cb.mean() / na, 2), diff_pp=round(diff, 2),
                      p_runs=float(p_perm), p_questions=p_q,
                      tarr_a=round(100 * a["TACr"] / na, 1), tarr_b=round(100 * b["TACr"] / na, 1),
                      tara_a=round(100 * a["TACa"] / na, 1), tara_b=round(100 * b["TACa"] / na, 1)))
padj = bh([x["p_questions"] for x in pairs])
for x, a in zip(pairs, padj):
    x["p_questions_bh"] = float(a)
summary["pairs"] = dict(n=len(pairs), p_runs_lt05=int(sum(1 for x in pairs if x["p_runs"] < 0.05)),
                        p_questions_lt05=int(sum(1 for x in pairs if x["p_questions"] < 0.05)),
                        p_questions_bh_lt05=int(sum(1 for x in pairs if x["p_questions_bh"] < 0.05)),
                        median_abs_diff_pp=float(np.median([abs(x["diff_pp"]) for x in pairs])),
                        max_abs_diff_pp=float(max(abs(x["diff_pp"]) for x in pairs)),
                        tarr_abs_diff_median=float(np.median([abs(x["tarr_a"] - x["tarr_b"]) for x in pairs])),
                        tarr_abs_diff_max=float(max(abs(x["tarr_a"] - x["tarr_b"]) for x in pairs)))

OUT.write_text(json.dumps(dict(summary=summary, rows=rows, pairs=pairs), ensure_ascii=False, indent=1))
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
