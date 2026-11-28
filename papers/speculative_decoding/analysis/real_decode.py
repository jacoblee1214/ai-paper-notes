"""실제 Pythia 쌍으로 Algorithm 1을 끝까지 돌려(한 시퀀스씩, KV 캐시 없이 전체 재계산) 반복마다의 받아들인 수 n과 위치별 beta를 모은다.
알고리즘 안의 연산량·시간은 의미가 없고(캐시 없음), 보는 것은 통계다: n의 히스토그램 대 capped geometric(식 1의 i.i.d. 가정),
반복당 평균 토큰 수 대 식 (1), 위치별 평균 beta(i.i.d.이면 평평).
  - 프롬프트: real_alpha.py와 같은 pile-10k 앞 N개(32토큰), 64토큰 생성, 표준화는 T만(top-k 없음)
  - 신뢰구간: 프롬프트 단위 부트스트랩(10000회). 한 프롬프트의 반복들은 서로 의존하므로 반복이 아니라 프롬프트가 표본 단위.
결과: real_decode_results.json
실행: HF_HUB_OFFLINE=1 python3 real_decode.py [N_PROMPTS]  (CPU)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import minimal_impl as M
import real_common as R

N_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PROMPT_LEN, NEW = 32, 64
CONFIGS = [("70m", 1.0, 5), ("70m", 1.0, 1), ("70m", 1.0, 2), ("70m", 1.0, 3), ("70m", 1.0, 8), ("70m", 0, 5),
           ("160m", 1.0, 5), ("160m", 0, 5)]
OUT = Path(__file__).with_name("real_decode_results.json")


def analyze(runs, gamma):
    """runs: 프롬프트마다 [(n, beta_list), ...]. 반환: 식 (1) 가정 점검 통계."""
    P = len(runs)
    n_all = np.array([n for r in runs for n, _ in r]); beta_all = np.array([b for r in runs for _, b in r])   # [I, gamma]
    I = len(n_all)
    visited = np.arange(gamma)[None, :] <= np.minimum(n_all, gamma - 1)[:, None]    # 위치 j(0부터)는 j <= n일 때만 실제로 평가된다(거절 위치까지)
    alpha_first = float(beta_all[:, 0].mean())                                       # 첫 위치는 모든 반복이 방문
    alpha_visited = float(beta_all[visited].mean())
    pmf = M.capped_geometric_pmf(alpha_visited, gamma)
    emp = np.bincount(n_all, minlength=gamma + 1) / I
    # 반복 수준 카이제곱(반복들이 프롬프트 안에서 의존하므로 p값은 낙관적, TV를 주 지표로 쓴다)
    exp = np.array(pmf) * I
    ok = exp >= 5
    c = np.append(np.bincount(n_all, minlength=gamma + 1)[ok], np.bincount(n_all, minlength=gamma + 1)[~ok].sum())
    e = np.append(exp[ok], exp[~ok].sum())
    if e[-1] < 5:
        c, e = c[:-1], e[:-1]
    chi = float(((c - e) ** 2 / e).sum()); chi_p = float(stats.chi2.sf(chi, len(c) - 1 - 1))   # alpha를 데이터로 추정했으므로 자유도 -1
    # 프롬프트 단위 부트스트랩: 반복당 토큰 수, 식 (1) 예측과의 차이
    per_prompt = [(np.array([n + 1 for n, _ in r]), np.array([b for _, b in r])) for r in runs]
    def stat_fn(idx):
        toks = np.concatenate([per_prompt[i][0] for i in idx]); bet = np.concatenate([per_prompt[i][1] for i in idx])
        ns = toks - 1
        vis = np.arange(gamma)[None, :] <= np.minimum(ns, gamma - 1)[:, None]
        a = bet[vis].mean()
        return toks.mean(), M.expected_tokens(a, gamma), toks.mean() - M.expected_tokens(a, gamma), a
    rng = np.random.default_rng(0)
    boots = np.array([stat_fn(rng.integers(0, P, P)) for _ in range(5000)])
    pt = stat_fn(np.arange(P))
    ci = lambda col: [float(np.quantile(boots[:, col], 0.025)), float(np.quantile(boots[:, col], 0.975))]
    # 위치별 평균 beta (방문한 반복만), 프롬프트 부트스트랩 신뢰구간
    pos = []
    for j in range(gamma):
        per = [np.array([b[j] for n, b in r if j <= min(n, gamma - 1)]) for r in runs]
        per = [x for x in per if len(x)]
        vals = np.concatenate(per)
        bs = [np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]).mean() for _ in range(2000)]
        pos.append(dict(position=j + 1, n_iter=int(len(vals)), mean_beta=float(vals.mean()),
                        ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]))
    # 위치 1의 beta와 첫 거절 이후... 의존성: 직전 위치가 받아들여졌을 때 다음 위치 beta의 평균(방문 반복 안)
    return dict(iterations=int(I), prompts=P, tokens_per_iter=float(pt[0]), tokens_per_iter_ci95=ci(0),
                eq1_prediction=float(pt[1]), eq1_ci95=ci(1), diff=float(pt[2]), diff_ci95=ci(2),
                alpha_first_position=alpha_first, alpha_visited=alpha_visited, alpha_visited_ci95=ci(3),
                n_hist_emp=emp.tolist(), n_hist_geometric=pmf, tv_n_hist_vs_geometric=float(0.5 * np.abs(emp - np.array(pmf)).sum()),
                chi2=chi, chi2_p_iter_level_optimistic=chi_p, beta_by_position=pos)


def main():
    t0 = time.time()
    tok = R.load_tok()
    prompts = R.load_prompts(tok, 100, PROMPT_LEN, seed=0)[:N_PROMPTS]
    draft = R.load_model("14m")
    results, models = {}, {}
    for size, T, gamma in CONFIGS:
        if size not in models:
            models = {size: R.load_model(size)}
        draft_fn, target_fn = R.make_fns(draft, models[size], T)
        runs, texts = [], []
        for i in range(N_PROMPTS):
            gen = torch.Generator().manual_seed(10_000 + i)
            seq, stats_ = M.speculative_decode(draft_fn, target_fn, prompts[i : i + 1], NEW, gamma, gen)
            runs.append(stats_); texts.append(seq)
        key = f"target={size},T={T},gamma={gamma}"
        results[key] = dict(target=size, T=T, gamma=gamma, **analyze(runs, gamma))
        results[key]["distinct_token_frac"] = len({t for s in texts for t in s}) / sum(len(s) for s in texts)
        r = results[key]
        print(key, "tok/iter", round(r["tokens_per_iter"], 3), [round(x, 3) for x in r["tokens_per_iter_ci95"]],
              "eq1", round(r["eq1_prediction"], 3), "alpha", round(r["alpha_visited"], 3), "TV", round(r["tv_n_hist_vs_geometric"], 3),
              "t", round(time.time() - t0), flush=True)
        OUT.write_text(json.dumps(dict(results, _meta=dict(n_prompts=N_PROMPTS, prompt_len=PROMPT_LEN, new_tokens=NEW, draft="pythia-14m",
                                                          seed_base=10_000, seconds=round(time.time() - t0))), indent=1))


if __name__ == "__main__":
    main()
