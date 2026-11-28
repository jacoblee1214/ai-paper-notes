"""표본 수준 정확성 검사: 실제 Pythia 쌍(초안 14M, 목표 70M·160M)에서 speculative decoding이 목표 모델 단독 샘플링과 같은 분포를 내는가.
  T=1, top-k=10 표준화(초안·목표 모두). 처음 2토큰의 결합분포는 최대 100칸이라 목표 모델 정방향 최대 11번으로 정확한 확률을 구할 수 있다.
  - 방법 A: Algorithm 1(gamma=3)로 처음 2토큰 N개 표집
  - 방법 B(대조): 목표 모델 단독 샘플링 N개
  - 두 표본을 정확한 결합분포와 카이제곱(기대 5 미만 칸은 묶음), TV와 귀무 TV 분포 비교; 검정 전체에 Holm 보정
  - T=0: 초안이 무엇이든(실제 14M, 무작위 초안) 처음 16토큰이 목표의 greedy 경로와 정확히 같은지
결과: real_e2e_results.json
실행: HF_HUB_OFFLINE=1 python3 real_e2e.py  (CPU, 약 5분)
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import minimal_impl as M
import real_common as R

N, CH, GAMMA, TOPK, K = 8000, 200, 3, 10, 2
CASES = [("70m", 0), ("70m", 1), ("70m", 2), ("160m", 3)]       # (목표, 프롬프트 번호)
OUT = Path(__file__).with_name("real_e2e_results.json")


def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def exact_joint(target, prompt):
    p1 = R.probs_last(target, prompt, 1, 1.0, TOPK)[0, 0]
    joint = {}
    for x1 in p1.nonzero().squeeze(-1).tolist():
        p2 = R.probs_last(target, torch.cat([prompt, torch.tensor([[x1]])], 1), 1, 1.0, TOPK)[0, 0]
        for x2 in p2.nonzero().squeeze(-1).tolist():
            joint[(x1, x2)] = float(p1[x1] * p2[x2])
    return joint


def test_counts(counter, joint, rng):
    cells = list(joint); probs = np.array([joint[c] for c in cells]); probs = probs / probs.sum()
    cnt = np.array([counter.get(c, 0) for c in cells]); off = sum(v for k, v in counter.items() if k not in joint)
    n = cnt.sum() + off; exp = probs * n; big = exp >= 5
    c = np.append(cnt[big], cnt[~big].sum() + off); e = np.append(exp[big], exp[~big].sum())
    if e[-1] < 5:
        c, e = c[:-1], e[:-1]
    stat = float(((c - e) ** 2 / e).sum()); p = float(stats.chi2.sf(stat, len(c) - 1))
    tv = 0.5 * float(np.abs(cnt / n - probs).sum())
    null = [0.5 * np.abs(rng.multinomial(n, probs) / n - probs).sum() for _ in range(300)]
    return dict(chi2=stat, dof=len(c) - 1, p=p, tv=tv, tv_null_mean=float(np.mean(null)), tv_null_q975=float(np.quantile(null, 0.975)),
                off_support=int(off), cells=len(cells), cells_ge5=int(big.sum()))


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    tok = R.load_tok()
    prompts = R.load_prompts(tok, 100, 32, seed=0)
    draft = R.load_model("14m")
    rows = []
    for size, pi in CASES:
        target = R.load_model(size)
        prompt = prompts[pi : pi + 1]
        draft_fn, target_fn = R.make_fns(draft, target, 1.0, TOPK)
        joint = exact_joint(target, prompt)
        for method in ("speculative", "target_only"):
            gen = torch.Generator().manual_seed(77 + pi)
            counter = Counter(); alphas = []
            for _ in range(N // CH):
                pre = prompt.expand(CH, -1).contiguous()
                if method == "speculative":
                    out = M.sample_first_k(draft_fn, target_fn, pre, K, GAMMA, gen)
                else:
                    out = M.autoregressive_sample(target_fn, pre, K, gen)
                counter.update(map(tuple, out.tolist()))
            r = dict(target=size, prompt=pi, method=method, N=N, gamma=GAMMA if method == "speculative" else None,
                     **test_counts(counter, joint, rng))
            if method == "speculative":
                q1 = R.probs_last(draft, prompt, 1, 1.0, TOPK)[0, 0]; p1 = R.probs_last(target, prompt, 1, 1.0, TOPK)[0, 0]
                r["beta_first_position"] = float(M.accept_prob(p1, q1))
            rows.append(r)
            print(r["target"], pi, method, "p", round(r["p"], 3), "TV", round(r["tv"], 4), "null", round(r["tv_null_mean"], 4),
                  "t", round(time.time() - t0), flush=True)
        del target
    for r, a in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = a
    # T=0: greedy 경로와 정확히 같은가
    greedy = []
    target = R.load_model("70m")
    d0, t0f = R.make_fns(draft, target, 0)
    rand_draft = lambda ids: torch.nn.functional.one_hot(torch.randint(0, 50304, (ids.shape[0],)), 50304).float()
    for pi in range(5):
        prompt = prompts[pi : pi + 1]
        ref = prompt
        for _ in range(16):
            ref = torch.cat([ref, R.probs_last(target, ref, 1, 0)[:, 0].argmax(-1, keepdim=True)], 1)
        ref = ref[0, 32:].tolist()
        for name, dfn in (("pythia-14m", d0), ("random", rand_draft)):
            for gamma in (1, 4):
                seq, st = M.speculative_decode(dfn, t0f, prompt, 16, gamma, torch.Generator().manual_seed(pi))
                greedy.append(dict(prompt=pi, draft=name, gamma=gamma, equal=seq == ref, iterations=len(st)))
    res = dict(sample_level=rows, greedy=greedy, greedy_all_equal=all(g["equal"] for g in greedy), top_k=TOPK, seconds=round(time.time() - t0))
    OUT.write_text(json.dumps(res, indent=1))
    print("greedy all equal", res["greedy_all_equal"], len(greedy))


if __name__ == "__main__":
    main()
