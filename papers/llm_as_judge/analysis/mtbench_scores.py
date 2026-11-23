"""MT-Bench 80문항 점수(GPT-4 단일 채점)의 모델 간 차이가 문항 표본 오차를 넘는가.

입력: data/hf_space_mtbench/gpt-4_single.jsonl (HF space lmsys/mt-bench의 사전 생성 판정. 34개 모델 x 80문항 x 2턴)
  - 이 파일은 논문 표 8의 학습 모델이 아니라 이후 저장소가 공개한 판정이다(GPT-4 판정 시점도 논문과 다를 수 있다).
  - 라이선스 'other'(HF 카드에 명시 없음)라 사본은 data/에만 두고 게시본에는 계산 요약만 넣는다.
통계식은 ../eval_error_bars/analysis/minimal_impl.py(Miller 2024의 식 (1), (7))와 같은 것을 여기에 짠다.
  단위: 문항 평균 s_i = (1턴 점수 + 2턴 점수)/2, i = 1..80. 모델 평균 = s_i의 평균 = 논문 표 8 방식의 160개 평균.
  비교: 문항별 짝지은 차이 d_i = s_A,i - s_B,i, SE = sqrt(Var(d)/n) (식 (7)).
출력: analysis/mtbench_scores_results.json
"""
import json
import os
import itertools

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(0)


def se_clt(s):
    s = np.asarray(s, float)
    return float(np.sqrt(s.var(ddof=1) / len(s)))


def se_paired(a, b):
    return se_clt(np.asarray(a, float) - np.asarray(b, float))


def main():
    rows = [json.loads(l) for l in open(f"{ROOT}/data/hf_space_mtbench/gpt-4_single.jsonl")]
    qs = sorted({r["question_id"] for r in rows})
    assert len(qs) == 80
    models = sorted({r["model"] for r in rows})
    qi = {q: i for i, q in enumerate(qs)}
    S = {m: np.full((80, 2), np.nan) for m in models}
    for r in rows:
        S[r["model"]][qi[r["question_id"]], r["turn"] - 1] = r["score"]
    allscores = np.concatenate([S[m].ravel() for m in models])
    out = {"n_models": len(models), "score_values": sorted(set(float(x) for x in allscores if not np.isnan(x))),
           "n_missing": int(np.isnan(allscores).sum())}
    ok = [m for m in models if not np.isnan(S[m]).any()]
    out["models_with_complete_scores"] = len(ok)
    out["n_score_minus1"] = {m: int((S[m] == -1).sum()) for m in models if (S[m] == -1).any()}  # -1 = 판정 문장에서 점수 추출 실패, 평균에 그대로 들어감(표 8 재현 값과 일치)
    Q = {m: S[m].mean(1) for m in ok}  # 문항 평균 (n=80)
    mean = {m: float(Q[m].mean()) for m in ok}
    out["paper_table8_check"] = {m: {"ours_160_mean": round(mean[m], 3), "paper": p} for m, p in
                                 {"gpt-4": 8.99, "gpt-3.5-turbo": 7.94, "alpaca-13b": 4.53, "llama-13b": 2.61}.items() if m in mean}
    out["turn_means"] = {m: [round(float(S[m][:, 0].mean()), 3), round(float(S[m][:, 1].mean()), 3)] for m in ok}
    order = sorted(ok, key=lambda m: -mean[m])
    out["ranking"] = [{"model": m, "mean": round(mean[m], 3), "se_clt_80q": round(se_clt(Q[m]), 3),
                       "ci95": [round(mean[m] - 1.96 * se_clt(Q[m]), 3), round(mean[m] + 1.96 * se_clt(Q[m]), 3)]} for m in order]

    # 모든 쌍: 짝지은 SE와 비짝 SE
    M = len(order)
    n = 80
    tcrit = stats.t.ppf(0.975, n - 1)
    pairs = []
    for i, j in itertools.combinations(range(M), 2):
        a, b = Q[order[i]], Q[order[j]]
        d = a - b
        sep = se_paired(a, b)
        seu = float(np.hypot(se_clt(a), se_clt(b)))
        t = d.mean() / sep
        p = 2 * stats.t.sf(abs(t), n - 1)
        pairs.append((i, j, float(d.mean()), sep, seu, float(np.corrcoef(a, b)[0, 1]), float(p)))
    P = np.array(pairs)
    sig_paired = (P[:, 6] < 0.05)
    # 비짝 z (정규 근사)
    zu = P[:, 2] / P[:, 4]
    sig_unpaired = np.abs(zu) > 1.96
    # Holm 보정(쌍별 p, 561개)
    pv = P[:, 6]
    idx = np.argsort(pv)
    holm = np.zeros(len(pv), bool)
    for rank, k in enumerate(idx):
        if pv[k] < 0.05 / (len(pv) - rank):
            holm[k] = True
        else:
            break
    out["pairs"] = {"n_pairs": len(pairs), "sig_paired_p05": int(sig_paired.sum()), "sig_unpaired_z196": int(sig_unpaired.sum()),
                    "sig_paired_holm05": int(holm.sum()),
                    "median_se_paired": round(float(np.median(P[:, 3])), 3), "median_se_unpaired": round(float(np.median(P[:, 4])), 3),
                    "median_score_corr": round(float(np.median(P[:, 5])), 3)}
    adj = []
    for k, (i, j, d, sep, seu, r, p) in enumerate(pairs):
        if j == i + 1:
            adj.append({"A": order[i], "B": order[j], "diff": round(d, 3), "se_paired": round(sep, 3), "se_unpaired": round(seu, 3),
                        "corr": round(r, 3), "p_paired": round(p, 4), "sig_p05": bool(p < 0.05)})
    out["adjacent_ranks"] = adj
    out["adjacent_summary"] = {"n": len(adj), "n_sig_p05": sum(a["sig_p05"] for a in adj),
                               "n_not_sig": sum(not a["sig_p05"] for a in adj)}
    # 관련 있는 저장소 규모: 점수 차이가 얼마 이상이면 구분되나
    out["diff_needed_for_p05_paired_median"] = round(float(np.median(P[:, 3]) * tcrit), 3)

    # 순위 부트스트랩: 문항 재표집
    B = 4000
    ranks = np.zeros((B, M), int)
    allQ = np.stack([Q[m] for m in order], 1)  # (80, M)
    for b in range(B):
        ix = RNG.integers(0, 80, 80)
        mu = allQ[ix].mean(0)
        ranks[b] = (-mu).argsort().argsort() + 1
    out["rank_ci"] = [{"model": order[i], "point_rank": i + 1,
                       "rank95": [int(np.percentile(ranks[:, i], 2.5)), int(np.percentile(ranks[:, i], 97.5))]} for i in range(M)]
    widths = [r["rank95"][1] - r["rank95"][0] for r in out["rank_ci"]]
    out["rank_ci_width_median"] = float(np.median(widths))

    # 논문이 다룬 6개 모델(GPT-4, GPT-3.5, Claude-v1, Vicuna-13B(v1.3으로 대체), Alpaca-13B, LLaMA-13B) 간 쌍 비교
    six = ["gpt-4", "gpt-3.5-turbo", "claude-v1", "vicuna-13b-v1.3", "alpaca-13b", "llama-13b"]
    six = [m for m in six if m in Q]
    sx = []
    for a, b in itertools.combinations(six, 2):
        d = Q[a] - Q[b]
        t = d.mean() / se_paired(Q[a], Q[b])
        sx.append({"A": a, "B": b, "diff": round(float(d.mean()), 3), "se_paired": round(se_paired(Q[a], Q[b]), 3),
                   "se_unpaired": round(float(np.hypot(se_clt(Q[a]), se_clt(Q[b]))), 3), "p": float(f"{2 * stats.t.sf(abs(t), 79):.3g}")})
    out["six_models_pairs"] = sx

    # 문항 군집 안의 1턴-2턴 상관 (문항 단위 군집을 쓰는 이유)
    out["within_question_turn_corr_median"] = round(float(np.median([np.corrcoef(S[m][:, 0], S[m][:, 1])[0, 1] for m in ok])), 3)
    # 턴을 독립 160개로 쓴 SE(과소)와 문항 군집 SE(80문항 평균) 비교
    r = []
    for m in ok:
        naive = float(np.sqrt(S[m].ravel().var(ddof=1) / 160))
        r.append(se_clt(Q[m]) / naive)
    out["se_question_over_naive160_median"] = round(float(np.median(r)), 3)

    with open(f"{ROOT}/analysis/mtbench_scores_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print({k: out[k] for k in ["n_models", "score_values", "n_missing", "models_with_complete_scores", "paper_table8_check", "pairs", "adjacent_summary",
                               "diff_needed_for_p05_paired_median", "rank_ci_width_median", "within_question_turn_corr_median", "se_question_over_naive160_median"]})
    for r in out["ranking"][:6] + out["ranking"][-3:]:
        print(r)
    for a in out["adjacent_ranks"]:
        print(a["A"], a["B"], a["diff"], a["se_paired"], a["p_paired"])
    print(out["six_models_pairs"])


if __name__ == "__main__":
    main()
