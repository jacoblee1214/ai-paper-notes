"""run_experiments.py 결과(문서별 NLL)로 perplexity 신뢰구간과 짝지은 비교를 계산한다.

- 신뢰구간: 문서 단위 부트스트랩(B=5000, 시드 0). ppl = exp(sum NLL / sum 토큰).
- 비교: 같은 문서를 뽑아 log ppl 차이(방법 A - 방법 B)의 분포를 만든다. 양측 p = 2 * min(P(d<=0), P(d>=0)) (하한 1/(B+1)).
- 모델마다 미리 정한 비교 묶음 전체에 Holm 보정.
"""
import json
import sys

import numpy as np

B = 5000
COMPARISONS = [  # (A, B)  d = log ppl(A) - log ppl(B)
    ("awq_w4", "rtn_w4"), ("awq_clip_w4", "rtn_w4"), ("fp16_act_0.01_w4", "rtn_w4"),
    ("awq_w3", "rtn_w3"), ("awq_clip_w3", "rtn_w3"), ("awq_clip_w3", "awq_w3"),
    ("fp16_act_0.01_w3", "rtn_w3"), ("fp16_weight_0.01_w3", "rtn_w3"), ("fp16_random_0.01_w3", "rtn_w3"),
    ("fp16_act_0.01_w3", "fp16_weight_0.01_w3"), ("fp16_act_0.01_w3", "fp16_random_0.01_w3"),
    ("awq_w3", "fp16_act_0.01_w3"),
    ("s_sweep_1.25_w3", "s_sweep_1.0_w3"), ("s_sweep_1.5_w3", "s_sweep_1.0_w3"),
    ("s_sweep_2.0_w3", "s_sweep_1.0_w3"), ("s_sweep_4.0_w3", "s_sweep_1.0_w3"),
]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def analyze(path):
    r = json.load(open(path))
    meth = r["methods"]
    n = len(meth["fp32"]["nll"])
    rng = np.random.RandomState(0)
    idx = rng.randint(0, n, size=(B, n))
    nll = {k: np.array(v["nll"]) for k, v in meth.items()}
    cnt = {k: np.array(v["cnt"]) for k, v in meth.items()}
    logppl = {k: nll[k][idx].sum(1) / cnt[k][idx].sum(1) for k in meth}  # 평균 NLL = log ppl
    point = {k: float(np.exp(nll[k].sum() / cnt[k].sum())) for k in meth}
    ci = {k: [float(np.exp(np.percentile(logppl[k], 2.5))), float(np.exp(np.percentile(logppl[k], 97.5)))] for k in meth}
    comps, ps = [], []
    for a, b in COMPARISONS:
        d = logppl[a] - logppl[b]
        dp = np.log(point[a]) - np.log(point[b])
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (B + 1))
        comps.append(dict(a=a, b=b, ratio=float(np.exp(dp)), ratio_ci=[float(np.exp(np.percentile(d, 2.5))), float(np.exp(np.percentile(d, 97.5)))], p=float(p)))
        ps.append(p)
    for c, adj in zip(comps, holm(np.array(ps))):
        c["p_holm"] = float(adj)
    gap = {}
    for nb in (4, 3):
        base, fp = point[f"rtn_w{nb}"], point["fp32"]
        for k in point:
            if k.endswith(f"_w{nb}") and k != f"rtn_w{nb}" and not k.startswith("s_sweep"):
                gap[k] = float((base - point[k]) / (base - fp))
    return dict(model=r["model"], n_docs=n, n_tokens=int(cnt["fp32"].sum()), B=B, point=point, ci=ci,
                comparisons=comps, gap_closed=gap, table2=r["table2"], awq_info=r["awq_info"],
                n_params_quantized=r["n_params_quantized"])


def paired(nll_a, cnt_a, nll_b, cnt_b):
    """같은 문서 부트스트랩으로 ppl(A)/ppl(B)와 95% CI."""
    n = len(nll_a)
    idx = np.random.RandomState(0).randint(0, n, size=(B, n))
    na, ca, nb, cb = map(np.array, (nll_a, cnt_a, nll_b, cnt_b))
    d = na[idx].sum(1) / ca[idx].sum(1) - nb[idx].sum(1) / cb[idx].sum(1)
    pt = na.sum() / ca.sum() - nb.sum() / cb.sum()
    return dict(ratio=float(np.exp(pt)), ci=[float(np.exp(np.percentile(d, 2.5))), float(np.exp(np.percentile(d, 97.5)))])


def calib_extra(name, main):
    """보정 문장 수·위치 0 제외 진단의 짝지은 비교. 모두 한 번의 보정 추출(시드 0)이라 보정 추출의 분산은 포함하지 않는다."""
    cs = json.load(open(f"{name}_calib_size.json"))
    p0 = json.load(open(f"{name}_pos0.json"))
    out = {}
    for nb in (4, 3):
        rtn = main["methods"][f"rtn_w{nb}"]
        for n in (4, 8, 32):
            c = cs[f"n{n}_w{nb}"]
            out[f"calib{n}_vs_rtn_w{nb}"] = paired(c["nll"], c["cnt"], rtn["nll"], rtn["cnt"])
        a, b = cs[f"n4_w{nb}"], cs[f"n32_w{nb}"]
        out[f"calib4_vs_calib32_w{nb}"] = paired(a["nll"], a["cnt"], b["nll"], b["cnt"])
        a, b = p0[f"skip1_w{nb}"], p0[f"skip0_w{nb}"]
        out[f"skippos0_vs_all_w{nb}"] = paired(a["nll"], a["cnt"], b["nll"], b["cnt"])
    return out


if __name__ == "__main__":
    out = {}
    for name in ("gpt2", "pythia160m"):
        out[name] = analyze(f"{name}_results.json")
        out[name]["calib_extra"] = calib_extra(name, json.load(open(f"{name}_results.json")))
    json.dump(out, open("analysis_results.json", "w"), indent=1)
    for name, a in out.items():
        print(f"== {name}: n_docs={a['n_docs']} tokens={a['n_tokens']}")
        for k, v in a["point"].items():
            print(f"  {k:24s} {v:8.3f}  [{a['ci'][k][0]:.3f}, {a['ci'][k][1]:.3f}]")
        for c in a["comparisons"]:
            print(f"  {c['a']:22s} vs {c['b']:22s} ratio {c['ratio']:.3f} [{c['ratio_ci'][0]:.3f},{c['ratio_ci'][1]:.3f}] p={c['p']:.4f} holm={c['p_holm']:.4f}")
        print("  gap closed:", {k: round(v, 2) for k, v in a["gap_closed"].items()})
        print("  table2:", {k: (round(v["p_changed"], 3), round(v["mean_ratio"], 3), round(v["err_ratio"], 3)) for k, v in a["table2"].items()})
        print("  calib_extra:", {k: (round(v["ratio"], 3), [round(x, 3) for x in v["ci"]]) for k, v in a["calib_extra"].items()})
