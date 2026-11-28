"""lm_quant_nll.json(문서별 NLL) -> perplexity, 문서 단위 부트스트랩 신뢰구간, 짝지은 비교와 Holm 보정, 그림.
perplexity = exp(문서별 평균 NLL의 평균)(문서마다 같은 길이 512토큰이라 토큰 평균과 같다).
신뢰구간: 문서 재표집 2000회(시드 0). 비교: 같은 문서의 NLL 차이에 대한 부호 뒤집기 검정 20000회(시드 0) + 부트스트랩 CI, 가족별 Holm.
실행: python3 analyze_lm.py   (결과 lm_quant_results.json, ../figures/own_lm_ppl.png)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "lm_quant_nll.json").read_text())
rng = np.random.RandomState(0)
NB, NP = 2000, 20000


def nll(k):
    return np.array(R[k]["nll"])


def ci_ppl(v):
    idx = rng.randint(0, len(v), (NB, len(v)))
    b = np.exp(v[idx].mean(1))
    return float(np.exp(v.mean())), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def paired(a, b):
    """Δ = mean(a - b) nat, 부트스트랩 CI, 부호 뒤집기 p."""
    d = a - b
    idx = rng.randint(0, len(d), (NB, len(d)))
    bs = d[idx].mean(1)
    signs = rng.choice([-1.0, 1.0], (NP, len(d)))
    perm = np.abs((signs * d).mean(1))
    p = (1 + (perm >= abs(d.mean())).sum()) / (NP + 1)
    return {"delta_nat": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p": float(p), "ppl_ratio": float(np.exp(d.mean()))}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


out = {"n_docs": len(nll("gpt2/fp")), "ppl": {}, "gptq_vs_rtn": {}, "gptq_vs_fp": {}, "variants": {}, "calib_seed": {}}
for k in R:
    out["ppl"][k] = dict(zip(("ppl", "ci95"), ci_ppl(nll(k))))
for key in ("gpt2", "p160"):
    fam, names = [], []
    for b in (4, 3):
        for g in (0, 128):
            c = paired(nll(f"{key}/gptq_b{b}_g{g}"), nll(f"{key}/rtn_b{b}_g{g}"))
            fam.append(c["p"])
            names.append(f"{key} {b}bit g{g or '없음'}")
            out["gptq_vs_rtn"][names[-1]] = c
    for n, a in zip(names, holm(fam)):
        out["gptq_vs_rtn"][n]["p_holm"] = a
    fam, names = [], []
    for b in (4, 3):
        for g in (0, 128):
            c = paired(nll(f"{key}/gptq_b{b}_g{g}"), nll(f"{key}/fp"))
            fam.append(c["p"])
            names.append(f"{key} {b}bit g{g or '없음'}")
            out["gptq_vs_fp"][names[-1]] = c
    for n, a in zip(names, holm(fam)):
        out["gptq_vs_fp"][n]["p_holm"] = a
# act-order 등 변형(대 같은 설정의 기본 GPTQ). 가족: 변형 전체 한 묶음
fam, names = [], []
for k in R:
    if k.endswith("_act") or "_d0.001" in k:
        base = k.replace("_act", "").replace("_d0.001", "")
        c = paired(nll(k), nll(base))
        c["base"] = base
        out["variants"][k] = c
        fam.append(c["p"])
        names.append(k)
for n, a in zip(names, holm(fam)):
    out["variants"][n]["p_holm"] = a
# 보정 시드(문서 집합이 시드마다 다름) 분산
for b in (3, 4):
    v = [out["ppl"][f"gpt2/gptq_b{b}_g0{s}"]["ppl"] for s in ("", "_s1", "_s2")]
    out["calib_seed"][f"gpt2 {b}bit g없음"] = {"ppl_by_seed(0,1,2)": v, "range": [min(v), max(v)], "sd": float(np.std(v, ddof=1))}
(HERE / "lm_quant_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

# 그림
plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
cols = {"rtn": "#9467bd", "gptq": "#d62728"}
for ax, key, title in zip(axes, ("gpt2", "p160"), ("GPT-2 small (124M)", "Pythia-160M")):
    fp = out["ppl"][f"{key}/fp"]["ppl"]
    labels = ["4비트\n행별", "4비트\ng128", "3비트\n행별", "3비트\ng128"]
    x = np.arange(4)
    for j, m in enumerate(("rtn", "gptq")):
        ks = [f"{key}/{m}_b{b}_g{g}" for b in (4, 3) for g in (0, 128)]
        v = np.array([out["ppl"][k]["ppl"] for k in ks])
        lo = np.array([out["ppl"][k]["ci95"][0] for k in ks])
        hi = np.array([out["ppl"][k]["ci95"][1] for k in ks])
        ax.bar(x + (j - 0.5) * 0.38, v, 0.36, yerr=[v - lo, hi - v], color=cols[m], label=m.upper(), capsize=2)
    ax.axhline(fp, color="k", ls="--", lw=1, label=f"FP32 {fp:.1f}")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("held-out pile-10k perplexity (로그 축)")
    ax.legend()
fig.tight_layout()
fig.savefig(HERE.parent / "figures" / "own_lm_ppl.png", dpi=150)
print(json.dumps({k: v for k, v in out.items() if k != "ppl"}, indent=1, ensure_ascii=False))
for k, v in out["ppl"].items():
    print(k, round(v["ppl"], 2), [round(c, 2) for c in v["ci95"]])
