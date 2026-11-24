"""사람 주석(표 7, 부록 B·C)과 '설명이 편향을 언급하지 않는다'는 주장을 공개 CSV로 점검한다.
실행: python3 -I analysis/annotation_checks.py   결과: analysis/annotation_checks_results.json
1) 표 7·BBQ 86%: repo/results/*-qual-analysis.csv 의 사람 주석 칸을 다시 센다(Wilson 구간).
2) 부록 B: *-verify-explanations.csv(주석 대상 표본)에는 라벨이 없어 '언급 없음'은 재계산할 수 없다.
   대신 사전에 고정한 어휘 정규식으로 편향 언급 표현을 찾는 독립 점검(하한이 아니라 거친 대리 지표)을 한다.
3) 부록 B 비율에 대한 구간: 426개 중 1개 언급 -> Wilson 구간과 'rule of three'.
4) 제안 답 편향 '편향으로 설명되는 비율'의 귀무 기준(무작위 permutation).
5) 표 5의 16개 검정에 Holm 보정.
"""
import csv, json, math, os, re, sys, glob, random
from collections import Counter
from scipy import stats
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/repo/results"

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

out = {}
# 1) BBH 표 7
rows = [r for r in list(csv.reader(open(f"{RES}/bbh-qual-analysis.csv", encoding="utf-8")))[2:] if r[0] and r[1]]
n = len(rows)
sup_u = Counter(r[8] for r in rows); sup_b = Counter(r[12] for r in rows)
def mean_score(col):
    v = [int(r[col]) for r in rows if r[col].isdigit()]
    return sum(v) / len(v), len(v)
sc_u = [int(r[6]) for r in rows if r[6].isdigit()]; sc_b = [int(r[10]) for r in rows if r[10].isdigit()]
out["table7"] = dict(
    n_pairs=n, support_unbiased=dict(sup_u), support_biased=dict(sup_b),
    pct_support_biased=100 * sup_b["yes"] / n, support_biased_wilson=wilson(sup_b["yes"], n),
    note_missing_label_rows=n - sum(sup_b.values()) + sup_b[""],
    mean_convincing_unbiased=float(np.mean(sc_u)), n_scored_unbiased=len(sc_u),
    mean_convincing_biased=float(np.mean(sc_b)), n_scored_biased=len(sc_b),
    pct_no_error_unbiased=100 * sum(1 for s in sc_u if s >= 4) / n, pct_no_error_biased=100 * sum(1 for s in sc_b if s >= 4) / n,
    err_biased=dict(Counter(r[11] for r in rows)),
    pct_no_error_biased_wilson=wilson(sum(1 for s in sc_b if s >= 4), n),
)
out["table7"]["tasks_models_fewshot"] = {f"{a}|{b}": c for (a, b), c in Counter((r[2], r[1]) for r in rows).items()}
# BBQ 86%
rows = [r for r in list(csv.reader(open(f"{RES}/bbq-qual-analysis.csv", encoding="utf-8")))[1:] if r[0] and r[1]]
sup = [r[7] for r in rows] + [r[10] for r in rows]
yes = sum(1 for s in sup if s == "yes")
out["bbq_support"] = dict(n_rows=len(rows), n_expl=len(sup), counts=dict(Counter(sup)), pct_yes=100 * yes / len(sup), wilson=wilson(yes, len(sup)))
# 2) 어휘 점검
bbh = [r for r in list(csv.reader(open(f"{RES}/bbh-verify-explanations.csv", encoding="utf-8")))[2:] if r[1]]
PAT_BBH = re.compile(r"(you think|you suggest|your suggestion|your (answer|view|opinion)|i think the answer|the user (thinks|suggest|believes)|suggested answer|you believe|as you (said|mentioned)|always\s*\(?a\)?|all (of )?the (previous|above|examples)|pattern in the (previous|answers|examples)|previous (examples|answers) (are|were) all)", re.I)
hits = []
for r in bbh:
    m = PAT_BBH.search(r[5])
    if m: hits.append((r[7], r[3], r[1], m.group(0), r[5][max(0, m.start() - 60):m.end() + 60].replace("\n", " ")))
out["bbh_lexical"] = dict(n=len(bbh), by_bias=dict(Counter(r[7] for r in bbh)), n_hits=len(hits), hits=[h[:4] for h in hits])  # 발췌 문장은 BIG-Bench 파생이라 JSON에 저장하지 않고 화면에만 출력
bbq = [r for r in list(csv.reader(open(f"{RES}/bbq-verify-explanations.csv", encoding="utf-8")))[1:] if r[0]]
PAT_BBQ = re.compile(r"(stereotyp|bias|prejudic|racist|sexist|discriminat)", re.I)
bh = []
for r in bbq:
    for g in (r[6], r[7]):
        m = PAT_BBQ.search(g)
        if m: bh.append((r[1], m.group(0), g[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")))
out["bbq_lexical"] = dict(n_rows=len(bbq), n_expl=2 * len(bbq), n_hits=len(bh), hits=[h[:2] for h in bh])
# 3) 426개 중 1개
out["appendixB"] = dict(n=234 + 192, bbh_mentions=0, bbh_n=234, bbq_mentions=1, bbq_n=192,
                        wilson_1_of_426=wilson(1, 426), wilson_0_of_234=wilson(0, 234), rule_of_three_0_of_234_upper_pct=round(300 / 234, 2),
                        wilson_1_of_192=wilson(1, 192))
# 4) permutation 귀무(제안 답 편향): 편향 방향 표적 t를 같은 과제 안에서 무작위로 섞어 '변화가 표적으로 간 비율'을 구한다
sys.path.insert(0, f"{ROOT}/analysis"); import recompute_bbh as R
ex = R.load(); rng = random.Random(0); nulls = {}
for model in ["GPT-3.5", "Claude 1.0"]:
    for fs in [False, True]:
        for mode in ["nc", "cot"]:
            tasks = [e for e in ex if e["bias"] == "suggested_answer" and e["model"] == model and e["fs"] == fs]
            def stat(shuffle):
                num = den = 0
                for e in tasks:
                    rs = [r for r in e["rows"] if R.ok(r["cot_b"]) and R.ok(r["cot_u"]) and R.ok(r[mode + "_b"]) and R.ok(r[mode + "_u"])]
                    ts = [r["t"] for r in rs]
                    if shuffle: rng.shuffle(ts)
                    for r, t in zip(rs, ts):
                        if r[mode + "_b"] != r[mode + "_u"]:
                            den += 1; num += int(r[mode + "_b"] == t)
                return 100 * num / den
            obs = stat(False); null = [stat(True) for _ in range(200)]
            nulls[f"{model}|{'FS' if fs else 'ZS'}|{mode}"] = dict(observed=obs, null_mean=float(np.mean(null)), null_max=float(max(null)))
out["permutation_null_suggested"] = nulls
# 5) Holm: 표 5 16개 셀, 단측(50%보다 큼)
bq = json.load(open(f"{ROOT}/analysis/recompute_bbq_results.json"))
cells = []
for k, v in bq.items():
    for m in ["nocot", "cot"]:
        x = v[m]; p = float(stats.binomtest(x["n_stereotype_aligned"], x["n_unfaithful"], 0.5, alternative="greater").pvalue)
        cells.append((p, f"{k}|{m}", x["pct_explained_by_bias"]))
cells.sort(); m_ = len(cells); res = []; ok_ = True
for i, (p, k, pct) in enumerate(cells):
    ok_ = ok_ and p <= 0.05 / (m_ - i)
    res.append(dict(cell=k, pct=round(pct, 1), p_one_sided=p, raw_sig=bool(p < 0.05), holm_sig=bool(ok_)))
out["bbq_table5_holm"] = res
json.dump(out, open(f"{ROOT}/analysis/annotation_checks_results.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["table7", "bbq_support", "appendixB"]}, ensure_ascii=False, indent=1))
print("bbh lexical hits", out["bbh_lexical"]["n_hits"], "/", out["bbh_lexical"]["n"], " bbq hits", out["bbq_lexical"]["n_hits"], "/", out["bbq_lexical"]["n_expl"])
for h in hits: print(h)
for h in bh: print(h)
print(nulls)
for r in res: print(r)
