"""repo/inter_annotator_agreement_annotations/*.pdf (pdftotext -layout 결과)의 TRUE/FALSE 표를
HF 공개 JSON(MAD_human_labelled_dataset.json)과 셀 단위로 대조하고, PDF 원표로 κ를 다시 계산한다.
전제: .scratch/ 에 pdftotext -layout 결과 txt가 있다(README 재실행 절 참고).
실행: python3 -I analysis/iaa_pdf_vs_hf.py
"""
import json, os, re, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SC = os.path.join(ROOT, ".scratch")
H = json.load(open(os.path.join(ROOT, "data/hf/MAD_human_labelled_dataset.json")))
ANN = ["annotator_1", "annotator_2", "annotator_3"]


def parse(fname, ntr):
    t = open(os.path.join(SC, fname)).read()
    rows = []
    for l in t.split("\n"):
        tk = re.findall(r"\b(TRUE|FALSE)\b", l)
        if tk:
            v = [x == "TRUE" for x in tk]
            if len(v) < 3 * ntr:  # 마지막 주석자 칸이 빈 행: 끝에 NaN을 채운다(HF도 같은 위치가 NaN)
                v = v + [np.nan] * (3 * ntr - len(v))
            rows.append(v)
    return np.array(rows, dtype=float).reshape(len(rows), ntr, 3)  # modes x traces x 3


def hf_block(rd):
    recs = [r for r in H if r["round"] == rd]
    out = np.full((len(recs[0]["annotations"]), len(recs), 3), np.nan)
    for t, r in enumerate(recs):
        for m, a in enumerate(r["annotations"]):
            for j, k in enumerate(ANN):
                v = a.get(k)
                if v is True or v is False:
                    out[m, t, j] = float(v)
    return out, [r["mas_name"] for r in recs]


def cohen(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    po = np.mean(x == y)
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return (po - pe) / (1 - pe)


def pair_mean(A):
    M = A.reshape(-1, 3)
    return float(np.mean([cohen(M[:, i], M[:, j]) for i, j in itertools.combinations(range(3), 2)]))


def fleiss(A):
    M = A.reshape(-1, 3)
    M = M[~np.isnan(M).any(axis=1)]
    k1 = M.sum(1)
    pi = (k1 * (k1 - 1) + (3 - k1) * (3 - k1 - 1)) / 6
    p1 = M.mean()
    pe = p1**2 + (1 - p1) ** 2
    return float((pi.mean() - pe) / (1 - pe))


res = {}
for name, f, rd, ntr in [("Round 2", "interannotator_annotations_round_2_trim.pdf.txt", "Round 2", 5),
                         ("Round 3", "interannotator_annotations_round3_trim.pdf.txt", "Round 3", 5),
                         ("Generalizability", "interannotator_annotations_new_benchmarks_trim.pdf.txt", "Generlazability", 4)]:
    P = parse(f, ntr)
    Hh, mas = hf_block(rd)
    # HF의 trace 순서가 PDF와 같다고 가정하고 셀을 비교한다. 순서가 다른 경우를 위해 trace 순열 중 일치가 최대인 것도 본다.
    same = (np.nan_to_num(P, nan=-1) == np.nan_to_num(Hh, nan=-1))
    best = None
    for perm in itertools.permutations(range(ntr)):
        s = (np.nan_to_num(P[:, list(perm), :], nan=-1) == np.nan_to_num(Hh, nan=-1)).mean()
        if best is None or s > best[0]:
            best = (s, perm)
    d = {
        "pdf_shape": list(P.shape), "hf_shape": list(Hh.shape),
        "cell_match_same_order": float(same.mean()),
        "best_perm_match": float(best[0]), "best_perm": list(best[1]),
        "hf_mas_order": mas,
        "pdf_cohen_pairwise_mean": pair_mean(P), "pdf_fleiss": fleiss(P),
        "hf_cohen_pairwise_mean": pair_mean(Hh), "hf_fleiss": fleiss(Hh),
        "pdf_positive_rate": float(np.nanmean(P)),
        "pdf_per_trace_kappa": [pair_mean(P[:, t:t+1, :]) if np.nanstd(P[:, t, :]) > 0 else None for t in range(ntr)],
    }
    res[name] = d
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})

# Round 2+3 합산(PDF 원표)
A = np.concatenate([parse("interannotator_annotations_round_2_trim.pdf.txt", 5), parse("interannotator_annotations_round3_trim.pdf.txt", 5)], axis=1)
res["PDF Round 2+3 pooled"] = {"cohen_pairwise_mean": pair_mean(A), "fleiss": fleiss(A)}
print("PDF R2+R3 pooled", res["PDF Round 2+3 pooled"])
res["PDF mean of round κ (R2,R3)"] = float(np.mean([res["Round 2"]["pdf_cohen_pairwise_mean"], res["Round 3"]["pdf_cohen_pairwise_mean"]]))
print(res["PDF mean of round κ (R2,R3)"])
json.dump(res, open(os.path.join(HERE, "iaa_pdf_vs_hf_results.json"), "w"), ensure_ascii=False, indent=1)

# ---- 추가: PDF 원표 기준 변형별 κ와 구간 ----
rng = np.random.default_rng(0)
G = parse("interannotator_annotations_new_benchmarks_trim.pdf.txt", 4)  # 모드 x trace(ChatDev,MetaGPT,Magentic,OpenManus) x 3
R2 = parse("interannotator_annotations_round_2_trim.pdf.txt", 5)
R3 = parse("interannotator_annotations_round3_trim.pdf.txt", 5)


def cell_boot(A, B=4000, metric=pair_mean):
    """셀(모드x trace)을 독립으로 보고 복원추출. 한 trace의 모드들이 독립이 아니라서 구간이 낙관적이다."""
    M = A.reshape(-1, 3)
    n = len(M)
    out = []
    for _ in range(B):
        S = M[rng.integers(0, n, n)].reshape(n, 1, 3)
        try:
            v = metric(S)
        except Exception:
            continue
        if np.isfinite(v):
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def trace_boot(A, B=4000, metric=pair_mean):
    nt = A.shape[1]
    out = []
    for _ in range(B):
        S = A[:, rng.integers(0, nt, nt), :]
        v = metric(S)
        if np.isfinite(v):
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], len(out)


def trace_means(A):
    return [pair_mean(A[:, t:t+1, :]) if np.nanstd(A[:, t, :]) > 0 else None for t in range(A.shape[1])]


variants = {
    "PDF Gen 4 traces pooled": G,
    "PDF Gen 새 MAS 2개(Magentic, OpenManus) pooled": G[:, 2:4, :],
    "PDF Round 2 pooled": R2,
    "PDF Round 3 pooled": R3,
    "PDF Round 2+3 pooled": np.concatenate([R2, R3], axis=1),
}
res["variants"] = {}
for k, A in variants.items():
    M = A.reshape(-1, 3)
    ok = ~np.isnan(M).any(1)
    tb, ntb = trace_boot(A)
    d = {"traces": A.shape[1], "cells": int(ok.sum()), "cohen": pair_mean(A), "fleiss": fleiss(A),
         "raw_agree": float(np.mean([np.mean(M[ok][:, i] == M[ok][:, j]) for i, j in itertools.combinations(range(3), 2)])),
         "pos_rate": float(M[ok].mean()), "cell_boot95": cell_boot(A), "trace_boot95": tb, "trace_boot_valid": ntb}
    res["variants"][k] = d
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in d.items()})
# trace별 κ 평균 방식(κ가 정의되지 않는 trace는 제외)
for k, A in variants.items():
    tm = [x for x in trace_means(A) if x is not None and np.isfinite(x)]
    res["variants"][k]["mean_of_trace_kappas"] = float(np.mean(tm)) if tm else None
    print(k, "mean of per-trace κ:", res["variants"][k]["mean_of_trace_kappas"], "n=", len(tm))
# 불일치 셀 목록(PDF 기준, Gen)
names = [r["failure mode"].split("\n")[0] for r in [x for x in H if x["round"] == "Generlazability"][0]["annotations"]]
dis = []
for m in range(G.shape[0]):
    for t in range(G.shape[1]):
        v = G[m, t, :]
        if len(set(v[~np.isnan(v)])) > 1:
            dis.append((names[m], ["ChatDev-MMLU", "MetaGPT-MMLU", "GAIA-Magentic", "GAIA-OpenManus"][t], v.tolist()))
res["gen_disagreements_pdf"] = dis
print("Gen 불일치", len(dis))
for x in dis: print("  ", x)
json.dump(res, open(os.path.join(HERE, "iaa_pdf_vs_hf_results.json"), "w"), ensure_ascii=False, indent=1)
