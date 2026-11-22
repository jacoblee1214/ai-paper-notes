"""MAST-Data-human(HF mcemri/MAD, MAD_human_labelled_dataset.json)의 3인 주석에서
주석자 간 일치도(Cohen κ, Fleiss κ)를 직접 계산한다. numpy만 사용.
실행: python3 -I analysis/iaa_recompute.py  (data/hf/ 에 JSON이 있어야 함)
"""
import json, itertools, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
H = json.load(open(os.path.join(ROOT, "data/hf/MAD_human_labelled_dataset.json")))
ANN = ["annotator_1", "annotator_2", "annotator_3"]
rng = np.random.default_rng(0)


def cells(rec):
    """trace 하나의 (모드 수 x 3) 행렬. NaN은 np.nan."""
    m = np.full((len(rec["annotations"]), 3), np.nan)
    for i, a in enumerate(rec["annotations"]):
        for j, k in enumerate(ANN):
            v = a.get(k)
            if v is True or v is False:
                m[i, j] = float(v)
    return m


def cohen(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    po = np.mean(x == y)
    p1x, p1y = x.mean(), y.mean()
    pe = p1x * p1y + (1 - p1x) * (1 - p1y)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def mean_pair_kappa(M):
    return float(np.nanmean([cohen(M[:, i], M[:, j]) for i, j in itertools.combinations(range(3), 2)]))


def fleiss(M):
    ok = ~np.isnan(M).any(axis=1)
    M = M[ok]
    n = 3
    k1 = M.sum(axis=1)
    pi = (k1 * (k1 - 1) + (n - k1) * (n - k1 - 1)) / (n * (n - 1))
    Pbar = pi.mean()
    p1 = M.mean()
    pe = p1**2 + (1 - p1) ** 2
    return float((Pbar - pe) / (1 - pe))


def raw_agree(M):
    ok = ~np.isnan(M).any(axis=1)
    M = M[ok]
    return float(np.mean([np.mean(M[:, i] == M[:, j]) for i, j in itertools.combinations(range(3), 2)]))


def pos_stats(M):
    ok = ~np.isnan(M).any(axis=1)
    M = M[ok]
    s = M.sum(axis=1)
    any1 = int((s >= 1).sum())
    all3 = int((s == 3).sum())
    return {"cells": int(len(M)), "pos_rate": float(M.mean()), "cells_any_flag": any1,
            "cells_all3_flag": all3, "all3_over_any": all3 / any1 if any1 else None}


def pabak(M):
    return 2 * raw_agree(M) - 1


def boot(trace_mats, B=4000, metric=mean_pair_kappa):
    """trace를 복원추출해 κ 구간(백분위 95%). 한 trace 안 셀은 묶어서 뽑는다."""
    n = len(trace_mats)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        M = np.vstack([trace_mats[i] for i in idx])
        v = metric(M)
        if not np.isnan(v):
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def block(name, recs):
    mats = [cells(r) for r in recs]
    M = np.vstack(mats)
    d = {
        "traces": len(recs),
        "modes_per_trace": sorted(set(len(m) for m in mats)),
        "cohen_pairwise_mean": mean_pair_kappa(M),
        "cohen_pairs": {f"{i+1}-{j+1}": cohen(M[:, i], M[:, j]) for i, j in itertools.combinations(range(3), 2)},
        "fleiss": fleiss(M),
        "raw_agreement": raw_agree(M),
        "pabak": pabak(M),
        "boot95_trace_resample": boot(mats),
        "per_trace_cohen_mean": [mean_pair_kappa(m) for m in mats],
        **pos_stats(M),
    }
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k not in ("cohen_pairs",)})
    return d


res = {}
rounds = {}
for r in H:
    rounds.setdefault(r["round"], []).append(r)
for rd, recs in rounds.items():
    res[rd] = block(rd, recs)

res["Rounds 2+3 (17-mode)"] = block("R2+R3", rounds["Round 2"] + rounds["Round 3"])
res["Rounds 1+2+3"] = block("R1+R2+R3", rounds["Round 1"] + rounds["Round 2"] + rounds["Round 3"])
gen = rounds["Generlazability"]
res["Generalizability, 새 MAS 2개만(Magentic, OpenManus)"] = block("Gen 2 new MAS", [r for r in gen if r["mas_name"].startswith("GAIA")])
res["전체 19 trace"] = block("ALL19", H)
# 라운드 단위 κ를 평균하는 방식
res["mean_of_round_kappas_R2_R3"] = float(np.mean([res["Round 2"]["cohen_pairwise_mean"], res["Round 3"]["cohen_pairwise_mean"]]))
res["mean_of_round_kappas_R1_R2_R3"] = float(np.mean([res[r]["cohen_pairwise_mean"] for r in ("Round 1", "Round 2", "Round 3")]))
res["mean_of_round_kappas_R2_R3_Gen"] = float(np.mean([res[r]["cohen_pairwise_mean"] for r in ("Round 2", "Round 3", "Generlazability")]))
print("mean κ R2,R3:", res["mean_of_round_kappas_R2_R3"], " R1-3:", res["mean_of_round_kappas_R1_R2_R3"])

# 중복 trace(Round 1과 Round 2의 ChatDev)
def key(r): return " ".join(r["trace"].split())[:3000]
dups = {}
for i, r in enumerate(H):
    dups.setdefault(key(r), []).append((i, r["round"], r["mas_name"]))
res["duplicate_traces"] = [v for v in dups.values() if len(v) > 1]
res["unique_traces"] = len(dups)
print("duplicates:", res["duplicate_traces"], "unique:", len(dups))

# 결측 셀
nan_cells = []
for r in H:
    for a in r["annotations"]:
        for k in ANN:
            if a.get(k) is None or (isinstance(a.get(k), float)):
                nan_cells.append((r["round"], r["mas_name"], a["failure mode"].split("\n")[0], k))
res["nan_cells"] = nan_cells
print("nan cells:", nan_cells)

json.dump(res, open(os.path.join(HERE, "iaa_results.json"), "w"), ensure_ascii=False, indent=1)
