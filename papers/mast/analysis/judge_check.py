"""LLM 판정자(표 2) 검증.
(1) 표 2 수치의 내부 정합성: 정확도·재현율·정밀도에서 유병률과 κ를 역산해 보고된 F1·κ와 맞는지 본다.
(2) 표본 수 가정별 κ 구간(모수적 부트스트랩). 홀드아웃 크기는 논문에 없다.
(3) 공개 LLM 라벨(MAST-Data)과 사람 PDF 라벨을 겹치는 trace에서 직접 대조. 겹치는 trace는 2개(14개 모드 기준)뿐이다.
실행: python3 -I analysis/judge_check.py   (.scratch/ 에 pdftotext -layout 결과와 data/hf/ JSON 필요)
"""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
rng = np.random.default_rng(0)
res = {}

# (1) 표 2 역산
def implied(acc, rec, prec):
    # P = 양성 비율. 오류율 = P*[(1-rec) + rec*(1/prec - 1)]
    P = (1 - acc) / ((1 - rec) + rec * (1 / prec - 1))
    pred = rec * P / prec
    pe = P * pred + (1 - P) * (1 - pred)
    kappa = (acc - pe) / (1 - pe)
    f1 = 2 * prec * rec / (prec + rec)
    return P, pred, kappa, f1

T2 = {"o1": (0.89, 0.62, 0.68, 0.64, 0.58), "o1 few-shot": (0.94, 0.77, 0.833, 0.80, 0.77)}
res["table2_implied"] = {}
for k, (acc, rec, prec, f1p, kp) in T2.items():
    P, pred, kappa, f1 = implied(acc, rec, prec)
    res["table2_implied"][k] = {"reported": {"acc": acc, "recall": rec, "precision": prec, "f1": f1p, "kappa": kp},
                                "implied_positive_rate": P, "implied_pred_positive_rate": pred, "implied_kappa": kappa, "implied_f1": f1}
    print(k, "역산 양성비율 %.3f 예측양성비율 %.3f κ %.3f(보고 %.2f) F1 %.3f(보고 %.2f)" % (P, pred, kappa, kp, f1, f1p))

# (2) 모수적 부트스트랩: 표 2의 혼동행렬 비율을 N개 셀로 뽑았을 때 κ 구간
def conf(acc, rec, prec, N):
    P, pred, _, _ = implied(acc, rec, prec)
    tp = rec * P
    fp = tp / prec - tp
    fn = P - tp
    tn = 1 - tp - fp - fn
    return np.array([tp, fp, fn, tn])


def kappa_from(c):
    tp, fp, fn, tn = c
    n = c.sum()
    po = (tp + tn) / n
    p1 = (tp + fn) / n
    q1 = (tp + fp) / n
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return (po - pe) / (1 - pe)


res["kappa_ci_by_N"] = {}
for k, (acc, rec, prec, _, kp) in T2.items():
    pr = conf(acc, rec, prec, 1)
    res["kappa_ci_by_N"][k] = {}
    for N in (70, 140, 210, 420):  # 5/10/15/30 trace x 14 모드
        sims = rng.multinomial(N, pr, size=20000)
        ks = np.array([kappa_from(s) for s in sims if s.sum() and (s[0] + s[2]) > 0 and (s[0] + s[1]) > 0])
        res["kappa_ci_by_N"][k][N] = [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5)), float(ks.mean()), int(np.round(pr[0] * N + pr[2] * N))]
        print(k, "N=%d 셀 → κ 95%% 구간 [%.2f, %.2f], 실제 양성 셀 약 %d" % (N, *res["kappa_ci_by_N"][k][N][:2], res["kappa_ci_by_N"][k][N][3]))

# o1 대 o1 few-shot κ 차이가 표본 오차 안인가(독립 가정, 같은 N)
for N in (140, 210, 420):
    a = rng.multinomial(N, conf(*T2["o1"][:3], 1), size=20000)
    b = rng.multinomial(N, conf(*T2["o1 few-shot"][:3], 1), size=20000)
    ka = np.array([kappa_from(s) for s in a if (s[0] + s[2]) > 0 and (s[0] + s[1]) > 0])
    kb = np.array([kappa_from(s) for s in b if (s[0] + s[2]) > 0 and (s[0] + s[1]) > 0])
    m = min(len(ka), len(kb))
    d = kb[:m] - ka[:m]
    res.setdefault("kappa_diff_fewshot_minus_zeroshot", {})[N] = {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "P(diff<=0)": float((d <= 0).mean())}
    print("κ 차이(few-shot - zero-shot) N=%d: 평균 %.2f 구간 [%.2f, %.2f] P(<=0)=%.4f" % (N, d.mean(), *np.percentile(d, [2.5, 97.5]), (d <= 0).mean()))

# (3) 겹치는 trace에서 LLM 판정과 사람 대조
SC = os.path.join(ROOT, ".scratch")
t = open(os.path.join(SC, "interannotator_annotations_new_benchmarks_trim.pdf.txt")).read()
rows = []
for l in t.split("\n"):
    tk = re.findall(r"\b(TRUE|FALSE)\b", l)
    if tk:
        v = [x == "TRUE" for x in tk]
        v += [np.nan] * (12 - len(v))
        rows.append(v)
G = np.array(rows, dtype=float).reshape(14, 4, 3)
F = json.load(open(os.path.join(ROOT, "data/hf/MAD_full_dataset.json")))
MODES = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3.1", "3.2", "3.3"]
pairs = {"GAIA-Magentic": (2, 786), "GAIA-OpenManus(MAST-Data에서는 OpenManus ProgramDev)": (3, 75)}
llm_all, hum_all = [], []
res["judge_vs_human"] = {}
for name, (col, idx) in pairs.items():
    llm = np.array([F[idx]["mast_annotation"][m] for m in MODES], dtype=float)
    h = G[:, col, :]
    maj = (np.nansum(h, axis=1) >= 2).astype(float)
    llm_all.append(llm); hum_all.append(maj)
    res["judge_vs_human"][name] = {"llm_flags": [m for m, v in zip(MODES, llm) if v == 1], "human_majority_flags": [m for m, v in zip(MODES, maj) if v == 1],
                                    "human_any_flags": [m for m, v in zip(MODES, np.nansum(h, axis=1) >= 1) if v], "agree_cells": int((llm == maj).sum()), "cells": 14}
    print(name, res["judge_vs_human"][name])
L = np.concatenate(llm_all); Hm = np.concatenate(hum_all)
tp = int(((L == 1) & (Hm == 1)).sum()); fp = int(((L == 1) & (Hm == 0)).sum()); fn = int(((L == 0) & (Hm == 1)).sum()); tn = int(((L == 0) & (Hm == 0)).sum())
res["judge_vs_human_pooled"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / 28, "kappa": kappa_from(np.array([tp, fp, fn, tn], float)) if (tp + fn) and (tp + fp) else None}
print("겹치는 2 trace 합산", res["judge_vs_human_pooled"])
json.dump(res, open(os.path.join(HERE, "judge_results.json"), "w"), ensure_ascii=False, indent=1)
