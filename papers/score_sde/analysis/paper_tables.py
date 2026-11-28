"""논문 표 1(표집기 비교), 표 2(NLL·FID), 표 3의 수치를 옮겨 재계산하고 통계 검정한다.

표 1은 CIFAR-10 FID 평균 ± 표준편차, 표집 5회(n=5). 표준편차가 표본 표준편차(ddof=1)인지 모집단 표준편차(ddof=0)인지
적혀 있지 않아 두 가정을 모두 계산한다. 학습 시드 변동은 표에 없다(체크포인트 하나로 5회 표집).
실행: python paper_tables.py  (수 초)  ->  paper_tables_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

N = 5
# 표 1: {SDE: {sampler: {column: (mean, std)}}}
T1 = {
    "VE": {
        "ancestral": {"P1000": (4.98, .06), "P2000": (4.88, .06), "PC1000": (3.62, .03)},
        "reverse_diffusion": {"P1000": (4.79, .07), "P2000": (4.74, .08), "C2000": (20.43, .07), "PC1000": (3.60, .02)},
        "probability_flow": {"P1000": (15.41, .15), "P2000": (10.54, .08), "PC1000": (3.51, .04)},
    },
    "VP": {
        "ancestral": {"P1000": (3.24, .02), "P2000": (3.24, .02), "PC1000": (3.21, .02)},
        "reverse_diffusion": {"P1000": (3.21, .02), "P2000": (3.19, .02), "C2000": (19.06, .06), "PC1000": (3.18, .01)},
        "probability_flow": {"P1000": (3.59, .04), "P2000": (3.23, .03), "PC1000": (3.06, .03)},
    },
}
# 표 2: (모델, SDE, NLL bits/dim, FID(ODE 표집))  --- NLL은 식 39의 정확한 우도(5회 허친슨 평균)
T2 = [
    ("DDPM", "VP", 3.28, 3.37), ("DDPM cont.", "VP", 3.21, 3.69), ("DDPM cont.", "sub-VP", 3.05, 3.56),
    ("DDPM++ cont.", "VP", 3.16, 3.93), ("DDPM++ cont.", "sub-VP", 3.02, 3.16),
    ("DDPM++ cont. deep", "VP", 3.13, 3.08), ("DDPM++ cont. deep", "sub-VP", 2.99, 2.92),
]
# 표 3: (모델, FID, IS)  PC/EM 표집, 학습 중 FID 최소 체크포인트
T3 = [
    ("DDPM", 3.17, 9.46), ("DDPM++", 2.78, 9.64), ("DDPM++ cont. (VP)", 2.55, 9.58), ("DDPM++ cont. (sub-VP)", 2.61, 9.56),
    ("DDPM++ cont. (deep, VP)", 2.41, 9.68), ("DDPM++ cont. (deep, sub-VP)", 2.41, 9.57), ("NCSN++", 2.45, 9.73),
    ("NCSN++ cont. (VE)", 2.38, 9.83), ("NCSN++ cont. (deep, VE)", 2.20, 9.89),
]


def welch(a, b, ddof1=True):
    f = math.sqrt(N / (N - 1)) if not ddof1 else 1.0   # ddof=0로 보고된 std를 표본 std로 환산
    r = stats.ttest_ind_from_stats(a[0], a[1] * f, N, b[0], b[1] * f, N, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def table1():
    rows = []
    for sde, d in T1.items():
        for pred, cols in d.items():
            rows.append((sde, f"PC1000({pred}) vs P2000({pred}) [같은 NFE 2000]", cols["PC1000"], cols["P2000"]))
        for pred, cols in d.items():
            rows.append((sde, f"PC1000({pred}) vs P1000({pred}) [교정기 추가, 연산 2배]", cols["PC1000"], cols["P1000"]))
        for col in ("P1000", "P2000", "PC1000"):
            rows.append((sde, f"reverse_diffusion vs ancestral at {col}", d["reverse_diffusion"][col], d["ancestral"][col]))
        rows.append((sde, "P2000 vs P1000 (reverse_diffusion) [연산 2배]", d["reverse_diffusion"]["P2000"], d["reverse_diffusion"]["P1000"]))
    out = []
    p_a, p_b = [], []
    for sde, label, a, b in rows:
        ta, pa = welch(a, b, True)
        tb, pb = welch(a, b, False)
        out.append({"sde": sde, "cmp": label, "a": a, "b": b, "diff": round(a[0] - b[0], 4), "t_ddof1": ta, "p_ddof1": pa,
                    "p_ddof0": pb})
        p_a.append(pa)
        p_b.append(pb)
    ha, hb = holm(p_a), holm(p_b)
    for r, x, y in zip(out, ha, hb):
        r["p_holm_ddof1"], r["p_holm_ddof0"] = x, y
    return out


def table2():
    out = {"pairs_sub_vp_vs_vp": []}
    by = {}
    for model, sde, nll, fid in T2:
        by[(model, sde)] = (nll, fid)
    for model in ("DDPM cont.", "DDPM++ cont.", "DDPM++ cont. deep"):
        a, b = by[(model, "sub-VP")], by[(model, "VP")]
        out["pairs_sub_vp_vs_vp"].append({"model": model, "nll_sub_minus_vp": round(a[0] - b[0], 3),
                                           "fid_sub_minus_vp": round(a[1] - b[1], 3)})
    d = [r["nll_sub_minus_vp"] for r in out["pairs_sub_vp_vs_vp"]]
    out["sign_test_nll_p_two_sided"] = float(stats.binomtest(sum(x < 0 for x in d), len(d), 0.5).pvalue)
    f = [r["fid_sub_minus_vp"] for r in out["pairs_sub_vp_vs_vp"]]
    out["fid_signs_sub_minus_vp"] = f
    # 같은 모델이 표 2와 표 3에서 다른 FID: 표집기와 체크포인트 선택 차이
    out["same_model_fid_table2_vs_table3"] = [
        {"model": "DDPM++ cont. (VP)", "t2_ode_last": 3.93, "t3_pc_best": 2.55},
        {"model": "DDPM++ cont. (sub-VP)", "t2_ode_last": 3.16, "t3_pc_best": 2.61},
        {"model": "DDPM++ cont. deep (VP)", "t2_ode_last": 3.08, "t3_pc_best": 2.41},
        {"model": "DDPM++ cont. deep (sub-VP)", "t2_ode_last": 2.92, "t3_pc_best": 2.41},
    ]
    # 초록의 세 수치가 한 모델에서 나오는가
    out["abstract_numbers"] = {"IS_9.89_FID_2.20": "NCSN++ cont. (deep, VE), 표 3 (PC 표집, 최소 FID 체크포인트)",
                               "NLL_2.99": "DDPM++ cont. (deep, sub-VP), 표 2 (ODE, 마지막 체크포인트), 같은 모델의 표 2 FID는 2.92, 표 3 FID는 2.41",
                               "same_model": False}
    out["ve_has_nll"] = False
    return out


def table3():
    fids = [t[1] for t in T3]
    # 5회 표집 잡음(표 1의 std 0.01-0.08)보다 작은 차이를 표시
    pairs = []
    for i in range(len(T3)):
        for j in range(i + 1, len(T3)):
            d = abs(T3[i][1] - T3[j][1])
            if d <= 0.07:
                pairs.append((T3[i][0], T3[j][0], round(d, 2)))
    return {"pairs_within_0p07_fid": pairs, "best_fid": min(fids),
            "note": "표 3은 학습 중 FID가 가장 낮은 체크포인트를 고른 값이라 선택 편향이 있고, 학습 시드 반복은 없다"}


def main():
    res = {"table1": table1(), "table2": table2(), "table3": table3()}
    out = Path(__file__).resolve().parent / "paper_tables_results.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for r in res["table1"]:
        print(f"{r['sde']:2s} {r['cmp'][:58]:58s} d={r['diff']:7.2f} p={r['p_ddof1']:.4f} holm={r['p_holm_ddof1']:.4f}")
    print(json.dumps(res["table2"], indent=1, ensure_ascii=False)[:1500])
    print(res["table3"])


if __name__ == "__main__":
    main()
