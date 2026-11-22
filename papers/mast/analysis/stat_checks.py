"""논문 표 5(개입 실험)와 표 7(성공/실패별 실패 모드 비율)의 표본 수 복원과 검정.
표의 비율에서 분모를 역산하고 Fisher 정확 검정, 비율 차이의 Wilson/Newcombe 구간을 구한다.
실행: python3 -I analysis/stat_checks.py
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def diff_ci(k1, n1, k2, n2):
    """Newcombe(1998) 방법 10: 두 비율 차이의 95% 구간."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def denominators(pct, nmax=400, tol=0.051):
    """보고된 퍼센트(소수 첫째 자리)가 k/n으로 설명되는 n 후보."""
    return [n for n in range(1, nmax + 1) if any(abs(100 * k / n - pct) <= tol for k in range(n + 1))]


# 표 5 ChatDev ProgramDev-v0: n=32 가정(논문 H.2: 32개 과제)
res["table5_chatdev_v0"] = {}
base = (8, 32)
for name, k in [("improved prompt", 11), ("new topology", 13)]:
    fe = stats.fisher_exact([[k, 32 - k], [base[0], 32 - base[0]]])
    lo, hi = diff_ci(k, 32, base[0], 32)
    res["table5_chatdev_v0"][name] = {"k": k, "n": 32, "pct": 100 * k / 32, "diff_pp": 100 * (k - base[0]) / 32, "fisher_p": fe[1], "diff95_pp": [100 * lo, 100 * hi]}
    print("ChatDev v0", name, res["table5_chatdev_v0"][name])
# 보고 퍼센트가 k/32로 설명되는지
res["table5_pct_vs_32"] = {str(p): any(abs(100 * k / 32 - p) < 0.051 for k in range(33)) for p in (25.0, 34.4, 40.6)}
# HumanEval n=164
res["table5_humaneval_n164"] = {str(p): [k for k in range(165) if abs(100 * k / 164 - p) < 0.051] for p in (89.6, 90.3, 91.5)}
print("HumanEval 164 분모 설명", res["table5_humaneval_n164"])
for name, k in [("improved prompt(90.2%로 반올림되는 148 사용)", 148), ("new topology", 150)]:
    fe = stats.fisher_exact([[k, 164 - k], [147, 164 - 147]])
    res["table5_humaneval_n164"][name + " fisher_p"] = fe[1]
    print("HumanEval", name, "p=%.3f" % fe[1])
# AG2 GSM-Plus: 200문항, ±는 보고값. 800 시행(4회 x 200)이면 84.75=678/800
for gpt, (b, i, t) in {"GPT-4": (84.75, 89.75, 85.50), "GPT-4o": (84.25, 89.00, 88.83)}.items():
    out = {}
    for name, v in [("improved prompt", i), ("new topology", t)]:
        # 독립 시행 800개 가정(낙관), 문항 200개가 유효 표본이라는 가정(보수)
        for n in (800, 200):
            kb, kv = round(b / 100 * n), round(v / 100 * n)
            fe = stats.fisher_exact([[kv, n - kv], [kb, n - kb]])
            out[f"{name} n={n}"] = fe[1]
    res[f"table5_ag2_{gpt}"] = out
    print("AG2", gpt, {k: round(v, 4) for k, v in out.items()})

# 표 7: ChatDev success n=10 fail n=20, MetaGPT success n=12 fail n=18 (퍼센트 배수에서 역산)
t7 = {
    "ChatDev": {"succ_n": 10, "fail_n": 20,
                "succ": [20.0, 0.0, 20.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 0.0, 10.0, 20.0],
                "fail": [25.0, 0.0, 20.0, 5.0, 10.0, 5.0, 5.0, 5.0, 5.0, 0.0, 15.0, 5.0, 10.0, 25.0]},
    "MetaGPT": {"succ_n": 12, "fail_n": 18,
                "succ": [33.3, 0.0, 16.7, 0.0, 0.0, 0.0, 8.3, 8.3, 0.0, 0.0, 8.3, 0.0, 16.7, 16.7],
                "fail": [16.7, 0.0, 22.2, 5.6, 11.1, 5.6, 5.6, 5.6, 5.6, 0.0, 16.7, 5.6, 5.6, 27.8]},
}
MODES = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3.1", "3.2", "3.3"]
res["table7"] = {}
pooled = {m: [0, 0, 0, 0] for m in MODES}  # succ_k, succ_n, fail_k, fail_n
for mas, d in t7.items():
    ks = [round(p / 100 * d["succ_n"]) for p in d["succ"]]
    kf = [round(p / 100 * d["fail_n"]) for p in d["fail"]]
    chk_s = all(abs(100 * k / d["succ_n"] - p) < 0.06 for k, p in zip(ks, d["succ"]))
    chk_f = all(abs(100 * k / d["fail_n"] - p) < 0.06 for k, p in zip(kf, d["fail"]))
    res["table7"][mas] = {"succ_n": d["succ_n"], "fail_n": d["fail_n"], "k_succ": ks, "k_fail": kf, "pct_consistent_succ": chk_s, "pct_consistent_fail": chk_f,
                          "flags_per_trace_succ": sum(ks) / d["succ_n"], "flags_per_trace_fail": sum(kf) / d["fail_n"]}
    print("표 7", mas, res["table7"][mas])
    for m, a, b in zip(MODES, ks, kf):
        pooled[m][0] += a; pooled[m][1] += d["succ_n"]; pooled[m][2] += b; pooled[m][3] += d["fail_n"]
# 논문 주장: 1.5와 2.4는 실패 run에서만(거의) 나타남 / 3.2·3.3은 성공 run에서도 흔함
res["table7_pooled_fisher"] = {}
for m in MODES:
    ks, ns, kf, nf = pooled[m]
    fe = stats.fisher_exact([[kf, nf - kf], [ks, ns - ks]])
    res["table7_pooled_fisher"][m] = {"succ": f"{ks}/{ns}", "fail": f"{kf}/{nf}", "p": fe[1]}
print("합산 Fisher", {m: (v["succ"], v["fail"], round(v["p"], 3)) for m, v in res["table7_pooled_fisher"].items() if m in ("1.5", "2.4", "3.2", "3.3", "1.1")})
# 총 실패 건수/ trace: 실패 run이 더 많은가 (조건부 이항 검정, 포아송 비율)
for mas, r in res["table7"].items():
    ks, kf = sum(r["k_succ"]), sum(r["k_fail"])
    ns, nf = r["succ_n"], r["fail_n"]
    # H0: 두 군의 trace당 비율 동일 -> kf ~ Binomial(ks+kf, nf/(ns+nf))
    bt = stats.binomtest(kf, ks + kf, nf / (ns + nf), alternative="greater")
    r["poisson_rate_test_p"] = bt.pvalue
    print("총 건수", mas, ks, kf, "trace당", ks / ns, kf / nf, "p(실패 run이 더 많음)=%.3f" % bt.pvalue)
json.dump(res, open(os.path.join(HERE, "stat_results.json"), "w"), ensure_ascii=False, indent=1)
