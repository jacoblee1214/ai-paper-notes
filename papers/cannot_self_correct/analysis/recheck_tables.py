"""Huang et al. (arXiv:2310.01798v2) 표 재계산과 통계 검정.

입력은 논문 표 2-8과 Figure 1의 공개 수치뿐이다(문항 단위 결과는 공개되지 않았다).
- 격자 검사: 보고된 정확도가 문항 수 n의 정수 개수로 나오는가
- Figure 1 -> 짝지은 2x2 분할표 복원, Table 3/4와 일치 검사, McNemar 정확 검정
- Table 3-6 각 칸: 표준 프롬프트 대비 비짝 검정(보수적), Wilson 구간
- Table 7: 다중 에이전트 토론 vs 자기 일관성 (n=1319, 비짝)
- 최소 검출 차이(MDE)
- v1 무작위 기준선 식 검산
실행: python3 -I analysis/recheck_tables.py  -> analysis/recheck_tables_results.json
"""
import json
import math
import os
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recheck_tables_results.json")
R = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def unpaired(p1, p2, n1, n2):
    """두 정확도(%)의 비짝 z 검정. 짝지은 상관이 양수면 실제 검정력은 이보다 높다."""
    a, b = p1 / 100, p2 / 100
    pp = (a * n1 + b * n2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (b - a) / se
    return round(100 * (b - a), 2), round(z, 2), round(2 * (1 - stats.norm.cdf(abs(z))), 4)


# ---------- 1. 격자 검사 ----------
# (이름, n, [보고 정확도])
grid = {
    "T2/T3 GSM8K GPT-3.5": (1319, [75.9, 84.3, 75.1, 74.7]),
    "T2/T3 CSQA GPT-3.5": (1221, [75.8, 89.7, 38.1, 41.8]),
    "T2/T3 HotpotQA GPT-3.5": (100, [26.0, 29.0, 25.0, 25.0]),
    "T2/T3 GSM8K GPT-4": (200, [95.5, 97.5, 91.5, 89.0]),
    "T2/T3 CSQA GPT-4": (200, [82.0, 85.5, 79.5, 80.0]),
    "T2/T3 HotpotQA GPT-4": (100, [49.0, 59.0, 49.0, 43.0]),
    "T4 GSM8K GPT-4-Turbo": (200, [91.5, 88.0, 90.0]),
    "T4 CSQA GPT-4-Turbo": (200, [84.0, 81.5, 83.0]),
    "T4 GSM8K Llama-2": (200, [62.0, 43.5, 36.5]),
    "T4 CSQA Llama-2": (200, [64.0, 37.5, 36.5]),
    "T7 GSM8K debate/SC": (1319, [76.7, 82.5, 83.2, 85.3, 83.0, 88.2]),
}
bad = []
for name, (n, vals) in grid.items():
    for v in vals:
        k = round(v / 100 * n)
        if round(100 * k / n, 1) != v:
            bad.append([name, v, k])
R["grid_check"] = {"n_values_checked": sum(len(v[1]) for v in grid.values()), "inconsistent": bad}

# ---------- 2. Figure 1 -> 짝지은 분할표 ----------
# 출처: Figure 1 파이 라벨(%). C2I=정답->오답, I2C=오답->정답, I2I=오답->다른 오답, NC=변화 없음
# acc0, acc2: Table 3 / Table 4의 표준 프롬프트와 round 2 정확도
fig1 = {
    "GSM8K GPT-3.5": dict(n=1319, C2I=8.8, I2C=7.6, I2I=8.9, NC=74.7, acc0=75.9, acc2=74.7),
    "CSQA GPT-3.5": dict(n=1221, C2I=39.8, I2C=5.8, I2I=11.6, NC=42.8, acc0=75.8, acc2=41.8),
    "GSM8K GPT-4": dict(n=200, C2I=8.0, I2C=1.5, I2I=0.0, NC=90.5, acc0=95.5, acc2=89.0),
    "CSQA GPT-4": dict(n=200, C2I=5.0, I2C=3.0, I2I=1.5, NC=90.5, acc0=82.0, acc2=80.0),
    "GSM8K GPT-4-Turbo": dict(n=200, C2I=2.5, I2C=1.0, I2I=0.5, NC=96.0, acc0=91.5, acc2=90.0),
    "CSQA GPT-4-Turbo": dict(n=200, C2I=6.0, I2C=5.0, I2I=1.0, NC=88.0, acc0=84.0, acc2=83.0),
    "GSM8K Llama-2": dict(n=200, C2I=31.0, I2C=5.5, I2I=23.5, NC=40.0, acc0=62.0, acc2=36.5),
    "CSQA Llama-2": dict(n=200, C2I=35.5, I2C=8.0, I2I=16.5, NC=40.0, acc0=64.0, acc2=36.5),
}
R["figure1_paired"] = {}
for name, d in fig1.items():
    n = d["n"]
    sum_pct = round(d["C2I"] + d["I2C"] + d["I2I"] + d["NC"], 1)
    b = round(d["C2I"] / 100 * n)
    c = round(d["I2C"] / 100 * n)
    net_fig = round(d["I2C"] - d["C2I"], 2)
    net_tab = round(d["acc2"] - d["acc0"], 2)
    # McNemar 정확 검정(양측)
    p_mc = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    diff = (c - b) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0)) / n  # 짝지은 차이의 Wald SE
    lo, hi = 100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)
    # 내재 오라클 분해: 'NC 정답' = acc0 - C2I, 초기 오답 중 I2C 비율
    init_wrong = 100 - d["acc0"]
    R["figure1_paired"][name] = dict(
        n=n, pct_sum=sum_pct, C2I_n=b, I2C_n=c,
        net_fig1_pts=net_fig, net_table_pts=net_tab, match=bool(abs(net_fig - net_tab) < 0.051),
        mcnemar_exact_p=round(float(p_mc), 4), paired_diff_pts=round(100 * diff, 2),
        paired_diff_ci95=[round(lo, 2), round(hi, 2)],
        harm_to_gain_ratio=round(b / c, 2) if c else None,
        I2C_share_of_initially_wrong_pct=round(100 * d["I2C"] / init_wrong, 1),
        C2I_share_of_initially_right_pct=round(100 * d["C2I"] / d["acc0"], 1),
    )

# ---------- 3. Table 3-6: 표준 대비 비짝 검정 ----------
rows = [
    # (이름, n, 표준, [(라벨, 값), ...])
    ("T3 GSM8K GPT-3.5", 1319, 75.9, [("r1", 75.1), ("r2", 74.7)]),
    ("T3 CSQA GPT-3.5", 1221, 75.8, [("r1", 38.1), ("r2", 41.8)]),
    ("T3 HotpotQA GPT-3.5", 100, 26.0, [("r1", 25.0), ("r2", 25.0)]),
    ("T3 GSM8K GPT-4", 200, 95.5, [("r1", 91.5), ("r2", 89.0)]),
    ("T3 CSQA GPT-4", 200, 82.0, [("r1", 79.5), ("r2", 80.0)]),
    ("T3 HotpotQA GPT-4", 100, 49.0, [("r1", 49.0), ("r2", 43.0)]),
    ("T4/T5 GSM8K GPT-4-Turbo", 200, 91.5, [("r1", 88.0), ("r2", 90.0), ("p2 r1", 90.0), ("p2 r2", 90.0), ("p3 r1", 91.0), ("p3 r2", 91.0)]),
    ("T4/T5 CSQA GPT-4-Turbo", 200, 84.0, [("r1", 81.5), ("r2", 83.0), ("p2 r1", 74.5), ("p2 r2", 81.0), ("p3 r1", 81.5), ("p3 r2", 83.5)]),
    ("T4/T6 GSM8K Llama-2", 200, 62.0, [("r1", 43.5), ("r2", 36.5), ("p2 r1", 46.5), ("p2 r2", 30.5), ("p3 r1", 58.0), ("p3 r2", 41.5)]),
    ("T4/T6 CSQA Llama-2", 200, 64.0, [("r1", 37.5), ("r2", 36.5), ("p2 r1", 26.0), ("p2 r2", 37.0), ("p3 r1", 24.0), ("p3 r2", 43.0)]),
]
R["unpaired_vs_standard"] = {}
for name, n, std, items in rows:
    out = {"std_wilson95": wilson(round(std / 100 * n), n)}
    for lab, v in items:
        dlt, z, p = unpaired(std, v, n, n)
        out[lab] = dict(value=v, delta=dlt, z=z, p=p, sig_unpaired_05=bool(p < 0.05))
    R["unpaired_vs_standard"][name] = out

# ---------- 4. Table 7: 토론 vs 자기 일관성 (n=1319, 비짝) ----------
t7 = {
    "SC3 vs MAD r1 (3 agents 기준, 82.5 vs 83.2)": (82.5, 83.2),
    "MAD r1(6) vs SC6 (83.2 vs 85.3)": (83.2, 85.3),
    "MAD r2(9) vs SC9 (83.0 vs 88.2)": (83.0, 88.2),
    "MAD r2(9) vs SC3 (82.5 vs 83.0)": (82.5, 83.0),
    "Std vs SC3 (76.7 vs 82.5)": (76.7, 82.5),
}
R["table7"] = {}
for k, (a, b) in t7.items():
    dlt, z, p = unpaired(a, b, 1319, 1319)
    R["table7"][k] = dict(delta=dlt, z=z, p=p)
# v1 보고: 표준 프롬프트 9회 반복 SD = 0.91점. 이항 표준오차와 비교
p = 0.767
R["table7_noise"] = dict(
    binomial_se_pts=round(100 * math.sqrt(p * (1 - p) / 1319), 2),
    v1_reported_sd_over_9_runs_pts=0.91,
)

# ---------- 5. 최소 검출 차이(비짝, 80% 검정력, 양측 5%) ----------
def mde(n, p0):
    za, zb = 1.96, 0.8416
    return round(100 * (za + zb) * math.sqrt(2 * p0 * (1 - p0) / n), 1)

R["mde_pts_unpaired"] = {
    f"n={n}, p0={p0}": mde(n, p0)
    for n, p0 in [(1319, 0.76), (1221, 0.76), (200, 0.9), (200, 0.6), (100, 0.26), (100, 0.49)]
}

# ---------- 6. 오라클 상한 분해 (GPT-3.5 GSM8K/CSQA) ----------
orc = {"GSM8K GPT-3.5": (75.9, 84.3, 7.6, 8.8), "CSQA GPT-3.5": (75.8, 89.7, 5.8, 39.8)}
R["oracle_decomposition"] = {
    k: dict(acc0=a0, oracle_acc=ao, oracle_gain=round(ao - a0, 1), intrinsic_I2C=ic, intrinsic_C2I=ci,
            intrinsic_net=round(ic - ci, 1),
            oracle_fix_share_of_wrong_pct=round(100 * (ao - a0) / (100 - a0), 1),
            intrinsic_fix_share_of_wrong_pct=round(100 * ic / (100 - a0), 1),
            intrinsic_break_share_of_right_pct=round(100 * ci / a0, 1))
    for k, (a0, ao, ic, ci) in orc.items()
}

# ---------- 7. v1 무작위 기준선(Table 2 of v1) ----------
def rand_base(x0, rounds, denom_offset):
    x = x0
    out = []
    for k in range(rounds):
        x = x + (1 - x) / (denom_offset - k)
        out.append(round(100 * x, 1))
    return out

R["v1_random_baseline"] = {
    "paper_formula_5_minus_k": {"GPT-3.5": rand_base(0.758, 4, 5), "GPT-4": rand_base(0.82, 4, 5)},
    "formula_4_minus_k": {"GPT-3.5": rand_base(0.758, 4, 4), "GPT-4": rand_base(0.82, 4, 4)},
    "v1_reported_round2": {"GPT-3.5": 87.9, "GPT-4": 91.0},
}

json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
