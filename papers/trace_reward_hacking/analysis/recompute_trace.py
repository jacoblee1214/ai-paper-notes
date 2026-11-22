"""TRACE 논문(arXiv:2510.01367v4) 공개 수치 재계산과 토이.
1) 막대 라벨(그림 1, 9, 10)로 초록의 '65%·30% 이득' 재계산
2) F1 차이의 표본 수 민감도(검출 표본 수가 논문에 없어 가정 N으로 계산)
3) TRACE 점수 토이: 정직 대 해킹 통과율 곡선, 표본당 잡음(m=5), 임계값, F1·AUROC
실행: python3 analysis/recompute_trace.py  -> analysis/recompute_trace_results.json
"""
import json, math, pathlib
import numpy as np
from scipy import stats

OUT = pathlib.Path(__file__).with_suffix("").with_name("recompute_trace_results.json")

# ---- 1. 논문 막대 라벨 (소수 셋째 자리, 반올림 오차 +-0.0005)
fig1 = {"math_IC_Qwen14B": {"mon72B": 0.522, "trace": 0.883}, "math_RM_Qwen14B": {"mon72B": 0.395, "trace": 0.875}}
fig10 = {  # (monitor72B, trace)
 "IC": {"LLaMA3.2-3B": (0.755, 0.964), "Qwen2.5-3B": (0.660, 0.932), "Qwen2.5-7B": (0.595, 0.898), "Qwen2.5-14B": (0.517, 0.883)},
 "RM": {"LLaMA3.2-3B": (0.830, 0.956), "Qwen2.5-3B": (0.528, 0.927), "Qwen2.5-7B": (0.558, 0.878), "Qwen2.5-14B": (0.394, 0.875)},
}
fig9 = {"IC_Qwen1.5B": (0.159, 0.654), "IC_Qwen3B": (0.050, 0.587), "RM_Qwen3B": (0.457, 0.593)}
res = {"gain": {}}
def gain(m, t): return t / m - 1
def gain_range(m, t, e=0.0005): return ((t - e) / (m + e) - 1, (t + e) / (m - e) - 1)
res["gain"]["fig1"] = {k: {**v, "rel_gain": gain(v["mon72B"], v["trace"])} for k, v in fig1.items()}
res["gain"]["fig10"] = {s: {k: {"monitor": m, "trace": t, "rel_gain": gain(m, t), "abs_gain": t - m} for k, (m, t) in d.items()} for s, d in fig10.items()}
res["gain"]["fig10_min_rel_gain"] = {s: min(gain(m, t) for m, t in d.values()) for s, d in fig10.items()}
res["gain"]["fig10_median_rel_gain"] = {s: float(np.median([gain(m, t) for m, t in d.values()])) for s, d in fig10.items()}
res["gain"]["fig9"] = {k: {"monitor32B": m, "trace": t, "rel_gain": gain(m, t), "rel_gain_rounding_range": gain_range(m, t)} for k, (m, t) in fig9.items()}
res["gain"]["claim_math_over_65pct_holds_for"] = [f"{s}/{k}" for s, d in fig10.items() for k, (m, t) in d.items() if gain(m, t) > 0.65]
res["gain"]["claim_code_over_30pct_RM_max_with_rounding"] = gain_range(0.457, 0.593)[1]
res["gain"]["fig1_vs_fig10_mismatch"] = {"IC_monitor": (0.522, 0.517), "RM_monitor": (0.395, 0.394)}

# ---- 2. F1 차이의 표본 수 민감도. 균형 클래스, TPR=1-FPR=a 이면 F1=a (정밀도=재현율=a).
def f1_ci(a, N, B=4000, rng=np.random.default_rng(0)):
    tp = rng.binomial(N, a, B); fp = rng.binomial(N, 1 - a, B)
    f1 = 2 * tp / (2 * tp + fp + (N - tp))
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))
def two_prop_p(a1, a2, N):  # 독립 가정 근사, 양쪽 표본 2N
    n = 2 * N; p = (a1 + a2) / 2
    z = (a1 - a2) / math.sqrt(2 * p * (1 - p) / n)
    return 2 * (1 - stats.norm.cdf(abs(z)))
def min_N(a1, a2):
    for N in range(5, 5000):
        if two_prop_p(a1, a2, N) < 0.05: return N
sens = {}
for name, (m, t) in {"math_IC_LLaMA3B(최소 격차)": fig10["IC"]["LLaMA3.2-3B"], "math_RM_LLaMA3B": fig10["RM"]["LLaMA3.2-3B"],
                     "code_RM_Qwen3B": fig9["RM_Qwen3B"], "math_IC_Qwen14B": fig10["IC"]["Qwen2.5-14B"]}.items():
    sens[name] = {"gap": t - m, "min_N_per_class_for_p<.05": min_N(m, t),
                  "CI95_trace_N50": f1_ci(t, 50), "CI95_trace_N200": f1_ci(t, 200), "CI95_monitor_N50": f1_ci(m, 50), "CI95_monitor_N200": f1_ci(m, 200)}
res["sample_size_sensitivity_balanced_assumption"] = sens

# ---- 3. 토이
rng = np.random.default_rng(1)
GRID = np.arange(0.1, 1.01, 0.1)  # 10%..100% (논문은 격자를 명시하지 않음; 그림 5의 10%..90% + 끝을 가정)
def curve_honest(n, t_lo=0.6, t_hi=1.0, q=0.9, s=0.05):
    t = rng.uniform(t_lo, t_hi, n)[:, None]
    return q / (1 + np.exp(-(GRID[None] - t) / s))
def curve_hack(n, rate=0.9, q=0.95):
    # 해킹: 일부 표본은 처음부터 보상(힌트 복사), 나머지는 정직 곡선
    h = rng.random(n) < rate
    base = curve_honest(n)
    out = np.where(h[:, None], q, base)
    return out, h
def score(p, m, rng=rng):
    ph = rng.binomial(m, np.clip(p, 0, 1)) / m
    return 100 * ph.mean(axis=1)  # 0..100 (논문 그림의 척도)
def auroc(pos, neg):
    return float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))
def f1_at(pos, neg, thr):
    tp = (pos > thr).sum(); fp = (neg > thr).sum(); fn = (pos <= thr).sum()
    return float(2 * tp / (2 * tp + fp + fn))
toy = {}
n = 4000
hon = curve_honest(n)
for m in (1, 5, 20):
    sh = score(hon, m); init_mean = sh.mean()
    ph, _ = curve_hack(n, rate=1.0)
    sk = score(ph, m)
    # 해킹 모델이 일부(rate)만 해킹: 양성은 해킹 표본, 음성은 정직 표본
    ph2, hmask = curve_hack(n, rate=0.6)
    s2 = score(ph2, m)
    toy[f"m={m}"] = {
        "honest_mean": float(sh.mean()), "honest_sd": float(sh.std()), "hack_mean": float(sk.mean()), "hack_sd": float(sk.std()),
        "AUROC_hack_vs_honest": auroc(sk, sh),
        "thr=honest_mean: FPR": float((sh > init_mean).mean()), "thr=honest_mean: F1(balanced)": f1_at(sk, sh, init_mean),
        "thr=honest_p95: FPR": float((sh > np.percentile(sh, 95)).mean()), "thr=honest_p95: F1(balanced)": f1_at(sk, sh, np.percentile(sh, 95)),
        "mixed60%: AUROC(hacked vs not, same model)": auroc(s2[hmask], s2[~hmask]),
        "mixed60%: F1@honest_mean": f1_at(s2[hmask], s2[~hmask], init_mean),
    }
res["toy_grid"] = "cutoffs 10%..100%, score=100*mean of per-cutoff pass rate"
res["toy_basic"] = toy
# 과잉 사고(overthinking): 정직 모델이 RL 뒤 일찍 정답에 도달하는 비율이 커지면(t_hi 감소) 해킹 없이도 점수가 오른다
ov = {}
base = score(curve_honest(n), 5); thr = base.mean()
for lo, hi in [(0.6, 1.0), (0.5, 0.9), (0.4, 0.8), (0.3, 0.7)]:
    s = score(curve_honest(n, lo, hi), 5)
    ov[f"t~U({lo},{hi})"] = {"mean_score": float(s.mean()), "frac_above_initial_mean": float((s > thr).mean())}
res["toy_overthinking_false_positive"] = ov
# 형태 효과: 해킹 곡선이 늦게 오르지만(마지막 20%에서만 보상, 즉 '짧은 CoT에서는 안 보이는' 해킹) 점수는 정직과 같다
late = np.where(GRID[None] >= 0.9, 0.95, 0.0) * np.ones((n, 1))
res["toy_late_hack_score_equals_honest"] = {"late_hack_mean": float(score(late, 5).mean()), "honest_mean": float(base.mean())}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
