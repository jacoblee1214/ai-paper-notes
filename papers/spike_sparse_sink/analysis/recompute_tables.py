"""논문 표(Table 1–9)와 식을 다시 계산하고 일관성을 검사한다. 표준 라이브러리 + numpy + scipy.
실행: python3 recompute_tables.py  -> recompute_tables_results.json
표 값은 paper/paper.txt에서 손으로 옮겼다(출처: 표 번호). 모델 학습·평가는 하지 않는다.
"""
import json, math
import numpy as np
from scipy.stats import spearmanr

R = {}

# ---------- Table 2: 어휘 전수 탐사 비율 ----------
t2 = {  # model: (vocab, spike_tokens, reported_ratio%)
    "Llama 2 7B": (32000, 31887, 99.65), "Llama 2 13B": (32000, 31889, 99.65), "Llama 3 8B": (128256, 127956, 99.77),
    "Qwen2.5 7B": (152064, 149587, 98.40), "Qwen2.5 14B": (152064, 149645, 98.40),
    "Qwen3 8B": (151936, 151830, 99.93), "Qwen3 14B": (151936, 151824, 99.93)}
R["table2"] = {m: {"recomputed_pct": round(100 * s / v, 3), "reported_pct": r, "diff_pp": round(100 * s / v - r, 3)}
               for m, (v, s, r) in t2.items()}

# ---------- Table 1: 블록 수 = 2L, step-up은 짝수(FFN)인가 ----------
t1 = {"Llama 2 7B": (64, [4], [62], 32), "Llama 2 13B": (80, [8], [78, 79], 40), "Llama 3 8B": (64, [4], [64], 32),
      "Qwen2.5 7B": (56, [8, 10], [54, 55], 28), "Qwen2.5 14B": (96, [10], [90, 92, 94, 95], 48),
      "Qwen3 8B": (72, [14], [70, 72], 36), "Qwen3 14B": (80, [14], [79], 40)}
R["table1"] = {m: {"blocks==2L": b == 2 * l, "stepup_all_even(FFN)": all(i % 2 == 0 for i in up),
                   "stepdown_odd(attention)_blocks": [i for i in dn if i % 2 == 1]}
               for m, (b, up, dn, l) in t1.items()}

# ---------- 학습 설정 일관성 (Table 9 vs 본문/공개 config) ----------
R["training_budget"] = {"table9_steps": 5000, "table9_batch_tokens": 2_000_000,
                        "table9_implied_tokens_B": 5000 * 2e6 / 1e9,
                        "text_baseline_tokens_B": 100, "repo_train_0_num_steps": 50000,
                        "repo_implied_tokens_B": 50000 * 2e6 / 1e9,
                        "repo_train_31_num_steps(200B row)": 100000, "repo_train_31_implied_tokens_B": 100000 * 2e6 / 1e9}

# ---------- 모델 파라미터 수(표 4·6의 비교 대상이 같은 크기인가) ----------
d, L_, V_ = 4096, 32, 32000
emb = 2 * V_ * d
def attn_params(nh, dh): return 3 * d * nh * dh + nh * dh * d
def total(ffn_per_layer, attn_per_layer): return emb + L_ * (ffn_per_layer + attn_per_layer)
a0 = attn_params(32, 128)
swi, gel, lin = 3 * d * 11008, 2 * d * 16512, d * d
R["params_table4_B"] = {
    "SwiGLU(baseline)": total(swi, a0) / 1e9, "GeLU(inter 16512)": total(gel, a0) / 1e9,
    "Linear(d x d)": total(lin, a0) / 1e9, "Attention-only(FFN 자리에 attention)": total(a0, a0) / 1e9}
R["params_table6_B"] = {f"dh{dh}_nh{nh}": total(swi, attn_params(nh, dh)) / 1e9
                       for dh, nh in [(128, 32), (64, 32), (32, 32), (16, 32), (8, 32), (128, 16), (128, 8),
                                      (8, 512), (16, 256), (32, 128), (64, 64), (256, 16)]}
tokens = 100e9
flops_per_run = 6 * total(swi, a0) * tokens
R["compute_estimate"] = {"params_B": total(swi, a0) / 1e9, "flops_per_100B_run_e21": flops_per_run / 1e21,
                         "H100_gpu_hours_at_40pct_MFU(989TF peak)": flops_per_run / (989e12 * 0.4) / 3600,
                         "note": "6ND 근사, 우리 추정. 논문은 GPU 시간을 적지 않았다."}

# ---------- Eq. 28 sink ratio: T=64, eps=0.3 의 의미 ----------
def harmonic(T): return sum(1 / t for t in range(1, T + 1))
sens = {}
for T in (64, 256, 1024, 4096):
    uniform_alpha0 = harmonic(T) / T                       # 균일 인과 attention의 위치 0 평균 수신량
    f_needed = (0.3 * T - 1) / (T - 1)                      # 모든 질의(t>=2)가 첫 토큰에 f를 몰아주면 alpha_0=(1+(T-1)f)/T
    sens[T] = {"uniform_causal_alpha0": round(uniform_alpha0, 4), "dump_fraction_needed_for_eps0.3": round(f_needed, 4)}
R["sink_ratio_T_dependence"] = sens
R["sink_ratio_eps_to_dump_fraction_T64"] = {str(e): round((e * 64 - 1) / 63, 4) for e in (0.1, 0.2, 0.3, 0.5, 0.8)}

# ---------- Eq. 20–21: 이차 증폭 항등식 ----------
rng = np.random.default_rng(0)
dm = 4096
S = rng.standard_normal((dm, dm)); S = (S + S.T) / 2
w, U = np.linalg.eigh(S)
star = U[:, -1]; lam = w[-1]
# 임의의 h를 RMSNorm하고 (s^T h~)^2 vs lam*d*cos^2 (맞는 식) vs lam*sqrt(d)*cos (논문 식 21 그대로)
chk = []
for mix in (0.05, 0.3, 0.7, 1.0):
    h = mix * star * 50 + rng.standard_normal(dm)
    ht = np.sqrt(dm) * h / np.linalg.norm(h)
    cos = float(star @ ht / np.linalg.norm(ht))
    exact = lam * float(star @ ht) ** 2
    chk.append({"cos": round(cos, 4), "lam(sTh)^2": exact, "lam*d*cos^2": lam * dm * cos ** 2,
                "paper_eq21_lam*sqrt(d)*cos": lam * math.sqrt(dm) * cos,
                "ratio_eq21/exact": lam * math.sqrt(dm) * cos / exact})
R["eq20_21_check"] = chk

# ---------- Eq. 25 근처-상수 벡터: spike 채널이 노름을 지배할 때 RMSNorm 뒤 코사인 하한 ----------
# 같은 spike 방향 s(비율 고정)와 나머지 r_a, r_b (|r|/|s| = rho) 일 때 최악은 r_a = -r_b 이고 s에 직교: cos = (1-rho^2)/(1+rho^2)
R["eq25_cos_lower_bound"] = {str(rho): round((1 - rho ** 2) / (1 + rho ** 2), 4) for rho in (0.01, 0.03, 0.1, 0.3, 1.0)}
# 수치 확인: 최악 구성
worst = []
for rho in (0.1, 0.3):
    s = np.zeros(dm); s[[1415, 2533]] = [2500.0, 1500.0]
    r = np.zeros(dm); r[0] = rho * np.linalg.norm(s)
    a, b = s + r, s - r
    na, nb = np.sqrt(dm) * a / np.linalg.norm(a), np.sqrt(dm) * b / np.linalg.norm(b)
    worst.append({"rho": rho, "cos_numeric": float(na @ nb / (np.linalg.norm(na) * np.linalg.norm(nb)))})
R["eq25_worst_case_numeric"] = worst
# 정리 B.3: |h~_i| <= sqrt(d), 등호는 단일 채널 지배일 때
x = np.zeros(dm); x[7] = 1e4; x += 1e-3 * rng.standard_normal(dm)
xt = np.sqrt(dm) * x / np.linalg.norm(x)
R["thmB3"] = {"max_abs_normalized": float(np.abs(xt).max()), "sqrt_d": math.sqrt(dm)}

# ---------- Table 3–8 (표에서 옮김): (ppl, sink%, spike) ----------
T = {
    "T3.lr": {"7.5e-5": (11.8, 18.6, 1447), "1.5e-4": (10.7, 31.8, 2251), "3e-4(base)": (10.1, 46.0, 3818),
              "6e-4": (10.0, 51.5, 3773), "1.2e-3": (10.2, 39.2, 2723)},
    "T3.minlr": {"3e-5(base)": (10.1, 46.0, 3818), "3e-4": (10.7, 56.8, 2870)},
    "T3.wd": {"0.0": (10.4, 33.8, 12275), "0.1(base)": (10.1, 46.0, 3818)},
    "T3.beta2": {"0.9": (10.1, 49.0, 2832), "0.95(base)": (10.1, 46.0, 3818), "0.999": (10.7, 20.9, 1855)},
    "T3.tokens": {"100B(base)": (10.1, 46.0, 3818), "200B": (9.5, 63.3, 1848)},
    "T4.ffn": {"GeLU": (10.1, 69.3, 3369), "Linear": (12.5, 58.9, 688), "Attention": (10.8, 73.9, 637), "SwiGLU(base)": (10.1, 46.0, 3818)},
    "T5.norm": {"Sandwich": (9.8, 44.7, 520), "Sandwich(QK)": (10.0, 42.0, 92), "DynamicTanh": (10.0, 61.0, 153), "PreNorm(base)": (10.1, 46.0, 3818)},
    "T6.nheads": {"8": (10.4, 37.1, 1253), "16": (10.3, 41.7, 1936), "32(base)": (10.1, 46.0, 3818)},
    "T6.dhead": {"8": (11.3, 4.1, 291), "16": (10.8, 9.8, 315), "32": (10.5, 27.9, 829), "64": (10.3, 37.7, 2112), "128(base)": (10.1, 46.0, 3818)},
    "T6.fixedcap": {"8/512": (10.7, 11.0, 1205), "16/256": (10.4, 30.8, 1750), "32/128": (10.3, 41.1, 1916),
                    "64/64": (10.2, 44.1, 2523), "128/32(base)": (10.1, 46.0, 3818), "256/16": (10.1, 52.1, 3429)},
    "T7.conditional": {"channel": (10.0, 4.5, 202), "head": (10.1, 6.4, 186), "single": (10.2, 31.2, 316)},
    "T7.unconditional": {"channel": (10.1, 42.2, 1922), "head": (10.1, 41.3, 1884), "single": (10.2, 44.3, 1797)},
    "T7.static": {"positional": (10.1, 41.1, 1755), "token_embedding": (10.0, 31.1, 1966)},
    "T8.trainlen": {"1/256": (12.4, 42.1, 5411), "1/1024": (10.6, 46.3, 4442), "1/4096(base)": (10.1, 46.0, 3818)},
    "T8.minctx": {"1024/4096": (10.1, 13.0, 38470), "1024/5120": (10.1, 8.0, 42365), "2048/4096": (10.6, 1.2, 7193), "2048/6144": (10.0, 5.8, 30634)},
}
R["tables_3_8_transcribed"] = {k: {n: list(v) for n, v in d_.items()} for k, d_ in T.items()}

BASE = (10.1, 46.0, 3818)
rows = {}  # 중복 제거: 기준선 제외한 모든 고유 행
for grp, d_ in T.items():
    for n, v in d_.items():
        if "base" in n: continue
        rows[f"{grp}:{n}"] = v
ppl = np.array([v[0] for v in rows.values()]); sink = np.array([v[1] for v in rows.values()]); spk = np.array([v[2] for v in rows.values()])
R["n_unique_nonbaseline_rows"] = len(rows)
R["spearman_all_rows"] = {
    "ppl_vs_sink": list(map(float, spearmanr(ppl, sink))),
    "ppl_vs_spike": list(map(float, spearmanr(ppl, spk))),
    "sink_vs_spike": list(map(float, spearmanr(sink, spk))),
    "note": "행은 서로 다른 개입이라 독립 표본이 아니고 시드가 하나라 p는 참고용이다."}
# 표 3 한정(논문이 'sink ratio is a robust proxy for optimization health'라고 한 근거)
t3 = {k: v for k, v in rows.items() if k.startswith("T3")}
p3 = np.array([v[0] for v in t3.values()]); s3 = np.array([v[1] for v in t3.values()]); k3 = np.array([v[2] for v in t3.values()])
R["spearman_table3_only"] = {"n": len(t3), "ppl_vs_sink": list(map(float, spearmanr(p3, s3))), "ppl_vs_spike": list(map(float, spearmanr(p3, k3)))}
R["table3_ppl_sink_inversions"] = [k for k, v in t3.items() if (v[0] > BASE[0] and v[1] > BASE[1]) or (v[0] < BASE[0] and v[1] < BASE[1])]

# 주장 검사
claims = {}
# C-a: 'spike 억제 + sink 유지' 후보 (spike <= 15% of base, sink >= 90% of base)
claims["spike_suppressed_sink_kept"] = [k for k, v in rows.items() if v[2] <= 0.15 * BASE[2] and v[1] >= 0.9 * BASE[1]]
# C-b: 'sink 억제 + spike 유지/증가' (sink <= 30% of base, spike >= 0.9*base)
claims["sink_suppressed_spike_kept_or_up"] = [k for k, v in rows.items() if v[1] <= 0.3 * BASE[1] and v[2] >= 0.9 * BASE[2]]
# C-c: 둘 다 억제
claims["both_suppressed"] = [k for k, v in rows.items() if v[1] <= 0.3 * BASE[1] and v[2] <= 0.15 * BASE[2]]
# C-d: ppl 손실 (기준 대비 +0.3 초과)
claims["ppl_worse_by_more_than_0.3"] = [k for k, v in rows.items() if v[0] - BASE[0] > 0.3]
R["decoupling_claim_rows"] = claims
# 게이트 conditional의 'eliminates sinks entirely' 문구 vs 표
R["gated_conditional_vs_text"] = {"channel_sink%": 4.5, "head_sink%": 6.4, "base_sink%": 46.0,
                                  "relative_drop_channel": 1 - 4.5 / 46.0, "relative_drop_head": 1 - 6.4 / 46.0,
                                  "text": "per-channel and per-head gates both eliminate attention sinks entirely"}
# 단조성 검사
dh = [v for v in T["T6.dhead"].values()]
R["table6_dhead_monotone"] = {"sink": all(dh[i][1] < dh[i + 1][1] for i in range(4)), "spike": all(dh[i][2] < dh[i + 1][2] for i in range(4))}
fc = list(T["T6.fixedcap"].values())
R["table6_fixedcap_monotone_sink"] = all(fc[i][1] < fc[i + 1][1] for i in range(5))
R["table6_fixedcap_spike_monotone"] = all(fc[i][2] < fc[i + 1][2] for i in range(5))
R["table8_minctx_sink_monotone_in_min_len"] = "1024→13.0/8.0, 2048→1.2/5.8: 같은 min 안에서 max를 늘리면 1024는 하락, 2048은 상승(비단조)"
# 반올림 해상도: ppl은 소수 1자리
R["ppl_resolution"] = {"rounding_halfwidth": 0.05, "relative_at_10.1_pct": round(0.05 / 10.1 * 100, 2),
                       "rows_with_abs_ppl_diff<=0.1_vs_base": [k for k, v in rows.items() if abs(v[0] - BASE[0]) <= 0.1 + 1e-9]}
json.dump(R, open("recompute_tables_results.json", "w"), indent=1, ensure_ascii=False)

# 출력 요약
print("Table2 diffs(pp):", {m: v["diff_pp"] for m, v in R["table2"].items()})
print("Table1:", R["table1"])
print("budget:", R["training_budget"])
print("params T4:", R["params_table4_B"])
print("compute:", R["compute_estimate"])
print("sink T dependence:", R["sink_ratio_T_dependence"])
print("eq21:", chk)
print("eq25:", R["eq25_cos_lower_bound"], worst)
print("thmB3:", R["thmB3"])
print("spearman all:", R["spearman_all_rows"])
print("spearman T3:", R["spearman_table3_only"], R["table3_ppl_sink_inversions"])
print("decoupling:", claims)
print("monotone:", R["table6_dhead_monotone"], R["table6_fixedcap_monotone_sink"], R["table6_fixedcap_spike_monotone"])
print("ppl within 0.1:", R["ppl_resolution"]["rows_with_abs_ppl_diff<=0.1_vs_base"])
