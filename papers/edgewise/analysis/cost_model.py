"""비용 식과 민감도, 보상(식 1)·이점(식 2) 검산, 학습 비용 손익분기.
(a) 기준선 노력 수준 민감도(그림 1 판독값), (b) 단가·캐시 가정 민감도(토큰 수가 없어 구조적 범위), (c) 위임 호출당 비용,
(d) 학습 비용 손익분기, (e) 식 1의 벌점 크기, (f) 식 2의 스케일 인자.
실행: python3 cost_model.py  ->  cost_model_results.json (analysis/ 안에서 실행)"""
import json, math
import numpy as np
from paper_tables import *

out = {}

# (a) 기준선 선택: GPT-5.5 노력 수준(low, medium, high, xhigh)과 Self-Delegation(같은 노력 수준의 클라우드)
a = {}
for b, d in FIG1_READ.items():
    eff = ("low", "medium", "high", "xhigh")
    gpt = {e: d[f"GPT {e}"] for e in eff}; slf = {e: d[f"Self {e}"] for e in eff}
    best_gpt = max(gpt.items(), key=lambda kv: kv[1][1])       # 정확도 최고인 GPT 노력
    rows = {}
    for e in eff:
        sc, sa = slf[e]
        same = 100 * (1 - sc / gpt[e][0])
        vs_x = 100 * (1 - sc / gpt["xhigh"][0])
        vs_best = 100 * (1 - sc / best_gpt[1][0])
        # 같은 정확도(자기 정확도 -1%p 이상)를 내는 가장 싼 GPT 노력
        ok = [(g[0], k) for k, g in gpt.items() if g[1] >= sa - 1.0]
        iso = min(ok) if ok else None
        rows[e] = {"self": (sc, sa), "same_effort_saving": round(same, 1), "vs_xhigh_saving": round(vs_x, 1),
                   "vs_best_acc_effort": (best_gpt[0], round(vs_best, 1)),
                   "cheapest_gpt_effort_within_1pp": None if iso is None else (iso[1], iso[0], round(100 * (1 - sc / iso[0]), 1))}
    a[b] = {"best_gpt_effort": best_gpt, "rows": rows,
            "xhigh_dominated_by": [e for e in ("low", "medium", "high") if gpt[e][1] >= gpt["xhigh"][1] and gpt[e][0] < gpt["xhigh"][0]]}
out["baseline_effort"] = a
# 표 1의 헤드라인(Self xhigh, GPT xhigh)과 비교
out["headline_vs_best_effort"] = {b: {"headline_saving": round(100 * (1 - FIG1_READ[b]["Self xhigh"][0] / FIG1_READ[b]["GPT xhigh"][0]), 1),
                                      "vs_GPT_high_saving": round(100 * (1 - FIG1_READ[b]["Self xhigh"][0] / FIG1_READ[b]["GPT high"][0]), 1),
                                      "acc_gain_vs_GPT_high_pp": round(FIG1_READ[b]["Self xhigh"][1] - FIG1_READ[b]["GPT high"][1], 1)} for b in FIG1_READ}

# (b) 단가·캐시 가정의 구조적 민감도
# 바닐라: 입력 I 토큰, 출력 O 토큰, 단가 1과 rho, 입력 캐시 적중 h, 캐시 단가 c. 위임: 입력 r_in*I, 출력 r_out*O.
def ratio(r_in, r_out, kappa, rho, h=0.0, c=0.1):
    p_in = 1 - h * (1 - c)
    van = p_in + kappa * rho
    return (r_in * p_in + r_out * kappa * rho) / van
b = []
base = dict(kappa=0.05, rho=5, h=0.0)
for target in (0.65, 0.47):          # TB와 SWE의 Pass@3 비용 비율(자기 위임 GPT)
    for delta in (-0.3, 0.0, 0.3):   # r_out - r_in
        # r_in을 풀어 기준 가정에서 목표 비율을 맞춘다
        lo, hi = 0.01, 1.5
        for _ in range(60):
            mid = (lo + hi) / 2
            if ratio(mid, mid + delta, **base) > target: hi = mid
            else: lo = mid
        r_in = (lo + hi) / 2; r_out = r_in + delta
        if r_in < 0 or r_out < 0: continue
        vals = {}
        for kappa in (0.02, 0.05, 0.10):
            for rho in (2, 5, 8):
                for h in (0.0, 0.9):
                    vals[f"k{kappa}_rho{rho}_h{h}"] = round(100 * (1 - ratio(r_in, r_out, kappa, rho, h)), 1)
        b.append({"target_ratio": target, "r_out_minus_r_in": delta, "r_in": round(r_in, 3), "r_out": round(r_out, 3),
                  "saving_min": min(vals.values()), "saving_max": max(vals.values()), "saving_at_base": round(100 * (1 - ratio(r_in, r_out, **base)), 1)})
out["price_sensitivity"] = b
out["price_sensitivity_note"] = "토큰 수(입력·출력·캐시)가 논문에 없어 위임 쪽 입력·출력 토큰 비율(r_in, r_out)을 기준 가정에서 목표 비율에 맞춰 풀고 가격·캐시 가정만 바꾼 구조적 범위. 절감률은 항상 1-r_in과 1-r_out 사이"

# (c) 위임 호출당 비용
c = {"with_token_adv": round(56.96 / (89 * 5.80), 4), "without_token_adv": round(55.05 / (89 * 8.97), 4)}
c["ratio"] = round(c["with_token_adv"] / c["without_token_adv"], 2)
c["vanilla_cost_per_task"] = round(90.31 / 89, 3); c["self_cost_per_task"] = round(56.96 / 89, 3)
out["per_delegation"] = c
# 무작위 위임: 비용 비율 / 위임 비율(40% 서술) = 위임 한 단계가 평균보다 몇 배 비싼가 (길이 동일 가정)
out["random_cost_over_share"] = {"TB_GPT_P3": round(209.13 / 279.73 / 0.40, 2), "SWE_GPT_P3": round(455.54 / 598.08 / 0.40, 2)}
out["self_cost_over_share_if_40pct"] = {"TB_GPT_P3": round(181.88 / 279.73 / 0.40, 2), "SWE_GPT_P3": round(283.02 / 598.08 / 0.40, 2)}

# (d) 학습 비용 손익분기 (논문은 학습 비용을 보고하지 않음)
sav = {"TB_GPT": (90.31 - 56.96) / 89, "SWE_GPT": (200.50 - 97.80) / 110, "TB_Opus": (64.78 - 42.98) / 89, "SWE_Opus": (199.06 - 74.28) / 110}
out["saving_per_task_usd"] = {k: round(v, 3) for k, v in sav.items()}
rollouts = TRAIN["steps"] * TRAIN["batch"] * TRAIN["rollouts"]
out["rl_rollouts"] = rollouts
be = {}
for per_roll in (0.1, 0.3, 0.6):
    for gpu_h_per_gpu in (24, 48, 96):
        for usd_gpu_h in (1.5, 3.0):
            C = rollouts * per_roll + TRAIN["gpus"] * gpu_h_per_gpu * usd_gpu_h + TRAIN["sft_examples"] * 1.0
            be[f"cloud${per_roll}/rollout_{gpu_h_per_gpu}h_${usd_gpu_h}/gpu-h"] = {"C_train_usd": round(C), "breakeven_tasks_TB_GPT": round(C / sav["TB_GPT"]), "breakeven_tasks_SWE_GPT": round(C / sav["SWE_GPT"])}
out["breakeven"] = be

# (e) 식 1: R = V + 1[V=1]/(1+lam*C)
e = {}
for lam in (0.02, 0.05, 0.1, 0.2, 0.5):
    row = {}
    for C in (3, 5.8, 9, 15):
        g = 1 / (1 + lam * C)
        row[str(C)] = {"bonus": round(g, 3), "dR_dC": round(-lam / (1 + lam * C) ** 2, 4), "min_rel_gain_in_P_to_justify_one_call": round(lam / ((1 + lam * C) ** 2 * (1 + g)), 4)}
    e[str(lam)] = row
out["eq1_penalty"] = e
out["eq1_range"] = {"fail": 0.0, "success_C0": 2.0, "success_C5.8_lam0.1": round(1 + 1 / (1 + 0.1 * 5.8), 3), "success_C8.97_lam0.1": round(1 + 1 / (1 + 0.1 * 8.97), 3), "success_Cinf": 1.0,
                    "gap_success_vs_fail_min": 1.0, "gap_C5.8_vs_C8.97": round(1 / 1.58 - 1 / 1.897, 3)}

# (f) 식 2: A_t = A_traj + beta*(N/N_route)*A_del
f = {}
for N_ in (2000, 8000):
    for Nr in (10, 30, 100):
        f[f"N{N_}_Nroute{Nr}"] = {"scale_N_over_Nroute": round(N_ / Nr, 1), "per_token_extra_adv_beta1_Adel1": round(N_ / Nr, 1), "total_mass_beta1_Adel1": N_}
out["eq2_scale"] = f
out["eq2_total_mass_note"] = "위임 토큰 전체에 더해지는 이점의 합 = beta*N*(토큰 가중 평균 A_del) -> N_route와 무관. beta 값은 논문에 없음"

json.dump(out, open("cost_model_results.json", "w"), ensure_ascii=False, indent=1, default=str)
for k in ("baseline_effort", "headline_vs_best_effort", "price_sensitivity", "per_delegation", "random_cost_over_share", "self_cost_over_share_if_40pct", "saving_per_task_usd", "rl_rollouts", "eq1_range"):
    print(k, json.dumps(out[k], ensure_ascii=False, default=str))
print("eq1 lam=0.1", json.dumps(out["eq1_penalty"]["0.1"]))
for k in list(be)[:6]: print(k, be[k])
print(min(v["C_train_usd"] for v in be.values()), max(v["C_train_usd"] for v in be.values()))
print(min(v["breakeven_tasks_TB_GPT"] for v in be.values()), max(v["breakeven_tasks_TB_GPT"] for v in be.values()), min(v["breakeven_tasks_SWE_GPT"] for v in be.values()), max(v["breakeven_tasks_SWE_GPT"] for v in be.values()))
