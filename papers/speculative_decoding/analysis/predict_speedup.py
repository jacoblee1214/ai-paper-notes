"""real_alpha / real_decode / real_timing 결과를 합쳐 gamma별 기대 속도 향상을 계산한다 (모형 계산이며 실제 시간 측정이 아니다).
  S_paper(gamma) = E[tokens] / (gamma c + 1)                    (Thm 3.8; 목표 호출 1회 = 1토큰 시간, 병렬 검증 공짜)
  S_meas(gamma)  = E[tokens] / (gamma c + v(gamma + 1))         v(k) = 목표 모델 k토큰 호출 시간 / 1토큰 호출 시간 (실측)
E[tokens]: 식 (1)에 alpha를 넣거나(i.i.d. 가정), real_decode에서 실제로 관측한 반복당 토큰 수.
결과: speedup_results.json
"""
import json
from pathlib import Path

import numpy as np

import minimal_impl as M

D = Path(__file__).parent
alpha_r = json.load(open(D / "real_alpha_results.json"))
dec = json.load(open(D / "real_decode_results.json"))
tim = json.load(open(D / "real_timing_results.json"))
GAMMAS = list(range(1, 12))


def v_of(vk, k):
    ks = sorted(int(x) for x in vk); ys = [vk[str(x)] if str(x) in vk else vk[x] for x in ks]
    return float(np.interp(k, ks, ys))


out = {}
for tgt in ("70m", "160m"):
    c = tim[f"c_14m_over_{tgt}"]["c"]; vk = tim[f"c_14m_over_{tgt}"]["verify_cost_ratio_by_k"]
    for T in (1.0, 0):
        a_dec = dec[f"target={tgt},T={T},gamma=5"]["alpha_visited"]
        a_tf = alpha_r[f"target={tgt},T={T}"]["alpha"]
        rows = []
        for g in GAMMAS:
            e = M.expected_tokens(a_dec, g)
            rows.append(dict(gamma=g, e_tokens=e, s_paper=e / (g * c + 1), s_meas=e / (g * c + v_of(vk, g + 1)),
                             s_paper_c0=M.walltime_factor(a_dec, g, 0.0)))
        best_p = max(rows, key=lambda r: r["s_paper"]); best_m = max(rows, key=lambda r: r["s_meas"])
        out[f"target={tgt},T={T}"] = dict(c=c, alpha_decode=a_dec, alpha_teacher_forced=a_tf, rows=rows,
                                          best_gamma_paper=best_p["gamma"], best_s_paper=best_p["s_paper"],
                                          best_gamma_meas=best_m["gamma"], best_s_meas=best_m["s_meas"],
                                          optimal_gamma_formula_c_measured=M.optimal_gamma(a_dec, c)[0])
# 관측한 반복당 토큰 수로 만든 속도(70M, T=1, gamma 스윕)
c = tim["c_14m_over_70m"]["c"]; vk = tim["c_14m_over_70m"]["verify_cost_ratio_by_k"]
sweep = []
for g in (1, 2, 3, 5, 8):
    r = dec[f"target=70m,T=1.0,gamma={g}"]
    ci = r["tokens_per_iter_ci95"]
    f = lambda e: e / (g * c + v_of(vk, g + 1))
    sweep.append(dict(gamma=g, obs_tokens=r["tokens_per_iter"], obs_ci95=ci, eq1=r["eq1_prediction"],
                      s_meas_obs=f(r["tokens_per_iter"]), s_meas_obs_ci95=[f(ci[0]), f(ci[1])], s_meas_eq1=f(r["eq1_prediction"]),
                      s_paper_obs=r["tokens_per_iter"] / (g * c + 1)))
out["sweep_70m_T1_observed"] = sweep
# 식 (1)에 넣는 alpha에 따른 예측: 첫 위치 beta(모든 반복이 방문, 편향 없음) 대 방문한 모든 위치의 평균 beta
out["decode_summary"] = {k: dict(tokens=v["tokens_per_iter"], ci95=v["tokens_per_iter_ci95"], alpha_first=v["alpha_first_position"],
                                 alpha_visited=v["alpha_visited"], eq1_alpha_first=M.expected_tokens(v["alpha_first_position"], v["gamma"]),
                                 eq1_alpha_visited=v["eq1_prediction"], tv=v["tv_n_hist_vs_geometric"], distinct=v["distinct_token_frac"])
                         for k, v in dec.items() if k.startswith("target")}
(D / "speedup_results.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if k.startswith("target"):
        print(k, "c", round(v["c"], 3), "alpha", round(v["alpha_decode"], 3), "best paper", v["best_gamma_paper"], round(v["best_s_paper"], 2),
              "best meas", v["best_gamma_meas"], round(v["best_s_meas"], 2))
for s in sweep:
    print(s["gamma"], round(s["obs_tokens"], 3), round(s["eq1"], 3), "S_meas obs", round(s["s_meas_obs"], 3), [round(x, 3) for x in s["s_meas_obs_ci95"]], "S_paper obs", round(s["s_paper_obs"], 3))
