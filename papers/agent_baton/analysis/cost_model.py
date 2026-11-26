"""비용 식 해체와 민감도.
C_AB = C_src(0..h) + C_recv(h..끝) [+ C_detector], 절감률 = 1 - C_AB / C_capable-only. Qwen은 0원(논문 규약).
(1) 절감률 = 1 - s(x): s는 capable 비용 중 첫 x 진행분의 몫. Table 3의 Qwen 행에서 s(x)=x^alpha를 역산.
(2) Qwen에 값을 매기면: Luna 행과 Qwen 행의 비용 차를 이용해 Luna의 인계 후 비용 L을 추정, Qwen = c*L.
(3) 탐지기 비용: 중앙값 338 ms, p95 1.11 s(논문 §4.2), H100 시간당 단가와 인계 전 턴 수 가정.
(4) E->C가 싸게 안 되는 이유가 구조적인가: s(x)=x^alpha로 예측한 절감 대 Table 1 관측.
실행: python3 cost_model.py -> cost_model_results.json"""
import json, math
import numpy as np
from paper_tables import *

out = {}
# (1) alpha 역산
alpha = []
for key, d in AB.items():
    if key[1] != "Qwen": continue
    for bi, b in enumerate(("DS", "TB")):
        x = HANDOFF_PCT[key[0]][bi] / 100
        red = 1 - d[b][1] / BASE[key[0]][b][1]
        share = 1 - red
        alpha.append({"pair": "->".join(key), "bench": b, "x": x, "share_spent": round(share, 3), "alpha": round(math.log(share) / math.log(x), 2)})
out["alpha_from_qwen_rows"] = alpha
out["alpha_range"] = (min(a["alpha"] for a in alpha), max(a["alpha"] for a in alpha))
out["alpha_median"] = round(float(np.median([a["alpha"] for a in alpha])), 2)

# (4) E->C 예측 절감 vs 관측 (Table 1, logfile, 30%, Qwen->capable)
ec = []
for (src, rcv), vals in T1.items():
    if src != "Qwen": continue
    for i in (1, 5):  # TB30 log, DS30 log
        b, pct, _ = T1_COLS[i]
        a, c = vals[i]
        obs = 1 - c / BASE[rcv][b][1]
        pred = {al: round(0.30 ** al, 3) for al in (1.0, 1.3, 1.5)}
        ec.append({"pair": f"Qwen->{rcv}", "bench": b, "obs_saving": round(obs, 3), "pred_saving_if_alpha": pred})
out["E_to_C_30pct_saving_obs_vs_pred"] = ec
# C->E 30% 관측: GLM DS 0.56 vs 3.16
ce = []
for (src, rcv), vals in T1.items():
    if rcv != "Qwen": continue
    for i in (0, 4):
        b, pct, _ = T1_COLS[i]
        ce.append({"pair": f"{src}->Qwen", "bench": b, "obs_saving": round(1 - vals[i][1] / BASE[src][b][1], 3), "pred_saving_alpha1.3": round(1 - 0.3 ** 1.3, 3)})
out["C_to_E_30pct_saving_obs_vs_pred"] = ce

# (2) Qwen에 값을 매겼을 때
L = {}
for s in ("Sol", "Opus", "GLM", "Kimi", "Gem"):
    for b in ("DS", "TB"):
        L[(s, b)] = AB[(s, "Luna")][b][1] - AB[(s, "Qwen")][b][1]
out["Luna_post_handoff_cost_L"] = {f"{k[0]}_{k[1]}": round(v, 2) for k, v in L.items()}
sens = {}
for c in (0, 0.1, 0.25, 0.5, 1.0):
    reds = []
    for s in ("Sol", "Opus", "GLM", "Kimi", "Gem"):
        for b in ("DS", "TB"):
            cost = AB[(s, "Qwen")][b][1] + c * L[(s, b)]
            reds.append(100 * (1 - cost / BASE[s][b][1]))
    luna_rows = [100 * (1 - AB[(s, "Luna")][b][1] / BASE[s][b][1]) for s in ("Sol", "Opus", "GLM", "Kimi", "Gem") for b in ("DS", "TB")]
    luna_src = [100 * (1 - AB[("Luna", "Qwen")][b][1] / BASE["Luna"][b][1]) for b in ("DS", "TB")]
    sens[f"c={c}"] = {"avg_qwen_rows_10": round(float(np.mean(reds)), 1), "pooled_22_luna_src_unchanged": round(float(np.mean(reds + luna_rows + luna_src)), 1)}
out["qwen_price_sensitivity"] = sens

# (3) 탐지기 비용
det = {}
for T in (30, 60, 120):
    for gpu in (2.0, 4.0):
        for lat, name in ((0.338, "median"), (1.11, "p95")):
            xs = [0.27, 0.58]  # DS, TB 대표 진행률(평균 인계 진행 26.6–41.1 / 55.6–68.9의 낮은 쪽)
            usd = [x * T * lat * gpu / 3600 for x in xs]
            det[f"T{T}_gpu{gpu}_{name}"] = [round(u, 4) for u in usd]
out["detector_usd_per_task(DS,TB)"] = det
# Luna->Qwen TB 비용 0.04 대비
out["detector_vs_LunaQwen_TB_cost_0.04"] = {k: round(v[1] / 0.04, 2) for k, v in det.items() if k.startswith("T60")}
worst = det["T120_gpu4.0_p95"]
adj = []
for key, d in AB.items():
    for bi, b in enumerate(("DS", "TB")):
        extra = worst[bi] if key[0] != "Luna" else worst[bi]
        adj.append(100 * (1 - (d[b][1] + extra) / BASE[key[0]][b][1]))
out["avg_reduction_with_worst_detector_cost"] = round(float(np.mean(adj)), 1)
base_adj = []
for key, d in AB.items():
    for bi, b in enumerate(("DS", "TB")):
        extra = det["T60_gpu2.0_median"][bi]
        base_adj.append(100 * (1 - (d[b][1] + extra) / BASE[key[0]][b][1]))
out["avg_reduction_with_mid_detector_cost"] = round(float(np.mean(base_adj)), 1)

# 평균의 구성: 비용 가중, 벤치마크별, 상한(최저값 TB Gem->Luna ...)
out["reduction_by_row"] = {"->".join(k) + "_" + b: round(100 * (1 - d[b][1] / BASE[k[0]][b][1]), 1) for k, d in AB.items() for b in ("DS", "TB")}

# 비용의 정의 문제: 시간당 비용이 아닌 청구액이므로 캐시 단가는 모델마다 다름 (논문은 공급자 공식 토큰 단가×보고된 토큰 수라고만 적음)
out["note"] = "입력·출력·캐시 토큰 분해와 단가는 논문에 없다(§2 마지막 문장). 절감률은 토큰 분해 없이 Table 3 비용 열의 비율로만 재계산했다."

# (5) 기준선을 "max 노력 단독"이 아니라 Table 6의 모든 단일 구성 중 정확도가 같거나 높은 가장 싼 것으로 바꾸면 (DeepSWE)
singles = [(r[0], r[1], r[2][0], r[2][1]) for r in T6]
rows5 = []
for r in T6:
    for tag, idx in (("->Qwen", 3), ("->Luna", 4)):
        if r[idx] is None: continue
        acc, cost = r[idx]
        for tol, name in ((0.0, "acc>=AB"), (5.0, "acc>=AB-5pp")):
            cand = [x for x in singles if x[2] >= acc - tol]
            best = min(cand, key=lambda x: x[3]) if cand else None
            rows5.append({"ab": f"{r[0]} {r[1]} {tag}", "ab_acc": acc, "ab_cost": cost, "rule": name,
                          "cheapest_single": None if best is None else f"{best[0]} {best[1]} {best[2]}/{best[3]}",
                          "saving_vs_cheapest_single_pct": None if best is None else round(100 * (1 - cost / best[3]), 1) if best[3] > 0 else None})
out["saving_vs_cheapest_adequate_single_DS"] = rows5
for rule in ("acc>=AB", "acc>=AB-5pp"):
    v = [x["saving_vs_cheapest_single_pct"] for x in rows5 if x["rule"] == rule and x["saving_vs_cheapest_single_pct"] is not None]
    out[f"saving_vs_cheapest_single_summary_{rule}"] = {"n": len(v), "mean": round(float(np.mean(v)), 1), "median": round(float(np.median(v)), 1), "negative": sum(1 for x in v if x < 0)}

json.dump(out, open("cost_model_results.json", "w"), ensure_ascii=False, indent=1)
for k in ("alpha_range", "alpha_median", "E_to_C_30pct_saving_obs_vs_pred", "C_to_E_30pct_saving_obs_vs_pred", "Luna_post_handoff_cost_L", "qwen_price_sensitivity",
          "saving_vs_cheapest_single_summary_acc>=AB", "saving_vs_cheapest_single_summary_acc>=AB-5pp", "detector_usd_per_task(DS,TB)", "detector_vs_LunaQwen_TB_cost_0.04", "avg_reduction_with_worst_detector_cost", "avg_reduction_with_mid_detector_cost"):
    print(k, json.dumps(out[k], ensure_ascii=False))
