"""표 3–8 재계산과 절제 차이의 크기 판정 (numpy, 수 초).

논문은 모든 표가 학습 1회(시드·분산 보고 없음)다. 따라서 차이가 우연보다 큰지를 직접 검정할 수 없고, 아래 세 가지 척도로 가늠한다.
 (a) 표 간 일관성: 같은 설정이 여러 표에 나오는 칸이 일치하는지. 평가 설정만 다른 칸(표 3 vs 표 7의 7B Linear)의 차이.
 (b) 그림 5 맞춤선의 잔차 산포(ln 단위): 크기·체크포인트가 다른 점이 거듭제곱선에서 흩어지는 정도. 시드 분산이 아니라 거친 상한 척도다.
 (c) 부호 일관성: 5개 모델 크기에서 같은 방향인가 (표 7, 부호 검정).
입력: paper_tables.py, scaling_results.json. 출력: ablation_results.json
"""
import json
from math import comb, log, sqrt
from pathlib import Path

import numpy as np

import paper_tables as T

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "scaling_results.json").read_text())["panels"]
SCATTER = {  # 그림 5 Transfusion 맞춤선 잔차 SD(ln 단위)
    "C4_PPL": S["c4_ppl"]["resid_sd_ln"]["Transfusion"], "Wiki_PPL": S["wiki_ppl"]["resid_sd_ln"]["Transfusion"],
    "Llama_Acc": S["llama_acc"]["resid_sd_ln"]["Transfusion"], "CIDEr": S["coco_cider"]["resid_sd_ln"]["Transfusion"],
    "FID": S["coco_fid"]["resid_sd_ln"]["Transfusion"], "CLIP": S["coco_clip"]["resid_sd_ln"]["Transfusion"],
}
out = {"scatter_sd_ln": SCATTER}


def lnd(a, b):
    return log(a / b)


def row_diff(a, b):
    """b 대비 a의 ln 비 (양수 = a가 큼)를 지표별로. 방향(좋아짐)도 붙인다."""
    r = {}
    for i, c in enumerate(T.COLS):
        d = lnd(a[i], b[i])
        better = None if d == 0 else ((d < 0) if c in T.LOWER_BETTER else (d > 0))
        r[c] = {"a": a[i], "b": b[i], "ln_ratio": round(d, 4), "pct": round(100 * (a[i] / b[i] - 1), 1), "units_of_scatter": round(abs(d) / SCATTER[c], 2), "a_better": better}
    return r


# (a) 표 3 vs 표 7의 같은 모델(7B, Linear): 평가 프로토콜만 다름(표 3: FID·CLIP 30k, CFG 5. 표 7: 5k, CFG 3)
t3 = T.TABLE3["Transfusion"]; t7 = T.TABLE7["7B"]["Linear"]
out["protocol_shift_table3_vs_table7_same_model"] = {c: {"table3": t3[i], "table7": t7[i], "pct": round(100 * (t7[i] / t3[i] - 1), 1)} for i, c in enumerate(T.COLS)}
out["table7_linear_7B_is_table3_transfusion"] = {"text_and_CIDEr_columns_equal": all(abs(t3[i] - t7[i]) <= 0.05 for i in (0, 1, 2, 3))}

# 표 간 일관성
cons = []
t5 = T.TABLE5; t6 = T.TABLE6; t7r = T.TABLE7["0.76B"]; t8 = T.TABLE8
cons.append(("T5 Linear-Bidir == T6 Linear-2 == T7 0.76B Linear", t5[("Linear", "Bidirectional")] == t6[("Linear", 2)] == t7r["Linear"]))
cons.append(("T5 U-Net-Bidir == T6 U-Net-2 == T7 0.76B U-Net == T8 0.76B 노이즈제한X", t5[("U-Net", "Bidirectional")] == t6[("U-Net", 2)] == t7r["U-Net"] == t8[("0.76B", False)]))
cons.append(("T7 7B U-Net == T8 7B 노이즈제한X", T.TABLE7["7B"]["U-Net"] == t8[("7B", False)]))
cons.append(("T4 Transfusion(+Diffusion) 텍스트 3칸 == T5 Linear 텍스트 3칸", T.TABLE4["Transfusion(+Diffusion)"] == t5[("Linear", "Bidirectional")][:3]))
cons.append(("T6 'None 1x1' 행: 다른 표에 없음(단독)", None))
out["cross_table_consistency"] = [{"check": a, "ok": b} for a, b in cons]

# 표 4 괄호 차이 재계산
t4 = T.TABLE4
rows = list(t4.items())
chk = []
for n_, prev, ref in ((rows[1][0], rows[1][1], rows[0][1]), (rows[2][0], rows[2][1], rows[0][1]), (rows[3][0], rows[3][1], rows[2][1])):
    chk.append({"row": n_, "diff_vs_ref": [round(prev[i] - ref[i], 1) for i in range(3)]})
out["table4_parenthesis_diffs_recomputed"] = chk
out["table4_text_cost_C4_ppl_pct"] = {"Transfusion_vs_Llama2": round(100 * (10.4 / 10.1 - 1), 1), "Chameleon_total_vs_Llama2": round(100 * (11.8 / 10.1 - 1), 1)}

# 표 5: 양방향 vs 인과
out["table5_bidirectional_vs_causal"] = {enc: row_diff(t5[(enc, "Bidirectional")], t5[(enc, "Causal")]) for enc in ("Linear", "U-Net")}
# 표 6: 패치 크기(U-Net) 기준 2x2 대비
out["table6_unet_patch_vs_2x2"] = {f"{p}x{p}": row_diff(T.TABLE6[("U-Net", p)], T.TABLE6[("U-Net", 2)]) for p in (4, 8)}
out["table6_linear_patch_vs_2x2"] = {f"{p}x{p}": row_diff(T.TABLE6[("Linear", p)], T.TABLE6[("Linear", 2)]) for p in (4, 8)}
out["table6_none1x1_vs_linear2x2"] = row_diff(T.TABLE6[("None", 1)], T.TABLE6[("Linear", 2)])
# 표 7: U-Net vs Linear, 크기별 + 부호 검정
t7res = {}
signs = {c: 0 for c in T.COLS}
for k, v in T.TABLE7.items():
    d = row_diff(v["U-Net"], v["Linear"])
    t7res[k] = d
    for c in T.COLS:
        signs[c] += int(d[c]["a_better"] is True)
out["table7_unet_vs_linear_by_size"] = t7res
out["table7_unet_better_count_of_5_sizes"] = signs
out["table7_sign_test_two_sided_p_if_5_of_5"] = 2 * comb(5, 5) / 2 ** 5
# 표 7 주장: "U-Net 작은 모델이 7B Linear보다 낫다"
lin7 = T.TABLE7["7B"]["Linear"]
claims = {}
for k in ("0.16B", "0.37B", "0.76B", "1.4B"):
    u = T.TABLE7[k]["U-Net"]
    claims[k] = {"FID_unet": u[4], "FID_7B_linear": lin7[4], "FID_diff": round(u[4] - lin7[4], 1), "FID_diff_pct": round(100 * (u[4] / lin7[4] - 1), 1),
                 "CIDEr_unet": u[3], "CIDEr_7B_linear": lin7[3], "CIDEr_diff_pct": round(100 * (u[3] / lin7[3] - 1), 1)}
out["table7_claim_small_unet_beats_7B_linear"] = {"rows": claims, "FID_protocol_shift_pct_(T3_vs_T7)": round(100 * (18.6 / 16.8 - 1), 1),
                                               "FID_scatter_pct": round(100 * (np.exp(SCATTER["FID"]) - 1), 1), "CIDEr_scatter_pct": round(100 * (np.exp(SCATTER["CIDEr"]) - 1), 1)}
# 표 8: 노이즈 제한
out["table8_noise_limit"] = {k: row_diff(t8[(k, True)], t8[(k, False)]) for k in ("0.76B", "7B")}

# 표 3: Transfusion vs Chameleon (같은 7B)
out["table3_transfusion_vs_chameleon"] = row_diff(T.TABLE3["Transfusion"], T.TABLE3["Chameleon"])

# 평가 표본: Llama 2 평가 묶음의 0-shot 7과제 평균 정확도의 이항 표준오차 (과제 크기는 통상 공개 크기, 논문에 적혀 있지 않음 → 미확인)
sizes = {"HellaSwag": 10042, "PIQA": 1838, "SIQA": 1954, "WinoGrande": 1267, "ARC-e": 2376, "ARC-c": 1172, "BoolQ": 3270}
n = np.array(list(sizes.values()))
res = {}
for acc in (0.52, 0.615):
    se_task = np.sqrt(acc * (1 - acc) / n)
    se_avg = float(np.sqrt((se_task ** 2).sum()) / len(n))
    res[str(acc)] = {"SE_avg_pp": round(100 * se_avg, 2), "SE_diff_independent_pp": round(100 * se_avg * sqrt(2), 2)}
out["llama_suite_average_accuracy_binomial_SE"] = {"task_sizes_(unverified_common_sizes)": sizes, "total_items": int(n.sum()), **res,
    "note": "같은 문항에 두 모델을 돌리므로 짝지은 차이의 SE는 독립 가정보다 작다(상관 미상). 시드·학습 변동은 포함하지 않는다."}
# 표 5·6·7의 Llama Acc 차이가 SE 안인가
out["llama_acc_diffs_vs_SE"] = {
    "table6_linear8x8_vs_2x2": round(47.7 - 51.7, 1), "table6_linear4x4_vs_2x2": round(49.8 - 51.7, 1), "table7_unet_vs_linear_max_abs_diff_pp": max(abs(T.TABLE7[k]["U-Net"][2] - T.TABLE7[k]["Linear"][2]) for k in T.TABLE7),
    "table5_bidir_vs_causal_linear_pp": round(51.7 - 51.4, 1),
}
(HERE / "ablation_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: out[k] for k in ("protocol_shift_table3_vs_table7_same_model", "cross_table_consistency", "table7_unet_better_count_of_5_sizes", "table7_claim_small_unet_beats_7B_linear", "llama_suite_average_accuracy_binomial_SE", "table4_parenthesis_diffs_recomputed")}, indent=1, ensure_ascii=False))
