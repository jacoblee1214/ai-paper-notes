"""논문 표(1, 2, 3, 5, 6, 7, 9–14)의 값을 옮겨 적어 산술 검사와 "시드 변동 대용 척도" 비교를 한다. numpy만 사용.

논문은 모든 표가 실행 1회이고 시드 반복·구간·표준편차가 없다(공식 설정 파일의 seed가 42 하나, `04` D8).
따라서 효과가 우연한 흔들림보다 큰지 가릴 직접 근거가 없다. 대용으로, 논문이 스스로 "영향 없음"이라고 한
변형들(FFN 활성함수, 헤드 수, 배치 크기, 위치 임베딩)의 Sink 값 흩어짐을 노이즈의 상한 추정으로 쓴다.
이 흩어짐에는 실제 효과가 섞여 있을 수 있어 과대추정(보수적)이다.
실행: python3 -I analysis/tables_recompute.py
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = {}

# ---- 논문 값 (표 번호는 arXiv v2 기준) ----
t2_wd = {0.0: (15.20, 3.72), 0.001: (15.39, 3.72), 0.01: (15.23, 3.72), 0.1: (18.18, 3.73),
         0.5: (41.08, 3.80), 1.0: (37.71, 3.90), 2.0: (6.13, 4.23), 5.0: (0.01, 5.24)}
t9_lr = [(8e-4, 10, 23.44, 3.79), (8e-4, 20, 32.23, 3.70), (4e-4, 20, 18.18, 3.73), (2e-4, 20, 11.21, 3.78),
         (2e-4, 40, 16.81, 3.68), (1e-4, 20, 2.90, 3.92), (1e-4, 80, 6.29, 3.67)]
t3_pe = {"NoPE": (20.35, 3.81), "Absolute": (32.73, 3.74), "Learnable": (33.13, 3.79), "Relative": (35.53, 5.45),
         "ALiBi": (20.78, 3.71), "Rotary": (18.18, 3.73)}
t10_batch = {"0.25M": 16.45, "0.5M": 17.19, "1M": 18.18, "2M": 16.19}
t11_ffn = {"ReLU": 16.90, "GeLU": 14.76, "Swish": 17.89, "ReGLU": 13.88, "GeGLU": 17.86, "SwiGLU": 18.18}
t12_heads = {"H=8": 18.18, "H=4": 10.68, "H=2": 12.95, "H=1": 19.50, "addition": 21.76}
t14_sigmoid = [(4e-4, 0.0, 0.64, 3.77), (4e-4, 0.1, 0.44, 3.70), (4e-4, 0.5, 0.18, 3.76), (4e-4, 1.0, 0.30, 4.06),
               (1e-3, 0.1, 0.81, 3.68), (1e-4, 0.1, 0.36, 4.08)]
t14_scale = {2.0: (29.10, 3.72), 1.0: (18.18, 3.73), 0.2: (9.41, 3.72), 0.1: (3.59, 3.76), 0.05: (4.53, 3.78)}
t5 = {"v*": ["0", "v'", "5v'", "20v'", "v''", "5v''", "20v''"],
      "sink_star": [73.34, 70.03, 44.43, 1.51, 69.74, 27.99, 0.00],
      "sink_1": [0.00, 0.06, 3.71, 25.88, 2.15, 5.93, 11.21]}
t13 = {"da": [1, 2, 4, 8, 16, 32, 64], "sink_star": [32.18, 30.88, 30.94, 31.39, 23.30, 51.23, 69.19]}
t1_chat = {"Mistral-7B": (97.49, 88.34), "LLaMA2-7B": (92.47, 92.88), "LLaMA2-13B": (91.69, 90.94), "LLaMA3-8B": (99.02, 98.85)}
t1_input = {"GPT2-XL": (77.00, 70.29, 62.28), "Mistral-7B": (97.49, 75.21, 0.00), "LLaMA2-7B": (92.47, 90.13, 0.00),
            "LLaMA3-8B": (99.02, 91.23, 0.00)}
t7_replace = {"no": 62.28, 5: 0.20, 10: 2.36, 15: 7.73, 20: 10.63, 25: 10.97}
t6_main = {"softmax": (18.18, 3.73), "sigmoid_nonorm": (0.44, 3.70), "sigmoid_norm": (30.24, 3.74),
           "elu_nonorm": (0.80, 3.69), "elu_kernel_norm": (53.65, 4.19), "identity_nonorm": (0.00, 3.99),
           "mlp_norm": (0.19, 3.85), "mlp_nonorm": (0.74, 3.91)}
text_1b = {"softmax": (45.11, 3.07), "sigmoid_nonorm": (2.46, 3.10)}

# ---- 1. 노이즈 대용 척도 ----
groups = {"FFN 활성함수(표 11)": list(t11_ffn.values()), "헤드 수·결합(표 12)": list(t12_heads.values()),
          "배치 크기(표 10)": list(t10_batch.values()),
          "위치 임베딩, Relative 제외(표 3)": [v[0] for k, v in t3_pe.items() if k != "Relative"]}
g_out, pooled_ss, pooled_df = {}, 0.0, 0
for k, v in groups.items():
    v = np.array(v)
    g_out[k] = {"n": len(v), "values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "range": float(v.max() - v.min())}
    pooled_ss += float(((v - v.mean()) ** 2).sum())
    pooled_df += len(v) - 1
sd_pool = math.sqrt(pooled_ss / pooled_df)  # 네 묶음 모두(상한)
ss_lo = sum(float(((np.array(groups[k]) - np.mean(groups[k])) ** 2).sum()) for k in ["FFN 활성함수(표 11)", "배치 크기(표 10)"])
df_lo = (6 - 1) + (4 - 1)
sd_low = math.sqrt(ss_lo / df_lo)  # FFN과 배치 크기만(위치 임베딩·헤드 수는 구조가 달라 실제 효과가 섞일 수 있음)
out["null_groups"] = g_out
out["null_pooled_sd"] = sd_pool
out["null_pooled_sd_low"] = {"sd": sd_low, "df": df_lo, "groups": ["FFN 활성함수(표 11)", "배치 크기(표 10)"]}
out["note_null"] = ("'영향 없음' 변형들의 합동 표준편차(자유도 %d). 변형 효과가 섞여 있어 시드 노이즈의 상한 추정이며, "
                    "시드 반복 값이 아니다." % pooled_df)

# ---- 2. 각 주장의 차이를 노이즈 척도로 환산 ----
def z(d):
    return d / sd_pool  # 보수적(큰 노이즈) 기준. z_low = d / sd_low 는 claims 출력 때 같이 계산

claims = {}
base = t2_wd[0.1][0]
claims["wd 0.0 대 0.1 (표 2)"] = {"diff_pp": base - t2_wd[0.0][0], "z": z(base - t2_wd[0.0][0])}
claims["wd 0.1 대 0.5 (표 2)"] = {"diff_pp": t2_wd[0.5][0] - base, "z": z(t2_wd[0.5][0] - base)}
claims["wd 0.1 대 1.0 (표 2)"] = {"diff_pp": t2_wd[1.0][0] - base, "z": z(t2_wd[1.0][0] - base)}
claims["wd 0.0 대 0.5 (표 2)"] = {"diff_pp": t2_wd[0.5][0] - t2_wd[0.0][0], "z": z(t2_wd[0.5][0] - t2_wd[0.0][0])}
claims["lr 4e-4 대 2e-4, 20k (표 9)"] = {"diff_pp": 18.18 - 11.21, "z": z(18.18 - 11.21)}
claims["lr 4e-4 대 1e-4, 20k (표 9)"] = {"diff_pp": 18.18 - 2.90, "z": z(18.18 - 2.90)}
claims["lr x steps = 8 일정: 8e-4/10k 대 4e-4/20k (표 9)"] = {"diff_pp": 23.44 - 18.18, "z": z(23.44 - 18.18)}
claims["lr x steps = 8 일정: 4e-4/20k 대 2e-4/40k (표 9)"] = {"diff_pp": 18.18 - 16.81, "z": z(18.18 - 16.81)}
claims["lr x steps = 8 일정: 4e-4/20k 대 1e-4/80k (표 9)"] = {"diff_pp": 18.18 - 6.29, "z": z(18.18 - 6.29)}
sig = np.array([r[2] for r in t14_sigmoid])
claims["softmax 18.18 대 sigmoid 6개 평균 (표 14 오른쪽)"] = {"diff_pp": 18.18 - float(sig.mean()), "z": z(18.18 - float(sig.mean())),
                                                           "sigmoid_values": sig.tolist(), "sigmoid_max": float(sig.max())}
claims["sigmoid 정규화 없음 0.44 대 있음 30.24 (표 6)"] = {"diff_pp": 30.24 - 0.44, "z": z(30.24 - 0.44)}
claims["PE: 최대-최소, Relative 제외 (표 3)"] = {"diff_pp": g_out["위치 임베딩, Relative 제외(표 3)"]["range"],
                                            "z": z(g_out["위치 임베딩, Relative 제외(표 3)"]["range"])}
claims["normalization scale 2.0 대 1.0 (표 14 왼쪽)"] = {"diff_pp": 29.10 - 18.18, "z": z(29.10 - 18.18)}
claims["normalization scale 1.0 대 0.2 (표 14 왼쪽)"] = {"diff_pp": 18.18 - 9.41, "z": z(18.18 - 9.41)}
claims["normalization scale 1.0 대 0.1 (표 14 왼쪽)"] = {"diff_pp": 18.18 - 3.59, "z": z(18.18 - 3.59)}
claims["1B: softmax 45.11 대 sigmoid 2.46 (본문 7.4절)"] = {"diff_pp": 45.11 - 2.46, "z": z(45.11 - 2.46)}
for v in claims.values():
    v["z_low"] = v["diff_pp"] / sd_low
    v["z_high"] = v.pop("z")
out["claim_effects_vs_null_sd"] = claims

# ---- 3. 학습률 표: lr x steps 가 일정할 때의 단조성과 손실 교락 ----
const = [(r[0], r[1], r[2], r[3]) for r in t9_lr if abs(r[0] * r[1] - 8e-4 * 10) < 1e-12]
out["lr_const_product"] = [{"lr": r[0], "steps_k": r[1], "sink": r[2], "val_loss": r[3]} for r in const]
x = np.array([r[3] for r in const])
y = np.array([r[2] for r in const])
out["lr_const_corr_loss_vs_sink"] = float(np.corrcoef(x, y)[0, 1])

# ---- 4. 합과 보존 (표 5): k 편향과 첫 토큰의 합 ----
tot = [a + b for a, b in zip(t5["sink_star"], t5["sink_1"])]
out["table5_sum_sink_star_plus_sink_1"] = dict(zip(t5["v*"], tot))

# ---- 5. 표 13: 학습 가능 차원 수에 대한 단조성 ----
d = np.diff(t13["sink_star"])
out["table13_diffs"] = d.tolist()
out["table13_monotone_increasing"] = bool((d >= 0).all())

# ---- 6. 표 1 ----
out["table1_chat_minus_base_pp"] = {k: v[1] - v[0] for k, v in t1_chat.items()}
out["table1_natural_to_random_drop_pp"] = {k: v[0] - v[1] for k, v in t1_input.items()}
# 첫 위치 임베딩 대체 (표 7)
out["table7_vs_baseline_pp"] = {str(k): v - t7_replace["no"] for k, v in t7_replace.items()}

# ---- 7. 손실 비교 가능성: sigmoid/elu 등 "비슷한 검증 손실" ----
base_loss = 3.73
out["table6_loss_delta_vs_softmax"] = {k: round(v[1] - base_loss, 2) for k, v in t6_main.items()}
out["table6_note"] = "정규화 없는 sigmoid/elu는 -0.04~-0.03(더 낮음), identity/MLP 커널은 +0.12~+0.26(더 나쁨)"

# ---- 8. 규모·예산 산술 ----
tok_per_step = 512 * 2048
steps = 20000
budget = tok_per_step * steps
out["budget"] = {"tokens_per_step": tok_per_step, "tokens_total_60M": budget,
                 "epochs_over_5B": budget / 5e9, "epochs_1B_40k": tok_per_step * 40000 / 5e9,
                 "epochs_by_data_amount": {"5B": budget / 5e9, "2.5B": budget / 2.5e9, "500M": budget / 5e8,
                                           "200M": budget / 2e8, "100M": budget / 1e8, "50M": budget / 5e7}}
d_, L_, ff, V = 768, 10, 1536, 50432
per_layer = 4 * d_ * d_ + 3 * d_ * ff
out["params_60M"] = {"per_layer": per_layer, "non_embedding": per_layer * L_, "embedding_plus_unembedding": 2 * V * d_,
                     "heads_total": L_ * 8, "pp_per_head_per_sequence": 100 / (L_ * 8)}
d1, L1, ff1 = 2048, 22, 5632
out["params_1B_config"] = {"non_embedding": L1 * (4 * d1 * d1 + 3 * d1 * ff1), "heads_total": 22 * 16}
# 헤드 단위 환산: 18.18% 는 80개 헤드 중 평균 몇 개
out["heads_equivalent"] = {k: v[0] / 100 * 80 for k, v in {"wd0.0": t2_wd[0.0], "wd0.1": t2_wd[0.1], "wd0.5": t2_wd[0.5]}.items()}

(ROOT / "analysis" / "tables_recompute_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("pooled null sd (4 groups)", round(sd_pool, 2), "df", pooled_df, "| FFN+batch only", round(sd_low, 2), "df", df_lo)
for k, v in g_out.items():
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a != "values"})
for k, v in claims.items():
    print(f"{k:60s} diff {v['diff_pp']:6.2f} pp  z_high={v['z_high']:.1f} z_low={v['z_low']:.1f}")
print("lr const", out["lr_const_product"], "corr(loss,sink)", round(out["lr_const_corr_loss_vs_sink"], 2))
print("table5 sums", out["table5_sum_sink_star_plus_sink_1"])
print("table13 diffs", out["table13_diffs"])
print("table1", out["table1_chat_minus_base_pp"], out["table1_natural_to_random_drop_pp"])
print("table6 loss delta", out["table6_loss_delta_vs_softmax"])
print("budget", out["budget"])
print("params", out["params_60M"], out["params_1B_config"], out["heads_equivalent"])
