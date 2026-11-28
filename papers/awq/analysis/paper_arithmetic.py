"""논문 표 재계산과 온디바이스 메모리·속도 산술. 결과는 paper_arithmetic_results.json.
표의 수치는 논문 v6 텍스트에서 옮긴 것이다(출처 표 번호를 키에 적음)."""
import json

R = {}

# ---- Table 2 (OPT-6.7B, W3 g128): 평균 Delta'/Delta * 1/s 를 평균 Delta'/Delta와 s로 다시 계산
s = [1, 1.25, 1.5, 2, 4]
ratio = [1, 1.005, 1.013, 1.038, 1.213]
reported = [1, 0.804, 0.676, 0.519, 0.303]
ppl = [23.54, 12.87, 12.48, 11.92, 12.36]
R["table2"] = dict(s=s, mean_ratio=ratio, reported_err_ratio=reported,
                   recomputed_err_ratio=[round(r / x, 3) for r, x in zip(ratio, s)],
                   max_abs_diff=max(abs(round(r / x, 3) - q) for r, x, q in zip(ratio, s, reported)),
                   best_s_by_ppl=s[ppl.index(min(ppl))])

# ---- Table 1 (INT3 g128, WikiText-2 PPL): FP16 위치 선택별로 RTN-FP16 격차를 얼마나 메우는가
fp16 = dict(zip(["1.3B", "6.7B", "13B"], [14.62, 10.86, 10.13]))
rtn = dict(zip(["1.3B", "6.7B", "13B"], [119.00, 23.54, 46.04]))
t1 = {
    "act_0.1%": [25.03, 11.58, 10.51], "act_1%": [16.91, 11.39, 10.43], "act_3%": [16.68, 11.36, 10.42],
    "weight_0.1%": [108.71, 23.41, 46.07], "weight_1%": [98.55, 22.37, 48.96], "weight_3%": [98.08, 22.45, 54.49],
    "random_0.1%": [119.76, 23.54, 44.87], "random_1%": [109.38, 24.23, 42.00], "random_3%": [61.49, 24.22, 39.71],
}
gap = {}
for k, v in t1.items():
    gap[k] = {m: round((rtn[m] - x) / (rtn[m] - fp16[m]), 3) for m, x in zip(fp16, v)}
R["table1_gap_closed"] = gap
R["table1_worse_than_rtn"] = [f"{k}/{m}" for k, v in t1.items() for m, x in zip(fp16, v) if x > rtn[m]]

# ---- Table 3 vs Table 1 일관성 (OPT-1.3B/6.7B/13B RTN, 1% FP16)
R["table3_vs_table1"] = {"rtn_1.3B_table1": 119.00, "rtn_1.3B_table3": 119.47}

# ---- Table 4 (WikiText-2 PPL, g128): AWQ가 RTN-FP16 격차를 메우는 비율, GPTQ-R와의 차이
t4 = {
    "Llama-2-7B": dict(fp=5.47, i3=dict(rtn=6.66, gptq=6.43, gptqr=6.42, awq=6.24), i4=dict(rtn=5.73, gptq=5.69, gptqr=5.63, awq=5.60)),
    "Llama-2-13B": dict(fp=4.88, i3=dict(rtn=5.52, gptq=5.48, gptqr=5.41, awq=5.32), i4=dict(rtn=4.98, gptq=4.98, gptqr=4.99, awq=4.97)),
    "Llama-2-70B": dict(fp=3.32, i3=dict(rtn=3.98, gptq=3.88, gptqr=3.86, awq=3.74), i4=dict(rtn=3.46, gptq=3.42, gptqr=3.43, awq=3.41)),
    "LLaMA-7B": dict(fp=5.68, i3=dict(rtn=7.01, gptq=8.81, gptqr=6.53, awq=6.35), i4=dict(rtn=5.96, gptq=6.22, gptqr=5.83, awq=5.78)),
    "LLaMA-13B": dict(fp=5.09, i3=dict(rtn=5.88, gptq=5.66, gptqr=5.64, awq=5.52), i4=dict(rtn=5.25, gptq=5.23, gptqr=5.20, awq=5.19)),
    "LLaMA-30B": dict(fp=4.10, i3=dict(rtn=4.88, gptq=4.88, gptqr=4.74, awq=4.61), i4=dict(rtn=4.23, gptq=4.24, gptqr=4.22, awq=4.21)),
    "LLaMA-65B": dict(fp=3.53, i3=dict(rtn=4.24, gptq=4.17, gptqr=4.21, awq=3.95), i4=dict(rtn=3.67, gptq=3.66, gptqr=3.66, awq=3.62)),
}
out4 = {}
for m, d in t4.items():
    row = {}
    for b in ("i3", "i4"):
        x = d[b]
        row[b] = dict(awq_minus_rtn=round(x["awq"] - x["rtn"], 2), awq_minus_gptqr=round(x["awq"] - x["gptqr"], 2),
                      awq_minus_best_other=round(x["awq"] - min(x["gptq"], x["gptqr"]), 2),
                      gap_closed=round((x["rtn"] - x["awq"]) / (x["rtn"] - d["fp"]), 2),
                      awq_degradation=round(x["awq"] - d["fp"], 2), rtn_degradation=round(x["rtn"] - d["fp"], 2))
    out4[m] = row
R["table4"] = out4
R["table4_int4_awq_vs_gptqr_diffs"] = sorted(out4[m]["i4"]["awq_minus_gptqr"] for m in out4)

# ---- 메모리 산술
def llama_params(L, h, ffn, V, kv_dim=None):
    kv = h if kv_dim is None else kv_dim
    attn = h * h * 2 + h * kv * 2           # q, o + k, v
    mlp = 3 * h * ffn
    lin = L * (attn + mlp)
    emb = 2 * V * h                         # embed + lm_head (비결합)
    norms = L * 2 * h + h
    return dict(linear=lin, embed_head=emb, norms=norms, total=lin + emb + norms)


MODELS = {"Llama-2-7B": llama_params(32, 4096, 11008, 32000), "Llama-2-13B": llama_params(40, 5120, 13824, 32000),
          "Llama-2-70B": llama_params(80, 8192, 28672, 32000, kv_dim=1024)}
mem = {}
GB = 1e9
GIB = 2 ** 30
for name, p in MODELS.items():
    fp16 = p["total"] * 2
    entry = dict(params=p["total"], fp16_GB=fp16 / GB, fp16_GiB=fp16 / GIB)
    for tag, extra_bits in (("g128_fp16_scale_fp16_zero", 32 / 128), ("g128_fp16_scale_int4_zero", 20 / 128), ("g128_scale_only", 16 / 128), ("no_group", 0.0)):
        w = p["linear"] * (4 + extra_bits) / 8 + (p["embed_head"] + p["norms"]) * 2
        entry["w4_" + tag] = dict(GB=w / GB, GiB=w / GIB, ratio_to_fp16=fp16 / w, bits_per_linear_weight=4 + extra_bits)
    mem[name] = entry
R["memory"] = mem

# KV cache fp16
def kv_bytes(L, kv_dim, ctx):
    return 2 * L * kv_dim * ctx * 2

R["kv_cache_fp16_GiB"] = {"Llama-2-7B_ctx4096": kv_bytes(32, 4096, 4096) / GIB, "Llama-2-13B_ctx4096": kv_bytes(40, 5120, 4096) / GIB,
                          "Llama-2-70B_ctx4096": kv_bytes(80, 1024, 4096) / GIB}
w7 = mem["Llama-2-7B"]["w4_g128_fp16_scale_fp16_zero"]["GiB"]
R["fits_24GiB_7B_W4_ctx4096_GiB"] = w7 + R["kv_cache_fp16_GiB"]["Llama-2-7B_ctx4096"]
R["fits_8GB_13B"] = dict(weights_GB=mem["Llama-2-13B"]["w4_g128_fp16_scale_fp16_zero"]["GB"], weights_GiB=mem["Llama-2-13B"]["w4_g128_fp16_scale_fp16_zero"]["GiB"],
                         kv_ctx1024_GiB=kv_bytes(40, 5120, 1024) / GIB)
R["fits_64GB_70B"] = dict(weights_GB=mem["Llama-2-70B"]["w4_g128_fp16_scale_fp16_zero"]["GB"], kv_ctx4096_GiB=R["kv_cache_fp16_GiB"]["Llama-2-70B_ctx4096"])

# 디코딩 상한: 토큰당 가중치 한 번 읽기 -> tok/s <= BW / bytes
bw = [68, 100, 120, 200, 273, 400, 546, 800, 1008]   # GB/s, 일반 값. 어느 기기인지는 사용자 기기 사양서로 확인 필요
dec = {}
for name in ("Llama-2-7B",):
    f = mem[name]["fp16_GB"]
    q = mem[name]["w4_g128_fp16_scale_fp16_zero"]["GB"]
    dec[name] = {str(b): dict(fp16=round(b / f, 1), w4=round(b / q, 1), ratio=round(f / q, 2)) for b in bw}
R["decode_upper_bound_tok_s"] = dec
R["paper_speedup_vs_ideal"] = dict(ideal_7B=mem["Llama-2-7B"]["fp16_GB"] / mem["Llama-2-7B"]["w4_g128_fp16_scale_fp16_zero"]["GB"],
                                    paper_claim_avg="3.2-3.3x (TinyChat vs HF FP16, 논문 §1)")
# 산술 강도 (디코딩, 배치 1): 파라미터당 2 FLOP, fp16 2 byte -> 1 FLOP/B; W4 -> 2 FLOP / 0.5 B
R["arith_intensity"] = dict(fp16=2 / 2, w4_nominal=2 / 0.5, w4_with_group=2 / ((4 + 32 / 128) / 8), roofline_4090_TFLOPS_at_1TBs=1.0 * 4,
                            paper_ridge=165 / 1.0)
# 논문 §5.4 4090 수치(Llama-2-7B): HF FP16 52 tok/s, 융합 FP16 62 tok/s, 양자화 커널로 3.1x 추가 -> 약 192 tok/s. 대역폭 1 TB/s(논문 §4.1)
bw4090 = 1000.0
f, q = mem["Llama-2-7B"]["fp16_GB"], mem["Llama-2-7B"]["w4_g128_fp16_scale_fp16_zero"]["GB"]
R["roofline_4090_llama2_7b"] = dict(ideal_fp16=bw4090 / f, ideal_w4=bw4090 / q, hf_fp16=52, fused_fp16=62, w4_derived=62 * 3.1,
                                    frac_of_ideal_hf=52 / (bw4090 / f), frac_of_ideal_fused=62 / (bw4090 / f), frac_of_ideal_w4=62 * 3.1 / (bw4090 / q),
                                    speedup_vs_hf=62 * 3.1 / 52, ideal_speedup=f / q)
print(R["roofline_4090_llama2_7b"])
R["gpt3_fp16_GB"] = 175e9 * 2 / GB
json.dump(R, open("paper_arithmetic_results.json", "w"), indent=1)
print(json.dumps({k: R[k] for k in ("table2", "table1_worse_than_rtn", "table4_int4_awq_vs_gptqr_diffs", "kv_cache_fp16_GiB", "fits_24GiB_7B_W4_ctx4096_GiB", "fits_8GB_13B", "fits_64GB_70B", "arith_intensity", "gpt3_fp16_GB")}, indent=1))
for n, e in mem.items():
    print(n, round(e["params"] / 1e9, 3), "B params; fp16 %.2f GB;" % e["fp16_GB"], {k: (round(v["GB"], 2), round(v["ratio_to_fp16"], 2)) for k, v in e.items() if k.startswith("w4_")})
print(R["table1_gap_closed"]["act_1%"], R["table1_gap_closed"]["weight_3%"])
print({m: (v["i4"]["gap_closed"], v["i3"]["gap_closed"]) for m, v in out4.items()})
print(dec["Llama-2-7B"]["100"], dec["Llama-2-7B"]["400"])
