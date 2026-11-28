"""GPTQ 논문의 산술 주장 재계산: 표 5·6·7·8 문장, OPT-175B 메모리, 양자화 시간 plausibility, 소형 기기 모델 크기.
표 값은 논문(arXiv:2210.17323v2) 표에서 옮긴 것이다. 하드웨어 사양(대역폭, FLOPS)은 공개 사양을 기억해 쓴 값이므로 plausibility 용도다.
실행: python3 paper_arithmetic.py  (결과 paper_arithmetic_results.json)
"""
import json
from pathlib import Path

GiB = 2 ** 30
R = {}

# ---- 1. 본문 문장 대 표 ----
T3 = {  # 표 3 OPT WikiText2: full, RTN4, GPTQ4, RTN3, GPTQ3
    "175B": (8.34, 10.54, 8.37, 7.3e3, 8.68), "13B": (10.13, 11.32, 10.31, 3.4e3, 11.61),
    "125M": (27.65, 37.28, 31.12, 1.3e3, 53.85)}
T5 = {"OPT": {"full": (8.34, 12.01, 10.13), "g1024": (8.45, 12.48, 10.47), "g128": (8.45, 12.37, 10.36),
              "gptq3": (8.68, 12.68, 10.67), "gptq4": (8.37, 12.26, 10.28)},
      "BLOOM": {"full": (8.11, 14.59, 11.71), "g1024": (8.35, 15.01, 11.98), "g128": (8.26, 14.89, 11.85),
                "gptq3": (8.64, 15.57, 12.27), "gptq4": (8.21, 14.75, 11.81)}}
chk = {}
chk["RTN4 OPT-175B 손실(본문 2.2)"] = round(T3["175B"][1] - T3["175B"][0], 2)
chk["GPTQ4 OPT-175B 손실(본문 0.03)"] = round(T3["175B"][2] - T3["175B"][0], 2)
chk["RTN4 175B가 13B FP(10.13)보다 나쁜가"] = T3["175B"][1] > T3["13B"][0]
d = {m: [round(T5[m]["gptq4"][i] - T5[m]["full"][i], 2) for i in range(3)] for m in T5}
chk["4bit GPTQ - FP (Wiki2, PTB, C4)(본문 '≤0.25')"] = d
g = {}
for m in T5:
    base = T5[m]["gptq3"]
    g[m] = {"g1024 개선(3열)": [round(base[i] - T5[m]["g1024"][i], 2) for i in range(3)],
            "g128이 g1024보다 추가 개선": [round(T5[m]["g1024"][i] - T5[m]["g128"][i], 2) for i in range(3)],
            "g128과 FP 차이": [round(T5[m]["g128"][i] - T5[m]["full"][i], 2) for i in range(3)]}
chk["3bit 그룹 효과(본문: g1024 평균 ≈0.2, g128 추가 ≈0.1, FP와 0.1–0.3)"] = g
avg = {m: {"g1024 평균 개선": round(sum(base - gg for base, gg in zip(T5[m]["gptq3"], T5[m]["g1024"])) / 3, 3),
           "g128 추가 평균": round(sum(a - b for a, b in zip(T5[m]["g1024"], T5[m]["g128"])) / 3, 3)} for m in T5}
chk["그룹 평균"] = avg
# 표 7(2-bit)
T7 = {"OPT-175B": (8.34, 9.58, 9.18, 8.94, 8.68), "BLOOM": (8.11, 9.55, 9.17, 8.83, 8.64)}
chk["2bit g128 증가(본문 <1.5), g32 증가(본문 0.6–0.7)"] = {k: (round(v[1] - v[0], 2), round(v[3] - v[0], 2)) for k, v in T7.items()}
# 표 8과 표 3의 OPT-125M FP
chk["OPT-125M FP: 표 3의 27.65 대 표 8의 27.66"] = (27.65, 27.66)
# 표 6
chk["표 6 속도비 재계산(4.53, 3.24)"] = [round(589 / 130, 2), round(230 / 71, 2)]
# ZeroQuant 외삽
chk["ZeroQuant-LKD 3h@1.3B를 175B로 선형 외삽(시간)"] = round(3 * 175 / 1.3)
chk["GPTQ 대 ZeroQuant 규모비(본문 '100× 더 큰')"] = round(175 / 1.3)
R["claims_vs_tables"] = chk

# ---- 2. OPT-175B 메모리 ----
d_, L, V, P, ffn = 12288, 96, 50272, 2050, 4 * 12288
attn = 4 * (d_ * d_ + d_)
mlp = d_ * ffn + ffn + ffn * d_ + d_
ln = 4 * d_
per_layer = attn + mlp + ln
lin_w = L * (4 * d_ * d_ + 2 * d_ * ffn)       # 양자화 대상 가중치(행렬만)
emb = V * d_ + P * d_
total = L * per_layer + emb + 2 * d_            # 마지막 LN
mem = {"총 파라미터(B)": total / 1e9, "양자화 대상 행렬(B)": lin_w / 1e9, "임베딩(B)": emb / 1e9,
       "FP16 GiB(본문 326)": total * 2 / GiB}
rows_total = L * (3 * d_ + d_ + ffn + d_)   # 출력 행 수(qkv, out, fc1, fc2)
for bits in (3, 4, 8):
    w = lin_w * bits / 8 + rows_total * (2 + bits / 8) + (total - lin_w) * 2  # 행별 scale(fp16) + zero(비트) + 나머지 FP16
    mem[f"{bits}bit GiB"] = w / GiB
kv = 2 * L * d_ * 2048 * 2
mem["KV 캐시 2048토큰 FP16 GiB(본문 9)"] = kv / GiB
mem["3bit + KV GiB (80GiB GPU 1장에 들어가나)"] = mem["3bit GiB"] + kv / GiB
mem["4bit + KV GiB"] = mem["4bit GiB"] + kv / GiB
mem["FP16 + KV GiB → 80GiB GPU 수(본문 5)"] = (total * 2 + kv) / GiB, (total * 2 + kv) / GiB / 80
mem["8bit + KV GiB → 80GiB GPU 수(본문 3)"] = mem["8bit GiB"] + kv / GiB, (mem["8bit GiB"] + kv / GiB) / 80
mem["그룹 오버헤드 비트/가중치: g1024, g128(본문 0.02, 0.15), 3bit, scale fp16 + zero 3bit"] = ((16 + 3) / 1024, (16 + 3) / 128)
mem["2bit g128, g32 평균 비트(본문 ≈2.2, ≈2.6)"] = (2 + (16 + 2) / 128, 2 + (16 + 2) / 32)
R["opt175b_memory"] = mem

# ---- 3. 양자화 시간 plausibility(FLOPs) ----
N = 128 * 2048
fl = {}
H_acc = L * 2 * N * (5 * d_ ** 2 + ffn ** 2)
upd = L * 2 * (4 * d_ * d_ ** 2 + ffn * d_ ** 2 + d_ * ffn ** 2)          # d_row·d_col² ×2 (열 갱신)
chol = L * (5 * (d_ ** 3) + ffn ** 3) * 1.0                                # Cholesky+역행렬 대략 ~ d³ 몇 배 (상한 근사)
fwd = L * 2 * 2 * N * 12 * d_ ** 2                                          # 블록 통과 2회(FP16, 텐서코어)
fl.update({"Hessian 누적 FLOPs": H_acc, "열 갱신 FLOPs": upd, "Cholesky 등(근사)": chol, "블록 통과 2회 FLOPs(FP16)": fwd})
fp32_peak = 19.5e12  # A100 FP32(TF32 제외) 공개 사양, 공식 코드는 TF32를 끈다
sec = (H_acc + upd + chol) / fp32_peak
fl["FP32 피크 가정 시간(h)"] = sec / 3600
fl["보고 4.2h / 피크 시간"] = 4.2 / (sec / 3600)
R["opt175b_quant_time_flops"] = fl
t2 = {"OPT": {"13B": 20.9 / 60, "30B": 44.9 / 60, "66B": 1.6, "175B": 4.2}, "BLOOM": {"1.7B": 2.9 / 60, "3B": 5.2 / 60, "7.1B": 10 / 60, "176B": 3.8}}
R["table2_time_per_billion_params_min"] = {m: {k: round(v * 60 / float(k[:-1]), 2) for k, v in t.items()} for m, t in t2.items()}

# ---- 4. 지연 시간과 대역폭 ----
bw = {}
w3 = (lin_w * 3 / 8 + rows_total * (2 + 3 / 8) + (total - lin_w) * 2)  # 바이트 (토큰마다 가중치를 한 번 읽는다)
w16 = total * 2
bw["A100 3bit 유효 대역폭 TB/s"] = w3 / 0.071 / 1e12
bw["A100 FP16 유효(순차 파이프라인이라 한 번에 한 GPU) TB/s"] = w16 / 0.230 / 1e12
bw["A6000 3bit 유효 TB/s"] = w3 / 0.130 / 1e12
bw["A6000 FP16 유효 TB/s"] = w16 / 0.589 / 1e12
bw["바이트 비율 상한 16/3(순수 대역폭 한계 속도비)"] = w16 / w3
bw["측정 속도비(A100, A6000)"] = (230 / 71, 589 / 130)
bw["공개 사양 피크(기억에 의함): A100 80GB 약 2.0 TB/s, A6000 0.768 TB/s"] = (2.0, 0.768)
R["latency_vs_bandwidth"] = bw

# ---- 5. 24 GiB 맥 ----
mac = {}
def bits_per_w(b, g):
    return b if g is None else b + (16 + b) / g
budgets = {"GPU 한도 2/3 (16 GiB)": 16.0, "GPU 한도 3/4 (18 GiB)": 18.0, "전부 24 GiB(비현실)": 24.0}
runtime = 1.5  # GiB: 활성, 런타임, 프레임워크 가정
fmts = {"FP16": (16, None), "8bit": (8, None), "4bit g128": (4, 128), "4bit 행별": (4, None), "3bit g128": (3, 128),
        "3bit 행별": (3, None), "2bit g128": (2, 128)}
for bname, B in budgets.items():
    for kvg in (0.0, 2.0):
        row = {}
        for fname, (b, g) in fmts.items():
            bpw = bits_per_w(b, g)
            row[fname] = {"bits/w": round(bpw, 3), "max_params_B": round((B - runtime - kvg) * GiB * 8 / bpw / 1e9, 1)}
        mac[f"{bname}, KV {kvg:g} GiB"] = row
R["mac_24GiB_max_params"] = mac
# 가중치만 놓고 몇 B 모델이 들어가나(임베딩 FP16 5%라 가정하지 않음, 단순)
sizes = {}
for n in (7, 13, 34, 70):
    sizes[f"{n}B"] = {f: round(n * 1e9 * bits_per_w(*v) / 8 / GiB, 1) for f, v in fmts.items()}
R["weights_GiB_by_model_size"] = sizes

# 소형 모델: 임베딩이 FP16로 남는 효과
def small(name, n_total, n_emb, n_lin):
    out = {}
    for fname in ("4bit 행별", "4bit g128", "3bit 행별", "3bit g128"):
        b, g = fmts[fname]
        by = n_lin * bits_per_w(b, g) / 8 + (n_total - n_lin) * 2
        out[fname] = {"MB": round(by / 1e6, 1), "유효 bits/param": round(by * 8 / n_total, 2), "FP16 대비": round(by / (n_total * 2), 3)}
    return out
R["small_models"] = {"GPT-2 small(124.4M, 선형 85.0M, 임베딩 wte 38.6M + wpe 0.8M)": small("gpt2", 124.44e6, 39.4e6, 12 * 12 * 768 ** 2),
                     "Pythia-160M(162M, 선형 85M, 임베딩·언임베딩 각 38.6M)": small("p160", 162.3e6, 77.2e6, 12 * 12 * 768 ** 2)}
Path(__file__).with_name("paper_arithmetic_results.json").write_text(json.dumps(R, indent=1, default=str, ensure_ascii=False))
print(json.dumps(R, indent=1, default=str, ensure_ascii=False))
