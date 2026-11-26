"""CPU-ARC 비용 주장 재계산: 표 2의 배수, 부록 C의 토큰·단가, 손익분기, 단가 민감도.

실행: python3 -I cost_check.py  (결과: cost_check_results.json)
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_tables import *  # noqa

here = os.path.dirname(os.path.abspath(__file__))
res = {}

def opus_cost(i, o, rd=0, wr=0, variant="A"):
    P = OPUS
    if variant == "A":      # In = 캐시 아닌 입력 전체
        return (i * P["in"] + o * P["out"] + rd * P["cache_rd"] + (wr or 0) * P["cache_wr"]) / 1e6
    if variant == "B":      # In에 cache read가 포함
        return ((i - rd) * P["in"] + o * P["out"] + rd * P["cache_rd"] + (wr or 0) * P["cache_wr"]) / 1e6
    if variant == "C":      # In에 cache read와 write 모두 포함
        return ((i - rd - (wr or 0)) * P["in"] + o * P["out"] + rd * P["cache_rd"] + (wr or 0) * P["cache_wr"]) / 1e6
    if variant == "D":      # cache write/read 없이 In·Out만
        return (i * P["in"] + o * P["out"]) / 1e6

def llama_cost(i, o):
    return (i * LLAMA1B["in"] + o * LLAMA1B["out"]) / 1e6

# 1) 표 2 배수 재계산 (표 2의 4자리 반올림 비용으로)
res["table2_multipliers"] = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    sav_round = r["src"] / r["cache"]
    bld_x = r["bld"] / r["src"]
    implied_cache = r["src"] / r["sav_x"]
    res["table2_multipliers"][k] = {
        "build_x_recomputed": round(bld_x, 2), "build_x_reported": r["build_x"],
        "savings_x_from_rounded_cache_cost": round(sav_round, 1), "savings_x_reported": r["sav_x"],
        "implied_cache_cost": round(implied_cache, 6),
        "savings_x_range_if_cache_cost_in_[c-0.00005,c+0.00005]": [
            round(r["src"] / (r["cache"] + 0.00005), 1), round(r["src"] / max(r["cache"] - 0.00005, 1e-9), 1)],
    }

# 2) 부록 C: 토큰 x 단가로 비용 재계산
res["appendix_c"] = {}
for k, roles in APPX_C.items():
    out = {}
    s = roles["source"]
    out["source_recomputed"] = round(opus_cost(s[0], s[1]), 4)
    out["source_reported"] = s[4]
    b = roles["builder"]
    out["builder_reported"] = b[4]
    out["builder_reported_over_variantB"] = round(b[4] / opus_cost(b[0], b[1], b[2], b[3], "B"), 4)
    out["builder_variants"] = {v: round(opus_cost(b[0], b[1], b[2], b[3], v), 4) for v in "ABCD"}
    cache_c = 0.0
    parts = {}
    for role in ("target", "extractor", "converter"):
        t = roles[role]
        c = llama_cost(t[0], t[1])
        parts[role] = {"recomputed": round(c, 6), "reported": t[4]}
        cache_c += c
    out["cache_parts"] = parts
    out["cache_recomputed"] = round(cache_c, 6)
    out["cache_reported_table2"] = TABLE2[k][8]
    out["savings_x_from_tokens"] = round(opus_cost(s[0], s[1]) / cache_c, 1)
    out["savings_x_reported"] = TABLE2[k][10]
    out["total_sum_of_role_costs"] = round(s[4] + b[4] + sum(roles[r][4] for r in ("target", "extractor", "converter")), 4)
    out["total_reported"] = roles["total"]
    res["appendix_c"][k] = out

# 3) 손익분기: B/(S-C) (논문 식) 과 성공률·시간 보정
res["breakeven"] = {}
for k, v in TABLE2.items():
    r = dict(zip(T2_KEYS, v))
    S, B = r["src"], r["bld"]
    C_tok = res["appendix_c"][k]["cache_recomputed"]
    n_paper = math.ceil(B / (S - r["cache"]))
    n_tok = math.ceil(B / (S - C_tok))
    # 교란 통과율을 정답률로 보면 (복합 교란): 정답 하나당 비용으로 보면 C/q
    q = r["comb"] / 100
    n_q = math.ceil(B / (S - C_tok / q)) if S > C_tok / q else None
    # 소스 모델 정확도를 같이 보정: 소스는 정답당 S/s_acc
    sacc = r["s_acc"] / 100
    n_q2 = math.ceil(B / (S / sacc - C_tok / q)) if S / sacc > C_tok / q else None
    res["breakeven"][k] = {"B": B, "S": S, "paper_formula_ceil": n_paper, "with_token_cache_cost": n_tok,
                           "per_correct_answer_combined_pass": n_q2, "note": "B/(S-C), 정답당 비용 보정은 S/s_acc 대 C/comb"}

# 4) 민감도: 헤드라인 배수가 단가·소스 모델 가정에 얼마나 움직이나
res["sensitivity"] = {}
for k in TABLE2:
    r = dict(zip(T2_KEYS, TABLE2[k]))
    S = r["src"]; C = res["appendix_c"][k]["cache_recomputed"]
    row = {}
    # (a) CPU 실제 비용: 캐시 비용이 보고값의 m배
    for m in (1, 2, 5, 10, 30):
        row[f"cache_cost_x{m}"] = round(S / (C * m), 1)
    # (b) 소스 모델이 더 싼 모델: S가 f배로 줄면 (Sonnet급 3x, Haiku급 5x 싼 경우 등)
    for f in (0.2, 0.33, 0.5):
        row[f"source_price_x{f}"] = round(S * f / C, 1)
    # (c) 소스의 thinking 출력이 반으로 (다른 모델 효율)
    s = APPX_C[k]["source"]
    row["source_output_tokens_half"] = round(opus_cost(s[0], s[1] / 2) / C, 1)
    # (d) 빌드 비용을 N회 호출에 분산한 실효 배수: S / (C + B/N)
    B = r["bld"]
    for N in (1, 10, 100, 1000, 10000):
        row[f"effective_with_build_N{N}"] = round(S / (C + B / N), 2)
    res["sensitivity"][k] = row

# 5) 배수 범위가 반올림 오차에서 어디까지 흔들리나
rep = [TABLE2[k][10] for k in TABLE2]
res["headline"] = {"reported_min": min(rep), "reported_max": max(rep),
                   "recomputed_from_tokens_min": min(res["appendix_c"][k]["savings_x_from_tokens"] for k in TABLE2),
                   "recomputed_from_tokens_max": max(res["appendix_c"][k]["savings_x_from_tokens"] for k in TABLE2),
                   "build_x_min": min(TABLE2[k][9] for k in TABLE2), "build_x_max": max(TABLE2[k][9] for k in TABLE2)}

# 5b) 토큰 수 비교: 캐시 호출이 쓰는 토큰이 원본 풀이보다 많다. 배수는 토큰 수가 아니라 토큰당 단가 차이에서 나온다.
res["token_ratio"] = {}
for k in TABLE2:
    s_t = APPX_C[k]["source"][0] + APPX_C[k]["source"][1]
    c_t = sum(APPX_C[k][r][0] + APPX_C[k][r][1] for r in ("target", "extractor", "converter"))
    S, C = res["appendix_c"][k]["source_recomputed"], res["appendix_c"][k]["cache_recomputed"]
    res["token_ratio"][k] = {"source_tokens": s_t, "cache_tokens": c_t, "cache_over_source_tokens": round(c_t / s_t, 1),
                             "price_per_token_ratio_source_over_cache": round((S / s_t) / (C / c_t), 0)}

# 6) CPU 지연 추정(가정): 논문에 지연이 없다. 1B 모델의 CPU 처리량을 입력 200-1000 tok/s, 출력 10-30 tok/s로 가정
res["cpu_latency_assumption"] = {"assumption": "1B 모델 CPU: prefill 200-1000 tok/s, decode 10-30 tok/s (논문 값 아님)"}
for k in ("AIME25", "GSM8K", "GPQA-Diamond"):
    ti = sum(APPX_C[k][r][0] for r in ("target", "extractor", "converter"))
    to = sum(APPX_C[k][r][1] for r in ("target", "extractor", "converter"))
    res["cpu_latency_assumption"][k] = {"in_tokens": ti, "out_tokens": to,
        "seconds_best": round(ti / 1000 + to / 30, 1), "seconds_worst": round(ti / 200 + to / 10, 1)}

with open(os.path.join(here, "cost_check_results.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res["headline"]))
for k in TABLE2:
    a = res["appendix_c"][k]
    print(k, "src", a["source_recomputed"], a["source_reported"], "| bld", a["builder_reported"], a["builder_variants"],
          "| cache", a["cache_recomputed"], a["cache_reported_table2"], "| sav", a["savings_x_from_tokens"], a["savings_x_reported"],
          "| tot", a["total_sum_of_role_costs"], a["total_reported"])
for k in TABLE2:
    print(k, res["breakeven"][k]["paper_formula_ceil"], res["breakeven"][k]["with_token_cache_cost"], res["breakeven"][k]["per_correct_answer_combined_pass"])
