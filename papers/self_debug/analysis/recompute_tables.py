#!/usr/bin/env python3
"""Self-Debug (arXiv:2304.05128 v2) 공개 표 재계산.

논문의 표 1-3, 5, 그림 6, 8, 표 4에서 옮긴 값만 쓴다. 모델 호출 없음, 문항별 결과 없음.
1) 격자 검사: 각 정확도가 (문항 수 N)과 정수 정답 수로 설명되는가
2) 그림 6(b)의 난이도별 값으로 전체 정답 수를 복원해 표 2(a)와 대조
3) 이득 분해: 기준선 -> 단순 피드백 -> 단위 테스트 피드백 -> 설명 -> 추적
4) Wilson 구간, 비짝 차이 구간(Newcombe), 짝지은 McNemar 가능 범위(최선·최악 배치)
5) 주장 검산: 2-3%p, 9%p, 12%p, 10배 표본 효율, 표 4 백분율의 분모

사용: python3 -I analysis/recompute_tables.py  (결과: analysis/recompute_tables_results.json)
"""
import json
import math
import os
from fractions import Fraction

from scipy.stats import binomtest

N = {"spider": 1034, "transcoder": 560, "mbpp": 500}
# Spider dev 난이도별 문항 수(easy, medium, hard, extra): 그림 6(b)로 역산해 확인한다(아래 2)
SPIDER_HARD_N = {"easy": 248, "medium": 446, "hard": 174, "extra": 166}

# ---- 표 2 (실행 포함 기본 설정) ----
T2 = {
    "spider": {  # Baseline, Simple, +Expl
        "cols": ["Baseline", "Simple", "Expl"],
        "Codex": [81.3, 81.3, 84.1], "GPT-3.5": [71.1, 72.2, 72.2],
        "GPT-4": [73.2, 73.4, 73.6], "StarCoder": [64.7, 64.9, 64.9],
    },
    "transcoder": {  # Baseline, Simple, UT, UT+Expl, UT+Trace
        "cols": ["Baseline", "Simple", "UT", "UT+Expl", "UT+Trace"],
        "Codex": [80.4, 89.3, 91.6, 92.5, 87.9], "GPT-3.5": [89.1, 91.6, 92.7, 92.7, 92.3],
        "GPT-4": [77.3, 80.9, 88.8, 90.4, 89.5], "StarCoder": [70.0, 72.9, 76.4, 76.6, 73.6],
    },
    "mbpp": {
        "cols": ["Baseline", "Simple", "UT", "UT+Expl", "UT+Trace"],
        "Codex": [61.4, 68.2, 69.4, 69.8, 70.8], "GPT-3.5": [67.6, 70.8, 72.2, 74.2, 72.8],
        "GPT-4": [72.8, 78.8, 80.6, 80.4, 80.2], "StarCoder": [47.2, 50.6, 52.2, 52.2, 53.2],
    },
}
# ---- 표 3 (단위 테스트 실행 없음) ----
T3 = {
    "transcoder": {"cols": ["Baseline", "Simple", "Expl", "Trace"],
                   "Codex": [80.4, 83.4, 83.9, 83.9], "GPT-3.5": [89.1, 89.1, 89.1, 89.1],
                   "GPT-4": [77.3, 78.2, 78.0, 78.4]},
    "mbpp": {"cols": ["Baseline", "Simple", "Expl", "Trace"],
             "Codex": [61.4, 57.6, 64.4, 66.2], "GPT-3.5": [67.6, 68.2, 68.2, 69.2],
             "GPT-4": [72.8, 76.0, 76.0, 76.4]},
}
# ---- 표 1, 표 5, 그림 6, 그림 8 ----
T1A = {"T5-3B+N-best": 80.6, "LEVER": 81.9, "Coder-Reviewer": 74.5, "MBR-Exec": 75.2, "SD Codex": 81.3, "SD Codex+Expl": 84.1}
T1B = {"MBR-Exec n=25": 63.0, "Reviewer n=25": 66.9, "LEVER n=100": 68.9, "Codex n=10": 72.2,
       "SD Simple": 73.6, "SD UT": 75.2, "SD UT+Expl": 75.6}
T5 = {"spider": {"cols": ["Base(Codex)", "Simple", "Expl"], "Codex": [81.3, 81.3, 84.1], "GPT-3.5": [81.3, 81.7, 83.0]},
      "transcoder": {"cols": ["Base(GPT-3.5)", "Simple", "UT", "UT+Expl"], "Codex": [89.1, 90.2, 92.1, 92.7], "GPT-3.5": [89.1, 91.6, 92.7, 92.7]},
      "mbpp": {"cols": ["Base(GPT-3.5)", "Simple", "UT", "UT+Expl"], "Codex": [67.6, 72.4, 73.2, 73.6], "GPT-3.5": [67.6, 70.8, 72.2, 74.2]}}
FIG6A = {"n": [1, 8, 16, 32], "baseline": [77.5, 78.7, 80.7, 81.3], "self_debug": [80.8, 82.5, 82.9, 84.1]}
FIG6B = {"without": {"easy": 90.7, "medium": 84.5, "hard": 76.4, "extra": 63.9},
         "with": {"easy": 92.3, "medium": 85.2, "hard": 80.5, "extra": 72.9}}
FIG8A = {"Simple": {1: 88.4, 3: 89.3}, "UT": {1: 91.1, 3: 91.6}, "UT+Expl": {1: 92.0, 3: 92.5}}
FIG8B = {"Codex": {1: 80.4, 5: 90.9, 25: 96.3}, "Codex+Expl": {1: 83.0, 5: 94.3, 25: 98.0}}
TABLE4 = {"spider": [25.7, 17.1, 14.3, 11.4, 8.6, 5.8, 5.7, 5.7, 5.7],
          "transcoder": [61.9, 38.1], "mbpp": [69.2, 30.8]}


def counts_for(rep, n):
    """보고값 rep(소수 1자리, %)을 만드는 정수 정답 수 k 전부(반올림 허용 오차 0.05)."""
    r = Fraction(str(rep))
    return [k for k in range(n + 1) if abs(Fraction(100 * k, n) - r) <= Fraction(5, 100)]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, k2, n):
    """비짝 두 비율(같은 n)의 차(k2-k1)에 대한 Newcombe hybrid-score 95% 구간."""
    l1, u1 = wilson(k1, n)
    l2, u2 = wilson(k2, n)
    p1, p2 = k1 / n, k2 / n
    return (p2 - p1) - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), (p2 - p1) + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def mcnemar_bounds(a, c, n):
    """기준선 정답 a, 변형 정답 c (같은 n문항). 불일치쌍: n10(기준선만 맞음), n01(변형만 맞음), n01-n10=c-a.
    n10은 0..min(a, n-c) 안의 어떤 값도 가능. 가장 유의한 배치(p_min_case)와 가장 덜 유의한 배치(p_max_case)의 정확 McNemar p를 낸다."""
    net = c - a
    out = {"net": net}
    lo = max(0, -net)  # n01 >= 0 이므로 n10 >= -net
    hi = min(a, n - c)
    for name, n10 in (("p_min_case", lo), ("p_max_case", hi)):
        n01 = n10 + net
        tot = n01 + n10
        p = 1.0 if tot == 0 else binomtest(min(n01, n10), tot, 0.5).pvalue
        out[name] = {"n10": n10, "n01": n01, "p": p}
    return out


def main():
    res = {}

    # 1) 격자 검사
    grid = []
    def chk(label, bench, vals):
        for i, v in enumerate(vals):
            ks = counts_for(v, N[bench])
            grid.append({"where": f"{label}[{i}]", "bench": bench, "value": v, "k": ks, "ok": bool(ks)})
    for b, t in T2.items():
        for m, v in t.items():
            if m != "cols":
                chk(f"표2 {b} {m}", b, v)
    for b, t in T3.items():
        for m, v in t.items():
            if m != "cols":
                chk(f"표3 {b} {m}", b, v)
    for b, t in T5.items():
        for m, v in t.items():
            if m != "cols":
                chk(f"표5 {b} {m}", b, v)
    chk("그림6a base", "spider", FIG6A["baseline"]); chk("그림6a SD", "spider", FIG6A["self_debug"])
    chk("표1a SD", "spider", [81.3, 84.1])
    chk("표1b Codex/SD", "mbpp", [72.2, 73.6, 75.2, 75.6])
    for s, d in FIG8A.items():
        chk(f"그림8a {s}", "transcoder", list(d.values()))
    for s, d in FIG8B.items():
        chk(f"그림8b {s}", "transcoder", list(d.values()))
    res["grid"] = {"n_values": len(grid), "n_ok": sum(g["ok"] for g in grid), "failed": [g for g in grid if not g["ok"]]}

    # 2) 그림 6(b) 난이도별 -> 전체
    tot = {}
    for key in ("without", "with"):
        ks = {}
        for lvl, v in FIG6B[key].items():
            c = counts_for(v, SPIDER_HARD_N[lvl])
            ks[lvl] = c
        tot[key] = {"per_level_k": ks, "sum": sum(c[0] for c in ks.values()), "uniquely_determined": all(len(c) == 1 for c in ks.values())}
    s0, s1 = tot["without"]["sum"], tot["with"]["sum"]
    res["fig6b"] = {"counts": tot, "overall_without_pct": 100 * s0 / N["spider"], "overall_with_pct": 100 * s1 / N["spider"],
                    "matches_table2a_codex": [round(100 * s0 / N["spider"], 1) == 81.3, round(100 * s1 / N["spider"], 1) == 84.1],
                    "gain_by_level_questions": {l: tot["with"]["per_level_k"][l][0] - tot["without"]["per_level_k"][l][0] for l in SPIDER_HARD_N},
                    "level_n_total": sum(SPIDER_HARD_N.values())}
    k0, k1 = tot["without"]["per_level_k"]["extra"][0], tot["with"]["per_level_k"]["extra"][0]
    res["fig6b"]["extra_hard_mcnemar"] = mcnemar_bounds(k0, k1, SPIDER_HARD_N["extra"])
    res["fig6b"]["extra_hard_wilson"] = {"without": wilson(k0, 166), "with": wilson(k1, 166)}

    # 3) 이득 분해 (표 2): 단계별 증분(%p)
    dec = {}
    for b in ("transcoder", "mbpp", "spider"):
        dec[b] = {}
        cols = T2[b]["cols"]
        for m in ("Codex", "GPT-3.5", "GPT-4", "StarCoder"):
            v = T2[b][m]
            dec[b][m] = {"total_best_minus_base": round(max(v[1:]) - v[0], 1)}
            if b != "spider":
                dec[b][m].update({"simple_minus_base": round(v[1] - v[0], 1), "UT_minus_simple": round(v[2] - v[1], 1),
                                  "Expl_minus_UT": round(v[3] - v[2], 1), "Trace_minus_UT": round(v[4] - v[2], 1),
                                  "share_of_UT_Expl_gain_from_simple": round((v[1] - v[0]) / (v[3] - v[0]), 3)})
            else:
                dec[b][m].update({"simple_minus_base": round(v[1] - v[0], 1), "Expl_minus_base": round(v[2] - v[0], 1)})
    res["decomposition"] = dec

    # 4) 통계: Wilson, 비짝 구간, 짝지은 McNemar 가능 범위
    def kof(b, v):
        ks = counts_for(v, N[b])
        return ks[0]

    stats = []
    def add(tag, bench, model, col_a, va, col_b, vb, note=""):
        n = N[bench]
        a, c = kof(bench, va), kof(bench, vb)
        mb = mcnemar_bounds(a, c, n)
        lo, hi = newcombe(a, c, n)
        stats.append({"tag": tag, "bench": bench, "model": model, "from": col_a, "to": col_b, "a": a, "c": c, "n": n,
                      "diff_pp": round(100 * (c - a) / n, 2), "newcombe95_pp": [round(100 * lo, 1), round(100 * hi, 1)],
                      "wilson_a": [round(100 * x, 1) for x in wilson(a, n)], "wilson_c": [round(100 * x, 1) for x in wilson(c, n)],
                      "mcnemar": mb, "note": note})
    for b in ("spider", "transcoder", "mbpp"):
        cols = T2[b]["cols"]
        for m in ("Codex", "GPT-3.5", "GPT-4", "StarCoder"):
            v = T2[b][m]
            best_i = max(range(1, len(v)), key=lambda i: v[i])
            add("T2 기준선->최고 변형", b, m, cols[0], v[0], cols[best_i], v[best_i])
            if b == "spider":
                add("T2 기준선->설명", b, m, cols[0], v[0], cols[2], v[2])
            else:
                add("T2 단순->UT+Expl(설명의 몫)", b, m, cols[1], v[1], cols[3], v[3])
                add("T2 UT->UT+Expl(설명의 몫)", b, m, cols[2], v[2], cols[3], v[3])
    for b in ("transcoder", "mbpp"):
        cols = T3[b]["cols"]
        for m, v in T3[b].items():
            if m == "cols":
                continue
            add("T3 실행 없음 기준선->Trace", b, m, cols[0], v[0], cols[3], v[3])
            add("T3 실행 없음 기준선->Simple", b, m, cols[0], v[0], cols[1], v[1])
    # 표본 효율 주장
    add("그림6a n=1 기준선 -> SD(n=1)", "spider", "Codex", "n=1", 77.5, "SD n=1", 80.8)
    add("그림6a n=16 기준선 -> SD(n=1)", "spider", "Codex", "n=16", 80.7, "SD n=1", 80.8, "10배 표본 효율 주장의 근거(차이 1문항)")
    add("그림6a n=32 기준선 -> SD(n=8)", "spider", "Codex", "n=32", 81.3, "SD n=8", 82.5)
    add("그림6a n=32 기준선 -> SD(n=32)", "spider", "Codex", "n=32", 81.3, "SD n=32", 84.1)
    add("표1b MBPP n=10 -> SD UT+Expl(n=10)", "mbpp", "Codex", "n=10", 72.2, "SD n=10", 75.6)
    add("표1b MBPP greedy SD(UT+Trace) vs n=10 기준선", "mbpp", "Codex", "SD greedy", 70.8, "n=10 기준선", 72.2, "SD가 n=10 표본 기준선에 못 미침")
    add("그림8 TransCoder n=5 기준선 -> SD greedy(UT+Expl,3턴)", "transcoder", "Codex", "n=5", 90.9, "SD greedy", 92.5)
    add("그림8 TransCoder SD greedy -> n=25 기준선", "transcoder", "Codex", "SD greedy", 92.5, "n=25", 96.3, "n=25 표본(단위 테스트 필터)이 SD greedy를 앞섬")
    add("표5 Spider Codex 초기 -> GPT-3.5가 설명 디버깅", "spider", "GPT-3.5", "Codex 초기", 81.3, "GPT-3.5 Expl", 83.0, "다른 모델이 초기 코드를 디버깅")
    res["stats"] = stats

    # 5) 주장 검산
    claims = {}
    d = [round(s - b, 1) for b, s in zip(FIG6A["baseline"], FIG6A["self_debug"])]
    claims["spider_gain_by_n_pp"] = d
    claims["spider_2_3pp_claim_holds_all_n"] = all(2 <= x <= 3 for x in d)
    claims["spider_gain_in_questions_by_n"] = [counts_for(s, N["spider"])[0] - counts_for(b, N["spider"])[0] for b, s in zip(FIG6A["baseline"], FIG6A["self_debug"])]
    claims["extra_hard_gain_pp"] = round(FIG6B["with"]["extra"] - FIG6B["without"]["extra"], 1)
    claims["transcoder_best_gain_pp"] = {m: round(max(T2["transcoder"][m][1:]) - T2["transcoder"][m][0], 1) for m in ("Codex", "GPT-3.5", "GPT-4", "StarCoder")}
    claims["mbpp_best_gain_pp"] = {m: round(max(T2["mbpp"][m][1:]) - T2["mbpp"][m][0], 1) for m in ("Codex", "GPT-3.5", "GPT-4", "StarCoder")}
    claims["transcoder_final_questions_unsolved"] = {m: N["transcoder"] - counts_for(max(T2["transcoder"][m][1:]), 560)[0] for m in ("Codex", "GPT-3.5", "GPT-4", "StarCoder")}
    # 표 4: 백분율의 분모
    t4 = {}
    for name, ps in TABLE4.items():
        fits = []
        for den in range(5, 101):
            if all(any(abs(100 * k / den - p) <= 0.051 for k in range(den + 1)) for p in ps) and abs(sum(ps) - 100) < 0.11:
                fits.append(den)
        t4[name] = {"smallest_denominators": fits[:6]}
    alt = [5.7 if x == 5.8 else x for x in TABLE4["spider"]]
    t4["spider_5.8_as_5.7_denominators"] = [d for d in range(5, 101) if all(any(abs(100 * k / d - p) <= 0.051 for k in range(d + 1)) for p in alt)][:6]
    t4["spider_sum_of_reported"] = round(sum(TABLE4["spider"]), 1)
    t4["spider_5.8_vs_5.7"] = "35개 분모에서 2/35=5.71이라 5.7이어야 하는 항목이 5.8로 적힘(합 100.0을 맞추려는 반올림 조정으로 보임)"
    claims["table4_denominators"] = t4
    res["claims"] = claims

    # 6) 최소 검출 차이(MDD, 양측 α=0.05, 검정력 0.8; 정규 근사 z=2.80)
    mdd = {}
    for b, p0 in (("spider", 0.80), ("transcoder", 0.85), ("mbpp", 0.65)):
        n = N[b]
        row = {"unpaired_pp": round(100 * 2.80 * math.sqrt(2 * p0 * (1 - p0) / n), 1)}
        for d in (0.03, 0.06, 0.12, 0.20):  # 불일치쌍 비율 (n10+n01)/N
            row[f"paired_disc{int(d*100)}pct_pp"] = round(100 * 2.80 * math.sqrt(d / n), 1)
        mdd[b] = row
    res["mdd"] = mdd
    print("MDD", json.dumps(mdd, ensure_ascii=False))

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "recompute_tables_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))

    # 요약 출력
    print("격자 검사:", res["grid"]["n_ok"], "/", res["grid"]["n_values"], "실패", [g["where"] for g in res["grid"]["failed"]])
    print("그림6b 합:", s0, s1, "->", round(res["fig6b"]["overall_without_pct"], 2), round(res["fig6b"]["overall_with_pct"], 2),
          "표2a와 일치", res["fig6b"]["matches_table2a_codex"], "난이도별 이득(문항)", res["fig6b"]["gain_by_level_questions"],
          "분모 합", res["fig6b"]["level_n_total"], "유일 결정", tot["without"]["uniquely_determined"], tot["with"]["uniquely_determined"])
    print("extra-hard McNemar:", res["fig6b"]["extra_hard_mcnemar"])
    for s in stats:
        m = s["mcnemar"]
        print(f'{s["tag"]:48s} {s["bench"]:10s} {s["model"]:9s} {s["from"]:>10s}->{s["to"]:<10s} {s["diff_pp"]:+5.1f}pp net={m["net"]:+4d} '
              f'N95={s["newcombe95_pp"]} p_min={m["p_min_case"]["p"]:.2g} (n10={m["p_min_case"]["n10"]}) p_max={m["p_max_case"]["p"]:.2g} (n10={m["p_max_case"]["n10"]})')
    print(json.dumps(res["claims"], ensure_ascii=False))
    print(json.dumps(res["decomposition"], ensure_ascii=False))


if __name__ == "__main__":
    main()
