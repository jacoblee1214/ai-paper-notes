"""논문의 편향 표(표 2, 3, 4, 9–12, 15, 6)의 산술 검산과 Wilson 구간, Fisher 정확 검정.

논문 표의 숫자는 손으로 옮겼다(쪽·표 번호는 주석). 분모는 표의 퍼센트 단위에서 역으로 구한다:
  표 2·9·11·12·15: 0.1%p 단위 반올림 값이 k/80과 맞으면 n=80(첫 턴 질문 80개).
  표 3: 23개 답. 표 4: 10 문항 x 위치 2 = 20.
GPT-4 쌍 비교 판정 파일(gpt4_pair.parquet)로는 표 11의 세 모델 쌍 일관성을 직접 비교한다.
출력: analysis/bias_tables_results.json
"""
import json
import os

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def recover_k(p, n):
    """퍼센트 p(0.1%p 단위 반올림)에서 n개 중 k를 찾는다. 맞는 k가 없으면 None."""
    ks = [k for k in range(n + 1) if round(k / n * 100, 1) == round(p, 1)]
    return ks[0] if ks else None


# 표 2 (본문 5쪽), 표 9 (부록 D.1, 25쪽): 행 = (판정자, 프롬프트) -> (일관, 첫 번째 편향, 두 번째 편향, 오류)
T2 = {
    ("claude-v1", "default"): (23.8, 75.0, 0.0, 1.2), ("claude-v1", "rename"): (56.2, 11.2, 28.7, 3.8),
    ("gpt-3.5-turbo", "default"): (46.2, 50.0, 1.2, 2.5), ("gpt-3.5-turbo", "rename"): (51.2, 38.8, 6.2, 3.8),
    ("gpt-4", "default"): (65.0, 30.0, 5.0, 0.0), ("gpt-4", "rename"): (66.2, 28.7, 5.0, 0.0),
}
T9_EXTRA = {
    ("claude-v1", "score"): (20.0, 80.0, 0.0, 0.0), ("claude-v1", "short"): (22.5, 75.0, 2.5, 0.0),
    ("gpt-3.5-turbo", "score"): (55.0, 33.8, 11.2, 0.0), ("gpt-3.5-turbo", "short"): (38.8, 57.5, 3.8, 0.0),
    ("gpt-4", "score"): (51.2, 46.2, 2.5, 0.0), ("gpt-4", "short"): (62.5, 35.0, 2.5, 0.0),
}
# 표 10 (부록 D.1): 범주별 (일관, 첫 번째, 두 번째)
T10 = {"writing": (42.0, 46.0, 12.0), "roleplay": (68.0, 30.0, 2.0), "reasoning": (76.0, 20.0, 4.0),
       "math": (86.0, 4.0, 10.0), "coding": (86.0, 14.0, 0.0), "extraction": (78.0, 12.0, 10.0),
       "stem": (44.0, 54.0, 2.0), "humanities": (36.0, 60.0, 4.0)}
# 표 11: GPT-4 default 프롬프트, 모델 쌍별 (일관, 첫 번째, 두 번째)
T11 = {"GPT-3.5 vs Claude-V1": (67.5, 23.8, 8.8), "GPT-3.5 vs Vicuna-13B": (73.8, 23.8, 2.5), "GPT-3.5 vs LLaMA-13B": (98.8, 1.2, 0.0)}
# 표 12: 영샷 대 퓨샷 (일관, 첫, 둘째, 오류)
T12 = {("claude-v1", "zero"): (23.8, 75.0, 0.0, 1.2), ("claude-v1", "few"): (63.7, 21.2, 11.2, 3.8),
       ("gpt-3.5-turbo", "zero"): (46.2, 50.0, 1.2, 2.5), ("gpt-3.5-turbo", "few"): (55.0, 16.2, 28.7, 0.0),
       ("gpt-4", "zero"): (65.0, 30.0, 5.0, 0.0), ("gpt-4", "few"): (77.5, 10.0, 12.5, 0.0)}


def table_rows(table, n, names):
    rows = {}
    for key, vals in table.items():
        r = {"n": n}
        ks = []
        for nm, v in zip(names, vals):
            k = recover_k(v, n)
            ks.append(k)
            r[nm] = {"paper_pct": v, "k": k, "wilson95": wilson(k, n) if k is not None else None}
        r["pct_sum"] = round(sum(vals), 1)
        r["k_sum"] = sum(k for k in ks if k is not None) if all(k is not None for k in ks) else None
        r["all_k_recovered"] = all(k is not None for k in ks)
        rows[" / ".join(key) if isinstance(key, tuple) else key] = r
    return rows


def main():
    out = {}
    names4 = ["consistent", "biased_first", "biased_second", "error"]
    out["table2_9"] = table_rows({**T2, **T9_EXTRA}, 80, names4)
    out["table12"] = table_rows({" ".join(k): v for k, v in T12.items()}, 80, names4)
    out["table11"] = table_rows(T11, 80, names4[:3])

    # 표 10: 범주당 n을 모르므로 퍼센트 단위에서 가능한 n을 찾는다
    t10 = {}
    for n in (10, 20, 25, 50, 100):
        ok = all(recover_k(v, n) is not None for vals in T10.values() for v in vals)
        t10[f"n_per_category={n}_consistent_with_percentages"] = ok
    cons = [v[0] for v in T10.values()]
    t10["mean_consistency_over_categories"] = round(float(np.mean(cons)), 2)
    t10["note"] = "범주당 10문항이면 10%p 단위여야 한다. 표 10의 값은 2%p 단위라 범주당 n=50이 필요하고, 합계 400은 표 2의 n=80과 맞지 않는다."
    t10["wilson_at_n50"] = {c: wilson(round(v[0] / 2), 50) for c, v in T10.items()}
    t10["wilson_at_n10_if_questions"] = {c: wilson(round(v[0] / 10), 10) for c, v in T10.items()}
    out["table10"] = t10

    # 표 2 핵심 대조 (질문 80개를 같이 쓴 반복 측정이라 정확한 쌍 검정은 못 한다: 비쌍 Fisher는 근사)
    cmp_ = {}
    def kk(j, p, i=0):
        return recover_k({**T2, **T9_EXTRA}[(j, p)][i], 80)
    cmp_["claude default vs rename 일관성"] = fisher(kk("claude-v1", "default"), 80, kk("claude-v1", "rename"), 80)
    cmp_["gpt-4 default vs rename 일관성"] = fisher(kk("gpt-4", "default"), 80, kk("gpt-4", "rename"), 80)
    cmp_["gpt-4 vs gpt-3.5 default 일관성"] = fisher(kk("gpt-4", "default"), 80, kk("gpt-3.5-turbo", "default"), 80)
    cmp_["gpt-4 default vs few-shot 일관성(표 12)"] = fisher(recover_k(65.0, 80), 80, recover_k(77.5, 80), 80)
    cmp_["gpt-4 default vs score 프롬프트 일관성"] = fisher(kk("gpt-4", "default"), 80, kk("gpt-4", "score"), 80)
    out["table2_fisher_unpaired_p"] = {k: round(v, 4) for k, v in cmp_.items()}

    # 표 3 장황함 공격 (23개 답) — 8.7% = 2/23, 91.3% = 21/23
    t3 = {}
    for j, fr in {"Claude-v1": 91.3, "GPT-3.5": 91.3, "GPT-4": 8.7}.items():
        k = [k for k in range(24) if round(k / 23 * 100, 1) == fr][0]
        t3[j] = {"fail_k": k, "n": 23, "pct": round(k / 23 * 100, 1), "wilson95": wilson(k, 23)}
    t3["gpt4_vs_gpt35_fisher_p"] = float(f"{fisher(2, 23, 21, 23):.3g}")
    out["table3_repetitive_list"] = t3

    # 표 4 수학 채점 실패 (10문항 x 2 위치 = 20 판정)
    t4 = {}
    for nm, k in {"default": 14, "cot": 6, "reference": 3}.items():
        t4[nm] = {"k": k, "n": 20, "pct": k / 20 * 100, "wilson95_n20": wilson(k, 20),
                  "wilson95_if_only_10_independent": wilson(round(k / 2), 10)}
    t4["default_vs_reference_fisher_p_n20"] = round(fisher(14, 20, 3, 20), 5)
    t4["default_vs_cot_fisher_p_n20"] = round(fisher(14, 20, 6, 20), 4)
    t4["cot_vs_reference_fisher_p_n20"] = round(fisher(6, 20, 3, 20), 4)
    out["table4_math_failure"] = t4

    # 표 6 Chatbot Arena S2: GPT-4와 다른 LLM 판정자의 사람 일치
    arena = {"G4": (0.87, 1944), "G4-S": (0.85, 1761), "G3.5": (0.83, 1567), "Claude": (0.84, 1475)}
    a = {}
    for j, (p, n) in arena.items():
        k = round(p * n)
        a[j] = {"rate": p, "n": n, "wilson95": wilson(k, n)}
    a["G4_vs_G35_diff_fisher_p"] = round(fisher(round(0.87 * 1944), 1944, round(0.83 * 1567), 1567), 5)
    a["S1_G4_vs_human"] = {"rate": 0.64, "n": 3066, "wilson95": wilson(round(0.64 * 3066), 3066)}
    a["note"] = "부록 F의 '(GPT-4) 66%와 87%'에서 66%는 표 6의 G4-사람 S1(64%)과 다르다. 표 6의 66%는 G4-G3.5·G4-Claude 칸이다."
    a["finetuned_vicuna_S2_vs_G4"] = {"vicuna_ft": 0.855, "G4": 0.87}
    out["table6_arena"] = a

    # 표 11을 GPT-4 쌍 비교 판정 파일(HF)의 첫 턴으로 직접 비교
    raw = pq.read_table(f"{ROOT}/data/hf_mtbench/gpt4_pair.parquet", columns=["question_id", "model_a", "model_b", "winner", "turn"]).to_pylist()
    pairs = {"GPT-3.5 vs Claude-V1": ("claude-v1", "gpt-3.5-turbo"), "GPT-3.5 vs Vicuna-13B": ("gpt-3.5-turbo", "vicuna-13b-v1.2"),
             "GPT-3.5 vs LLaMA-13B": ("gpt-3.5-turbo", "llama-13b")}
    t11 = {}
    for nm, (x, y) in pairs.items():
        for turn in (1, 2):
            rows = [r for r in raw if r["turn"] == turn and {r["model_a"], r["model_b"]} == {x, y}]
            inc = sum(r["winner"] == "tie (inconsistent)" for r in rows)
            t11[f"{nm} turn{turn}"] = {"n": len(rows), "consistent_rate": round(1 - inc / len(rows), 3), "paper_table11": T11[nm][0],
                                       "wilson95": wilson(len(rows) - inc, len(rows))}
    t11["note"] = "논문 표 11은 첫 질문 80개이며 모델 답은 같지만 판정 실행·모델 버전(vicuna-13b-v1.2)이 다를 수 있다. 이 파일의 '일관'은 두 순서 결과가 같은 경우(동점 포함)로 가정한 값."
    out["table11_vs_released_gpt4_pair"] = t11

    with open(f"{ROOT}/analysis/bias_tables_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # 요약
    for k, v in out["table2_9"].items():
        print(k, "k_sum", v["k_sum"], "recovered", v["all_k_recovered"], "consistent", v["consistent"]["k"], v["consistent"]["wilson95"])
    print(out["table10"])
    print(out["table2_fisher_unpaired_p"])
    print(json.dumps(out["table3_repetitive_list"], ensure_ascii=False))
    print(json.dumps(out["table4_math_failure"], ensure_ascii=False))
    print(json.dumps(out["table6_arena"], ensure_ascii=False))
    print(json.dumps(out["table11_vs_released_gpt4_pair"], ensure_ascii=False, indent=0))
    print({k: (v["all_k_recovered"], v["k_sum"]) for k, v in out["table11"].items()}, {k: (v["all_k_recovered"], v["k_sum"]) for k, v in out["table12"].items()})


if __name__ == "__main__":
    main()
