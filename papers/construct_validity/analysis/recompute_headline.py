"""논문 §4·§5·그림 3·그림 2의 집계 비율을 공개 코딩 자료로 재계산한다.

입력: data/hf/LLM_benchmark_codebook.csv (HF ambean/construct-validity-review, CC BY-SA 4.0, 455행 = 445 벤치마크 + 중복 코딩 10행)
분모를 세 가지로 비교한다.
  all455   : 행 455개 전부 (저자 노트북 coding_results.ipynb가 쓴 방식)
  first445 : bibkey별 첫 행만 (논문이 말하는 445개)
  last445  : bibkey별 마지막 행만 (중복 행의 코딩이 달라서 결과가 흔들리는지 보려는 민감도)
각 비율에 Wilson 95% 구간을 붙인다. 표준 라이브러리 + scipy 없이 직접 계산한다.
실행: python3 -I analysis/recompute_headline.py   (저장소 폴더 루트에서)
"""
import ast
import csv
import collections
import json
import math
import re

csv.field_size_limit(10**9)
SRC = "data/hf/LLM_benchmark_codebook.csv"
OUT = "analysis/recompute_headline_results.json"


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def lst(s):
    """'_clean' 열은 파이썬 리스트 문자열이다."""
    s = (s or "").strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        return [x for x in v if x != ""] if isinstance(v, list) else [str(v)]
    except Exception:
        return [s]


rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
assert len(rows) == 455
by = collections.defaultdict(list)
for r in rows:
    by[r["bibkey"]].append(r)
first = [v[0] for v in by.values()]
last = [v[-1] for v in by.values()]
assert len(first) == 445
DENOMS = {"all455": rows, "first445": first, "last445": last}


# ---- 술어 ----
def has(col, label):
    return lambda r: label in lst(r[col])


def only(col, label):
    return lambda r: lst(r[col]) == [label]


def anyof(col, labels):
    return lambda r: any(x in labels for x in lst(r[col]))


def onlyof(col, labels):
    return lambda r: len(lst(r[col])) > 0 and all(x in labels for x in lst(r[col]))


REUSE = {"Another benchmark", "Human exams"}
ITEMS = [
    # (키, 논문 위치, 논문 값(%), 술어)
    ("defined", "§4 Phenomenon", 78.2, lambda r: r["phenomenon_defined"] == "Yes"),
    ("not_defined(No)", "Fig3 Undefined", 21.7, lambda r: r["phenomenon_defined"] == "No"),
    ("defined&widely", "Fig3 Widely-agreed", 40.9,
     lambda r: r["phenomenon_defined"] == "Yes" and "Widely-agreed" in lst(r["phenomenon_contested_clean"])),
    ("defined&contested", "Fig3 Contested", 37.4,
     lambda r: r["phenomenon_defined"] == "Yes" and "Contested" in lst(r["phenomenon_contested_clean"])),
    ("composite", "§4 composite", 61.2, lambda r: r["definition_integrity"].startswith("Composite")),
    ("single_whole", "§4 single whole", 36.5, lambda r: r["definition_integrity"].startswith("Single")),
    ("eco_constructed_any", "§4 constructed", 40.7, has("task_ecology_clean", "Constructed")),
    ("eco_constructed_only", "§4 only constructed", 28.5, only("task_ecology_clean", "Constructed")),
    ("eco_partial_any", "§4 partial real", 32.3, has("task_ecology_clean", "Partial")),
    ("eco_representative_any", "§4 representative", 36.9, has("task_ecology_clean", "Representative")),
    ("eco_complete_any", "§4 complete real (<10%)", 10.0, has("task_ecology_clean", "Complete")),
    ("src_single", "§4 single source", 33.6, lambda r: len(lst(r["task_source_clean"])) == 1),
    ("src_author_crafted", "§4 handcrafted", 43.3, has("task_source_clean", "Author-crafted")),
    ("src_another_benchmark", "§4 reuse benchmark", 42.6, has("task_source_clean", "Another benchmark")),
    ("src_llm_generated", "§4 LLM-generated", 31.2, has("task_source_clean", "LLM-generated")),
    ("src_human_exams", "§4 human exams", 38.2, has("task_source_clean", "Human exams")),
    ("src_only_another_benchmark", "§4 only another benchmark", 7.7, only("task_source_clean", "Another benchmark")),
    ("src_reuse_any(bench|exam)", "Fig3 Partial+Only reused", 48.1, anyof("task_source_clean", REUSE)),
    ("src_reuse_only", "Fig3 Only reused", 9.9, onlyof("task_source_clean", REUSE)),
    ("src_reuse_partial", "Fig3 Partially reused", 38.2,
     lambda r: anyof("task_source_clean", REUSE)(r) and not onlyof("task_source_clean", REUSE)(r)),
    ("src_not_reused", "Fig3 Not reused", 51.9, lambda r: not anyof("task_source_clean", REUSE)(r)),
    ("src_synthetic(LLM|proc)", "§5.3 synthetic", 47.5, anyof("task_source_clean", {"LLM-generated", "Procedurally-generated"})),
    ("conv_only", "Fig3 Only convenience", 12.3, only("dataset_sampling_method_clean", "Convenience")),
    ("conv_any", "§4/§5.3 convenience (27.0 vs 39.3)", 27.0, has("dataset_sampling_method_clean", "Convenience")),
    ("conv_partial", "Fig3 Partial convenience", 27.0,
     lambda r: "Convenience" in lst(r["dataset_sampling_method_clean"]) and lst(r["dataset_sampling_method_clean"]) != ["Convenience"]),
    ("conv_none", "Fig3 No convenience", 60.7, lambda r: "Convenience" not in lst(r["dataset_sampling_method_clean"])),
    ("samp_targeted_any", "§4 targeted", 55.2, has("dataset_sampling_method_clean", "Targeted")),
    ("samp_criterion_any", "§4 criterion", 46.2, has("dataset_sampling_method_clean", "Criterion")),
    ("samp_random_any", "§4/§5.3 random", 17.1, has("dataset_sampling_method_clean", "Random")),
    ("samp_targeted_only", "§4 only targeted", 17.5, only("dataset_sampling_method_clean", "Targeted")),
    ("samp_criterion_only", "§4 only criterion", 10.8, only("dataset_sampling_method_clean", "Criterion")),
    ("samp_random_only", "§4 only random", 7.0, only("dataset_sampling_method_clean", "Random")),
    ("fmt_free_any", "§4 free response", 46.4, has("response_format_clean", "Free response")),
    ("fmt_mc_any", "§4 multiple choice", 40.0, has("response_format_clean", "Multiple choice")),
    ("fmt_short_any", "§4 short free", 38.5, has("response_format_clean", "Short free response")),
    ("fmt_structured_any", "§4/§5.2 structured", 21.1, has("response_format_clean", "Structured")),
    ("fmt_free_only", "§4 only free", 17.8, only("response_format_clean", "Free response")),
    ("fmt_mc_only", "§4 only MC", 18.5, only("response_format_clean", "Multiple choice")),
    ("met_exact_any", "§4 exact match", 81.3, has("metric_definition_clean", "Exact match")),
    ("met_exact_only", "§4 only exact match", 40.7, only("metric_definition_clean", "Exact match")),
    ("met_soft_any", "§4 soft match", 20.9, has("metric_definition_clean", "Soft match")),
    ("met_judge_any", "§4 LLM-as-a-judge", 17.1, has("metric_definition_clean", "LLM-as-a-Judge")),
    ("met_human_any", "§4 human ratings", 13.0, has("metric_definition_clean", "Human ratings")),
    ("stats(Std|Tests)", "§4/Fig3 statistics", 16.0, anyof("metric_statistics_clean", {"Std", "Tests"})),
    ("stats(Tests only)", "(참고) 검정만", None, has("metric_statistics_clean", "Tests")),
    ("validity_discussed", "§4/Fig3 validity", 53.4, has("results_author_validity_clean", "Yes")),
    ("cmp_other_benchmarks", "§4 comparison", 35.2, lambda r: r["results_comparison"] == "Yes"),
    ("human_baseline", "§4 human baseline", 32.4, lambda r: r["results_human_baseline"] == "Yes"),
    ("realism_yes_or_itself", "§4 realistic setting", 31.2,
     lambda r: set(lst(r["results_realism_clean"])) & {"Comparison made", "Realistic"} != set()),
    ("realism_comparison_made_only", "(참고) 실제 비교를 한 것만", None, has("results_realism_clean", "Comparison made")),
    ("realism_benchmark_itself", "(참고) 벤치마크가 이미 현실적", None, has("results_realism_clean", "Realistic")),
    ("tax_reasoning", "§4 reasoning", 18.5, lambda r: r["phenomenon_taxonomy_root"] == "Reasoning"),
    ("tax_alignment", "§4 alignment", 8.1, lambda r: r["phenomenon_taxonomy_root"] == "Alignment"),
    ("tax_code", "§4 code generation", 5.7, lambda r: r["phenomenon_taxonomy_root"] == "Code Generation"),
]

res = {"n_rows": len(rows), "n_unique_bibkeys": len(first), "items": {}}
print(f"{'item':32s} {'paper':>6s} | " + " | ".join(f"{d:^22s}" for d in DENOMS))
for key, loc, paper, pred in ITEMS:
    ent = {"where": loc, "paper_pct": paper}
    cells = []
    for dn, rs in DENOMS.items():
        k = sum(1 for r in rs if pred(r))
        n = len(rs)
        lo, hi = wilson(k, n)
        ent[dn] = {"k": k, "n": n, "pct": round(100 * k / n, 2), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
        cells.append(f"{k:3d}/{n} {100*k/n:5.1f} [{100*lo:4.1f},{100*hi:4.1f}]")
    p = ent["all455"]["pct"]
    q = ent["first445"]["pct"]
    if paper is not None:
        ent["matches_all455"] = abs(p - paper) < 0.06
        ent["matches_first445"] = abs(q - paper) < 0.06
    if paper is not None:
        k455 = ent["all455"]["k"]
        # 논문 값을 반올림으로 재현하는 분모 (440..460 중). 분자는 455행 기준
        ent["implied_denominators_for_k455"] = [n for n in range(440, 461) if round(100 * k455 / n, 1) == paper]
        k445 = ent["first445"]["k"]
        ent["implied_denominators_for_k445"] = [n for n in range(440, 461) if round(100 * k445 / n, 1) == paper]
    res["items"][key] = ent
    pv = "  -  " if paper is None else f"{paper:5.1f}"
    print(f"{key:32s} {pv:>6s} | " + " | ".join(cells))

# ---- 원본(자유서술 포함) 응답 기준 보조 계산 ----
raw = {}
def rawcount(col, val):
    k = sum(1 for r in rows if r[col] == val)
    lo, hi = wilson(k, 455)
    return {"k": k, "n": 455, "pct": round(100 * k / 455, 2), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
raw["results_realism=='Yes'(실제 비교)"] = rawcount("results_realism", "Yes")
raw["results_realism=='The benchmark is itself realistic'"] = rawcount("results_realism", "The benchmark is itself realistic")
raw["results_author_validity=='Yes'(엄격)"] = rawcount("results_author_validity", "Yes")
raw["phenomenon_defined=='Yes'"] = rawcount("phenomenon_defined", "Yes")
res["raw_strict"] = raw
print("\n원본 응답 엄격 계산", json.dumps(raw, ensure_ascii=False, indent=1))

# ---- 정의 여부와 contested의 모순 (부록 B: Contested 225는 정의 없다고 답한 행도 포함) ----
cross = collections.Counter()
for r in rows:
    cross[(r["phenomenon_defined"] or "(blank)", ";".join(lst(r["phenomenon_contested_clean"])) or "(blank)")] += 1
res["defined_x_contested_all455"] = {f"{a} | {b}": n for (a, b), n in sorted(cross.items())}
print("\n정의 여부 x contested (455행)")
for (a, b), n in sorted(cross.items()):
    print(f"  defined={a:8s} contested={b:16s} {n}")

# ---- 그림 3의 음영: 다섯 항목을 모두 모범 사례로 만족하는 벤치마크 ----
def best5(r):
    return (r["phenomenon_defined"] == "Yes" and "Widely-agreed" in lst(r["phenomenon_contested_clean"])
            and "Convenience" not in lst(r["dataset_sampling_method_clean"])
            and not anyof("task_source_clean", REUSE)(r)
            and bool(set(lst(r["metric_statistics_clean"])) & {"Std", "Tests"})
            and "Yes" in lst(r["results_author_validity_clean"]))


res["best5"] = {}
for dn, rs in DENOMS.items():
    k = sum(1 for r in rs if best5(r))
    lo, hi = wilson(k, len(rs))
    res["best5"][dn] = {"k": k, "n": len(rs), "pct": round(100 * k / len(rs), 2), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
print("\n다섯 항목 모두 모범 사례(그림 3 음영):", res["best5"])
# 항목 수별 분포 (몇 개를 만족하는가)
def n_ok(r):
    return sum([
        r["phenomenon_defined"] == "Yes" and "Widely-agreed" in lst(r["phenomenon_contested_clean"]),
        "Convenience" not in lst(r["dataset_sampling_method_clean"]),
        not anyof("task_source_clean", REUSE)(r),
        bool(set(lst(r["metric_statistics_clean"])) & {"Std", "Tests"}),
        "Yes" in lst(r["results_author_validity_clean"]),
    ])
dist = collections.Counter(n_ok(r) for r in first)
res["n_best_practices_met_first445"] = {str(k): dist[k] for k in sorted(dist)}
print("충족 개수 분포(445):", dict(sorted(dist.items())))

# ---- 연도별 (그림 2B): bibkey 끝 4자리를 연도로 쓴다 ----
def year(r):
    m = re.search(r"(20\d\d)[a-z]?$", r["bibkey"])
    return int(m.group(1)) if m else None


yr = collections.Counter(year(r) for r in first)
yv = collections.Counter(year(r) for r in first if "Yes" in lst(r["results_author_validity_clean"]))
res["by_year_first445"] = {str(y): {"n": yr[y], "validity_discussed": yv[y],
                                    "pct": round(100 * yv[y] / yr[y], 1)} for y in sorted(k for k in yr if k)}
print("\n연도별 (n, 타당성 논의):", {y: (yr[y], yv[y]) for y in sorted(k for k in yr if k)}, "연도 미파싱", yr.get(None, 0))

# ---- 중복 행: 같은 bibkey에서 코딩이 서로 다른 항목 ----
dups = {k: v for k, v in by.items() if len(v) > 1}
diff = collections.Counter()
for k, v in dups.items():
    for col in ["phenomenon_defined", "phenomenon_contested_clean", "results_author_validity_clean", "metric_statistics_clean", "dataset_sampling_method_clean", "task_source_clean"]:
        if len({x[col] for x in v}) > 1:
            diff[col] += 1
res["duplicates"] = {"bibkeys": len(dups), "extra_rows": sum(len(v) - 1 for v in dups.values()), "disagree_by_column": dict(diff)}
print("\n중복 bibkey", len(dups), "추가 행", sum(len(v) - 1 for v in dups.values()), "열별 불일치", dict(diff))

# ---- 검토자별 부담 (이름은 저장하지 않고 분포만) ----
cc = list(csv.DictReader(open("data/gh/clean_codebook.csv", encoding="utf-8")))
load = sorted(collections.Counter(r["main_coder"] for r in cc if r["main_coder"]).values(), reverse=True)
res["reviewer_load_clean_codebook"] = {"n_reviewers": len(load), "sorted_loads": load, "total": sum(load)}
print("\n검토자 수", len(load), "상위 부담", load[:5], "합", sum(load))

# ---- 자유서술 키워드: 권고 5(오염), 7(오류 분석)은 코드북 항목이 없다. 참고용 키워드 적중 수 ----
FREE = ["short_summary", "contribution", "purpose_extra", "task_definition_detail", "task_source_detail",
        "procedural_extra", "notes_extra", "results_author_validity_detail", "task_ecology_detail",
        "task_dataset_size_detail", "metric_definition_detail"]
kw = {"contamination": r"contaminat|leak", "error analysis": r"error analysis|failure mode|error type", "held-out": r"held[- ]out|hold[- ]out|private test"}
res["freetext_keyword_hits_first445"] = {}
for name, pat in kw.items():
    k = sum(1 for r in first if re.search(pat, " ".join(r[c] for c in FREE), re.I))
    lo, hi = wilson(k, 445)
    res["freetext_keyword_hits_first445"][name] = {"k": k, "pct": round(100 * k / 445, 1), "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}
print("\n자유서술 키워드(코딩이 아니라 참고):", res["freetext_keyword_hits_first445"])

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved", OUT)
