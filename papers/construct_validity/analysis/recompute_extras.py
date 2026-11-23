"""보조 계산: (1) 시험 세트 크기 분포와 표본 오차, (2) 연도별 타당성 논의 비율의 추세 검정.
입력: data/hf/LLM_benchmark_codebook.csv (bibkey별 첫 행 = 445편)
실행: python3 -I analysis/recompute_extras.py  (논문 폴더 루트에서)
"""
import ast, csv, collections, json, math, re
import numpy as np
from scipy import stats

csv.field_size_limit(10**9)
rows = list(csv.DictReader(open("data/hf/LLM_benchmark_codebook.csv", encoding="utf-8")))
first = {}
for r in rows:
    first.setdefault(r["bibkey"], r)
first = list(first.values())
res = {}

# (1) 시험 세트 크기: 숫자만 있는 응답(쉼표, 'items' 등 단어 허용)만 파싱한다. 여러 하위 과제나 서술형은 제외.
pat = re.compile(r"^\s*~?\s*([\d][\d,\.]*)\s*(k|K)?\s*(items|examples|samples|questions|instances|problems|test)?\s*$")
sizes = []
for r in first:
    m = pat.match(r["task_dataset_size"].replace(" ", " ").replace("\xa0", " "))
    if m:
        txt = m.group(1).replace(",", "")
        try:
            v = float(txt) * (1000 if m.group(2) else 1)
        except ValueError:
            continue
        sizes.append(v)
sizes = np.array(sorted(sizes))
n = len(sizes)
res["test_size"] = {
    "n_parsed": n, "n_total": len(first), "n_blank": sum(1 for r in first if not r["task_dataset_size"].strip()),
    "median": float(np.median(sizes)), "q25": float(np.percentile(sizes, 25)), "q75": float(np.percentile(sizes, 75)),
    "share_le_200": round(float((sizes <= 200).mean()), 3), "share_le_500": round(float((sizes <= 500).mean()), 3),
    "share_le_1000": round(float((sizes <= 1000).mean()), 3),
    "halfwidth_pp_at_half_for_n": {str(k): round(100 * 1.96 * math.sqrt(0.25 / k), 1) for k in (100, 200, 500, 1000, 2000)},
    "note": "숫자만 적힌 응답만 파싱(서술형·하위 과제별 크기 제외). 표본 오차는 정확도 0.5 근처의 단일 점수의 95% 반폭(%p).",
}
print("test size", json.dumps(res["test_size"], ensure_ascii=False))

# (2) 연도별 '타당성 논의' 비율
def lst(s):
    s = (s or "").strip()
    if not s: return []
    try:
        v = ast.literal_eval(s); return v if isinstance(v, list) else [str(v)]
    except Exception:
        return [s]
by = collections.defaultdict(lambda: [0, 0])
for r in first:
    y = int(re.search(r"(20\d\d)[a-z]?$", r["bibkey"]).group(1))
    by[y][0] += 1
    by[y][1] += "Yes" in lst(r["results_author_validity_clean"])
res["validity_by_year"] = {str(y): {"n": a, "yes": b, "pct": round(100 * b / a, 1)} for y, (a, b) in sorted(by.items())}
# 2022-2024만(연도별 n>=30) 3x2 카이제곱, 그리고 연도를 순서 변수로 쓴 로지스틱 대신 Cochran-Armitage 추세(수식 직접)
yrs = [2022, 2023, 2024]
tab = np.array([[by[y][1], by[y][0] - by[y][1]] for y in yrs])
chi2, p, dof, _ = stats.chi2_contingency(tab)
x = np.array(yrs, float); nn = tab.sum(1); yy = tab[:, 0]
pbar = yy.sum() / nn.sum()
T = (x * (yy - nn * pbar)).sum()
var = pbar * (1 - pbar) * ((nn * x**2).sum() - (nn * x).sum() ** 2 / nn.sum())
z = T / math.sqrt(var); pz = 2 * (1 - stats.norm.cdf(abs(z)))
res["validity_trend_2022_2024"] = {"table_yes_no": tab.tolist(), "chi2": round(float(chi2), 3), "dof": int(dof), "p": round(float(p), 3),
                                    "cochran_armitage_z": round(float(z), 3), "p_trend": round(float(pz), 3)}
print("validity by year", res["validity_by_year"]); print("trend", res["validity_trend_2022_2024"])
# (3) 하위 요소를 가진(복합) 현상 중 하위 요소를 따로 재는 비율 (권고 1의 세 번째 항목)
comp = [r for r in first if r["definition_integrity"].startswith("Composite")]
cd = collections.Counter(r["definition_integrity_detail"] or "(blank)" for r in comp)
res["composite_measured_separately"] = {"n_composite": len(comp), "counts": dict(cd), "yes_pct": round(100 * cd["Yes"] / len(comp), 1)}
print("composite", res["composite_measured_separately"])

# (4) 코드북 안의 중복 bibkey 9개(IRR 41쌍과 겹치지 않음)의 항목별 일치
allrows = list(csv.DictReader(open("data/hf/LLM_benchmark_codebook.csv", encoding="utf-8")))
g = collections.defaultdict(list)
for r in allrows: g[r["bibkey"]].append(r)
dup = {k: v for k, v in g.items() if len(v) > 1}
merged_keys = {r["bibkey"] for r in csv.DictReader(open("data/gh/merged_annotations.csv", encoding="utf-8"))}
agree = {}
for col in ["phenomenon_defined", "phenomenon_contested_clean", "definition_integrity", "results_author_validity_clean", "task_source_clean", "dataset_sampling_method_clean", "metric_statistics_clean", "task_ecology_clean"]:
    agree[col] = sum(1 for v in dup.values() if v[0][col] == v[1][col])
res["codebook_duplicates"] = {"n_bibkeys": len(dup), "overlap_with_merged_annotations": len(set(dup) & merged_keys), "agree_count_of_9": agree}
print("dups", res["codebook_duplicates"])
json.dump(res, open("analysis/recompute_extras_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
