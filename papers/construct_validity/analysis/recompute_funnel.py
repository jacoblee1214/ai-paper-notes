"""PRISMA식 선별 흐름(그림 1, 그림 4, 표 7, 표 8)의 산술과 키워드 선별 재현.

입력: data/hf/included_papers_r2.csv  (키워드 통과 2,189편과 LLM 선별 열)
      data/gh/final_list.csv          (LLM 선별 후 522편)
      data/gh/manual_subset.csv       (사람이 직접 라벨한 50편)
      data/gh/papers_dataset/*        (저자가 모은 학회 원본 목록, ICML 없음)
      data/hf/LLM_benchmark_codebook.csv (최종 445편)
실행: python3 -I analysis/recompute_funnel.py  (논문 폴더 루트에서; lxml 없이 xml.etree 사용)
"""
import collections
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET

csv.field_size_limit(10**9)
OUT = "analysis/recompute_funnel_results.json"
res = {}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1))


T = lambda s: " ".join(s.lower().split())
r2 = list(csv.DictReader(open("data/hf/included_papers_r2.csv", encoding="utf-8")))
final = list(csv.DictReader(open("data/gh/final_list.csv", encoding="utf-8")))
man = list(csv.DictReader(open("data/gh/manual_subset.csv", encoding="utf-8")))

# ---- 표 7 산술 ----
steps = [("keyword", 2189, None), ("not a benchmark", 1251, 938), ("modality", 92, 846), ("not primarily benchmark", 324, 522)]
chk = []
rem = 2189
for name, ex, left in steps[1:]:
    rem -= ex
    chk.append({"step": name, "excluded": ex, "remaining_paper": left, "remaining_ours": rem, "ok": rem == left})
res["table7_arithmetic"] = chk
print("표 7 산술", chk)

# ---- 공개 열로 표 7 단계 재구성 ----
c = collections.Counter((x["inclusion"], x["modality"], x["contribution"]) for x in r2)
res["r2_label_counts"] = {"|".join(k): v for k, v in sorted(c.items())}
step1_ex = sum(v for (i, m, ct), v in c.items() if i != "Yes")
mod_ex = sum(v for (i, m, ct), v in c.items() if i == "Yes" and m in ("Video", "Audio", "Other", "Image"))
rest = sum(v for (i, m, ct), v in c.items() if i == "Yes" and m in ("Language", "Multimodal"))
res["rebuild"] = {"step1_excluded_ours": step1_ex, "step2_excluded_if_Video_Audio_Other_Image": mod_ex,
                  "after_step2_Language_Multimodal": rest,
                  "Image_only_excluded": c[("Yes", "Image", "")],
                  "Language_by_contribution": {k[2]: v for k, v in c.items() if k[0] == "Yes" and k[1] == "Language"},
                  "Multimodal": c[("Yes", "Multimodal", "")]}
rm = {T(x["title"]): x for x in r2}
fin = collections.Counter()
for x in final:
    y = rm.get(T(x["title"]))
    fin[(y["modality"], y["contribution"]) if y else ("(r2에 없음)", "")] += 1
res["final522_composition"] = {"|".join(k): v for k, v in fin.items()}
res["final522_by_venue"] = dict(collections.Counter(x["venue"] for x in final))
res["final522_by_year"] = dict(sorted(collections.Counter(x["year"] for x in final).items()))
lang_b = c[("Yes", "Language", "Benchmark")]
res["rebuild"]["Language_Benchmark_in_r2"] = lang_b
res["rebuild"]["Language_Benchmark_in_final522"] = fin[("Language", "Benchmark")]
res["rebuild"]["Multimodal_in_final522"] = fin[("Multimodal", "")]
res["rebuild"]["step3_excluded_Language"] = sum(v for (i, m, ct), v in c.items() if i == "Yes" and m == "Language") - fin[("Language", "Benchmark")]
res["rebuild"]["step3_excluded_Multimodal"] = c[("Yes", "Multimodal", "")] - fin[("Multimodal", "")]
res["rebuild"]["step3_excluded_total_ours"] = res["rebuild"]["step3_excluded_Language"] + res["rebuild"]["step3_excluded_Multimodal"]
print("재구성", json.dumps(res["rebuild"], ensure_ascii=False))
print("522 구성", res["final522_composition"], res["final522_by_venue"])

# ---- 2018-2019는 14편? ----
yr = collections.Counter(x["year"] for x in r2)
res["r2_by_year"] = dict(sorted(yr.items()))
res["r2_2018_2019"] = yr["2018"] + yr["2019"]
res["r2_by_venue"] = dict(collections.Counter(x["venue"] for x in r2))

# ---- 표 8: 사람 라벨 50편 x LLM 라벨 ----
conf = collections.Counter()
for x in man:
    y = rm[T(x["title"])]
    conf[(x["inclusion"], y["inclusion"])] += 1
TI, FI, TE, FE = conf[("Yes", "Yes")], conf[("No", "Yes")], conf[("No", "No")], conf[("Yes", "No")]
res["table8_step1_from_data"] = {"TI": TI, "FI": FI, "TE": TE, "FE": FE, "paper": {"TI": 14, "FI": 9, "TE": 22, "FE": 5}}
print("표 8 1단계(자료)", res["table8_step1_from_data"])
# 표 8 전체 행 (논문): TI 8, FI 2, TE 39, FE 1
ti, fi, te, fe = 8, 2, 39, 1
prec, rec = ti / (ti + fi), ti / (ti + fe)
f1 = 2 * prec * rec / (prec + rec)
res["table8_overall"] = {"TI": ti, "FI": fi, "TE": te, "FE": fe, "n": ti + fi + te + fe,
                         "precision": round(prec, 3), "precision_wilson95": wilson(ti, ti + fi),
                         "recall": round(rec, 3), "recall_wilson95": wilson(ti, ti + fe),
                         "f1": round(f1, 3), "paper_precision": 0.80, "paper_recall": 0.89, "paper_f1": 0.84}
# 사람 기준으로 적격한 논문의 비율과 외삽
pos = ti + fe
lo, hi = wilson(pos, 50)
res["extrapolation"] = {
    "human_eligible_in_sample": pos, "sample": 50,
    "eligible_rate_pct": round(100 * pos / 50, 1), "eligible_rate_wilson95": [lo, hi],
    "expected_eligible_in_2189": round(2189 * pos / 50), "expected_eligible_wilson95": [round(2189 * lo / 100), round(2189 * hi / 100)],
    "llm_selected": 522, "final_included": 445,
    "expected_missed_if_recall_0.889": round(445 / rec - 445),
    "missed_range_recall_wilson95": [round(445 / (wilson(ti, pos)[1] / 100) - 445), round(445 / (wilson(ti, pos)[0] / 100) - 445)],
    "note": "표본이 50편이고 사람 기준 적격이 9편이라 재현율 8/9의 구간이 매우 넓다. 최종 445 대비 놓친 편수는 외삽일 뿐이다.",
}
print("표 8 전체", res["table8_overall"])
print("외삽", res["extrapolation"])
# 단계별 순차 표(표 8 2-3행)의 비교 모집단이 설명되는가
res["table8_steps_note"] = {"step2_compared": 14, "step3_compared": 9,
                            "LLM_kept_after_step1": TI + FI, "human_kept_after_step1": TI + FE,
                            "note": "2단계 비교 14편은 1단계에서 사람과 LLM이 모두 통과시킨 편수(TI=14)와 같고, 3단계 9편은 2단계 통과(13+0)와도 14-5 등 어떤 조합과도 맞지 않는다. 논문이 비교 대상을 정의하지 않았다."}
print("표 8 단계 설명", res["table8_steps_note"])

# ---- 522 -> 445 ----
lo, hi = wilson(445, 522)
res["manual_stage"] = {"in": 522, "included": 445, "rate_pct": round(100 * 445 / 522, 1), "wilson95": [lo, hi],
                       "paper_claim": "정밀도 약 85%"}
print("522->445", res["manual_stage"])

# ---- 키워드 선별 재현 (저자 노트북 규칙: 'LLM' 대소문자 구분 또는 'language model' 소문자화, 'benchmark' 소문자화) ----
def hit(title, abs_):
    t = title or ""
    a = abs_ or ""
    llm = ("LLM" in t) or ("language model" in t.lower()) or ("LLM" in a) or ("language model" in a.lower())
    bm = ("benchmark" in t.lower()) or ("benchmark" in a.lower())
    return llm and bm


D = "data/gh/papers_dataset"
tot_papers = collections.Counter()
hits = collections.Counter()
for fn in sorted(os.listdir(D)):
    p = os.path.join(D, fn)
    if fn.endswith(".csv"):
        m = re.match(r"(ICLR|NeurIPS)_(\d{4})\.csv", fn)
        key = (m.group(1), m.group(2))
        for r in csv.DictReader(open(p, encoding="utf-8")):
            tot_papers[key] += 1
            if hit(r["name"], r["abstract"]):
                hits[key] += 1
    elif fn.endswith(".xml"):
        m = re.match(r"(\d{4})\.(\w+)\.xml", fn)
        key = ("ACL", m.group(1))
        root = ET.parse(p).getroot()
        for pa in root.iter("paper"):
            tot_papers[key] += 1
            t = "".join(pa.find("title").itertext()) if pa.find("title") is not None else ""
            ab = pa.find("abstract")
            doi = pa.find("doi")
            if ab is None or doi is None:  # 저자 코드: 초록과 doi가 있어야 남긴다
                continue
            if hit(t, "".join(ab.itertext())):
                hits[key] += 1
ours = collections.Counter()
r2c = collections.Counter()
for x in r2:
    r2c[(x["venue"], x["year"])] += 1
rows = []
for key in sorted(set(list(hits.keys()) + [k for k in r2c if k[0] != "ICML"])):
    rows.append({"venue": key[0], "year": key[1], "records_in_repo_files": tot_papers[key], "keyword_hits_ours": hits[key], "in_2189_list": r2c[key]})
res["keyword_recount_by_venue_year"] = rows
sum_hits = {v: sum(h for (vv, y), h in hits.items() if vv == v) for v in ("ACL", "ICLR", "NeurIPS")}
sum_r2 = {v: sum(n for (vv, y), n in r2c.items() if vv == v) for v in ("ACL", "ICLR", "NeurIPS", "ICML")}
res["keyword_recount_totals"] = {"ours": sum_hits, "paper_list_2189": sum_r2,
                                 "records_total_repo_files_without_ICML": sum(tot_papers.values())}
print("키워드 재집계 합", sum_hits, "목록 2189 쪽", sum_r2, "저장소 파일 속 논문 수(ICML 없음)", sum(tot_papers.values()))
for r in rows:
    flag = "" if r["keyword_hits_ours"] == r["in_2189_list"] else "  <-- 불일치"
    print(f"  {r['venue']:8s} {r['year']} files={r['records_in_repo_files']:5d} ours={r['keyword_hits_ours']:4d} list={r['in_2189_list']:4d}{flag}")

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
