"""부록 D 표 9(평가자 간 일치도)를 저자의 이중 코딩 자료로 재계산한다.

입력: data/gh/merged_annotations.csv (82행 = 41개 논문 x 2명)  <- 논문은 "46편"이라고 쓴다.
      paper/paper.txt 의 표 9 (필드별 일치율, BPK)
방법: 논문 부록 D의 정의를 그대로 쓴다.
  - 단일 라벨: 정확히 같으면 일치. 다중 라벨: 선택 집합의 Jaccard 평균을 일치율로 쓰고,
    Jaccard > 0.3이면 "일치"로 바꿔 이진 BPK = 2*Po - 1 (k=2)을 계산.
  - BPK = (Po - 1/k) / (1 - 1/k), k = 유효한 응답 범주 수. 다중 클래스의 k는 자료에서 알 수 없어
    (a) 두 평가자가 실제로 고른 서로 다른 범주 수, (b) 표 9의 값에서 역산한 k 두 가지를 낸다.
  - 빈 응답도 하나의 값으로 센다(표 9 phenomenon_defined 22/41 = 53.66%가 이렇게만 나온다).
추가로 Cohen's kappa와 Gwet's AC1을 같은 자료로 계산한다(저자 노트북이 낸 값과 비교).
실행: python3 -I analysis/recompute_irr.py  (논문 폴더 루트에서)
"""
import collections
import csv
import json
import math
import re

csv.field_size_limit(10**9)
MULTI = {"phenomenon_short", "task_train_val", "response_format", "metric_definition", "task_source",
         "dataset_sampling_method", "metric_aggregation"}


def norm1(v):
    """라벨 정규화: 설명 괄호 이하를 떼고 공백 정리. 'Contested (e.g. ...)' -> 'Contested'."""
    v = v.strip()
    v = re.split(r"\s+\(", v, maxsplit=1)[0]
    return v.strip().rstrip(",").lower()


def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x for x in (norm1(x) for x in out) if x]


def parse_table9(path):
    rows = {}
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"\s*([a-z_]+)\s+(Binary|Multi-class|Multi-label)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
        if m:
            rows[m.group(1)] = (m.group(2), float(m.group(3)), float(m.group(4)))
    return rows


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cohen(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def gwet_ac1(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    cnt = collections.Counter(a + b)
    pe = sum((cnt[c] / (2 * n)) ** 2 for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


rows = list(csv.DictReader(open("data/gh/merged_annotations.csv", encoding="utf-8")))
pairs = collections.defaultdict(list)
for r in rows:
    pairs[r["bibkey"]].append(r)
assert all(len(v) == 2 for v in pairs.values())
T9 = parse_table9("paper/paper.txt")
print("표 9 파싱", len(T9), "필드;  자료의 이중 코딩 논문 수", len(pairs))

out = {"n_pairs_in_data": len(pairs), "n_rows": len(rows), "paper_says_n": 46,
       "fields": {}, "missing_in_data": [f for f in T9 if f not in rows[0]]}
hdr = f"{'field':32s} {'type':11s} {'paper%':>6s} {'ours%':>6s} {'paperBPK':>8s} {'oursBPK(k=obs)':>14s} {'k_implied':>9s} {'kappa':>6s} {'AC1':>6s}"
print(hdr)
sum_p, sum_b, sum_pp, sum_pb, cnt = 0, 0, 0, 0, 0
kap, ac1 = [], []
for f, (typ, pp, pb) in T9.items():
    if f not in rows[0]:
        print(f"{f:32s} {typ:11s} {pp:6.2f}   (merged_annotations.csv에 열 없음)  paperBPK={pb}")
        continue
    A, B, J = [], [], []
    for k, v in pairs.items():
        x, y = v[0][f], v[1][f]
        if f in MULTI:
            sa, sb = set(split_top(x)), set(split_top(y))
            u = sa | sb
            J.append(len(sa & sb) / len(u) if u else 1.0)
        else:
            A.append(norm1(x))
            B.append(norm1(y))
    n = len(pairs)
    if f in MULTI:
        pct = 100 * sum(J) / n
        po = sum(j > 0.3 for j in J) / n
        bpk = 2 * po - 1
        k_obs = 2
        kappa = ac = float("nan")
    else:
        po = sum(x == y for x, y in zip(A, B)) / n
        pct = 100 * po
        k_obs = len(set(A) | set(B))
        if typ == "Binary":
            k_obs = 2
        bpk = (po - 1 / k_obs) / (1 - 1 / k_obs)
        kappa, ac = cohen(A, B), gwet_ac1(A, B)
        kap.append(kappa)
        ac1.append(ac)
    # 표 9 값에서 k 역산 (단일 라벨만 의미가 있다)
    inv = (pp / 100 - pb) / (1 - pb) if pb != 1 else float("nan")
    k_imp = 1 / inv if (typ != "Multi-label" and inv > 0) else float("nan")
    lo, hi = wilson(round(po * n), n)
    out["fields"][f] = {"type": typ, "n_pairs": n, "paper_pct": pp, "our_pct": round(pct, 2),
                        "paper_bpk": pb, "our_bpk_k_observed": round(bpk, 3), "k_used": k_obs,
                        "k_implied_by_table9": None if math.isnan(k_imp) else round(k_imp, 2),
                        "cohen_kappa": None if math.isnan(kappa) else round(kappa, 3),
                        "gwet_ac1": None if math.isnan(ac) else round(ac, 3),
                        "po_wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
                        "pct_diff": round(pct - pp, 2)}
    print(f"{f:32s} {typ:11s} {pp:6.2f} {pct:6.2f} {pb:8.3f} {bpk:14.3f} {k_imp:9.2f} {kappa:6.3f} {ac:6.3f}")
    sum_p += pct
    sum_b += bpk
    sum_pp += pp
    sum_pb += pb
    cnt += 1

# 가설 검사: 두 번째 검토자가 Exclude라 빈 응답이 28/30인 한 쌍을 inclusion 외 필드에서 "일치"로 세면 표 9와 맞는가
excl_keys = [k for k, v in pairs.items() if any(x["inclusion"] != "Include" for x in v)]
hyp = {}
for f, (typ, pp, pb) in T9.items():
    if f not in rows[0] or f == "inclusion":
        continue
    tot = 0.0
    for k, v in pairs.items():
        if k in excl_keys:
            tot += 1.0
            continue
        x, y = v[0][f], v[1][f]
        if f in MULTI:
            sa, sb = set(split_top(x)), set(split_top(y)); u = sa | sb
            tot += len(sa & sb) / len(u) if u else 1.0
        else:
            tot += norm1(x) == norm1(y)
    hyp[f] = round(100 * tot / len(pairs), 2)
out["hypothesis_excluded_pair_counted_as_agree"] = {
    "pair_bibkeys": excl_keys,
    "n_fields_within_0.1": sum(1 for f, v in hyp.items() if abs(v - T9[f][1]) < 0.1),
    "n_fields": len(hyp),
    "mean_pct": round(sum(hyp.values()) / len(hyp), 2),
    "mean_pct_paper_same_fields": round(sum(T9[f][1] for f in hyp) / len(hyp), 2),
    "per_field": hyp}
print("가설(제외 쌍을 일치로 셈):", {k: v for k, v in out["hypothesis_excluded_pair_counted_as_agree"].items() if k != "per_field"})
ok = [v for v in out["fields"].values() if abs(v["pct_diff"]) < 0.1]
out["summary"] = {
    "n_fields_compared": cnt,
    "n_fields_pct_within_0.1": len(ok),
    "mean_pct_ours": round(sum_p / cnt, 2), "mean_pct_paper_same_fields": round(sum_pp / cnt, 2),
    "mean_bpk_ours_k_observed": round(sum_b / cnt, 3), "mean_bpk_paper_same_fields": round(sum_pb / cnt, 3),
    "mean_bpk_paper_all30": 0.524, "mean_pct_paper_all30": 68.11,
    "mean_cohen_kappa_single_label": round(sum(kap) / len(kap), 3),
    "mean_gwet_ac1_single_label": round(sum(ac1) / len(ac1), 3),
    "n_single_label_fields": len(kap),
}
# 표 9 평균 자체의 산술 검산 (30행 평균)
pcts = [v[1] for v in T9.values()]
bpks = [v[2] for v in T9.values()]
out["table9_mean_check"] = {"rows": len(T9), "mean_pct": round(sum(pcts) / len(pcts), 2), "mean_bpk": round(sum(bpks) / len(bpks), 3),
                            "paper_mean_pct": 68.11, "paper_mean_bpk": 0.524,
                            "n_fields_bpk_lt_0.4": sum(b < 0.4 for b in bpks), "n_fields_bpk_ge_0.8": sum(b >= 0.8 for b in bpks)}
# 정의 여부 필드의 핵심 수치
out["table9_phenomenon_defined"] = T9.get("phenomenon_defined")
print("\n요약", json.dumps(out["summary"], ensure_ascii=False, indent=1))
print("표 9 평균 검산", out["table9_mean_check"])
print("자료에 없는 필드", out["missing_in_data"])
json.dump(out, open("analysis/recompute_irr_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
