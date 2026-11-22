"""공개 차별 테스트 파일로 보는 보조 분석: 의심 패치를 뒷받침하는 테스트 수가 수작업 오답 판정과 관련 있는가.

입력: data/unz/PatchDiff/results/RQ2_*_difftests.json, RQ34.csv
출력: analysis/difftests_stats_results.json
실행: python3 -I analysis/difftests_stats.py
"""
import csv, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "data", "unz", "PatchDiff", "results")
TOOLS = ["CodeStory", "LearnByInteract", "OpenHands"]
n_entries, n_support = {}, {}
out = {}
for t in TOOLS:
    d = json.load(open(os.path.join(R, f"RQ2_{t}_difftests.json")))
    for i, l in d.items():
        sup = [e for e in l if e["differential_tests"]]
        if sup:
            n_support[(t, i)] = len(sup)
            n_entries[(t, i)] = len(l)
    v = np.array([n_support[k] for k in n_support if k[0] == t])
    out[t] = dict(suspicious=int(len(v)), supporting_files_median=float(np.median(v)), supporting_files_min=int(v.min()),
                  single_file_support=int((v == 1).sum()), single_file_pct=round(100 * float((v == 1).mean()), 1))
allv = np.array(list(n_support.values()))
out["pooled"] = dict(suspicious=int(len(allv)), single_file_support=int((allv == 1).sum()), single_file_pct=round(100 * float((allv == 1).mean()), 1),
                     median=float(np.median(allv)), p25=float(np.percentile(allv, 25)), p75=float(np.percentile(allv, 75)),
                     total_generated_files_median=float(np.median(list(n_entries.values()))))

rows = list(csv.DictReader(open(os.path.join(R, "RQ34.csv"))))
inc, unc, cor = [], [], []
for r in rows:
    k = (r["tool"], r["instance_id"])
    x = n_support[k]
    (inc if r["correctness"].startswith("incorrect") else unc if r["correctness"] == "uncertain" else cor).append(x)
u = stats.mannwhitneyu(inc, unc, alternative="two-sided")
out["support_vs_label"] = dict(incorrect_n=len(inc), uncertain_n=len(unc), correct_n=len(cor),
                               incorrect_median=float(np.median(inc)), uncertain_median=float(np.median(unc)), correct_values=cor,
                               mannwhitney_p=float(f"{u.pvalue:.3g}"),
                               single_file_among_incorrect=f"{sum(1 for x in inc if x == 1)}/{len(inc)}",
                               single_file_among_uncertain=f"{sum(1 for x in unc if x == 1)}/{len(unc)}")
# 한 인스턴스를 여러 도구가 의심받는 경우: 표본에서 같은 인스턴스가 여러 도구로 중복 들어왔는지
from collections import Counter
c = Counter(r["instance_id"] for r in rows)
out["duplicate_instances_in_sample"] = dict(unique_instances=len(c), rows=len(rows), instances_in_2plus_tools=sum(1 for v in c.values() if v > 1))
with open(os.path.join(HERE, "difftests_stats_results.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
