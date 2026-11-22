"""공개 패치 파일만으로 보는 보조 분석: 생성 패치 크기·정답 패치와의 동일성이 의심 판정과 관련 있는가.

입력: data/unz/PatchDiff/ (Zenodo 압축 해제본). 코드 실행 없음, 텍스트 비교만.
출력: analysis/patch_size_results.json
실행: python3 -I analysis/patch_size_analysis.py
"""
import json, os, re
import pyarrow.ipc as ipc
from scipy import stats
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P = os.path.join(ROOT, "data", "unz", "PatchDiff"); R = os.path.join(P, "results")
TOOLS = {"CodeStory": "20241221_codestory_midwit_claude-3-5-sonnet_swe-search",
         "LearnByInteract": "20250110_learn_by_interact_claude3.5",
         "OpenHands": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022_verified"}

with open(os.path.join(P, "data/dataset/swebench_verified/data-00000-of-00001.arrow"), "rb") as f:
    tbl = ipc.open_stream(f).read_all().to_pylist()
gold = {r["instance_id"]: r["patch"] for r in tbl}


def stat(patch, src_only=False):
    """파일 수와 변경 줄 수. src_only=True 면 새로 만든 파일과 test·재현 스크립트 경로를 뺀 소스 변경만 센다."""
    n_files, n_lines = 0, 0
    for blk in re.split(r"(?m)^(?=diff --git )", patch):
        m = re.match(r"diff --git a/(\S+)", blk)
        if not m:
            continue
        path = m.group(1)
        if src_only and ("new file mode" in blk or re.search(r"(^|/)(tests?|testing)(/|_)|test_|_test\.py|reproduce|repro", path)):
            continue
        n_files += 1
        n_lines += sum(1 for l in blk.splitlines() if (l.startswith("+") and not l.startswith("+++")) or (l.startswith("-") and not l.startswith("---")))
    return n_files, n_lines


def norm(patch):
    """바뀐 줄만 모아 공백을 접고 주석(#)·빈 줄을 뺀 비교 키."""
    ls = []
    for l in patch.splitlines():
        if (l.startswith("+") and not l.startswith("+++")) or (l.startswith("-") and not l.startswith("---")):
            body = l[1:].strip()
            if not body or body.startswith("#"):
                continue
            ls.append(l[0] + re.sub(r"\s+", " ", body))
    return tuple(ls)


out = {}
rows = []
for t, folder in TOOLS.items():
    res = set(json.load(open(os.path.join(P, "data/tool_results", folder, "results.json")))["resolved"])
    preds = {}
    for line in open(os.path.join(P, "data/tool_results", folder, "all_preds.jsonl")):
        x = json.loads(line); preds[x["instance_id"]] = x["model_patch"]
    d = json.load(open(os.path.join(R, f"RQ2_{t}_difftests.json")))
    susp = {i for i, l in d.items() if any(e["differential_tests"] for e in l)}
    tested = set(d)
    ident = {i for i in res if norm(preds[i]) == norm(gold[i])}
    out.setdefault("per_tool", {})[t] = dict(plausible=len(res), identical_to_gold_normalized=len(ident),
                                             tested_by_patchdiff=len(tested), plausible_not_tested=len(res - tested),
                                             not_tested_and_not_identical=len((res - tested) - ident),
                                             identical_and_susp=len(ident & susp),
                                             susp_among_non_identical=f"{len(susp)}/{len(res - ident)}")
    for i in res:
        gf, gl = stat(gold[i], True); pf, pl = stat(preds[i], True); af, al = stat(preds[i])
        rows.append(dict(tool=t, inst=i, s=int(i in susp), ident=int(i in ident), tested=int(i in tested), gf=gf, gl=gl, pf=pf, pl=pl, af=af, al=al))

# 동일 패치를 뺀 의심 비율, 크기별 의심 비율
non = [r for r in rows if not r["ident"]]
k = sum(r["s"] for r in non); n = len(non)
out["suspicious_among_non_identical"] = dict(k=k, n=n, pct=round(100 * k / n, 1),
                                             wilson95=[round(100 * x, 1) for x in (stats.binomtest(k, n).proportion_ci(method="wilson").low, stats.binomtest(k, n).proportion_ci(method="wilson").high)])
def cmp(name, f):
    a = np.array([f(r) for r in non if r["s"]]); b = np.array([f(r) for r in non if not r["s"]])
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return dict(metric=name, susp_median=float(np.median(a)), nonsusp_median=float(np.median(b)),
                susp_mean=round(float(a.mean()), 2), nonsusp_mean=round(float(b.mean()), 2), mannwhitney_p=float(f"{u.pvalue:.3g}"))
out["size_tests"] = [cmp("생성 패치 변경 줄 수", lambda r: r["pl"]), cmp("정답 패치 변경 줄 수", lambda r: r["gl"]),
                     cmp("생성/정답 변경 줄 비", lambda r: r["pl"] / max(r["gl"], 1)),
                     cmp("생성 패치 파일 수", lambda r: r["pf"]), cmp("생성-정답 파일 수 차", lambda r: r["pf"] - r["gf"])]
out["raw_patch_includes_extra_files"] = dict(
    median_changed_lines_all_files=float(np.median([r["al"] for r in rows])), median_changed_lines_source_only=float(np.median([r["pl"] for r in rows])),
    share_patches_with_extra_nonsource_files_pct=round(100 * sum(1 for r in rows if r["af"] > r["pf"]) / len(rows), 1))
tested_non = [r for r in non if r["tested"]]
kt = sum(r["s"] for r in tested_non)
out["susp_among_tested_by_patchdiff"] = dict(k=kt, n=len(tested_non), pct=round(100 * kt / len(tested_non), 1),
                                             untested_non_identical=len(non) - len(tested_non))
# 정답 패치보다 파일을 더 건드린 패치의 의심 비율
for lab, cond in [("생성 파일 수 > 정답", lambda r: r["pf"] > r["gf"]), ("생성 파일 수 <= 정답", lambda r: r["pf"] <= r["gf"]),
                  ("생성 줄 수 > 2배 정답", lambda r: r["pl"] > 2 * r["gl"]), ("생성 줄 수 <= 2배 정답", lambda r: r["pl"] <= 2 * r["gl"])]:
    sub = [r for r in non if cond(r)]
    out.setdefault("rate_by_group", {})[lab] = dict(n=len(sub), susp=sum(r["s"] for r in sub), pct=round(100 * sum(r["s"] for r in sub) / len(sub), 1))
a = out["rate_by_group"]
for x, y, key in [("생성 파일 수 > 정답", "생성 파일 수 <= 정답", "fisher_files"), ("생성 줄 수 > 2배 정답", "생성 줄 수 <= 2배 정답", "fisher_lines2x")]:
    out[key] = round(float(stats.fisher_exact([[a[x]["susp"], a[x]["n"] - a[x]["susp"]], [a[y]["susp"], a[y]["n"] - a[y]["susp"]]])[1]), 4)
with open(os.path.join(HERE, "patch_size_results.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
