"""SWE-bench (arXiv:2310.06770) 재계산: 데이터셋 통계, 주 결과표, 구간, 짝지은 검정, 라벨 잡음.
입력: data/SWE-bench*/test.parquet, data/exp/<split>/<제출>/results.json (모두 gitignore, README의 재실행 절 참조)
실행: python3 -I analysis/recompute_tables.py   (폴더 swe_bench/ 에서)
"""
import json, re, math, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
out = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (c - h), 2), round(100 * (c + h), 2)


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))


def load(name):
    df = pd.read_parquet(D / name / "test.parquet")
    for c in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        df[c] = df[c].map(lambda s: json.loads(s) if isinstance(s, str) else list(s))
    return df


full, lite, ver = load("SWE-bench"), load("SWE-bench_Lite"), load("SWE-bench_Verified")


def patch_stats(p):
    files = set(re.findall(r"^diff --git a/(\S+) b/", p, re.M))
    lines = p.split("\n")
    add = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    ctx, cur = set(), None
    for l in lines:
        if l.startswith("diff --git"):
            cur = l.split(" b/")[-1]
        m = re.match(r"^@@ [^@]+ @@\s*(.*)$", l)
        if m and m.group(1).strip():
            ctx.add((cur, m.group(1).strip()))
    hunks = sum(1 for l in lines if l.startswith("@@"))
    return dict(files=len(files), add=add, rem=rem, lines=add + rem, funcs=len(ctx), hunks=hunks)


ps = pd.DataFrame(list(full.patch.map(patch_stats)))
full = pd.concat([full.reset_index(drop=True), ps], axis=1)
full["f2p"] = full.FAIL_TO_PASS.map(len)
full["p2p"] = full.PASS_TO_PASS.map(len)
full["words"] = full.problem_statement.str.split().map(len)
full["year"] = full.created_at.str[:4].astype(int)
full["img"] = full.problem_statement.str.contains(r"!\[")

# ---------- 1. 데이터셋 통계 (Table 1, 10, 11, 21, 본문 수치) ----------
paper_t1 = dict(words=(195.1, 4477), lines=(32.8, 5888), files=(1.7, 31), funcs=(3.0, 36), f2p=(9.1, 1633))
d1 = {"n": len(full), "n_repos": int(full.repo.nunique())}
for k, (pm, px) in paper_t1.items():
    d1[k] = dict(ours_mean=round(float(full[k].mean()), 2), ours_max=int(full[k].max()), paper_mean=pm, paper_max=px)
tot = full.f2p + full.p2p
d1["tests_total"] = dict(ours_mean=round(float(tot.mean()), 1), ours_max=int(tot.max()), paper_mean=120.8, paper_max=9459)
d1["p2p_median"] = dict(ours=float(full.p2p.median()), paper=51)
d1["f2p_ge2_share"] = dict(ours=round(float((full.f2p >= 2).mean()), 3), paper=0.40)
d1["p2p_zero_instances"] = int((full.p2p == 0).sum())
d1["image_share"] = dict(ours=round(float(full.img.mean()), 3), paper=0.02,
                         matplotlib=round(float(full.img[full.repo == "matplotlib/matplotlib"].mean()), 3), paper_matplotlib=0.32,
                         seaborn=round(float(full.img[full.repo == "mwaskom/seaborn"].mean()), 3), paper_seaborn=0.10)
# Table 10 최종 수
t10 = {"astropy/astropy": 95, "django/django": 850, "pallets/flask": 11, "matplotlib/matplotlib": 184, "pylint-dev/pylint": 57,
       "pytest-dev/pytest": 119, "psf/requests": 44, "scikit-learn/scikit-learn": 229, "mwaskom/seaborn": 22,
       "sphinx-doc/sphinx": 187, "sympy/sympy": 386, "pydata/xarray": 110}
rc = full.repo.value_counts().to_dict()
d1["table10_repo_counts_match"] = all(rc[k] == v for k, v in t10.items())
# Table 11 저장소별 (F2P, P2P, 편집 줄, 파일)
t11 = {  # repo: (files, lines, f2p, p2p)
    "astropy/astropy": (1.5, 36.0, 21.7, 191.0), "django/django": (1.5, 18.5, 8.8, 85.9), "pallets/flask": (1.6, 35.4, 1.4, 32.5),
    "matplotlib/matplotlib": (1.5, 58.9, 2.4, 242.4), "pylint-dev/pylint": (1.8, 36.0, 6.8, 47.0), "pytest-dev/pytest": (1.4, 24.5, 4.1, 60.7),
    "psf/requests": (1.64, 25.5, 7.6, 87.1), "scikit-learn/scikit-learn": (1.68, 44.0, 7.5, 150.7), "mwaskom/seaborn": (1.77, 30.1, 12.9, 86.8),
    "sphinx-doc/sphinx": (1.51, 30.6, 2.3, 45.1), "sympy/sympy": (1.9, 36.3, 2.2, 74.5), "pydata/xarray": (2.45, 124.8, 58.5, 297.5)}
rows = []
for repo, (pf, pl, pf2p, pp2p) in t11.items():
    g = full[full.repo == repo]
    rows.append(dict(repo=repo, files=(round(g.files.mean(), 2), pf), lines=(round(g.lines.mean(), 1), pl),
                     f2p=(round(g.f2p.mean(), 1), pf2p), p2p=(round(g.p2p.mean(), 1), pp2p)))
d1["table11_ours_vs_paper"] = rows
# Table 21 연도별 합계 (paper: 2023 244, 2022 395, 2021 383, 2020 427, 2019 437, 2018 165, <2018 89)
yr = full.year.value_counts().sort_index().to_dict()
ours21 = {"2023": yr.get(2023, 0), "2022": yr.get(2022, 0), "2021": yr.get(2021, 0), "2020": yr.get(2020, 0), "2019": yr.get(2019, 0),
          "2018": yr.get(2018, 0), "<2018": int(sum(v for k, v in yr.items() if k < 2018))}
d1["table21_totals"] = dict(ours=ours21, paper=dict(zip(ours21, [244, 395, 383, 427, 437, 165, 89])), paper_sum=244 + 395 + 383 + 427 + 437 + 165 + 89, ours_sum=sum(ours21.values()))
# Lite / Verified 구성
ids = set(full.instance_id)
d1["lite"] = dict(n=len(lite), subset_of_full=set(lite.instance_id) <= ids, repos=int(lite.repo.nunique()),
                  per_repo=lite.repo.value_counts().to_dict())
lp = pd.DataFrame(list(lite.patch.map(patch_stats)))
d1["lite"].update(gold_files_max=int(lp.files.max()), gold_hunks_max=int(lp.hunks.max()),
                  words_min=int(lite.problem_statement.str.split().map(len).min()),
                  image_in_issue=int(lite.problem_statement.str.contains(r"!\[").sum()))
d1["verified"] = dict(n=len(ver), subset_of_full=set(ver.instance_id) <= ids, repos=int(ver.repo.nunique()),
                      per_repo=ver.repo.value_counts().to_dict(), in_lite=len(set(ver.instance_id) & set(lite.instance_id)))
out["dataset_stats"] = d1

# ---------- 2. 주 결과표 ----------
MODELS = {"Claude 2": "20231010_rag_claude2", "ChatGPT-3.5": "20231010_rag_gpt35", "SWE-Llama 7b": "20231010_rag_swellama7b",
          "SWE-Llama 13b": "20231010_rag_swellama13b", "Claude 3 Opus": "20240402_rag_claude3opus", "GPT-4 (1106)": "20240402_rag_gpt4"}
PAPER5 = {  # model: (full resolved%, full apply%, lite resolved%, lite apply%)
    "Claude 3 Opus": (3.79, 46.56, 4.33, 51.67), "Claude 2": (1.97, 43.07, 3.00, 33.00), "ChatGPT-3.5": (0.17, 26.33, 0.33, 10.00),
    "GPT-4 (1106)": (1.31, 26.90, 2.67, 29.67), "SWE-Llama 7b": (0.70, 51.74, 1.33, 38.00), "SWE-Llama 13b": (0.70, 53.62, 1.00, 38.00)}
PAPER14 = {"Claude 2": (988, 302), "ChatGPT-3.5": (604, 363), "SWE-Llama 13b": (1230, 369), "SWE-Llama 7b": (1187, 340)}  # Applies, Fixed
res = {}
for split in ("test", "lite", "verified"):
    for m, d in MODELS.items():
        res[(split, m)] = json.load(open(D / "exp" / split / d / "results.json"))
N = {"test": 2294, "lite": 300, "verified": 500}
t5 = []
for m in MODELS:
    row = {"model": m}
    for split in ("test", "lite", "verified"):
        r = res[(split, m)]
        k, a = len(r["resolved"]), len(r["applied"])
        row[split] = dict(resolved=k, n=N[split], pct=round(100 * k / N[split], 2), wilson95=wilson(k, N[split]),
                          applied_file=a, applied_pct_file=round(100 * a / N[split], 2))
    p = PAPER5[m]
    row["paper_table5"] = dict(full_resolved=p[0], full_apply=p[1], lite_resolved=p[2], lite_apply=p[3])
    row["match_full_resolved"] = abs(row["test"]["pct"] - p[0]) < 0.006
    row["match_lite_resolved"] = abs(row["lite"]["pct"] - p[2]) < 0.006
    if m in PAPER14:
        ap, fx = PAPER14[m]
        row["table14_applies_minus_fixed"] = ap - fx
        row["table14_identity"] = (ap - fx) == row["test"]["applied_file"]
    t5.append(row)
out["table5"] = t5
# resolved ⊂ applied?
out["resolved_subset_of_applied"] = {m: set(res[("test", m)]["resolved"]) <= set(res[("test", m)]["applied"]) for m in MODELS}

# ---------- 3. 짝지은 검정 ----------
def pairwise(split):
    names = list(MODELS)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = set(res[(split, names[i])]["resolved"]), set(res[(split, names[j])]["resolved"])
            b, c = len(A - B), len(B - A)
            rows.append(dict(a=names[i], b=names[j], only_a=b, only_b=c, both=len(A & B), p=mcnemar_exact(b, c)))
    ps_ = np.array([r["p"] for r in rows])
    order = np.argsort(ps_)
    holm = np.empty(len(ps_))
    run = 0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (len(ps_) - rank) * ps_[idx]))
        holm[idx] = run
    for r, h in zip(rows, holm):
        r["p"] = round(float(r["p"]), 5)
        r["p_holm"] = round(float(h), 5)
    return rows
out["mcnemar_full"] = pairwise("test")
out["mcnemar_lite"] = pairwise("lite")

# ---------- 4. 오라클 대 BM25: 주변 합만으로 가능한 p 범위 ----------
def margin_range(a, b, n):
    lo = max(0, a + b - n)
    hi = min(a, b)
    ps_ = [mcnemar_exact(a - k, b - k) for k in range(lo, hi + 1)]
    return dict(oracle=a, bm25=b, n=n, p_min=float(min(ps_)), p_max=float(max(ps_)))
ORACLE = {"Claude 2": 110, "SWE-Llama 13b": 91, "SWE-Llama 7b": 69, "ChatGPT-3.5": 12}  # Table 18 % x 2294 (Claude 2 110, SL13b 91은 본문)
out["oracle_vs_bm25_margin_bounds"] = {m: margin_range(a, len(res[("test", m)]["resolved"]), 2294) for m, a in ORACLE.items()}
out["oracle_counts_check"] = {m: dict(count=a, pct_from_count=round(100 * a / 2294, 2)) for m, a in ORACLE.items()}
out["oracle_paper_pct"] = {"Claude 2": 4.80, "ChatGPT-3.5": 0.52, "SWE-Llama 7b": 3.01, "SWE-Llama 13b": 3.97}
# SWE-Llama 13b 91개 중 Claude 2가 푼 비율 42%(본문 5절): 38개 안팎
out["gpt4_subset"] = dict(n=574, share_of_2294=round(574 / 2294, 3),
                          oracle_k=10, oracle_pct=round(100 * 10 / 574, 2), oracle_wilson=wilson(10, 574),
                          bm25_k=0, bm25_wilson=wilson(0, 574),
                          claude2_on_subset_oracle_pct=4.01, claude2_k_approx=round(0.0401 * 574),
                          claude2_wilson=wilson(round(0.0401 * 574), 574),
                          gpt4_vs_claude2_margin=margin_range(round(0.0401 * 574), 10, 574),
                          year_split_2023_k_of_n=dict(note="Table 7 GPT-4 After 2023 = 0.0 (표본 약 61, 25% 부분집합)", wilson_0_of_61=wilson(0, 61)))
out["table7_checks"] = dict(before_2023_n=int((full.year < 2023).sum()), after_2023_n=int((full.year >= 2023).sum()))

# ---------- 5. 문항 난이도와 해결 ----------
fi = full.set_index("instance_id")
solved_any = set().union(*[set(res[("test", m)]["resolved"]) for m in MODELS])
out["solved_by_any_model"] = len(solved_any)
sv = fi.loc[sorted(solved_any)]
uns = fi.drop(index=sorted(solved_any))
mw = {}
for col in ("lines", "files", "hunks", "f2p", "p2p", "words"):
    u = stats.mannwhitneyu(sv[col], uns[col])
    mw[col] = dict(median_solved=float(sv[col].median()), median_unsolved=float(uns[col].median()), p=float(u.pvalue))
out["solved_vs_unsolved_instance_features"] = mw
c3 = set(res[("test", "Claude 3 Opus")]["resolved"])
out["claude3opus_by_year_trend"] = {}
yrs = fi.year.clip(lower=2016)
tab = pd.DataFrame({"y": yrs, "s": fi.index.isin(c3)})
g = tab.groupby("y").s.agg(["sum", "count"])
out["claude3opus_by_year_trend"]["table"] = {int(k): [int(v["sum"]), int(v["count"])] for k, v in g.iterrows()}
# Cochran-Armitage 근사: 로지스틱 대신 점-이연 스피어만
rho, pv = stats.spearmanr(tab.y, tab.s)
out["claude3opus_by_year_trend"].update(spearman_rho=float(rho), p=float(pv))
repo_tab = pd.crosstab(fi.repo, fi.index.isin(c3))
chi = stats.chi2_contingency(repo_tab.values)
out["claude3opus_repo_heterogeneity"] = dict(chi2=float(chi[0]), p=float(chi[1]), note="기대 빈도 작은 칸 많음. 근사치")

# ---------- 6. Verified 선택 효과와 라벨 잡음 ----------
vids = set(ver.instance_id)
noise = {}
for m in MODELS:
    R = set(res[("test", m)]["resolved"])
    kv, kn = len(R & vids), len(R - vids)
    nv, nn = len(vids), 2294 - len(vids)
    fe = stats.fisher_exact([[kv, nv - kv], [kn, nn - kn]])
    ver_file = set(res[("verified", m)]["resolved"])
    noise[m] = dict(verified=f"{kv}/{nv}", rest=f"{kn}/{nn}", pct_verified=round(100 * kv / nv, 2), pct_rest=round(100 * kn / nn, 2),
                    odds_ratio=round(float(fe[0]), 2) if fe[0] != float("inf") else None, fisher_p=float(fe[1]),
                    verified_file_k=len(ver_file), intersection_k=kv, same_set=(ver_file == (R & vids)), only_in_verified_run=sorted(ver_file - (R & vids)), only_in_full_run=sorted((R & vids) - ver_file))
out["verified_vs_rest"] = noise
# OpenAI 2024 주석 수치(공식 글 인용)
N_ANN = 1699
oa = dict(n_annotated=N_ANN, under_pct=38.3, tests_pct=61.1, filtered_pct=68.3, verified_n=500,
          under_n=round(.383 * N_ANN), tests_n=round(.611 * N_ANN), filtered_n=round(.683 * N_ANN))
oa["filtered_wilson95"] = wilson(oa["filtered_n"], N_ANN)
oa["tests_wilson95"] = wilson(oa["tests_n"], N_ANN)
oa["under_wilson95"] = wilson(oa["under_n"], N_ANN)
oa["min_overlap_under_and_tests_pct"] = round(38.3 + 61.1 - 100, 1)  # 합집합<=100 이므로 겹침 >= -0.6 (제약 없음)
oa["overlap_if_only_these_two_cause_filter_pct"] = round(38.3 + 61.1 - 68.3, 1)
oa["kept_pct"] = round(100 - 68.3, 1)
oa["kept_n"] = N_ANN - oa["filtered_n"]
# 해결률에 대한 함의: 라벨 잡음 p(문항이 부적격)=0.683이면, 해결 판정이 거짓 양성/거짓 음성일 수 있는 영역
oa["verified_gap_note"] = "Verified 부분집합에서의 해결률이 나머지보다 높은 것은 부적격 문항을 걸러낸 효과와 일치하지만 인과 확정은 아님"
out["verified_annotation_openai2024"] = oa


# ---------- 7. P2P가 빈 문항 (회귀 검사 없음) ----------
empty = set(full.instance_id[full.p2p == 0])
out["empty_p2p"] = dict(n=len(empty), share=round(len(empty) / 2294, 4),
                        in_lite=len(empty & set(lite.instance_id)), in_verified=len(empty & vids),
                        resolved_by_model={m: len(set(res[("test", m)]["resolved"]) & empty) for m in MODELS},
                        resolved_rate_empty_vs_rest_claude3opus=[len(c3 & empty), len(empty), len(c3 - empty), 2294 - len(empty)])
fe2 = stats.fisher_exact([[len(c3 & empty), len(empty) - len(c3 & empty)], [len(c3 - empty), 2294 - len(empty) - len(c3 - empty)]])
out["empty_p2p"]["fisher_p_claude3opus"] = float(fe2[1])
out["f2p_single_share"] = round(float((full.f2p == 1).mean()), 3)

# ---------- 8. 표 2: 문맥 한도별 (문항 단위 결과가 없어 주변 합으로 가능한 p 범위) ----------
T2 = {"Claude 2": (1.96, 1.87, 1.22), "SWE-Llama 7b": (0.70, 0.31, 0.00), "SWE-Llama 13b": (0.70, 0.48, 0.00)}
t2o = {}
for m, (a, b, c) in T2.items():
    ka, kb, kc = (round(x * 2294 / 100) for x in (a, b, c))
    t2o[m] = dict(counts_13k_27k_50k=(ka, kb, kc), p_range_13k_vs_27k=margin_range(ka, kb, 2294),
                  p_range_13k_vs_50k=margin_range(ka, kc, 2294), wilson_13k=wilson(ka, 2294), wilson_50k=wilson(kc, 2294))
out["table2_context_margin_bounds"] = t2o
# 표 5 Claude 2 1.97이 어떤 k/2294에 해당하는가
out["claude2_1_97_candidates"] = {k: round(100 * k / 2294, 3) for k in (45, 46)}

# ---------- 9. 라벨 잡음 외삽: 주석 표본이 전체의 무작위 표본이라고 가정할 때 걸러진 문항의 해결률 ----------
# overall = kept * r_kept + (1-kept) * r_filt, kept = 1 - 0.683, r_kept = Verified 해결률(500문항이 주석 통과분 539개 중 선택됨)
ext = {}
for m in MODELS:
    k_all = len(res[("test", m)]["resolved"]) / 2294
    kv = len(res[("verified", m)]["resolved"])
    lo, hi = wilson(kv, 500)
    row = {}
    for lab, rk in (("point", kv / 500), ("wilson_lo", lo / 100), ("wilson_hi", hi / 100)):
        row[lab] = round(100 * (k_all - 0.317 * rk) / 0.683, 2)
    ext[m] = dict(r_all_pct=round(100 * k_all, 2), r_kept_pct=round(100 * kv / 500, 2), r_filtered_implied_pct=row)
out["implied_filtered_resolve_rate"] = ext

# ---------- 10. HF 데이터 대 v2.0.0 번들 데이터(2024-06-27, swebench_updated.json) ----------
bundle = ROOT / "repo" / "v2.0.0" / "data" / "swebench_updated.json"
if bundle.exists():
    a = pd.DataFrame(json.load(open(bundle))).set_index("instance_id")
    b = full.set_index("instance_id")
    def _L(v):
        return json.loads(v) if isinstance(v, str) else list(v)
    diff = [i for i in b.index if not (_L(a.loc[i, "PASS_TO_PASS"]) == b.loc[i, "PASS_TO_PASS"] and _L(a.loc[i, "FAIL_TO_PASS"]) == b.loc[i, "FAIL_TO_PASS"]
                                      and a.loc[i, "patch"] == b.loc[i, "patch"] and a.loc[i, "test_patch"] == b.loc[i, "test_patch"])]
    out["hf_vs_v2_0_0_bundle"] = dict(n=len(b), identical=len(b) - len(diff), differing=diff)

# ---------- 11. 테스트 패치의 구조: 새 파일을 만드는 문항, 되돌리기 정규식(--- a/)이 잡는 파일 ----------
def tp_files(tp):
    news = len(re.findall(r"^--- /dev/null", tp, re.M))
    olds = len(re.findall(r"^--- a/", tp, re.M))
    return news, olds
tf = full.test_patch.map(tp_files)
full["tp_new"] = [a for a, b in tf]
full["tp_old"] = [b for a, b in tf]
out["test_patch_structure"] = dict(
    instances_with_new_test_file=int((full.tp_new > 0).sum()), share=round(float((full.tp_new > 0).mean()), 3),
    instances_only_new_files=int(((full.tp_new > 0) & (full.tp_old == 0)).sum()),
    mean_files_in_test_patch=round(float((full.tp_new + full.tp_old).mean()), 2),
    instances_touching_nontest_ext=int(full.test_patch.str.contains(r"^diff --git a/\S+\.(?:md|rst|txt|json|png)\b", regex=True, flags=re.M).sum()))

(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
print("done")
