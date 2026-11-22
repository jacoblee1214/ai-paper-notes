"""APPS GPT-4 원자료(all-gpt4-repair-results-tokens.jsonl, 저자 공개)로 논문 Figure 3(b)을 정확 기댓값으로 재계산하고,
과제(task) 단위 부트스트랩으로 개선분의 불확실성을 본다.

모델 (논문 4절, 저자 코드 paper/analysis/sample-and-estimate.py의 batch 모드, --with-replacement와 같은 식):
- 과제 t의 기본 프로그램 50개 중 통과 비율 p_t.
- 수리 기반 self-repair (np, nfr=1): np번 복원추출한 기본 프로그램 중 하나라도 통과하면 성공.
  실패한 기본 프로그램 b마다 수리 후보 중 통과 비율 q_b, nfr번 복원추출 → 실패 확률 (1-q_b)^nfr.
  한 번 뽑기에서 '실패하고 수리도 모두 실패'할 확률 a_t = (1/50) * sum_{b 실패} (1-q_b)^nfr,
  성공 확률 S_sr(t) = 1 - a_t^np.
- 기준선 pass@k (k = np*(1+nfr)): S_base(t,k) = 1 - (1-p_t)^k.
- 정규화 값 = mean_t S_sr / mean_t S_base(k). 논문 히트맵 셀과 같은 정의다.

검산: S_base(t,1)의 평균이 processed JSON의 (1,0) 값 0.39502와 맞는지 본다.
사용: python3 -I apps_gpt4_exact_check.py <all-gpt4-repair-results-tokens.jsonl>
"""
import json
import random
import sys
from pathlib import Path

path = sys.argv[1]
tasks = []
with open(path) as f:
    for line in f:
        d = json.loads(line)
        comps = d["completions"]
        n = len(comps)
        passed = [bool(c["binary"]) or c["fault"] == "passed" for c in comps]
        p = sum(passed) / n
        rep_lens = sorted({len(c["repairs"]) for c in comps if c["repairs"]})
        qs = []
        for c, ok in zip(comps, passed):
            if ok:
                continue
            reps = c["repairs"] or []
            qs.append(sum(bool(r["binary"]) for r in reps) / len(reps) if reps else 0.0)
        tasks.append({"prob_path": d["prob_path"], "n": n, "p": p, "qs": qs, "rep_lens": rep_lens})

print("tasks:", len(tasks), " completions per task:", sorted({t["n"] for t in tasks}))
print("repair list lengths (failing programs):", sorted({x for t in tasks for x in t["rep_lens"]}))


def s_base(t, k):
    return 1 - (1 - t["p"]) ** k


def s_sr(t, np_, nfr):
    a = sum((1 - q) ** nfr for q in t["qs"]) / t["n"]
    return 1 - a ** np_


def mean(xs):
    return sum(xs) / len(xs)


res = {"n_tasks": len(tasks)}
res["baseline_check_k1_mean"] = mean([s_base(t, 1) for t in tasks])
res["processed_json_(1,0)"] = 0.39502000000000004

grid = {}
for nfr in [1, 3, 5, 10]:
    for np_ in [1, 2, 5, 10, 25]:
        k = np_ * (1 + nfr)
        if k > 50:
            continue
        sr = mean([s_sr(t, np_, nfr) for t in tasks])
        base = mean([s_base(t, k) for t in tasks])
        grid[f"np={np_},nfr={nfr}"] = {"k": k, "sr": sr, "base": base, "norm": sr / base}
res["grid"] = grid

# 과제 단위 부트스트랩(10,000회): 대표 설정 두 개의 정규화 값과 차이의 95% 구간
rng = random.Random(0)
focus = [(10, 1), (2, 10), (25, 1)]
boot = {f: [] for f in focus}
for _ in range(10000):
    samp = [tasks[rng.randrange(len(tasks))] for _ in tasks]
    for np_, nfr in focus:
        k = np_ * (1 + nfr)
        sr = mean([s_sr(t, np_, nfr) for t in samp])
        base = mean([s_base(t, k) for t in samp])
        boot[(np_, nfr)].append(sr / base)
res["task_bootstrap_norm_95ci"] = {}
for f, xs in boot.items():
    xs.sort()
    res["task_bootstrap_norm_95ci"][f"np={f[0]},nfr={f[1]}"] = [xs[250], xs[9750]]

# 같은 과제에서 쌍을 이룬 차이(sr - base)의 과제 단위 표준오차 (np=10, nfr=1)
diffs = [s_sr(t, 10, 1) - s_base(t, 20) for t in tasks]
m = mean(diffs)
sd = (sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5
res["paired_diff_np10_nfr1"] = {"mean_diff": m, "task_se": sd / len(diffs) ** 0.5,
                                "ci95": [m - 1.96 * sd / len(diffs) ** 0.5, m + 1.96 * sd / len(diffs) ** 0.5]}

# Table 2 검산: 수리 후보 중 통과 비율 (전체 수리 후보 기준)
tot = passing = 0
with open(path) as f:
    for line in f:
        for c in json.loads(line)["completions"]:
            for r in c["repairs"] or []:
                tot += 1
                passing += bool(r["binary"])
res["repair_success_rate_all_repairs"] = passing / tot
res["repair_total"] = tot

out = Path(__file__).with_name("apps_gpt4_exact_results.json")
out.write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=2))
for k, v in grid.items():
    print(k, {x: round(y, 4) for x, y in v.items()})
