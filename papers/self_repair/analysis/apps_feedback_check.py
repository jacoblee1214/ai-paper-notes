"""논문 Figure 5(APPS)의 핵심 대비를 원자료로 재계산한다: 피드백 모델을 바꿨을 때 수리 성능이 달라지는가.

- 기준선: 각 파일 자체의 pass@k (저자 코드처럼 파일마다 자기 (k,0) 기준선으로 정규화, k = np*2)
- A: GPT-3.5 자기 수리 (같은 파일의 기본 프로그램과 수리 후보, nfr=1)
- B: GPT-3.5 기본 프로그램 + GPT-4 피드백 (mp-chatgpt-mf-gpt4-nf-10 파일, nfr=1)
정규화 값은 analysis/apps_gpt4_exact_check.py와 같은 정확 기댓값 모델이다(복원추출, 과제 평균).
과제 단위 부트스트랩 95% 구간을 함께 낸다.

사용: python3 -I apps_feedback_check.py <base_jsonl> <mp_jsonl>
"""
import json
import random
import sys
from pathlib import Path


def load(path):
    tasks = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            comps = d["completions"]
            passed = [bool(c["binary"]) or c["fault"] == "passed" for c in comps]
            qs = []
            for c, ok in zip(comps, passed):
                if ok:
                    continue
                reps = c["repairs"] or []
                qs.append(sum(bool(r["binary"]) for r in reps) / len(reps) if reps else 0.0)
            tasks.append({"n": len(comps), "p": sum(passed) / len(comps), "qs": qs})
    return tasks


def s_base(t, k):
    return 1 - (1 - t["p"]) ** k


def s_sr(t, np_, nfr=1):
    a = sum((1 - q) ** nfr for q in t["qs"]) / t["n"]
    return 1 - a ** np_


def mean(xs):
    return sum(xs) / len(xs)


base_path, mp_path = sys.argv[1], sys.argv[2]
base_tasks = load(base_path)   # 기준선과 A(자기 수리)
mp_tasks = load(mp_path)       # B(GPT-4 피드백)
assert len(base_tasks) == len(mp_tasks) == 300
# A와 B는 같은 과제 순서여야 짝을 이룬다
res = {"n_tasks": 300, "rows": {},
       "mean_p_base_file": mean([t["p"] for t in base_tasks]), "mean_p_mp_file": mean([t["p"] for t in mp_tasks])}
rng = random.Random(0)
idx_boot = [[rng.randrange(300) for _ in range(300)] for _ in range(5000)]
for np_ in [1, 2, 5, 10, 25]:
    k = 2 * np_
    base = mean([s_base(t, k) for t in base_tasks])
    a = mean([s_sr(t, np_) for t in base_tasks]) / base
    base_mp = mean([s_base(t, k) for t in mp_tasks])
    b = mean([s_sr(t, np_) for t in mp_tasks]) / base_mp
    ra, rb, dd = [], [], []
    for ix in idx_boot:
        bb = mean([s_base(base_tasks[i], k) for i in ix])
        ra.append(mean([s_sr(base_tasks[i], np_) for i in ix]) / bb)
        bm = mean([s_base(mp_tasks[i], k) for i in ix])
        rb.append(mean([s_sr(mp_tasks[i], np_) for i in ix]) / bm)
        dd.append(rb[-1] - ra[-1])
    ra.sort(); rb.sort(); dd.sort()
    res["rows"][f"np={np_},k={k}"] = {
        "A_gpt35_self_repair_norm": round(a, 4), "A_ci95": [round(ra[125], 4), round(ra[4875], 4)],
        "B_gpt35_plus_gpt4_feedback_norm": round(b, 4), "B_ci95": [round(rb[125], 4), round(rb[4875], 4)],
        "B_minus_A_ci95": [round(dd[125], 4), round(dd[4875], 4)],
    }
out = Path(__file__).with_name("apps_feedback_results.json")
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
