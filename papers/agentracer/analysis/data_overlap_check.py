"""math 테스트 질문이 train에도 있는지, 있으면 정답 단계와 주체 라벨이 같은지 센다.
또 repo의 data/tracertraj-code-test.parquet 행 수와 릴리스 coding/test.jsonl을 비교한다.
실행: python3 -I analysis/data_overlap_check.py -> analysis/data_overlap_check.json
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "data", "release", "unz", "agentracer-data-v1.0.0")
OUT = os.path.join(HERE, "data_overlap_check.json")
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
res = {}
for dom in ["math", "coding"]:
    tr = load(os.path.join(ROOT, dom, "train.jsonl")); te = load(os.path.join(ROOT, dom, "test.jsonl"))
    by_q = {}
    for r in tr: by_q.setdefault(r["question"], []).append(r)
    shared = [r for r in te if r["question"] in by_q]
    same_step = sum(any(str(x["mistake_step"]) == str(r["mistake_step"]) for x in by_q[r["question"]]) for r in shared)
    same_agent = sum(any(str(x["mistake_agent"]) == str(r["mistake_agent"]) for x in by_q[r["question"]]) for r in shared)
    res[dom] = {"test_rows": len(te), "test_rows_sharing_question_text_with_train": len(shared),
                "of_those_some_train_copy_has_same_mistake_step": same_step,
                "of_those_some_train_copy_has_same_mistake_agent": same_agent,
                "train_rows_per_question_max": max(len(v) for v in by_q.values())}
try:
    import pandas as pd
    pq = os.path.join(HERE, "..", "data", "..", "repo", "data", "tracertraj-code-test.parquet")
    df = pd.read_parquet(pq)
    res["repo_parquet_rows"] = len(df); res["repo_parquet_columns"] = list(df.columns)
except Exception as e:
    res["repo_parquet_error"] = repr(e)[:200]
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
