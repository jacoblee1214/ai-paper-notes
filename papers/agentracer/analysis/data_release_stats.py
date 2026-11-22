"""AgenTracer 공개 데이터 릴리스(v1.0.0)의 행 수, 라벨 분포, train/test 질문 중복을 센다.
실행: python3 -I analysis/data_release_stats.py  (입력: data/release/unz/agentracer-data-v1.0.0/)
결과: analysis/data_release_stats.json
"""
import json, os, collections, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "release", "unz", "agentracer-data-v1.0.0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_release_stats.json")
res = {}
def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
for dom in ["agentic", "coding", "math"]:
    d = {}
    sets = {}
    for split in ["train", "test"]:
        rows = load(os.path.join(ROOT, dom, f"{split}.jsonl"))
        sets[split] = rows
        agents = collections.Counter(str(r.get("mistake_agent")) for r in rows)
        steps = [r.get("mistake_step") for r in rows]
        nhist = [len(r.get("history") or []) for r in rows]
        step_is_int = sum(isinstance(s, int) or (isinstance(s, str) and s.strip().lstrip("-").isdigit()) for s in steps)
        d[split] = {
            "rows": len(rows),
            "unique_question_ID": len({r.get("question_ID") for r in rows}),
            "unique_question_text": len({r.get("question") for r in rows}),
            "top_mistake_agent": agents.most_common(6),
            "n_distinct_agent_labels": len(agents),
            "mistake_step_int_like": step_is_int,
            "mistake_step_min_max": [min(s for s in steps if isinstance(s,int)) if any(isinstance(s,int) for s in steps) else None,
                                     max(s for s in steps if isinstance(s,int)) if any(isinstance(s,int) for s in steps) else None],
            "history_len_median": sorted(nhist)[len(nhist)//2] if nhist else None,
        }
    tr_q = {r.get("question") for r in sets["train"]}
    tr_id = {r.get("question_ID") for r in sets["train"]}
    te = sets["test"]
    d["overlap"] = {
        "test_rows_with_question_text_in_train": sum(r.get("question") in tr_q for r in te),
        "test_rows_with_question_ID_in_train": sum(r.get("question_ID") in tr_id for r in te),
    }
    res[dom] = d
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
