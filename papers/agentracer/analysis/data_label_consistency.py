"""라벨 일관성: mistake_step이 history의 step 필드와 맞는지, 그 단계의 name이 mistake_agent와 같은지 센다.
실행: python3 -I analysis/data_label_consistency.py -> analysis/data_label_consistency.json
"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "release", "unz", "agentracer-data-v1.0.0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_label_consistency.json")
out = {}
for dom in ["agentic", "coding", "math"]:
    for split in ["train", "test"]:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, dom, f"{split}.jsonl"), encoding="utf-8") if l.strip()]
        n = len(rows); step_found = 0; agent_match = 0; agent_checkable = 0; examples = []
        for r in rows:
            s = str(r["mistake_step"]).strip()
            ents = [e for e in r["history"] if str(e.get("step")) == s]
            if ents:
                step_found += 1
                if "name" in ents[0]:
                    agent_checkable += 1
                    if str(ents[0]["name"]) == str(r["mistake_agent"]):
                        agent_match += 1
                    elif len(examples) < 3:
                        examples.append({"qid": r["question_ID"], "label": r["mistake_agent"], "entry_name": ents[0]["name"]})
        out[f"{dom}/{split}"] = {"rows": n, "step_value_found_in_history_step_field": step_found,
                                 "agent_name_checkable": agent_checkable, "agent_label_equals_step_actor": agent_match,
                                 "mismatch_examples": examples}
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
