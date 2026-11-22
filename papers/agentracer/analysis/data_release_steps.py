"""공개 릴리스에서 mistake_step 값 형식과 범위, 이력 길이를 확인한다.
실행: python3 -I analysis/data_release_steps.py  -> analysis/data_release_steps.json
"""
import json, os, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "release", "unz", "agentracer-data-v1.0.0")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_release_steps.json")
out = {}
for dom in ["agentic", "coding", "math"]:
    for split in ["train", "test"]:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, dom, f"{split}.jsonl"), encoding="utf-8") if l.strip()]
        types = collections.Counter(type(r["mistake_step"]).__name__ for r in rows)
        vals = []
        for r in rows:
            try: vals.append(int(r["mistake_step"]))
            except Exception: pass
        h = [len(r["history"]) for r in rows]
        in_range = sum(0 <= v < hl for v, hl in zip(vals, h)) if len(vals) == len(h) else None
        out[f"{dom}/{split}"] = {
            "mistake_step_types": dict(types),
            "step_min": min(vals) if vals else None, "step_max": max(vals) if vals else None,
            "history_len_min": min(h), "history_len_max": max(h),
            "step_within_history_bounds": in_range,
            "step0_share": round(sum(v == 0 for v in vals) / len(vals), 4) if vals else None,
            "history_entry_keys": sorted(set(k for r in rows[:50] for e in r["history"] for k in (e.keys() if isinstance(e, dict) else ["<str>"])))[:12],
        }
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
