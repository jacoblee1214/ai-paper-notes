"""repo/ (공식 코드 저장소 clone)의 output_data에서 HumanEval 과제별 통과 여부를 뽑아
analysis/source_data/pertask_humaneval.json 으로 저장한다. 실행 없이 저장된 결과 파일만 읽는다.
사용: python3 -I analysis/extract_pertask.py   (repo/ 가 있어야 함)
"""
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LDB = os.path.join(ROOT, "repo/LLMDebugger/output_data")
REF = os.path.join(ROOT, "repo/reflexion/output_data/reflexion/humaneval")

# 별칭 -> (디렉터리 패턴(run*/ 앞까지), CSV의 (strategy, model))
AGENTS = {
    "GPT-4 zero-shot":      (f"{LDB}/simple/humaneval/gpt-4-turbo-2024-04-09", ("simple", "gpt-4-turbo-2024-04-09")),
    "GPT-3.5 zero-shot":    (f"{LDB}/simple/humaneval/gpt-3.5-turbo-0125", ("simple", "gpt-3.5-turbo-0125")),
    "Retry (GPT-4)":        (f"{LDB}/simple_repeat/humaneval/gpt-4-turbo-0409-5repeats-no-temp-incr", ("simple_repeat", "gpt-4-turbo-0409-5repeats-no-temp-incr")),
    "Warming (GPT-4)":      (f"{LDB}/simple_incr_temp/humaneval/gpt-4-turbo-0409-5step", ("simple_incr_temp", "gpt-4-turbo-0409-5step")),
    "Escalation":           (f"{LDB}/simple_boosting/humaneval/l3_8b-gpt35-l3_70b-gpt4-0409", ("simple_boosting", "l3_8b-gpt35-l3_70b-gpt4-0409")),
    "Reflexion (GPT-4)":    (f"{REF}/gpt-4-turbo-2024-04-09", ("reflexion", "gpt-4-turbo-2024-04-09")),
    "LDB (GPT-4)":          (f"{LDB}/ldb/humaneval/gpt-4-turbo-2024-04-09+gpt-4-turbo-2024-04-09", ("ldb", "gpt-4-turbo-2024-04-09+gpt-4-turbo-2024-04-09")),
    "LDB (Reflexion, GPT-4)": (f"{LDB}/ldb/humaneval/gpt-4-turbo-2024-04-09+reflexion", ("ldb", "gpt-4-turbo-2024-04-09+reflexion")),
    "LDB (GPT-4, GPT-3.5)": (f"{LDB}/ldb/humaneval/gpt-3.5-turbo-0125+gpt-4-turbo-2024-04-09", ("ldb", "gpt-3.5-turbo-0125+gpt-4-turbo-2024-04-09")),
    "LDB (GPT-3.5)":        (f"{LDB}/ldb/humaneval/gpt-3.5-turbo-0125+gpt-3.5-turbo-0125", ("ldb", "gpt-3.5-turbo-0125+gpt-3.5-turbo-0125")),
    "LDB (Reflexion, GPT-3.5)": (f"{LDB}/ldb/humaneval/gpt-3.5-turbo-0125+reflexion", ("ldb", "gpt-3.5-turbo-0125+reflexion")),
}

def norm(t):
    # Reflexion 출력은 MultiPL-E 형식 이름(HumanEval_23_strlen)과 task_id(HumanEval/32)가 섞여 있다
    return "HumanEval/" + t.split("_")[1] if t.startswith("HumanEval_") else t


out = {}
for name, (d, key) in AGENTS.items():
    runs = {}
    for r in sorted(glob.glob(d + "/run*")):
        fs = [f for f in glob.glob(r + "/*.jsonl")]
        if len(fs) != 1:
            print("WARN", name, r, fs, file=sys.stderr); continue
        rows = [json.loads(x) for x in open(fs[0])]
        ids = [norm(x.get("task_id") or x.get("name")) for x in rows]
        runs[os.path.basename(r)] = {"ids": ids, "solved": [bool(x["is_solved"]) for x in rows]}
    out[name] = {"csv_key": list(key), "runs": runs}
    print(name, {k: (len(v["solved"]), sum(v["solved"])) for k, v in runs.items()})
json.dump(out, open(os.path.join(ROOT, "analysis/source_data/pertask_humaneval.json"), "w"))

# ---------------------------------------------------------------- HotPotQA (표 A3): 과제별 결과를 compact JSON으로
HP = os.path.join(ROOT, "repo/HotpotQA/retrieval_score")
hp = {}
for model, pat in [("GPT-3.5", "gpt_3.5*eval_results.json"), ("Llama-3-70B", "llama3_70B*eval_results.json")]:
    hp[model] = {}
    for dname, sub in [("uncompiled", "uncompiled"), ("format", "formatting_instr_only"), ("few_shot", "few-shot"),
                       ("random_search", "bootstrap_random_search"), ("joint", "joint_optim")]:
        f = glob.glob(f"{HP}/{sub}/{pat}")[0]
        E = json.load(open(f))
        hp[model][dname] = []
        for e in E:
            runs = {}
            for rk in e["eval_gold_passages_retrieved_accuracies"]:
                runs[rk] = {
                    "gold": [bool(t["gold_passages_retrieved"]) for t in e["per_task_results"][rk]],
                    "em": e["eval_accuracies"][rk], "cost_200": e["eval_costs"][rk],
                }
            hp[model][dname].append({"compile_cost": e["compile_cost"], "dev_acc": e.get("trainset_acc"), "runs": runs})
json.dump(hp, open(os.path.join(ROOT, "analysis/source_data/hotpotqa_compact.json"), "w"))
print("hotpotqa compact saved")
