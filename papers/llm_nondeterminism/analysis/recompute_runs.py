"""공개 실행별 원출력(repo/experiments/v3/runs.tgz)에서 논문 지표를 다시 계산한다.

공식 코드는 실행하지 않는다. evaluate.py와 helper_functions.py의 답 추출 규칙(정적으로 읽은 내용)을
직접 다시 구현하고, 공개 평가 결과 stability_eval.csv와 조건별로 대조한다.

입력: data/v3/all_runs/<조건 폴더>/*.csv (10회 실행, 문항별 응답), repo/experiments/v3/stability_eval.csv
출력: analysis/recompute_results.json (조건별 요약과 문항x실행 정답 행렬)
실행: python3 -I analysis/recompute_runs.py
"""
import ast
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "v3" / "all_runs"
EVAL_CSV = ROOT / "repo" / "experiments" / "v3" / "stability_eval.csv"
OUT = ROOT / "analysis" / "recompute_results.json"

csv.field_size_limit(sys.maxsize)


class BlownUP(LookupError):
    pass


def parse_parenthesized(answers, response):
    """helper_functions.parse_parenthesized_answers 와 같은 규칙(정적으로 읽고 다시 구현)."""
    matches = set()
    backoff = ["answer is", "boxed choice", "exact_word", "case_insensitive", "sub_string", "llm"]
    sents = response.split("\n")
    salience = [sents[-1:]]
    if len(sents) > 1:
        salience.append(sents[-3:])
    if len(sents) > 3:
        salience.append(sents)
    for ss in salience:
        text = "\n".join(ss)
        for mt in backoff:
            for a in answers:
                if mt == "answer is":
                    pat = r"(T|t)he answer is " + a.replace("(", r"\(").replace(")", r"\)")
                elif mt == "exact_word":
                    pat = a.replace("(", r"\(").replace(")", r"\)")
                elif mt == "case_insensitive":
                    pat = a.replace(")", r"\)").replace("(", r"(?i)\(")
                elif mt == "sub_string":
                    pat = a.replace(")", r"\b").replace("(", r"(?i)\b")
                else:
                    continue
                if re.search(pat, text):
                    matches.add(a)
            if len(matches) == 1:
                return matches.pop()
            if len(matches) > 1:
                raise BlownUP(text)
    return None


def parse_string(answers, response):
    """helper_functions.parse_string (navigate)."""
    if not isinstance(response, str):
        return None
    matches = set()
    for mt in ["exact_word", "case_insensitive"]:
        for a in answers:
            if mt == "exact_word" and re.search(rf"\b{a}\b", response):
                matches.add(a)
            if mt == "case_insensitive" and re.search(rf"(?i)\b{a}\b", response):
                matches.add(a)
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            raise BlownUP(response)
    return None


def answer_fn(task, response):
    if task == "navigate":
        return parse_string(["Yes", "No"], response)
    letters = ["(A)", "(B)", "(C)", "(D)"]
    if task == "geometric_shapes":
        letters = [f"({c})" for c in "ABCDEFGHIJK"]
    return parse_parenthesized(letters, response)


def correct_fn(task, response, gt, answer):
    if task == "college_mathematics":  # JSON 응답 특수 처리(tasks/college_mathematics.py)
        try:
            a = json.loads(response)["Answer"]
            return a == gt or a == re.sub(r"[()]", "", gt)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    return gt == answer


def evaluate_dir(d: Path):
    files = sorted(d.glob("*.csv"), key=lambda p: int(re.search(r"-(\d+)\.csv$", p.name).group(1)))
    runs = []
    meta = None
    for f in files:
        df = pd.read_csv(f, usecols=["model", "task", "task_config", "model_config", "rubric_id", "gt", "run", "response", "date"])
        runs.append(df)
    n_runs = len(runs)
    first = runs[0]
    meta = dict(model=first["model"].iloc[0], task=first["task"].iloc[0],
                shots=json.loads(first["task_config"].iloc[0])["shots"], date=first["date"].iloc[0],
                top_p_k=json.loads(first["model_config"].iloc[0])["top_p_k"])
    task = meta["task"]
    qids = sorted(first["rubric_id"].unique())
    nq = len(qids)
    # 문항 x 실행
    resp = {r: runs[r].set_index("rubric_id")["response"] for r in range(n_runs)}
    gts = first.set_index("rubric_id")["gt"]
    correct = np.zeros((nq, n_runs), dtype=int)
    tar_raw = tar_ans = 0
    n_parse_none = n_blown = 0
    raw_same_but_ans_disagree = 0
    tacr_by_N = {k: 0 for k in range(2, n_runs + 1)}  # 앞 N회 실행만 썼을 때 원출력이 모두 같은 문항 수
    n_nan = 0
    n_q_parse_issue = 0  # 응답 하나라도 답 추출에 실패(복수 답 또는 없음)한 문항 수
    for qi, q in enumerate(qids):
        raw, ans = set(), set()
        seen = set()
        raw_list = [resp[r].loc[q] for r in range(n_runs)]
        for k in tacr_by_N:
            if len(set(raw_list[:k])) == 1:
                tacr_by_N[k] += 1
        issue = False
        for r in range(n_runs):
            s = raw_list[r]
            raw.add(s)
            if not isinstance(s, str):
                n_nan += 1
                continue
            try:
                a = answer_fn(task, s)
            except BlownUP:
                ans.add(("blown", r))
                n_blown += 1
                issue = True
                continue
            if a is None:  # evaluate.py: 문항당 첫 실패만 ans에 들어가고 나머지는 건너뜀
                n_parse_none += 1
                issue = True
                if "none" not in seen:
                    seen.add("none")
                    ans.add(("none", r))
                continue
            ans.add(a)
            if correct_fn(task, s, gts.loc[q], a):
                correct[qi, r] = 1
        n_q_parse_issue += issue
        if len(raw) == 1:
            tar_raw += 1
        if len(ans) == 1:
            tar_ans += 1
        if len(raw) == 1 and len(ans) != 1:
            raw_same_but_ans_disagree += 1
    per_run = correct.sum(0).tolist()
    best = int((correct.sum(1) > 0).sum())
    worst = int((correct.sum(1) == n_runs).sum())
    return meta, dict(n_questions=nq, N=n_runs, TACr=tar_raw, TACa=tar_ans, correct_count_per_run=per_run,
                      best_count=best, worst_count=worst, n_parse_none=n_parse_none, n_blown=n_blown, n_nan=n_nan, n_q_parse_issue=n_q_parse_issue, tacr_by_N=tacr_by_N,
                      raw_same_but_ans_disagree=raw_same_but_ans_disagree), correct


def main():
    ev = pd.read_csv(EVAL_CSV, index_col=0)
    ev["shots"] = ev["task_config"].apply(lambda s: ast.literal_eval(s)["shots"])
    ev["cc"] = ev["correct_count_per_run"].apply(ast.literal_eval)
    results = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or not list(d.glob("*.csv")):
            continue
        meta, res, mat = evaluate_dir(d)
        sel = ev[(ev["model"] == meta["model"]) & (ev["task"] == meta["task"]) & (ev["date"] == meta["date"])]
        sel = sel[sel["shots"].astype(str) == str(meta["shots"])]
        row = {"dir": d.name, **meta, **res}
        if len(sel) == 1:
            e = sel.iloc[0]
            row["csv_match"] = dict(
                TACr=bool(e["TACr"] == res["TACr"]), TACa=bool(e["TACa"] == res["TACa"]),
                per_run=bool(e["cc"] == res["correct_count_per_run"]),
                best=bool(round(float(e["best_possible_accuracy"].strip("%")) * res["n_questions"] / 100) == res["best_count"]),
                worst=bool(round(float(e["worst_possible_accuracy"].strip("%")) * res["n_questions"] / 100) == res["worst_count"]),
            )
            row["csv_TACa"] = int(e["TACa"])
            row["csv_per_run"] = e["cc"]
        else:
            row["csv_match"] = None
        row["matrix"] = mat.tolist()
        results.append(row)
        print(meta["model"], meta["task"], meta["shots"], meta["date"], row["csv_match"], flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False))
    ok = [r for r in results if r["csv_match"]]
    print("conditions", len(results), "matched-in-csv", len(ok))
    for k in ["TACr", "TACa", "per_run", "best", "worst"]:
        print(k, sum(r["csv_match"][k] for r in ok), "/", len(ok))


if __name__ == "__main__":
    main()
