"""논문 표 2·3(원문 paper.txt)을 읽어 (1) 표 안 일관성, (2) 공개 실행별 원출력 재계산(recompute_results.json)과의 일치를 검사한다.

(1) 표 안 일관성(논문 정의에서 항상 성립해야 하는 관계)
  - TARr <= TARa 이어야 한다(문자열이 모두 같으면 파싱한 답도 같다). 어기면 파서 실패 처리 때문이다.
  - BestAcc - WorstAcc <= 100 - TARa 이어야 한다(정답 여부가 갈린 문항은 모두 답이 갈린 문항이다).
  - WorstAcc <= 중앙값 <= BestAcc.
  - 모든 정확도는 k/문항수(소수 첫째 자리 반올림)여야 한다.
(2) 표의 각 칸이 공개 배치(날짜) 중 어느 것과 일치하는가.

실행: python3 -I analysis/table_check.py   (먼저 recompute_runs.py)
출력: analysis/table_check_results.json
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TXT = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
RES = json.loads((ROOT / "analysis" / "recompute_results.json").read_text())
OUT = ROOT / "analysis" / "table_check_results.json"

MODELS = ["gpt-35-turbo", "gpt-4o", "llama3-8b", "llama3-70b", "mixtral-8x7b"]
TASKS = {"navigation": "navigate", "geo. shapes": "geometric_shapes", "logical deduct.": "logical_deduction",
         "public rel.": "public_relations", "Europ. hist.": "high_school_european_history",
         "ruin names": "ruin_names", "prof. account": "professional_accounting", "college math": "college_mathematics"}
NQ = {"navigate": 250, "geometric_shapes": 250, "logical_deduction": 250, "ruin_names": 250,
      "high_school_european_history": 165, "college_mathematics": 100, "professional_accounting": 282,
      "public_relations": 110}


def parse_table(title_marker_start, title_marker_end):
    seg = TXT[TXT.index(title_marker_start):TXT.index(title_marker_end)]
    acc, tar = {}, {}
    mode = "acc"
    for line in seg.splitlines():
        if "TAR Results" in line:
            mode = "tar"
            continue
        for name, key in TASKS.items():
            if line.strip().startswith(name):
                nums = [float(x) for x in re.findall(r"\d+\.\d*", line[len(line) - len(line.lstrip()) + len(name):])]
                (acc if mode == "acc" else tar)[key] = nums
    return acc, tar


# 표 2(few-shot)는 "Task gpt3.5 ..." 머리글 ~ "Table 2:" 사이, 표 3(zero-shot)은 부록 머리글 ~ "Table 3:" 사이
T2 = parse_table("Task                  gpt3.5             gpt4o           llama8b", "Table 2:")
T3 = parse_table("Task                  gpt3.5              gpt4o           llama8b", "Table 3:")


def cond_values(table, task, mi):
    acc, tar = table
    a = acc[task][3 * mi:3 * mi + 3]  # best, median, worst
    t = tar[task][2 * mi:2 * mi + 2]  # TARa, TARr
    return dict(best=a[0], median=a[1], worst=a[2], TARa=t[0], TARr=t[1])


def batch_values(r):
    n = r["n_questions"]
    pr = np.array(r["correct_count_per_run"]) / n * 100
    return dict(best=r["best_count"] / n * 100, median=float(np.median(pr)), worst=r["worst_count"] / n * 100,
                TARa=r["TACa"] / n * 100, TARr=r["TACr"] / n * 100, maxmin=float(pr.max() - pr.min()))


def close(a, b):
    return abs(a - b) <= 0.051


out = {"internal": [], "vs_batches": []}
for tname, table, shots in [("표 2 few-shot", T2, "few"), ("표 3 zero-shot", T3, "0")]:
    for mi, model in enumerate(MODELS):
        for task in NQ:
            v = cond_values(table, task, mi)
            n = NQ[task]
            probs = []
            if v["TARr"] > v["TARa"] + 1e-9:
                probs.append("TARr>TARa")
            if v["best"] - v["worst"] > 100 - v["TARa"] + 0.051:
                probs.append("Best-Worst>100-TARa")
            if not (v["worst"] - 0.051 <= v["median"] <= v["best"] + 0.051):
                probs.append("median 순서")
            for k in ("best", "median", "worst", "TARa", "TARr"):
                x = v[k]
                if not any(abs(round(j / n * 100, 1) - x) < 0.051 or abs(round(j / n * 100 * 2) / 2 - x) < 0.051 for j in range(0, n + 1)):
                    # 중앙값은 두 실행의 평균일 수 있어 j/n 사이 값도 허용
                    if k != "median":
                        probs.append(f"{k}={x} 는 k/{n} 격자에 없음")
            out["internal"].append(dict(table=tname, model=model, task=task, **v, problems=probs))
            cands = [r for r in RES if r["model"] == model and r["task"] == task and str(r["shots"]) == shots]
            best_match, rows = None, []
            for r in cands:
                b = batch_values(r)
                nmatch = sum(close(v[k], b[k]) for k in ("best", "median", "worst", "TARa", "TARr"))
                rows.append(dict(date=r["date"], matches=nmatch, **{k: round(b[k], 1) for k in b}))
            rows.sort(key=lambda x: -x["matches"])
            out["vs_batches"].append(dict(table=tname, model=model, task=task, paper=v, n_batches=len(cands),
                                          best_batch=rows[0] if rows else None, all_batches=rows))

# 요약
sumry = {}
for tname in ["표 2 few-shot", "표 3 zero-shot"]:
    rows = [x for x in out["vs_batches"] if x["table"] == tname]
    full = sum(1 for x in rows if x["best_batch"] and x["best_batch"]["matches"] == 5)
    ge4 = sum(1 for x in rows if x["best_batch"] and x["best_batch"]["matches"] >= 4)
    tarr_gt = [(x["model"], x["task"], x["TARa"], x["TARr"]) for x in out["internal"]
               if x["table"] == tname and "TARr>TARa" in x["problems"]]
    other = [(x["model"], x["task"], x["problems"]) for x in out["internal"] if x["table"] == tname
             and [p for p in x["problems"] if p != "TARr>TARa"]]
    sumry[tname] = dict(conditions=len(rows), five_of_five_match=full, ge_four_match=ge4,
                        TARr_gt_TARa=tarr_gt, other_problems=other)
out["summary"] = sumry
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(sumry, ensure_ascii=False, indent=1))
for x in out["vs_batches"]:
    if x["best_batch"] and x["best_batch"]["matches"] < 5:
        print("불일치", x["table"], x["model"], x["task"], x["paper"], "->", x["best_batch"], "batches", x["n_batches"])
