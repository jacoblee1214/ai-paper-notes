"""공식 저장소(repo/, 정적)의 Table 2용 시험 파일을 읽어 논문이 말한 '비교환 요소'와 입력 형식을 확인한다. 코드는 실행하지 않는다.
입력: repo/benchmarks/benchmarks.zip (4.4 MB). 결과: analysis/repo_checks_results.json
실행: python3 -I analysis/repo_checks.py   (repo/가 있어야 함: git clone --depth 1 https://github.com/tatsu-lab/test_set_contamination repo)
"""
import json
import re
import zipfile
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
z = zipfile.ZipFile(ROOT / "repo" / "benchmarks" / "benchmarks.zip")


def lines(name):
    return z.read(name).decode("utf-8").splitlines(keepends=True)


res = {}
for name in ["boolq/dev.jsonl", "arc/ARC-Easy-Test.jsonl", "piqa/tests.jsonl", "gsm8k/test.jsonl", "lambada/lambada_test_plain_text.txt"]:
    L = lines(name)
    first = L[:5000]  # 공식 코드의 max_examples=5000 절단
    res[name] = {"n_lines": len(L), "n_unique": len(set(L)), "n_used_first5000": len(first),
                 "n_unique_in_first5000": len(set(first)), "all_end_with_newline": all(x.endswith("\n") for x in L[:-1])}
ids = [json.loads(l)["id"] for l in lines("arc/ARC-Easy-Test.jsonl")]
nums = [int(re.findall(r"\d+", i)[-1]) for i in ids]
rho = spearmanr(range(len(nums)), nums)
res["arc_id_order_spearman"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
mm = [n for n in z.namelist() if n.startswith("mmlu_json/") and n.endswith(".json")]
sizes = {n.split("/")[1][:-5]: len(json.loads(z.read(n))) for n in mm}
elem = json.loads(z.read("mmlu_json/business_ethics.json"))
res["mmlu"] = {"n_files": len(sizes), "total_examples": sum(sizes.values()),
               "professional_law_in_zip": "professional_law" in sizes, "professional_law_n": sizes.get("professional_law"),
               "element_ends_with_newline": elem[0].endswith("\n")}
Path(__file__).with_name("repo_checks_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
