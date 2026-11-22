"""공개 데이터셋(HF fjzzq2002/impossible_*)의 변형(mutation) 구조 감사.

- 실행 없이 텍스트와 AST만 본다. 참조 해답이 없으므로 "정말 불가능한가"는 구조로만 확인한다.
- 입력: data/lcb/*.parquet, data/swe/*.parquet  (README의 curl 명령으로 받는다)
- 출력: audit_dataset_results.json
"""
import ast
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "audit_dataset_results.json"


def load(kind):
    return {s: pd.read_parquet(DATA / kind / f"{s}.parquet") for s in ["original", "oneoff", "conflicting"]}


# ---------------------------------------------------------------- LiveCodeBench
def norm(node):
    return ast.dump(node, annotate_fields=False)


def assert_pairs(src):
    """check() 안의 `assert candidate(...) == RHS` 형태를 (호출 정규화, RHS 정규화) 목록으로."""
    tree = ast.parse(src)
    pairs = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare) and len(n.test.ops) == 1:
            left, right = n.test.left, n.test.comparators[0]
            op = type(n.test.ops[0]).__name__
            pairs.append((norm(left), op, norm(right)))
    return pairs


def lcb_audit(d):
    res = {"n": {k: len(v) for k, v in d.items()}}
    o, c, r = d["oneoff"], d["conflicting"], d["original"]
    res["same_task_ids_across_splits"] = bool((o.task_id == r.task_id).all() and (c.task_id == r.task_id).all())
    res["original_test_equals_original_split_test"] = {
        "oneoff": int((o.original_test == r.test).sum()),
        "conflicting": int((c.original_test == r.test).sum()),
    }
    # 구문 오류
    bad = {}
    for s in ["oneoff", "conflicting"]:
        bad[s] = sum(1 for t in d[s].test if not _parses(t))
    res["unparsable_mutated_tests"] = bad
    # check 함수 정의가 없는 변형(= check(...) 호출이 NameError) 또는 구문 오류인 변형: 정직한 풀이로는 물론, 테스트를 안 건드리면 어떤 풀이로도 통과 불가
    broken = {}
    for s in ["oneoff", "conflicting"]:
        ids = []
        for tid, t in zip(d[s].task_id, d[s].test):
            if "def check(" not in t or not _parses(t):
                ids.append(tid)
        broken[s] = ids
    res["structurally_broken_mutated_tests"] = broken

    # oneoff: 줄 단위 diff 유형
    types = Counter()
    n_changed_lines = []
    for ot, mt in zip(o.original_test, o.test):
        sm = difflib.SequenceMatcher(a=ot.splitlines(), b=mt.splitlines(), autojunk=False)
        rem = add = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                rem += i2 - i1
                add += j2 - j1
        n_changed_lines.append((rem, add))
        if (rem, add) == (1, 1):
            types["1줄 교체"] += 1
        elif rem == 0 and add >= 1:
            types["추가만"] += 1
        elif rem >= 1 and add >= 1:
            types["여러 줄 교체"] += 1
        elif rem >= 1 and add == 0:
            types["삭제만"] += 1
        else:
            types["변경 없음"] += 1
    res["oneoff_diff_shape"] = dict(types)
    res["oneoff_unchanged"] = int(sum(1 for ot, mt in zip(o.original_test, o.test) if ot == mt))

    # oneoff: 1줄 교체인데 비교 대상(RHS)만 바뀐 경우 / 비교 연산자가 바뀐 경우
    rhs_only = 0
    other = 0
    for ot, mt in zip(o.original_test, o.test):
        a = [x for x in ot.splitlines()]
        b = [x for x in mt.splitlines()]
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        ops = [x for x in sm.get_opcodes() if x[0] != "equal"]
        if len(ops) == 1 and ops[0][0] == "replace" and (ops[0][2] - ops[0][1]) == 1 and (ops[0][4] - ops[0][3]) == 1:
            la, lb = a[ops[0][1]].strip(), b[ops[0][3]].strip()
            try:
                ta = ast.parse(la).body[0]
                tb = ast.parse(lb).body[0]
                if isinstance(ta, ast.Assert) and isinstance(tb, ast.Assert) and norm(ta.test.left) == norm(tb.test.left) \
                        and type(ta.test.ops[0]) is type(tb.test.ops[0]):
                    rhs_only += 1
                else:
                    other += 1
            except Exception:
                other += 1
    res["oneoff_single_line_rhs_only"] = rhs_only
    res["oneoff_single_line_other"] = other

    # conflicting: 같은 호출(좌변)·같은 연산자에 서로 다른 RHS가 공존하는가 (AST로 감지)
    detect = 0
    unresolved = []
    added_only = 0
    for tid, ot, mt in zip(c.task_id, c.original_test, c.test):
        sm = difflib.SequenceMatcher(a=ot.splitlines(), b=mt.splitlines(), autojunk=False)
        ops = [x for x in sm.get_opcodes() if x[0] != "equal"]
        if all(x[0] == "insert" for x in ops):
            added_only += 1
        try:
            ps = assert_pairs(mt)
        except Exception:
            ps = []
        by_left = {}
        for l, op, rr in ps:
            by_left.setdefault((l, op), set()).add(rr)
        ok = any(len(v) >= 2 for (l, op), v in by_left.items() if op == "Eq")
        # `is None` 대 `is not None` 같은 쌍도 허용
        if not ok:
            by_left2 = {}
            for l, op, rr in ps:
                by_left2.setdefault(l, set()).add((op, rr))
            ok = any(len(v) >= 2 for v in by_left2.values())
        if ok:
            detect += 1
        else:
            unresolved.append(tid)
    res["conflicting_added_only"] = added_only
    res["conflicting_ast_contradiction_detected"] = detect
    res["conflicting_ast_not_detected"] = len(unresolved)
    res["conflicting_ast_not_detected_ids"] = unresolved
    return res


def _parses(src):
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------- SWE-bench
def patch_lines(p):
    return [l for l in p.splitlines() if not l.startswith("@@") and not l.startswith("index ")]


def swe_audit(d):
    res = {"n": {k: len(v) for k, v in d.items()}}
    o, c, r = d["oneoff"], d["conflicting"], d["original"]
    ids = set(r.instance_id)
    res["same_ids_across_splits"] = bool(ids == set(o.instance_id) == set(c.instance_id))
    res["repo_counts"] = dict(Counter(r.repo).most_common())
    res["difficulty_counts"] = dict(Counter(r.difficulty))
    res["original_test_patch_equals_original_split_test_patch"] = {
        "oneoff": int((o.set_index("instance_id").original_test_patch == r.set_index("instance_id").test_patch).sum()),
        "conflicting": int((c.set_index("instance_id").original_test_patch == r.set_index("instance_id").test_patch).sum()),
    }
    res["mutated_equals_original"] = {
        "oneoff": int((o.test_patch == o.original_test_patch).sum()),
        "conflicting": int((c.test_patch == c.original_test_patch).sum()),
    }

    def shape(df):
        """원본 test_patch 대비 변형 patch의 줄 단위 변화를 '+'/'-' 줄 다중집합 차이로 분류한다.

        P_add: 변형에만 있는 '+' 줄, P_rem: 원본에만 있는 '+' 줄, M_add: 변형에만 있는 '-' 줄(기존 테스트 줄 삭제/교체).
        """
        types = Counter()
        sizes = []
        for _, row in df.iterrows():
            def split(p):
                plus = Counter(l[1:] for l in p.splitlines() if l.startswith("+") and not l.startswith("+++"))
                minus = Counter(l[1:] for l in p.splitlines() if l.startswith("-") and not l.startswith("---"))
                return plus, minus
            pa, ma = split(row.original_test_patch)
            pb, mb = split(row.test_patch)
            p_add = sum((pb - pa).values())
            p_rem = sum((pa - pb).values())
            m_add = sum((mb - ma).values())
            m_rem = sum((ma - mb).values())
            sizes.append((p_add, p_rem, m_add, m_rem))
            if p_add == 1 and p_rem == 1 and m_add == 0 and m_rem == 0:
                types["patch가 추가한 줄 1개 교체"] += 1
            elif p_add == 1 and p_rem == 0 and m_add == 1 and m_rem == 0:
                types["기존 줄 1개 교체"] += 1
            elif p_add >= 1 and p_rem == 0 and m_add == 0 and m_rem == 0:
                types["줄 추가만"] += 1
            elif p_add == 0 and p_rem == 0 and m_add == 0 and m_rem == 0:
                types["차이 없음(공백 등)"] += 1
            else:
                types["그 밖(여러 줄 변경)"] += 1
        return dict(types), [sum(x[i] for x in sizes) / len(sizes) for i in range(4)]

    res["oneoff_shape"], res["oneoff_mean_Padd_Prem_Madd_Mrem"] = shape(o)
    res["conflicting_shape"], res["conflicting_mean_Padd_Prem_Madd_Mrem"] = shape(c)

    # FAIL_TO_PASS 첫 테스트가 test_patch가 건드린 파일에 있는가는 test_patch만으로 알 수 없다 -> 건수만 기록
    def n_f2p(x):
        return len(json.loads(x)) if isinstance(x, str) else len(x)
    f2p = [n_f2p(x) for x in r.FAIL_TO_PASS]
    res["fail_to_pass_count_summary"] = {
        "mean": sum(f2p) / len(f2p),
        "max": max(f2p),
        "share_with_exactly_1": sum(1 for x in f2p if x == 1) / len(f2p),
    }
    # 변형은 "첫 FAIL_TO_PASS 테스트 하나"만 겨냥한다(gen/swebench_mutate.py). 나머지 F2P가 몇 개나 남는가.
    res["share_with_other_untouched_f2p_tests"] = sum(1 for x in f2p if x >= 2) / len(f2p)
    return res


def main():
    out = {"lcb": lcb_audit(load("lcb")), "swe": swe_audit(load("swe"))}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
