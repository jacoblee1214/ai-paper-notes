"""공식 데이터(meg-tong/sycophancy-eval)의 구성 감사. repo/datasets/*.jsonl만 읽고 모델은 부르지 않습니다.
결과: dataset_audit_results.json
실행 전에: git clone --depth 1 https://github.com/meg-tong/sycophancy-eval.git repo   (커밋 9a16942)
"""
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "repo", "datasets")


def load(name):
    return [json.loads(l) for l in open(os.path.join(D, name + ".jsonl"))]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main():
    out = {}
    fb = load("feedback")
    c = collections.Counter((r["base"]["dataset"], r["metadata"]["prompt_template_type"] or "(none)") for r in fb)
    out["feedback_rows"] = len(fb)
    out["feedback_by_dataset_template"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(c.items())}
    texts = collections.defaultdict(list)
    for r in fb:
        b = r["base"]
        texts[b["dataset"]].append(b.get("text") or b.get("correct_solution"))
    out["feedback_unique_passages"] = {k: len(set(v)) for k, v in texts.items()}
    out["feedback_expected_passages"] = {"arguments": 300 * 5, "math": 1000 * 5, "poems": 400 * 5}
    out["feedback_unique_passages_per_template"] = {k: len(set(v)) for k, v in texts.items()}
    # 수학 피드백은 항상 정답 풀이(correct_solution)에 대한 것인지
    m = [r for r in fb if r["base"]["dataset"] == "math"]
    out["feedback_math_prompt_contains_correct_solution"] = sum(r["base"]["correct_solution"] in r["prompt"][0]["content"] for r in m)
    out["feedback_math_prompt_contains_wrong_solution_text"] = sum(r["base"]["wrong_solution"] in r["prompt"][0]["content"] for r in m)
    out["feedback_math_rows"] = len(m)
    arg = [r for r in fb if r["base"]["dataset"] == "arguments" and r["metadata"]["prompt_template_type"] == ""]
    out["feedback_argument_target_rating_counts"] = dict(sorted(collections.Counter(r["base"]["rating"] for r in arg).items()))
    poems = [r for r in fb if r["base"]["dataset"] == "poems" and r["metadata"]["prompt_template_type"] == ""]
    out["feedback_poem_generator_counts"] = dict(collections.Counter(r["base"]["model"] for r in poems))

    ans = load("answer")
    c = collections.Counter((r["base"]["dataset"], r["metadata"]["prompt_template"][:60]) for r in ans)
    out["answer_rows"] = len(ans)
    out["answer_by_dataset_template"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(c.items())}
    leaks = collections.Counter()
    for r in ans:
        b = r["base"]
        if "answer" in b:  # trivia_qa 형식: 별칭 목록
            al = {norm(a) for a in b["answer"]}
            leaks[(b["dataset"], "incorrect_in_alias_list")] += norm(b["incorrect_answer"]) in al
            leaks[(b["dataset"], "incorrect_equals_correct")] += norm(b["incorrect_answer"]) == norm(b["correct_answer"])
        else:
            leaks[(b["dataset"], "incorrect_equals_correct")] += norm(b["incorrect_answer"]) == norm(b["correct_answer"])
    out["answer_label_checks_counts_over_all_templates"] = {f"{k[0]}|{k[1]}": int(v) for k, v in leaks.items()}
    base_q = {(r["base"]["dataset"], r["base"]["question"]) for r in ans}
    out["answer_unique_questions"] = {d: len({q for (dd, q) in base_q if dd == d}) for d in {r["base"]["dataset"] for r in ans}}

    ays = load("are_you_sure")
    c = collections.Counter(r["base"]["dataset"] for r in ays)
    out["are_you_sure_rows"] = len(ays)
    out["are_you_sure_by_dataset"] = dict(c)
    out["are_you_sure_mc_correct_letter_distribution"] = {
        ds: dict(sorted(collections.Counter(r["base"]["correct_letter"] for r in ays if r["base"]["dataset"] == ds).items()))
        for ds in ["math_mc_cot", "mmlu_mc_cot", "truthful_qa_mc", "aqua_mc"]}
    mathq = [r for r in ays if r["base"]["dataset"] == "math_mc_cot"]
    out["are_you_sure_math_options_per_question"] = dict(collections.Counter(len(r["base"]["answers"].split("\n")) for r in mathq))
    tq = [r for r in ays if r["base"]["dataset"] == "truthful_qa_mc"]
    out["are_you_sure_truthfulqa_mc_options_per_question"] = dict(sorted(collections.Counter(len(r["base"]["letters"]) for r in tq).items()))
    out["are_you_sure_prefilled_assistant_turn_rows"] = sum(len(r["prompt"]) == 2 for r in ays)
    out["are_you_sure_dataset_has_no_challenge_turn"] = "are_you_sure.jsonl에는 '확실해요?' 턴과 모델 첫 답이 없다. 첫 질문만 들어 있어 챌린지 단계는 사용자가 직접 구현해야 한다."
    out["repo_has_mimicry_dataset"] = os.path.exists(os.path.join(D, "mimicry.jsonl"))
    out["repo_files"] = sorted(os.listdir(D))
    json.dump(out, open(os.path.join(HERE, "dataset_audit_results.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
