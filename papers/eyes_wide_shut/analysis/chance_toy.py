"""MMVP의 쌍 단위 채점이 만드는 우연 수준을 구조(공개 Questions.csv)와 numpy 토이로 확인한다 (CPU, 수 초).

입력: ../data/mmvp/Questions.csv, ../data/mmvp_vlm/Questions.csv (HF MMVP/MMVP, MMVP/MMVP_VLM; data/는 gitignore)
출력: chance_results.json

C1 파일 구조: 300행이 150쌍(연속 2행)이고 한 쌍의 두 질문·보기가 같다. 정답이 쌍마다 (홀수행 = (a), 짝수행 = (b))인지.
C2 이미지를 보지 않는 응답 규칙(항상 (a), 항상 (b), 항상 Yes, 항상 첫 보기 글자 ...)의 쌍 점수를 실제 파일로 채점.
C3 몬테카를로: 이미지와 무관하게 P((a)) = p 로 답하는 응답자의 쌍 점수 = p(1-p). p=0.5에서만 25%, 상한이 25%.
   이미지를 보고 비율 d의 쌍을 맞히고 나머지는 이미지를 보지 않는 모델의 점수 = d + (1-d) p(1-p).
C4 같은 쌍에서 두 답이 상관 rho로 같아지는 응답자(이미지가 거의 같아서 같은 답을 내는 경향)의 점수.
C5 MMVP-VLM: 두 텍스트 x 두 이미지, 텍스트마다 어느 이미지가 더 맞는지 고르는 쌍 채점. 이미지 임베딩이 같은 모델이 쓰는 규칙
   (항상 img1, 두 텍스트에 같은 이미지)은 0점, 독립 무작위는 25%.
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
res = {}

rows = list(csv.DictReader(open(DATA / "mmvp" / "Questions.csv", encoding="utf-8")))
n = len(rows)
pairs = [(rows[i], rows[i + 1]) for i in range(0, n, 2)]
res["C1_structure"] = {
    "n_rows": n,
    "n_pairs": len(pairs),
    "pair_same_question_and_options": sum(a["Question"] == b["Question"] and a["Options"].strip() == b["Options"].strip() for a, b in pairs),
    "pair_answers_differ": sum(a["Correct Answer"] != b["Correct Answer"] for a, b in pairs),
    "odd_row_answer_a": sum(a["Correct Answer"] == "(a)" for a, b in pairs),
    "even_row_answer_b": sum(b["Correct Answer"] == "(b)" for a, b in pairs),
    "n_options_per_question": sorted({r["Options"].count("(") for r in rows}),
    "n_yes_no_questions": sum(r["Options"].strip() == "(a) Yes (b) No" for r in rows),
    "n_yes_no_pairs": sum(a["Options"].strip() == "(a) Yes (b) No" for a, b in pairs),
    "n_distinct_option_strings": len({r["Options"].strip() for r in rows}),
}


def pair_score(policy):
    """policy(question_row) -> '(a)' 또는 '(b)'. 쌍 단위 점수(%)."""
    ok = 0
    for a, b in pairs:
        ok += policy(a) == a["Correct Answer"] and policy(b) == b["Correct Answer"]
    return 100.0 * ok / len(pairs)


def question_score(policy):
    return 100.0 * sum(policy(r) == r["Correct Answer"] for r in rows) / len(rows)


policies = {
    "always (a)": lambda r: "(a)",
    "always (b)": lambda r: "(b)",
    "always Yes on yes/no, (a) otherwise": lambda r: "(a)",
    "always No on yes/no, (b) otherwise": lambda r: "(b)",
}
res["C2_image_blind_deterministic"] = {k: {"pair_score_pct": pair_score(f), "question_score_pct": question_score(f)} for k, f in policies.items()}

rng = np.random.default_rng(0)
N = 150
reps = 200000
sim = {}
for p in (0.5, 0.4, 0.3, 0.2, 0.1, 0.063):
    k = rng.binomial(N, p * (1 - p), size=reps) / N * 100  # 쌍마다 독립: 첫 질문 (a) 확률 p, 둘째 질문 (b) 확률 1-p
    sim[f"p={p}"] = {"expected_pct": round(100 * p * (1 - p), 2), "mc_mean": round(float(k.mean()), 2), "mc_sd_over_150_pairs": round(float(k.std()), 2),
                      "P(score>=25)": round(float((k >= 25).mean()), 4)}
res["C3_image_blind_stochastic"] = sim

# d: 이미지를 실제로 쓰는 쌍의 비율, 나머지는 p=0.5 로 이미지를 보지 않고 답함 -> 점수 = d + (1-d)*0.25
res["C3b_discrimination_lower_bound"] = {name: {"score": s, "d_lower": round(max(0.0, (s - 25.0) / 75.0) * 100, 1)} for name, s in
                                         [("Gemini 40.7", 40.7), ("GPT-4V 38.7", 38.7), ("PaliGemma-448 47.3 (paligemma/ 03)", 47.3), ("LLaVA-1.5 24.7", 24.7), ("Human 95.7", 95.7)]}

# C4: 두 답이 상관 rho 로 같아지는 모델: 쌍의 두 질문에 대한 P(둘 다 맞음)
def corr_pair_score(p_a, rho):
    """각 질문에서 P((a))=p_a, 두 질문 응답의 상관계수 rho. 맞는 조합은 (a, b). 이진 변수의 이변량 분포로 계산."""
    # X1 = 1{(a) on Q1}, X2 = 1{(a) on Q2}. 맞는 조합: X1=1, X2=0.
    cov = rho * p_a * (1 - p_a)
    p11 = p_a * p_a + cov
    return p_a - p11  # P(X1=1) - P(X1=1, X2=1)


c4 = {}
for rho in (0.0, 0.5, 0.8, 0.95, 1.0):
    c4[f"rho={rho}"] = {f"p_a={pa}": round(100 * corr_pair_score(pa, rho), 2) for pa in (0.5, 0.3)}
res["C4_correlated_answers"] = c4

# C5 MMVP-VLM
v = list(csv.DictReader(open(DATA / "mmvp_vlm" / "Questions.csv", encoding="utf-8")))
types = [v[i]["Type"] for i in range(0, len(v), 30)]
res["C5_vlm_structure"] = {
    "n_rows": len(v), "n_pairs": len(v) // 2, "types_in_order": types, "pairs_per_type": 15,
    "ground_truth_by_parity": "gt = img1 if question_id odd else img2 (evaluate_vlm.py L76-77)",
    "always_img1_pair_score": 0.0,
    "independent_random_pair_score": 25.0,
    "text_prefix_in_code": "'a photo of ' + statement (evaluate_vlm.py L52-53; 논문 본문에 없음)",
}
# 이미지 임베딩이 같은 인코더: 두 텍스트에 대해 같은 이미지를 고른다. 점수 = 0. 점수가 약간 다른 이미지 쪽으로 기울면 소폭 > 0.
ind = (rng.random((reps, 15)) < 0.5) & (rng.random((reps, 15)) < 0.5)
res["C5_vlm_structure"]["mc_independent_pair_score_15_pairs_mean_pct"] = round(float(ind.mean() * 100), 2)
res["C5_vlm_structure"]["mc_independent_sd_per_pattern_pct"] = round(float((ind.mean(axis=1) * 100).std()), 2)

(HERE / "chance_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
