"""CV-Bench(HF nyu-visionx/CV-Bench)의 질문·답 주석(data/test_2d.jsonl, test_3d.jsonl)으로 구성, 우연 수준, 이미지 없는 단서를 계산한다.

이미지는 받지 않았다. 입력: data/*.jsonl (HF 저장소 파일, Apache-2.0).  출력: analysis/cvbench_results.json
이 스크립트는 `python3 -I analysis/cvbench_stats.py`로 실행한다(내려받은 파일을 읽으므로 -I).
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for f in ["test_2d.jsonl", "test_3d.jsonl"] for l in open(ROOT / "data" / f)]
out = {"n_total": len(rows)}


def ans_idx(r):
    return ord(r["answer"][1]) - 65


# 1. 구성
out["by_source_task"] = {f"{s}|{t}": n for (s, t), n in sorted(collections.Counter((r["source"], r["task"]) for r in rows).items())}
out["by_type"] = dict(collections.Counter(r["type"] for r in rows))
out["by_source_dataset"] = {k: v for k, v in sorted(collections.Counter(r["source_dataset"] for r in rows).items())}
out["n_choices_by_task"] = {t: dict(collections.Counter(len(r["choices"]) for r in rows if r["task"] == t)) for t in ["Count", "Relation", "Depth", "Distance"]}
# 이미지 재사용
img = collections.defaultdict(set)
for r in rows:
    img[r["source"] + "|" + r["source_dataset"]].add(r["source_filename"])
nq = collections.Counter(r["source"] + "|" + r["source_dataset"] for r in rows)
out["distinct_source_images"] = {k: {"questions": nq[k], "images": len(v), "q_per_image": round(nq[k] / len(v), 2)} for k, v in img.items()}
out["distinct_source_images_total"] = len({r["source_filename"] for r in rows})

# 2. 우연 수준
by_src = collections.defaultdict(list)
for r in rows:
    by_src[r["source"]].append(1 / len(r["choices"]))
chance_src = {s: float(np.mean(v)) for s, v in by_src.items()}
# 보고 지표: 1/2 * ((ADE + COCO)/2 + Omni3D)   (HF README, 논문 부록 C)
comb = 0.5 * ((chance_src["ADE20K"] + chance_src["COCO"]) / 2 + chance_src["Omni3D"])
by_type = collections.defaultdict(list)
for r in rows:
    by_type[r["type"]].append(1 / len(r["choices"]))
out["chance"] = {"by_source": {k: round(100 * v, 2) for k, v in chance_src.items()},
                 "combined_metric": round(100 * comb, 2),
                 "pooled_all_items": round(100 * float(np.mean([1 / len(r["choices"]) for r in rows])), 2),
                 "2D_pooled": round(100 * float(np.mean(by_type["2D"])), 2), "3D_pooled": round(100 * float(np.mean(by_type["3D"])), 2)}

# 3. 이미지 없는 단서
cnt = [r for r in rows if r["task"] == "Count"]


def val(r, i):
    return int(r["choices"][i])


def medoid_pick(r):
    v = np.array([int(x) for x in r["choices"]])
    cost = np.abs(v[:, None] - v[None, :]).sum(1)
    cand = np.flatnonzero(cost == cost.min())
    return cand  # 동점이면 무작위 -> 기대 정확도 1/len(cand)


acc = []
for r in cnt:
    cand = medoid_pick(r)
    acc.append((ans_idx(r) in cand) / len(cand))
heur_count = float(np.mean(acc))
# 모달 순위 규칙(선택지 수별 정렬 순위 하나를 고정)
rank_acc = {}
for k in (4, 5, 6):
    sub = [r for r in cnt if len(r["choices"]) == k]
    ranks = collections.Counter(sorted(int(x) for x in r["choices"]).index(int(r["choices"][ans_idx(r)])) for r in sub)
    best_rank, hits = ranks.most_common(1)[0]
    rank_acc[k] = {"n": len(sub), "modal_rank": best_rank, "hits": hits, "acc": round(hits / len(sub), 3)}
offsets = collections.Counter(tuple(sorted(int(x) - int(r["choices"][ans_idx(r)]) for x in r["choices"])) for r in cnt)
out["count_task"] = {
    "n": len(cnt),
    "medoid_heuristic_acc": round(100 * heur_count, 1),
    "modal_rank_rule": rank_acc,
    "modal_rank_rule_overall": round(100 * sum(v["hits"] for v in rank_acc.values()) / len(cnt), 1),
    "answer_value_1_share": round(100 * np.mean([int(r["choices"][ans_idx(r)]) == 1 for r in cnt]), 1),
    "top_decoy_offset_patterns": [[list(k), v] for k, v in offsets.most_common(6)],
    "share_of_items_with_offsets_(-1,0,1,2)|(-2..2)|(-3,-2,-1,0,1,2)|(-4,-2,-1,0,1,2)": round(
        100 * (offsets[(-1, 0, 1, 2)] + offsets[(-2, -1, 0, 1, 2)] + offsets[(-3, -2, -1, 0, 1, 2)] + offsets[(-4, -2, -1, 0, 1, 2)]) / len(cnt), 1),
}
# 합성: 이미지 없는 규칙으로 CV-Bench 점수 (Count는 medoid, 나머지는 50% = 무작위)
src_acc = {}
for s in ["ADE20K", "COCO", "Omni3D"]:
    sub = [r for r in rows if r["source"] == s]
    a = []
    for r in sub:
        if r["task"] == "Count":
            cand = medoid_pick(r); a.append((ans_idx(r) in cand) / len(cand))
        else:
            a.append(1 / len(r["choices"]))
    src_acc[s] = float(np.mean(a))
combined_h = 0.5 * ((src_acc["ADE20K"] + src_acc["COCO"]) / 2 + src_acc["Omni3D"])
n2d = sum(1 for r in rows if r["type"] == "2D")
acc2d = float(np.mean([(medoid_pick(r) is not None and (ans_idx(r) in medoid_pick(r)) / len(medoid_pick(r))) if r["task"] == "Count" else 0.5
                       for r in rows if r["type"] == "2D"]))
out["no_image_rule"] = {"by_source": {k: round(100 * v, 1) for k, v in src_acc.items()}, "combined_metric": round(100 * combined_h, 1),
                        "2D_pooled": round(100 * acc2d, 1), "3D": 50.0,
                        "note": "Count는 선택지 값의 medoid를 고르는 규칙(동점 시 무작위), 나머지 과제는 50% 우연 수준으로 둠(이미지 없이는 더 알 수 없다고 가정)"}
# 같은 합성을 '선택지 수별 모달 순위' 규칙(3개 숫자를 이 벤치마크에서 맞춘 표본 내 규칙)으로 계산
modal = {k: v["modal_rank"] for k, v in rank_acc.items()}


def rank_hit(r):
    srt = sorted(int(x) for x in r["choices"])
    return float(int(r["choices"][ans_idx(r)]) == srt[modal[len(srt)]])


src_rank = {}
for s in ["ADE20K", "COCO", "Omni3D"]:
    src_rank[s] = float(np.mean([rank_hit(r) if r["task"] == "Count" else 1 / len(r["choices"]) for r in rows if r["source"] == s]))
acc2d_rank = float(np.mean([rank_hit(r) if r["task"] == "Count" else 0.5 for r in rows if r["type"] == "2D"]))
out["no_image_rule_rank"] = {"by_source": {k: round(100 * v, 1) for k, v in src_rank.items()},
                             "2D_by_formula(ADE+COCO)/2": round(50 * (src_rank["ADE20K"] + src_rank["COCO"]), 1),
                             "combined_metric": round(100 * 0.5 * ((src_rank["ADE20K"] + src_rank["COCO"]) / 2 + src_rank["Omni3D"]), 1),
                             "note": "표본 내 규칙(선택지 4/5/6개일 때 정렬 순위 1/2/3번째를 고름). Count 외 과제는 50%"}
# 답 위치(글자) 편향과 텍스트 단서
out["answer_letter_by_task"] = {t: dict(collections.Counter(r["answer"] for r in rows if r["task"] == t)) for t in ["Count", "Relation", "Depth", "Distance"]}
rel = [r for r in rows if r["task"] == "Relation"]
ans_text = collections.Counter(r["choices"][ans_idx(r)] for r in rel)
out["relation_answer_text"] = dict(ans_text)
out["relation_majority_baseline"] = {"left_right": round(100 * max(ans_text["left"], ans_text["right"]) / (ans_text["left"] + ans_text["right"]), 1),
                                     "above_below": round(100 * max(ans_text["above"], ans_text["below"]) / (ans_text["above"] + ans_text["below"]), 1)}
for t in ["Depth", "Distance"]:
    sub = [r for r in rows if r["task"] == t]
    out[f"{t}_answer_is_first_listed"] = round(100 * np.mean([ans_idx(r) == 0 for r in sub]), 1)

# 4. 정확도 구간 (보고 지표 기준, p는 예시)
def wilson_half(p, n, z=1.96):
    den = 1 + z * z / n
    return 100 * z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den


n_ade, n_coco, n_omni = (sum(1 for r in rows if r["source"] == s) for s in ["ADE20K", "COCO", "Omni3D"])
out["n_by_source"] = {"ADE20K": n_ade, "COCO": n_coco, "Omni3D": n_omni}
res = {}
for p in (0.5, 0.6, 0.7, 0.8):
    var = 0.25 * (0.25 * (p * (1 - p) / n_ade + p * (1 - p) / n_coco) + p * (1 - p) / n_omni)
    res[f"p={p}"] = {"combined_se_pts": round(100 * np.sqrt(var), 2), "ci95_half_pts": round(196 * np.sqrt(var), 2),
                     "total_n_wilson_half": round(float(wilson_half(p, len(rows))), 2)}
out["combined_metric_se"] = res
# MDE (쌍체 q=0.2) 보고 지표: 두 모델 차이
for q in (0.1, 0.2, 0.3):
    var = 0.25 * (0.25 * (q / n_ade + q / n_coco) + q / n_omni)
    out[f"combined_metric_mde_q{q}"] = round(100 * 2.80 * np.sqrt(var), 2)
out["relative_size"] = {"vs_RealWorldQA_765": round(len(rows) / 765, 2), "vs_MMVP_300_questions": round(len(rows) / 300, 2), "vs_MMVP_150_pairs": round(len(rows) / 150, 2)}
(ROOT / "analysis" / "cvbench_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: out[k] for k in ["by_source_task", "chance", "count_task", "no_image_rule", "relation_majority_baseline",
                                      "no_image_rule_rank", "distinct_source_images", "combined_metric_se", "combined_metric_mde_q0.2", "n_by_source"]}, ensure_ascii=False, indent=1))
