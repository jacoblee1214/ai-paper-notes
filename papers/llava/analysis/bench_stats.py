"""LLaVA-Bench 통계 점검 (API 호출 없음, numpy 만). 결과는 bench_results.json.

1. 문항 수와 범주 구성: COCO 90문항(공식 저장소 playground/data/coco2014_val_qa_eval/qa90_questions.jsonl),
   In-the-Wild 60문항(HF liuhaotian/llava-bench-in-the-wild 의 questions.jsonl, 8 KB, data/ 에 받아 둠).
2. 논문 표 4·5 재계산: '전체' 열이 범주 문항 수 가중 평균과 맞는지, 괄호 안 차이가 맞는지.
3. 상대 점수 = Σ(후보 점수) / Σ(참조 점수) 의 불확실성. LLaVA 의 문항별 판정 점수는 공개되지 않았다.
   대신 같은 저장소가 담고 있는 Vicuna-80 판정 4세트(GPT-4, 같은 1~10 점수 프로토콜, llava/eval/table/review)로
   문항별 점수의 산포와 상관을 재고, 그 값으로 델타 방법과 부트스트랩 구간을 낸다. 다른 벤치마크의 산포를 빌린 것이다.
4. 판정 코드의 구조적 선택: 참조가 항상 어시스턴트 1, 파싱 실패는 [-1, -1] 로 평균에 들어감 -> 영향 산술.
실행: python3 llava/analysis/bench_stats.py  (저장소 루트에서)
"""
import collections
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repo"
OUT = Path(__file__).with_name("bench_results.json")
rng = np.random.default_rng(0)


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def counts():
    coco = load_jsonl(REPO / "playground/data/coco2014_val_qa_eval/qa90_questions.jsonl")
    itw = load_jsonl(ROOT / "data/llava_bench_itw/questions.jsonl")
    ctx = load_jsonl(ROOT / "data/llava_bench_itw/context.jsonl")
    return {
        "coco": {"n": len(coco), "images": len({q["image"] for q in coco}), "by_category": dict(collections.Counter(q["category"] for q in coco))},
        "itw": {"n": len(itw), "images": len({q["image"] for q in itw}), "contexts": len(ctx), "by_category": dict(collections.Counter(q["category"] for q in itw))},
    }


def table_checks(n_itw):
    """표 5의 '전체'는 범주 문항 수 가중 평균(참조 점수 합 가중에 가깝다)과 얼마나 맞는가."""
    t5 = {  # 표 5: (conv, detail, complex, all)
        "OpenFlamingo": (19.3, 19.0, 19.1, 19.1),
        "BLIP-2": (54.6, 29.1, 32.9, 38.1),
        "LLaVA": (57.3, 52.5, 81.7, 67.3),
        "LLaVA_dagger": (58.8, 49.2, 81.4, 66.7),
    }
    n = np.array([n_itw["conv"], n_itw["detail"], n_itw["complex"]])
    out = {}
    for k, (a, b, c, tot) in t5.items():
        w = float((n * np.array([a, b, c])).sum() / n.sum())
        out[k] = {"reported_all": tot, "n_weighted": round(w, 2), "unweighted": round((a + b + c) / 3, 2), "diff_weighted": round(w - tot, 2)}
    t4 = {  # 표 4: (conv, detail, complex, all), COCO 는 범주당 30문항
        "Full data": (83.1, 75.3, 96.5, 85.1),
        "Detail + Complex": (81.5, 73.3, 90.8, 81.9),
        "Conv + 5% Detail + 10% Complex": (81.0, 68.4, 91.5, 80.5),
        "Conversation": (76.5, 59.8, 84.9, 73.8),
        "No Instruction Tuning": (22.0, 24.0, 18.5, 21.5),
    }
    out4 = {}
    full = t4["Full data"]
    for k, (a, b, c, tot) in t4.items():
        out4[k] = {"reported_all": tot, "mean_of_three": round((a + b + c) / 3, 2), "diff": round((a + b + c) / 3 - tot, 2),
                   "delta_vs_full_recomputed": [round(x - y, 1) for x, y in zip((a, b, c, tot), full)]}
    # 표 4 괄호 안 차이(논문 표기)와 뺄셈 재계산의 불일치 칸
    rep = {"Detail + Complex": (-1.6, -2.0, -5.7, -3.2), "Conv + 5% Detail + 10% Complex": (-2.1, -7.1, -5.0, -4.4),
           "Conversation": (-6.6, -16.2, -12.4, -11.3), "No Instruction Tuning": (-61.1, -51.3, -78.0, -63.6)}
    cols = ["conv", "detail", "complex", "all"]
    mism = []
    for k, d in rep.items():
        for c, x, y in zip(cols, d, out4[k]["delta_vs_full_recomputed"]):
            if abs(x - y) > 0.051:
                mism.append({"row": k, "col": c, "reported": x, "recomputed": y, "gap": round(x - y, 1)})
    return {"table5": out, "table4": out4, "table4_delta_mismatches": mism, "table4_delta_cells_checked": 16}


def parse_first_line(text):
    m = re.match(r"^\s*(\d+(?:\.\d+)?)[ ,]+(\d+(?:\.\d+)?)\s*$", text.split("\n")[0])
    return (float(m.group(1)), float(m.group(2))) if m else None


def vicuna80():
    """GPT-4 판정 Vicuna-80 4세트. 문항별 (어시스턴트1, 어시스턴트2) 점수와 파싱 실패 수."""
    base = REPO / "llava/eval/table/review"
    out, arrays = {}, {}
    for name in ["alpaca-13b", "bard", "gpt35", "llama-13b"]:
        rows = load_jsonl(base / f"review_{name}_vicuna-13b.jsonl")
        parsed = [parse_first_line(r["text"]) for r in rows]
        t = np.array([p for p in parsed if p])
        a, b = t[:, 0], t[:, 1]
        arrays[name] = t
        out[name] = {"n_reviews": len(rows), "parse_failures": sum(p is None for p in parsed), "mean1": round(a.mean(), 2), "mean2": round(b.mean(), 2),
                     "ratio_pct": round(b.sum() / a.sum() * 100, 1), "sd1": round(a.std(ddof=1), 2), "sd2": round(b.std(ddof=1), 2),
                     "corr": round(float(np.corrcoef(a, b)[0, 1]), 2), "tie_rate": round(float((a == b).mean()), 2),
                     "integer_scores_share": round(float((t == np.round(t)).mean()), 2)}
    return out, arrays


def ratio(a_ref, b_cand):
    return b_cand.sum() / a_ref.sum() * 100


def bootstrap_ratio(t, n_boot=5000):
    n = len(t)
    idx = rng.integers(0, n, (n_boot, n))
    r = t[idx][:, :, 1].sum(1) / t[idx][:, :, 0].sum(1) * 100
    return np.percentile(r, [2.5, 97.5]).round(1).tolist()


def delta_se(n, mean_ref, ratio_frac, sd_ref, sd_cand, corr):
    """비 r = Σb/Σa 의 델타 방법 표준오차(점수 %p). 문항을 독립 표본으로 본다."""
    var = (sd_cand**2 + ratio_frac**2 * sd_ref**2 - 2 * ratio_frac * corr * sd_ref * sd_cand) / (n * mean_ref**2)
    return 100 * np.sqrt(var)


def simulate_ratio_se(n, mean_ref, ratio_frac, sd_ref, sd_cand, corr, n_sim=20000):
    """정규 점수(1~10 로 자름)로 문항 n개를 뽑아 상대 점수의 표준편차를 직접 잰다. 델타 방법 점검용."""
    cov = np.array([[sd_ref**2, corr * sd_ref * sd_cand], [corr * sd_ref * sd_cand, sd_cand**2]])
    x = rng.multivariate_normal([mean_ref, mean_ref * ratio_frac], cov, size=(n_sim, n))
    x = np.clip(x, 1, 10)
    r = x[:, :, 1].sum(1) / x[:, :, 0].sum(1) * 100
    return float(r.std())


def uncertainty_grid():
    """상대 점수 ~ 67%(In-the-Wild 의 LLaVA)와 85%(COCO 전체)에서 문항 수별 표준오차."""
    rows = []
    for label, n, ratio_pct in [("ITW all (n=60)", 60, 67.3), ("ITW conv (n=17)", 17, 57.3), ("ITW detail (n=15)", 15, 52.5),
                                ("ITW complex (n=28)", 28, 81.7), ("COCO all (n=90)", 90, 85.1), ("COCO per category (n=30)", 30, 83.1)]:
        for sd_c in (1.5, 2.0, 2.5):
            for corr in (0.0, 0.3, 0.5):
                se = delta_se(n, 8.5, ratio_pct / 100, 1.0, sd_c, corr)
                rows.append({"setting": label, "n": n, "ratio_pct": ratio_pct, "sd_cand": sd_c, "corr": corr,
                             "se_pts": round(se, 2), "ci95_half_width": round(1.96 * se, 1)})
    check = {"delta": round(delta_se(60, 8.5, 0.673, 1.0, 2.0, 0.3), 2), "simulated": round(simulate_ratio_se(60, 8.5, 0.673, 1.0, 2.0, 0.3), 2)}
    return rows, check


def paired_difference():
    """같은 90문항에서 두 학습 설정의 상대 점수 차(표 4 의 행 비교)의 표준오차. 두 모델의 문항 점수 상관 rho 에 따라 달라진다."""
    out = []
    for label, d in [("Full vs Conv+5%D+10%C (+4.6)", 4.6), ("Full vs Detail+Complex (+3.2)", 3.2), ("Conversation vs Conv+5%D+10%C (+6.7)", 6.7)]:
        for rho in (0.0, 0.5, 0.8):
            se1 = delta_se(90, 8.5, 0.80, 1.0, 2.0, 0.3)
            se_diff = float(np.sqrt(2 * se1**2 * (1 - rho)))
            out.append({"comparison": label, "diff": d, "rho_between_models": rho, "se_diff": round(se_diff, 2), "z": round(d / se_diff, 2)})
    return out


def itw_vs_baselines():
    """In-the-Wild 에서 LLaVA 67.3 대 BLIP-2 38.1, OpenFlamingo 19.1 의 차이가 문항 표본 오차 대비 얼마인가(가정: 후보 SD 2.0, 상관 0.3)."""
    out = []
    se_l = delta_se(60, 8.5, 0.673, 1.0, 2.0, 0.3)
    for name, r in [("BLIP-2", 38.1), ("OpenFlamingo", 19.1)]:
        se_b = delta_se(60, 8.5, r / 100, 1.0, 2.0, 0.3)
        for rho in (0.0, 0.5):
            se_d = float(np.sqrt(se_l**2 + se_b**2 - 2 * rho * se_l * se_b))
            out.append({"vs": name, "diff_pts": round(67.3 - r, 1), "rho_between_models": rho, "se_diff": round(se_d, 2), "z": round((67.3 - r) / se_d, 1)})
    return out


def failure_effect():
    """파싱 실패 [-1,-1] 가 그대로 평균에 들어갈 때 비의 이동. 실패한 문항의 실제 점수가 (참조 8.0, 후보 r*8.0) 이라고 가정."""
    out = []
    n = 60
    for ratio_true in (0.673, 0.85):
        ref = np.full(n, 8.0)
        cand = ref * ratio_true
        base = ratio(ref, cand)
        for k in (1, 2, 3):
            ref2, cand2 = ref.copy(), cand.copy()
            ref2[:k], cand2[:k] = -1.0, -1.0
            out.append({"true_ratio_pct": round(base, 1), "failures": k, "reported_ratio_pct": round(ratio(ref2, cand2), 2), "shift": round(ratio(ref2, cand2) - base, 2)})
    return out


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (centre - half), 2), round(100 * (centre + half), 2)


def scienceqa():
    """ScienceQA 시험 분할 4,241문항(논문 L476)에서 표 7·8 정확도의 이항 구간과 차이의 z.
    짝지은 비교의 SE 는 두 모델이 서로 다르게 맞히는 비율(불일치율 d)에 달려 있다: SE_diff = sqrt(d / n). d 를 모르므로 가정값을 나열한다."""
    n = 4241
    acc = {"LLaVA (before last layer)": 90.92, "LLaVA+GPT-4 judge": 92.53, "LLaVA+GPT-4 complement": 90.97, "MM-CoT Large": 91.68,
           "LLaMA-Adapter": 85.19, "last layer": 89.96, "answer first": 89.77, "no pretraining stage": 85.81, "7B": 89.84, "GPT-4 2-shot": 82.69}
    out = {"n": n, "wilson95": {k: wilson(v / 100, n) for k, v in acc.items()}, "diffs": []}
    comps = [("before-last vs last layer", 90.92, 89.96), ("13B vs 7B", 90.92, 89.84), ("reasoning-first vs answer-first", 90.92, 89.77),
             ("with vs without stage 1", 90.92, 85.81), ("GPT-4 judge ensemble vs MM-CoT Large", 92.53, 91.68), ("GPT-4 judge ensemble vs LLaVA alone", 92.53, 90.92)]
    for name, a, b in comps:
        p1, p2 = a / 100, b / 100
        se_unp = float(np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n))
        row = {"comparison": name, "diff_pts": round(a - b, 2), "z_unpaired": round((a - b) / 100 / se_unp, 2)}
        for d in (0.03, 0.06, 0.10):
            row[f"z_paired_d{int(d*100):02d}"] = round((a - b) / 100 / np.sqrt(d / n), 2)
        out["diffs"].append(row)
    out["mde_paired_pts"] = {f"d{int(d*100):02d}": round(2.80 * np.sqrt(d / n) * 100, 2) for d in (0.03, 0.06, 0.10)}
    return out


def main():
    r = {"counts": counts()}
    r["scienceqa"] = scienceqa()
    r["table_recompute"] = table_checks(r["counts"]["itw"]["by_category"])
    v80, arrays = vicuna80()
    r["vicuna80_reviews"] = v80
    r["vicuna80_bootstrap_ci95_ratio"] = {k: bootstrap_ratio(t) for k, t in arrays.items()}
    r["vicuna80_ci_half_width_vs_point"] = {k: [round(ratio(t[:, 0], t[:, 1]), 1), bootstrap_ratio(t)] for k, t in arrays.items()}
    grid, check = uncertainty_grid()
    r["uncertainty_grid"] = grid
    r["delta_vs_simulation_check_n60"] = check
    r["paired_row_differences_table4"] = paired_difference()
    r["itw_vs_baselines"] = itw_vs_baselines()
    r["parse_failure_effect"] = failure_effect()
    r["paper_reported_std"] = {"table5_LLaVA_all": 2.0, "note": "3회 추론 실행의 표준편차(디코딩 변동). 문항 표본 오차는 들어 있지 않다"}
    OUT.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(json.dumps({k: r[k] for k in ["counts", "table_recompute", "vicuna80_reviews", "vicuna80_bootstrap_ci95_ratio", "delta_vs_simulation_check_n60", "parse_failure_effect"]}, indent=1, ensure_ascii=False))
    for row in r["uncertainty_grid"]:
        if row["sd_cand"] == 2.0 and row["corr"] == 0.3:
            print(row)
    for row in r["paired_row_differences_table4"]:
        print(row)


if __name__ == "__main__":
    main()
