"""논문 Appendix E(인간 피드백 실험, 40 프로그램 x 4 피드백 x 25 수리 후보)를 재계산한다.

입력: 논문 Appendix E 표를 옮겨 적은 값(각 칸은 25개 수리 후보 중 통과한 개수).
출력: analysis/recompute_human_study_results.json
검증 대상: Table 1 (GPT-4 33.30%, 인간 52.60%, 1.58x), 난이도별 값, 대응 표본 검정.
"""
import json, random
from math import comb
from pathlib import Path

# (task, difficulty, program, GPT-4 #1, GPT-4 #2, Human #1, Human #2)
ROWS = [
    (2106, "interview", "A", 7, 10, 10, 0), (2106, "interview", "B", 0, 2, 20, 16),
    (2673, "interview", "A", 4, 7, 17, 24), (2673, "interview", "B", 3, 25, 25, 25),
    (2923, "interview", "A", 0, 0, 0, 0), (2923, "interview", "B", 0, 0, 0, 0),
    (3070, "competition", "A", 0, 0, 0, 0), (3070, "competition", "B", 3, 0, 5, 0),
    (3286, "competition", "A", 2, 6, 10, 25), (3286, "competition", "B", 0, 0, 0, 4),
    (3754, "competition", "A", 0, 0, 0, 0), (3754, "competition", "B", 0, 0, 0, 0),
    (4182, "introductory", "A", 25, 25, 25, 24), (4182, "introductory", "B", 25, 0, 25, 25),
    (4195, "introductory", "A", 25, 3, 24, 23), (4195, "introductory", "B", 23, 25, 25, 25),
    (4281, "introductory", "A", 0, 4, 0, 0), (4281, "introductory", "B", 0, 0, 0, 0),
    (4333, "introductory", "A", 25, 0, 25, 0), (4333, "introductory", "B", 23, 24, 24, 25),
    (4347, "introductory", "A", 0, 0, 7, 25), (4347, "introductory", "B", 0, 0, 25, 25),
    (4426, "introductory", "A", 25, 25, 25, 25), (4426, "introductory", "B", 25, 25, 25, 25),
    (4450, "introductory", "A", 0, 0, 0, 0), (4450, "introductory", "B", 24, 0, 22, 24),
    (4507, "introductory", "A", 0, 0, 0, 0), (4507, "introductory", "B", 0, 0, 1, 0),
    (4514, "introductory", "A", 15, 21, 1, 16), (4514, "introductory", "B", 0, 0, 25, 0),
    (4704, "introductory", "A", 0, 25, 0, 25), (4704, "introductory", "B", 25, 25, 24, 23),
    (4741, "introductory", "A", 25, 25, 25, 25), (4741, "introductory", "B", 25, 25, 25, 25),
    (4855, "introductory", "A", 0, 1, 17, 25), (4855, "introductory", "B", 0, 2, 3, 23),
    (4873, "introductory", "A", 0, 0, 0, 0), (4873, "introductory", "B", 0, 0, 0, 18),
    (4952, "introductory", "A", 0, 0, 2, 25), (4952, "introductory", "B", 24, 8, 24, 21),
]
N = 25  # 수리 후보 수
assert len(ROWS) == 40 and len({r[0] for r in ROWS}) == 20


def rate(vals):
    return sum(vals) / (N * len(vals))


def summarize(rows):
    g = [x for r in rows for x in r[3:5]]
    h = [x for r in rows for x in r[5:7]]
    gp, hp = sum(g), sum(h)
    tot = N * len(g)
    return {"n_programs": len(rows), "gpt4_passes": gp, "human_passes": hp,
            "total_repairs_each": tot, "gpt4_rate": gp / tot, "human_rate": hp / tot,
            "ratio": (hp / tot) / (gp / tot) if gp else None}


def sign_test_exact(pos, neg):
    """두 쪽 부호 검정(동점 제외)."""
    n = pos + neg
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    out = {"overall": summarize(ROWS)}
    for d in ["introductory", "interview", "competition"]:
        out[d] = summarize([r for r in ROWS if r[1] == d])

    # 프로그램 단위 쌍 비교: 프로그램마다 GPT-4 평균 통과율과 인간 평균 통과율
    diffs = []
    for r in ROWS:
        g = (r[3] + r[4]) / (2 * N)
        h = (r[5] + r[6]) / (2 * N)
        diffs.append((h - g, r))
    pos = sum(1 for d, _ in diffs if d > 0)
    neg = sum(1 for d, _ in diffs if d < 0)
    tie = sum(1 for d, _ in diffs if d == 0)
    out["program_level"] = {"human_better": pos, "gpt4_better": neg, "tie": tie,
                            "sign_test_two_sided_p": sign_test_exact(pos, neg)}

    # 태스크 단위 부트스트랩: 20개 태스크를 복원추출해 비율(human/gpt4)의 95% CI
    rng = random.Random(0)
    tasks = sorted({r[0] for r in ROWS})
    by_task = {t: [r for r in ROWS if r[0] == t] for t in tasks}
    ratios = []
    for _ in range(10000):
        samp = [rng.choice(tasks) for _ in tasks]
        rows = [r for t in samp for r in by_task[t]]
        s = summarize(rows)
        if s["ratio"] is not None:
            ratios.append(s["ratio"])
    ratios.sort()
    out["task_bootstrap_ratio_95ci"] = [ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]]

    # 프로그램 단위 부호 순열 검정(평균 차이의 부호를 무작위로 뒤집어 귀무분포를 만든다)
    obs = sum(d for d, _ in diffs) / len(diffs)
    cnt = 0
    R = 20000
    for _ in range(R):
        s = sum(d if rng.random() < 0.5 else -d for d, _ in diffs) / len(diffs)
        if abs(s) >= abs(obs):
            cnt += 1
    out["program_level_sign_flip_perm_p"] = (cnt + 1) / (R + 1)
    out["mean_paired_diff_human_minus_gpt4"] = obs

    path = Path(__file__).with_name("recompute_human_study_results.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
