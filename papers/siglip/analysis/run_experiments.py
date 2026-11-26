"""softmax(CLIP) 대 sigmoid(SigLIP) 합성 도형 토이 비교. 논문 재현이 아니라 이해용이다.

  python analysis/run_experiments.py train --seeds 0 --out analysis/toy_seed0.json   # 시드 하나(CPU 1스레드, 약 15분)
  python analysis/run_experiments.py summary                                         # toy_seed*.json -> toy_results.json

실행 격자 (시드마다)
  regime fixed_steps      : 배치 16, 32, 128 x 2500 스텝. 배치가 작을수록 본 이미지 수도 줄어든다 (clip 분석이 지적한 교란).
  regime matched_examples : 본 예제 수 320,000으로 맞춤. 배치 16, 32, 128, 512 -> 20000, 10000, 2500, 625 스텝. 논문 그림 2도 본 예제 수를 맞춘다.
  손실: softmax, sigmoid(t=10, b=-10), 그리고 b 초기화 절제 sigmoid_b0(b=0)는 matched 배치 128, 512에서만.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import toy_train as T  # noqa: E402

HERE = Path(__file__).parent
BASE_STEPS = 2500
BASE_BATCH = 128
EXAMPLES = BASE_STEPS * BASE_BATCH  # 320,000
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}  # 양측 95% t 임계값, 키는 자유도


def grid():
    """(regime 이름들, loss, batch, steps) 목록. 배치 128, 2500스텝은 두 regime이 공유한다."""
    runs = []
    for loss in ("softmax", "sigmoid"):
        for b in (16, 32, 128):
            runs.append((("fixed_steps",) + (("matched_examples",) if b == 128 else ()), loss, b, BASE_STEPS))
        for b in (16, 32, 512):
            runs.append((("matched_examples",), loss, b, EXAMPLES // b))
    for b in (128, 512):
        runs.append((("matched_examples",), "sigmoid_b0", b, EXAMPLES // b))
    return runs


def cmd_train(a):
    out = {"runs": []}
    for seed in a.seeds:
        for regimes, loss, b, steps in grid():
            r = T.run_one(seed, loss, b, steps)
            r["regimes"] = list(regimes)
            out["runs"].append(r)
            print(seed, loss, b, steps, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
            Path(a.out).write_text(json.dumps(out))


def interval(xs):
    n = len(xs)
    mean = statistics.fmean(xs)
    if n < 2:
        return [mean, mean, mean]
    half = T975[n - 1] * statistics.stdev(xs) / math.sqrt(n)
    return [mean, mean - half, mean + half]


def cmd_summary(_):
    runs = []
    for f in sorted(HERE.glob("toy_seed*.json")):
        runs += json.loads(f.read_text())["runs"]
    metrics = ["zs_held_out_photo", "zs_seen_photo", "logit_pos_mean", "logit_neg_mean", "t", "b"]
    cells = {}  # (regime, loss, batch) -> seed -> run
    for r in runs:
        for g in r["regimes"]:
            cells.setdefault((g, r["loss"], r["batch"]), {})[r["seed"]] = r
    summary = {"n_seeds": len({r["seed"] for r in runs}), "cells": {}, "paired": {}}
    for (g, loss, b), by_seed in sorted(cells.items()):
        seeds = sorted(by_seed)
        summary["cells"][f"{g}|{loss}|{b}"] = {
            "steps": by_seed[seeds[0]]["steps"], "examples_seen": by_seed[seeds[0]]["examples_seen"], "seeds": seeds,
            **{m: interval([by_seed[s][m] for s in seeds]) for m in metrics if by_seed[seeds[0]][m] is not None}}
    try:
        from scipy import stats
    except ImportError:
        stats = None
    for (g, loss, b), by_seed in sorted(cells.items()):
        for ref in (("softmax",) if loss == "sigmoid" else ("sigmoid",) if loss == "sigmoid_b0" else ()):
            other = cells.get((g, ref, b))
            if not other:
                continue
            seeds = sorted(set(by_seed) & set(other))
            for m in ("zs_held_out_photo", "zs_seen_photo"):
                x, y = [by_seed[s][m] for s in seeds], [other[s][m] for s in seeds]
                d = [p - q for p, q in zip(x, y)]
                rec = {"diff_mean_ci95": interval(d), "per_seed_diff": d, "n_seeds": len(seeds),
                       "wins": sum(v > 0 for v in d)}
                if stats is not None:
                    rec["paired_t_p"] = float(stats.ttest_rel(x, y).pvalue)
                summary["paired"][f"{g}|b={b}|{loss}-{ref}|{m}"] = rec
    (HERE / "toy_results.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary["paired"].items():
        print(k, [round(x, 3) for x in v["diff_mean_ci95"]], "wins", v["wins"], "/", v["n_seeds"], "p=", v.get("paired_t_p"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seeds", type=int, nargs="+", default=[0])
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("summary")
    s.set_defaults(fn=cmd_summary)
    a = p.parse_args()
    a.fn(a)
