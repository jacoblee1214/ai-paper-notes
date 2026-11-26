"""합성 쌍 토이 실험. 논문 재현이 아니라 CLIP 손실의 성질을 작은 규모에서 보는 용도다.

  python analysis/run_experiments.py train --config baseline --seeds 0 1 2 3 4 --out analysis/toy_baseline.json
  python analysis/run_experiments.py gap --seeds 0 1 2          # gap_results.json (모달리티 간격)
  python analysis/run_experiments.py summary                    # toy_*.json -> toy_results.json (시드 평균, t 구간)

설정
  baseline   배치 128, 표준 대칭 InfoNCE
  hardneg    배치 128, 두 물체 장면마다 모양을 맞바꾼 캡션을 이미지 -> 텍스트 부정 후보로 추가 (NegCLIP식)
  smallbatch 배치 32, 표준 InfoNCE (배치 안에 맞바꾼 쌍이 거의 없을 때)
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as m  # noqa: E402

HERE = Path(__file__).parent
CONFIGS = {
    "baseline": dict(batch=128, hard_neg=False),
    "hardneg": dict(batch=128, hard_neg=True),
    "smallbatch": dict(batch=32, hard_neg=False),
}
T975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}  # 양측 95% t 임계값, 자유도 n-1 = key


def cmd_train(a):
    torch.set_num_threads(2)
    out = {"config": a.config, "steps": a.steps, **CONFIGS[a.config], "seeds": {}}
    for seed in a.seeds:
        t = time.time()
        model, res = m.train(seed, steps=a.steps, device="cpu", **CONFIGS[a.config])
        ev = m.evaluate(model, seed)
        ev["train_seconds"] = round(time.time() - t)
        ev["final_loss_mean_last5"] = sum(v for _, v in res.losses[-5:]) / 5
        ev["curve"] = res.losses
        out["seeds"][str(seed)] = ev
        print(a.config, seed, {k: round(v, 3) for k, v in ev.items() if isinstance(v, float)}, flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


def cmd_gap(a):
    """모달리티 간격: 초기화 직후와 학습 후(baseline 설정), 그리고 학습 후 배율을 인위로 바꾼 값은 쓰지 않는다."""
    torch.set_num_threads(2)
    out = {"steps": a.steps, "seeds": {}}
    for seed in a.seeds:
        torch.manual_seed(seed)
        init = m.modality_gap(m.TinyCLIP(), seed)
        model, res = m.train(seed, steps=a.steps, device="cpu", batch=128)
        out["seeds"][str(seed)] = {"init": init, "trained": m.modality_gap(model, seed),
                                    "logit_scale": model.logit_scale.exp().item()}
        print("gap", seed, out["seeds"][str(seed)], flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


def interval(xs):
    n = len(xs)
    mean = statistics.fmean(xs)
    if n < 2:
        return [mean, mean, mean]
    half = T975[n] * statistics.stdev(xs) / math.sqrt(n)
    return [mean, mean - half, mean + half]


def cmd_summary(_):
    keys = ["zs_held_out_bare", "zs_held_out_photo", "zs_held_out_ensemble", "zs_seen_photo",
            "bind_attr_acc", "bind_order_acc", "wrong_color_acc", "logit_scale", "final_loss_mean_last5"]
    summary = {}
    for f in sorted(HERE.glob("toy_*.json")):
        if f.name == "toy_results.json":
            continue
        d = json.loads(f.read_text())
        runs = list(d["seeds"].values())
        summary[d["config"]] = {
            "steps": d["steps"], "batch": d["batch"], "hard_neg": d["hard_neg"], "n_seeds": len(runs),
            "per_seed": {k: [r[k] for r in runs] for k in keys},
            "mean_ci95": {k: interval([r[k] for r in runs]) for k in keys},
            "n_eval": {"held_out": runs[0]["n_held_out"], "seen": runs[0]["n_seen"], "two": runs[0]["n_two"]},
        }
    # 같은 시드끼리 짝지은 차이(시드 0-4가 같은 초기화·같은 데이터 생성 시드를 씀). t 구간과 양측 짝 t 검정.
    pairs = {}
    try:
        from scipy import stats
    except ImportError:
        stats = None
    for a, b in (("hardneg", "baseline"), ("baseline", "smallbatch")):
        if a in summary and b in summary:
            for k in ("zs_held_out_photo", "bind_attr_acc", "bind_order_acc"):
                d = [x - y for x, y in zip(summary[a]["per_seed"][k], summary[b]["per_seed"][k])]
                rec = {"diff_mean_ci95": interval(d), "per_seed_diff": d}
                if stats is not None:
                    rec["paired_t_p"] = float(stats.ttest_rel(summary[a]["per_seed"][k], summary[b]["per_seed"][k]).pvalue)
                pairs[f"{a}-{b}:{k}"] = rec
    summary["_paired"] = pairs
    (HERE / "toy_results.json").write_text(json.dumps(summary, indent=1))
    for c, s in summary.items():
        if c == "_paired":
            for k, v in s.items():
                print(k, [round(x, 3) for x in v["diff_mean_ci95"]], v.get("paired_t_p"))
            continue
        print(c, s["n_seeds"], {k: [round(x, 3) for x in v] for k, v in s["mean_ci95"].items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", choices=list(CONFIGS), required=True)
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    t.add_argument("--steps", type=int, default=2500)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)
    g = sub.add_parser("gap")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--steps", type=int, default=2500)
    g.add_argument("--out", default=str(HERE / "gap_results.json"))
    g.set_defaults(fn=cmd_gap)
    s = sub.add_parser("summary")
    s.set_defaults(fn=cmd_summary)
    a = p.parse_args()
    a.fn(a)
