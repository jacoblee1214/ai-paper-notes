"""LLaVA 토이 실험 (합성 장면, CPU). 논문 재현이 아니다.

arm(설정)마다 같은 시드의 사전학습된 비전 인코더와 LM 에서 출발한다(짝지은 비교).
  base            선형 투영, 끝에서 두 번째 층, 1단계(투영만) + 2단계(투영 + LM)
  one_stage       선형, -2, 2단계만 (투영기는 무작위 초기화에서 LM 과 같이 학습)  -- Prismatic 의 단일 단계
  one_stage_long  선형, -2, 2단계만이지만 스텝 수를 1단계 + 2단계 합과 같게(계산량 통제)
  mlp             2층 MLP 투영(LLaVA-1.5), -2, 두 단계
  mlp_one_stage   2층 MLP, -2, 2단계만
  last_layer      선형, -1(마지막 층 출력), 두 단계
  layer_m3        선형, -3, 두 단계
사용: python llava/analysis/run_experiments.py train --seeds 0 1  (저장소 루트에서)
      python llava/analysis/run_experiments.py summary
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "toy_runs"

HP = dict(vision_steps=400, lm_steps=1000, s1_steps=1200, s2_steps=2000, batch=64,
          lr_s1=2e-3, lr_s2=1e-3, pre_lr_vision=2e-3, pre_lr_lm=1e-3)

ARMS = {
    # 이름: (투영기, 층, s1 스텝, s2 스텝 배율 설명)
    "base": dict(projector="linear", layer=-2, s1=HP["s1_steps"], s2=HP["s2_steps"]),
    "one_stage": dict(projector="linear", layer=-2, s1=0, s2=HP["s2_steps"]),
    "one_stage_long": dict(projector="linear", layer=-2, s1=0, s2=HP["s1_steps"] + HP["s2_steps"]),
    "mlp": dict(projector="mlp2x_gelu", layer=-2, s1=HP["s1_steps"], s2=HP["s2_steps"]),
    "mlp_one_stage": dict(projector="mlp2x_gelu", layer=-2, s1=0, s2=HP["s2_steps"]),
    "last_layer": dict(projector="linear", layer=-1, s1=HP["s1_steps"], s2=HP["s2_steps"]),
    "layer_m3": dict(projector="linear", layer=-3, s1=HP["s1_steps"], s2=HP["s2_steps"]),
}

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


@torch.no_grad()
def probe_layers(vision, seed):
    """고정된 인코더의 층별 패치 특징이 '이 칸의 색(배경 포함 7종)'과 '물체가 있는가'를 얼마나 선형으로 읽히게 두는가.
    릿지 회귀(닫힌 식)로 학습 장면에서 맞추고 새 장면에서 정확도를 잰다. 층 선택 가설(끝에서 두 번째가 낫다)의 진단용이다."""
    gen = torch.Generator().manual_seed(7_000 + seed)

    def feats(n):
        sc = M.sample_scenes(n, gen)
        hs = vision.hidden_states(M.render(sc, gen))
        grid = torch.full((n, M.G * M.G), 0, dtype=torch.long)
        for j in range(M.MAX_OBJ):
            idx = torch.nonzero(sc.cells[:, j] >= 0).squeeze(1)
            grid[idx, sc.cells[idx, j]] = 1 + sc.colors[idx, j]
        return hs, grid.reshape(-1), sc.nobj

    hs_tr, y_tr, n_tr = feats(600)
    hs_te, y_te, n_te = feats(300)
    out = {}
    for name, li in [("m1", -1), ("m2", -2), ("m3", -3), ("m4", -4)]:
        x_tr = hs_tr[li][:, 1:].reshape(-1, vision.dim)
        x_te = hs_te[li][:, 1:].reshape(-1, vision.dim)
        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
        a = torch.cat([(x_tr - mu) / sd, torch.ones(len(x_tr), 1)], 1)
        b = torch.cat([(x_te - mu) / sd, torch.ones(len(x_te), 1)], 1)
        t = torch.nn.functional.one_hot(y_tr, 1 + M.NCOL).float()
        w = torch.linalg.solve(a.T @ a + 1.0 * torch.eye(a.shape[1]), a.T @ t)
        acc = float(((b @ w).argmax(1) == y_te).float().mean())
        # 같은 특징에서 "이 패치가 64칸 중 어디인가"(위치 읽힘). 위치는 장면과 무관한 정답이라 학습 장면 수가 적어도 된다.
        pos_tr = torch.arange(M.G * M.G).repeat(600)
        pos_te = torch.arange(M.G * M.G).repeat(300)
        tp = torch.nn.functional.one_hot(pos_tr, M.G * M.G).float()
        wp = torch.linalg.solve(a.T @ a + 1.0 * torch.eye(a.shape[1]), a.T @ tp)
        pos_acc = float(((b @ wp).argmax(1) == pos_te).float().mean())
        # "한 패치 토큰만 보고 장면의 물체 수(1-4)를 맞히는가": 전역 정보가 패치 토큰에 섞여 있는 정도. 논문의 '마지막 층은 전역적' 가설의 진단.
        tc = torch.nn.functional.one_hot(n_tr.repeat_interleave(M.G * M.G), M.MAX_OBJ + 1).float()
        wc = torch.linalg.solve(a.T @ a + 1.0 * torch.eye(a.shape[1]), a.T @ tc)
        cnt_acc = float(((b @ wc).argmax(1) == n_te.repeat_interleave(M.G * M.G)).float().mean())
        out[name] = {"patch_color_probe_acc": acc, "patch_position_probe_acc": pos_acc, "patch_count_probe_acc": cnt_acc,
                     "mean_token_norm": float(hs_tr[li][:, 1:].norm(dim=-1).mean())}
    return out


def run_seed(seed, arm_names, threads, device):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    vision = M.TinyViT().to(device)
    v_loss = M.pretrain_vision(vision, gen, HP["vision_steps"], HP["batch"], HP["pre_lr_vision"])
    lm = M.TinyLM().to(device)
    lm_loss = M.pretrain_lm(lm, gen, HP["lm_steps"], HP["batch"], HP["pre_lr_lm"])
    vision, lm = vision.cpu(), lm.cpu()                       # arm 마다 복사해 쓰므로 CPU 에 보관
    res = {"seed": seed, "hp": HP, "device": device, "vision_pretrain_loss": v_loss, "lm_pretrain_loss": lm_loss,
           "probe": probe_layers(vision, seed), "arms": {}}
    print(f"[seed {seed}] pretrain done {time.time() - t0:.0f}s vis {v_loss:.3f} lm {lm_loss:.3f}", flush=True)
    for name in arm_names:
        cfg = ARMS[name]
        model = M.build_model(vision, lm, cfg["projector"], cfg["layer"], seed).to(device)
        g = torch.Generator().manual_seed(1_000 + seed)       # 모든 arm 이 같은 데이터 순서에서 출발
        t = time.time()
        row = {"cfg": cfg}
        if cfg["s1"]:
            row["s1_loss"] = M.train_stage(model, 1, g, cfg["s1"], HP["batch"], HP["lr_s1"])
            row["s1_caption"] = M.caption_eval(model)
        row["s2_loss"] = M.train_stage(model, 2, g, cfg["s2"], HP["batch"], HP["lr_s2"])
        row["eval"] = M.evaluate(model)
        row["seconds"] = time.time() - t
        res["arms"][name] = row
        print(f"[seed {seed}] {name:15s} spatial {row['eval']['spatial']:.3f} macro {row['eval']['macro']:.3f} "
              f"({row['seconds']:.0f}s)", flush=True)
        RUNS.mkdir(exist_ok=True)
        (RUNS / f"seed{seed}.json").write_text(json.dumps(res, indent=1))
    return res


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    half = T975[n - 1] * x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def summary():
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("seed*.json"))]
    seeds = [r["seed"] for r in runs]
    arms = [a for a in ARMS if all(a in r["arms"] for r in runs)]
    metrics = ["spatial", "macro"] + M.QTYPES
    out = {"seeds": seeds, "n_seeds": len(seeds), "hp": runs[0]["hp"], "arms": {}, "paired_vs_base": {}, "probe": {}}
    for a in arms:
        out["arms"][a] = {m: dict(zip(("mean", "lo", "hi"), ci([r["arms"][a]["eval"][m] for r in runs]))) for m in metrics}
        out["arms"][a]["per_seed_spatial"] = [r["arms"][a]["eval"]["spatial"] for r in runs]
        if "s1_caption" in runs[0]["arms"][a]:
            out["arms"][a]["s1_content_acc"] = dict(zip(("mean", "lo", "hi"), ci([r["arms"][a]["s1_caption"]["content_acc"] for r in runs])))
            out["arms"][a]["s1_loss"] = dict(zip(("mean", "lo", "hi"), ci([r["arms"][a]["s1_loss"] for r in runs])))
        out["arms"][a]["s2_train_loss"] = dict(zip(("mean", "lo", "hi"), ci([r["arms"][a]["s2_loss"] for r in runs])))
    for a in arms:
        if a == "base":
            continue
        d = {}
        for m in metrics:
            diffs = [r["arms"][a]["eval"][m] - r["arms"]["base"]["eval"][m] for r in runs]
            mean, lo, hi = ci(diffs)
            d[m] = {"mean": mean, "lo": lo, "hi": hi, "n_positive": int(sum(x > 0 for x in diffs)), "diffs": diffs}
        out["paired_vs_base"][a] = d
    for k in ["m1", "m2", "m3", "m4"]:
        out["probe"][k] = {
            "patch_color_probe_acc": dict(zip(("mean", "lo", "hi"), ci([r["probe"][k]["patch_color_probe_acc"] for r in runs]))),
            "patch_position_probe_acc": dict(zip(("mean", "lo", "hi"), ci([r["probe"][k]["patch_position_probe_acc"] for r in runs]))),
            "patch_count_probe_acc": dict(zip(("mean", "lo", "hi"), ci([r["probe"][k]["patch_count_probe_acc"] for r in runs]))),
            "mean_token_norm": dict(zip(("mean", "lo", "hi"), ci([r["probe"][k]["mean_token_norm"] for r in runs]))),
        }
    out["pretrain"] = {"vision_loss": ci([r["vision_pretrain_loss"] for r in runs]), "lm_loss": ci([r["lm_pretrain_loss"] for r in runs])}
    (HERE / "toy_results.json").write_text(json.dumps(out, indent=1))
    for a in arms:
        e = out["arms"][a]
        print(f"{a:15s} spatial {e['spatial']['mean']:.3f} [{e['spatial']['lo']:.3f},{e['spatial']['hi']:.3f}]  "
              + " ".join(f"{m} {e[m]['mean']:.3f}" for m in M.QTYPES))
    for a, d in out["paired_vs_base"].items():
        x = d["spatial"]
        print(f"  {a:15s} vs base spatial {x['mean']:+.3f} [{x['lo']:+.3f},{x['hi']:+.3f}] pos {x['n_positive']}/{len(seeds)}")
    print("probe (color, position, count-from-one-patch, norm)", {k: (round(v['patch_color_probe_acc']['mean'], 3), round(v['patch_position_probe_acc']['mean'], 3), round(v['patch_count_probe_acc']['mean'], 3), round(v['mean_token_norm']['mean'], 2)) for k, v in out["probe"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "summary"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    if a.cmd == "train":
        for s in a.seeds:
            run_seed(s, a.arms, a.threads, a.device)
    else:
        summary()
