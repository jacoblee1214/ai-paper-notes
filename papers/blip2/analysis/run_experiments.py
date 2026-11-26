"""BLIP-2 토이 실험: 질의 수 K, Q-Former 대 MLP 투영기, 1단계(표현 학습) 유무. 논문 재현이 아니다.

저장소 루트에서:
  PY=python   # torch가 설치된 파이썬(README 재실행 방법 참고)
  $PY blip2/analysis/run_experiments.py decoder                    # 고정 디코더를 텍스트로 사전학습(한 번, data/decoder.pt)
  $PY blip2/analysis/run_experiments.py train --seed 0 --out blip2/analysis/toy_seed0.json
  $PY blip2/analysis/run_experiments.py summary                    # toy_results.json (시드 평균, t 95% 구간, 짝지은 차이)
시드 하나가 CPU 10-25분이라 시드마다 프로세스를 따로 띄워 병렬로 돌린다.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as m  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
DEC_CFG = dict(d=96, heads=4, layers=3, f=192)
QF_KW = dict(d=64, heads=4, layers=4, ffn=128, d_img=32, embed_dim=32)
KS = [1, 4, 16, 32]
KS_S1 = [4, 16]            # 1단계(4000스텝급)는 비싸서 K=4, 16에서만
S1_STEPS, S2_STEPS, DEC_STEPS = 2500, 2000, 8000
S1_BATCH, S1_LR = 64, 2e-3
WIDE_HIDDEN = 1850        # 은닉 1850이면 MLP 투영기가 약 24.6만 파라미터로 Q-Former(약 24.5만)와 비슷해짐


def load_decoder():
    dec = m.TinyDecoder(**DEC_CFG)
    dec.load_state_dict(torch.load(DATA / "decoder.pt"))
    return m.freeze(dec)


def cmd_decoder(args):
    DATA.mkdir(exist_ok=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    dec = m.TinyDecoder(**DEC_CFG)
    t = time.time()
    m.pretrain_decoder(dec, gen, steps=args.steps)
    torch.save(dec.state_dict(), DATA / "decoder.pt")
    out = {"max_steps": args.steps, "stopped_at": dec.stopped_at, "config": DEC_CFG, "params": sum(p.numel() for p in dec.parameters()),
           "seconds": round(time.time() - t),
           "oracle_caption_acc": m.oracle_caption_acc(dec, torch.Generator().manual_seed(99), n=1000)}
    (HERE / "decoder_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def stage1_eval(qf, enc, gen, n=100):
    """1단계 진단: 질의 K개의 ITC 이미지->텍스트 검색 정확도(후보 n개), ITM 이진 정확도(어려운 음성)."""
    sc = m.sample_scenes(n, gen)
    cap = m.caption_ids(sc, gen)
    with torch.no_grad():
        feats = enc(m.render_patches(sc), m.G)
        q, t = qf(feats, cap, "itc")
        sim, _ = m.itc_similarity(q, t[:, 0], qf.vision_proj, qf.text_proj)
        r1 = (sim.argmax(1) == torch.arange(n)).float().mean().item()
        neg_t, _ = m.sample_hard_negatives(sim, 0.07, gen)
        q_pos, _ = qf(feats, cap, "itm")
        q_neg, _ = qf(feats, cap[neg_t], "itm")
        pos = qf.itm_head(q_pos).mean(1).argmax(-1) == 1
        neg = qf.itm_head(q_neg).mean(1).argmax(-1) == 0
        dec = cap.clone()
        dec[:, 0] = m.DEC
        _, tg = qf(feats, dec, "itg")
        pred = qf.lm_head(tg[:, :-1]).argmax(-1)
        tgt = dec[:, 1:]
        tok_acc = ((pred == tgt) & (tgt != m.PAD)).sum().item() / (tgt != m.PAD).sum().item()
    return {"itc_i2t_r1": r1, "itm_acc": 0.5 * (pos.float().mean().item() + neg.float().mean().item()), "itg_token_acc": tok_acc}


def cmd_train(args):
    seed = args.seed
    torch.set_num_threads(args.threads)
    enc = m.freeze(m.FrozenImageEncoder())
    dec = load_decoder()
    results = {"seed": seed, "s1_steps": args.s1_steps, "s2_steps": args.s2_steps, "configs": {}}
    out = Path(args.out)
    only = set(args.only) if args.only else None
    if only and out.exists():                      # 일부 설정만 추가로 돌릴 때는 기존 결과에 합친다
        results = json.loads(out.read_text())
    ev = lambda vlm: m.qa_accuracy(vlm, torch.Generator().manual_seed(7000 + seed), per_type=500)

    def run(name, make_connector, s1_qf=None, steps=None, extra=None):
        if only and name not in only:
            return
        t0 = time.time()
        torch.manual_seed(seed * 100 + sum(map(ord, name)) % 97)
        gen = torch.Generator().manual_seed(seed * 1000 + 5)
        conn = make_connector()
        vlm = m.VLM(enc, conn, dec)
        losses = m.train_connector(vlm, gen, steps=steps or args.s2_steps)
        acc = ev(vlm)
        rec = {"acc": acc, "mean_acc": sum(acc.values()) / len(acc), "final_loss": sum(losses[-100:]) / 100,
               "trainable_params": sum(p.numel() for p in vlm.parameters() if p.requires_grad),
               "visual_tokens": int(vlm.prefix(m.render_patches(m.sample_scenes(2, gen))).shape[1]),
               "seconds": round(time.time() - t0)}
        if extra:
            rec.update(extra)
        results["configs"][name] = rec
        out.write_text(json.dumps(results, indent=1))
        print(f"seed {seed} {name}: mean {rec['mean_acc']:.3f} {acc} ({rec['seconds']}s)", flush=True)

    for K in KS:
        cfg = m.QFConfig(k_queries=K, **QF_KW)
        # 1단계 없이 곧바로 2단계
        run(f"qf_K{K}_nos1", lambda: m.QFormerConnector(m.QFormer(cfg), DEC_CFG["d"]))
        if K not in KS_S1 or (only and f"qf_K{K}_s1" not in only):
            continue
        # 1단계(ITC+ITM+ITG) 뒤 2단계. 질의, 교차 attention, 공유 self-attention이 그대로 이어진다
        torch.manual_seed(seed * 100 + K)
        qf = m.QFormer(cfg)
        t0 = time.time()
        hist = m.train_stage1(qf, enc, torch.Generator().manual_seed(seed * 1000 + 1), steps=args.s1_steps, batch=S1_BATCH, lr=S1_LR, log_every=args.s1_steps // 5)
        diag = stage1_eval(qf, enc, torch.Generator().manual_seed(8000 + seed))
        extra = {"stage1_seconds": round(time.time() - t0), "stage1_hist": hist, "stage1_diag": diag}
        run(f"qf_K{K}_s1", lambda: m.QFormerConnector(qf, DEC_CFG["d"]), extra=extra)
    # 통제: 1단계 없이 2단계를 (1단계 + 2단계)만큼 오래 (K=16) — "더 오래 학습해서"가 아닌지
    cfg = m.QFConfig(k_queries=16, **QF_KW)
    run("qf_K16_nos1_long", lambda: m.QFormerConnector(m.QFormer(cfg), DEC_CFG["d"]), steps=args.s2_steps + args.s1_steps)
    # MLP 투영기: 패치 64개를 모두 넘김, 2x2 풀링으로 16개
    run("mlp_all64", lambda: m.MLPConnector(32, DEC_CFG["d"]))
    # 통제: MLP가 Q-Former보다 파라미터가 20배 적고(1.2만 대 24.5만) 같은 스텝 수에서 덜 수렴했을 수 있다
    run("mlp_all64_wide", lambda: m.MLPConnector(32, DEC_CFG["d"], hidden=WIDE_HIDDEN))
    run("mlp_all64_long", lambda: m.MLPConnector(32, DEC_CFG["d"]), steps=args.s2_steps + args.s1_steps)
    run("mlp_pool16", lambda: m.MLPConnector(32, DEC_CFG["d"], pool=2))


def t_ci(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, mean, mean
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - h, mean + h


def cmd_summary(args):
    files = sorted(HERE.glob("toy_seed*.json"))
    runs = [json.loads(f.read_text()) for f in files]
    names = list(runs[0]["configs"])
    types = m.QTYPES
    summ = {"seeds": [r["seed"] for r in runs], "s1_steps": runs[0]["s1_steps"], "s2_steps": runs[0]["s2_steps"], "configs": {}, "paired": {}}
    for n in names:
        rows = [r["configs"][n] for r in runs if n in r["configs"]]
        e = {"n_seeds": len(rows), "visual_tokens": rows[0]["visual_tokens"], "trainable_params": rows[0]["trainable_params"]}
        for key in ["mean_acc"] + types:
            xs = [(r["mean_acc"] if key == "mean_acc" else r["acc"][key]) for r in rows]
            mean, lo, hi = t_ci(xs)
            e[key] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "per_seed": [round(x, 4) for x in xs]}
        if "stage1_diag" in rows[0]:
            e["stage1_diag"] = {k: round(sum(r["stage1_diag"][k] for r in rows) / len(rows), 4) for k in rows[0]["stage1_diag"]}
        summ["configs"][n] = e

    def paired(a, b, key="mean_acc"):
        da = []
        for r in runs:
            if a in r["configs"] and b in r["configs"]:
                get = lambda c: r["configs"][c]["mean_acc"] if key == "mean_acc" else r["configs"][c]["acc"][key]
                da.append(get(a) - get(b))
        if not da:
            return None
        mean, lo, hi = t_ci(da)
        return {"mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "per_seed": [round(x, 4) for x in da], "n": len(da)}

    for key in ["mean_acc"] + types:
        for a, b in [("qf_K16_s1", "mlp_all64"), ("qf_K4_s1", "mlp_all64"),
                     ("qf_K16_s1", "mlp_pool16"), ("qf_K16_s1", "qf_K16_nos1"), ("qf_K16_s1", "qf_K16_nos1_long"),
                     ("qf_K4_s1", "qf_K4_nos1"), ("qf_K16_nos1_long", "qf_K16_nos1"),
                     ("qf_K16_nos1", "qf_K4_nos1"), ("qf_K4_nos1", "qf_K1_nos1"), ("qf_K32_nos1", "qf_K16_nos1"), ("qf_K16_nos1", "mlp_all64"),
                     ("qf_K32_nos1", "mlp_all64"), ("mlp_all64", "mlp_pool16"),
                     ("mlp_all64_wide", "mlp_all64"), ("qf_K16_nos1", "mlp_all64_wide"), ("mlp_all64_long", "mlp_all64"),
                     ("qf_K16_nos1_long", "mlp_all64_long"), ("qf_K16_s1", "mlp_all64_long")]:
            summ["paired"].setdefault(f"{a} - {b}", {})[key] = paired(a, b, key)
    (HERE / "toy_results.json").write_text(json.dumps(summ, indent=1))
    for n, e in summ["configs"].items():
        print(f"{n:20s} tok={e['visual_tokens']:3d} mean {e['mean_acc']['mean']:.3f} {e['mean_acc']['ci95']} "
              + " ".join(f"{t}={e[t]['mean']:.2f}" for t in types))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decoder")
    p.add_argument("--steps", type=int, default=DEC_STEPS)
    p = sub.add_parser("train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "toy_seed0.json"))
    p.add_argument("--s1-steps", type=int, default=S1_STEPS)
    p.add_argument("--s2-steps", type=int, default=S2_STEPS)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--only", nargs="*", default=None, help="이 이름의 설정만 (기존 결과 파일에 합침)")
    sub.add_parser("summary")
    a = ap.parse_args()
    {"decoder": cmd_decoder, "train": cmd_train, "summary": cmd_summary}[a.cmd](a)
