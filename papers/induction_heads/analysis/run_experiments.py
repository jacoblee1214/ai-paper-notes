"""실험 실행: (a) 합성 반복 토큰 데이터의 단계 변화, (b) 자연 텍스트 2층 MLP 모델, (c) 31번에 넘길 artifacts.

  python analysis/run_experiments.py synthetic            # synthetic_results.json (CPU 약 8분)
  python analysis/run_experiments.py natural              # natural_results.json + artifacts/ (MPS 또는 CPU)
  python analysis/run_experiments.py artifacts            # 저장된 체크포인트로 activation만 다시 뽑기

torch가 설치된 python(MPS 또는 CPU)으로 돌린다. 결과 json은 analysis/ 에 쓴다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_impl import (BOS, Cfg, TinyLM, ablation_effects, build_corpus, copy_scores_eig, dup_losses, dup_windows, fixed_repeat_batch,
                          head_table, icl_score, kcomp_trace, per_pos_loss, pick_device, previous_token_score,
                          repeat_batch, repeat_losses, train, train_batch, val_position_loss, val_windows)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART = ROOT / "artifacts"

# ---------------------------------------------------------------- (a) 합성 데이터
SYN = dict(V=64, T=64, Lmin=6, Lmax=20, d=128, n_head=4, B=64, lr=5e-4, warmup=50, steps=1500, eval_every=10)
SYN_CONFIGS = {
    "1L": dict(n_layer=1, smear=False),
    "2L": dict(n_layer=2, smear=False),
    "1L_smear": dict(n_layer=1, smear=True),
}


def crossing(steps, vals, thr, below=True):
    for s, v in zip(steps, vals):
        if (v <= thr) if below else (v >= thr):
            return s
    return None


def run_synthetic(seeds, device, out):
    c = SYN
    res = {"setup": c, "configs": {}}
    for name, kw in SYN_CONFIGS.items():
        res["configs"][name] = {}
        for seed in seeds:
            t0 = time.time()
            torch.manual_seed(seed)
            cfg = Cfg(vocab=c["V"], d=c["d"], n_layer=kw["n_layer"], n_head=c["n_head"], ctx=c["T"], smear=kw["smear"])
            model = TinyLM(cfg).to(device)
            gdata = torch.Generator().manual_seed(1000 + seed)
            gev = torch.Generator().manual_seed(999)       # 평가 세트는 시드와 무관하게 같다
            ev, evL = repeat_batch(gev, 256, c["T"], c["V"], c["Lmin"], c["Lmax"])
            ev = ev.to(device)

            def ef(m, step):
                with torch.no_grad():
                    r, f = repeat_losses(m(ev), ev, evL)
                    ht = head_table(m, torch.Generator().manual_seed(5), device, L=12, n_rep=4, hi=c["V"])
                return {"rep": r.item(), "first": f.item(), "prefix": ht["prefix"].tolist()}

            hist = train(model, lambda s: repeat_batch(gdata, c["B"], c["T"], c["V"], c["Lmin"], c["Lmax"])[0],
                         c["steps"], c["lr"], ef, c["eval_every"], c["warmup"], device=device)
            steps = [h["step"] for h in hist]
            rep = [h["rep"] for h in hist]
            pmax = [float(np.max(h["prefix"])) for h in hist]
            r0, r1 = rep[0], float(np.mean(rep[-5:]))
            p1 = float(np.mean(pmax[-5:]))
            entry = {
                "rep_loss_start": r0, "rep_loss_end": r1, "first_loss_end": float(np.mean([h["first"] for h in hist[-5:]])),
                "prefix_max_end": p1,
                "tau_loss": crossing(steps, rep, (r0 + r1) / 2) if r0 - r1 > 1.0 else None,
                "tau_prefix": crossing(steps, pmax, 0.5 * p1, below=False) if p1 > 0.3 else None,
                "history": {"step": steps, "rep": rep, "first": [h["first"] for h in hist], "prefix_max": pmax},
            }
            # 학습 끝 상태의 헤드별 측정과 ablation
            with torch.no_grad():
                ht = head_table(model, torch.Generator().manual_seed(7), device, L=12, n_rep=4, hi=c["V"],
                                natural_tokens=ev[:64])
                pm = ht["prefix"]
                groups = {"layer0_all": [(0, h) for h in range(c["n_head"])],
                          "prefix_gt_0.3": [(l, h) for l in range(pm.shape[0]) for h in range(c["n_head"]) if pm[l, h] > 0.3]}
                groups = {k: v for k, v in groups.items() if v}
                if kw["n_layer"] == 2:
                    groups["layer1_all"] = [(1, h) for h in range(c["n_head"])]
                ab = ablation_effects(model, ev, lambda lg, t: repeat_losses(lg, t, evL)[0], device, groups)
            entry["heads"] = {k: v.tolist() for k, v in ht.items()}
            entry["ablation_rep_loss"] = ab
            if kw["n_layer"] == 2:
                with torch.no_grad():
                    pw = ht["prev_token"][0].to(device)
                    entry["kcomp_trace_layer1"] = [kcomp_trace(model, 1, h, pw) for h in range(c["n_head"])]
            res["configs"][name][str(seed)] = entry
            print(f"[syn] {name} seed {seed}: rep {r0:.2f}->{r1:.3f}, prefix_max {p1:.2f}, tau_loss {entry['tau_loss']}, "
                  f"tau_prefix {entry['tau_prefix']} ({time.time() - t0:.0f}s)", flush=True)
    Path(out).write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------- (b) 자연 텍스트
NAT = dict(V=2048, d=128, n_layer=2, n_head=4, d_mlp=512, ctx=256, B=32, lr=1e-3, warmup=100, steps=3000, eval_every=50,
           L=25, n_rep=4, lo=52, hi=1948, dup=0.0)


def natural_eval_fns(windows, device):
    c = NAT

    def rep_random(model, gen):
        toks = fixed_repeat_batch(gen, 64, c["L"], c["n_rep"], c["lo"], c["hi"]).to(device)
        with torch.no_grad():
            r, f = repeat_losses(model(toks), toks, c["L"])
        return r.item(), f.item()

    def ef(model, step):
        pl = val_position_loss(model, windows, device)
        with torch.no_grad():
            ht = head_table(model, torch.Generator().manual_seed(5), device, L=c["L"], n_rep=c["n_rep"], lo=c["lo"],
                            hi=c["hi"], B=32, natural_tokens=windows[:16])
        r, f = rep_random(model, torch.Generator().manual_seed(11))
        d1, d2 = dup_losses(model, windows, device)
        return {"dup_first": d1, "dup_second": d2, "val_loss": pl.mean().item(), "icl": icl_score(pl), "pos_loss": [round(x, 4) for x in pl.tolist()],
                "rep_random": r, "first_random": f, "prefix": ht["prefix"].tolist()}
    return ef, rep_random


def natural_ablation(model, windows, device, rep_random):
    """헤드를 하나씩 끄고(전체 / 패턴 유지) 검증 손실, ICL 점수, 반복 무작위 열의 반복 구간 손실을 잰다."""
    out = {"full": {}, "preserve": {}}
    c = NAT

    def eval_with(ablate, preserve):
        s, n = 0, 0
        with torch.no_grad():
            for i in range(0, len(windows), 64):
                t = windows[i:i + 64].to(device)
                fr = None
                if preserve:
                    _, cc = model(t, cache=True)
                    fr = [p.detach() for p in cc["patterns"]]
                s = s + per_pos_loss(model(t, ablate=ablate, freeze=fr), t).sum(0)
                n += t.shape[0]
            pl = (s / n).cpu()
            toks = fixed_repeat_batch(torch.Generator().manual_seed(11), 64, c["L"], c["n_rep"], c["lo"], c["hi"]).to(device)
            fr = None
            if preserve:
                _, cc = model(toks, cache=True)
                fr = [p.detach() for p in cc["patterns"]]
            lg = model(toks, ablate=ablate, freeze=fr)
            rr, _ = repeat_losses(lg, toks, c["L"])
            # 반복 구절 세트(둘째 사본 손실)
            dwin, hh = dup_windows(windows)
            d1 = d2 = 0.0
            for i in range(0, len(dwin), 64):
                t = dwin[i:i + 64].to(device)
                fr = None
                if preserve:
                    _, cc = model(t, cache=True)
                    fr = [p.detach() for p in cc["patterns"]]
                l = per_pos_loss(model(t, ablate=ablate, freeze=fr), t)
                d1 += l[:, :hh].sum().item()
                d2 += l[:, hh:2 * hh].sum().item()
            n = len(dwin) * hh
        return {"val_loss": pl.mean().item(), "icl": icl_score(pl), "rep_random": rr.item(), "dup_first": d1 / n,
                "dup_second": d2 / n}

    out["base"] = eval_with(None, False)
    for mode, pres in (("full", False), ("preserve", True)):
        for l in range(model.cfg.n_layer):
            for h in range(model.cfg.n_head):
                out[mode][f"{l}.{h}"] = eval_with((l, h), pres)
    return out


def run_natural(seeds, device, out, save_artifacts=True):
    c = NAT
    t0 = time.time()
    corp = build_corpus(vocab=c["V"])
    print(f"[nat] corpus: {corp['n_tokens']} tokens, train {len(corp['train'])}, val {len(corp['val'])}, "
          f"sha256 {corp['sha256'][:16]} ({time.time() - t0:.0f}s)", flush=True)
    windows = val_windows(corp["val"], c["ctx"])
    ef, rep_random = natural_eval_fns(windows, device)
    res = {"setup": c, "corpus": {"n_tokens": corp["n_tokens"], "train_tokens": int(len(corp["train"])),
                                  "val_tokens": int(len(corp["val"])), "val_windows": int(len(windows)),
                                  "sha256": corp["sha256"], "python": sys.version.split()[0]},
           "seeds": {}}
    for seed in seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        cfg = Cfg(vocab=c["V"], d=c["d"], n_layer=c["n_layer"], n_head=c["n_head"], d_mlp=c["d_mlp"], ctx=c["ctx"])
        model = TinyLM(cfg).to(device)
        gdata = torch.Generator().manual_seed(1000 + seed)
        hist = train(model, lambda s: train_batch(corp["train"], gdata, c["B"], c["ctx"], c["dup"]), c["steps"], c["lr"], ef,
                     c["eval_every"], c["warmup"], device=device)
        steps = [h["step"] for h in hist]
        pmax = [float(np.max(h["prefix"])) for h in hist]
        icl = [h["icl"] for h in hist]
        rr = [h["rep_random"] for h in hist]
        entry = {"history": {"step": steps, "val_loss": [h["val_loss"] for h in hist], "icl": icl,
                             "rep_random": rr, "first_random": [h["first_random"] for h in hist],
                             "dup_first": [h["dup_first"] for h in hist], "dup_second": [h["dup_second"] for h in hist],
                             "prefix_max": pmax, "prefix": [h["prefix"] for h in hist]},
                 "pos_loss_end": hist[-1]["pos_loss"], "pos_loss_trace": {str(h["step"]): h["pos_loss"] for h in hist[::4]}}
        r0, r1 = rr[0], float(np.mean(rr[-3:]))
        p1 = float(np.mean(pmax[-3:]))
        entry["summary"] = {"val_loss_end": hist[-1]["val_loss"], "icl_end": float(np.mean(icl[-3:])), "icl_start": icl[0],
                            "rep_random_start": r0, "rep_random_end": r1, "prefix_max_end": p1,
                            "tau_rep_random": crossing(steps, rr, (r0 + r1) / 2) if r0 - r1 > 1.0 else None,
                            "tau_prefix": crossing(steps, pmax, 0.5 * p1, below=False) if p1 > 0.3 else None}
        with torch.no_grad():
            ht = head_table(model, torch.Generator().manual_seed(7), device, L=c["L"], n_rep=c["n_rep"], lo=c["lo"],
                            hi=c["hi"], B=32, natural_tokens=windows[:64])
            pw = ht["prev_token"][0].to(device)
            entry["kcomp_trace_layer1"] = [kcomp_trace(model, 1, h, pw) for h in range(c["n_head"])]
        entry["heads"] = {k: v.tolist() for k, v in ht.items()}
        entry["ablation"] = natural_ablation(model, windows, device, rep_random)
        res["seeds"][str(seed)] = entry
        print(f"[nat] seed {seed}: val {entry['summary']['val_loss_end']:.3f} icl {entry['summary']['icl_end']:.3f} "
              f"rep_random {r0:.2f}->{r1:.3f} prefix_max {p1:.2f} tau_rep {entry['summary']['tau_rep_random']} "
              f"tau_prefix {entry['summary']['tau_prefix']} ({time.time() - t0:.0f}s)", flush=True)
        if seed == seeds[0] and save_artifacts:
            ART.mkdir(exist_ok=True)
            torch.save({"cfg": cfg.dict(), "state": {k: v.cpu() for k, v in model.state_dict().items()}, "seed": seed,
                        "corpus_sha256": corp["sha256"]}, ART / f"model_seed{seed}.pt")
            (ART / "vocab.json").write_text(json.dumps(corp["itos"], ensure_ascii=False))
            make_artifacts(model, windows, device)
    Path(out).write_text(json.dumps(res))
    return res


@torch.no_grad()
def make_artifacts(model, windows, device, n_seq=512):
    """검증 창 앞 n_seq개(= 131,072 토큰)에서 activation을 뽑아 float16 npy로 저장한다.
    tokens.npy (n_seq, ctx) int16 / resid_post_l{0,1}.npy, mlp_out_l{0,1}.npy (n_seq, ctx, d) float16.
    resid_post_l{l}은 층 l의 attention과 MLP를 모두 더한 직후 잔차(ln_f 이전), mlp_out_l{l}은 층 l MLP가 잔차에 더하는 값."""
    ART.mkdir(exist_ok=True)
    toks = windows[:n_seq]
    outs = {"resid_post": [[] for _ in range(model.cfg.n_layer)], "mlp_out": [[] for _ in range(model.cfg.n_layer)]}
    model.eval()
    for i in range(0, len(toks), 64):
        _, c = model(toks[i:i + 64].to(device), cache=True)
        for l in range(model.cfg.n_layer):
            outs["resid_post"][l].append(c["resid_post"][l].half().cpu())
            outs["mlp_out"][l].append(c["mlp_out"][l].half().cpu())
    np.save(ART / "tokens.npy", toks.numpy().astype(np.int16))
    for kind in outs:
        for l in range(model.cfg.n_layer):
            np.save(ART / f"{kind}_l{l}.npy", torch.cat(outs[kind][l]).numpy())
    print("[artifacts]", {p.name: p.stat().st_size for p in sorted(ART.iterdir())}, flush=True)


def regen_artifacts(device):
    ck = torch.load(ART / "model_seed0.pt", map_location="cpu")
    model = TinyLM(Cfg(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    corp = build_corpus(vocab=ck["cfg"]["vocab"])
    assert corp["sha256"] == ck["corpus_sha256"], "코퍼스가 학습 때와 다릅니다(파이썬 표준 라이브러리 버전 차이)"
    make_artifacts(model, val_windows(corp["val"], ck["cfg"]["ctx"]), device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["synthetic", "natural", "artifacts"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None, help="스모크 시험용 학습 스텝 덮어쓰기")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--dup", type=float, default=None, help="반복 구절 증강 비율(0이면 자연 텍스트만)")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.what == "synthetic":
        if a.steps:
            SYN["steps"] = a.steps
        run_synthetic(a.seeds or [0, 1, 2, 3, 4], pick_device(a.device or "cpu"), a.out or HERE / "synthetic_results.json")
    elif a.what == "natural":
        if a.steps:
            NAT["steps"] = a.steps
        if a.lr:
            NAT["lr"] = a.lr
        if a.batch:
            NAT["B"] = a.batch
        if a.dup is not None:
            NAT["dup"] = a.dup
        if a.eval_every:
            NAT["eval_every"] = a.eval_every
        run_natural(a.seeds or [0, 1, 2], pick_device(a.device or "auto"), a.out or HERE / "natural_results.json",
                    save_artifacts=not a.no_artifacts)
    else:
        regen_artifacts(pick_device(a.device or "auto"))
