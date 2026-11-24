"""시드별 실험: 토이 학습 -> 인과 추적 -> ROME 층/위치 훑기 -> 구성 요소 제거 -> 연속 편집.
실행: python run_experiments.py <seed> ...  또는 aux:<seed> ...   (CPU, 시드당 약 15-20분)  -> results_seed<seed>.json, results_aux_seed<seed>.json
모델 체크포인트는 ../.scratch/에 둔다(커밋하지 않음)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import minimal_impl as mi

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent / ".scratch"
CFG = dict(d=128, n_layer=6, n_head=4, steps=3000, bs=256)
VKW = dict(steps=150, lr=0.3, clamp=10.0, wd=0.5, kl_factor=0.0625)  # v* 최적화 설정(토이에 맞춰 조정, 04 D2)
N_TRACE = 200
N_EDIT = 24
PLEN = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pick_edits(world, rng, n):
    s = rng.randint(0, world.n_subj, n)
    r = rng.randint(0, world.n_rel, n)
    o_old = world.obj[s, r]
    o_new = np.array([rng.choice([x for x in range(*world.obj_range(int(ri))) if x != int(oo)]) for ri, oo in zip(r, o_old)])
    return s, r, o_old, o_new


def run_edit(model, world, layer, s, r, o_old, o_new, C, rng, pos, seed, key_mode="mean", vkw=None):
    """한 사실을 고쳐 평가하고, 편집 전 지표와 함께 돌려준다."""
    vkw = vkw or VKW
    before = mi.evaluate_edit(model, world, s, r, o_old, o_new, seed=seed)
    dW, info = mi.rome_edit(model, world, layer, s, r, o_new, C, rng, pos=pos, key_mode=key_mode, **vkw)
    with mi.Edited(model, layer, dW):
        after = mi.evaluate_edit(model, world, s, r, o_old, o_new, seed=seed)
    after["d_nb_prob"] = before["nb_prob"] - after["nb_prob"]
    after["d_glob_prob"] = before["glob_prob"] - after["glob_prob"]
    after["d_glob_acc"] = before["glob_acc"] - after["glob_acc"]
    after["nll"], after["dW_fro"] = info["nll"], info["dW_fro"]
    after["es_before"] = before["es"]
    return after


def main(seed, variant="base"):
    """variant='base': 마지막 토큰 손실만. 'aux': 주어 마지막 토큰에서 속성 분포를 맞추는 보조 손실을 더함."""
    t0 = time.time()
    world = mi.World(seed=seed)
    tag = "" if variant == "base" else f"_{variant}"
    ck = SCRATCH / f"model{tag}_seed{seed}.pt"
    model = mi.GPT(world.vocab, CFG["d"], CFG["n_layer"], CFG["n_head"], world.max_len)
    if ck.exists():
        model.load_state_dict(torch.load(ck))
        model.eval()
        [p.requires_grad_(False) for p in model.parameters()]
    else:
        log(f"seed {seed}: 학습")
        model = mi.train_model(world, seed=seed, d=CFG["d"], n_layer=CFG["n_layer"], n_head=CFG["n_head"],
                               steps=CFG["steps"], bs=CFG["bs"], log=log, aux_bs=(64 if variant == "aux" else 0))
        torch.save(model.state_dict(), ck)
    out = dict(seed=seed, variant=variant, cfg=CFG, vkw=VKW)
    out["acc_templateA"] = float(mi.fact_accuracy(model, world))
    out["acc_templateB"] = float(mi.fact_accuracy(model, world, tmpl=1))
    out["acc_noprefix"] = float(mi.fact_accuracy(model, world, plen=0))
    log("정확도", out["acc_templateA"], out["acc_templateB"], out["acc_noprefix"])
    L = model.n_layer

    # ---------------- 인과 추적 ----------------
    rng = np.random.RandomState(seed + 10)
    s_t = rng.randint(0, world.n_subj, N_TRACE)
    r_t = rng.randint(0, world.n_rel, N_TRACE)
    res = mi.causal_trace(model, world, s_t, r_t, noise_mult=3.0, n_draws=10, plen=PLEN, windows=(1, 3), seed=seed)
    out["trace_gn3"] = mi.summarize_trace(res)
    te = res["p_clean"] - res["p_corrupt"]
    log("추적 GN 3σ ATE", out["trace_gn3"]["ate"])
    out["sigma"] = mi.embedding_std(model, world)
    # 잡음 크기 대 토큰 간 거리
    ids = torch.from_numpy(np.unique(world.subj))
    E = model.tok(ids)
    dist = torch.cdist(E, E)
    med = float(dist[dist > 0].median())
    out["noise_norm_over_median_token_dist"] = {str(m): m * out["sigma"] * (model.d ** 0.5) / med for m in (0.5, 1, 2, 3, 6)}
    # 잡음 크기 훑기와 STR (잔차 상태와 창 3의 MLP/attn만)
    out["trace_sweep"] = {}
    for mult in (0.5, 1.0, 2.0, 3.0, 6.0):
        rs = mi.causal_trace(model, world, s_t[:100], r_t[:100], noise_mult=mult, n_draws=10, plen=PLEN, windows=(3,), seed=seed, sever=False)
        out["trace_sweep"][str(mult)] = mi.summarize_trace(rs)
        log("  noise", mult, "ATE", out["trace_sweep"][str(mult)]["ate"])
    rs = mi.causal_trace(model, world, s_t[:100], r_t[:100], mode="STR", n_draws=10, plen=PLEN, windows=(3,), seed=seed, sever=False)
    out["trace_str"] = mi.summarize_trace(rs)
    log("  STR ATE", out["trace_str"]["ate"])

    # ---------------- ROME: 층 x 위치 훑기 ----------------
    Cs, toks = mi.key_second_moment(model, world, (100, 1000, 20000))
    out["C_tokens"] = {str(k): v for k, v in toks.items()}
    out["C_cond_full"] = [float(torch.linalg.cond(c)) for c in Cs[20000]]
    erng = np.random.RandomState(seed + 20)
    # 편집 대상은 추적한 사실 중 앞 N_EDIT개로 해서 사실별 추적 효과와 짝지을 수 있게 한다
    es_, er_ = s_t[:N_EDIT], r_t[:N_EDIT]
    eo_old = world.obj[es_, er_]
    eo_new = np.array([erng.choice([x for x in range(*world.obj_range(int(ri))) if x != int(oo)]) for ri, oo in zip(er_, eo_old)])
    out["edit_facts"] = dict(s=es_.tolist(), r=er_.tolist(), o_old=eo_old.tolist(), o_new=eo_new.tolist())
    out["tracing_per_fact"] = {
        "te": te[:N_EDIT].tolist(),
        "resid": res["ie"]["resid"][:N_EDIT].tolist(),        # (fact, layer, pos)
        "mlp_w1": res["ie"]["mlp_w1"][:N_EDIT].tolist(),
        "attn_w1": res["ie"]["attn_w1"][:N_EDIT].tolist(),
    }
    keys = ["es", "ps", "pm", "pp", "nb_near", "ns_near", "d_nb_prob", "nb_rand", "same_subj", "d_glob_prob", "d_glob_acc", "nll", "dW_fro"]
    sweep = {}
    for pos in (("last", "final", "first") if variant == "base" else ("last", "final")):  # aux 변형은 시간 때문에 first 생략
        sweep[pos] = {}
        for layer in range(L):
            rows = []
            for i in range(N_EDIT):
                rows.append(run_edit(model, world, layer, int(es_[i]), int(er_[i]), int(eo_old[i]), int(eo_new[i]),
                                     Cs[20000][layer], np.random.RandomState(1000 * seed + i), pos, seed=i))
            sweep[pos][str(layer)] = {k: [r_[k] for r_ in rows] for k in keys}
            log(f"  sweep {pos} L{layer}: es {np.mean(sweep[pos][str(layer)]['es']):.2f} ps {np.mean(sweep[pos][str(layer)]['ps']):.2f} "
                f"nb {np.mean(sweep[pos][str(layer)]['nb_near']):.2f} pm {np.mean(sweep[pos][str(layer)]['pm']):.2f}")
    out["sweep"] = sweep
    out["sweep_keys"] = keys

    # ---------------- 구성 요소 제거: 가장 잘 듣는 (위치, 층)에서 ----------------
    best = max(((p, l) for p in ("last", "final") for l in range(L)),
               key=lambda pl: np.mean(sweep[pl[0]][str(pl[1])]["pm"]))
    out["ablation_site"] = dict(pos=best[0], layer=best[1])
    log("ablation site", best)
    bp, bl = best
    variants = {
        "full (C: 99,960 토큰, 키 평균)": dict(C=Cs[20000][bl]),
        "C = I": dict(C=None),
        "C: 4,996 토큰": dict(C=Cs[1000][bl]),
        "C: 506 토큰": dict(C=Cs[100][bl]),
        "공식 코드식 비대칭 키": dict(C=Cs[20000][bl], key_mode="plain"),
        "KL 항 없음": dict(C=Cs[20000][bl], vkw={**VKW, "kl_factor": 0.0}),
        "정규화·KL 모두 없음": dict(C=Cs[20000][bl], vkw={**VKW, "kl_factor": 0.0, "wd": 0.0}),
        "노름 상한 4배(공식 설정)": dict(C=Cs[20000][bl], vkw={**VKW, "clamp": 4.0}),
    }
    abl = {}
    for name, kw in variants.items():
        rows = []
        for i in range(N_EDIT):
            rows.append(run_edit(model, world, bl, int(es_[i]), int(er_[i]), int(eo_old[i]), int(eo_new[i]), kw["C"],
                                 np.random.RandomState(1000 * seed + i), bp, seed=i, key_mode=kw.get("key_mode", "mean"), vkw=kw.get("vkw")))
        abl[name] = {k: [r_[k] for r_ in rows] for k in keys}
        log(f"  abl {name}: es {np.mean(abl[name]['es']):.2f} ps {np.mean(abl[name]['ps']):.2f} nb {np.mean(abl[name]['nb_near']):.2f} "
            f"d_nb_prob {np.mean(abl[name]['d_nb_prob']):.4f} d_glob_acc {np.mean(abl[name]['d_glob_acc']):.4f} dW {np.mean(abl[name]['dW_fro']):.1f}")
    out["ablation"] = abl

    # ---------------- 연속 편집 (같은 층에 80번) ----------------
    n_seq = 80 if variant == "base" else 0  # aux 변형은 생략
    ss, sr, so_old, so_new = pick_edits(world, np.random.RandomState(seed + 30), max(n_seq, 1))
    # 같은 (주어, 관계)가 두 번 나오지 않게
    seen, keep = set(), []
    for i in range(n_seq):
        if (ss[i], sr[i]) not in seen:
            seen.add((ss[i], sr[i]))
            keep.append(i)
    ss, sr, so_old, so_new = ss[keep], sr[keep], so_old[keep], so_new[keep]
    w_orig = model.blocks[bl].proj.weight.clone()
    track = []
    edited = []
    for i in range(len(ss)):
        dW, info = mi.rome_edit(model, world, bl, int(ss[i]), int(sr[i]), int(so_new[i]), Cs[20000][bl], np.random.RandomState(5000 + i), pos=bp, **VKW)
        model.blocks[bl].proj.weight.data += dW
        edited.append(i)
        if (i + 1) % 10 == 0 or i == 0:
            # 편집한 사실들의 유지율(top-1이 새 목적어) 과 편집하지 않은 사실의 정확도
            B = len(edited)
            ee = np.array(edited)
            tok, last, _, _ = world.build(ss[ee], sr[ee], np.zeros(B, int), np.full(B, 2), world.random_fills(np.random.RandomState(0), B))
            pe = mi.probs_for(model, tok, last).argmax(-1).numpy()
            kept = float((pe == so_new[ee]).mean())
            mask = np.ones((world.n_subj, world.NR), bool)
            mask[ss[ee], sr[ee]] = False
            gs_, gr_ = np.where(mask)
            sel = np.random.RandomState(1).choice(len(gs_), 600, replace=False)
            gs_, gr_ = gs_[sel], gr_[sel]
            tok, last, _, _ = world.build(gs_, gr_, np.zeros(600, int), np.full(600, 2), world.random_fills(np.random.RandomState(0), 600))
            pg = mi.probs_for(model, tok, last).argmax(-1).numpy()
            acc = float((pg == world.obj[gs_, gr_]).mean())
            track.append(dict(n_edits=i + 1, edited_retained=kept, unedited_acc=acc, dW_fro=info["dW_fro"]))
            log(f"  seq {i + 1}: edited retained {kept:.2f} unedited acc {acc:.3f}")
    model.blocks[bl].proj.weight.data.copy_(w_orig)
    out["sequential"] = track
    out["time_s"] = time.time() - t0
    (HERE / f"results{tag}_seed{seed}.json").write_text(json.dumps(out), encoding="utf-8")
    log(f"seed {seed} 끝 {out['time_s']:.0f}s")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        # 사용: run_experiments.py 0 1 2  (기본 변형)  또는  aux:0 aux:1 (보조 손실 변형)
        v, sd = arg.split(":") if ":" in arg else ("base", arg)
        main(int(sd), v)
