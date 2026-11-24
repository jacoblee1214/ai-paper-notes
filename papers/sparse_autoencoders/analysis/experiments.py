"""실험 실행. 한 번에 하나씩 돌린다(CPU).

  python experiments.py a      # 정답 있는 시험대: 합성 superposition 데이터에서 alpha x R 격자 (results_a.json)
  python experiments.py a2     # 직교 정답에서 소프트 임계 이론(FVU, L0, 기울기)과 학습 결과 비교 (results_a2.json)
  python experiments.py b      # 29번 모델 activation에 SAE: FVU, L0, 죽은 특징, loss recovered (results_b.json)
  python experiments.py c      # 해석 가능성 대용 지표: SAE 대 기저·PCA·무작위, 랜덤 초기화 모델 대조 (results_c.json)
  python experiments.py d      # 29번 A1: 층 1 head 출력과 SAE 특징 (results_d.json)

b, c, d는 ../../induction_heads/artifacts/ (29번 산출물, 읽기만)가 있어야 한다. 없으면 29번 README의 재생성 명령을 먼저 실행.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import minimal_impl as M

HERE = Path(__file__).resolve().parent
ART = HERE.parent.parent / "induction_heads" / "artifacts"
N_TRAIN_SEQ = 384          # 앞 384개 창은 SAE 학습, 뒤 128개 창은 평가(손실, 해석 지표)
torch.set_num_threads(8)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- a: 정답 있는 시험대
def run_a(out=HERE / "results_a.json"):
    alphas = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    ratios = [1, 2, 4, 8]
    seeds = [0, 1, 2]
    steps = 6000
    settings = {"main": dict(n=64, d=32, density=0.05), "hard": dict(n=64, d=16, density=0.05)}
    runs = []
    t0 = time.time()
    for sname, st in settings.items():
        for seed in seeds:
            G = M.make_truth(st["n"], st["d"], seed)
            samp = M.synthetic_sampler(G, st["density"])
            Xe = samp(20000, torch.Generator().manual_seed(9000 + seed))
            for R in ratios:
                for a in alphas:
                    s = M.train_sae(samp, st["d"], R * st["d"], a, steps=steps, seed=seed)
                    e = M.evaluate(s, Xe)
                    D = s.dictionary()
                    slope, n_match = M.shrinkage_slope(s, G, st["density"])
                    runs.append(dict(setting=sname, seed=seed, R=R, alpha=a, tied=False, **e,
                                     mmcs_truth=float(M.mmcs(G, D).mean()), frac_rec=M.recovered_frac(G, D),
                                     mmcs_learned=float(M.mmcs(D, G).mean()), frac_learned_match=float((M.mmcs(D, G) > 0.9).float().mean()),
                                     slope=slope, n_match=n_match, slope_theory=M.shrinkage_theory(a)))
                    log(f"[a] {sname} seed{seed} R{R} a{a}: fvu {e['fvu']:.3f} l0 {e['l0']:.1f} dead {e['dead_frac']:.2f} "
                        f"rec {runs[-1]['frac_rec']:.2f} ({time.time() - t0:.0f}s)")
    # 묶은 가중치 대 별도 디코더, 밀도 0.15 (main, R=2, alpha 세 값)
    extra = []
    st = settings["main"]
    for seed in seeds:
        G = M.make_truth(st["n"], st["d"], seed)
        for tied in (False, True):
            for dens in (0.05, 0.15):
                samp = M.synthetic_sampler(G, dens)
                Xe = samp(20000, torch.Generator().manual_seed(9000 + seed))
                for a in (0.03, 0.1, 0.3):
                    s = M.train_sae(samp, st["d"], 2 * st["d"], a, tied=tied, steps=steps, seed=seed)
                    e = M.evaluate(s, Xe)
                    D = s.dictionary()
                    extra.append(dict(seed=seed, R=2, alpha=a, tied=tied, density=dens, **e, mmcs_truth=float(M.mmcs(G, D).mean()),
                                      frac_rec=M.recovered_frac(G, D)))
                    log(f"[a-extra] seed{seed} tied{tied} p{dens} a{a}: fvu {e['fvu']:.3f} l0 {e['l0']:.1f} rec {extra[-1]['frac_rec']:.2f}")
    json.dump(dict(settings=settings, alphas=alphas, ratios=ratios, seeds=seeds, steps=steps, lr=1e-3, batch=512, runs=runs, extra=extra),
              open(out, "w"), indent=1)


# --------------------------------------------------------------------------- a2: 직교 정답에서 수축 이론 검산
def soft_threshold_theory(alpha, p, n):
    """직교·단위 정답 방향, a ~ Bernoulli(p) * U[0,1]에서 최적 코드 c = max(a - t, 0), t = alpha/2 (식 (4)를 특징별로 분해).
    표본당 재구성 오차 = n p [t^3/3 + t^2 (1 - t)], 총분산 = n (p/3 - p^2/4), L0 = n p (1 - t) (알파 <= 2)."""
    t = alpha / 2
    err = n * p * (t ** 3 / 3 + t ** 2 * (1 - t))
    var = n * (p / 3 - p ** 2 / 4)
    return dict(fvu=err / var, l0=n * p * (1 - t), slope=M.shrinkage_theory(alpha))


def run_a2(out=HERE / "results_a2.json"):
    n = d = 16
    p = 0.1
    rows = []
    for alpha in (0.0, 0.03, 0.1, 0.3, 0.6, 1.0):
        for seed in (0, 1, 2):
            G = M.make_truth(n, d, seed, orthogonal=True)
            samp = M.synthetic_sampler(G, p)
            Xe = samp(50000, torch.Generator().manual_seed(9000 + seed))
            s = M.train_sae(samp, d, n, alpha, steps=6000, seed=seed)
            e = M.evaluate(s, Xe)
            slope, nm = M.shrinkage_slope(s, G, p)
            rows.append(dict(alpha=alpha, seed=seed, fvu=e["fvu"], l0=e["l0"], slope=slope, n_match=nm,
                             mmcs=float(M.mmcs(G, s.dictionary()).mean()), theory=soft_threshold_theory(alpha, p, n)))
            log(f"[a2] alpha {alpha} seed {seed}: fvu {e['fvu']:.4f} (th {rows[-1]['theory']['fvu']:.4f}) l0 {e['l0']:.2f} "
                f"(th {rows[-1]['theory']['l0']:.2f}) slope {slope:.3f} (th {rows[-1]['theory']['slope']:.3f})")
    json.dump(dict(n=n, d=d, density=p, rows=rows), open(out, "w"), indent=1)


# --------------------------------------------------------------------------- 29번 activation 불러오기
def load_site(name):
    """(512, 256, 128) float16 -> 위치 0(BOS)을 빼고 학습/평가로 나눈 float32 텐서와 정규화 상수."""
    a = torch.from_numpy(np.load(ART / f"{name}.npy").astype(np.float32))
    tr, ev = a[:N_TRAIN_SEQ, 1:].reshape(-1, a.shape[-1]), a[N_TRAIN_SEQ:, 1:].reshape(-1, a.shape[-1])
    scale = float((tr ** 2).sum(-1).mean().sqrt())     # 평균 ||x||^2 = 1이 되게 나눔
    return tr / scale, ev / scale, scale


def load_tokens():
    return torch.from_numpy(np.load(ART / "tokens.npy").astype(np.int64))


STEPS_B = 8000
ALPHAS_B = [0.01, 0.03, 0.1, 0.3]
ALPHA_C = 0.1              # c, d, 시드 일치에 쓰는 alpha (L0 약 45/512, FVU 약 0.2)


# --------------------------------------------------------------------------- b: 29번 모델의 SAE
def run_b(out=HERE / "results_b.json"):
    model = M.load_model(ART)
    toks = load_tokens()
    ev_t = toks[N_TRAIN_SEQ:]
    L_clean = M.lm_loss(model, ev_t)
    sites = ["resid_post_l0", "resid_post_l1", "mlp_out_l0"]
    res = dict(L_clean=L_clean, steps=STEPS_B, alphas=ALPHAS_B, runs=[], baselines={})
    t0 = time.time()
    for site in sites:
        tr, ev, scale = load_site(site)
        hook_name = site.replace("_l", "_")
        mean_vec = tr.mean(0) * scale
        L_zero = M.lm_loss(model, ev_t, {hook_name: M.splice_hook(lambda x: torch.zeros_like(x))})
        L_mean = M.lm_loss(model, ev_t, {hook_name: M.splice_hook(lambda x: mean_vec.expand_as(x).clone())})
        res["baselines"][site] = dict(L_zero=L_zero, L_mean=L_mean, scale=scale, n_train=tr.shape[0], n_eval=ev.shape[0])
        log(f"[b] {site}: clean {L_clean:.3f} zero-abl {L_zero:.3f} mean-abl {L_mean:.3f}")
        for R in (2, 4, 8):
            for a in ALPHAS_B:
                s = M.train_sae(tr, 128, 128 * R, a, steps=STEPS_B, seed=0)
                e = M.evaluate(s, ev)
                e_tr = M.evaluate(s, tr)
                L_rec = M.lm_loss(model, ev_t, {hook_name: M.splice_hook(M.sae_recon_fn(s, scale))})
                r = dict(site=site, R=R, alpha=a, **e, fvu_train=e_tr["fvu"], L_rec=L_rec,
                         rec_zero=M.loss_recovered(L_clean, L_rec, L_zero), rec_mean=M.loss_recovered(L_clean, L_rec, L_mean))
                res["runs"].append(r)
                log(f"[b] {site} R{R} a{a}: fvu {e['fvu']:.3f} (train {e_tr['fvu']:.3f}) l0 {e['l0']:.1f} dead {e['dead_frac']:.2f} "
                    f"L_rec {L_rec:.3f} rec(zero) {r['rec_zero']:.3f} rec(mean) {r['rec_mean']:.3f} ({time.time() - t0:.0f}s)")
    # 시드 간 일치: resid_post_l1, R=4, alpha=ALPHA_C, 시드 3개
    tr, ev, scale = load_site("resid_post_l1")
    ds = [M.train_sae(tr, 128, 512, ALPHA_C, steps=STEPS_B, seed=s).dictionary() for s in (0, 1, 2)]
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            m = M.mmcs(ds[i], ds[j])
            pairs.append(dict(pair=[i, j], mmcs=float(m.mean()), frac_gt_09=float((m > 0.9).float().mean())))
    res["seed_agreement"] = dict(site="resid_post_l1", R=4, alpha=ALPHA_C, pairs=pairs)
    log("[b] seed agreement", pairs)
    json.dump(res, open(out, "w"), indent=1)


# --------------------------------------------------------------------------- c: 해석 가능성 대용 지표
TOPK = 20


def _stats(acts, tok_ev):
    share, ent, tops = M.top_token_stats(acts, tok_ev, TOPK)
    return share, ent, tops, M.summarize_stats(share, ent)


def run_c(out=HERE / "results_c.json"):
    toks = load_tokens()
    vocab = json.load(open(ART / "vocab.json"))
    tok_ev = toks[N_TRAIN_SEQ:, 1:].reshape(-1)
    res = dict(topk=TOPK, sites={}, examples={})
    for site, alpha in (("resid_post_l1", ALPHA_C), ("resid_post_l0", ALPHA_C), ("mlp_out_l0", ALPHA_C)):
        tr, ev, scale = load_site(site)
        site_res = {}
        for name, kind in (("basis", "basis"), ("random_128", "random"), ("pca_2x128", "pca")):
            acts = M.baseline_acts(ev, kind, n=128) if kind != "pca" else M.baseline_acts(ev, "pca")
            site_res[name] = _stats(acts, tok_ev)[3]
        site_res["random_512"] = _stats(M.baseline_acts(ev, "random", n=512), tok_ev)[3]
        for R in (2, 4, 8):
            s = M.train_sae(tr, 128, 128 * R, alpha, steps=STEPS_B, seed=0)
            with torch.no_grad():
                acts = s.encode(ev)
            share, ent, tops, summ = _stats(acts, tok_ev)
            ev_stats = M.evaluate(s, ev)
            summ.update(fvu=ev_stats["fvu"], l0=ev_stats["l0"], dead_frac=ev_stats["dead_frac"])
            site_res[f"sae_R{R}"] = summ
            if site == "resid_post_l1" and R == 4:
                ok = np.nonzero(~np.isnan(share))[0]
                order = ok[np.argsort(-share[ok], kind="stable")]
                pick = list(order[:5]) + list(order[-5:])
                res["examples"]["resid_post_l1_R4"] = [
                    dict(feature=int(f), top1_share=float(share[f]), entropy=float(ent[f]),
                         top_tokens=[(vocab[t], c) for t, c in tops[f]], act_freq=float((acts[:, f] > 0).float().mean()))
                    for f in pick]
        res["sites"][site] = site_res
        log(f"[c] {site}", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in site_res.items()}))
    # 대조: 랜덤 초기화(학습 안 한) 모델의 resid_post_l1에서 SAE (같은 alpha, R=4)
    torch.manual_seed(123)
    rnd = M.TinyLM(M.load_model(ART).cfg).eval()
    with torch.no_grad():
        acts_r = torch.cat([rnd(toks[i:i + 64], cache=True)[1]["resid_post"][1][:, 1:].reshape(-1, 128) for i in range(0, 512, 64)])
    sc = float((acts_r[:N_TRAIN_SEQ * 255] ** 2).sum(-1).mean().sqrt())
    tr_r, ev_r = acts_r[:N_TRAIN_SEQ * 255] / sc, acts_r[N_TRAIN_SEQ * 255:] / sc
    ctrl = {}
    for name, kind in (("basis", "basis"), ("random_128", "random")):
        ctrl[name] = _stats(M.baseline_acts(ev_r, kind, n=128), tok_ev)[3]
    s = M.train_sae(tr_r, 128, 512, ALPHA_C, steps=STEPS_B, seed=0)
    with torch.no_grad():
        summ = _stats(s.encode(ev_r), tok_ev)[3]
    summ.update(M.evaluate(s, ev_r))
    ctrl["sae_R4"] = summ
    res["random_init_control"] = ctrl
    log("[c] random-init control", ctrl)
    json.dump(res, open(out, "w"), indent=1)


# --------------------------------------------------------------------------- d: 29번 A1
def run_d(out=HERE / "results_d.json"):
    model = M.load_model(ART)
    toks = load_tokens()
    L = 64
    tr, ev, scale = load_site("resid_post_l1")
    rep_tr = M.make_repeat_tokens(toks[:192], L)       # 학습 창에서 만든 반복 시퀀스(SAE-mix 학습용)
    rep_ev = M.make_repeat_tokens(toks[N_TRAIN_SEQ:], L)   # 평가 창에서 만든 반복 시퀀스(평가용)
    with torch.no_grad():
        x_rep_tr = torch.cat([model(rep_tr[i:i + 64], cache=True)[1]["resid_post"][1][:, 1:].reshape(-1, 128) for i in range(0, 192, 64)]) / scale
    res = dict(L=L, repeat_loss={}, saes={})
    # 반복 구간에서 모델이 정말 복사하는가: 복사본별 손실
    with torch.no_grad():
        lg = model(rep_ev)
        pl = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), rep_ev[:, 1:].reshape(-1), reduction="none").view(rep_ev.shape[0], -1)
    copy_t = ((torch.arange(1, rep_ev.shape[1]) - 1) // L)       # 대상 토큰 위치 1..T-1의 복사본 번호
    res["repeat_loss"] = {f"copy{c}": float(pl[:, copy_t == c].mean()) for c in range(int(copy_t.max()) + 1)}
    log("[d] per-copy loss", res["repeat_loss"])
    variants = {"nat": tr, "mix": torch.cat([tr, x_rep_tr])}
    for vname, data in variants.items():
        s = M.train_sae(data, 128, 512, ALPHA_C, steps=STEPS_B, seed=0)
        c, c_na, cp, mean_dir = M.head_dependence(model, s, scale, rep_ev, L, layer=1)
        D = s.dictionary()
        cos = D @ mean_dir.T                              # [F, H]
        first = c[cp == 0].mean(0)
        later = c[cp >= 1].mean(0)
        live = later > 1e-6
        sel = later / (first + 1e-3 * later.max().clamp(min=1e-9) + 1e-9)   # 반복 구간 / 첫 등장 평균 activation (0 나눔 방지용 작은 상수)
        drop = (c - c_na)[cp >= 1].mean(0)                 # 모든 층 1 head의 직접 출력을 빼면 반복 구간 activation이 평균 얼마나 줄어드나
        rel_drop = drop / (later + 1e-9)
        # 후보: 반복 구간 평균 activation이 첫 등장의 3배 이상이고 반복 구간에서 자주 켜지는 특징
        freq_rep = (c[cp >= 1] > 0).float().mean(0)
        cand = live & (sel >= 3) & (freq_rep >= 0.02)
        maxcos = cos.max(1).values
        from scipy.stats import spearmanr
        rho_drop = float(spearmanr(maxcos[live].numpy(), rel_drop[live].numpy())[0])
        rho_sel = float(spearmanr(maxcos[live].numpy(), torch.log(sel[live]).numpy())[0])
        # 무작위 방향의 코사인 분포(기준): 같은 차원의 단위 벡터 512개와 head 평균 방향의 최대 코사인
        g = torch.Generator().manual_seed(7)
        rnd_cos = (F.normalize(torch.randn(512, 128, generator=g), dim=1) @ mean_dir.T).max(1).values
        order = torch.argsort(-maxcos * live.float())[:8]
        top_cos = [dict(feature=int(f), maxcos=float(maxcos[f]), head=int(cos[f].argmax()), selectivity=float(sel[f]),
                        rel_drop=float(rel_drop[f]), freq_repeat=float(freq_rep[f])) for f in order]
        cand_idx = torch.nonzero(cand).flatten()
        cand_info = sorted([dict(feature=int(f), maxcos=float(maxcos[f]), head=int(cos[f].argmax()), selectivity=float(sel[f]),
                                 rel_drop=float(rel_drop[f]), freq_repeat=float(freq_rep[f])) for f in cand_idx], key=lambda r: -r["selectivity"])[:10]
        res["saes"][vname] = dict(
            n_train=int(data.shape[0]), n_live_in_repeat=int(live.sum()), n_candidates=int(cand.sum()),
            head_cos_between=[[float(v) for v in row] for row in (mean_dir @ mean_dir.T)],
            top_by_head_cos=top_cos, candidates=cand_info, spearman_maxcos_reldrop=rho_drop, spearman_maxcos_logsel=rho_sel,
            maxcos_dict_mean=float(maxcos.mean()), maxcos_dict_p95=float(torch.quantile(maxcos, 0.95)),
            maxcos_random_mean=float(rnd_cos.mean()), maxcos_random_p95=float(torch.quantile(rnd_cos, 0.95)),
            mean_rel_drop_all_live=float(rel_drop[live].mean()),
            mean_rel_drop_candidates=float(rel_drop[cand].mean()) if cand.any() else None,
            mean_maxcos_candidates=float(maxcos[cand].mean()) if cand.any() else None)
        log(f"[d] {vname}: live {int(live.sum())} cand {int(cand.sum())} rho_drop {rho_drop:.2f} rho_sel {rho_sel:.2f} "
            f"maxcos dict {float(maxcos.mean()):.2f} rnd {float(rnd_cos.mean()):.2f}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    {"a": run_a, "a2": run_a2, "b": run_b, "c": run_c, "d": run_d}[sys.argv[1]]()
