"""R2: Hase et al.(2023, "Does Localization Inform Editing?")의 사실별 분석을 GPT-2 small에서 축소 재현한다.

사실 ~100개 x 층 12개마다 (1) ROME 편집 성공(반사실 목표로 고쳤을 때)과 (2) 같은 층의 인과 추적 효과(gpt2_causal_trace.py 결과)를
짝지어 층별·사실별 상관, 층만으로 설명되는 편집 성공 대비 추적 효과가 더하는 설명량, 사실별 최적 층 선택을 비교한다.

ROME 구현은 공식 코드(compute_u/compute_v, hparams/ROME/gpt2-xl.json)를 따른다: 21개 문맥(없음 + 모델이 생성한 길이 5짜리 10개, 길이 10짜리 10개)의 평균 키 k*,
v* = Adam(lr 0.5, 최대 20단계, KL 0.0625, weight decay 0.5, 노름 상한 4배, 손실 < 0.05이면 중단), 비대칭 키(u = C^-1 k*, 분모는 문맥 없는 프롬프트의 키).
공식 코드와 다른 점: C는 위키백과 대신 pile-10k 텍스트 ~20만 토큰으로 추정, GPT-2 small용 hparams가 없어 XL 값을 그대로 씀(층만 바꿈).

사용:  python3 gpt2_localization_vs_edit.py [--n_facts 100] [--device mps|cpu] [--key_mode plain|mean]
필요: torch, transformers, numpy, scipy, matplotlib, pyarrow, huggingface_hub (openai-community/gpt2, NeelNanda/pile-10k가 로컬 캐시에 있어야 함)
선행: gpt2_causal_trace.py (gpt2_causal_trace_results.json)
출력: gpt2_localization_vs_edit_results.json, ../figures/own_gpt2_localization_vs_edit.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from gpt2_causal_trace import HERE, load_knowns, load_model
from minimal_impl import rank_one_update

SCRATCH = HERE.parent / ".scratch"
LAYERS = list(range(12))


# ----------------------------------------------------------------------------------------
# C = E[k k^T] 추정
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def estimate_C(tok, model, n_tokens, batch=16, max_len=512, seed=0):
    """모든 층 mlp.c_proj 입력(키, 3072차원)의 비중심 2차 모멘트. pile-10k 텍스트를 앞에서부터 읽어 모든 토큰 위치를 쓴다."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("NeelNanda/pile-10k", repo_type="dataset"))
    texts = pq.read_table(next(root.glob("data/*.parquet")), columns=["text"]).column("text").to_pylist()
    L, D = len(model.transformer.h), model.config.n_embd * 4
    acc = [torch.zeros(D, D, dtype=torch.float64) for _ in range(L)]
    cur = {}
    hs = [blk.mlp.c_proj.register_forward_pre_hook(lambda mod, a, l=l: cur.__setitem__(l, a[0]))
          for l, blk in enumerate(model.transformer.h)]
    n, i = 0, 0
    try:
        while n < n_tokens:
            chunk = texts[i:i + batch]
            i += batch
            tok.pad_token = tok.eos_token
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids, mask = enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device).bool()
            model(ids, attention_mask=mask.long(), logits_to_keep=1)
            for l in range(L):
                x = cur[l][mask]
                acc[l] += (x.T @ x).cpu().double()
            n += int(mask.sum())
    finally:
        for h in hs:
            h.remove()
    return [a / n for a in acc], n


def get_C(tok, model, n_tokens):
    path = SCRATCH / f"gpt2_C_pile10k_{n_tokens}.pt"
    if path.exists():
        d = torch.load(path)
        return [c.double() for c in d["C"]], d["n"]
    C, n = estimate_C(tok, model, n_tokens)
    SCRATCH.mkdir(exist_ok=True)
    torch.save(dict(C=[c.float() for c in C], n=n), path)
    return C, n


# ----------------------------------------------------------------------------------------
# ROME (GPT-2, HF Conv1D: c_proj.weight 모양 (3072, 768) = 수학적 W (768, 3072)의 전치)
# ----------------------------------------------------------------------------------------
def make_context_prefixes(tok, model, seed=0):
    """공식 get_context_templates: 모델이 <|endoftext|>에서 top-5 표집으로 만든 길이 5짜리 10개, 길이 10짜리 10개 + 문맥 없음."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    eos = tok.eos_token_id
    out = [""]
    for length, n in ((5, 10), (10, 10)):
        ids = torch.full((1, 1), eos, device=model.device)
        gen = model.generate(ids, do_sample=True, top_k=5, max_new_tokens=length, num_return_sequences=n,
                             pad_token_id=eos, attention_mask=torch.ones_like(ids))
        out += [tok.decode(g_[1:], skip_special_tokens=True).replace("\n", " ") + ". " for g_ in gen]
    return out


class Editor:
    def __init__(self, tok, model, C, prefixes, key_mode="plain", steps=20, lr=0.5, kl=0.0625, wd=0.5, clamp=4.0):
        self.tok, self.m, self.C, self.prefixes = tok, model, C, prefixes
        self.key_mode, self.steps, self.lr, self.kl, self.wd, self.clamp = key_mode, steps, lr, kl, wd, clamp
        self.dev = model.device

    def _row(self, prefix, template, subject, extra_ids=()):
        text = prefix + template.format(subject)
        start = len(prefix) + template.index("{}")
        enc = self.tok(text, return_offsets_mapping=True)
        idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < start + len(subject)]
        return enc["input_ids"] + list(extra_ids), idx[-1]

    def _batch(self, rows):
        T = max(len(r) for r in rows)
        ids = torch.full((len(rows), T), self.tok.eos_token_id)
        mask = torch.zeros(len(rows), T, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        return ids.to(self.dev), mask.to(self.dev)

    def compute_dW(self, layer, template, subject, target_ids):
        m = self.m
        rows, lk = zip(*[self._row(p, template, subject, target_ids[:-1]) for p in self.prefixes])
        kl_ids = self.tok(subject + " is a", return_offsets_mapping=True)
        kl_row, kl_lk = kl_ids["input_ids"], max(i for i, (a, b) in enumerate(kl_ids["offset_mapping"]) if a < len(subject))
        rows, lk = list(rows) + [kl_row], list(lk) + [kl_lk]
        n = len(self.prefixes)
        ids, mask = self._batch(rows)
        ar = torch.arange(len(rows), device=self.dev)
        lkt = torch.tensor(lk, device=self.dev)
        tgt = torch.full(ids.shape, -100, device=self.dev)
        for i in range(n):
            tgt[i, len(rows[i]) - len(target_ids):len(rows[i])] = torch.tensor(target_ids, device=self.dev)
        tmask = (tgt != -100).float()
        blk = m.transformer.h[layer].mlp
        st = {"delta": torch.zeros(m.config.n_embd, device=self.dev, requires_grad=True), "k": None, "out": None}

        def pre(mod, a):
            if st["k"] is None:
                st["k"] = a[0][ar, lkt].detach().clone()  # 모든 행(문맥)의 키

        def post(mod, a, out):
            if st["out"] is None:
                st["out"] = out[0, lk[0]].detach().clone()  # 문맥 없는 프롬프트의 원래 MLP 출력 (v 초기값)
            out = out.clone()
            out[ar, lkt] = out[ar, lkt] + st["delta"]
            return out

        hs = [blk.c_proj.register_forward_pre_hook(pre), blk.register_forward_hook(post)]
        opt = torch.optim.Adam([st["delta"]], lr=self.lr)
        kl0 = None
        info = {}
        try:
            for it in range(self.steps):
                opt.zero_grad()
                logits = m(ids, attention_mask=mask).logits.float()
                klp = torch.log_softmax(logits[n, kl_lk], -1)
                if kl0 is None:
                    kl0 = klp.detach().clone()
                lp = torch.log_softmax(logits, -1)
                g = torch.gather(lp, 2, torch.where(tgt != -100, tgt, 0).unsqueeze(2)).squeeze(2)
                nll = (-(g * tmask).sum(1) / len(target_ids))[:n].mean()
                kl = self.kl * (klp.exp() * (klp - kl0)).sum()
                reg = self.wd * st["delta"].norm() / st["out"].norm() ** 2
                loss = nll + kl + reg
                info = dict(loss=float(loss.detach()), nll=float(nll.detach()), steps=it + 1)
                if loss < 5e-2 or it == self.steps - 1:
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    mx = self.clamp * st["out"].norm()
                    if st["delta"].norm() > mx:
                        st["delta"].mul_(mx / st["delta"].norm())
        finally:
            for h in hs:
                h.remove()
        target = (st["out"] + st["delta"]).detach().cpu().double()
        k_mean, k_plain = st["k"][:n].mean(0).cpu().double(), st["k"][0].cpu().double()
        cur_out = st["out"].cpu().double()
        C = self.C[layer]
        W = blk.c_proj.weight.detach().T.cpu().double()
        b = blk.c_proj.bias.detach().cpu().double()
        if self.key_mode == "mean":
            dW = rank_one_update(W, b, k_mean, target, C)
        else:  # 공식 코드: u는 평균 키, 분모와 현재 출력은 문맥 없는 프롬프트
            u = torch.linalg.solve(C, k_mean)
            dW = torch.outer((target - cur_out) / (k_plain @ u), u)
        info.update(dW_fro=float(dW.norm()), delta_norm=float(st["delta"].norm()), v_norm=float(cur_out.norm()))
        return dW.T.float().to(self.dev), info  # c_proj.weight 모양 (3072, 768)

    @torch.no_grad()
    def seq_probs(self, template, subject, target_ids, prefixes=("",)):
        """목표 문자열 전체의 확률 P[o | prefix + prompt] (교사 강제, 토큰 확률의 곱), prefix마다 하나."""
        rows = [self._row(p, template, subject, target_ids)[0] for p in prefixes]
        ids, mask = self._batch(rows)
        lp = torch.log_softmax(self.m(ids, attention_mask=mask).logits.float(), -1)
        out = []
        for i, r in enumerate(rows):
            T0 = len(r) - len(target_ids)
            out.append(float(sum(lp[i, T0 - 1 + j, t] for j, t in enumerate(target_ids)).exp()))
        return np.array(out)

    def seq_prob(self, template, subject, target_ids):
        return float(self.seq_probs(template, subject, target_ids)[0])


def pick_targets(knowns, facts, tok, seed=0):
    """반사실 목표: 같은 relation_id의 다른 사실의 속성 중 정답과 첫 토큰이 다른 것(공개 CounterFact가 같은 관계에서 새 목표를 고르는 방식을 흉내).
    known_1000에는 반사실 목표가 없으므로 이렇게 만든다."""
    rng = np.random.RandomState(seed)
    out = {}
    for f in facts:
        t0 = tok(" " + f["attribute"])["input_ids"][0]
        cands = sorted({k["attribute"] for k in knowns if k["relation_id"] == f["relation_id"] and k["attribute"] != f["attribute"]
                        and tok(" " + k["attribute"])["input_ids"][0] != t0})
        out[f["known_id"]] = cands[rng.randint(len(cands))]
    return out


# ----------------------------------------------------------------------------------------
# 분석
# ----------------------------------------------------------------------------------------
def r2(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return 1 - res.var() / y.var()


def analyze(E, X_by, n_boot=2000, seed=0):
    """E (N, L) 편집 성공, X_by: 이름 -> (N, L) 추적 효과. 사실 단위 부트스트랩."""
    rng = np.random.RandomState(seed)
    N, L = E.shape
    boots = rng.randint(0, N, (n_boot, N))
    out = {}
    layer_onehot = np.tile(np.eye(L), (N, 1))  # (N*L, L), 사실 우선 순서

    def stats(idx, X):
        e, x = E[idx], X[idx]
        res = {}
        res["per_layer_rho"] = [spearmanr(x[:, l], e[:, l])[0] if x[:, l].std() > 0 and e[:, l].std() > 0 else np.nan for l in range(L)]
        res["pooled_rho"] = spearmanr(x.ravel(), e.ravel())[0]
        ey, xy = e.ravel(), x.ravel()
        oh = np.tile(np.eye(L), (len(idx), 1))
        r_layer, r_trace, r_both = r2(ey, oh), r2(ey, xy[:, None]), r2(ey, np.column_stack([oh, xy]))
        res.update(r2_layer=r_layer, r2_trace=r_trace, r2_both=r_both, r2_gain=r_both - r_layer)
        # 사실 안에서 층 프로파일의 순위 상관(추적이 층 순서를 맞히는가)
        w = [spearmanr(x[i], e[i])[0] for i in range(len(idx)) if x[i].std() > 0 and e[i].std() > 0]
        res["within_fact_rho"] = float(np.mean(w))
        # 층 선택: 사실별 추적 최대 층 / 전체 평균이 가장 좋은 고정 층 / 사실별 최적(oracle) / 무작위 층
        tl = x.argmax(1)
        res["succ_trace_layer"] = float(e[np.arange(len(idx)), tl].mean())
        res["succ_fixed_layer"] = float(e.mean(0).max())  # 같은 표본으로 고른 최선의 고정 층(낙관적)
        res["succ_oracle"] = float(e.max(1).mean())
        res["succ_random"] = float(e.mean())
        res["argmax_rho"] = spearmanr(tl, e.argmax(1))[0]
        res["trace_layer_is_best"] = float((e[np.arange(len(idx)), tl] >= e.max(1) - 1e-9).mean())
        return res

    for name, X in X_by.items():
        point = stats(np.arange(N), X)
        bs = [stats(b, X) for b in boots[:500]] if n_boot else []  # 무거워서 부트스트랩 500회
        ci = {}
        for k, v in point.items():
            arr = np.array([b[k] for b in bs], dtype=float)
            ci[k] = (np.nanpercentile(arr, [2.5, 97.5], axis=0)).tolist()
        out[name] = dict(point=point, ci95=ci)
    return out


def plot(E, X_by, an, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "AppleSDGothicNeo" in f or "NotoSansCJK" in f:
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    N, L = E.shape
    x = np.arange(L)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axs[0]
    se = E.std(0) / np.sqrt(N)
    ax.errorbar(x, E.mean(0), 1.96 * se, color="C3", marker="o", ms=4, label=f"편집 성공 ({label})")
    ax.set_ylabel(f"편집 성공 ({label})", color="C3"); ax.set_xlabel("층"); ax.set_title("층별 평균: 편집 성공과 추적 효과")
    ax2 = ax.twinx()
    for (nm, X), c in zip(X_by.items(), ("C0", "C2")):
        ax2.plot(x, X.mean(0), color=c, marker="s", ms=4, label=f"추적: {nm}")
    ax2.set_ylabel("추적 효과 (IE, 주어 마지막 토큰)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    ax = axs[1]
    for k, ((nm, _), c) in enumerate(zip(X_by.items(), ("C0", "C2"))):
        rho = np.array(an[nm]["point"]["per_layer_rho"], dtype=float)
        ci = np.array(an[nm]["ci95"]["per_layer_rho"], dtype=float)
        ax.errorbar(x + 0.12 * (k - 0.5), rho, [rho - ci[0], ci[1] - rho], fmt="o", color=c, ms=4, capsize=2,
                    label=f"{nm}  (전체 쌍 ρ = {an[nm]['point']['pooled_rho']:.2f})")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("층"); ax.set_ylabel("사실 간 Spearman ρ (95% 부트스트랩)")
    ax.set_title("층별: 추적 효과 대 편집 성공 (사실 간)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[2]
    nm = list(X_by)[0]
    p, ci = an[nm]["point"], an[nm]["ci95"]
    keys = [("succ_random", "무작위 층"), ("succ_trace_layer", "추적 최대 층\n(사실별)"), ("succ_fixed_layer", "최선의 고정 층"), ("succ_oracle", "사실별 최적 층\n(oracle)")]
    vals = [p[k] for k, _ in keys]
    lo = [p[k] - ci[k][0] for k, _ in keys]; hi = [ci[k][1] - p[k] for k, _ in keys]
    ax.bar(range(4), vals, yerr=[lo, hi], color=["0.7", "C0", "C1", "C4"], capsize=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in keys], fontsize=8); ax.set_ylim(0.9, 1.0)
    ax.set_ylabel(f"평균 편집 성공 ({label})"); ax.set_title(f"층 고르는 방법별 편집 성공 (추적: {nm})"); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"GPT-2 small, 사실 {N}개 x 층 {L}개: 인과 추적 효과가 편집 성공을 예측하는가")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_facts", type=int, default=100)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--key_mode", default="plain", choices=["plain", "mean"])
    ap.add_argument("--c_tokens", type=int, default=200000)
    ap.add_argument("--trace", default=str(HERE / "gpt2_causal_trace_results.json"))
    ap.add_argument("--out", default=str(HERE / "gpt2_localization_vs_edit_results.json"))
    ap.add_argument("--fig", default=str(HERE.parent / "figures" / "own_gpt2_localization_vs_edit.png"))
    ap.add_argument("--analyze_only", action="store_true", help="편집 결과(또는 .shard* 파일들)를 읽어 분석만 한다")
    ap.add_argument("--shard", default="", help="k/n: 사실을 n개로 나눈 k번째만 편집하고 <out>.shard<k>에 저장(분석은 --analyze_only로)")
    args = ap.parse_args()

    trace = json.load(open(args.trace))
    knowns = load_knowns()
    by_id = {k["known_id"]: k for k in knowns}
    rng = np.random.RandomState(0)
    sel = sorted(rng.choice(len(trace["facts"]), min(args.n_facts, len(trace["facts"])), replace=False))
    tfacts = [trace["facts"][i] for i in sel]
    facts = [by_id[t["known_id"]] for t in tfacts]

    if args.analyze_only:
        shards = sorted(Path(args.out).parent.glob(Path(args.out).name + ".shard*"))
        if shards:
            pool = {e["known_id"]: e for sh in shards for e in json.load(open(sh))["edits"]}
            edits = [pool[f["known_id"]] for f in facts]
            cfg = json.load(open(shards[0]))["config"]
            json.dump(dict(config=cfg, edits=edits), open(args.out, "w"), ensure_ascii=False)
        else:
            edits = json.load(open(args.out))["edits"]
    else:
        if args.shard:
            k_, n_ = map(int, args.shard.split("/"))
            facts = facts[k_::n_]
            args.out += f".shard{k_}"
        tok, model = load_model(args.device)
        t0 = time.time()
        C, n_tok = get_C(tok, model, args.c_tokens)
        print(f"C: 토큰 {n_tok}개, 층 {len(C)}개, {time.time() - t0:.0f}s", flush=True)
        prefixes = make_context_prefixes(tok, model)
        ed = Editor(tok, model, C, prefixes, key_mode=args.key_mode)
        targets = pick_targets(knowns, facts, tok)
        held_out = make_context_prefixes(tok, model, seed=1)[1:]  # 편집에 쓰지 않은 새 문맥 20개(일반화 시험)
        edits = []
        for n, f in enumerate(facts):
            tn = tok(" " + targets[f["known_id"]])["input_ids"]
            to = tok(" " + f["attribute"])["input_ids"]
            pn0 = ed.seq_prob(f["template"], f["subject"], tn)
            po0 = ed.seq_prob(f["template"], f["subject"], to)
            gn0 = ed.seq_probs(f["template"], f["subject"], tn, held_out)
            rec = dict(known_id=f["known_id"], subject=f["subject"], template=f["template"], target_true=f["attribute"],
                       target_new=targets[f["known_id"]], p_new_before=pn0, p_true_before=po0, layers=[])
            for l in LAYERS:
                dW, info = ed.compute_dW(l, f["template"], f["subject"], tn)
                w = model.transformer.h[l].mlp.c_proj.weight
                old = w.data.clone()
                w.data += dW
                try:
                    pn, po = ed.seq_prob(f["template"], f["subject"], tn), ed.seq_prob(f["template"], f["subject"], to)
                    gn = ed.seq_probs(f["template"], f["subject"], tn, held_out)
                    go = ed.seq_probs(f["template"], f["subject"], to, held_out)
                finally:
                    w.data.copy_(old)
                rec["layers"].append(dict(layer=l, p_new=pn, p_true=po, es=float(pn > po), rs=(pn - pn0) / (1 - pn0),
                                          gen_p=float(gn.mean()), gen_es=float((gn > go).mean()),
                                          gen_rs=float(np.mean((gn - gn0) / (1 - gn0))), **info))
            edits.append(rec)
            if n % 5 == 0:
                print(f"  fact {n + 1}/{len(facts)} {time.time() - t0:.0f}s  ES {np.mean([[x['es'] for x in e['layers']] for e in edits], 0).round(2).tolist()}", flush=True)
        json.dump(dict(config=dict(model="openai-community/gpt2", n_facts=len(facts), layers=LAYERS, key_mode=args.key_mode, c_tokens=n_tok,
                                   c_source="NeelNanda/pile-10k", device=args.device, steps=20, lr=0.5, kl_factor=0.0625, v_weight_decay=0.5,
                                   clamp_norm_factor=4.0, n_contexts=len(prefixes), prefixes=prefixes, held_out_prefixes=held_out,
                                   target_rule="same relation_id, different attribute, different first token (seed 0)",
                                   fact_sample="seed-0 random sample of the correct facts in gpt2_causal_trace_results.json"),
                       edits=edits), open(args.out, "w"), ensure_ascii=False)
        if args.shard:
            return

    NL = 11  # 분석은 층 0-10. 층 11의 MLP 출력은 주어 위치에서 뒤 토큰이 읽을 수 없어 구조적으로 편집이 안 먹힌다(별도 보고).
    M = {k: np.array([[x[k] for x in e["layers"]] for e in edits]) for k in ("rs", "es", "p_new", "gen_rs", "gen_es", "gen_p")}
    ate = np.array([t["p_clean"] - t["p_corrupt"] for t in tfacts])
    ie = lambda k: np.array([t["ie_subj_last"][k] for t in tfacts])
    X_hidden, X_mlp, X_mlp3 = ie("hidden"), ie("mlp_w1"), ie("mlp_w3")
    X_by = {"hidden": X_hidden[:, :NL], "MLP 창 1": X_mlp[:, :NL]}
    res = dict(edits=edits)
    res["analysis"] = {k: analyze(M[k][:, :NL], X_by) for k in ("gen_rs", "rs", "gen_es", "es", "gen_p")}
    res["analysis"]["gen_rs_mlp3"] = analyze(M["gen_rs"][:, :NL], {"MLP 창 3": X_mlp3[:, :NL]})
    res["analysis"]["gen_rs_hidden_normalized"] = analyze(M["gen_rs"][:, :NL], {"hidden / ATE": (X_hidden / np.maximum(ate[:, None], 1e-3))[:, :NL]})
    res["mean_by_layer"] = {k: M[k].mean(0).tolist() for k in M}
    res["mean_by_layer"].update(hidden=X_hidden.mean(0).tolist(), mlp_w1=X_mlp.mean(0).tolist(), mlp_w3=X_mlp3.mean(0).tolist())
    res["ate_mean"] = float(ate.mean())
    old = json.load(open(args.out)) if Path(args.out).exists() else {}
    old.update(res)
    json.dump(old, open(args.out, "w"), ensure_ascii=False)
    plot(M["gen_rs"][:, :NL], X_by, res["analysis"]["gen_rs"], args.fig, "일반화 rewrite score")
    for k in ("gen_rs", "rs", "gen_es", "es"):
        for nm, a in res["analysis"][k].items():
            p, c = a["point"], a["ci95"]
            print(f"[{k}/{nm}] 전체 쌍 ρ {p['pooled_rho']:.3f} {np.round(c['pooled_rho'], 3).tolist()}; R2 층 {p['r2_layer']:.3f} 추적 {p['r2_trace']:.3f} 둘 다 {p['r2_both']:.3f} "
                  f"(증가 {p['r2_gain']:.3f} {np.round(c['r2_gain'], 3).tolist()}); 사실 내 ρ {p['within_fact_rho']:.3f}")
            print(f"    층별 ρ {np.round(p['per_layer_rho'], 2).tolist()}")
            print(f"    성공: 추적 최대 층 {p['succ_trace_layer']:.3f}, 최선 고정 층 {p['succ_fixed_layer']:.3f}, oracle {p['succ_oracle']:.3f}, 무작위 {p['succ_random']:.3f}")
    for k in ("rs", "es", "gen_rs", "gen_es", "gen_p"):
        print(f"층별 평균 {k}", np.round(M[k].mean(0), 3).tolist())


if __name__ == "__main__":
    main()
