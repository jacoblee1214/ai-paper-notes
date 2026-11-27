"""R1: GPT-2 small에서 인과 추적(Meng et al. 2022, Sec. 2, 부록 B)을 층 x 위치 지도로 구한다.

논문 Fig. 2(GPT-2 XL)와 같은 절차: 주어 토큰 임베딩(wte 출력)에 N(0, (3*sigma)^2) 잡음을 더해 만든 '잡음 실행'에서,
상태 하나(또는 MLP/attention 출력의 층 창)만 깨끗한 실행의 값으로 되돌리고 정답 확률의 회복량(IE)을 잰다.
정답 = 깨끗한 실행의 top-1 토큰, 잡음은 사실마다 10개(공식 코드 samples=10), sigma = 데이터셋 전체 주어들의 wte 출력 원소 std.

창 위치: 논문 부록 B.2는 [l-4, ..., l+5]라고 쓰지만 공식 코드(trace_important_window)는 [l-5, ..., l+4]다(04의 D7).
기본은 코드 방식이고, MLP 창 10에 대해서만 논문 방식도 함께 구해 차이를 본다(kind 'mlp_w10_paper').

사용:  python3 gpt2_causal_trace.py [--n_facts N] [--device mps|cpu]
필요: torch, transformers(openai-community/gpt2 가중치가 로컬 캐시에 있어야 함), numpy, matplotlib
데이터: ../data/known_1000.json (공식 ROME knowns, `curl -L https://rome.baulab.info/data/dsets/known_1000.json -o ../data/known_1000.json`)
출력: gpt2_causal_trace_results.json, ../figures/own_gpt2_trace.png
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent
MODEL_NAME = "openai-community/gpt2"
BUCKETS = ["prior", "subj_first", "subj_mid", "subj_last", "further", "last"]
KINDS = ["hidden", "mlp_w1", "mlp_w3", "mlp_w10", "mlp_w10_paper", "attn_w1", "attn_w3", "attn_w10"]


def load_model(device):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def load_knowns(path=None):
    return json.load(open(path or HERE.parent / "data" / "known_1000.json"))


def subject_range(tok, template, subject):
    """프롬프트 토큰 중 주어 문자 구간과 겹치는 토큰 범위 [b, e)와 토큰 id."""
    prompt = template.format(subject)
    start = template.index("{}")
    end = start + len(subject)
    enc = tok(prompt, return_offsets_mapping=True)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end]
    return enc["input_ids"], idx[0], idx[-1] + 1


def embedding_std(tok, model, subjects):
    """공식 collect_embedding_std: 주어 문자열 전체의 wte 출력 원소 std (주어마다 토큰화해 이어 붙임)."""
    outs = []
    with torch.no_grad():
        for s in subjects:
            ids = torch.tensor([tok(s)["input_ids"]], device=model.device)
            outs.append(model.transformer.wte(ids)[0].flatten())
    return torch.cat(outs).std().item()


def window(l, w, L, paper=False):
    """코드: [l - w//2, ..., l - (-w//2) - 1]  (w=10이면 [l-5, l+4]). 논문 방식은 한 칸 뒤로(w=10이면 [l-4, l+5])."""
    lo = l - w // 2 + (1 if paper else 0)
    hi = l - (-w // 2) + (1 if paper else 0)
    return [x for x in range(lo, hi) if 0 <= x < L]


class Tracer:
    """한 프롬프트에 대한 인과 추적. 모든 위치 x 잡음 표본을 배치해 층(창)마다 한 번에 돌린다."""

    def __init__(self, model, n_noise=10):
        self.m = model
        self.L = len(model.transformer.h)
        self.n_noise = n_noise

    @torch.no_grad()
    def clean_run(self, ids):
        """깨끗한 실행: 정답 토큰, 정답 확률, 층별 hidden/mlp/attn 출력."""
        m, rec = self.m, {"hidden": {}, "mlp": {}, "attn": {}}
        hs = []
        for l, blk in enumerate(m.transformer.h):
            hs.append(blk.register_forward_hook(lambda mod, i, o, l=l: rec["hidden"].__setitem__(l, o[0].detach().clone())))
            hs.append(blk.mlp.register_forward_hook(lambda mod, i, o, l=l: rec["mlp"].__setitem__(l, o[0].detach().clone())))
            hs.append(blk.attn.register_forward_hook(lambda mod, i, o, l=l: rec["attn"].__setitem__(l, o[0][0].detach().clone())))
        try:
            logits = m(ids[None], logits_to_keep=1).logits[0, -1]
        finally:
            for h in hs:
                h.remove()
        p = torch.softmax(logits.float(), -1)
        ans = int(p.argmax())
        return ans, float(p[ans]), rec

    @torch.no_grad()
    def patched(self, ids, b, e, noise, answer, rec, kind, layers, positions):
        """kind in {'hidden','mlp','attn'}. positions 각각에 대해 layers의 해당 상태를 깨끗한 값으로 되돌린 실행의 정답 확률.
        반환 (len(positions), n_noise). positions=[]이면 되돌림 없는 잡음 실행(1 x n_noise)."""
        m, S = self.m, self.n_noise
        P = max(len(positions), 1)
        x = ids[None].repeat(P * S, 1)
        nz = noise.repeat(P, 1, 1)  # (P*S, e-b, d): 위치마다 같은 잡음 S개
        pos_idx = torch.tensor(positions if positions else [0], device=x.device).repeat_interleave(S)
        row = torch.arange(P * S, device=x.device)
        hs = [m.transformer.wte.register_forward_hook(lambda mod, i, o: torch.cat([o[:, :b], o[:, b:e] + nz, o[:, e:]], 1))]
        if positions:
            for l in layers:
                src = rec[kind][l][pos_idx]  # (P*S, d)

                def fix(t, src=src):
                    t = t.clone()
                    t[row, pos_idx] = src
                    return t
                blk = m.transformer.h[l]
                if kind == "hidden":
                    hs.append(blk.register_forward_hook(lambda mod, i, o, fix=fix: fix(o)))
                elif kind == "mlp":
                    hs.append(blk.mlp.register_forward_hook(lambda mod, i, o, fix=fix: fix(o)))
                else:
                    hs.append(blk.attn.register_forward_hook(lambda mod, i, o, fix=fix: (fix(o[0]),) + tuple(o[1:])))
        try:
            logits = m(x, logits_to_keep=1).logits[:, -1]
        finally:
            for h in hs:
                h.remove()
        p = torch.softmax(logits.float(), -1)[:, answer]
        return p.view(P, S).cpu().numpy()

    def trace(self, ids, b, e, sigma_noise, seed):
        T = len(ids)
        g = torch.Generator().manual_seed(seed)
        noise = (torch.randn(self.n_noise, e - b, self.m.config.n_embd, generator=g) * sigma_noise).to(ids.device)
        ans, p_clean, rec = self.clean_run(ids)
        p_corr = float(self.patched(ids, b, e, noise, ans, rec, "hidden", [], [])[0].mean())
        pos = list(range(T))
        out = {k: np.zeros((self.L, T)) for k in KINDS}
        for l in range(self.L):
            out["hidden"][l] = self.patched(ids, b, e, noise, ans, rec, "hidden", [l], pos).mean(1) - p_corr
            for kind in ("mlp", "attn"):
                for w in (1, 3, 10):
                    out[f"{kind}_w{w}"][l] = self.patched(ids, b, e, noise, ans, rec, kind, window(l, w, self.L), pos).mean(1) - p_corr
            out["mlp_w10_paper"][l] = self.patched(ids, b, e, noise, ans, rec, "mlp", window(l, 10, self.L, paper=True), pos).mean(1) - p_corr
        return dict(answer=ans, p_clean=p_clean, p_corrupt=p_corr, ie=out)


def bucketize(ie, b, e, T):
    """(L, T) -> (L, 6): 주어 앞 / 주어 첫 / 주어 중간 / 주어 마지막 / 주어 뒤 / 마지막 토큰. 해당 토큰이 없으면 NaN.
    주어가 한 토큰이면 그 토큰은 '주어 마지막'으로만 센다."""
    L = ie.shape[0]
    out = np.full((L, 6), np.nan)
    if b > 0:
        out[:, 0] = ie[:, :b].mean(1)
    if e - b > 1:
        out[:, 1] = ie[:, b]
    if e - b > 2:
        out[:, 2] = ie[:, b + 1:e - 1].mean(1)
    out[:, 3] = ie[:, e - 1]
    if e < T - 1:
        out[:, 4] = ie[:, e:T - 1].mean(1)
    out[:, 5] = ie[:, T - 1]
    return out


def select_correct(tok, model, knowns):
    """GPT-2 small의 top-1 다음 토큰이 정답 속성의 첫 토큰(앞 공백 포함)인 사실만 남긴다."""
    keep = []
    with torch.no_grad():
        for k in knowns:
            ids = torch.tensor([tok(k["template"].format(k["subject"]))["input_ids"]], device=model.device)
            top = int(model(ids, logits_to_keep=1).logits[0, -1].argmax())
            if top == tok(" " + k["attribute"])["input_ids"][0]:
                keep.append(k)
    return keep


def boot_ci(x, n=2000, seed=0):
    """사실에 대한 부트스트랩 평균의 95% 구간. x (N, ...)."""
    rng = np.random.RandomState(seed)
    N = len(x)
    idx = rng.randint(0, N, (n, N))
    means = np.nanmean(x[idx], axis=1)
    return np.nanpercentile(means, 2.5, axis=0), np.nanpercentile(means, 97.5, axis=0)


def summarize(per_fact):
    """per_fact: kind -> (N, L, 6) 버킷 배열."""
    res = {}
    for k, a in per_fact.items():
        mean = np.nanmean(a, axis=0)
        lo, hi = boot_ci(a)
        s = dict(mean=mean.tolist(), lo=lo.tolist(), hi=hi.tolist())
        l, j = np.unravel_index(np.nanargmax(mean), mean.shape)
        s["peak"] = dict(layer=int(l), bucket=BUCKETS[j], value=float(mean[l, j]), lo=float(lo[l, j]), hi=float(hi[l, j]))
        s["subj_last_peak_layer"] = int(np.argmax(mean[:, 3]))
        s["subj_last_peak"] = float(mean[:, 3].max())
        s["last_peak_layer"] = int(np.argmax(mean[:, 5]))
        s["last_peak"] = float(mean[:, 5].max())
        s["per_fact_max_over_cells_mean"] = float(np.nanmax(a.reshape(len(a), -1), axis=1).mean())
        res[k] = s
    return res


def plot(summary, ates, n, path):
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
    names = ["주어 앞", "주어 첫", "주어 중간", "주어 마지막", "주어 뒤", "마지막 토큰"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (k, title) in zip(axs[0], [("hidden", "hidden state 복원"), ("mlp_w3", "MLP 복원 (층 창 3)"), ("attn_w3", "attention 복원 (층 창 3)")]):
        g = np.array(summary[k]["mean"])
        im = ax.imshow(g.T, aspect="auto", origin="upper", cmap="Purples", vmin=0)
        ax.set_yticks(range(6)); ax.set_yticklabels(names)
        ax.set_xlabel("층"); ax.set_title(f"{title}  (최대 {np.nanmax(g):.3f})")
        for j in range(6):
            for l in range(g.shape[0]):
                if not np.isnan(g[l, j]) and g[l, j] > 0.5 * np.nanmax(g):
                    ax.text(l, j, f"{g[l, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
        plt.colorbar(im, ax=ax, fraction=0.046)
    x = np.arange(len(summary["hidden"]["mean"]))

    def line(ax, k, j, label, **kw):
        m = np.array(summary[k]["mean"])[:, j]
        lo = np.array(summary[k]["lo"])[:, j]
        hi = np.array(summary[k]["hi"])[:, j]
        (ln,) = ax.plot(x, m, label=label, **kw)
        ax.fill_between(x, lo, hi, alpha=0.15, color=ln.get_color())
    ax = axs[1, 0]
    line(ax, "hidden", 3, "주어 마지막 토큰"); line(ax, "hidden", 5, "마지막 토큰"); line(ax, "hidden", 1, "주어 첫 토큰")
    ax.set_title("hidden state AIE (95% 구간)")
    ax = axs[1, 1]
    for w in (1, 3, 10):
        line(ax, f"mlp_w{w}", 3, f"MLP 창 {w}, 주어 마지막")
    line(ax, "mlp_w3", 5, "MLP 창 3, 마지막 토큰", ls="--")
    ax.set_title("MLP AIE")
    ax = axs[1, 2]
    for w in (1, 3, 10):
        line(ax, f"attn_w{w}", 5, f"attention 창 {w}, 마지막 토큰")
    line(ax, "attn_w3", 3, "attention 창 3, 주어 마지막", ls="--")
    ax.set_title("attention AIE")
    for ax in axs[1]:
        ax.set_xlabel("층"); ax.set_ylabel("AIE (정답 확률 회복)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"GPT-2 small 인과 추적 (사실 {n}개, 잡음 3σ, ATE {np.mean(ates):.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_facts", type=int, default=0, help="0이면 맞힌 사실 전부")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--noise_mult", type=float, default=3.0)
    ap.add_argument("--out", default=str(HERE / "gpt2_causal_trace_results.json"))
    ap.add_argument("--fig", default=str(HERE.parent / "figures" / "own_gpt2_trace.png"))
    args = ap.parse_args()

    tok, model = load_model(args.device)
    knowns = load_knowns()
    sigma = embedding_std(tok, model, [k["subject"] for k in knowns])
    correct = select_correct(tok, model, knowns)
    print(f"knowns {len(knowns)}, GPT-2 small이 맞힌 사실 {len(correct)}, sigma {sigma:.4f}, 잡음 {args.noise_mult * sigma:.4f}")
    facts = correct[:args.n_facts] if args.n_facts else correct
    tr = Tracer(model)
    per_fact = {k: [] for k in KINDS}
    meta, ates = [], []
    t0 = time.time()
    for n, k in enumerate(facts):
        ids, b, e = subject_range(tok, k["template"], k["subject"])
        ids_t = torch.tensor(ids, device=model.device)
        r = tr.trace(ids_t, b, e, args.noise_mult * sigma, seed=1)
        T = len(ids)
        # 정의에서 따라 나오는 검산: 마지막 층 마지막 토큰의 hidden을 되돌리면 IE = TE
        assert abs(r["ie"]["hidden"][-1, -1] - (r["p_clean"] - r["p_corrupt"])) < 1e-4
        for key in KINDS:
            per_fact[key].append(bucketize(r["ie"][key], b, e, T))
        ates.append(r["p_clean"] - r["p_corrupt"])
        meta.append(dict(known_id=k["known_id"], subject=k["subject"], n_tokens=T, subj_range=[b, e], answer=r["answer"],
                         p_clean=r["p_clean"], p_corrupt=r["p_corrupt"],
                         ie_subj_last={q: r["ie"][q][:, e - 1].tolist() for q in ("hidden", "mlp_w1", "mlp_w3")}))
        if n % 20 == 0:
            print(f"  {n + 1}/{len(facts)}  {time.time() - t0:.0f}s", flush=True)
    per_fact = {k: np.stack(v) for k, v in per_fact.items()}
    summary = summarize(per_fact)
    ates = np.array(ates)
    lo, hi = boot_ci(ates[:, None])
    res = dict(config=dict(model=MODEL_NAME, n_knowns=len(knowns), n_correct=len(correct), n_facts=len(facts), noise_mult=args.noise_mult,
                           sigma=sigma, n_noise=10, seed=1, device=args.device, buckets=BUCKETS, answer="clean top-1",
                           window_note="mlp/attn 창 w는 공식 코드식 [l-w//2, l-(-w//2)-1]; mlp_w10_paper는 논문 부록 B.2식(한 칸 뒤)"),
               ate=dict(mean=float(ates.mean()), lo=float(lo[0]), hi=float(hi[0])),
               p_clean_mean=float(np.mean([m["p_clean"] for m in meta])), p_corrupt_mean=float(np.mean([m["p_corrupt"] for m in meta])),
               summary=summary, facts=meta)
    json.dump(res, open(args.out, "w"), ensure_ascii=False)
    plot(summary, ates, len(facts), args.fig)
    print(f"ATE {ates.mean():.3f} [{lo[0]:.3f}, {hi[0]:.3f}]")
    for k in KINDS:
        s = summary[k]
        print(f"{k:14s} 최대 {s['peak']['value']:.3f} (층 {s['peak']['layer']}, {s['peak']['bucket']}); 주어 마지막 최대 {s['subj_last_peak']:.3f} (층 {s['subj_last_peak_layer']}); 마지막 토큰 최대 {s['last_peak']:.3f} (층 {s['last_peak_layer']})")


if __name__ == "__main__":
    main()
