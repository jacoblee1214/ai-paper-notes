"""R1: 공개 사전학습 소형 모델(GPT-2 small, 비교용 Pythia)에서 induction head 점수, ICL 점수, head ablation 효과를 잰다.

TransformerLens 없이 transformers + PyTorch 훅만 쓴다.
  - attention 패턴: output_attentions=True (attn_implementation="eager")
  - head별 출력: 출력 투영(GPT-2 attn.c_proj, GPT-NeoX attention.dense)의 입력(head별 z)을 forward_pre_hook으로 잡고
    head 구간을 0(zero ablation) 또는 보정 데이터의 평균(mean ablation)으로 바꾼다.
점수 정의는 minimal_impl.py의 것을 그대로 쓴다(prefix_matching_score, copying logit 점수, OV 고유값 점수, icl_score 부호 규약).

입력: 로컬 캐시의 openai-community/gpt2, EleutherAI/pythia-{14m,70m,160m}, 데이터 NeelNanda/pile-10k (열 text).
실행: python gpt2_induction.py            -> gpt2_induction_results.json, ../figures/own_gpt2_*.png
      python gpt2_induction.py --models gpt2 --quick   (작은 표본으로 빠르게)
오프라인 캐시만 쓰려면 HF_HUB_OFFLINE=1.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from minimal_impl import pick_device, prefix_matching_score, previous_token_score

HERE = Path(__file__).resolve().parent
MODELS = {"gpt2": "openai-community/gpt2", "pythia-14m": "EleutherAI/pythia-14m",
          "pythia-70m": "EleutherAI/pythia-70m", "pythia-160m": "EleutherAI/pythia-160m"}


# --------------------------------------------------------------------------- 모델 어댑터
class Net:
    """GPT-2 또는 GPT-NeoX 계열 HF 모델을 head 단위로 다루는 얇은 래퍼."""

    def __init__(self, name, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.device = name, device
        repo = MODELS[name]
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, attn_implementation="eager", torch_dtype=torch.float32)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.gpt2 = name == "gpt2"
        self.L = cfg.n_layer if self.gpt2 else cfg.num_hidden_layers
        self.H = cfg.n_head if self.gpt2 else cfg.num_attention_heads
        self.d = cfg.n_embd if self.gpt2 else cfg.hidden_size
        self.dh = self.d // self.H
        if self.gpt2:
            self.blocks = list(self.model.transformer.h)
            self.oproj = [b.attn.c_proj for b in self.blocks]
            self.E = self.model.transformer.wte.weight
            self.lnf = self.model.transformer.ln_f.weight
        else:
            self.blocks = list(self.model.gpt_neox.layers)
            self.oproj = [b.attention.dense for b in self.blocks]
            self.E = self.model.gpt_neox.embed_in.weight
            self.lnf = self.model.gpt_neox.final_layer_norm.weight
        self.U = self.model.get_output_embeddings().weight          # (V, d)
        self.eot = self.tok.eos_token_id                              # 문서 시작 토큰(BOS 역할)
        self.state = {"capture": None, "ablate": {}}                  # ablate: {층: {head: (dh,) 벡터}}
        for l, m in enumerate(self.oproj):
            m.register_forward_pre_hook(self._hook(l))

    def _hook(self, l):
        def fn(mod, args):
            z = args[0]
            if self.state["capture"] is not None:
                self.state["capture"][l] = z.detach()
            ab = self.state["ablate"].get(l)
            if ab:
                z = z.clone().view(*z.shape[:2], self.H, self.dh)
                for h, v in ab.items():
                    z[:, :, h] = v
                z = z.view(*z.shape[:2], self.d)
                return (z,) + tuple(args[1:])
            return None
        return fn

    def W_O(self, l):
        """(H, dh, d): head별 출력 행렬."""
        w = self.oproj[l].weight
        w = w if self.gpt2 else w.T                                   # Conv1D는 (in,out), Linear는 (out,in)
        return w.view(self.H, self.dh, self.d)

    def W_V(self, l):
        """(H, d, dh): head별 값 행렬."""
        if self.gpt2:
            w = self.blocks[l].attn.c_attn.weight[:, 2 * self.d:]     # (d, d)
            return w.view(self.d, self.H, self.dh).permute(1, 0, 2)
        w = self.blocks[l].attention.query_key_value.weight           # (3d, d), head 순서로 [q,k,v] 인터리브
        return w.view(self.H, 3, self.dh, self.d)[:, 2].permute(0, 2, 1)

    def ln1(self, l):
        b = self.blocks[l]
        return (b.ln_1 if self.gpt2 else b.input_layernorm).weight

    @torch.no_grad()
    def run(self, toks, patterns=False, capture=False, ablate=None):
        self.state["capture"] = {} if capture else None
        self.state["ablate"] = ablate or {}
        out = self.model(toks, output_attentions=patterns)
        cap = self.state["capture"]
        self.state["capture"], self.state["ablate"] = None, {}
        return out.logits, (list(out.attentions) if patterns else None), cap


def per_pos_loss(logits, toks):
    """(B, T-1): 인덱스 i는 토큰 i+1을 맞히는 손실."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1),
                           reduction="none").view(toks.shape[0], -1)


# --------------------------------------------------------------------------- 데이터
def repeat_tokens(net, gen, B, L, n_rep):
    """eot + 길이 L 무작위 열 x n_rep. 무작위 토큰은 특수 토큰을 뺀 어휘에서 균등 추출."""
    V = len(net.tok)
    base = torch.randint(0, V - 1, (B, L), generator=gen)
    base = base + (base >= net.eot).long()                             # eot 하나만 건너뜀
    return torch.cat([torch.full((B, 1), net.eot), base.repeat(1, n_rep)], 1).to(net.device)


def pile_windows(net, n_docs, ctx, skip=0):
    """pile-10k에서 토큰 수 >= ctx-1인 문서의 앞 ctx-1 토큰, 앞에 eot를 붙여 길이 ctx. 앞에서 skip개 건너뜀(보정/평가 분리)."""
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    root = snapshot_download("NeelNanda/pile-10k", repo_type="dataset")
    texts = []
    for f in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
        texts += pq.read_table(f, columns=["text"]).column("text").to_pylist()
    out, seen = [], 0
    for t in texts:
        ids = net.tok(t[:20000])["input_ids"]
        if len(ids) >= ctx - 1:
            seen += 1
            if seen > skip:
                out.append([net.eot] + ids[:ctx - 1])
            if len(out) == n_docs:
                break
    return torch.tensor(out).to(net.device)


# --------------------------------------------------------------------------- (a) head 점수
@torch.no_grad()
def head_scores(net, seeds, B, L, n_rep, bs=16):
    """모든 head의 prefix-matching, copying(logit 경로), 이전 토큰 점수. 시드별 평균을 반환 (층, head)."""
    per_seed = []
    for s in seeds:
        gen = torch.Generator().manual_seed(s)
        toks_all = repeat_tokens(net, gen, B, L, n_rep)
        pm = torch.zeros(net.L, net.H)
        cl = torch.zeros(net.L, net.H)
        pt = torch.zeros(net.L, net.H)
        for i in range(0, B, bs):
            toks = toks_all[i:i + bs]
            _, pats, cap = net.run(toks, patterns=True, capture=True)
            for l in range(net.L):
                pm[l] += prefix_matching_score(pats[l], L, n_rep).mean(0).cpu() * len(toks)
                pt[l] += previous_token_score(pats[l]).mean(0).cpu() * len(toks)
                cl[l] += copy_logit(net, l, cap[l], pats[l], toks, L, n_rep).cpu() * len(toks)
        per_seed.append({"prefix": pm / B, "copy_logit": cl / B, "prev_token": pt / B})
    return {k: torch.stack([p[k] for p in per_seed]) for k in per_seed[0]}      # (seed, 층, head)


def copy_logit(net, l, z, pat, toks, L, n_rep):
    """minimal_impl.copy_scores_logit과 같은 정의(직접 logit 경로, 어휘 평균 뺀 ReLU, 가장 크게 attend한 토큰의 몫,
    (몫-1/K)/(1-1/K))를 한 층에 대해 계산. 어휘 전체 logit 대신 표본 토큰 K개의 logit과 어휘 평균 logit만 쓰는 정확한 축약.
    ln_f는 중심화와 gamma만 반영(표준편차 나눗셈은 목적지마다 양의 상수라 몫에 영향 없음)."""
    B, T = toks.shape
    dest = torch.arange(1 + L, 1 + L * n_rep, device=toks.device)
    ho = torch.einsum("bthk,hkm->bhtm", z.view(B, T, net.H, net.dh), net.W_O(l))[:, :, dest]    # (B,H,D,d)
    ho = ho - ho.mean(-1, keepdim=True)
    Wg = net.U * net.lnf[None, :]                                                                # (V,d)
    Wm = Wg.mean(0)
    src = pat[:, :, dest].argmax(-1)                                                             # (B,H,D)
    att_tok = torch.gather(toks[:, None, :].expand(-1, net.H, -1), 2, src)
    out = torch.zeros(B, net.H)
    for b in range(B):
        ids = torch.unique(toks[b, 1:])
        K = len(ids)
        lg = F.relu(ho[b] @ (Wg[ids] - Wm).T)                                                    # (H,D,K)
        pos = (att_tok[b][..., None] == ids[None, None, :]).float().argmax(-1)                    # (H,D)
        num = torch.gather(lg, 2, pos[..., None]).squeeze(-1)
        den = lg.sum(-1)
        valid = den > 1e-9
        s = (num / den.clamp_min(1e-9) - 1 / K) / (1 - 1 / K)
        out[b] = torch.stack([s[h][valid[h]].mean() if valid[h].any() else torch.tensor(float("nan"), device=s.device)
                              for h in range(net.H)]).cpu()
    return torch.nan_to_num(out, nan=0.0).mean(0)


@torch.no_grad()
def copy_eig_scores(net):
    """OV 고유값 copying 점수: 토큰 -> 토큰 회로의 sum Re lambda / sum |lambda| (ln1 gamma 포함, 정규화 크기는 무시)."""
    UE = (net.U * net.lnf[None, :]).T @ net.E                    # (d,d)
    out = torch.zeros(net.L, net.H)
    for l in range(net.L):
        D = net.ln1(l)
        Wv, Wo = net.W_V(l), net.W_O(l)
        for h in range(net.H):
            O = (D[:, None] * Wv[h]) @ Wo[h]                    # (d,d) 잔차 -> 잔차
            # E·O·U 의 0이 아닌 고유값은 d x d 곱 O·(U'^T E)와 같다
            M = O @ UE
            lam = torch.linalg.eigvals(M.cpu().double())
            out[l, h] = (lam.real.sum() / lam.abs().sum().clamp_min(1e-12)).item()
    return out


# --------------------------------------------------------------------------- (c) ablation
@torch.no_grad()
def mean_z(net, toks, bs=8):
    """보정 데이터에서 head별 z의 평균(배치와 위치 평균): {층: (H, dh)}."""
    s = {l: torch.zeros(net.H, net.dh, device=net.device) for l in range(net.L)}
    n = 0
    for i in range(0, len(toks), bs):
        _, _, cap = net.run(toks[i:i + bs], capture=True)
        for l in range(net.L):
            z = cap[l].view(*cap[l].shape[:2], net.H, net.dh)
            s[l] += z.sum((0, 1))
        n += toks[i:i + bs].shape[0] * toks.shape[1]
    return {l: v / n for l, v in s.items()}


def make_ablate(heads, mode, means):
    ab = {}
    for (l, h) in heads:
        ab.setdefault(l, {})[h] = 0.0 if mode == "zero" else means[l][h]
    return ab


@torch.no_grad()
def seq_losses(net, toks, ablate=None, bs=16):
    """(N, T-1) 위치별 손실."""
    return torch.cat([per_pos_loss(net.run(toks[i:i + bs], ablate=ablate)[0], toks[i:i + bs]).cpu()
                      for i in range(0, len(toks), bs)])


def boot_ci(x, n=2000, seed=0):
    """x (N,) 시퀀스별 값의 평균과 95% 부트스트랩 CI(시퀀스 재표집)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def rep_metric(loss, L):
    """반복 시퀀스(eot + L + L)에서 둘째 사본 손실(대상 토큰 위치 L+2..2L, 첫 대상은 예측 불가라 뺌)과 첫 사본 손실(1..L). (N,) 각각."""
    return loss[:, L + 1:2 * L].mean(1), loss[:, :L].mean(1)


def icl_terms(loss, early=50, late=500, w=5):
    """loss (N, 511): 인덱스 i는 토큰 i+1의 손실. 단일 토큰(early, late)과 +-w 창 평균. 반환 (early, late, early창, late창) (N,)."""
    e, l = loss[:, early - 1], loss[:, late - 1]
    ew = loss[:, early - 1 - w:early + w].mean(1)
    lw = loss[:, late - 1 - w:min(late + w, loss.shape[1])].mean(1)
    return e, l, ew, lw


def ablation_suite(net, top, rep_toks, rep_ref, nat_toks, nat_ref, L, n_random, seed, sizes, singles=True):
    heads_all = [(l, h) for l in range(net.L) for h in range(net.H)]
    pool = [x for x in heads_all if x not in top]
    rng = np.random.default_rng(seed)
    draws = {k: [[pool[i] for i in rng.choice(len(pool), k, replace=False)] for _ in range(n_random)] for k in sizes}
    means_rep, means_nat = mean_z(net, rep_ref), mean_z(net, nat_ref)

    base_rep = seq_losses(net, rep_toks)
    base_nat = seq_losses(net, nat_toks, bs=8)
    b2, b1 = rep_metric(base_rep, L)
    be, bl, bew, blw = icl_terms(base_nat)
    res = {"sizes": list(sizes), "baseline": {"rep_second": boot_ci(b2.numpy()), "rep_first": boot_ci(b1.numpy()),
                        "loss_tok50": boot_ci(be.numpy()), "loss_tok500": boot_ci(bl.numpy()),
                        "icl_single": boot_ci((be - bl).numpy()), "icl_window": boot_ci((bew - blw).numpy())},
           "n_rep_seq": len(rep_toks), "n_docs": len(nat_toks), "n_random_draws": n_random}

    def effect(heads_list, mode, nat=True):
        """heads_list: head 집합의 리스트(랜덤 baseline은 여러 개, 평균). 시퀀스별 (ablated - clean)의 평균과 CI."""
        r2s, r1s, ics, icw = [], [], [], []
        print(f"    {mode} {heads_list[0]}... x{len(heads_list)}", flush=True)
        for heads in heads_list:
            lr = seq_losses(net, rep_toks, make_ablate(heads, mode, means_rep))
            a2, a1 = rep_metric(lr, L)
            r2s.append((a2 - b2).numpy()); r1s.append((a1 - b1).numpy())
            if nat:
                ln = seq_losses(net, nat_toks, make_ablate(heads, mode, means_nat), bs=8)
                ae, al, aew, alw = icl_terms(ln)
                ics.append(((ae - al) - (be - bl)).numpy()); icw.append(((aew - alw) - (bew - blw)).numpy())
        avg = lambda xs: np.mean(xs, 0)
        out = {"rep_second": boot_ci(avg(r2s)), "rep_first": boot_ci(avg(r1s))}
        if nat:
            out.update({"icl_single": boot_ci(avg(ics)), "icl_window": boot_ci(avg(icw))})
        if len(heads_list) > 1:   # 추첨별 평균 효과의 범위(시퀀스 평균 후)
            out["draw_range_rep_second"] = [float(min(x.mean() for x in r2s)), float(max(x.mean() for x in r2s))]
            if nat:
                out["draw_range_icl_window"] = [float(min(x.mean() for x in icw)), float(max(x.mean() for x in icw))]
        return out

    for mode in ("zero", "mean"):
        r = {}
        if singles:
            # 단일 head: 반복 손실은 전부, 자연 텍스트 ICL은 비용 때문에 상위 5개만
            r["single"] = {f"{l}.{h}": effect([[(l, h)]], mode, nat=i < 5) for i, (l, h) in enumerate(top)}
        for k in sizes:   # 상위 k개 공동 ablation과 같은 크기의 무작위 head 집합(top 밖에서 추첨)
            r[f"joint{k}"] = effect([top[:k]], mode)
            r[f"random{k}"] = effect(draws[k], mode)
        res[mode] = r
    return res


# --------------------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "pythia-14m", "pythia-70m"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "gpt2_induction_results.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--topk", type=int, default=0, help="공동 ablation 집합 크기. 0이면 전체 head 수의 1/12 (GPT-2 small = 12)")
    a = ap.parse_args()
    dev = pick_device(a.device)
    q = a.quick
    cfg = {"score_L": 25, "score_n_rep": 4, "score_B": 16 if q else 64, "score_seeds": [0, 1] if q else [0, 1, 2],
           "abl_L": 50, "abl_n_rep": 2, "abl_B": 32 if q else 128, "abl_seed": 100, "ref_seed": 200,
           "icl_ctx": 512, "icl_docs": 40 if q else 120, "icl_ref_docs": 20 if q else 50,
           "n_random": 3 if q else 5}
    results = {"config": cfg, "device": str(dev), "models": {}}
    for name in a.models:
        t0 = time.time()
        net = Net(name, dev)
        print(f"[{name}] layers={net.L} heads={net.H}", flush=True)
        sc = head_scores(net, cfg["score_seeds"], cfg["score_B"], cfg["score_L"], cfg["score_n_rep"])
        mean = {k: v.mean(0) for k, v in sc.items()}
        sd = {k: v.std(0) for k, v in sc.items()}
        # 우연 수준: 균등 causal attention의 prefix 점수
        T = 1 + 25 * 4
        chance = float(prefix_matching_score(torch.tril(torch.ones(T, T)) / torch.arange(1, T + 1)[:, None], 25, 4))
        r = {"layers": net.L, "heads": net.H, "prefix": mean["prefix"].tolist(), "prefix_sd_over_seeds": sd["prefix"].tolist(),
             "copy_logit": mean["copy_logit"].tolist(), "prev_token": mean["prev_token"].tolist(), "prefix_chance_L25n4": chance}
        if net.gpt2:
            r["copy_eig"] = copy_eig_scores(net).tolist()
        flat = [(mean["prefix"][l, h].item(), l, h) for l in range(net.L) for h in range(net.H)]
        flat.sort(reverse=True)
        r["ranking"] = [{"head": f"{l}.{h}", "prefix": p, "copy_logit": mean["copy_logit"][l, h].item(),
                         "copy_eig": (r["copy_eig"][l][h] if net.gpt2 else None),
                         "prev_token": mean["prev_token"][l, h].item()} for p, l, h in flat[:15]]
        k = a.topk or max(2, round(net.L * net.H / 12))
        top = [(l, h) for _, l, h in flat[:k]]
        r["top_heads"] = [f"{l}.{h}" for l, h in top]
        print(f"  scores done {time.time() - t0:.0f}s", flush=True)
        print(f"  top-{k} heads: {r['top_heads']}", flush=True)
        # ablation 데이터: 점수 계산에 쓰지 않은 시드
        gen = torch.Generator().manual_seed(cfg["abl_seed"])
        rep_toks = repeat_tokens(net, gen, cfg["abl_B"], cfg["abl_L"], cfg["abl_n_rep"])
        rep_ref = repeat_tokens(net, torch.Generator().manual_seed(cfg["ref_seed"]), 64, cfg["abl_L"], cfg["abl_n_rep"])
        nat_ref = pile_windows(net, cfg["icl_ref_docs"], cfg["icl_ctx"])
        nat_toks = pile_windows(net, cfg["icl_docs"], cfg["icl_ctx"], skip=cfg["icl_ref_docs"])
        r["ablation"] = ablation_suite(net, top, rep_toks, rep_ref, nat_toks, nat_ref, cfg["abl_L"], cfg["n_random"],
                                       seed=0, sizes=sorted({min(5, len(top)), len(top)}), singles=net.gpt2)
        results["models"][name] = r
        print(f"  ablation done {time.time() - t0:.0f}s", flush=True)
        print(f"  done {time.time() - t0:.0f}s", flush=True)
        del net
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    plot(results)


# --------------------------------------------------------------------------- 그림
def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for nm in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        if any(nm in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = nm
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig_dir = HERE.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    g = results["models"].get("gpt2")
    if g is None:
        return
    # 그림 1: GPT-2 head 점수 지도
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for a_, key, title in zip(ax, ["prefix", "copy_logit", "prev_token"],
                              ["prefix-matching 점수 (반복 무작위 열)", "copying 점수 (logit 경로)", "이전 토큰 점수"]):
        m = np.array(g[key])
        im = a_.imshow(m, cmap="viridis", aspect="auto", origin="lower", vmin=0 if key != "copy_logit" else None)
        a_.set_title(title, fontsize=10)
        a_.set_xlabel("head")
        a_.set_ylabel("층")
        a_.set_xticks(range(m.shape[1]))
        a_.set_yticks(range(m.shape[0]))
        plt.colorbar(im, ax=a_, fraction=0.046)
        if key == "prefix":
            for hd in g["top_heads"]:
                l, h = map(int, hd.split("."))
                a_.plot(h, l, marker="o", mfc="none", mec="red", ms=11)
    fig.suptitle("GPT-2 small head 점수 (빨간 원 = prefix 점수 상위 head)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "own_gpt2_head_scores.png", dpi=150)
    plt.close(fig)
    # 그림 2: ablation 효과 (단일 / 공동 / 무작위)
    ab = g.get("ablation")
    if not ab or "zero" not in ab:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for a_, key, ttl in zip(ax, ["rep_second", "icl_window"],
                            ["반복 무작위 열의 둘째 사본 손실 증가 (nats)", "자연 텍스트 ICL 점수 변화 (창 평균, nats)"]):
        labels, vals, cols = [], [], []
        for mode, hatch in (("zero", ""), ("mean", "//")):
            for hd, e in ab[mode].get("single", {}).items():
                if key not in e:
                    continue
                labels.append(f"{hd}\n{mode}"); vals.append(e[key]); cols.append("#0072b2")
            for k in ab["sizes"]:
                labels.append(f"공동 상위{k}\n{mode}"); vals.append(ab[mode][f"joint{k}"][key]); cols.append("#d55e00")
                labels.append(f"무작위 {k}개\n{mode}"); vals.append(ab[mode][f"random{k}"][key]); cols.append("#8c8c8c")
        x = np.arange(len(vals))
        m = np.array([v[0] for v in vals]); lo = np.array([v[1] for v in vals]); hi = np.array([v[2] for v in vals])
        a_.bar(x, m, color=cols, yerr=[m - lo, hi - m], capsize=2)
        a_.set_xticks(x)
        a_.set_xticklabels(labels, fontsize=6, rotation=90)
        a_.axhline(0, color="k", lw=0.6)
        a_.set_title(ttl, fontsize=10)
    fig.suptitle("GPT-2 small head ablation (막대 = 평균, 선 = 시퀀스 부트스트랩 95% CI; 파랑 = 단일, 주황 = 공동, 회색 = 같은 수의 무작위 head)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "own_gpt2_ablation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
