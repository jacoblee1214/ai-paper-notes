"""스트리밍 perplexity 비교: 전체(dense) / 윈도만 / 싱크 s개 + 최근 윈도 / 슬라이딩 윈도 + 재계산.

데이터: NeelNanda/pile-10k(Pile 앞 1만 문서)를 고정 시드로 섞어 문서를 이어 붙인 스트림 n개(각 스트림은 문서 첫머리에서 시작).
모델: Pythia-14M/70M/160M(RoPE), GPT-2 small(학습된 절대 위치). 가중치와 데이터는 로컬 캐시만 쓴다(HF_HUB_OFFLINE=1).
측정: 스트림마다 위치 t(입력 토큰 번호)에서 토큰 t+1의 nll. 구간별 평균 nll -> exp = perplexity.
통계: 신뢰구간은 스트림 단위 부트스트랩(2000회), 비교는 스트림 짝지은 부호 뒤집기 검정(20000회) 후 Holm 보정.
실행: python3 run_streaming_ppl.py [모델키 ...]   (키: p160 p70 p14 gpt2 gpt2long, 기본은 전부)
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from minimal_impl import MiniLM, nll_recompute

HERE = Path(__file__).resolve().parent
OUT = HERE / "streaming_ppl_results.json"
SEED, N_STREAMS, N_BOOT, N_PERM, STRIDE = 0, 24, 2000, 20000, 64
SINKS = [0, 1, 2, 4, 8]
DEV = os.environ.get("STREAM_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
# 캐시 크기별로 돌릴 싱크 수와 스트림별 스트라이드(CPU 시간 때문에 Pythia-160M은 C=512만 전체 격자)
SINKS_BY = {("p160", 256): [0, 4], ("p160", 1024): [0, 4]}
STRIDE_BY = {"p160": 128}

# 키: (HF 이름, 스트림 길이 T, 캐시 크기 C 목록, 위치 모드 목록, 재계산 C 목록, 사전학습 문맥 길이)
PLAN = {
    "p160": ("EleutherAI/pythia-160m", 4096, [256, 512, 1024], ["orig"], [512], 2048),
    "p70": ("EleutherAI/pythia-70m", 4096, [512], ["orig"], [512], 2048),
    "p14": ("EleutherAI/pythia-14m", 4096, [512], ["orig"], [], 2048),
    "gpt2": ("openai-community/gpt2", 1024, [128, 256, 512], ["orig", "slot"], [256], 1024),
    "gpt2long": ("openai-community/gpt2", 2048, [256], ["slot"], [256], 1024),
}


def load_docs():
    p = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                        repo_type="dataset", local_files_only=True)
    return pd.read_parquet(p)["text"].tolist()


def make_streams(tok, docs, T, n):
    """문서를 시드 고정 순서로 이어 붙여(문서 뒤에 EOS) 길이 T 스트림 n개. 스트림은 문서 첫머리에서 시작."""
    order = np.random.RandomState(SEED).permutation(len(docs))
    streams, buf, k = [], [], 0
    while len(streams) < n:
        t = docs[order[k]]
        k += 1
        if len(t) < 200:
            continue
        buf += tok(t).input_ids + [tok.eos_token_id]
        if len(buf) >= T:
            streams.append(buf[:T])
            buf = []
    return torch.tensor(streams)


def boot_ci(x, rng):
    x = np.asarray(x)
    means = x[rng.randint(0, len(x), (N_BOOT, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def perm_p(d, rng):
    """짝지은 차이 d(스트림별)의 부호 뒤집기 검정, 양측. 평균 차이가 0이라는 귀무가설."""
    d = np.asarray(d)
    obs = abs(d.mean())
    sg = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return float(((np.abs((sg * d).mean(1)) >= obs - 1e-12).sum() + 1) / (N_PERM + 1))


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def buckets(T, C, pre_ctx):
    """위치 t(입력 토큰 번호) 구간. pre: 캐시가 안 찼음(t < C), mid: C <= t < 사전학습 문맥, late: 그 이후."""
    b = {"pre": (0, C), "post_all": (C, T - 1)}
    if pre_ctx > C:
        b["mid"] = (C, min(pre_ctx, T - 1))
    if T - 1 > pre_ctx:
        b["late"] = (max(pre_ctx, C), T - 1)
    return b


def summarize(nll, bk, rng, cols=None):
    """nll: (n, T-1) 또는 (n, len(cols)) numpy. cols: 열에 해당하는 위치 배열(재계산·스트라이드용)."""
    pos = np.arange(nll.shape[1]) if cols is None else np.asarray(cols)
    out = {}
    for name, (lo, hi) in bk.items():
        sel = (pos >= lo) & (pos < hi)
        if sel.sum() == 0:
            continue
        per = nll[:, sel].mean(1)                      # 스트림별 평균 nll
        out[name] = {"nll": float(per.mean()), "ppl": float(np.exp(per.mean())),
                     "ppl_ci95": [float(np.exp(v)) for v in boot_ci(per, rng)], "n_pos": int(sel.sum()),
                     "per_stream_nll": [round(float(v), 5) for v in per]}
    return out


def run_model(key, docs, results):
    name, T, Cs, modes, rc_Cs, pre_ctx = PLAN[key]
    rng = np.random.RandomState(SEED)
    stride = STRIDE_BY.get(key, STRIDE)
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    m = MiniLM.from_hf(hf).to(DEV)
    del hf
    ids = make_streams(tok, docs, T, N_STREAMS).to(DEV)
    nl_id = tok("\n").input_ids[0]
    R = results.setdefault(key, {"model": name, "T": T, "n_streams": N_STREAMS, "pre_ctx": pre_ctx,
                                 "device": DEV, "configs": {}})
    t0 = time.time()
    # 전체 attention. GPT-2는 1024 안에서만 가능. Pythia는 사전학습 문맥(2048)을 넘으면 외삽이다.
    if key != "gpt2long":
        d = m.nll(ids).cpu().numpy()
        R["dense"] = {"all": summarize(d, {"all": (0, T - 1), "le_pre_ctx": (0, pre_ctx), "gt_pre_ctx": (pre_ctx, T - 1)}, rng)}
        R["dense"]["curve_bin128"] = [float(v) for v in d[:, :(T - 1) // 128 * 128].reshape(N_STREAMS, -1, 128).mean((0, 2))]
        R["_dense_nll"] = d
        print(key, "dense", round(time.time() - t0), "s", flush=True)
    for C in Cs:
        bk = buckets(T, C, pre_ctx)
        stride_pos = [t for t in range(C, T - 1) if t % stride == stride - 1]
        sinks = SINKS_BY.get((key, C), SINKS)
        for mode in modes:
            for s in sinks:
                if s >= C:
                    continue
                d = m.nll(ids, spec=(s, C - s), pos_mode=mode).cpu().numpy()
                cfg = {"C": C, "s": s, "mode": mode, "buckets": summarize(d, bk, rng),
                       "stride": summarize(d[:, stride_pos], bk, rng, stride_pos),
                       "curve_bin128": [float(v) for v in d[:, :(T - 1) // 128 * 128].reshape(N_STREAMS, -1, 128).mean((0, 2))]}
                R["configs"][f"C{C}|s{s}|{mode}"] = cfg
                R.setdefault("_nll", {})[f"C{C}|s{s}|{mode}"] = d[:, stride_pos]
                if DEV == "mps":
                    torch.mps.empty_cache()
                print(key, C, mode, f"s={s}", {k: round(v['ppl'], 2) for k, v in cfg["buckets"].items()},
                      round(time.time() - t0), "s", flush=True)
            if 4 in sinks:
                # 표 1 변형: 처음 4토큰을 줄바꿈 토큰으로 바꿔 입력(예측 대상은 그대로), 싱크 4개 유지
                ids2 = ids.clone()
                ids2[:, :4] = nl_id
                # 대상 토큰은 원래 ids이므로 입력만 바꾼 nll을 직접 계산한다
                d = nll_with_inputs(m, ids2, ids, (4, C - 4), mode)
                R["configs"][f"C{C}|s4nl|{mode}"] = {"C": C, "s": 4, "mode": mode, "buckets": summarize(d, bk, rng),
                                                     "stride": summarize(d[:, stride_pos], bk, rng, stride_pos)}
                R["_nll"][f"C{C}|s4nl|{mode}"] = d[:, stride_pos]
        if C in rc_Cs:
            ts = stride_pos
            r = nll_recompute(m, ids, C, ts, bs=24).cpu().numpy()
            R["configs"][f"C{C}|recompute"] = {"C": C, "stride": summarize(r, bk, rng, ts)}
            R["_nll"][f"C{C}|recompute"] = r
            print(key, C, "recompute", round(time.time() - t0), "s", flush=True)
    R["stride"] = stride
    add_comparisons(R, Cs, modes, rng)
    R.pop("_nll", None)  # 요약만 남김(스트림별 평균은 summarize 안에 있음)
    R.pop("_dense_nll", None)
    return R


@torch.no_grad()
def nll_with_inputs(m, inp, tgt, spec, mode):
    import torch.nn.functional as F
    res = []
    for i in range(0, inp.shape[0], 4):
        h = m.hidden(inp[i:i + 4], spec, mode)
        parts = []
        for a in range(0, inp.shape[1] - 1, 512):
            b = min(a + 512, inp.shape[1] - 1)
            parts.append(F.cross_entropy(m.head(h[:, a:b]).float().transpose(1, 2), tgt[i:i + 4, a + 1:b + 1], reduction="none"))
        res.append(torch.cat(parts, 1))
    return torch.cat(res).cpu().numpy()


def add_comparisons(R, Cs, modes, rng):
    """C마다 post 구간(스트라이드 위치) 스트림별 평균 nll 차이. 비교 목록 전체에 Holm 보정."""
    comps = []
    for C in Cs:
        for mode in modes:
            def g(k):
                c = R["configs"].get(f"C{C}|{k}")
                return None if c is None else np.array(c["stride"]["post_all"]["per_stream_nll"])
            base = g(f"s4|{mode}")
            pairs = [(f"s0|{mode}", f"s4|{mode}"), (f"s1|{mode}", f"s4|{mode}"), (f"s2|{mode}", f"s4|{mode}"),
                     (f"s8|{mode}", f"s4|{mode}"), (f"s4nl|{mode}", f"s4|{mode}")]
            if mode == modes[0]:
                pairs.append((f"s4|{mode}", "recompute"))
            for a, b in pairs:
                x, y = g(a), g(b)
                if x is None or y is None:
                    continue
                d = x - y
                comps.append({"C": C, "A": a, "B": b, "mean_diff_nll": float(d.mean()), "ci95": boot_ci(d, rng),
                              "p_perm": perm_p(d, rng), "n_streams": len(d)})
    adj = holm([c["p_perm"] for c in comps])
    for c, a in zip(comps, adj):
        c["p_holm"] = a
    R["comparisons_post_stride"] = comps


def main():
    keys = sys.argv[1:] or list(PLAN)
    docs = load_docs()
    results = json.load(open(OUT)) if OUT.exists() else {}
    results["_meta"] = {"seed": SEED, "n_streams": N_STREAMS, "stride": f"{STRIDE} (p160은 {STRIDE_BY['p160']})", "n_boot": N_BOOT, "n_perm": N_PERM,
                        "data": "NeelNanda/pile-10k text, 문서를 시드 0 순서로 이어 붙임(문서 뒤 EOS), 문서 첫머리에서 시작",
                        "torch": torch.__version__, "dtype": "float32", "device": DEV}
    for k in keys:
        run_model(k, docs, results)
        json.dump(results, open(OUT, "w"), ensure_ascii=False)
        print("saved", k, flush=True)


if __name__ == "__main__":
    main()
