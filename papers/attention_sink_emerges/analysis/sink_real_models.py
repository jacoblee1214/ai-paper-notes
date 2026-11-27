"""공개 가중치(Pythia-14M/70M/160M, GPT-2)로 논문의 Sink 지표를 BOS 유무, 임계값 eps, 길이 T를 바꿔 재측정한다.

지표(논문 절 3.2, 공식 코드 eval_sink_open_source.compute_attention_sink 와 같은 정의):
  alpha_k = 1/(T-k+1) * sum_{i=k}^{T} A_{i,k}      (헤드 (l,h), 시퀀스 하나)
  Sink_k^eps = (1/L)(1/H) sum_{l,h} 1[alpha_k^{l,h} > eps]
공식 코드처럼 시퀀스마다 헤드별로 임계값을 적용한 뒤 시퀀스와 헤드에 대해 평균한다. 여기서는 k=1(첫 토큰)과 k=2를 잰다.

조건(문서 d의 토큰열 x, 모델 최대 입력 길이 S=1024로 한 번 순전파하고 앞 T개로 자른다.
인과 마스크라 앞 T개의 attention은 길이 T 입력을 따로 넣은 것과 같다):
  start : x[0:S]            문서 첫 토큰부터, BOS 없음 (논문 절 4, 코드의 기본 조건)
  bos   : [BOS] + x[0:S-1]  BOS를 앞에 붙임 (총 길이 S). BOS = 토크나이저의 <|endoftext|>
  mid   : x[o:o+S]          문서 중간의 무작위 위치 o 에서 시작, BOS 없음 (학습 때의 패킹 조각처럼 첫 토큰이 문서 중간)
같은 문서 집합과 같은 순서를 모든 조건에 쓴다(짝 비교). 신뢰구간은 문서에 대한 부트스트랩(2000회, 95%).

실행: HF_HUB_OFFLINE=1 python3 -I analysis/sink_real_models.py   (torch, transformers, pandas, pyarrow 필요. 모델과 데이터는 로컬 HF 캐시)
모델: EleutherAI/pythia-14m, -70m, -160m, openai-community/gpt2 (최종 체크포인트). 데이터: NeelNanda/pile-10k (Pile 앞 10k 문서)
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["EleutherAI/pythia-14m", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "openai-community/gpt2"]
S = 1024  # 두 모델군이 모두 받는 최대 길이(GPT-2의 문맥 한계)
TS = [64, 256, 1024]
EPS = [0.1, 0.2, 0.3, 0.4, 0.5]  # 논문 기본값 0.3 포함
N_DOCS, SEED, N_BOOT = 200, 0, 2000
CONDS = ["start", "bos", "mid"]


def load_docs():
    p = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                        repo_type="dataset", local_files_only=True)
    return pd.read_parquet(p)["text"].tolist()


def pick_docs(texts, tokenizers):
    """모든 토크나이저에서 S+76 토큰 이상인 문서 중 고정 시드로 N_DOCS개."""
    ok = []
    for i, t in enumerate(texts):
        if len(t) < 3000:  # 1100토큰이 안 될 문서를 글자 수로 먼저 거른다(토큰당 약 4자 이상)
            continue
        if all(len(tok(t).input_ids) >= S + 76 for tok in tokenizers):
            ok.append(i)
    rng = np.random.RandomState(SEED)
    chosen = sorted(rng.choice(ok, N_DOCS, replace=False).tolist())
    return chosen, len(ok)


@torch.no_grad()
def alphas(model, ids):
    """ids: (S,) 토큰. 반환: a1[T], a2[T] 각각 (L,H) 배열의 딕셔너리."""
    out = model(input_ids=ids[None], output_attentions=True)
    a1 = {T: [] for T in TS}
    a2 = {T: [] for T in TS}
    for A in out.attentions:  # (1,H,S,S), 행이 질의 i, 열이 key k
        A = A[0].float()
        c0 = A[:, :, 0].cumsum(-1)  # (H,S): sum_{i<=t} A[i,0]
        c1 = A[:, :, 1].cumsum(-1)
        for T in TS:
            a1[T].append((c0[:, T - 1] / T).cpu().numpy())  # 1/(T-1+1) sum_{i=1}^T A_{i,1}
            a2[T].append((c1[:, T - 1] / (T - 1)).cpu().numpy())  # 1/(T-2+1) sum_{i=2}^T A_{i,2}
    return {T: np.stack(a1[T]) for T in TS}, {T: np.stack(a2[T]) for T in TS}


def boot_ci(x, rng):
    """x: (n,) 문서별 값. 평균의 부트스트랩 95% 구간."""
    idx = rng.randint(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summ(x, rng):
    return {"mean": float(x.mean()), "ci95": boot_ci(x, rng)}


def main():
    texts = load_docs()
    toks = {m: AutoTokenizer.from_pretrained(m) for m in MODELS}
    chosen, n_pool = pick_docs(texts, list(toks.values()))
    rng_mid = np.random.RandomState(SEED + 1)
    mid_frac = rng_mid.rand(N_DOCS)  # 문서별 시작 위치 비율(모델 간 같은 비율을 쓴다)
    res = {"meta": {"n_docs": N_DOCS, "seed": SEED, "pool_docs": n_pool, "S": S, "T": TS, "eps": EPS, "n_boot": N_BOOT,
                    "data": "NeelNanda/pile-10k (text 열), 고정 시드 표본, 문서 길이 >= S+76 토큰(두 토크나이저 모두)",
                    "torch": torch.__version__, "device": "cpu", "dtype": "float32", "attn": "eager"},
           "models": {}}
    for m in MODELS:
        tok = toks[m]
        model = AutoModelForCausalLM.from_pretrained(m, attn_implementation="eager", torch_dtype=torch.float32).eval()
        cfg = model.config
        L, H = cfg.num_hidden_layers, cfg.num_attention_heads
        bos = tok.bos_token_id
        info = {"L": L, "H": H, "bos_token": tok.bos_token, "bos_id": bos, "eos_id": tok.eos_token_id,
                "adds_bos_by_default": bool(tok("hello").input_ids[0] == bos),
                "max_pos": int(cfg.max_position_embeddings)}
        A1 = {c: {T: np.zeros((N_DOCS, L, H)) for T in TS} for c in CONDS}
        A2 = {c: {T: np.zeros((N_DOCS, L, H)) for T in TS} for c in CONDS}
        for j, d in enumerate(chosen):
            x = tok(texts[d]).input_ids
            o = 1 + int(mid_frac[j] * (len(x) - S - 1))  # 1 <= o <= len-S
            seqs = {"start": x[:S], "bos": [bos] + x[:S - 1], "mid": x[o:o + S]}
            for c, s in seqs.items():
                assert len(s) == S
                a1, a2 = alphas(model, torch.tensor(s))
                for T in TS:
                    A1[c][T][j], A2[c][T][j] = a1[T], a2[T]
            if j % 50 == 0:
                print(m, j, flush=True)
        rng = np.random.RandomState(SEED + 2)
        mr = {"info": info, "cond": {}, "paired": {}, "layer_head_alpha1_T64": {}}
        for c in CONDS:
            mr["cond"][c] = {}
            for T in TS:
                a1 = A1[c][T]
                a2 = A2[c][T]
                mr["cond"][c][str(T)] = {
                    "alpha1_headmean": summ(a1.mean((1, 2)), rng),
                    "alpha1_headmax": summ(a1.max((1, 2)), rng),
                    "alpha2_headmean": summ(a2.mean((1, 2)), rng),
                    "uniform_baseline": sum(1.0 / i for i in range(1, T + 1)) / T,
                    "sink1": {str(e): summ(100 * (a1 > e).mean((1, 2)), rng) for e in EPS},
                    "sink2": {str(e): summ(100 * (a2 > e).mean((1, 2)), rng) for e in EPS},
                }
            mr["layer_head_alpha1_T64"][c] = A1[c][64].mean(0).round(4).tolist()
        # 짝 비교(같은 문서): 평균 차이와 부트스트랩 구간
        def diff(x, y):
            return summ(x - y, rng)
        sk = lambda c, T, e: 100 * (A1[c][T] > e).mean((1, 2))
        for e in EPS:
            mr["paired"][f"bos_minus_start_sink1_T64_eps{e}"] = diff(sk("bos", 64, e), sk("start", 64, e))
            mr["paired"][f"mid_minus_start_sink1_T64_eps{e}"] = diff(sk("mid", 64, e), sk("start", 64, e))
        for c in CONDS:
            for e in (0.3,):
                mr["paired"][f"{c}_T1024_minus_T64_sink1_eps{e}"] = diff(sk(c, 1024, e), sk(c, 64, e))
                mr["paired"][f"{c}_T256_minus_T64_sink1_eps{e}"] = diff(sk(c, 256, e), sk(c, 64, e))
        mr["paired"]["bos_minus_start_alpha1_T64"] = diff(A1["bos"][64].mean((1, 2)), A1["start"][64].mean((1, 2)))
        # 한 헤드 이상 sink가 있는 층(기본 조건, 헤드-문서 평균 alpha_1 > 0.3)
        mr["heads_mean_alpha1_gt_0.3_start_T64"] = int((A1["start"][64].mean(0) > 0.3).sum())
        res["models"][m] = mr
        del model
    (ROOT / "analysis" / "sink_real_models_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for m, mr in res["models"].items():
        print(m, mr["info"])
        for c in CONDS:
            for T in TS:
                r = mr["cond"][c][str(T)]
                print(f"  {c:5s} T={T:4d} a1={r['alpha1_headmean']['mean']:.3f} base={r['uniform_baseline']:.3f} "
                      f"sink1(0.3)={r['sink1']['0.3']['mean']:.1f} {r['sink1']['0.3']['ci95']}")


if __name__ == "__main__":
    main()
