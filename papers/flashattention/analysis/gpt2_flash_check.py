"""사전학습 GPT-2 small의 실제 Q, K, V(N=1024, d=64, 12층 x 12헤드 = 144개)에서 타일링 attention의 정확성과 HBM 접근 수를 확인한다.

논문 실험 설정(GPT-2, 시퀀스 1024, 헤드 차원 64)과 같은 모양이다. 학습은 하지 않고 순전파만 한다(CPU, 수 분 이내).
  - 입력: pile-10k에서 토큰 수가 1024 이상인 첫 문서 5개의 앞 1024토큰 (문서 5개 x 144헤드)
  - 비교: float64 정답, flash(float32/float16 입력), naive(float32/float16 입력, S와 P를 저장하는 PyTorch 관례)
  - 인과 마스크 사용 (GPT-2는 인과 모델)
필요: torch, transformers, pyarrow, huggingface_hub (openai-community/gpt2, NeelNanda/pile-10k가 로컬 캐시에 있어야 함)
실행: HF_HUB_OFFLINE=1 python3 gpt2_flash_check.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import minimal_impl as mi

torch.set_num_threads(4)
OUT = Path(__file__).parent / "gpt2_flash_check_results.json"
N_DOCS, N_CTX, M_SRAM = 5, 1024, 98304  # M: A100 SM당 SRAM 192 KB를 fp16 원소 수로 환산(논문 2.1절의 192KB)


def collect_qkv(model, ids):
    """각 층 attn.c_attn 출력에서 Q, K, V를 꺼낸다. 반환 (L, H, N, d) 세 개."""
    out = {}
    hs = [blk.attn.c_attn.register_forward_hook(lambda mod, a, o, l=l: out.__setitem__(l, o.detach()))
          for l, blk in enumerate(model.transformer.h)]
    with torch.no_grad():
        model(ids)
    for h in hs:
        h.remove()
    H, D = model.config.n_head, model.config.n_embd
    qs, ks, vs = [], [], []
    for l in range(len(model.transformer.h)):
        q, k, v = out[l][0].split(D, dim=-1)
        f = lambda x: x.reshape(x.shape[0], H, D // H).permute(1, 0, 2).contiguous()
        qs.append(f(q)); ks.append(f(k)); vs.append(f(v))
    return torch.stack(qs), torch.stack(ks), torch.stack(vs)


def main():
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
    root = Path(snapshot_download("NeelNanda/pile-10k", repo_type="dataset"))
    texts = pq.read_table(next(root.glob("data/*.parquet")), columns=["text"]).column("text").to_pylist()
    docs = []
    for t in texts:
        ids = tok(t, return_tensors="pt")["input_ids"]
        if ids.shape[1] >= N_CTX:
            docs.append(ids[:, :N_CTX])
        if len(docs) == N_DOCS:
            break
    rows = []
    for di, ids in enumerate(docs):
        Q, K, V = collect_qkv(model, ids)
        for l in range(Q.shape[0]):
            for h in range(Q.shape[1]):
                q, k, v = Q[l, h], K[l, h], V[l, h]
                ref = mi.naive_reference(q, k, v, causal=True)
                S = (q.double() @ k.double().T) / 8.0
                S = S.masked_fill(torch.ones(N_CTX, N_CTX, dtype=torch.bool).triu(1), float("-inf"))
                row = dict(doc=di, layer=l, head=h, max_logit=float(S[S > -1e30].abs().max()))
                row["f32_flash"] = float((mi.flash_forward(q, k, v, M_SRAM, causal=True)[0].double() - ref).abs().max())
                Sn = ((q @ k.T) / 8.0).masked_fill(torch.ones(N_CTX, N_CTX, dtype=torch.bool).triu(1), float("-inf"))
                row["f32_naive"] = float(((torch.softmax(Sn, -1) @ v).double() - ref).abs().max())
                q16, k16, v16 = q.half(), k.half(), v.half()
                ref16 = mi.naive_reference(q16, k16, v16, causal=True)
                row["f16_flash"] = float((mi.flash_forward(q16, k16, v16, M_SRAM, causal=True)[0].double() - ref16).abs().max())
                row["f16_flash_roundP"] = float((mi.flash_forward(q16, k16, v16, M_SRAM, causal=True, p_round_dtype=torch.float16)[0].double() - ref16).abs().max())
                S16 = ((q16 @ k16.T) / 8.0).masked_fill(torch.ones(N_CTX, N_CTX, dtype=torch.bool).triu(1), float("-inf"))
                row["f16_naive"] = float(((torch.softmax(S16, -1) @ v16).double() - ref16).abs().max())
                rows.append(row)
    A = {k: np.array([r[k] for r in rows]) for k in ("max_logit", "f32_flash", "f32_naive", "f16_flash", "f16_flash_roundP", "f16_naive")}
    # 층별 요약(문서와 헤드 평균)
    per_layer = []
    for l in range(12):
        sel = np.array([r["layer"] == l for r in rows])
        per_layer.append(dict(layer=l, max_logit_max=float(A["max_logit"][sel].max()), max_logit_median=float(np.median(A["max_logit"][sel])),
                              f16_flash_max=float(A["f16_flash"][sel].max()), f16_naive_max=float(A["f16_naive"][sel].max())))
    # 문서를 군집으로 보는 부트스트랩: 문서 5개를 복원추출해 로그 오차비의 평균 신뢰구간
    ratio = np.log(A["f16_naive"] / A["f16_flash"])
    docs_idx = np.array([r["doc"] for r in rows])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        pick = rng.integers(0, N_DOCS, N_DOCS)
        boots.append(np.exp(np.mean(np.concatenate([ratio[docs_idx == p] for p in pick]))))
    # HBM 접근 수: 인과 마스크의 건너뛴 블록 포함, 헤드 하나 기준(원소 수)
    cf, cb = mi.count_flash(N_CTX, 64, M_SRAM, causal=True)
    cf_d, cb_d = mi.count_flash(N_CTX, 64, M_SRAM, causal=False)
    sf, sb = mi.count_standard(N_CTX, 64)
    summary = dict(
        n_heads_total=len(rows), n_docs=len(docs), n_ctx=N_CTX, M_sram=M_SRAM,
        max_logit=dict(max=float(A["max_logit"].max()), median=float(np.median(A["max_logit"]))),
        f32_flash_max=float(A["f32_flash"].max()), f32_naive_max=float(A["f32_naive"].max()),
        f16_flash_max=float(A["f16_flash"].max()), f16_naive_max=float(A["f16_naive"].max()),
        f16_flash_roundP_max=float(A["f16_flash_roundP"].max()), f16_flash_roundP_median=float(np.median(A["f16_flash_roundP"])),
        frac_heads_roundP_better_than_naive=float(np.mean(A["f16_flash_roundP"] < A["f16_naive"])),
        f16_flash_median=float(np.median(A["f16_flash"])), f16_naive_median=float(np.median(A["f16_naive"])),
        frac_heads_flash_better_fp16=float(np.mean(A["f16_flash"] < A["f16_naive"])),
        geomean_err_ratio_naive_over_flash=float(np.exp(ratio.mean())),
        geomean_ratio_ci95_doc_bootstrap=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        corr_logit_vs_log_ratio=float(np.corrcoef(np.log(A["max_logit"]), ratio)[0, 1]),
        hbm_per_head_elements=dict(flash_fwd_causal=cf.total, flash_bwd_causal=cb.total, flash_fwd_dense=cf_d.total, flash_bwd_dense=cb_d.total,
                                   standard_fwd=sf.total, standard_bwd=sb.total,
                                   ratio_dense_fwd_bwd=(sf.total + sb.total) / (cf_d.total + cb_d.total)),
    )
    OUT.write_text(json.dumps(dict(summary=summary, per_layer=per_layer, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))
    for p in per_layer:
        print(p)


if __name__ == "__main__":
    main()
