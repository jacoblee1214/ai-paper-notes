"""Pythia-14M(초안)과 Pythia-70M·160M(목표)의 토큰별 수락률 beta = sum_x min(p, q)를 정확한 합으로 잰다.
논문 4.2절과 같은 방식: 목표 모델이 샘플링한 텍스트 위에서 두 모델의 분포를 구한다(teacher forcing).
  - 프롬프트: pile-10k 문서의 첫 32토큰 100개(시드 0), 목표 모델이 96토큰을 생성(T = 0, 0.5, 1; top-k 없음)
  - 문서 단위 부트스트랩(10000회)으로 alpha의 95% 신뢰구간, beta 분포(분위수, 히스토그램), 위치별 평균, 목표 분포의 날카로움과의 상관
결과: real_alpha_results.json, real_alpha_beta.npz
실행: HF_HUB_OFFLINE=1 python3 real_alpha.py  (CPU, 약 3분)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import minimal_impl as M
import real_common as R

N_PROMPTS, PROMPT_LEN, NEW = 100, 32, 96
OUT = Path(__file__).with_name("real_alpha_results.json")


@torch.inference_mode()
def generate(target, prompts, T, seed):
    torch.manual_seed(seed)
    kw = dict(do_sample=False) if T == 0 else dict(do_sample=True, temperature=T, top_k=0, top_p=1.0)
    out = target.generate(prompts, attention_mask=torch.ones_like(prompts), max_new_tokens=NEW, min_new_tokens=NEW,
                          pad_token_id=0, **kw)
    return out


def main():
    t0 = time.time()
    tok = R.load_tok()
    prompts = R.load_prompts(tok, N_PROMPTS, PROMPT_LEN, seed=0)
    draft = R.load_model("14m")
    res, beta_store = {}, {}
    for size in ("70m", "160m"):
        target = R.load_model(size)
        for T in (0, 0.5, 1.0):
            full = generate(target, prompts, T, seed=1)
            L = full.shape[1]
            betas, p_max, p_ent = [], [], []
            for s in range(0, N_PROMPTS, 5):
                ids = full[s : s + 5]
                k = NEW
                pk = R.probs_last(target, ids[:, :-1], k, T if T > 0 else 0)      # 위치 L-1-k .. L-2의 다음 토큰 분포
                qk = R.probs_last(draft, ids[:, :-1], k, T if T > 0 else 0)
                betas.append(M.accept_prob(pk, qk))
                p_max.append(pk.max(-1).values)
                p_ent.append(-(pk * pk.clamp_min(1e-30).log()).sum(-1))
            beta = torch.cat(betas).numpy(); pmax = torch.cat(p_max).numpy(); ent = torch.cat(p_ent).numpy()
            key = f"target={size},T={T}"
            beta_store[key] = beta
            per_prompt = beta.mean(1)
            lo, hi = R.bootstrap_ci(list(per_prompt))
            rho, rho_p = stats.spearmanr(beta.ravel(), pmax.ravel()) if T > 0 else (float('nan'), float('nan'))   # T=0이면 p는 one-hot이라 상관이 정의되지 않음
            q = np.quantile(beta, [0.05, 0.25, 0.5, 0.75, 0.95])
            res[key] = dict(
                target=size, T=T, n_prompts=N_PROMPTS, tokens=int(beta.size), prompt_len=PROMPT_LEN, new_tokens=NEW,
                alpha=float(beta.mean()), alpha_ci95_prompt_bootstrap=[lo, hi],
                beta_quantiles_5_25_50_75_95=q.tolist(), beta_std=float(beta.std()),
                frac_beta_ge_0_99=float((beta >= 0.99).mean()), frac_beta_le_0_01=float((beta <= 0.01).mean()),
                hist_10bins=np.histogram(beta, bins=10, range=(0, 1))[0].tolist(),
                alpha_by_position_block=[float(beta[:, i : i + 16].mean()) for i in range(0, NEW, 16)],
                spearman_beta_vs_target_pmax=None if T == 0 else float(rho), spearman_p=None if T == 0 else float(rho_p),
                mean_target_entropy_nats=None if T == 0 else float(ent.mean()), mean_target_pmax=None if T == 0 else float(pmax.mean()),
                distinct_tokens_frac=float(len(set(full[:, PROMPT_LEN:].flatten().tolist())) / full[:, PROMPT_LEN:].numel()),
            )
            print(key, round(res[key]["alpha"], 4), [round(v, 4) for v in res[key]["alpha_ci95_prompt_bootstrap"]],
                  "t", round(time.time() - t0), flush=True)
        del target
    res["_meta"] = dict(draft="pythia-14m", seconds=round(time.time() - t0), seed_gen=1, seed_prompts=0)
    OUT.write_text(json.dumps(res, indent=1))
    np.savez_compressed(Path(__file__).with_name("real_alpha_beta.npz"), **{k.replace(",", "_").replace("=", ""): v for k, v in beta_store.items()})


if __name__ == "__main__":
    main()
