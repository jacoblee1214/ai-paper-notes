"""Pythia 초안·목표 모델 공통 도구. 가중치(EleutherAI/pythia-14m|70m|160m)와 텍스트(NeelNanda/pile-10k)는 로컬 캐시만 쓴다(HF_HUB_OFFLINE=1).
추론만 한다(학습·미세조정 없음)."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import minimal_impl as M

torch.set_num_threads(4)
PARQUET = "data/train-00000-of-00001-4746b8785c874cc7.parquet"


def load_tok():
    return AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")


def load_model(size):
    return AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-{size}", dtype=torch.float32).eval()


def load_prompts(tok, n, prompt_len, seed=0):
    """pile-10k를 시드 고정 순서로 섞어, 첫 prompt_len 토큰이 있는 문서의 앞부분 n개. 반환 [n, prompt_len] long."""
    p = hf_hub_download("NeelNanda/pile-10k", PARQUET, repo_type="dataset", local_files_only=True)
    docs = pd.read_parquet(p)["text"].tolist()
    order = np.random.RandomState(seed).permutation(len(docs))
    out = []
    for i in order:
        if len(docs[i]) < 400:
            continue
        ids = tok(docs[i]).input_ids
        if len(ids) >= prompt_len:
            out.append(ids[:prompt_len])
        if len(out) == n:
            break
    return torch.tensor(out)


@torch.inference_mode()
def probs_last(model, ids, k, temperature=1.0, top_k=None, chunk=64):
    """ids[B, L]의 마지막 k개 위치에서의 다음 토큰 분포 [B, k, V] (표준화 적용). logits 메모리를 줄이려고 마지막 k 위치만 계산한다."""
    outs = []
    for s in range(0, ids.shape[0], chunk):
        h = model.base_model(input_ids=ids[s : s + chunk]).last_hidden_state[:, -k:]
        outs.append(M.standardize(model.get_output_embeddings()(h).float(), temperature, top_k))
    return torch.cat(outs, 0)


def make_fns(draft, target, temperature=1.0, top_k=None):
    draft_fn = lambda ids: probs_last(draft, ids, 1, temperature, top_k)[:, 0]
    target_fn = lambda ids, k: probs_last(target, ids, k, temperature, top_k)
    return draft_fn, target_fn


def bootstrap_ci(per_prompt_values, stat=np.mean, reps=10000, seed=0):
    """프롬프트 단위 부트스트랩 95% 신뢰구간. per_prompt_values: 프롬프트마다 하나의 값(또는 1차원 배열 리스트)."""
    rng = np.random.default_rng(seed)
    n = len(per_prompt_values)
    vals = [stat(np.array([per_prompt_values[i] for i in rng.integers(0, n, n)])) for _ in range(reps)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
