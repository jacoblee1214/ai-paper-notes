"""논문 3.3절(사전학습 때 모든 샘플 앞에 학습 가능한 싱크 토큰 하나)의 아주 작은 토이.

바이트 수준 4층 RoPE 디코더(약 0.8M 파라미터)를 pile-10k 바이트로 처음부터 학습한다. 두 가지를 같은 조건으로 비교한다.
  vanilla : 무작위로 뽑은 바이트 조각 128개를 입력으로 학습(조각의 첫 토큰이 일정하지 않음)
  sink    : 같은 방식으로 뽑은 조각 앞에 학습 가능한 싱크 토큰(id 256)을 붙여 입력 길이 128(싱크 + 127바이트)로 학습(예측 대상 128개로 같음)
평가: 학습에 안 쓴 문서 뒤쪽(인덱스 9000 이상)을 이은 길이 512 스트림 32개(문서 첫머리에서 시작).
스트림 앞에 싱크 토큰을 붙인 것은 sink 모델뿐이다. 캐시 크기 C=64의 "x+y" 구성(x개 초기 토큰 + 최근 y개)으로 스트리밍 perplexity를 잰다.
이 규모는 논문(Pythia-160M, 143k 스텝, 8xA6000)과 비교할 수 없는 토이다. 싱크가 안 생기는 것도 결과다.
실행: python3 sink_token_toy.py   (MPS에서 수 분)
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from minimal_impl import MiniLM, rope

HERE = Path(__file__).resolve().parent
DEV = os.environ.get("STREAM_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
SINK_ID, V = 256, 257
L_TRAIN, BATCH, STEPS, LR = 128, 32, 1200, 2e-3
C, T_EVAL, N_EVAL, N_BOOT = 64, 512, 32, 2000
SEEDS = [0, 1]


def load_bytes():
    p = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet",
                        repo_type="dataset", local_files_only=True)
    texts = pd.read_parquet(p)["text"].tolist()
    train = np.frombuffer("\n".join(texts[:9000]).encode("utf-8", "ignore"), dtype=np.uint8)
    ev, buf = [], []
    for t in texts[9000:]:
        buf += list(t.encode("utf-8", "ignore"))
        if len(buf) >= T_EVAL and len(t) > 100:
            ev.append(buf[:T_EVAL])
            buf = []
        elif len(buf) >= T_EVAL:
            buf = []
        if len(ev) == N_EVAL:
            break
    return torch.tensor(train.astype(np.int64)), torch.tensor(ev)


def new_model(seed):
    torch.manual_seed(seed)
    return MiniLM("neox", 4, 4, 128, V, max_pos=512, rot=16).to(DEV)


def train(variant, seed, data):
    m = new_model(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    n = len(data) - L_TRAIN - 2
    for step in range(STEPS):
        st = torch.randint(0, n, (BATCH,), generator=g)
        chunk = torch.stack([data[i:i + L_TRAIN + 1] for i in st.tolist()]).to(DEV)
        if variant == "sink":
            x = torch.cat([torch.full((BATCH, 1), SINK_ID, device=DEV), chunk[:, :L_TRAIN - 1]], 1)   # 입력 128(싱크 + 127바이트)
            y = chunk[:, :L_TRAIN]                                                                    # 대상 128
        else:
            x, y = chunk[:, :L_TRAIN], chunk[:, 1:L_TRAIN + 1]                                    # 입력 128, 대상 128
        lr = LR * min(1.0, (step + 1) / 100) * (0.5 * (1 + np.cos(np.pi * step / STEPS)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        loss = F.cross_entropy(m.head(m.hidden(x)).transpose(1, 2), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return m, float(loss)


@torch.no_grad()
def stream_ppl(m, ev, variant, s):
    x = torch.cat([torch.full((len(ev), 1), SINK_ID), ev], 1) if variant == "sink" else ev
    x = x.to(DEV)
    spec = None if s is None else (s, C - s)
    nll = m.nll(x, spec=spec, bs=8).cpu().numpy()
    pos = np.arange(nll.shape[1])
    per = nll[:, pos >= C].mean(1)                 # t >= C 구간(캐시가 넘친 뒤)
    pre = nll[:, (pos >= 8) & (pos < C)].mean(1)
    rng = np.random.RandomState(0)
    bs = per[rng.randint(0, len(per), (N_BOOT, len(per)))].mean(1)
    return {"ppl_post": float(np.exp(per.mean())), "ppl_post_ci95": [float(np.exp(np.percentile(bs, q))) for q in (2.5, 97.5)],
            "ppl_pre": float(np.exp(pre.mean()))}


def first_token_mass(m, ev, variant):
    """마지막 층까지 평균한 '첫 토큰에 가는 attention 비율'(t=C+63 위치, 전체 attention)을 대략 재서 싱크가 생겼는지 본다."""
    x = (torch.cat([torch.full((len(ev), 1), SINK_ID), ev], 1) if variant == "sink" else ev)[:, :128].to(DEV)
    masses = []

    def attn(q, k, v):
        pos = torch.arange(q.shape[2], device=q.device)
        qr, kr = rope(q, pos, m.rot), rope(k, pos, m.rot)
        sc = (qr @ kr.transpose(-1, -2)) * m.hd ** -0.5
        sc = sc.masked_fill(~(pos[None] <= pos[:, None]), float("-inf"))
        p = torch.softmax(sc, -1)
        masses.append(p[:, :, 64:, 0].mean().item())     # 질의 위치 64 이상에서 첫 토큰이 받는 질량
        return p @ v

    with torch.no_grad():
        h = m.embed(x, torch.arange(x.shape[1], device=DEV))
        for L in m.layers:
            h = m.block(L, h, attn)
    return [round(v, 4) for v in masses]


def main():
    data, ev = load_bytes()
    res = {"config": {"layers": 4, "d": 128, "heads": 4, "rot": 16, "train_len": L_TRAIN, "batch": BATCH, "steps": STEPS,
                      "lr": LR, "cache_C": C, "T_eval": T_EVAL, "n_eval_streams": N_EVAL, "seeds": SEEDS,
                      "train_bytes": int(len(data)), "device": DEV}, "runs": {}}
    t0 = time.time()
    for variant in ("vanilla", "sink"):
        for seed in SEEDS:
            m, loss = train(variant, seed, data)
            r = {"final_train_loss": loss, "first_token_attn_mass_by_layer": first_token_mass(m, ev, variant),
                 "stream": {}}
            for s in (0, 1, 2, 4, 8):
                r["stream"][f"{s}+{C - s}"] = stream_ppl(m, ev, variant, s)
            res["runs"][f"{variant}|seed{seed}"] = r
            print(variant, seed, round(loss, 3), {k: round(v["ppl_post"], 2) for k, v in r["stream"].items()},
                  r["first_token_attn_mass_by_layer"], round(time.time() - t0), "s", flush=True)
    json.dump(res, open(HERE / "sink_token_toy_results.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
