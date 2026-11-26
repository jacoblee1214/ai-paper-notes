"""합성 도형 토이 학습 루프: softmax(CLIP) 손실과 sigmoid(SigLIP) 손실을 같은 데이터·모델·옵티마이저로 비교한다.
논문 재현이 아니라 이해용 토이다. 데이터와 모델은 toy_data_model.py(clip/ 토이 복사본), 손실은 minimal_impl.py.

학습 설정은 clip/analysis/minimal_impl.py의 train()과 같다: AdamW(lr 2e-3, wd 0.01), OneCycleLR(pct_start 0.1), 시드마다 데이터 생성기 시드 1000+seed.
"""
from __future__ import annotations

import math
import time

import torch

import minimal_impl as L
import toy_data_model as D

LOSSES = {
    # 손실 이름: (t' 초기값, b 초기값, t' 상한)
    "softmax": (D.LOGIT_SCALE_INIT, None, D.LOGIT_SCALE_MAX),  # CLIP 기본: t 초기 1/0.07, 상한 100
    "sigmoid": (L.T_INIT_PRIME, L.B_INIT, None),               # SigLIP 기본: t 초기 10, b 초기 -10, 상한 없음
    "sigmoid_b0": (L.T_INIT_PRIME, 0.0, None),                 # 4.9절 표 4의 b=0 초기화 절제
}


def class_words(c: int, s: int, template: str) -> list[str]:
    return D.caption_words(template, [(c, s)])


@torch.no_grad()
def zero_shot_acc(model: D.TinyDual, img: torch.Tensor, y: torch.Tensor, template: str = "photo") -> float:
    """클래스(색 * NS + 모양) 64개의 이름 프롬프트로 만든 제로샷 분류기의 정확도."""
    ws = []
    for c in range(D.NC):
        for s in range(D.NS):
            ws.append(model.encode_text(D._tokens([class_words(c, s, template)])).squeeze(0))
    W = torch.stack(ws, dim=1)
    return float(((model.encode_image(img) @ W).argmax(-1) == y).float().mean())


@torch.no_grad()
def evaluate(model: D.TinyDual, seed: int, n: int = 2000) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(777_000 + seed)
    out = {}
    for name, ho in (("held_out", True), ("seen", False)):
        img, y = D.make_single_eval(n, gen, ho)
        out[f"zs_{name}_photo"] = zero_shot_acc(model, img, y, "photo")
        out[f"zs_{name}_bare"] = zero_shot_acc(model, img, y, "bare")
    # 새 배치 128쌍에서 양성·음성 로짓의 평균 (sigmoid 최적에서 갈라지는지 보기)
    img, text, _ = D.make_batch(128, gen)
    ie, te = model.encode_image(img), model.encode_text(text)
    t = model.logit_scale.exp()
    b = model.logit_bias if model.logit_bias is not None else torch.zeros(())
    lg = t * ie @ te.T + b
    eye = torch.eye(128, dtype=torch.bool)
    out["logit_pos_mean"] = float(lg[eye].mean())
    out["logit_neg_mean"] = float(lg[~eye].mean())
    out["cos_pos_mean"] = float((ie @ te.T)[eye].mean())
    out["cos_neg_mean"] = float((ie @ te.T)[~eye].mean())
    out["t"] = float(t)
    out["b"] = float(model.logit_bias) if model.logit_bias is not None else None
    return out


def train(seed: int, loss: str, batch: int, steps: int, lr: float = 2e-3, device: str = "cpu", log_every: int = 200):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    t_init, b_init, t_max = LOSSES[loss]
    model = D.TinyDual(t_init=t_init, b_init=b_init, t_max=t_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    curve = []
    for step in range(steps):
        img, text, _ = D.make_batch(batch, gen)
        ie, te = model(img.to(device), text.to(device))
        t = model.logit_scale.exp()
        if loss == "softmax":
            l = L.softmax_loss(ie, te, t)
        else:
            l = L.sigmoid_loss(ie, te, t, model.logit_bias)
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        sched.step()
        model.clamp_logit_scale()
        if step % log_every == 0 or step == steps - 1:
            curve.append((step, l.item()))
    return model, curve


def run_one(seed: int, loss: str, batch: int, steps: int, threads: int = 1) -> dict:
    torch.set_num_threads(threads)
    t0 = time.time()
    model, curve = train(seed, loss, batch, steps)
    ev = evaluate(model, seed)
    ev.update(seed=seed, loss=loss, batch=batch, steps=steps, examples_seen=batch * steps,
              train_seconds=round(time.time() - t0, 1), curve=curve)
    return ev
