"""Transfusion의 혼합 마스크와 두 손실(L = L_LM + λ·L_DDPM)을 한 모델에서 함께 계산하는 최소 토이 (PyTorch, CPU, 수 분).

논문 대응: §3 Transfusion Attention(인과 + 이미지 블록 안 양방향, 그림 4), Training Objective(식 4), Inference(BOI에서 diffusion으로 전환),
          §4.1(캡션 먼저 80%), §4.3.1(양방향 절제), §4.3.4(이미지가 캡션 앞일 때 노이즈 t ≤ 500 제한).
토이이지 논문 재현이 아니다: 클래스 4개, 이미지 = 클래스별 고정 패턴 8패치(차원 4) + 작은 잡음, 2층 Transformer(d=64).

T1 마스크 규칙을 독립 구현 둘(블록 id 방식, big_vision식 cumsum 방식)로 만들어 같은지, 그림 4와 같은 배열을 그린다.
T2 정보 누출 점검: 이미지 앞 텍스트·앞 패치는 뒤 패치를 못 보고, 같은 이미지 패치끼리는 서로 보며, 뒤 텍스트는 이미지를 본다.
T3 한 순전파에서 LM 손실(텍스트 위치)과 DDPM 손실(이미지 위치)을 함께 계산하고 λ에 따른 기울기 크기를 본다.
T4 학습(3 시드): 양방향 / 인과만 / 양방향 + 노이즈 제한. 평가: 이미지 먼저(B)와 캡션 먼저(A)에서의 ε-MSE, 이미지→캡션 정확도(깨끗한 이미지),
   캡션→이미지 생성(LM이 BOI를 내면 DDIM 50스텝으로 전환, 분류 정확도).
출력: mask_loss_results.json
실행: python3 mask_loss_toy.py   (torch 필요, 이 맥에서 CPU로 약 6분)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)
K, NP, DL = 4, 8, 4          # 클래스 수, 이미지당 패치 수, 패치(잠재) 차원
BOS, BOI, EOI, END = 4, 5, 6, 7
V = 8                        # 어휘: 0-3 클래스 단어, BOS, BOI, EOI, END
T_TRAIN = 1000
LAM = 5.0                    # 논문 §4.1: λ = 5


# ---------------------------------------------------------------- 마스크
def block_ids_mask(is_img, blk):
    """블록 id 방식. is_img[i]: 위치 i가 이미지 패치인가, blk[i]: 이미지 번호(텍스트는 -1).
    규칙(논문 §3): 모든 위치는 인과(j<=i), 같은 이미지 안의 패치끼리는 양방향."""
    n = len(is_img)
    m = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            m[i, j] = (j <= i) or (is_img[i] and is_img[j] and blk[i] == blk[j])
    return m


def cumsum_mask(ar):
    """big_vision make_attn_mask: attn[i,j] = cumsum(ar)[j] <= cumsum(ar)[i]. ar=1이면 새 인과 단계를 연다."""
    c = np.cumsum(ar)
    return c[None, :] <= c[:, None]


def layout(kind):
    """kind 'A': [BOS, cap, BOI, img*8, EOI, END] (캡션 먼저), 'B': [BOS, BOI, img*8, EOI, cap, END]."""
    if kind == "A":
        toks = ["BOS", "cap", "BOI"] + ["p"] * NP + ["EOI", "END"]
    else:
        toks = ["BOS", "BOI"] + ["p"] * NP + ["EOI", "cap", "END"]
    is_img = np.array([t == "p" for t in toks])
    blk = np.where(is_img, 0, -1)
    ar = np.array([0 if (t == "p" and toks[i - 1] == "p") else 1 for i, t in enumerate(toks)])  # 이미지 첫 패치부터 뒤 패치는 0
    return toks, is_img, blk, ar


def two_image_layout():
    toks = ["t", "BOI"] + ["p"] * 3 + ["EOI", "t", "BOI"] + ["p"] * 3 + ["EOI", "t"]
    is_img = np.array([t == "p" for t in toks])
    blk = np.full(len(toks), -1)
    blk[2:5], blk[8:11] = 0, 1
    ar = np.array([0 if (t == "p" and toks[i - 1] == "p") else 1 for i, t in enumerate(toks)])
    return toks, is_img, blk, ar


def draw(m, toks):
    w = max(len(t) for t in toks)
    lines = [" " * (w + 1) + " ".join(f"{t:>{w}}" for t in toks)]
    for i, t in enumerate(toks):
        lines.append(f"{t:>{w}} " + " ".join(f"{'#' if m[i, j] else '.':>{w}}" for j in range(len(toks))))
    return lines


results = {}
# T1, T2 -------------------------------------------------------------
t1 = {}
for name, (toks, is_img, blk, ar) in {"A": layout("A"), "B": layout("B"), "two_images": two_image_layout()}.items():
    m1, m2 = block_ids_mask(is_img, blk), cumsum_mask(ar)
    t1[name] = {"identical_block_vs_cumsum": bool((m1 == m2).all()), "n": len(toks), "matrix": draw(m1, toks)}
results["T1_mask_equivalence"] = t1
toks, is_img, blk, ar = two_image_layout()
m = block_ids_mask(is_img, blk)
img0, img1 = np.where(blk == 0)[0], np.where(blk == 1)[0]
chk = {
    "within_image_bidirectional": bool(m[np.ix_(img0, img0)].all() and m[np.ix_(img1, img1)].all()),
    "first_image_cannot_see_second_image": bool(not m[np.ix_(img0, img1)].any()),
    "second_image_sees_first_image_fully": bool(m[np.ix_(img1, img0)].all()),
    "BOI_before_image_cannot_see_patches": bool(not m[1, img0].any() and not m[7, img1].any()),
    "EOI_after_image_sees_all_patches": bool(m[5, img0].all() and m[11, img1].all()),
    "text_after_image_sees_image": bool(m[6, img0].all() and m[12, img1].all()),
    "num_attended_pairs": int(m.sum()), "causal_pairs": int(np.tril(np.ones_like(m)).sum()),
    "extra_pairs_from_bidirectional": int(m.sum() - np.tril(np.ones_like(m)).sum()),
    "extra_pairs_expected": 2 * (3 * 2 // 2),   # 블록 둘, 각 3패치: 위 삼각 3쌍
}
results["T2_info_flow"] = chk
print("T1", {k: v["identical_block_vs_cumsum"] for k, v in t1.items()}, "\n".join(t1["A"]["matrix"]))
print("T2", chk)


# ---------------------------------------------------------------- 모델
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h, s.n1, s.n2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):
        B, N, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # bool: True = 볼 수 있음
        x = x + s.o(a.transpose(1, 2).reshape(B, N, D))
        return x + s.ff(s.n2(x))


class Toy(nn.Module):
    def __init__(s, d=64, h=4, L=2, n=13):
        super().__init__()
        s.emb = nn.Embedding(V, d)                        # 텍스트 임베딩(입력)
        s.pin = nn.Linear(DL, d)                          # 이미지: 선형 패치 인코더(논문 기본 변형)
        s.tmlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))  # timestep 임베딩을 모든 패치에 더함(논문 각주 6)
        s.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        s.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        s.ln = nn.LayerNorm(d)
        s.lm_head = nn.Linear(d, V)                       # 텍스트 출력: 어휘 분포
        s.eps_head = nn.Linear(d, DL)                     # 이미지 출력: 선형 패치 디코더(ε 예측)
        s.d = d

    def temb(s, t):
        half = s.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half) / half)
        a = t[:, None].float() * f[None]
        return s.tmlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(s, toks, patches, t, is_img, mask):
        """toks: [B,N] (이미지 위치는 무시), patches: [B,N,DL] (이미지 위치만 사용), t: [B] 이미지당 timestep."""
        x = s.emb(toks.clamp(max=V - 1))
        px = s.pin(patches) + s.temb(t)[:, None]
        x = torch.where(is_img[None, :, None], px, x) + s.pos[: x.shape[1]]
        for b in s.blocks:
            x = b(x, mask)
        h = s.ln(x)
        return s.lm_head(h), s.eps_head(h)


def cosine_abar(T=T_TRAIN, sft=0.008):
    t = torch.arange(T + 1) / T
    f = torch.cos((t + sft) / (1 + sft) * math.pi / 2) ** 2
    ab = (f / f[0])[1:]
    return ab.clamp(1e-5, 0.9999)                       # 논문: 코사인 일정(Nichol & Dhariwal)


ABAR = cosine_abar()
g0 = torch.Generator().manual_seed(1234)
PATTERN = torch.randn(K, NP, DL, generator=g0)           # 클래스별 이미지 패턴
SIGMA = 0.1


def make_batch(B, gen, p_caption_first=0.8):
    cls = torch.randint(0, K, (B,), generator=gen)
    x0 = PATTERN[cls] + SIGMA * torch.randn(B, NP, DL, generator=gen)
    first = torch.rand(B, generator=gen) < p_caption_first    # True = 캡션 먼저(A)
    return cls, x0, first


def assemble(cls, first):
    """layout A/B 토큰 [B,13]과 이미지 위치 마스크를 만든다(이미지 위치는 두 레이아웃에서 다르다)."""
    B = len(cls)
    toksA = torch.stack([torch.full((B,), BOS), cls, torch.full((B,), BOI)] + [torch.zeros(B, dtype=torch.long)] * NP + [torch.full((B,), EOI), torch.full((B,), END)], 1)
    toksB = torch.stack([torch.full((B,), BOS), torch.full((B,), BOI)] + [torch.zeros(B, dtype=torch.long)] * NP + [torch.full((B,), EOI), cls, torch.full((B,), END)], 1)
    return torch.where(first[:, None], toksA, toksB)


def masks(causal_only):
    out = {}
    for kind in "AB":
        toks, is_img, blk, ar = layout(kind)
        m = np.tril(np.ones((len(toks), len(toks)), bool)) if causal_only else block_ids_mask(is_img, blk)
        out[kind] = (torch.tensor(m), torch.tensor(is_img))
    return out


def run(model, mk, cls, x0, first, t, eps):
    """한 번의 순전파로 두 손실을 함께 계산한다. 레이아웃 A/B는 샘플마다 달라 둘로 나눠 처리하고 평균은 전체 토큰·이미지 기준."""
    B = len(cls)
    xt = ABAR[t].sqrt()[:, None, None] * x0 + (1 - ABAR[t]).sqrt()[:, None, None] * eps
    lm_sum = 0.0
    n_lm = 0
    ddpm = []
    per = {}
    for kind, sel in (("A", first), ("B", ~first)):
        if sel.sum() == 0:
            continue
        mask, is_img = mk[kind]
        toks = assemble(cls[sel], torch.full((int(sel.sum()),), kind == "A"))
        patches = torch.zeros(int(sel.sum()), toks.shape[1], DL)
        patches[:, is_img] = xt[sel]
        logits, epsh = model(toks, patches, t[sel], is_img, mask)
        # LM 손실: 위치 i의 출력이 토큰 i+1(텍스트)을 예측. 입력이 이미지 패치이거나 BOI이면 LM 손실 없음(논문 각주 8: BOI 입력엔 손실 없음)
        tgt = toks[:, 1:]
        inp_is_text = ~is_img[:-1]
        inp_not_boi = toks[:, :-1] != BOI
        tgt_is_text = ~is_img[1:]
        w = (inp_is_text[None] & tgt_is_text[None] & inp_not_boi).float()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt.reshape(-1), reduction="none").view_as(w)
        lm_sum = lm_sum + (ce * w).sum()
        n_lm += float(w.sum())
        # DDPM 손실: 이미지 하나당 ||ε - ε̂||² (원소 평균), 이미지 수로 평균
        d = ((epsh[:, is_img] - eps[sel]) ** 2).mean(dim=(1, 2))
        ddpm.append(d)
        per[kind] = float(d.mean())
    l_lm = lm_sum / n_lm
    l_ddpm = torch.cat(ddpm).mean()
    return l_lm, l_ddpm, per


# T3: 기울기 크기 ------------------------------------------------------
torch.manual_seed(0)
model = Toy()
mk = masks(False)
gen = torch.Generator().manual_seed(0)
cls, x0, first = make_batch(512, gen)
t = torch.randint(0, T_TRAIN, (512,), generator=gen)
eps = torch.randn(x0.shape, generator=gen)
l_lm, l_dd, per = run(model, mk, cls, x0, first, t, eps)
g = {}
for nm, l in (("LM", l_lm), ("DDPM", l_dd)):
    model.zero_grad()
    l.backward(retain_graph=True)
    g[nm] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)))
results["T3_loss_and_gradient_scale_at_init"] = {
    "L_LM": float(l_lm), "L_DDPM": float(l_dd), "L_LM_expected_uniform_V8": math.log(V), "L_DDPM_expected_random_init_approx": 1.0,
    "grad_norm_LM": g["LM"], "grad_norm_DDPM": g["DDPM"], "grad_ratio_DDPM_over_LM": g["DDPM"] / g["LM"],
    "total_with_lambda5": float(l_lm + LAM * l_dd),
}
print("T3", results["T3_loss_and_gradient_scale_at_init"])


# T4: 학습·평가 --------------------------------------------------------
@torch.no_grad()
def evaluate(model, mk, seed):
    model.eval()
    gen = torch.Generator().manual_seed(10_000 + seed)
    N = 4000
    out = {}
    for tt in (100, 500, 900):
        for kind in "AB":
            cls = torch.randint(0, K, (N,), generator=gen)
            x0 = PATTERN[cls] + SIGMA * torch.randn(N, NP, DL, generator=gen)
            first = torch.full((N,), kind == "A")
            t = torch.full((N,), tt)
            eps = torch.randn(x0.shape, generator=gen)
            _, _, per = run(model, mk, cls, x0, first, t, eps)
            out[f"eps_mse_{kind}_t{tt}"] = per[kind]
    # 이미지→캡션(레이아웃 B): 입력 이미지의 노이즈 수준 t = 0(깨끗함, 추론 때의 조건), 300, 700 에서 캡션 정확도와 교차엔트로피
    for tt in (0, 300, 700):
        cls = torch.randint(0, K, (N,), generator=gen)
        x0 = PATTERN[cls] + SIGMA * torch.randn(N, NP, DL, generator=gen)
        mask, is_img = mk["B"]
        toks = assemble(cls, torch.zeros(N, dtype=torch.bool))
        ab = ABAR[tt]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn(x0.shape, generator=gen)
        patches = torch.zeros(N, 13, DL); patches[:, is_img] = xt
        logits, _ = model(toks, patches, torch.full((N,), tt), is_img, mask)
        lg = logits[:, 1 + 1 + NP]                           # EOI 위치의 출력이 캡션을 예측
        out[f"caption_acc_image_t{tt}"] = float((lg[:, :K].argmax(-1) == cls).float().mean())
        out[f"caption_ce_image_t{tt}"] = float(F.cross_entropy(lg, cls))
    # 캡션→이미지(레이아웃 A): LM이 캡션 뒤 BOI를 내는지, 그 뒤 DDIM 50스텝 diffusion으로 8패치를 만든다
    cls = torch.randint(0, K, (N,), generator=gen)
    mask, is_img = mk["A"]
    toks = assemble(cls, torch.ones(N, dtype=torch.bool))
    patches = torch.zeros(N, 13, DL)
    logits, _ = model(toks, patches, torch.zeros(N, dtype=torch.long), is_img, mask)
    out["LM_emits_BOI_after_caption"] = float((logits[:, 1].argmax(-1) == BOI).float().mean())   # 위치 1(캡션 토큰)의 출력이 BOI
    x = torch.randn(N, NP, DL, generator=gen)
    steps = torch.linspace(T_TRAIN - 1, 0, 51).long()
    for a, b in zip(steps[:-1], steps[1:]):
        patches[:, is_img] = x
        _, e = model(toks, patches, torch.full((N,), int(a)), is_img, mask)
        e = e[:, is_img]
        x0h = ((x - (1 - ABAR[a]).sqrt() * e) / ABAR[a].sqrt()).clamp(-3, 3)   # x0 추정을 데이터 범위로 자름(DDPM 구현의 clip_denoised와 같은 관행. 없으면 t≈T에서 오차가 300배로 커짐)
        x = ABAR[b].sqrt() * x0h + (1 - ABAR[b]).sqrt() * e if int(b) > 0 else x0h
    d = ((x[:, None] - PATTERN[None]) ** 2).mean((2, 3))
    out["generation_class_acc"] = float((d.argmin(1) == cls).float().mean())
    out["generation_mse_to_true_pattern"] = float(d[torch.arange(N), cls].mean())
    model.train()
    return out


def train(variant, seed, steps=2500, bs=256):
    torch.manual_seed(seed)
    model = Toy()
    mk = masks(variant["causal"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    hist = []
    for it in range(steps):
        cls, x0, first = make_batch(bs, gen)
        t = torch.randint(0, T_TRAIN, (bs,), generator=gen)
        if variant["noise_limit"]:
            t = torch.where(~first, t.clamp(max=500), t)       # 이미지가 캡션 앞일 때만 t ≤ 500 (논문 §4.3.4)
        eps = torch.randn(x0.shape, generator=gen)
        l_lm, l_dd, _ = run(model, mk, cls, x0, first, t, eps)
        loss = l_lm + LAM * l_dd
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == steps - 1:
            hist.append((it, float(l_lm), float(l_dd)))
    return model, mk, hist


if __name__ == "__main__":
    VARIANTS = {
        "bidirectional": dict(causal=False, noise_limit=False),
        "causal_only": dict(causal=True, noise_limit=False),
        "bidirectional_noise_limit500": dict(causal=False, noise_limit=True),
    }
    runs = {}
    t0 = time.time()
    for name, var in VARIANTS.items():
        runs[name] = []
        for seed in range(3):
            model, mk, hist = train(var, seed)
            ev = evaluate(model, mk, seed)
            ev["train_hist_(it,L_LM,L_DDPM)"] = hist
            runs[name].append(ev)
            print(name, seed, {k: round(v, 4) for k, v in ev.items() if not k.startswith("train")}, f"{time.time()-t0:.0f}s", flush=True)
    results["T4_runs"] = runs
    keys = [k for k in runs["bidirectional"][0] if not k.startswith("train")]
    results["T4_mean_sd_over_3_seeds"] = {n: {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))] for k in keys} for n, rs in runs.items()}
    results["meta"] = {"lambda": LAM, "T_train": T_TRAIN, "steps": 2500, "batch": 256, "d": 64, "layers": 2, "seeds": 3, "device": "cpu", "seconds": time.time() - t0}
    (HERE / "mask_loss_results.json").write_text(json.dumps(results, indent=1))
