"""Speculative sampling / decoding (Leviathan et al., 2023, Algorithm 1)의 최소 구현과 논문 식.

  - standardize      : logits -> 확률(온도, top-k, T=0은 one-hot). 논문 2.2절의 "표준화 샘플링"
  - speculative_sampling : 토큰 하나짜리 speculative sampling (2.3절, 부록 A.1)
  - speculative_step : Algorithm 1 한 번 (배치 지원, 접두사 길이가 같은 B개 시퀀스)
  - sample_first_k   : 처음 K토큰을 speculative decoding으로 뽑음(길이가 갈라지면 같은 길이끼리 묶어 진행)
  - speculative_decode : 한 시퀀스를 max_new 토큰까지 디코딩 (반복 호출)
  - 식: alpha_from_dists(Thm 3.5), expected_tokens(식 1), walltime_factor(Thm 3.8),
        ops_factor(Thm 3.11), optimal_gamma(3.5절)

모델은 함수로 받는다(같은 길이의 접두사 ids[B, L] -> 확률).
  draft_fn(ids)     -> [B, V]      다음 토큰 분포 q
  target_fn(ids, k) -> [B, k, V]   ids의 마지막 k개 위치에서의 다음 토큰 분포 p (한 번의 병렬 호출)
"""
import math
import torch


# ---------------------------------------------------------------- 표준화 (2.2절)
def standardize(logits, temperature=1.0, top_k=None):
    """logits[..., V] -> 확률. T=0이면 argmax one-hot, top_k가 있으면 상위 k개만 남기고 재정규화."""
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    z = logits / temperature
    if top_k is not None and top_k < z.shape[-1]:
        kth = z.topk(top_k, dim=-1).values[..., -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    return torch.softmax(z, dim=-1)


# ---------------------------------------------------------------- 토큰 하나 (2.3절)
def accept_prob(p, q):
    """beta = sum_x min(p, q) (Thm 3.5). 마지막 축이 어휘."""
    return torch.minimum(p, q).sum(-1)


def residual(p, q, lenience=1.0):
    """p' = norm(max(0, p - l*q)) (2.3절, l=1). p == l*q 라서 분모가 0이면 거절이 일어나지 않으므로 p를 돌려준다."""
    r = (p - lenience * q).clamp_min(0)
    z = r.sum(-1, keepdim=True)
    return torch.where(z > 0, r / z.clamp_min(1e-30), p)


def speculative_sampling(p, q, gen, residual_fn=residual, lenience=1.0):
    """p, q: [B, V]. x ~ q를 뽑아 min(1, p/(l q)) 확률로 받고, 거절하면 residual에서 다시 뽑는다.
    반환: (x[B], accepted[B] bool)"""
    x = torch.multinomial(q, 1, generator=gen).squeeze(-1)
    px, qx = p.gather(-1, x[:, None]).squeeze(-1), q.gather(-1, x[:, None]).squeeze(-1)
    r = torch.rand(x.shape, generator=gen, dtype=p.dtype)
    accepted = r * lenience * qx <= px          # r <= p/(l q)
    y = torch.multinomial(residual_fn(p, q), 1, generator=gen).squeeze(-1)
    return torch.where(accepted, x, y), accepted


# ---------------------------------------------------------------- Algorithm 1
@torch.no_grad()
def speculative_step(draft_fn, target_fn, prefix, gamma, gen):
    """Algorithm 1 (SpeculativeDecodingStep). prefix[B, L] (long).
    반환 dict: n[B] (받아들인 추측 수), x[B, gamma] (추측), t[B] (목표 모델에서 뽑은 마지막 토큰),
               beta[B, gamma] (각 위치의 sum min(p_i, q_i); 진단용), tokens: 길이 n+1 리스트의 리스트."""
    B = prefix.shape[0]
    ids, qs, xs = prefix, [], []
    for _ in range(gamma):                                   # gamma개 추측을 Mq로 자기회귀 샘플링
        q = draft_fn(ids)
        x = torch.multinomial(q, 1, generator=gen)
        qs.append(q); xs.append(x); ids = torch.cat([ids, x], 1)
    ps = target_fn(ids, gamma + 1)                           # Mp 병렬 호출: p_1 .. p_{gamma+1}
    V = ps.shape[-1]
    if gamma == 0:
        t = torch.multinomial(ps[:, 0], 1, generator=gen).squeeze(-1)
        return dict(n=torch.zeros(B, dtype=torch.long), x=prefix[:, :0], t=t,
                    beta=torch.zeros(B, 0), tokens=[[int(v)] for v in t])
    q = torch.stack(qs, 1)                                   # [B, gamma, V]
    x = torch.cat(xs, 1)                                     # [B, gamma]
    px = ps[:, :gamma].gather(2, x[..., None]).squeeze(-1)
    qx = q.gather(2, x[..., None]).squeeze(-1)
    r = torch.rand(B, gamma, generator=gen, dtype=ps.dtype)
    rejected = r * qx > px                                   # r_i > p_i(x_i)/q_i(x_i)
    gam = torch.full((B,), gamma, dtype=torch.long)
    n = torch.where(rejected.any(1), rejected.to(torch.long).argmax(1), gam)   # 첫 거절 위치(없으면 gamma)
    q_pad = torch.cat([q, torch.zeros(B, 1, V, dtype=q.dtype)], 1)              # n = gamma이면 q = 0 -> residual = p
    ar = torch.arange(B)
    p0 = residual(ps[ar, n], q_pad[ar, n])
    t = torch.multinomial(p0, 1, generator=gen).squeeze(-1)
    tokens = [x[b, : n[b]].tolist() + [int(t[b])] for b in range(B)]
    return dict(n=n, x=x, t=t, beta=accept_prob(ps[:, :gamma], q), tokens=tokens)


@torch.no_grad()
def sample_first_k(draft_fn, target_fn, prefix, K, gamma, gen):
    """prefix[B, L]에서 처음 K 토큰을 speculative decoding으로 뽑는다. 반환 [B, K].
    한 번 호출로 1..gamma+1 토큰이 나오므로 시퀀스 길이가 갈라진다. 같은 길이끼리 묶어 한 번씩 더 돈다."""
    B, L = prefix.shape
    out = torch.zeros(B, K, dtype=torch.long)
    have = torch.zeros(B, dtype=torch.long)
    while (have < K).any():
        for m in have[have < K].unique().tolist():
            idx = (have == m).nonzero().squeeze(-1)
            cur = torch.cat([prefix[idx], out[idx, :m]], 1)
            res = speculative_step(draft_fn, target_fn, cur, gamma, gen)
            for j, b in enumerate(idx.tolist()):
                toks = res["tokens"][j][: K - m]
                out[b, m : m + len(toks)] = torch.tensor(toks)
                have[b] = m + len(toks)
    return out


@torch.no_grad()
def speculative_decode(draft_fn, target_fn, prefix, max_new, gamma, gen):
    """prefix[1, L]에서 max_new 토큰을 만든다. 반환 (생성 토큰 리스트, 반복마다의 (n, beta 리스트))."""
    seq, stats = prefix, []
    produced = 0
    while produced < max_new:
        res = speculative_step(draft_fn, target_fn, seq, gamma, gen)
        toks = res["tokens"][0]
        stats.append((int(res["n"][0]), res["beta"][0].tolist()))
        seq = torch.cat([seq, torch.tensor([toks])], 1)
        produced += len(toks)
    return seq[0, prefix.shape[1] : prefix.shape[1] + max_new].tolist(), stats


@torch.no_grad()
def autoregressive_sample(target_fn, prefix, n_tokens, gen):
    """기준선: 목표 모델에서 한 토큰씩 샘플링. prefix[B, L] -> [B, n_tokens]."""
    ids = prefix
    for _ in range(n_tokens):
        p = target_fn(ids, 1)[:, 0]
        ids = torch.cat([ids, torch.multinomial(p, 1, generator=gen)], 1)
    return ids[:, prefix.shape[1]:]


# ---------------------------------------------------------------- 논문의 식
def alpha_from_dists(p, q):
    """alpha = sum_x min(p, q) = 1 - D_LK(p, q) (Lemma 3.3, Thm 3.5)."""
    return accept_prob(p, q)


def d_lk(p, q):
    """D_LK(p, q) = sum_x |p - M|, M = (p+q)/2 (Def 3.2) = TV(p, q)."""
    return (p - (p + q) / 2).abs().sum(-1)


def expected_tokens(alpha, gamma):
    """식 (1): (1 - alpha^(gamma+1)) / (1 - alpha). alpha = 1이면 극한값 gamma + 1."""
    if alpha >= 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def walltime_factor(alpha, gamma, c):
    """Thm 3.8: (1 - alpha^(gamma+1)) / ((1 - alpha)(gamma c + 1))."""
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def ops_factor(alpha, gamma, c_hat):
    """Thm 3.11: 총 연산량 증가 배수 (1 - alpha)(gamma c_hat + gamma + 1) / (1 - alpha^(gamma+1))."""
    return (gamma * c_hat + gamma + 1) / expected_tokens(alpha, gamma)


def optimal_gamma(alpha, c, gamma_max=64):
    """3.5절: walltime_factor를 최대로 하는 정수 gamma(>=1)와 그 값."""
    best = max(range(1, gamma_max + 1), key=lambda g: walltime_factor(alpha, g, c))
    return best, walltime_factor(alpha, best, c)


def capped_geometric_pmf(alpha, gamma):
    """n = 받아들인 추측 수의 분포 (i.i.d. beta 가정): P(n = k) = alpha^k (1-alpha) (k < gamma), P(n = gamma) = alpha^gamma."""
    pmf = [alpha ** k * (1 - alpha) for k in range(gamma)] + [alpha ** gamma]
    return pmf
