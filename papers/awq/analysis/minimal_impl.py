"""AWQ(Activation-aware Weight Quantization, Lin et al.) 최소 구현 (PyTorch, CPU).

논문 §3: 가중치 W[out, in]는 입력 채널(열) 단위로 s를 곱해 양자화하고, 입력 X는 s로 나눈다.
    s = s_X^alpha  (식 5),   alpha* = argmin || Q(W diag(s)) diag(s)^-1 X - W X ||  (식 4)
여기서는 "층 하나(선형 층 하나)의 출력 오차"를 손실로 쓴다(논문 식 4). 공식 코드는 모듈 출력
(attention, MLP)을 손실로 쓴다(04 D3).

구성
  quantize_group      그룹별 INT-n 균일 양자화(zero-point 있는 비대칭, 공식 코드 pseudo_quantize_tensor와 같은 규칙)
  quantize_sym        논문 식 1의 대칭 absmax 양자화(분석용 토이)
  act_scale           채널별 평균 |x|
  awq_search          alpha 격자 탐색, 손실 기록 반환
  auto_clip           그룹별 가중치 clipping 탐색
  fold_scale          W diag(s), X diag(s)^-1 로 변환(양자화 전 등가성 검사용)
"""
from __future__ import annotations

import math
import torch


# ---------------------------------------------------------------- 양자화 함수
def quantize_group(w: torch.Tensor, n_bit: int, group_size: int = 128, zero_point: bool = True,
                   clip_max: torch.Tensor | None = None) -> torch.Tensor:
    """w[out, in]를 입력 방향 group_size개씩 묶어 시뮬레이션 양자화(dequantize한 값 반환).

    zero_point=True: scale = (max-min)/(2^n-1), zero = round(-min/scale)를 [0, 2^n-1]로 자름.
    clip_max[out, n_group, 1]이 있으면 그룹을 [-clip_max, clip_max]로 먼저 자른다.
    """
    out, cin = w.shape
    gs = group_size if group_size > 0 else cin
    assert cin % gs == 0
    wg = w.reshape(out, cin // gs, gs)
    if clip_max is not None:
        wg = torch.maximum(torch.minimum(wg, clip_max), -clip_max)
    if zero_point:
        mx = wg.amax(dim=-1, keepdim=True)
        mn = wg.amin(dim=-1, keepdim=True)
        qmax = 2 ** n_bit - 1
        scale = (mx - mn).clamp(min=1e-5) / qmax
        zero = (-torch.round(mn / scale)).clamp(0, qmax)
        q = (torch.clamp(torch.round(wg / scale) + zero, 0, qmax) - zero) * scale
    else:
        mx = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        qmax = 2 ** (n_bit - 1) - 1
        scale = mx / qmax
        q = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax) * scale
    return q.reshape(out, cin)


def quantize_sym(w: torch.Tensor, n_bit: int, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """논문 식 1: Q(w) = Delta * Round(w / Delta), Delta = max|w| / 2^(N-1) (그룹별). (Q(w), Delta) 반환."""
    out, cin = w.shape
    gs = group_size if group_size > 0 else cin
    wg = w.reshape(out, cin // gs, gs)
    delta = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 2 ** (n_bit - 1)
    return (delta * torch.round(wg / delta)).reshape(out, cin), delta


# ---------------------------------------------------------------- AWQ
def act_scale(x: torch.Tensor) -> torch.Tensor:
    """x[n, in] -> 채널별 평균 |x|."""
    return x.abs().reshape(-1, x.shape[-1]).mean(0)


def make_scales(x_mean: torch.Tensor, alpha: float) -> torch.Tensor:
    """s = x_mean^alpha, 공식 코드처럼 s / sqrt(max(s) * min(s))로 정규화(전체에 상수를 곱해도 양자화 결과는 같다)."""
    s = x_mean.pow(alpha).clamp(min=1e-4)
    return s / (s.max() * s.min()).sqrt()


def scaled_quantize(w: torch.Tensor, s: torch.Tensor, n_bit: int, group_size: int = 128,
                    clip_max: torch.Tensor | None = None) -> torch.Tensor:
    """Q(W diag(s)) diag(s)^-1. 입력 쪽 diag(s)^-1 X를 앞 연산에 합치는 대신 가중치에 되돌려 놓은 같은 함수."""
    return quantize_group(w * s.view(1, -1), n_bit, group_size, clip_max=clip_max) / s.view(1, -1)


def layer_loss(w: torch.Tensor, wq: torch.Tensor, x: torch.Tensor) -> float:
    """||(Wq - W) x||^2 의 원소 평균 (식 4의 제곱 오차 평균)."""
    return float(((x @ wq.T - x @ w.T) ** 2).mean())


@torch.no_grad()
def awq_search(w: torch.Tensor, x: torch.Tensor, n_bit: int, group_size: int = 128, n_grid: int = 20,
               include_one: bool = False, x_stat: torch.Tensor | None = None):
    """alpha 격자 탐색. w[out,in], x[n,in]. x_stat이 있으면 평균 |x|를 거기서 구한다(탐색 손실은 x로 계산).

    격자: alpha = i / n_grid, i = 0..n_grid-1 (공식 코드). include_one=True면 alpha=1도 넣는다.
    반환: dict(alpha, scales, loss(최적), loss_rtn(alpha=0), history(list[(alpha, loss)]))
    """
    x_mean = act_scale(x if x_stat is None else x_stat)
    alphas = [i / n_grid for i in range(n_grid)] + ([1.0] if include_one else [])
    y_ref = x @ w.T
    hist, best = [], None
    for a in alphas:
        s = make_scales(x_mean, a)
        wq = scaled_quantize(w, s, n_bit, group_size)
        loss = float(((x @ wq.T - y_ref) ** 2).mean())
        hist.append((a, loss))
        if best is None or loss < best[1]:   # 같으면 앞(작은 alpha)을 유지
            best = (a, loss, s)
    return dict(alpha=best[0], scales=best[2], loss=best[1], loss_rtn=hist[0][1], history=hist)


@torch.no_grad()
def auto_clip(w: torch.Tensor, x: torch.Tensor, n_bit: int, group_size: int = 128, n_grid: int = 20,
              max_shrink: float = 0.5, n_sample_tokens: int = 512) -> torch.Tensor:
    """(출력 행, 그룹)마다 clip 상한을 원래 absmax의 (1 - i/n_grid), i < max_shrink*n_grid 중에서 고른다.

    손실은 그 그룹 입력 조각만 쓴 출력 오차 (공식 코드 auto_clip_layer와 같은 정의).
    반환 clip_max[out, n_group, 1].
    """
    out, cin = w.shape
    gs = group_size if group_size > 0 else cin
    ng = cin // gs
    wg = w.reshape(out, ng, gs)
    if x.shape[0] > n_sample_tokens:
        idx = torch.linspace(0, x.shape[0] - 1, n_sample_tokens).long()
        x = x[idx]
    xg = x.reshape(x.shape[0], ng, gs)
    org_out = torch.einsum("tgk,ogk->otg", xg, wg)
    org_max = wg.abs().amax(dim=-1, keepdim=True)
    best_max = org_max.clone()
    min_err = torch.full_like(org_max, 1e9)
    for i in range(int(max_shrink * n_grid)):
        mv = org_max * (1 - i / n_grid)
        q = quantize_group(w, n_bit, group_size, clip_max=mv)
        cur = torch.einsum("tgk,ogk->otg", xg, q.reshape(out, ng, gs))
        err = (cur - org_out).pow(2).mean(dim=1).unsqueeze(-1)   # [out, ng, 1]
        better = err < min_err
        min_err = torch.where(better, err, min_err)
        best_max = torch.where(better, mv, best_max)
    return best_max


def fold_scale(w: torch.Tensor, x: torch.Tensor, s: torch.Tensor):
    """양자화 전 등가 변환: (W diag(s), X diag(s)^-1). (W diag(s))(diag(s)^-1 x) = W x."""
    return w * s.view(1, -1), x / s.view(1, -1)


# ---------------------------------------------------------------- 선택 보호(논문 Table 1)
def salient_channels(kind: str, w: torch.Tensor, x_mean: torch.Tensor, frac: float,
                     gen: torch.Generator | None = None) -> torch.Tensor:
    """FP16으로 둘 입력 채널 색인. kind: act(평균|x| 상위), weight(열 L2 노름 상위), random."""
    cin = w.shape[1]
    k = max(1, math.ceil(frac * cin))
    if kind == "act":
        return torch.topk(x_mean, k).indices
    if kind == "weight":
        return torch.topk(w.norm(dim=0), k).indices
    if kind == "random":
        return torch.randperm(cin, generator=gen)[:k]
    raise ValueError(kind)


def mixed_precision_quantize(w: torch.Tensor, idx: torch.Tensor, n_bit: int, group_size: int = 128) -> torch.Tensor:
    """전체를 RTN 양자화한 뒤 idx 열만 원래 값으로 되돌린다(그룹 범위 계산에는 그 열도 들어간다)."""
    wq = quantize_group(w, n_bit, group_size)
    wq[:, idx] = w[:, idx]
    return wq
