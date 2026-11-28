"""GPTQ 최소 구현(선형층 하나)과 비교 대상(RTN, 순차 OBQ).

식 번호는 Frantar et al., "GPTQ" (arXiv:2210.17323)의 번호다.
  - 층 목적함수(식 1): argmin_Ŵ ||WX - ŴX||²,  H = 2 X Xᵀ (X: d_col x 토큰 수)
  - OBQ 한 단계(식 2): w_q 하나를 격자에 맞추고 남은 가중치를 δ = -(w_q - quant(w_q)) / [H_F⁻¹]_qq · (H_F⁻¹)[:,q] 만큼 옮김
  - H_F⁻¹ 갱신(식 3): 가우스 소거 한 단계
  - GPTQ: 모든 행이 같은 열 순서를 쓰고(Step 1), B열씩 묶어 늦게 갱신하고(Step 2, 식 4-5),
          H⁻¹의 Cholesky 위삼각 인자를 한 번 계산해 쓴다(Step 3, 알고리즘 1) + 대각 dampening λ.

규약: W는 (d_row, d_col). 활성 X는 (토큰 수 n, d_col)로 받고 H = 2/n · XᵀX (n으로 나눈 것은 척도일 뿐 argmin을 바꾸지 않는다).
격자는 논문과 같은 행별 비대칭 min-max(0을 포함)이며 그룹 크기를 주면 열 g개마다 행별로 따로 잡는다.
"""
import torch


# ---------- 양자화 격자 ----------
def grid_params(x, bits):
    """행별 비대칭 min-max 격자. x: (rows, k) -> scale, zero: (rows, 1)."""
    maxq = 2 ** bits - 1
    xmin = x.min(1, keepdim=True).values.clamp(max=0)
    xmax = x.max(1, keepdim=True).values.clamp(min=0)
    flat = (xmin == 0) & (xmax == 0)
    xmin = torch.where(flat, -torch.ones_like(xmin), xmin)
    xmax = torch.where(flat, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    return scale, zero


def quantize(x, scale, zero, bits):
    maxq = 2 ** bits - 1
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def rtn(W, bits, group=None):
    """가장 가까운 격자점으로 반올림(데이터를 보지 않음)."""
    d_col = W.shape[1]
    if group is None:
        s, z = grid_params(W, bits)
        return quantize(W, s, z, bits)
    Q = torch.empty_like(W)
    for i in range(0, d_col, group):
        blk = W[:, i:i + group]
        s, z = grid_params(blk, bits)
        Q[:, i:i + group] = quantize(blk, s, z, bits)
    return Q


# ---------- 목적함수 ----------
def hessian(X):
    """H = 2/n XᵀX, X: (n, d_col)."""
    X = X.reshape(-1, X.shape[-1])
    return 2.0 / X.shape[0] * (X.T @ X)


def layer_error(W, Wq, X):
    """식 1의 값을 토큰 수로 나눈 것: mean_n ||(W - Ŵ) x_n||²."""
    X = X.reshape(-1, X.shape[-1])
    D = (W - Wq) @ X.T
    return (D ** 2).sum().item() / X.shape[0]


def dampened(H, damp):
    """H + λI, λ = damp · mean(diag H)."""
    lam = damp * torch.diag(H).mean()
    return H + lam * torch.eye(H.shape[0], dtype=H.dtype)


# ---------- 순차 OBQ (비교 기준: 느리지만 식 2를 그대로 씀) ----------
def obq_fixed_order(W, H, bits, order, damp=0.01, group=None):
    """열 순서 `order`를 고정하고 식 2로 한 열씩 양자화. 각 단계에서 H_F를 새로 역행렬 계산한다(식 3 미사용).
    행은 서로 독립이라 행마다 따로 푼다. 그룹 격자는 그룹 첫 열에 이르렀을 때 '현재 갱신된' 가중치로 잡는다."""
    Hd = dampened(H, damp)
    d_row, d_col = W.shape
    order = list(order)
    Q = W.clone()
    base_s, base_z = grid_params(W, bits)
    for r in range(d_row):
        w = W[r].clone()
        s, z = base_s[r:r + 1], base_z[r:r + 1]
        for step, q in enumerate(order):
            F = order[step:]
            if group is not None and step % group == 0:
                cols = order[step:step + group]
                s, z = grid_params(w[cols].unsqueeze(0), bits)
            Hinv = torch.linalg.inv(Hd[F][:, F])
            wq = quantize(w[q].reshape(1, 1), s, z, bits).item()
            delta = -(w[q] - wq) / Hinv[0, 0] * Hinv[:, 0]
            w[F] += delta
            w[q] = wq
        Q[r] = w
    return Q


def obq_greedy(W, H, bits, damp=0.01, orders=None):
    """원래 OBQ의 탐욕 순서: 행마다 오차가 가장 작은 가중치부터(식 2의 argmin). 식 3으로 H⁻¹를 줄여 간다.
    OBQ 논문의 보조 휴리스틱(이상치 먼저 반올림 등)은 넣지 않았다. orders가 리스트면 행별 양자화 순서를 담는다."""
    Hd = dampened(H, damp)
    d_row, d_col = W.shape
    Q = W.clone()
    base_s, base_z = grid_params(W, bits)
    for r in range(d_row):
        w = W[r].clone()
        s, z = base_s[r:r + 1], base_z[r:r + 1]
        F = list(range(d_col))
        Hinv = torch.linalg.inv(Hd)
        picked = []
        for _ in range(d_col):
            wF = w[F]
            qF = quantize(wF.unsqueeze(0), s, z, bits).squeeze(0)
            cost = (qF - wF) ** 2 / torch.diagonal(Hinv)
            j = int(torch.argmin(cost))
            q = F[j]
            picked.append(q)
            delta = -(wF[j] - qF[j]) / Hinv[j, j] * Hinv[:, j]
            w[F] += delta
            w[q] = qF[j]
            keep = [i for i in range(len(F)) if i != j]
            Hinv = Hinv - torch.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]  # 식 3
            Hinv = Hinv[keep][:, keep]
            F = [F[i] for i in keep]
        Q[r] = w
        if orders is not None:
            orders.append(picked)
    return Q


# ---------- GPTQ ----------
def gptq(W, H, bits, block=128, damp=0.01, group=None, act_order=False, stale_group_params=False):
    """알고리즘 1. 반환: (Q, predicted_loss).

    act_order: 열을 diag(H) 내림차순으로 양자화(논문 본문에는 없고 공식 저장소에 나중에 추가된 휴리스틱).
    stale_group_params: True이면 그룹 격자를 '블록 시작 시점'의 W에서 잡는다(공식 저장소 코드의 방식;
        그룹 크기가 블록 크기보다 작을 때만 블록 안 갱신이 빠진 값이 된다). 기본 False는 현재 갱신된 값.
    predicted_loss: Σ (w - q)² / [U_jj]² / 2 (OBS가 예측한 목적함수 증가량, dampening 포함 목적함수에 대한 값).
    """
    W = W.clone()
    H = H.clone()
    d_row, d_col = W.shape
    dead = torch.diag(H) == 0
    idx = torch.arange(d_col)
    H[idx[dead], idx[dead]] = 1
    W[:, dead] = 0
    s0, z0 = grid_params(W, bits)

    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    H = dampened(H, damp)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)  # H⁻¹ = Uᵀ U ; U[j, j:] 는 H_F⁻¹의 j행(대각부터)을 √[H_F⁻¹]_jj 로 나눈 것

    Q = torch.zeros_like(W)
    loss = 0.0
    s, z = s0, z0
    for i1 in range(0, d_col, block):
        i2 = min(i1 + block, d_col)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if group is not None and (i1 + i) % group == 0:
                src = W[:, i1 + i:i1 + i + group] if stale_group_params else \
                    torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :group]
                s, z = grid_params(src, bits)
            w = W1[:, i]
            d = U1[i, i]
            q = quantize(w.unsqueeze(1), s, z, bits).squeeze(1)
            Q1[:, i] = q
            loss += (((w - q) / d) ** 2).sum().item() / 2
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ U1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]  # 식 4의 지연 갱신
    if act_order:
        Q = Q[:, invperm]
    return Q, loss
