"""FlashAttention(Dao et al., 2022) 최소 구현: 타일링 + online softmax 순전파(Alg. 1/2), 재계산 역전파(Alg. 4),
블록 희소(Alg. 5), 그리고 HBM 읽기·쓰기 원소 수를 세는 시뮬레이터.

이 파일은 이해용 토이다. SRAM은 "원소 M개 크기의 버퍼"라는 모형이고, 블록은 그 크기에 맞춰 나눈다.
HBM 접근은 알고리즘 줄마다 "HBM에서 SRAM으로 올리는 원소 수(read)"와 "SRAM에서 HBM으로 쓰는 원소 수(write)"를
IOCounter에 더해 센다. 수치는 PyTorch로 실제로 계산하므로 출력 O와 기울기를 정답과 비교할 수 있다.
GPU 커널이 아니라서 속도는 의미가 없다(파이썬 루프). 드롭아웃은 구현하지 않았다.

실행 예: python3 -c "import minimal_impl as mi; print(mi.hbm_forward_closed_form(1024, 64, 65536))"
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------------------
# HBM 접근 계수기
# --------------------------------------------------------------------------------------
@dataclass
class IOCounter:
    """HBM과 SRAM 사이를 오간 원소 수. kind는 'qkv'(입력), 'o'(출력·기울기), 'stat'(m, l), 'mat'(N x N 행렬)."""

    reads: dict = field(default_factory=lambda: defaultdict(int))
    writes: dict = field(default_factory=lambda: defaultdict(int))
    peak_sram: int = 0  # 한 반복에서 SRAM에 동시에 올라가는 원소 수의 최댓값

    def read(self, kind: str, n: int):
        self.reads[kind] += int(n)

    def write(self, kind: str, n: int):
        self.writes[kind] += int(n)

    def sram(self, n: int):
        self.peak_sram = max(self.peak_sram, int(n))

    @property
    def total(self) -> int:
        return sum(self.reads.values()) + sum(self.writes.values())


def block_sizes(M: int, d: int) -> tuple[int, int]:
    """Alg. 1 줄 1: Bc = ceil(M/4d), Br = min(Bc, d)."""
    bc = math.ceil(M / (4 * d))
    return bc, min(bc, d)


def sram_footprint(br: int, bc: int, d: int) -> int:
    """Alg. 1 한 반복에서 SRAM에 올라가는 원소 수: K_j, V_j, Q_i, O_i, l_i, m_i, S_ij(제자리에서 P~_ij가 됨)."""
    return 2 * bc * d + 2 * br * d + 2 * br + br * bc


def _causal_block_skip(i0: int, i1: int, j0: int, j1: int) -> bool:
    """블록의 모든 원소가 인과 마스크로 가려지면 True (열 시작이 행 끝보다 뒤)."""
    return j0 > i1 - 1


def _element_mask(i0, i1, j0, j1, causal, device):
    if not causal:
        return None
    rows = torch.arange(i0, i1, device=device)[:, None]
    cols = torch.arange(j0, j1, device=device)[None, :]
    return cols > rows  # True인 곳을 -inf로


# --------------------------------------------------------------------------------------
# 표준 attention (Alg. 0, Alg. 3): N x N 행렬을 HBM에 쓴다
# --------------------------------------------------------------------------------------
def _count_standard_forward(counter, N, d):
    counter.read("qkv", 2 * N * d); counter.write("mat", N * N)      # 줄 1: Q, K 읽고 S 쓰기
    counter.read("mat", N * N); counter.write("mat", N * N)          # 줄 2: S 읽고 P 쓰기
    counter.read("mat", N * N); counter.read("qkv", N * d)           # 줄 3: P, V 읽고
    counter.write("o", N * d)                                        # O 쓰기


def _count_standard_backward(counter, N, d):
    counter.read("mat", N * N); counter.read("o", N * d); counter.write("o", N * d)      # 줄 1: P, dO 읽고 dV
    counter.read("o", N * d); counter.read("qkv", N * d); counter.write("mat", N * N)    # 줄 2: dO, V 읽고 dP
    counter.read("mat", 2 * N * N); counter.write("mat", N * N)                          # 줄 3: P, dP 읽고 dS
    counter.read("mat", N * N); counter.read("qkv", N * d); counter.write("o", N * d)    # 줄 4: dS, K 읽고 dQ
    counter.read("mat", N * N); counter.read("qkv", N * d); counter.write("o", N * d)    # 줄 5: dS, Q 읽고 dK


def standard_forward(Q, K, V, tau=None, causal=False, counter: IOCounter | None = None):
    N, d = Q.shape
    tau = 1.0 / math.sqrt(d) if tau is None else tau
    S = tau * (Q @ K.T)
    if causal:
        S = S.masked_fill(torch.ones(N, N, dtype=torch.bool, device=Q.device).triu(1), float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = P @ V
    if counter is not None:
        _count_standard_forward(counter, N, d)
    return O, P


def standard_backward(Q, K, V, dO, P, tau=None, counter: IOCounter | None = None):
    """Alg. 3. P를 HBM에서 읽는다."""
    N, d = Q.shape
    tau = 1.0 / math.sqrt(d) if tau is None else tau
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - (P * dP).sum(-1, keepdim=True))
    dQ = tau * (dS @ K)
    dK = tau * (dS.T @ Q)
    if counter is not None:
        _count_standard_backward(counter, N, d)
    return dQ, dK, dV


# --------------------------------------------------------------------------------------
# FlashAttention 순전파 (Alg. 1 / Alg. 2 / Alg. 5)
# --------------------------------------------------------------------------------------
def _acc_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def flash_forward(Q, K, V, M: int, tau=None, causal=False, block_mask=None, counter: IOCounter | None = None,
                  compute: bool = True, block_override: tuple[int, int] | None = None, p_round_dtype=None):
    """반환: O (N,d, Q와 같은 dtype), m (N,), l (N,)  [HBM에 남기는 통계, 논문 표기].

    M: SRAM 크기(원소 수). block_mask: (Tr, Tc) bool, False인 블록은 건너뜀(Alg. 5).
    causal: 원소 단위 인과 마스크. 완전히 가려진 블록은 건너뜀(이 건너뜀은 논문 Alg. 1에는 없고 구현의 최적화).
    compute=False: 수치 계산 없이 반복 구조와 HBM 계수만 실행한다(큰 N 훑기용).
    p_round_dtype: 주면 P~_ij를 이 dtype으로 반올림한 뒤 P~V를 곱한다(실제 GPU 커널은 텐서코어 입력용으로 P~를 fp16/bf16으로 내린다. 기본은 반올림 없음).
    """
    N, d = Q.shape
    tau = 1.0 / math.sqrt(d) if tau is None else tau
    Bc, Br = block_sizes(M, d) if block_override is None else block_override
    Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)
    if block_mask is not None:
        assert block_mask.shape == (Tr, Tc)
    dt = _acc_dtype(Q)
    if compute:
        O = torch.zeros(N, d, dtype=dt, device=Q.device)
        l = torch.zeros(N, dtype=dt, device=Q.device)
        m = torch.full((N,), float("-inf"), dtype=dt, device=Q.device)
    for j in range(Tc):
        j0, j1 = j * Bc, min((j + 1) * Bc, N)
        if counter is not None:
            counter.read("qkv", 2 * (j1 - j0) * d)                                  # 줄 6: K_j, V_j
        if compute:
            Kj, Vj = K[j0:j1].to(dt), V[j0:j1].to(dt)
        for i in range(Tr):
            i0, i1 = i * Br, min((i + 1) * Br, N)
            if block_mask is not None and not bool(block_mask[i, j]):
                continue
            if causal and _causal_block_skip(i0, i1, j0, j1):
                continue
            br, bc = i1 - i0, j1 - j0
            if counter is not None:
                counter.read("qkv", br * d)                                         # 줄 8: Q_i
                counter.read("o", br * d)                                           #       O_i
                counter.read("stat", 2 * br)                                        #       l_i, m_i
                counter.sram(sram_footprint(br, bc, d))
                counter.write("o", br * d)                                          # 줄 12: O_i
                counter.write("stat", 2 * br)                                       # 줄 13: l_i, m_i
            if not compute:
                continue
            Qi = Q[i0:i1].to(dt)
            Sij = tau * (Qi @ Kj.T)                                                 # 줄 9
            em = _element_mask(i0, i1, j0, j1, causal, Q.device)
            if em is not None:
                Sij = Sij.masked_fill(em, float("-inf"))
            m_t = Sij.max(dim=1).values                                             # 줄 10
            m_t_safe = torch.where(torch.isinf(m_t), torch.zeros_like(m_t), m_t)
            P_t = torch.exp(Sij - m_t_safe[:, None])
            l_t = P_t.sum(dim=1)
            if p_round_dtype is not None:
                P_t = P_t.to(p_round_dtype).to(dt)
            m_new = torch.maximum(m[i0:i1], m_t)                                    # 줄 11
            m_new_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            a = torch.exp(torch.where(torch.isinf(m[i0:i1]), torch.full_like(m_new, float("-inf")), m[i0:i1]) - m_new_safe)
            b = torch.exp(torch.where(torch.isinf(m_t), torch.full_like(m_new, float("-inf")), m_t) - m_new_safe)
            l_new = a * l[i0:i1] + b * l_t
            O[i0:i1] = (l[i0:i1, None] * a[:, None] * O[i0:i1] + b[:, None] * (P_t @ Vj)) / l_new[:, None]  # 줄 12
            l[i0:i1], m[i0:i1] = l_new, m_new                                       # 줄 13
    if not compute:
        return None, None, None
    return O.to(Q.dtype), m, l


def logsumexp_from_stats(m, l):
    """L = m + log l. 이것 하나로 P = exp(S - L)을 복원할 수 있어 역전파는 (m, l) 두 개 대신 L만 있으면 된다."""
    return m + torch.log(l)


# --------------------------------------------------------------------------------------
# FlashAttention 역전파 (Alg. 4): S, P를 저장하지 않고 블록마다 다시 계산
# --------------------------------------------------------------------------------------
def flash_backward(Q, K, V, O, dO, m, l, M: int, tau=None, causal=False, block_mask=None,
                   counter: IOCounter | None = None, compute: bool = True,
                   block_override: tuple[int, int] | None = None):
    N, d = Q.shape
    tau = 1.0 / math.sqrt(d) if tau is None else tau
    Bc, Br = block_sizes(M, d) if block_override is None else block_override
    Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)
    dt = _acc_dtype(Q)
    if compute:
        dQ = torch.zeros(N, d, dtype=dt, device=Q.device)
        dK = torch.zeros(N, d, dtype=dt, device=Q.device)
        dV = torch.zeros(N, d, dtype=dt, device=Q.device)
        Of, dOf, Qf = O.to(dt), dO.to(dt), Q.to(dt)
        L = logsumexp_from_stats(m, l)
        Dvec = (dOf * Of).sum(-1)                                                   # 식 (4): D_i = dO_i . O_i
    for j in range(Tc):
        j0, j1 = j * Bc, min((j + 1) * Bc, N)
        bc = j1 - j0
        if counter is not None:
            counter.read("qkv", 2 * bc * d)                                         # 줄 7: K_j, V_j
        if compute:
            Kj, Vj = K[j0:j1].to(dt), V[j0:j1].to(dt)
            dKj = torch.zeros(bc, d, dtype=dt, device=Q.device)
            dVj = torch.zeros(bc, d, dtype=dt, device=Q.device)
        for i in range(Tr):
            i0, i1 = i * Br, min((i + 1) * Br, N)
            if block_mask is not None and not bool(block_mask[i, j]):
                continue
            if causal and _causal_block_skip(i0, i1, j0, j1):
                continue
            br = i1 - i0
            if counter is not None:
                counter.read("qkv", br * d)                                         # 줄 10: Q_i
                counter.read("o", 3 * br * d)                                       #        O_i, dO_i, dQ_i
                counter.read("stat", 2 * br)                                        #        l_i, m_i
                counter.sram(2 * bc * d + 2 * bc * d + 4 * br * d + 2 * br + br * bc)  # K,V,dK,dV + Q,O,dO,dQ + l,m + S
                counter.write("o", br * d)                                          # 줄 21: dQ_i
            if not compute:
                continue
            Sij = tau * (Qf[i0:i1] @ Kj.T)                                          # 줄 11
            em = _element_mask(i0, i1, j0, j1, causal, Q.device)
            if em is not None:
                Sij = Sij.masked_fill(em, float("-inf"))
            Pij = torch.exp(Sij - L[i0:i1, None])                                   # 줄 13 (재계산)
            dVj += Pij.T @ dOf[i0:i1]                                               # 줄 16
            dPij = dOf[i0:i1] @ Vj.T                                                # 줄 17
            dSij = Pij * (dPij - Dvec[i0:i1, None])                                 # 줄 20
            dQ[i0:i1] += tau * (dSij @ Kj)                                          # 줄 21
            dKj += tau * (dSij.T @ Qf[i0:i1])                                       # 줄 22
        if counter is not None:
            counter.write("o", 2 * bc * d)                                          # 줄 24: dK_j, dV_j
        if compute:
            dK[j0:j1], dV[j0:j1] = dKj, dVj
    if not compute:
        return None, None, None
    return dQ.to(Q.dtype), dK.to(Q.dtype), dV.to(Q.dtype)


# --------------------------------------------------------------------------------------
# 닫힌 식 (계수기와 대조용)
# --------------------------------------------------------------------------------------
def hbm_standard_forward(N, d):
    return 4 * N * d + 4 * N * N


def hbm_flash_forward_closed_form(N, d, M):
    """Alg. 1 순전파: K,V 한 번(2Nd) + Tc번의 (Q,O 읽기 2Nd, 통계 읽기 2N, O 쓰기 Nd, 통계 쓰기 2N)."""
    bc, _ = block_sizes(M, d)
    tc = math.ceil(N / bc)
    return 2 * N * d + tc * (3 * N * d + 4 * N)


def hbm_flash_backward_closed_form(N, d, M):
    """Alg. 4: K,V 읽기와 dK,dV 쓰기 4Nd + Tc번의 (Q,O,dO,dQ 읽기 4Nd, 통계 2N, dQ 쓰기 Nd)."""
    bc, _ = block_sizes(M, d)
    tc = math.ceil(N / bc)
    return 4 * N * d + tc * (5 * N * d + 2 * N)


hbm_forward_closed_form = hbm_flash_forward_closed_form


def count_flash(N, d, M, block_mask=None, causal=False, backward=True):
    """수치 계산 없이 반복 구조만 돌려 HBM 계수를 얻는다. (forward 계수, backward 계수)."""
    z = torch.empty(N, d)
    cf = IOCounter()
    flash_forward(z, z, z, M, causal=causal, block_mask=block_mask, counter=cf, compute=False)
    cb = None
    if backward:
        cb = IOCounter()
        flash_backward(z, z, z, z, z, None, None, M, causal=causal, block_mask=block_mask, counter=cb, compute=False)
    return cf, cb


def count_standard(N, d):
    """수치 계산 없이 표준 attention의 HBM 계수만 얻는다. (forward 계수, backward 계수)."""
    cf, cb = IOCounter(), IOCounter()
    _count_standard_forward(cf, N, d)
    _count_standard_backward(cb, N, d)
    return cf, cb


def naive_reference(Q, K, V, tau=None, causal=False, block_mask=None, Br=None, Bc=None):
    """정답: float64로 S, P를 모두 만들어 계산한 attention. block_mask를 주면 원소 단위로 펼쳐 적용한다."""
    N, d = Q.shape
    tau = 1.0 / math.sqrt(d) if tau is None else tau
    S = tau * (Q.double() @ K.double().T)
    mask = torch.zeros(N, N, dtype=torch.bool)
    if causal:
        mask |= torch.ones(N, N, dtype=torch.bool).triu(1)
    if block_mask is not None:
        full = block_mask.repeat_interleave(Br, 0).repeat_interleave(Bc, 1)[:N, :N]
        mask |= ~full
    S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return P @ V.double()
