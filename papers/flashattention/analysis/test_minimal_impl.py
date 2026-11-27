"""minimal_impl.py 성질 시험. 실행: python -m pytest analysis/test_minimal_impl.py -q  (CPU, 약 30초)"""
import math

import pytest
import torch

import minimal_impl as mi

torch.set_num_threads(4)


def qkv(N, d, seed=0, dtype=torch.float64, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return tuple((scale * torch.randn(N, d, generator=g, dtype=torch.float64)).to(dtype) for _ in range(3))


# ---- 정확성: 순전파 --------------------------------------------------------------------------


@pytest.mark.parametrize("N,d,M", [(64, 16, 256), (100, 8, 64), (128, 32, 4096), (97, 16, 200), (50, 4, 16)])
def test_forward_matches_naive_float64(N, d, M):
    Q, K, V = qkv(N, d, seed=N)
    O, m, l = mi.flash_forward(Q, K, V, M)
    assert torch.allclose(O, mi.naive_reference(Q, K, V), atol=1e-12, rtol=0)


def test_forward_independent_of_block_size():
    """블록 크기는 IO만 바꾸고 결과는 (반올림 오차 안에서) 같다."""
    Q, K, V = qkv(96, 16, seed=1)
    outs = [mi.flash_forward(Q, K, V, 0, block_override=b)[0] for b in [(1, 1), (7, 5), (16, 16), (96, 96), (13, 96)]]
    for o in outs[1:]:
        assert torch.allclose(o, outs[0], atol=1e-12)


def test_forward_causal_matches_naive():
    Q, K, V = qkv(80, 16, seed=2)
    O, _, _ = mi.flash_forward(Q, K, V, 256, causal=True)
    assert torch.allclose(O, mi.naive_reference(Q, K, V, causal=True), atol=1e-12)


def test_stats_give_logsumexp():
    """m + log l 은 각 행의 logsumexp와 같다. 역전파가 이것 하나로 P를 되살릴 수 있는 이유."""
    Q, K, V = qkv(64, 16, seed=3)
    _, m, l = mi.flash_forward(Q, K, V, 256)
    S = (Q @ K.T) / math.sqrt(16)
    assert torch.allclose(mi.logsumexp_from_stats(m, l), torch.logsumexp(S, dim=1), atol=1e-12)


def test_large_logits_stable():
    """max-shift 덕에 로짓이 수백이어도 overflow 없이 정답과 같다."""
    Q, K, V = qkv(64, 16, seed=4, scale=12.0)  # 로짓 크기 약 36 * 4 = 수백
    S = (Q @ K.T) / 4
    assert S.abs().max() > 100
    O, _, _ = mi.flash_forward(Q, K, V, 128)
    assert torch.isfinite(O).all()
    assert torch.allclose(O, mi.naive_reference(Q, K, V), atol=1e-10)


def test_float32_and_float16_tolerance():
    Q, K, V = qkv(128, 32, seed=5, dtype=torch.float32)
    ref = mi.naive_reference(Q, K, V)
    O32 = mi.flash_forward(Q, K, V, 1024)[0]
    assert (O32.double() - ref).abs().max() < 1e-5
    Q16, K16, V16 = (x.half() for x in (Q, K, V))
    ref16 = mi.naive_reference(Q16, K16, V16)  # 같은 fp16 입력에서의 정답
    O16 = mi.flash_forward(Q16, K16, V16, 1024)[0]
    assert O16.dtype == torch.float16
    assert (O16.double() - ref16).abs().max() < 2e-3  # fp16 출력 반올림(2^-11 정도) 수준


# ---- 정확성: 역전파 --------------------------------------------------------------------------


def autograd_grads(Q, K, V, dO, causal=False, block_mask=None, Br=None, Bc=None):
    Qg, Kg, Vg = (x.clone().requires_grad_(True) for x in (Q, K, V))
    O = mi.naive_reference(Qg, Kg, Vg, causal=causal, block_mask=block_mask, Br=Br, Bc=Bc)
    O.backward(dO)
    return Qg.grad, Kg.grad, Vg.grad


@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_autograd(causal):
    N, d, M = 90, 16, 256
    Q, K, V = qkv(N, d, seed=6)
    dO = torch.randn(N, d, generator=torch.Generator().manual_seed(60), dtype=torch.float64)
    O, m, l = mi.flash_forward(Q, K, V, M, causal=causal)
    dQ, dK, dV = mi.flash_backward(Q, K, V, O, dO, m, l, M, causal=causal)
    rQ, rK, rV = autograd_grads(Q, K, V, dO, causal=causal)
    for a, b in [(dQ, rQ), (dK, rK), (dV, rV)]:
        assert torch.allclose(a, b, atol=1e-11)


def test_backward_matches_standard_backward():
    N, d = 64, 8
    Q, K, V = qkv(N, d, seed=7)
    dO = torch.randn(N, d, generator=torch.Generator().manual_seed(70), dtype=torch.float64)
    O, P = mi.standard_forward(Q, K, V)
    ref = mi.standard_backward(Q, K, V, dO, P)
    _, m, l = mi.flash_forward(Q, K, V, 128)
    got = mi.flash_backward(Q, K, V, O, dO, m, l, 128)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-11)


def test_backward_does_not_need_P_stored():
    """역전파에 쓰이는 HBM 입력은 Q,K,V,O,dO,m,l뿐(N x N 항목이 없다)."""
    c = mi.IOCounter()
    mi.flash_backward(*[torch.zeros(64, 8)] * 5, None, None, 128, counter=c, compute=False)
    assert c.reads.get("mat", 0) == 0 and c.writes.get("mat", 0) == 0


# ---- 블록 희소 (Alg. 5) ----------------------------------------------------------------------


def test_block_sparse_matches_masked_naive():
    N, d, M = 96, 8, 128
    Br, Bc = mi.block_sizes(M, d)[1], mi.block_sizes(M, d)[0]
    Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)
    g = torch.Generator().manual_seed(8)
    mask = torch.rand(Tr, Tc, generator=g) < 0.4
    mask[torch.arange(Tr), (torch.arange(Tr) * Tc // Tr)] = True  # 모든 행 블록에 최소 한 블록
    Q, K, V = qkv(N, d, seed=8)
    O, m, l = mi.flash_forward(Q, K, V, M, block_mask=mask)
    ref = mi.naive_reference(Q, K, V, block_mask=mask, Br=Br, Bc=Bc)
    assert torch.allclose(O, ref, atol=1e-12)
    dO = torch.randn(N, d, generator=torch.Generator().manual_seed(80), dtype=torch.float64)
    got = mi.flash_backward(Q, K, V, O, dO, m, l, M, block_mask=mask)
    for a, b in zip(got, autograd_grads(Q, K, V, dO, block_mask=mask, Br=Br, Bc=Bc)):
        assert torch.allclose(a, b, atol=1e-11)


# ---- HBM 접근 수 ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,d,M", [(256, 16, 1024), (300, 8, 256), (512, 64, 16384), (128, 32, 4096)])
def test_forward_counter_matches_closed_form(N, d, M):
    cf, cb = mi.count_flash(N, d, M)
    assert cf.total == mi.hbm_flash_forward_closed_form(N, d, M)
    assert cb.total == mi.hbm_flash_backward_closed_form(N, d, M)


def test_counter_same_with_and_without_compute():
    Q, K, V = qkv(64, 8, seed=9)
    c1, c2 = mi.IOCounter(), mi.IOCounter()
    mi.flash_forward(Q, K, V, 128, counter=c1, compute=True)
    mi.flash_forward(Q, K, V, 128, counter=c2, compute=False)
    assert c1.total == c2.total and dict(c1.reads) == dict(c2.reads)


def test_standard_counter_theta_nd_plus_n2():
    cf, cb = mi.count_standard(512, 64)
    assert cf.total == mi.hbm_standard_forward(512, 64) == 4 * 512 * 64 + 4 * 512 * 512
    assert cb.writes["mat"] + cb.reads["mat"] == 7 * 512 * 512  # Alg. 3은 N x N 행렬을 7번 오간다


def test_theorem2_scaling_exponents():
    """HBM 접근 ~ N^2 d^2 / M: N, d, M 각각 각각 두 배(나머지는 고정)일 때의 비가 (4, 4, 1/2)에 가깝다 (lower-order 항 때문에 근사)."""
    base = mi.count_flash(1024, 32, 16384, backward=False)[0].total
    assert mi.count_flash(2048, 32, 16384, backward=False)[0].total / base == pytest.approx(4, rel=0.05)
    assert mi.count_flash(1024, 64, 16384, backward=False)[0].total / base == pytest.approx(4, rel=0.15)
    assert mi.count_flash(1024, 32, 32768, backward=False)[0].total / base == pytest.approx(0.5, rel=0.1)


def test_block_sparse_hbm_linear_in_s():
    N, d, M = 512, 16, 1024
    Bc, Br = mi.block_sizes(M, d)
    Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)
    totals = []
    for s in (0.25, 0.5, 1.0):
        g = torch.Generator().manual_seed(11)
        mask = torch.rand(Tr, Tc, generator=g) < s
        mask[0, 0] = True
        totals.append((mask.float().mean().item(), mi.count_flash(N, d, M, block_mask=mask, backward=False)[0].total))
    # 선형: a = 2Nd + nnz * Br*(3d+4)  (정확한 식)
    for s, a in totals:
        nnz = round(s * Tr * Tc)
        assert a == 2 * N * d + nnz * Br * (3 * d + 4)


def test_sram_footprint_over_budget_is_constant_factor():
    """Alg. 1의 블록 크기는 SRAM을 M의 상수 배(<=1.5)까지 쓴다 - Theta 안에서는 무해하지만 M을 엄밀한 한계로 읽으면 안 된다."""
    worst = 0
    for d in (16, 32, 64, 128):
        for M in (4 * d, 8 * d, 4 * d * d, 8 * d * d, 4096, 65536, 98304):
            bc, br = mi.block_sizes(M, d)
            worst = max(worst, mi.sram_footprint(br, bc, d) / M)
    assert worst <= 1.5
