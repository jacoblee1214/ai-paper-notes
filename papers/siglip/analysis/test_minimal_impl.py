"""minimal_impl.py(SigLIP sigmoid 손실)의 핵심 성질 시험. CPU에서 수 초.

  python -m pytest analysis/test_minimal_impl.py -q
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as M  # noqa: E402

torch.manual_seed(0)


def rand_emb(n, d, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x = F.normalize(torch.randn(n, d, generator=g, dtype=dtype), dim=-1)
    y = F.normalize(x + 0.8 * torch.randn(n, d, generator=g, dtype=dtype), dim=-1)  # 짝이 어느 정도 가까움
    return x, y


def explicit_sigmoid_loss(x, y, t, b):
    """쌍마다 이중 반복문으로 쓴 정의: -(1/n) sum_i sum_j log(1 / (1 + exp(z_ij (-t x_i.y_j + b)))) (3.2절 식 그대로)."""
    n = x.shape[0]
    tot = 0.0
    for i in range(n):
        for j in range(n):
            z = 1.0 if i == j else -1.0
            tot += -math.log(1.0 / (1.0 + math.exp(z * (-(t * float(x[i] @ y[j])) + b))))
    return tot / n


def test_matches_paper_formula_and_algorithm1():
    """알고리즘 1(logits = t x.y + b)은 3.2절 식을 b의 부호만 뒤집어 쓴 것과 같다.
    3.2절 식 1/(1 + e^{z(-t x.y + b)}) = sigmoid(z(t x.y - b))는 b 부호가 알고리즘 1·코드와 반대다 (04의 D1).
    그대로 읽으면 b=-10 초기화가 로짓을 +10으로 밀어 올려 초기 손실이 폭발한다."""
    x, y = rand_emb(12, 16)
    t, b = 7.3, -3.1
    got = float(M.sigmoid_loss(x, y, t, b))
    assert got == pytest.approx(explicit_sigmoid_loss(x, y, t, -b), rel=1e-10)
    assert abs(got - explicit_sigmoid_loss(x, y, t, b)) > 1.0
    # b = -10, t = 10 에서 식을 글자 그대로 쓰면 초기 손실이 알고리즘 1의 몇 배가 된다
    xr, yr = rand_emb(64, 512)
    assert explicit_sigmoid_loss(xr, yr, 10.0, -10.0) > 50 * float(M.sigmoid_loss(xr, yr, 10.0, -10.0))  # 630.7 대 9.5


@pytest.mark.parametrize("D", [1, 2, 4])
@pytest.mark.parametrize("ckpt", [False, True])
def test_chunked_equals_full_value_and_grads(D, ckpt):
    n = 16
    x, y = rand_emb(n, 24)
    x, y = x.clone().requires_grad_(), y.clone().requires_grad_()
    tp = torch.tensor(1.9, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    full = M.sigmoid_loss(x, y, tp.exp(), b)
    g_full = torch.autograd.grad(full, [x, y, tp, b])
    stats = {}
    ch = M.sigmoid_loss_chunked(x, y, tp.exp(), b, D, use_checkpoint=ckpt, stats=stats)
    g_ch = torch.autograd.grad(ch, [x, y, tp, b])
    assert float(ch.detach()) == pytest.approx(float(full.detach()), rel=1e-12)
    for a, c in zip(g_full, g_ch):
        assert torch.allclose(a, c, atol=1e-12)
    assert stats["max_block"] == (n // D) ** 2  # 한 번에 만드는 로짓 블록은 b x b


def test_pairwise_independent_row_permutation_invariant():
    """쌍마다 독립인 항의 합이라서 (이미지, 텍스트) 쌍을 같은 순열로 섞어도 값이 같다. 텍스트만 섞으면 짝이 깨져 값이 바뀐다."""
    x, y = rand_emb(10, 16)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(3))
    a = M.sigmoid_loss(x, y, 10.0, -10.0)
    assert float(M.sigmoid_loss(x[perm], y[perm], 10.0, -10.0)) == pytest.approx(float(a), rel=1e-12)
    assert abs(float(M.sigmoid_loss(x, y[perm], 10.0, -10.0)) - float(a)) > 1e-3
    # 장치에 어떤 쌍이 배정되는지도 상관없다
    assert float(M.sigmoid_loss_chunked(x[perm], y[perm], 10.0, -10.0, 5)) == pytest.approx(float(a), rel=1e-12)


def test_init_loss_matches_analytic_for_random_embeddings():
    """무작위 임베딩, t=10, b=-10: 모든 로짓이 b 근처(cos ~ N(0, 1/d))라서 L ~ softplus(10) + (B-1) softplus(-10)."""
    n, d = 256, 4096
    g = torch.Generator().manual_seed(1)
    x = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    y = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    got = float(M.sigmoid_loss(x, y, 10.0, -10.0))
    want = M.loss_const_cos(n, 10.0, -10.0, c_pos=0.0, c_neg=0.0)
    assert got == pytest.approx(want, rel=0.02)  # t*cos의 표준편차 0.16에서 오는 2차 보정 약 1%
    # b = 0 이면 모든 쌍의 손실이 ~ log 2 라서 B배 커진다
    got0 = float(M.sigmoid_loss(x, y, 10.0, 0.0))
    assert got0 == pytest.approx(n * math.log(2.0), rel=0.02)
    assert got0 / got > 15  # n=256: 178 대 10.0 (n=32768이면 약 2.2e4 대 11.5)


def test_bias_balance_point_and_initial_bias_gradient():
    for B in (64, 32768):
        bb = M.bias_balance_point(B)
        assert M.bias_grad_const_cos(B, 10.0, bb, c_pos=0.0, c_neg=0.0) == pytest.approx(0.0, abs=1e-9)
    # B = 32768: b=0이면 음성 쪽 기울기가 압도 (~B/2), b=-10이면 균형점 -ln(B-1)=-10.4에 가까워 기울기가 작다
    g0 = M.bias_grad_const_cos(32768, 10.0, 0.0, c_pos=0.0, c_neg=0.0)
    g10 = M.bias_grad_const_cos(32768, 10.0, -10.0, c_pos=0.0, c_neg=0.0)
    assert g0 == pytest.approx(16383.0, rel=1e-3)
    assert abs(g10) < 0.6 and abs(g0) / abs(g10) > 1e4
    assert M.bias_balance_point(32768) == pytest.approx(-10.397, abs=1e-3)


def test_perfect_alignment_drives_loss_to_zero():
    """직교 임베딩(양성 cos 1, 음성 cos 0)에서 b = -t/2 이고 t가 크면 손실이 0으로 간다."""
    B = 8
    e = torch.eye(B, dtype=torch.float64)
    t, b = 60.0, -30.0
    assert float(M.sigmoid_loss(e, e, t, b)) < 1e-9
    assert float(M.sigmoid_loss(e, e, t, b)) == pytest.approx(M.loss_const_cos(B, t, b), rel=1e-6)
    # 같은 임베딩을 한 칸 어긋나게 짝지으면 양성 로짓이 -30이라 손실이 크다
    assert float(M.sigmoid_loss(e, e.roll(1, 0), t, b)) > 10


def test_manual_gradients_match_autograd():
    n, d = 9, 12
    x = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    y = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
    x.data = F.normalize(x.data, dim=-1)
    y.data = F.normalize(y.data, dim=-1)
    tp = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    loss = M.sigmoid_loss(x, y, tp.exp(), b)
    ax, ay, at, ab = torch.autograd.grad(loss, [x, y, tp, b])
    mx, my, mt, mb = M.sigmoid_grads_manual(x.detach(), y.detach(), float(tp.exp().detach()), float(b.detach()))
    assert torch.allclose(ax, mx, atol=1e-12) and torch.allclose(ay, my, atol=1e-12)
    assert float(at) == pytest.approx(float(mt), rel=1e-10) and float(ab) == pytest.approx(float(mb), rel=1e-10)


def test_softmax_loss_matches_explicit_formula():
    x, y = rand_emb(8, 16)
    t = 12.0
    lg = t * x @ y.T
    li = -(lg.diag() - torch.logsumexp(lg, 1)).mean()  # 이미지 -> 텍스트
    lt = -(lg.diag() - torch.logsumexp(lg, 0)).mean()  # 텍스트 -> 이미지
    assert float(M.softmax_loss(x, y, t)) == pytest.approx(float((li + lt) / 2), rel=1e-12)
    # 직교 임베딩의 닫힌 식: log(1 + (B-1) e^{-t})
    e = torch.eye(8, dtype=torch.float64)
    assert float(M.softmax_loss(e, e, 3.0)) == pytest.approx(M.softmax_loss_const_cos(8, 3.0), rel=1e-12)


def test_softmax_is_shift_invariant_sigmoid_is_not():
    """softmax는 한 행의 로짓에 상수를 더해도 같고, sigmoid는 아니다. 그래서 sigmoid에는 편향 b가 필요하다."""
    x, y = rand_emb(8, 16)
    lg = 12.0 * x @ y.T
    lab = torch.arange(8)
    assert float(F.cross_entropy(lg, lab)) == pytest.approx(float(F.cross_entropy(lg + 5.0, lab)), rel=1e-12)
    assert abs(float(M.sigmoid_loss(x, y, 12.0, 0.0)) - float(M.sigmoid_loss(x, y, 12.0, 5.0))) > 1.0


def test_optimal_logit_split_for_orthogonal_embeddings():
    """직교 임베딩에서 b 최적: 양성 로짓 ~ t/2 - ln(B-1)/2, 음성 ~ -t/2 - ln(B-1)/2, 손실 ~ 2 sqrt(B-1) e^{-t/2}."""
    B, t = 1024, 20.0
    o = M.optimal_orthogonal(B, t)
    assert o["b_star"] == pytest.approx(o["approx_b"], abs=0.01)
    assert o["loss"] == pytest.approx(o["approx_loss"], rel=0.01)
    assert o["logit_pos"] > 0 > o["logit_neg"]
    assert o["logit_pos"] + o["logit_neg"] == pytest.approx(-math.log(B - 1), abs=0.02)  # 중점이 -ln(B-1)/2 의 두 배
    # 최적 b에서 손실이 이웃 b보다 작다
    for db in (-0.5, 0.5):
        assert M.loss_const_cos(B, t, o["b_star"] + db) > o["loss"]
    # 음성 수가 늘수록 두 로짓이 함께 내려가고 최소 손실은 sqrt(B-1)로 늘어난다
    o2 = M.optimal_orthogonal(4 * B, t)
    assert o2["logit_neg"] < o["logit_neg"] and o2["loss"] > o["loss"]
    assert o2["loss"] / o["loss"] == pytest.approx(2.0, rel=0.02)


def test_free_embedding_optimization_separates_logits():
    """임베딩을 직접 최적화(t', b 학습)하면 양성 로짓은 양수, 음성 로짓은 음수로 갈린다."""
    torch.manual_seed(0)
    n, d = 16, 32
    xr = torch.randn(n, d, requires_grad=True)
    yr = torch.randn(n, d, requires_grad=True)
    tp = torch.tensor(M.T_INIT_PRIME, requires_grad=True)
    b = torch.tensor(M.B_INIT, requires_grad=True)
    opt = torch.optim.Adam([xr, yr, tp, b], lr=0.05)
    first = None
    for _ in range(300):
        loss = M.sigmoid_loss(F.normalize(xr, dim=-1), F.normalize(yr, dim=-1), tp.exp(), b)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    lg = tp.exp() * F.normalize(xr, dim=-1) @ F.normalize(yr, dim=-1).T + b
    eye = torch.eye(n, dtype=torch.bool)
    assert float(loss) < first / 20
    assert float(lg[eye].min()) > 0 > float(lg[~eye].max())


def test_memory_arithmetic_and_saved_activations():
    m = M.memory_elements(32768, 256)
    assert m["b"] == 128 and m["naive"] == 32768**2 and m["sigmoid_chunk"] == 128**2
    assert m["naive"] // m["sigmoid_chunk"] == 256**2
    assert m["recv_vectors_softmax_two_gathers"] == 2 * m["recv_vectors_sigmoid_permutes"]
    # 일반 autograd로 청크를 쓰면 블록 로짓이 역전파까지 저장돼 합이 전체 행렬과 같고, 체크포인트를 쓰면 저장되지 않는다
    n, D = 32, 4
    x, y = rand_emb(n, 16, dtype=torch.float32)
    x, y = x.clone().requires_grad_(), y.clone().requires_grad_()
    b = n // D
    plain = M.count_saved_block_elements(lambda: M.sigmoid_loss_chunked(x, y, 10.0, -10.0, D), (b, b))
    ckpt = M.count_saved_block_elements(lambda: M.sigmoid_loss_chunked(x, y, 10.0, -10.0, D, use_checkpoint=True), (b, b))
    assert plain >= D * D * b * b and ckpt == 0


def test_toy_training_smoke():
    """토이 모델로 두 손실 모두 3스텝이 돌고 손실이 유한하다 (편향과 온도 상한 규칙 포함)."""
    import toy_train as T
    for name in ("softmax", "sigmoid", "sigmoid_b0"):
        model, curve = T.train(0, name, batch=8, steps=3, log_every=1)
        assert all(math.isfinite(v) for _, v in curve)
        if name == "softmax":
            assert model.logit_bias is None and model.logit_scale.exp().item() <= 100 + 1e-4
        else:
            assert model.logit_bias is not None
