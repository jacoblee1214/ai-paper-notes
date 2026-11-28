import pytest
import torch

from minimal_impl import (act_scale, auto_clip, awq_search, fold_scale, layer_loss, make_scales,
                          mixed_precision_quantize, quantize_group, quantize_sym, salient_channels,
                          scaled_quantize)


def _layer(seed=0, out=64, cin=256, n=512, n_out=4, boost=20.0, dtype=torch.float32):
    """입력 채널 일부(n_out개)의 활성이 크고 가중치는 정규분포인 층."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out, cin, generator=g, dtype=dtype) * 0.05
    ch = torch.ones(cin, dtype=dtype)
    hot = torch.randperm(cin, generator=g)[:n_out]
    ch[hot] = boost
    x = torch.randn(n, cin, generator=g, dtype=dtype) * ch
    return w, x, hot


def test_fold_scale_is_exact_in_float64():
    w, x, _ = _layer(dtype=torch.float64)
    s = make_scales(act_scale(x), 0.5)
    w2, x2 = fold_scale(w, x, s)
    assert (x2 @ w2.T - x @ w.T).abs().max() < 1e-12


def test_alpha0_is_rtn():
    w, x, _ = _layer()
    s0 = make_scales(act_scale(x), 0.0)
    assert torch.equal(s0, torch.ones_like(s0))
    assert torch.equal(scaled_quantize(w, s0, 4, 128), quantize_group(w, 4, 128))
    r = awq_search(w, x, 4, 128)
    assert r["history"][0][0] == 0.0
    assert r["history"][0][1] == pytest.approx(layer_loss(w, quantize_group(w, 4, 128), x), rel=1e-6)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n_bit", [3, 4])
def test_awq_loss_not_worse_than_rtn(seed, n_bit):
    w, x, _ = _layer(seed)
    r = awq_search(w, x, n_bit, 128)
    assert r["loss"] <= r["loss_rtn"] + 1e-12
    # 다시 계산한 손실과 기록이 같다
    wq = scaled_quantize(w, r["scales"], n_bit, 128)
    assert layer_loss(w, wq, x) == pytest.approx(r["loss"], rel=1e-5)


def test_awq_strictly_better_when_salient_channels_exist():
    w, x, _ = _layer(1, boost=30.0)
    r = awq_search(w, x, 3, 128)
    assert r["alpha"] > 0 and r["loss"] < 0.8 * r["loss_rtn"]


def test_group_levels_and_error_bound():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, 256, generator=g)
    for n_bit in (3, 4):
        q = quantize_group(w, n_bit, 128)
        for r in range(8):
            for gi in range(2):
                assert q[r, gi * 128:(gi + 1) * 128].unique().numel() <= 2 ** n_bit
        wg = w.reshape(8, 2, 128)
        step = (wg.amax(-1) - wg.amin(-1)) / (2 ** n_bit - 1)
        err = (q - w).reshape(8, 2, 128).abs().amax(-1)
        assert (err <= step * 0.5 + step + 1e-6).all()  # 반올림 step/2, zero 반올림으로 최대 step/2 더


def test_global_rescale_invariance():
    w, x, _ = _layer()
    s = make_scales(act_scale(x), 0.6)
    a = scaled_quantize(w, s, 4, 128)
    b = scaled_quantize(w, s * 8.0, 4, 128)
    assert (a - b).abs().max() < 1e-5


def test_auto_clip_not_worse_per_group():
    w, x, _ = _layer(2, out=16, cin=256, n=256)
    cm = auto_clip(w, x, 3, 128, n_sample_tokens=256)
    out, ng = 16, 2
    xg = x.reshape(-1, ng, 128)
    ref = torch.einsum("tgk,ogk->otg", xg, w.reshape(out, ng, 128))
    e0 = (torch.einsum("tgk,ogk->otg", xg, quantize_group(w, 3, 128).reshape(out, ng, 128)) - ref).pow(2).mean(1)
    e1 = (torch.einsum("tgk,ogk->otg", xg, quantize_group(w, 3, 128, clip_max=cm).reshape(out, ng, 128)) - ref).pow(2).mean(1)
    assert (e1 <= e0 + 1e-10).all()
    assert (cm <= w.reshape(out, ng, 128).abs().amax(-1, keepdim=True) + 1e-9).all()


def test_salient_selection_and_mixed_precision():
    w, x, hot = _layer(3, boost=50.0)
    xm = act_scale(x)
    idx = salient_channels("act", w, xm, 4 / 256)
    assert set(idx.tolist()) == set(hot.tolist())
    wq = mixed_precision_quantize(w, idx, 3, 128)
    assert torch.equal(wq[:, idx], w[:, idx])
    other = torch.ones(256, dtype=torch.bool)
    other[idx] = False
    assert not torch.equal(wq[:, other], w[:, other])


def test_error_ratio_is_one_over_s_when_delta_unchanged():
    """논문 식 2: Delta' = Delta이면 정규화한 오차 비가 1/s. 첫 열(작은 값)만 s배."""
    g = torch.Generator().manual_seed(0)
    out = 20000
    w = torch.randn(out, 128, generator=g)
    w[:, 0] *= 0.2                      # 이 열은 작아서 s를 곱해도 그룹 최댓값이 안 바뀐다
    base, _ = quantize_sym(w, 4, 128)
    e0 = (base[:, 0] - w[:, 0]).abs().mean()
    for s in (1.5, 2.0):
        ws = w.clone()
        ws[:, 0] *= s
        q, _ = quantize_sym(ws, 4, 128)
        e1 = ((q[:, 0] / s) - w[:, 0]).abs().mean()
        ratio_delta = quantize_sym(ws, 4, 128)[1] / quantize_sym(w, 4, 128)[1]
        assert ratio_delta.mean() == pytest.approx(1.0, abs=0.02)
        assert float(e1 / e0) == pytest.approx(1.0 / s, rel=0.08)
