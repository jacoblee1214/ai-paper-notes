"""minimal_impl.py 의 성질을 확인하는 테스트. CPU, 시드 고정, 1분 이내.
실행: python -m pytest analysis/test_minimal_impl.py -q  (논문 폴더 vit/ 에서)
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    np.random.seed(0)


def small(**kw):
    cfg = dict(img_size=32, patch=8, in_ch=3, num_classes=10, depth=2, dim=32, mlp=64, heads=4)
    cfg.update(kw)
    return M.ViT(**cfg)


def randomize(model):
    """헤드가 0 으로 시작하므로 출력을 비교하려면 가중치를 흔든다."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model.eval()


# 1. 패치 수
@pytest.mark.parametrize("h,w,p", [(224, 224, 16), (224, 224, 14), (384, 384, 16), (512, 512, 16), (64, 96, 8)])
def test_num_patches_and_tokens(h, w, p):
    if h % p or w % p:
        with pytest.raises(ValueError):
            M.num_patches(h, w, p)
        return
    n = M.num_patches(h, w, p)
    assert n == (h // p) * (w // p) == h * w // (p * p)
    x = torch.randn(2, 3, h, w)
    assert M.patchify(x, p).shape == (2, n, 3 * p * p)


def test_non_divisible_raises():
    with pytest.raises(ValueError):
        M.num_patches(224, 224, 14 + 1)  # 224 % 15 != 0
    assert M.num_patches(224, 224, 14) == 256


# 2. conv 패치 임베딩 == reshape + linear
def test_conv_embedding_equals_linear():
    emb = M.PatchEmbed(8, 3, 32)
    x = torch.randn(4, 3, 32, 40)
    assert torch.allclose(emb(x), emb.forward_linear(x), atol=1e-5)


# 3. 위치 임베딩 없으면 패치 순서에 대해 불변, 있으면 달라진다
def shuffle_patches(x, patch, perm):
    p = M.patchify(x, patch)[:, perm]  # (B, N, C*P*P), 패치 순서를 섞음
    b, n, _ = p.shape
    g = int(round(n ** 0.5))
    c = x.shape[1]
    p = p.reshape(b, g, g, c, patch, patch).permute(0, 3, 1, 4, 2, 5)
    return p.reshape(b, c, g * patch, g * patch)


def test_permutation_invariance_without_pos_emb():
    x = torch.randn(3, 3, 32, 32)
    perm = torch.randperm(16)
    assert not torch.equal(perm, torch.arange(16))
    m = randomize(small(use_pos=False))
    with torch.no_grad():
        y, y_perm = m(x), m(shuffle_patches(x, 8, perm))
    assert torch.allclose(y, y_perm, atol=1e-4)  # [CLS] 출력(logit)이 같음


def test_position_embedding_breaks_invariance():
    x = torch.randn(3, 3, 32, 32)
    perm = torch.randperm(16)
    m = randomize(small(use_pos=True))
    with torch.no_grad():
        m.pos_embed.copy_(torch.randn_like(m.pos_embed))
        y, y_perm = m(x), m(shuffle_patches(x, 8, perm))
    assert not torch.allclose(y, y_perm, atol=1e-3)


def test_shuffle_helper_roundtrip():
    x = torch.randn(1, 3, 32, 32)
    perm = torch.randperm(16)
    inv = torch.argsort(perm)
    assert torch.equal(shuffle_patches(shuffle_patches(x, 8, perm), 8, inv), x)


# 4. 위치 임베딩 보간
def test_interp_identity_on_same_grid():
    pos = torch.randn(1, 1 + 14 * 14, 64)
    out = M.resize_pos_embed(pos, (14, 14), (14, 14))
    assert torch.allclose(out, pos, atol=1e-6)


def test_interp_keeps_cls_and_shape():
    pos = torch.randn(1, 1 + 14 * 14, 16)
    out = M.resize_pos_embed(pos, (14, 14), (24, 24))
    assert out.shape == (1, 1 + 24 * 24, 16)
    assert torch.equal(out[:, 0], pos[:, 0])


def test_interp_matches_scipy_zoom_like_official_code():
    pos = torch.randn(1, 1 + 14 * 14, 8)
    out = M.resize_pos_embed(pos, (14, 14), (24, 24))[:, 1:].numpy()
    g = pos[0, 1:].numpy().reshape(14, 14, 8)
    ref = scipy.ndimage.zoom(g, (24 / 14, 24 / 14, 1), order=1).reshape(1, 24 * 24, 8)
    assert np.allclose(out, ref, atol=1e-5)


def test_interp_linear_ramp_exact_and_corners():
    # 행·열에 대해 선형인 임베딩은 정확히 보존되고(양 끝점 정렬), 네 모서리 값이 유지된다
    gh = gw = 7
    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    grid = torch.stack([yy / (gh - 1), xx / (gw - 1)], dim=-1).reshape(1, gh * gw, 2)
    out = M.resize_pos_embed(grid, (gh, gw), (13, 13), has_cls=False).reshape(13, 13, 2)
    ty, tx = torch.meshgrid(torch.linspace(0, 1, 13), torch.linspace(0, 1, 13), indexing="ij")
    assert torch.allclose(out[..., 0], ty, atol=1e-6) and torch.allclose(out[..., 1], tx, atol=1e-6)
    g = grid.reshape(gh, gw, 2)
    assert torch.equal(out[0, 0], g[0, 0]) and torch.allclose(out[-1, -1], g[-1, -1])


def test_model_runs_at_higher_resolution_after_interp():
    m = randomize(small())
    with torch.no_grad():
        m.pos_embed = torch.nn.Parameter(M.resize_pos_embed(m.pos_embed, (4, 4), (6, 6)))
        y = m(torch.randn(2, 3, 48, 48))
    assert y.shape == (2, 10) and torch.isfinite(y).all()


# 5. 파라미터 수: 해석식 == numel, 표 1과 비교
@pytest.mark.parametrize("kw", [
    dict(img_size=32, patch=8, depth=2, dim=32, mlp=64, heads=4, num_classes=10),
    dict(img_size=48, patch=8, depth=3, dim=48, mlp=96, heads=6, num_classes=7, pool="gap"),
    dict(img_size=32, patch=4, depth=1, dim=16, mlp=64, heads=2, num_classes=3, use_pos=False),
])
def test_param_formula_matches_numel_small(kw):
    m = M.ViT(**kw)
    pool = kw.get("pool", "cls")
    f = M.count_params_formula(kw["depth"], kw["dim"], kw["mlp"], kw["patch"], kw["img_size"], kw["num_classes"],
                               pool=pool)
    n = sum(p.numel() for p in m.parameters())
    if not kw.get("use_pos", True):
        f -= ((kw["img_size"] // kw["patch"]) ** 2 + (1 if pool == "cls" else 0)) * kw["dim"]
    assert n == f


def test_param_count_vit_b16_matches_table1():
    m = M.make_vit("B/16", img_size=224, num_classes=1000)
    n = sum(p.numel() for p in m.parameters())
    c = M.CONFIGS["B"]
    assert n == M.count_params_formula(c["depth"], c["dim"], c["mlp"], 16, 224, 1000)
    assert abs(n / 1e6 - 86) < 1.0  # 표 1: 86M


def test_param_formula_vs_table1_L_H():
    for name, patch, table in [("L", 16, 307), ("H", 14, 632)]:
        c = M.CONFIGS[name]
        n = M.count_params_formula(c["depth"], c["dim"], c["mlp"], patch, 224, 1000) / 1e6
        # H/14 는 632M 과 0.1% 이내. L/16 은 304.3M 으로 표의 307M 보다 작다(03 절 참조).
        if name == "H":
            assert abs(n - table) / table < 0.002
        else:
            assert 303.0 < n < 305.5


# 6. 헤드와 [CLS]/GAP
def test_head_zero_init_gives_zero_logits():
    m = small().eval()
    with torch.no_grad():
        assert torch.count_nonzero(m(torch.randn(2, 3, 32, 32))) == 0


def test_cls_and_gap_shapes_and_token_count():
    for pool in ("cls", "gap"):
        m = randomize(small(pool=pool))
        z = m.tokens(torch.randn(2, 3, 32, 32))
        assert z.shape[1] == 16 + (1 if pool == "cls" else 0)
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)


# 7. MSA: 헤드별 루프와 같음(부록 A 식 5-8)
def test_msa_equals_per_head_loop():
    msa = M.MSA(32, 4).eval()
    z = torch.randn(2, 9, 32)
    with torch.no_grad():
        ref_heads = []
        qkv = msa.qkv(z)  # (B, N, 3D)
        for h in range(4):
            dh = 8
            q = qkv[..., 0 * 32 + h * dh: 0 * 32 + (h + 1) * dh]
            k = qkv[..., 1 * 32 + h * dh: 1 * 32 + (h + 1) * dh]
            v = qkv[..., 2 * 32 + h * dh: 2 * 32 + (h + 1) * dh]
            a = torch.softmax(q @ k.transpose(1, 2) / dh ** 0.5, dim=-1)
            ref_heads.append(a @ v)
        ref = msa.out(torch.cat(ref_heads, dim=-1))
        out, attn = msa(z, return_attn=True)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 9), atol=1e-5)


# 8. 연산량 해석식 == torch FlopCounterMode (독립 계산)
def test_macs_formula_matches_flop_counter():
    for pool in ("cls", "gap"):
        kw = dict(img_size=32, patch=8, depth=2, dim=32, mlp=64, heads=4, num_classes=10, pool=pool)
        m = randomize(M.ViT(**kw))
        x = torch.randn(1, 3, 32, 32)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x)
        macs = M.macs_formula(kw["depth"], kw["dim"], kw["mlp"], kw["patch"], kw["img_size"], kw["num_classes"],
                              pool=pool)
        assert fc.get_total_flops() == 2 * macs


def test_attention_share_grows_with_resolution():
    c = M.CONFIGS["L"]
    shares = [M.attn_share(c["depth"], c["dim"], c["mlp"], 16, r) for r in (224, 384, 512)]
    assert shares[0] < shares[1] < shares[2]
    # MLP 크기 M = 4D 이면 비율은 T / (6D)
    t = (224 // 16) ** 2 + 1
    assert abs(shares[0] - t / (6 * c["dim"])) < 1e-12


def test_table6_exaflops_vit_rows_within_2p5_percent():
    # JFT-300M 303M장 x 에폭. 표 6의 ViT 행(exaFLOPs)과 비교
    jft = 303e6
    rows = [("B", 32, 7, 55), ("B", 16, 7, 224), ("L", 32, 7, 196), ("L", 16, 7, 783), ("L", 16, 14, 1567),
            ("H", 14, 14, 4262)]
    for s, p, ep, table in rows:
        c = M.CONFIGS[s]
        macs = M.macs_formula(c["depth"], c["dim"], c["mlp"], p, 224, 1000)
        ex = M.pretrain_exaflops(macs, jft * ep)
        # B/32 만 +2.0%(표의 55는 정수 반올림 값), 나머지는 0.25% 이내
        assert abs(ex - table) / table < 0.025, (s, p, ep, ex, table)
