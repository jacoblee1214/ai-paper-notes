"""pytest: SAE 최소 구현의 핵심 성질. 13개, 약 15초 (CPU).  실행: python -m pytest analysis/test_minimal_impl.py -q (torch, numpy, pytest, scipy 필요)"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import minimal_impl as M

ART = Path(__file__).resolve().parents[2] / "induction_heads" / "artifacts"


def _bed(n, d, p, seed=0, orthogonal=False):
    G = M.make_truth(n, d, seed, orthogonal)
    samp = M.synthetic_sampler(G, p)
    Xe = samp(20000, torch.Generator().manual_seed(77))
    return G, samp, Xe


def test_generate_batch_density_and_range():
    x = M.generate_batch(20000, 0.1, 30, torch.Generator().manual_seed(0))
    assert x.shape == (20000, 30)
    assert abs(float((x > 0).float().mean()) - 0.1) < 0.005
    assert float(x.max()) <= 1.0 and float(x.min()) >= 0.0
    assert abs(float(x[x > 0].mean()) - 0.5) < 0.01


def test_mmcs_is_signed_and_directional():
    G = M.make_truth(8, 8, 0, orthogonal=True)
    assert float(M.mmcs(G, G).mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(M.mmcs(G, -G).mean()) < 0.1                 # 부호 있는 코사인: 반대 방향은 일치가 아님
    D = torch.cat([G, torch.randn(4, 8)])                    # 사전이 더 크면 정답->사전은 1, 사전->정답은 낮아짐
    assert float(M.mmcs(G, D).mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(M.mmcs(D, G).mean()) < 1.0


def test_decoder_rows_stay_unit_norm():
    G, samp, _ = _bed(16, 16, 0.1)
    for tied in (False, True):
        s = M.train_sae(samp, 16, 32, 0.03, tied=tied, steps=300)
        assert torch.allclose(s.W_dec.norm(dim=1), torch.ones(32), atol=1e-5)
    assert M.SAE(16, 32, tied=True).W_enc is None                 # 묶은 가중치: 인코더 행렬 = 디코더 행렬
    s_t = M.SAE(16, 32, tied=True)
    assert s_t.encoder_matrix() is s_t.W_dec


def test_orthogonal_sparse_truth_is_recovered():
    G, samp, Xe = _bed(16, 16, 0.1, orthogonal=True)
    s = M.train_sae(samp, 16, 16, 0.03, steps=6000, seed=0)
    D = s.dictionary()
    assert float(M.mmcs(G, D).mean()) > 0.95
    assert M.recovered_frac(G, D) > 0.9
    assert M.evaluate(s, Xe)["fvu"] < 0.02


def test_alpha_zero_with_large_dictionary_reconstructs():
    # 정답 특징이 4배 겹쳐 있어도(64개를 16차원에), alpha=0이고 사전이 8배(128)면 FVU가 거의 0
    G, samp, Xe = _bed(64, 16, 0.05)
    s = M.train_sae(samp, 16, 128, 0.0, steps=6000)
    assert M.evaluate(s, Xe)["fvu"] < 0.01


def test_larger_alpha_lowers_l0_and_raises_fvu():
    G, samp, Xe = _bed(64, 32, 0.05)
    ev = {a: M.evaluate(M.train_sae(samp, 32, 64, a, steps=2500), Xe) for a in (0.01, 0.1, 1.0)}
    assert ev[0.01]["l0"] > ev[0.1]["l0"] > ev[1.0]["l0"]
    assert ev[0.01]["fvu"] < ev[0.1]["fvu"] < ev[1.0]["fvu"]


def test_shrinkage_theory_matches_numeric_and_training():
    assert M.shrinkage_theory(0.0) == pytest.approx(1.0)
    a = torch.rand(2_000_000, generator=torch.Generator().manual_seed(1))
    for alpha in (0.1, 0.5, 1.0, 3.0):
        c = torch.relu(a - alpha / 2)
        num = float((c * a).sum() / (a * a).sum())
        assert M.shrinkage_theory(alpha) == pytest.approx(num, abs=2e-3)
    # 직교 정답 + 큰 alpha: 학습한 기울기가 이론값에 가깝고 1보다 작음
    G, samp, _ = _bed(16, 16, 0.1, orthogonal=True)
    s = M.train_sae(samp, 16, 16, 0.5, steps=6000)
    slope, n = M.shrinkage_slope(s, G, 0.1)
    assert n >= 12 and slope < 0.85 and abs(slope - M.shrinkage_theory(0.5)) < 0.1


def test_loss_recovered_formula():
    assert M.loss_recovered(2.0, 2.0, 5.0) == pytest.approx(1.0)
    assert M.loss_recovered(2.0, 5.0, 5.0) == pytest.approx(0.0)
    assert M.loss_recovered(2.0, 3.5, 5.0) == pytest.approx(0.5)


def _tiny_model():
    torch.manual_seed(0)
    cfg = M.Cfg(vocab=50, d=16, n_layer=2, n_head=2, d_mlp=32, ctx=20)
    return M.TinyLM(cfg).eval(), torch.randint(1, 50, (6, 20))


def test_identity_reconstruction_does_not_change_loss():
    model, t = _tiny_model()
    base = M.lm_loss(model, t)
    # x_hat = relu(x) - relu(-x) = x 를 정확히 내는 SAE: 사전 [I; -I], b=0
    s = M.SAE(16, 32, tied=True)
    with torch.no_grad():
        s.W_dec.copy_(torch.cat([torch.eye(16), -torch.eye(16)]))
        s.b.zero_()
    for name in ("resid_post_0", "resid_post_1", "mlp_out_0"):
        L = M.lm_loss(model, t, {name: M.splice_hook(M.sae_recon_fn(s, scale=3.0))})
        assert L == pytest.approx(base, abs=1e-5)
    # 지우면 손실이 달라지고 BOS(위치 0)는 건드리지 않는다
    L0 = M.lm_loss(model, t, {"resid_post_1": M.splice_hook(lambda x: torch.zeros_like(x))})
    assert abs(L0 - base) > 1e-3
    x = torch.randn(2, 5, 16)
    y = M.splice_hook(lambda z: torch.zeros_like(z))(x)
    assert torch.equal(y[:, 0], x[:, 0]) and float(y[:, 1:].abs().sum()) == 0.0


@pytest.mark.skipif(not (ART / "model_seed0.pt").exists(), reason="29번 artifacts 없음")
def test_model_load_matches_stored_activations():
    model = M.load_model(ART)
    t = torch.from_numpy(np.load(ART / "tokens.npy")[:4].astype(np.int64))
    with torch.no_grad():
        _, c = model(t, cache=True)
    for name, l, key in (("resid_post_l1", 1, "resid_post"), ("mlp_out_l0", 0, "mlp_out")):
        ref = np.load(ART / f"{name}.npy", mmap_mode="r")[:4].astype(np.float32)
        assert np.abs(c[key][l].numpy() - ref).max() < 0.02       # 저장은 float16


def test_top_token_stats_detects_planted_monosemantic_feature():
    g = torch.Generator().manual_seed(0)
    N = 5000
    tok = torch.randint(0, 40, (N,), generator=g)
    mono = torch.where(tok == 7, torch.rand(N, generator=g) + 1.0, torch.zeros(N))      # 토큰 7에서만
    poly = torch.rand(N, generator=g) * (torch.rand(N, generator=g) < 0.5)               # 토큰과 무관
    share, ent, tops = M.top_token_stats(torch.stack([mono, poly], 1), tok, k=20)
    assert share[0] == 1.0 and ent[0] == 0.0 and tops[0][0][0] == 7
    assert share[1] < 0.4 and ent[1] > 2.5
    dead = torch.zeros(N, 1)
    assert np.isnan(M.top_token_stats(dead, tok, 20)[0][0])


def test_make_repeat_tokens_structure():
    t = torch.randint(1, 100, (3, 40))
    t[:, 0] = 0
    r = M.make_repeat_tokens(t, L=8)
    assert r.shape == t.shape and torch.equal(r[:, 0], t[:, 0])
    assert torch.equal(r[:, 1:9], t[:, 1:9]) and torch.equal(r[:, 9:17], t[:, 1:9]) and torch.equal(r[:, 33:39], t[:, 1:7])


def test_head_dependence_shapes_on_tiny_model():
    model, t = _tiny_model()
    s = M.SAE(16, 24, seed=1)
    c, c_na, cp, mean_dir = M.head_dependence(model, s, 1.0, M.make_repeat_tokens(t, 5), 5, layer=1)
    assert c.shape == c_na.shape == (6 * 19, 24) and cp.shape == (6 * 19,)
    assert mean_dir.shape == (2, 16) and torch.allclose(mean_dir.norm(dim=1), torch.ones(2), atol=1e-5)
