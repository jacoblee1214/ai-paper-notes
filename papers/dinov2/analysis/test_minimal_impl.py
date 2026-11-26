"""minimal_impl.py 의 성질을 확인하는 테스트. CPU, 시드 고정, 1분 이내.
실행: python -m pytest analysis/test_minimal_impl.py -q  (논문 폴더 dinov2/ 에서)
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


# ---------------------------------------------------------------- Sinkhorn-Knopp

def test_sk_rows_are_distributions_columns_balanced():
    z = torch.randn(256, 32) * 0.1  # 코사인 로짓 규모(머리가 L2 정규화이므로 |로짓| <= 1), T=0.05
    q = M.sinkhorn_knopp(z, 0.05, n_iter=3)
    assert torch.allclose(q.sum(1), torch.ones(256), atol=1e-5)  # 각 샘플은 확률분포(행 합 1): 마지막 단계가 샘플 정규화
    col = q.sum(0) / 256  # 프로토타입 질량 비율, 균등이면 1/K
    assert float((col * 32 - 1).abs().max()) < 0.05  # 3회 뒤 근사 균등(이 규모에서 관측 0.02). 로짓이 더 퍼지면 3회로는 모자란다


def test_sk_converges_to_uniform_prototype_mass_with_more_iterations():
    z = torch.randn(256, 32) * 0.3  # 로짓이 퍼질수록 수렴이 느리다(3회 뒤 오차 0.6, 10회 0.05)
    err = [float(((M.sinkhorn_knopp(z, 0.05, n).sum(0) / 256) * 32 - 1).abs().max()) for n in (1, 3, 10, 50)]
    assert err[0] > err[1] > err[2] > err[3]  # 반복할수록 열 합 오차가 줄어든다
    assert err[3] < 1e-3


def test_sk_is_diagonal_scaling_of_exp_kernel():
    z = torch.randn(64, 16)
    temp = 0.1
    q = M.sinkhorn_knopp(z, temp, n_iter=5)
    log_ratio = torch.log(q) - z / temp  # 이면 a_i + b_k 꼴이어야 한다(Q = diag(u) exp(z/T) diag(v))
    resid = log_ratio - log_ratio.mean(0, keepdim=True) - log_ratio.mean(1, keepdim=True) + log_ratio.mean()
    assert float(resid.abs().max()) < 1e-4


def test_sk_prevents_dominant_prototype_but_softmax_does_not():
    z = torch.randn(256, 16) * 0.1
    z[:, 0] += 2.0  # 모든 샘플에서 0번 프로토타입이 우세
    p_soft = F.softmax(z / 0.05, dim=-1)
    q = M.sinkhorn_knopp(z, 0.05, 50)
    assert float(p_soft.mean(0).max()) > 0.99
    assert float(q.mean(0).max()) < 1.2 / 16


# ---------------------------------------------------------------- 중심화와 날카롭게 하기

def test_center_update_is_ema_of_batch_means():
    c = M.DINOCenter(8, momentum=0.9)
    manual = torch.zeros(1, 8)
    for _ in range(5):
        z = torch.randn(32, 8) + 1.0
        c.update(z)
        manual = 0.9 * manual + 0.1 * z.mean(0, keepdim=True)
    assert torch.allclose(c.center, manual, atol=1e-6)


def test_centering_removes_a_constant_bias():
    z = torch.randn(128, 16) * 0.3
    b = torch.zeros(1, 16)
    b[0, 3] = 5.0  # 모든 샘플에 같은 큰 편향
    c = M.DINOCenter(16)
    c.center = b + z.mean(0, keepdim=True)  # 중심이 편향을 학습한 상태
    p_with = c.probs(z + b, 0.1)
    p_without = F.softmax((z - z.mean(0, keepdim=True)) / 0.1, dim=-1)
    assert torch.allclose(p_with, p_without, atol=1e-5)
    assert float(F.softmax((z + b) / 0.1, dim=-1)[:, 3].min()) > 0.99  # 중심화 없으면 한 차원 지배


def test_centering_alone_collapses_to_uniform_for_constant_outputs():
    c = M.DINOCenter(16, momentum=0.9)
    z = torch.randn(1, 16).repeat(64, 1)  # 입력과 상관없이 같은 출력
    for _ in range(200):
        c.update(z)
    p = c.probs(z, 0.04)
    assert torch.allclose(p, torch.full_like(p, 1 / 16), atol=1e-3)  # 온도를 낮춰도 균등: 날카롭게 하기는 차이를 만들지 못한다


def test_sharpening_lowers_entropy_and_amplifies_small_differences():
    z = torch.randn(128, 16) * 0.05  # 중심화 뒤 작은 차이
    ents = [float(entropy(F.softmax(z / t, dim=-1)).mean()) for t in (0.5, 0.1, 0.04, 0.01)]
    assert ents[0] > ents[1] > ents[2] > ents[3]
    assert ents[0] > math.log(16) - 0.01  # 온도가 크면 거의 균등
    assert ents[3] < 1.0  # 낮은 온도에서는 같은 로짓이 뾰족해진다


# ---------------------------------------------------------------- KoLeo

def test_koleo_smaller_when_spread_larger_when_overlapping():
    n = 64
    ang = torch.arange(n) * 2 * math.pi / n
    spread = torch.stack([ang.cos(), ang.sin()], dim=1)  # 원 위에 균등
    clustered = torch.stack([torch.cos(0.2 * torch.rand(n)), torch.sin(0.2 * torch.rand(n))], dim=1)
    dup = spread.clone()
    dup[: n // 2] = dup[n // 2:]  # 절반이 다른 절반과 겹침
    ls, lc, ld = (float(M.koleo_loss(x)) for x in (spread, clustered, dup))
    assert ls < lc < ld
    assert ld > 10  # 겹치면 -log(eps) 에 가까워진다


def test_koleo_gradient_pushes_points_apart():
    x = (torch.tensor([[1.0, 0.0], [0.99, 0.14], [-1.0, 0.0]])).requires_grad_(True)
    M.koleo_loss(x).backward()
    xn = F.normalize(x.detach(), dim=-1)
    step = F.normalize(x.detach() - 0.05 * x.grad, dim=-1)
    d_before = (xn[0] - xn[1]).norm()
    d_after = (step[0] - step[1]).norm()
    assert d_after > d_before  # 가까운 두 점 사이 거리가 커진다


def test_kl_entropy_matches_gaussian_and_uniform():
    torch.manual_seed(1)
    x = torch.randn(4000, 2)
    assert abs(M.kl_entropy(x) - 2.8379) < 0.08  # N(0, I_2): H = log(2 pi e)
    u = torch.rand(4000, 3)
    assert abs(M.kl_entropy(u) - 0.0) < 0.08  # [0,1]^3 균등: H = 0


def test_koleo_is_affine_in_kl_entropy():
    torch.manual_seed(2)
    x = F.normalize(torch.randn(500, 8), dim=-1)
    n, d = x.shape
    log_vd = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)
    const = float(torch.special.digamma(torch.tensor(float(n)))) + 0.5772156649015329 + log_vd
    assert abs(M.kl_entropy(x) - (-d * float(M.koleo_loss(x, eps=0.0)) + const)) < 1e-3


# ---------------------------------------------------------------- EMA

def test_ema_extremes_and_closed_form():
    s, t = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
    t0 = [p.clone() for p in t.parameters()]
    M.ema_update(t, s, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(t.parameters(), t0))  # m=1: 교사 그대로
    M.ema_update(t, s, 0.0)
    assert all(torch.equal(a, b) for a, b in zip(t.parameters(), s.parameters()))  # m=0: 학생 복사

    m, steps = 0.9, 20
    t = torch.nn.Linear(4, 3)
    t0 = [p.clone() for p in t.parameters()]
    for _ in range(steps):
        M.ema_update(t, s, m)  # 학생이 고정이면 t_k - s = m^k (t_0 - s)
    for a, b, c in zip(t.parameters(), t0, s.parameters()):
        assert torch.allclose(a - c, m ** steps * (b - c), atol=1e-6)


def test_ema_is_exponentially_weighted_average_of_student_history():
    m = 0.8
    s_hist = torch.randn(10)
    t = torch.tensor([0.0])
    for x in s_hist:
        t = m * t + (1 - m) * x
    manual = sum((1 - m) * m ** (len(s_hist) - 1 - k) * s_hist[k] for k in range(len(s_hist)))
    assert abs(float(t) - float(manual)) < 1e-6  # t_0 = 0 이므로 첫 항 m^T t_0 은 0


def test_schedules():
    assert M.cosine_schedule(0.994, 1.0, 0, 100) == pytest.approx(0.994)
    assert M.cosine_schedule(0.994, 1.0, 100, 100) == pytest.approx(1.0)
    assert M.teacher_temp_schedule(0, 100) == pytest.approx(0.04)
    assert M.teacher_temp_schedule(100, 100) == pytest.approx(0.07)
    assert M.teacher_temp_schedule(500, 100) == pytest.approx(0.07)


# ---------------------------------------------------------------- 손실 구성

def test_dino_loss_term_count_and_pairing():
    b, k = 4, 8
    sg = torch.randn(2, b, k)
    sl = torch.randn(3, b, k)
    tp = F.softmax(torch.randn(2, b, k), dim=-1)
    loss = M.dino_cls_loss(sg, sl, tp, 0.1)
    ce = lambda s, t: -(t * F.log_softmax(s / 0.1, dim=-1)).sum(-1).mean()  # noqa: E731
    manual = (ce(sg[1], tp[0]) + ce(sg[0], tp[1]) + sum(ce(s, tp[j]) for j in range(2) for s in sl)) / (2 + 2 * 3)
    assert torch.allclose(loss, manual, atol=1e-6)  # 전역 교차 2쌍 + 지역 6쌍, 같은 크롭끼리는 짝짓지 않음


def test_dino_loss_minimized_when_student_matches_teacher():
    t = F.softmax(torch.randn(2, 6, 8) * 3, dim=-1)
    s_match = (torch.log(t) * 0.1)  # log_softmax(s/0.1) = log t
    l_match = M.dino_cls_loss(s_match, s_match[:1].repeat(2, 1, 1), t, 0.1)
    l_rand = M.dino_cls_loss(torch.randn(2, 6, 8), torch.randn(2, 6, 8), t, 0.1)
    assert l_match < l_rand


def test_ibot_loss_only_uses_masked_patches():
    b, n, k = 3, 10, 6
    s = torch.randn(b, n, k)
    t = F.softmax(torch.randn(b, n, k), dim=-1)
    mask = torch.zeros(b, n, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, 5:7] = True
    l1 = M.ibot_patch_loss(s, t, mask)
    s2, t2 = s.clone(), t.clone()
    s2[~mask] = torch.randn_like(s2[~mask])  # 가리지 않은 위치를 바꿔도
    t2[~mask] = F.softmax(torch.randn(int((~mask).sum()), k), dim=-1)
    assert torch.allclose(l1, M.ibot_patch_loss(s2, t2, mask))
    assert float(M.ibot_patch_loss(s, t, torch.zeros_like(mask))) == 0.0  # 가린 패치가 없으면 0
    ce = -(t * F.log_softmax(s / 0.1, dim=-1)).sum(-1)
    manual = (ce[0, :3].mean() + ce[1, 5:7].mean() + 0) / 3  # 이미지마다 평균, B 로 나눔(마스크 없는 이미지는 0)
    assert torch.allclose(l1, manual, atol=1e-6)


def test_mask_token_replaces_patches_before_position_embedding():
    vit = M.TinyViT().eval()
    x = torch.randn(2, 3, 32, 32)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :10] = True
    with torch.no_grad():
        _, p_plain = vit(x)
        _, p_mask = vit(x, mask=mask)
        x2 = x.clone()
        x2[:, :, :8, :] = torch.randn_like(x2[:, :, :8, :])  # 첫 두 패치 행(0..15번 패치)을 바꿔도
        _, p_mask2 = vit(x2, mask=mask)
    assert not torch.allclose(p_plain, p_mask)
    # 마스크된 패치 0..9 의 입력 내용은 사라졌지만, 다른 패치와의 attention 으로 출력은 달라질 수 있다.
    # 확인 가능한 것: 임베딩 단계에서 mask_token 이 들어갔는지(내용 독립)
    emb = vit.embed(x)
    emb2 = vit.embed(x2)
    z = torch.where(mask.unsqueeze(-1), vit.mask_token, emb)
    z2 = torch.where(mask.unsqueeze(-1), vit.mask_token, emb2)
    assert torch.equal(z[:, :10], z2[:, :10])


def test_backbone_handles_local_crop_size_via_pos_interpolation():
    vit = M.TinyViT(img_size=32, patch=4)
    cls, patches = vit(torch.randn(3, 3, 16, 16))
    assert cls.shape == (3, 64) and patches.shape == (3, 16, 64)
    cls, patches = vit(torch.randn(3, 3, 32, 32))
    assert patches.shape == (3, 64, 64)


# ---------------------------------------------------------------- 교사 기울기, 전체 한 스텝

def make_batch(model_b=8, n_local=2):
    img, _ = M.make_shapes(model_b)
    gc = torch.stack([M.color_augment(M.random_resized_crop(img, 32, (0.32, 1.0))) for _ in range(2)])
    lc = torch.stack([M.color_augment(M.random_resized_crop(img, 16, (0.05, 0.32))) for _ in range(n_local)])
    mk = torch.stack([M.random_patch_masks(model_b, 64) for _ in range(2)])
    mk[:, 0, :5] = True  # 마스크가 하나도 없는 경우를 피한다
    return gc, lc, mk


@pytest.mark.parametrize("mode", ["center", "sk", "none"])
def test_no_gradient_reaches_teacher_and_student_gets_gradient(mode):
    model = M.DINOv2Toy(n_proto=16)
    gc, lc, mk = make_batch()
    loss, _ = model.loss(gc, lc, mk, 0.05, centering=mode, koleo_weight=0.1)
    loss.backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(not p.requires_grad for p in model.teacher.parameters())
    for name in ("backbone", "dino_head", "ibot_head"):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.student[name].parameters()), name


def test_teacher_starts_equal_to_student_and_ema_moves_it():
    model = M.DINOv2Toy(n_proto=16)
    for a, b in zip(model.teacher.parameters(), model.student.parameters()):
        assert torch.equal(a, b)
    with torch.no_grad():
        for p in model.student.parameters():
            p.add_(0.1)
    model.update_teacher(0.9)
    a, b = next(model.teacher.parameters()), next(model.student.parameters())
    assert torch.allclose(a, b - 0.1 + 0.1 * 0.1, atol=1e-6)  # 0.9 t + 0.1 s = s - 0.9 * 0.1


def test_center_is_updated_only_in_center_mode_and_after_softmax():
    model = M.DINOv2Toy(n_proto=16)
    gc, lc, mk = make_batch()
    model.loss(gc, lc, mk, 0.05, centering="sk", koleo_weight=0)
    assert float(model.dino_center.center.abs().sum()) == 0.0
    model.loss(gc, lc, mk, 0.05, centering="center", koleo_weight=0)
    assert float(model.dino_center.center.abs().sum()) > 0.0


def test_one_training_step_reduces_loss_on_same_batch():
    model = M.DINOv2Toy(n_proto=16)
    opt = torch.optim.AdamW(model.student.parameters(), lr=3e-3)
    gc, lc, mk = make_batch(16)
    l0, _ = model.loss(gc, lc, mk, 0.05, centering="sk", koleo_weight=0)
    for _ in range(10):
        opt.zero_grad()
        loss, _ = model.loss(gc, lc, mk, 0.05, centering="sk", koleo_weight=0)
        loss.backward()
        opt.step()
    assert float(loss.detach()) < float(l0.detach())


# ---------------------------------------------------------------- 크롭과 마스크

def test_full_scale_crop_without_flip_is_identity():
    x = torch.rand(4, 3, 32, 32)
    torch.manual_seed(5)
    out = M.random_resized_crop(x, 32, (1.0, 1.0))
    assert all(torch.allclose(o, i, atol=1e-5) or torch.allclose(o, i.flip(-1), atol=1e-5) for o, i in zip(out, x))
    assert out.shape == x.shape


def test_mask_ratio_range_and_fraction_masked():
    mk = M.random_patch_masks(2000, 64, prob=0.5, ratio=(0.1, 0.5))
    frac = mk.float().mean(1)
    assert abs(float((frac > 0).float().mean()) - 0.5) < 0.05  # 절반의 이미지만 마스크
    assert float(frac.max()) <= 0.5 + 1 / 64
    assert float(frac[frac > 0].min()) >= 0.1 - 1 / 64


def test_shapes_have_all_classes_and_valid_range():
    x, y = M.make_shapes(600)
    assert x.shape == (600, 3, 32, 32) and float(x.min()) >= 0 and float(x.max()) <= 1
    assert set(y.tolist()) == set(range(len(M.SHAPES)))
