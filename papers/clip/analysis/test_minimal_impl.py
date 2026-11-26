"""minimal_impl.py의 핵심 성질 시험. 시드 고정, CPU, 1분 이내.
실행: python -m pytest analysis/test_minimal_impl.py -q  (논문 폴더에서)"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as m  # noqa: E402


def unit(*shape, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=g, dtype=dtype), dim=-1)


# ---- 식 검산 ---------------------------------------------------------------

def test_random_embeddings_loss_is_log_N():
    """독립 무작위 단위 벡터에서 손실 = log N + O(scale^2 / 2d). scale=1이면 오차는 0.002 수준이다."""
    N, d = 1024, 512
    loss = m.clip_loss(unit(N, d, seed=1), unit(N, d, seed=2), 1.0).item()
    assert abs(loss - math.log(N)) < 0.02


def test_aligned_orthonormal_closed_form_and_temperature():
    """완벽히 정렬된 직교 임베딩: 한 행의 softmax는 정답 1개와 오답 N-1개(코사인 0)뿐이므로
    손실 = log(1 + (N-1) e^{-s})이다. 온도를 낮추면(s를 키우면) 0으로 간다."""
    N = 8
    E = torch.eye(N, dtype=torch.float64)
    prev = None
    for s in (0.5, 2.0, 8.0, 32.0, 100.0):
        loss = m.clip_loss(E, E, s).item()
        assert loss == pytest.approx(math.log(1 + (N - 1) * math.exp(-s)), rel=1e-9, abs=1e-12)
        assert prev is None or loss < prev
        prev = loss
    assert prev < 1e-30 + 1e-12  # s = 100이면 사실상 0


def test_loss_is_symmetric_and_average_of_two_directions():
    """I와 T를 바꿔도 같고, 두 방향 cross entropy의 평균이다 (그림 3의 loss = (loss_i + loss_t) / 2)."""
    I, T = unit(16, 12, seed=3), unit(16, 12, seed=4)
    s = 7.0
    a, b = m.clip_loss(I, T, s), m.clip_loss(T, I, s)
    assert torch.isclose(a, b)
    logits = s * I @ T.T
    y = torch.arange(16)
    row, col = F.cross_entropy(logits, y), F.cross_entropy(logits.T, y)
    assert torch.isclose(a, (row + col) / 2)
    assert not torch.isclose(row, col)  # 두 방향은 일반적으로 다르므로 평균이 의미 있다


def test_matches_figure3_pseudocode():
    """그림 3을 numpy식으로 직접 풀어 쓴 것과 같다."""
    I, T = unit(10, 6, seed=5), unit(10, 6, seed=6)
    t = 2.0  # log 매개변수
    logits = I @ T.T * math.exp(t)
    def ce(lg):  # 행마다 softmax 후 대각 원소의 -log
        p = torch.softmax(lg, dim=1)
        return -torch.log(torch.diagonal(p)).mean()
    ref = (ce(logits) + ce(logits.T)) / 2
    assert torch.isclose(m.clip_loss(I, T, math.exp(t)), ref)


def test_gradient_wrt_log_temperature_matches_derivation():
    """dL/dt, t = log scale. 한 방향(행) 손실 L_i = -s C_ii + logsumexp_j s C_ij 에서
    dL_i/dt = s (sum_j P_ij C_ij - C_ii) (P = softmax). 대칭 손실은 두 방향 평균."""
    I, T = unit(12, 5, seed=7), unit(12, 5, seed=8)
    t = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    m.clip_loss(I, T, t.exp()).backward()
    s, C = math.exp(1.5), I @ T.T
    def d(Cm):
        P = torch.softmax(s * Cm, dim=1)
        return s * ((P * Cm).sum(1) - torch.diagonal(Cm)).mean()
    assert t.grad.item() == pytest.approx(((d(C) + d(C.T)) / 2).item(), rel=1e-9)


def test_infonce_mutual_information_lower_bound():
    """I(X;Y) >= log N - L_NCE (Oord et al. 2018). 이변량 정규에서 최적 비판자
    f = log p(y|x)/p(y)를 쓰면 추정량이 (1) min(I, log N)을 넘지 않고 (2) I가 log N보다 훨씬 크면 log N에 붙는다."""
    g = torch.Generator().manual_seed(0)
    N, B = 16, 4000
    for rho in (0.5, 0.9, 0.999, 0.9999999):
        I_true = -0.5 * math.log(1 - rho**2)
        x = torch.randn(B, N, generator=g, dtype=torch.float64)
        y = rho * x + math.sqrt(1 - rho**2) * torch.randn(B, N, generator=g, dtype=torch.float64)
        f = lambda xx, yy: -0.5 * math.log(1 - rho**2) - (yy - rho * xx) ** 2 / (2 * (1 - rho**2)) + yy**2 / 2
        score = f(x[:, :, None], y[:, None, :])  # [B, N(x), N(y)]
        L = F.cross_entropy(score.reshape(B * N, N), torch.arange(N).repeat(B)).item()
        est = math.log(N) - L
        assert est <= min(I_true, math.log(N)) + 0.03, (rho, est, I_true)
        if rho == 0.9999999:  # I = 7.7 >> log 16 = 2.77: 추정량은 log N에 붙는다
            assert est > math.log(N) - 0.05


# ---- 온도 ------------------------------------------------------------------

def test_logit_scale_init_and_clamp():
    model = m.TinyCLIP()
    assert model.logit_scale.exp().item() == pytest.approx(1 / 0.07, rel=1e-5)  # 초기 온도 0.07
    model.logit_scale.data.fill_(10.0)  # exp = 22026
    model.clamp_logit_scale()
    assert model.logit_scale.exp().item() == pytest.approx(100.0, rel=1e-5)


def test_logit_scale_stays_below_100_during_training():
    """정렬된 쌍은 온도를 계속 낮추려는 기울기를 준다. 큰 학습률로 밀어도 clamp 덕에 100을 넘지 않는다."""
    model = m.TinyCLIP()
    E = torch.eye(8)
    opt = torch.optim.Adam([model.logit_scale], lr=0.5)
    for _ in range(60):
        loss = m.clip_loss(E, E, model.logit_scale.exp())
        opt.zero_grad(); loss.backward(); opt.step()
        model.clamp_logit_scale()
    assert model.logit_scale.exp().item() == pytest.approx(100.0, rel=1e-4)  # 상한에 닿았고 넘지 않음


# ---- 텍스트 인코더 -----------------------------------------------------------

def test_eot_is_largest_token_id():
    assert m.EOT == len(m.VOCAB) - 1 and m.EOT == max(m.TOK.values())


def test_text_encoder_pools_eot_and_is_causal():
    """EOT 뒤의 토큰(패딩)을 바꿔도 출력이 같고, EOT 앞의 토큰을 바꾸면 달라진다."""
    torch.manual_seed(0)
    model = m.TinyCLIP().eval()
    text = m._tokens([m.caption_words("photo", [(0, 1)])])
    eot_pos = int(text.argmax(-1))
    base = model.encode_text(text)
    after = text.clone()
    after[0, eot_pos + 1:] = m.TOK["red"]  # EOT 뒤를 다른 토큰으로 채움
    # argmax가 EOT를 그대로 가리키려면 EOT가 최대여야 한다 (위 시험). red < EOT이므로 유지된다.
    assert int(after.argmax(-1)) == eot_pos
    assert torch.allclose(model.encode_text(after), base, atol=1e-6)
    before = text.clone()
    before[0, 4] = m.TOK["blue"]  # EOT 앞 토큰 변경
    assert not torch.allclose(model.encode_text(before), base, atol=1e-4)


# ---- 제로샷 -----------------------------------------------------------------

def test_zero_shot_equals_cosine_argmax_and_ensemble_in_embedding_space():
    torch.manual_seed(0)
    model = m.TinyCLIP().eval()
    gen = torch.Generator().manual_seed(1)
    img, _ = m.make_single_eval(50, gen, held_out=False)
    for tmpls in (["photo"], ["photo", "drawing"], ["bare", "photo", "drawing"]):
        W = m.zero_shot_weights(model, tmpls)
        assert W.shape == (model.text.proj.shape[1], m.NC * m.NS)
        assert torch.allclose(W.norm(dim=0), torch.ones(W.shape[1]), atol=1e-5)  # 앙상블 뒤 다시 정규화
        # 직접 만든 기준: 클래스마다 템플릿 임베딩을 정규화하고 평균, 정규화, 코사인의 argmax
        cols = []
        for c in range(m.NC):
            for s in range(m.NS):
                e = F.normalize(model.encode_text(m._tokens([m.class_words(c, s, t) for t in tmpls])), dim=-1).mean(0)
                cols.append(F.normalize(e, dim=0))
        ref = (F.normalize(model.encode_image(img), dim=-1) @ torch.stack(cols, 1)).argmax(-1)
        assert torch.equal(m.zero_shot_predict(model, img, tmpls), ref)


# ---- hard negative ----------------------------------------------------------

def test_hard_negatives_only_add_to_image_to_text_side():
    I, T, H = unit(8, 6, seed=9), unit(8, 6, seed=10), unit(3, 6, seed=11)
    s = 5.0
    base = m.clip_loss(I, T, s)
    assert torch.isclose(m.clip_loss(I, T, s, H[:0]), base)  # 빈 hard negative는 영향 없음
    with_h = m.clip_loss(I, T, s, H)
    assert with_h > base  # 분모에 항이 더해지므로 이미지 -> 텍스트 손실이 커진다
    logits = s * I @ torch.cat([T, H]).T
    y = torch.arange(8)
    expect = (F.cross_entropy(logits, y) + F.cross_entropy((s * I @ T.T).T, y)) / 2
    assert torch.isclose(with_h, expect)


# ---- 합성 데이터와 학습 -------------------------------------------------------

def test_data_properties():
    gen = torch.Generator().manual_seed(0)
    img, text, swap = m.make_batch(256, gen, hard_neg=True)
    assert img.shape == (256, 3, m.CANVAS, m.CANVAS) and 0 <= img.min() and img.max() <= 1
    assert text.shape == (256, m.CONTEXT)
    # 학습 캡션에 학습에서 뺀 (색, 모양) 조합이 없어야 한다
    c_ids = [m.TOK[w] for w in m.COLORS]
    s_ids = [m.TOK[w] for w in m.SHAPES]
    held = {(m.TOK[m.COLORS[c]], m.TOK[m.SHAPES[s]]) for c, s in m.HELD_OUT}
    for row in text.tolist():
        for a, b in zip(row, row[1:]):
            assert (a, b) not in held
    # hard negative는 같은 단어 가방이지만 다른 문장이다
    st, idx = swap
    assert st.shape[0] == idx.shape[0] > 0
    for k in range(min(20, idx.shape[0])):
        a, b = text[int(idx[k])].tolist(), st[k].tolist()
        assert sorted(a) == sorted(b) and a != b


def test_initial_loss_is_log_N_and_training_reduces_it():
    N = 64
    gen = torch.Generator().manual_seed(0)
    img, text, _ = m.make_batch(N, gen)
    torch.manual_seed(0)
    model = m.TinyCLIP()
    with torch.no_grad():
        ie, te, s = model(img, text)
        l0 = m.clip_loss(ie, te, s).item()
    # 무작위 초기화 + 배율 14.3에서 손실은 log N 근처 (특징이 서로 비슷해 로짓 차가 작다)
    assert abs(l0 - math.log(N)) < 0.5
    model, res = m.train(seed=0, steps=120, batch=N, log_every=20)
    assert res.losses[-1][1] < l0 - 0.5
    assert model.logit_scale.exp().item() <= 100.0 + 1e-4
