"""minimal_impl.py의 핵심 성질 시험. 시드 고정, CPU, 1분 이내.
실행: python -m pytest analysis/test_minimal_impl.py -q  (논문 폴더에서)"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import minimal_impl as m  # noqa: E402

torch.manual_seed(0)


def small_qf(K=4, layers=4, seed=0):
    torch.manual_seed(seed)
    cfg = m.QFConfig(d=32, heads=4, layers=layers, ffn=64, k_queries=K, d_img=16, embed_dim=16)
    qf = m.QFormer(cfg).eval()
    qf.query_tokens.data.normal_(0, 1.0)          # 질의 토큰 초기값 0.02는 작아서 경로 차이가 잘 안 보이므로 키운다
    return qf


def rand_img(B=3, P=20, d=16, seed=1):
    return torch.randn(B, P, d, generator=torch.Generator().manual_seed(seed))


def rand_text(B=3, T=6, seed=2):
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(3, m.V, (B, T), generator=g)
    t[:, 0] = m.CLS
    return t


# ---- 마스크: 그림 2 오른쪽 -------------------------------------------------

def test_mask_matrices_match_figure2():
    K, T = 2, 3
    itc = m.build_mask("itc", K, T)[0, 0]
    itg = m.build_mask("itg", K, T)[0, 0]
    itm = m.build_mask("itm", K, T)[0, 0]
    assert itm.all()
    # ITC: 질의 블록과 텍스트 블록이 서로 분리
    assert not itc[:K, K:].any() and not itc[K:, :K].any() and itc[:K, :K].all() and itc[K:, K:].all()
    # ITG: 질의는 텍스트를 못 보고, 텍스트는 모든 질의와 앞쪽(자기 포함) 텍스트를 본다
    assert not itg[:K, K:].any()
    assert itg[K:, :K].all()
    assert torch.equal(itg[K:, K:], torch.tril(torch.ones(T, T, dtype=torch.bool)))


def test_mask_hides_padding_columns():
    pad = torch.tensor([[False, False, True]])
    for mode in ("itc", "itg", "itm"):
        mk = m.build_mask(mode, 2, 3, pad)[0, 0]
        assert not mk[:, 4].any()


# ---- 정보 흐름: 각 마스크가 막아야 하는 의존성 ---------------------------------

def test_itc_text_does_not_depend_on_image_and_queries_not_on_text():
    qf = small_qf()
    img1, img2 = rand_img(seed=1), rand_img(seed=5)
    t1, t2 = rand_text(seed=2), rand_text(seed=7)
    q_a, t_a = qf(img1, t1, "itc")
    _, t_b = qf(img2, t1, "itc")
    q_c, _ = qf(img1, t2, "itc")
    assert torch.allclose(t_a, t_b, atol=1e-6)      # 텍스트 표현은 이미지가 바뀌어도 같음
    assert torch.allclose(q_a, q_c, atol=1e-6)      # 질의 표현은 텍스트가 바뀌어도 같음


def test_itc_joint_pass_equals_separate_passes():
    """LAVIS는 ITC에서 이미지 경로와 텍스트 경로를 따로 돌린다. 마스크를 쓴 한 번의 순전파와 같아야 한다."""
    qf = small_qf()
    img, txt = rand_img(), rand_text()
    q_joint, t_joint = qf(img, txt, "itc")
    q_alone, _ = qf(img, None, "query")
    _, t_alone = qf(img, txt, "text")
    assert torch.allclose(q_joint, q_alone, atol=1e-5)
    assert torch.allclose(t_joint, t_alone, atol=1e-5)


def test_itg_queries_do_not_see_text_but_text_sees_image_through_queries():
    qf = small_qf()
    img1, img2 = rand_img(seed=1), rand_img(seed=5)
    t1, t2 = rand_text(seed=2), rand_text(seed=7)
    q_a, t_a = qf(img1, t1, "itg")
    q_b, _ = qf(img1, t2, "itg")
    q_alone, _ = qf(img1, None, "query")
    assert torch.allclose(q_a, q_b, atol=1e-6)           # 질의 출력은 텍스트를 바꿔도 같음
    assert torch.allclose(q_a, q_alone, atol=1e-5)       # 텍스트 없이 돌린 것과도 같음 (LAVIS가 질의의 KV를 재사용하는 근거)
    _, t_c = qf(img2, t1, "itg")
    assert (t_a - t_c).abs().max() > 1e-3                # 텍스트는 이미지에 의존(질의를 거쳐서)


def test_itg_is_causal_over_text():
    qf = small_qf()
    img, t1 = rand_img(), rand_text(T=8)
    t2 = t1.clone()
    t2[:, 5:] = (t2[:, 5:] + 1) % m.V + 3
    _, o1 = qf(img, t1, "itg")
    _, o2 = qf(img, t2, "itg")
    assert torch.allclose(o1[:, :5], o2[:, :5], atol=1e-6)    # 앞쪽 토큰 출력은 뒤쪽 입력에 무관
    assert (o1[:, 5:] - o2[:, 5:]).abs().max() > 1e-3


def test_itm_is_bidirectional():
    qf = small_qf()
    img, t1, t2 = rand_img(), rand_text(seed=2), rand_text(seed=7)
    q_a, _ = qf(img, t1, "itm")
    q_b, _ = qf(img, t2, "itm")
    assert (q_a - q_b).abs().max() > 1e-3                 # 질의 출력이 텍스트에 의존
    _, t_a = qf(rand_img(seed=1), t1, "itm")
    _, t_b = qf(rand_img(seed=5), t1, "itm")
    assert (t_a - t_b).abs().max() > 1e-3                 # 텍스트 출력이 이미지에 의존


def test_image_info_reaches_text_only_after_first_cross_layer():
    """교차 attention은 층 0에 있으므로(0, 2, ...) ITG에서 텍스트가 이미지 정보를 받는 것은 층 1부터다."""
    qf = small_qf(layers=4)
    assert qf.cross_layers() == [0, 2]
    txt = rand_text()
    _, _, l1 = qf(rand_img(seed=1), txt, "itg", return_layers=True)
    _, _, l2 = qf(rand_img(seed=5), txt, "itg", return_layers=True)
    K = qf.cfg.k_queries
    d_text = [(a[:, K:] - b[:, K:]).abs().max().item() for a, b in zip(l1, l2)]
    d_query = [(a[:, :K] - b[:, :K]).abs().max().item() for a, b in zip(l1, l2)]
    assert d_text[0] < 1e-6 and d_text[1] > 1e-4
    assert d_query[0] > 1e-4


# ---- 출력 토큰 수 ---------------------------------------------------------------

def test_output_token_count_is_K_for_any_patch_count():
    qf = small_qf(K=5)
    for P in (9, 36, 64, 257):
        q, _ = qf(rand_img(B=2, P=P), None, "query")
        assert q.shape == (2, 5, 32)


def test_mlp_connector_token_count_grows_with_patches():
    enc = m.FrozenImageEncoder()
    for g in (6, 8, 10):
        feats = enc(torch.rand(2, g * g, 4), g)
        assert m.MLPConnector(32, 48)(feats, g).shape == (2, g * g, 48)
        assert m.MLPConnector(32, 48, pool=2)(feats, g).shape[1] == (g // 2) ** 2


# ---- ITC, ITM ------------------------------------------------------------------

def test_itc_score_is_max_over_queries():
    qf = small_qf()
    img, txt = rand_img(B=4), rand_text(B=4)
    q, t = qf(img, txt, "itc")
    sim, sim_q2t = m.itc_similarity(q, t[:, 0], qf.vision_proj, qf.text_proj)
    zq = F.normalize(qf.vision_proj(q), dim=-1)
    zt = F.normalize(qf.text_proj(t[:, 0]), dim=-1)
    for i in range(4):
        for j in range(4):
            manual = max(float((zq[i, k] @ zt[j]).detach()) for k in range(qf.cfg.k_queries))
            assert sim[i, j].item() == pytest.approx(manual, abs=1e-6)
    # 질의가 하나뿐이면 일반 CLIP식 코사인과 같다
    qf1 = small_qf(K=1)
    q1, t1 = qf1(img, txt, "itc")
    s1, _ = m.itc_similarity(q1, t1[:, 0], qf1.vision_proj, qf1.text_proj)
    cos = F.normalize(qf1.vision_proj(q1[:, 0]), dim=-1) @ F.normalize(qf1.text_proj(t1[:, 0]), dim=-1).t()
    assert torch.allclose(s1, cos, atol=1e-6)


def test_hard_negatives_exclude_positive_and_follow_similarity():
    B = 6
    sim = torch.zeros(B, B)
    sim[0, 3] = 0.9                                   # 이미지 0에게 텍스트 3이 훨씬 비슷함
    g = torch.Generator().manual_seed(0)
    picks = torch.stack([m.sample_hard_negatives(sim, 0.07, g)[0] for _ in range(300)])
    assert not (picks == torch.arange(B)[None]).any()                     # 자기 짝은 절대 안 뽑힘
    assert (picks[:, 0] == 3).float().mean() > 0.95                      # 비슷한 쪽이 주로 뽑힘
    assert len(set(picks[:, 1].tolist())) > 2                            # 비슷함이 같으면 다양하게 (argmax가 아님)


def test_stage1_loss_shapes_and_itm_layout():
    qf = small_qf()
    B = 4
    img, cap = rand_img(B=B), m.caption_ids(m.sample_scenes(B, torch.Generator().manual_seed(0)),
                                           torch.Generator().manual_seed(1))
    ls = m.stage1_losses(qf, img, cap, torch.Generator().manual_seed(2))
    assert set(ls) == {"loss_itc", "loss_itm", "loss_itg"}
    assert all(v.dim() == 0 and torch.isfinite(v) for v in ls.values())
    # 초기 ITM은 2-클래스 로짓이라 대략 log 2 근처, ITC는 log B 근처(라벨 평활화 포함)
    assert 0.3 < ls["loss_itm"].item() < 1.5


def test_stage1_overfits_tiny_batch():
    """30스텝 안에 작은 배치에서 세 손실의 합이 줄어든다(구현이 학습 가능한 상태인지)."""
    torch.manual_seed(0)
    qf = m.QFormer(m.QFConfig(d=32, heads=4, layers=2, ffn=64, k_queries=4, d_img=32, embed_dim=16))
    enc = m.freeze(m.FrozenImageEncoder())
    gen = torch.Generator().manual_seed(0)
    sc = m.sample_scenes(16, gen)
    feats, cap = enc(m.render_patches(sc), m.G), m.caption_ids(sc, gen)
    opt = torch.optim.AdamW(qf.parameters(), lr=3e-3)
    tot = []
    for _ in range(30):
        ls = m.stage1_losses(qf, feats, cap, gen)
        loss = sum(ls.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot.append(loss.item())
    assert tot[-1] < tot[0] - 0.5


# ---- 2단계: 고정과 기울기 --------------------------------------------------------

def test_frozen_modules_get_no_gradient_and_do_not_move():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    enc = m.freeze(m.FrozenImageEncoder())
    dec = m.freeze(m.TinyDecoder(d=32, heads=4, layers=1, f=64))
    qf = m.QFormer(m.QFConfig(d=32, heads=4, layers=2, ffn=64, k_queries=4, d_img=32, embed_dim=16))
    vlm = m.VLM(enc, m.QFormerConnector(qf, 32), dec)
    dec_before = [p.detach().clone() for p in dec.parameters()]
    enc_before = [p.detach().clone() for p in enc.parameters()]
    sc = m.sample_scenes(8, gen)
    q, a, _ = m.mixed_qa(sc, gen)
    vlm.loss(m.render_patches(sc), q, a).backward()
    assert all(p.grad is None for p in dec.parameters())
    assert all(p.grad is None for p in enc.parameters())
    assert qf.query_tokens.grad is not None and qf.query_tokens.grad.abs().sum() > 0
    assert vlm.connector.proj.weight.grad is not None
    trainable = [p for p in vlm.parameters() if p.requires_grad]
    assert len(trainable) == len(list(qf.parameters())) + 2          # Q-Former 전부 + 투영(가중치, 편향)
    torch.optim.AdamW(trainable, lr=1e-2).step()
    assert all(torch.equal(a_, b) for a_, b in zip(dec_before, dec.parameters()))
    assert all(torch.equal(a_, b) for a_, b in zip(enc_before, enc.parameters()))


def test_soft_prompt_is_prepended_before_text():
    """소프트 프롬프트가 디코더 입력 앞에 붙고(길이 K), 답 위치 예측은 그 뒤 텍스트에서 나온다."""
    torch.manual_seed(0)
    dec = m.freeze(m.TinyDecoder(d=32, heads=4, layers=1, f=64))
    pre = torch.randn(2, 5, 32)
    ids = torch.randint(3, m.V, (2, 7))
    out = dec(ids, prefix=pre)
    assert out.shape == (2, 12, m.V)
    # 인과성: 프롬프트 출력은 뒤의 텍스트에 무관
    ids2 = ids.clone()
    ids2[:, 0] = 3 + (ids2[:, 0] - 3 + 1) % (m.V - 3)
    assert torch.allclose(out[:, :5], dec(ids2, prefix=pre)[:, :5], atol=1e-6)


# ---- 데이터와 파라미터 수 ----------------------------------------------------------

def test_scenes_have_distinct_colors_and_cells_and_qa_is_consistent():
    gen = torch.Generator().manual_seed(0)
    sc = m.sample_scenes(500, gen)
    for b in range(500):
        n = int(sc["nobj"][b])
        cs, cl = sc["cells"][b, :n].tolist(), sc["colors"][b, :n].tolist()
        assert len(set(cs)) == n and len(set(cl)) == n and all(c >= 0 for c in cs)
        assert (sc["cells"][b, n:] == -1).all()
    raw = m.render_patches(sc)
    assert torch.equal(raw[..., 3].sum(1), sc["nobj"].float())
    q, a = m.make_qa(sc, "where", gen)
    cell = (a[:, 0] - m.ROW0) * m.G + (a[:, 1] - m.COL0)
    col = q[:, 3] - m.COLOR0
    for b in range(50):
        assert raw[b, cell[b], 3] == 1 and torch.allclose(raw[b, cell[b], :3], m.RGB[col[b]])


def test_toy_param_formula_matches_module():
    for cfg in (m.QFConfig(), m.QFConfig(d=48, layers=6, k_queries=3, cross_freq=3, d_img=20, ffn=96)):
        qf = m.QFormer(cfg)
        assert m.toy_param_count(cfg) == sum(p.numel() for p in qf.parameters())


def test_cross_attention_every_other_layer_and_188M():
    qf = m.QFormer(m.QFConfig(layers=12))
    assert qf.cross_layers() == [0, 2, 4, 6, 8, 10]
    n1 = m.qformer_param_count()                       # BERT-base 크기, ViT-g(1408), 질의 32개
    assert abs(n1 - 188e6) / 188e6 < 0.01              # 논문의 188M과 1% 이내
    assert m.qformer_param_count(query_ffn=False) < 135e6   # 질의용 FFN을 빼면 188M이 안 나옴
    # 2단계에서 학습되는 양: Table 2의 104M(ViT-L), 107M(OPT-2.7B, 은닉 2560), 108M(FlanT5-XXL, 4096)
    assert abs(m.qformer_param_count(stage=2, dv=1024, E=2560) - 104e6) / 104e6 < 0.01
    assert abs(m.qformer_param_count(stage=2, E=2560) - 107e6) / 107e6 < 0.01
    assert abs(m.qformer_param_count(stage=2, E=4096) - 108e6) / 108e6 < 0.01
    assert 32 * 768 == 24576 and 257 * 1024 == 263168     # 출력 토큰 32 x 768 대 ViT-L/14의 257 x 1024


def test_decoder_pretraining_makes_caption_reading_possible():
    """짧게 학습해도 캡션을 읽고 개수를 세는 질문(count)은 우연(25%) 이상으로 답해야 한다."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    dec = m.pretrain_decoder(m.TinyDecoder(d=64, heads=4, layers=2, f=128), gen, steps=150, batch=64)
    sc = m.sample_scenes(200, gen)
    cap = m.caption_ids(sc, gen)[:, 1:]
    q, a = m.make_qa(sc, "count", gen)
    ids = torch.cat([cap, q], 1)
    pred = dec(ids, key_pad=ids == m.PAD)[:, -1].argmax(-1)
    assert (pred == a[:, 0]).float().mean() > 0.4


def test_qa_labels_are_consistent_with_scenes():
    gen = torch.Generator().manual_seed(3)
    sc = m.sample_scenes(400, gen)
    q, a = m.make_qa(sc, "count", gen)
    assert torch.equal(a[:, 0] - m.NUM0, sc["nobj"])
    q, a = m.make_qa(sc, "exist", gen)
    asked = q[:, 3] - m.COLOR0
    present = (sc["colors"] == asked[:, None]).any(1)
    assert torch.equal(a[:, 0] == m.TOK["yes"], present)
    assert 0.3 < present.float().mean() < 0.9                      # 두 답이 모두 충분히 나옴
    q, a = m.make_qa(sc, "row_is", gen)
    asked_color, asked_row = q[:, 2] - m.COLOR0, q[:, 4] - m.ROW0
    for b in range(100):
        j = int((sc["colors"][b] == asked_color[b]).nonzero()[0])
        true_row = int(sc["cells"][b, j]) // m.G
        assert (a[b, 0] == m.TOK["yes"]) == (true_row == int(asked_row[b]))
    assert 0.3 < (a[:, 0] == m.TOK["yes"]).float().mean() < 0.7
