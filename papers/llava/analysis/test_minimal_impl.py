"""minimal_impl.py 의 성질을 확인하는 테스트. CPU, 시드 고정, 1분 이내.
실행: python -m pytest llava/analysis/test_minimal_impl.py -q  (저장소 루트에서)
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as M  # noqa: E402


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def small_model(projector="linear", select_layer=-2):
    vis = M.TinyViT(dim=32, depth=3, heads=4, mlp=64)
    lm = M.TinyLM(dim=48, depth=2, heads=4, mlp=96)
    return M.LlavaToy(vis, lm, projector, select_layer)


def batch(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    sc = M.sample_scenes(n, g)
    q, a, _ = M.mixed_qa(sc, g)
    ids, labels = M.build_instruct(q, a)
    return sc, M.render(sc, g), ids, labels, q, a


# 1. 시각 토큰 수: 패치 수와 같고 CLS 는 센 적이 없다
@pytest.mark.parametrize("size,expected", [(224, 256), (336, 576)])
def test_visual_token_count_matches_patch_grid(size, expected):
    assert M.num_visual_tokens(size, 14) == expected
    assert M.num_visual_tokens(size, 14, include_cls=True) == expected + 1


def test_toy_visual_tokens_equal_patches_without_cls():
    m = small_model()
    _, img, *_ = batch()
    feats = m.vision_features(img)
    assert feats.shape[1] == M.num_visual_tokens(M.IMG, M.PATCH) == M.G * M.G == 25
    assert m.encode_images(img).shape == (6, 25, m.lm.dim)


# 2. 끝에서 두 번째 층 선택
def test_select_layer_minus2_is_second_to_last_block_without_cls():
    m = small_model(select_layer=-2)
    _, img, *_ = batch()
    hs = m.vision.hidden_states(img)
    assert len(hs) == m.vision.depth + 1                      # 임베딩 + 블록 수
    assert torch.equal(m.vision_features(img), hs[-2][:, 1:])
    assert torch.equal(hs[-2], hs[m.vision.depth - 1])        # 블록 depth-1 의 출력
    assert not torch.allclose(hs[-1][:, 1:], hs[-2][:, 1:])   # 마지막 층과 다르다


def test_select_cls_patch_keeps_cls():
    hs = [torch.zeros(2, 5, 3)] * 4
    assert M.select_features(hs, -2, "patch").shape == (2, 4, 3)
    assert M.select_features(hs, -2, "cls_patch").shape == (2, 5, 3)


# 3. 투영기
def test_projector_output_dim_and_structure():
    lin = M.build_projector("linear", 32, 48)
    mlp = M.build_projector("mlp2x_gelu", 32, 48)
    assert isinstance(lin, torch.nn.Linear) and (lin.in_features, lin.out_features) == (32, 48)
    kinds = [type(x).__name__ for x in mlp]
    assert kinds == ["Linear", "GELU", "Linear"]
    x = torch.randn(2, 7, 32)
    assert lin(x).shape == mlp(x).shape == (2, 7, 48)


def test_projector_param_formulas_match_modules():
    for kind in ["linear", "mlp2x_gelu", "mlp3x_gelu"]:
        mod = M.build_projector(kind, 32, 48)
        assert sum(p.numel() for p in mod.parameters()) == M.projector_params(kind, 32, 48)


def test_real_scale_projector_and_tower_params():
    assert M.projector_params("linear", 1024, 4096) == 4_198_400          # 1024*4096 + 4096
    assert M.projector_params("mlp2x_gelu", 1024, 4096) == 4_198_400 + 4096 * 4096 + 4096
    assert M.clip_vit_params(1024, 24, 4096, 14, 224) == 303_179_776
    with torch.device("meta"):
        big = M.TinyViT(img=224, patch=14, dim=1024, depth=24, heads=16, mlp=4096)
    assert big.tower_params() == M.clip_vit_params(1024, 24, 4096, 14, 224)   # 식 = 실제 모듈 구조


def test_context_budget():
    assert M.context_budget(2048, 576) == 1472
    assert M.context_budget(2048, 2880) < 0                   # AnyRes 최대 토큰은 2048 문맥에 안 들어간다


# 4. 병합과 라벨 마스크
def test_merge_inserts_visual_tokens_and_ignores_their_labels():
    m = small_model()
    _, img, ids, labels, *_ = batch()
    vis = m.encode_images(img)
    emb, lab = M.merge_visual_tokens(ids, labels, m.lm.embed, vis)
    slot = int(torch.nonzero(ids[0] == M.IMAGE_TOKEN_INDEX)[0])
    t = M.G * M.G
    assert emb.shape[1] == ids.shape[1] - 1 + t and lab.shape == emb.shape[:2]
    assert torch.equal(emb[:, slot:slot + t], vis)
    assert (lab[:, slot:slot + t] == M.IGNORE_INDEX).all()
    assert torch.equal(emb[:, :slot], m.lm.embed(ids[:, :slot]))


def test_instruct_labels_only_on_answer_and_eos():
    sc, _, ids, labels, q, a = batch(n=40)
    keep = labels != M.IGNORE_INDEX
    # 질문, 역할 표지, 시스템, 이미지 자리, <bos> 의 라벨은 모두 무시값
    assert (labels[:, : ids.shape[1] - 3] == M.IGNORE_INDEX).all()
    # 라벨이 있는 위치는 답 토큰들과 <eos> 뿐
    for i in range(ids.shape[0]):
        toks = labels[i][keep[i]].tolist()
        ans = [t for t in a[i].tolist() if t != M.PAD]
        assert toks == ans + [M.EOS]


def test_plain_stage1_labels_are_caption_and_eos_only():
    g = torch.Generator().manual_seed(3)
    sc = M.sample_scenes(8, g)
    caps = M.caption_tokens(sc)
    ids, labels = M.build_plain(caps)
    for i, cap in enumerate(caps):
        assert ids[i, 0] == M.BOS and ids[i, 1] == M.IMAGE_TOKEN_INDEX
        assert labels[i][labels[i] != M.IGNORE_INDEX].tolist() == cap + [M.EOS]
        assert (labels[i, :2] == M.IGNORE_INDEX).all()


def test_loss_equals_manual_cross_entropy_on_answer_positions_only():
    m = small_model().eval()
    _, img, ids, labels, *_ = batch(n=5)
    logits, lab, loss = m.logits_and_loss(img, ids, labels)
    tgt = lab[:, 1:]
    mask = tgt != M.IGNORE_INDEX
    logp = torch.log_softmax(logits[:, :-1], -1)
    manual = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    # 질문 토큰을 바꿔도 라벨이 무시되는 위치의 손실 기여는 없다: 무시 위치 수 + 유효 위치 수 = 전체
    assert int(mask.sum()) == int((labels != M.IGNORE_INDEX).sum())


# 5. 단계별 동결
def snapshot(mod):
    return {k: v.clone() for k, v in mod.state_dict().items()}


def one_step(model, stage, steps=1):
    params = M.set_stage(model, stage)
    opt = torch.optim.SGD(params, lr=0.1)
    _, img, ids, labels, *_ = batch(n=8)
    for _ in range(steps):
        loss = model(img, ids, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_stage1_trains_only_projector():
    m = small_model()
    v0, l0, p0 = snapshot(m.vision), snapshot(m.lm), snapshot(m.projector)
    params = M.set_stage(m, 1)
    assert {id(p) for p in params} == {id(p) for p in m.projector.parameters()}
    one_step(m, 1)
    assert all(p.grad is None for p in m.vision.parameters())
    assert all(p.grad is None for p in m.lm.parameters())
    assert all(p.grad is not None for p in m.projector.parameters())
    assert all(torch.equal(v0[k], v) for k, v in m.vision.state_dict().items())
    assert all(torch.equal(l0[k], v) for k, v in m.lm.state_dict().items())
    assert any(not torch.equal(p0[k], v) for k, v in m.projector.state_dict().items())


@pytest.mark.parametrize("projector", ["linear", "mlp2x_gelu"])
def test_stage2_trains_projector_and_lm_but_not_vision(projector):
    m = small_model(projector)
    v0, l0 = snapshot(m.vision), snapshot(m.lm)
    one_step(m, 2)
    assert all(p.grad is None for p in m.vision.parameters())
    assert all(p.grad is not None for p in m.projector.parameters())
    assert any(p.grad is not None for p in m.lm.parameters())
    assert all(torch.equal(v0[k], v) for k, v in m.vision.state_dict().items())
    assert any(not torch.equal(l0[k], v) for k, v in m.lm.state_dict().items())


def test_vision_features_carry_no_graph():
    m = small_model()
    M.set_stage(m, 2)
    _, img, *_ = batch()
    assert not m.vision_features(img).requires_grad


# 6. 인과성과 이미지 사용
def test_lm_is_causal_and_image_changes_answer_logits():
    m = small_model().eval()
    _, img, ids, labels, *_ = batch(n=3)
    logits, _, _ = m.logits_and_loss(img, ids, labels)
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 1) % M.V                    # 마지막 토큰만 바꾼다
    logits2, _, _ = m.logits_and_loss(img, ids2, labels)
    assert torch.allclose(logits[:, :-1], logits2[:, :-1], atol=1e-5)    # 앞 위치 로짓은 그대로
    img2 = img + 0.5 * torch.randn_like(img)
    logits3, _, _ = m.logits_and_loss(img2, ids, labels)
    assert not torch.allclose(logits[:, -4:], logits3[:, -4:], atol=1e-4)  # 이미지가 답 위치 로짓에 닿는다


def test_truncation_to_max_len_drops_the_tail_like_official_code():
    m = small_model()
    _, img, ids, labels, *_ = batch(n=2)
    emb, lab = M.merge_visual_tokens(ids, labels, m.lm.embed, m.encode_images(img), max_len=30)
    assert emb.shape[1] == 30 and lab.shape[1] == 30
    assert (lab != M.IGNORE_INDEX).sum() == 0                # 답이 꼬리에 있어서 잘려 나가면 손실 대상이 사라진다


# 7. 데이터 생성
def test_scenes_render_and_qa_are_consistent():
    g = torch.Generator().manual_seed(5)
    sc = M.sample_scenes(200, g)
    assert (sc.nobj >= 1).all() and (sc.nobj <= M.MAX_OBJ).all()
    for b in range(200):
        n = int(sc.nobj[b])
        cells = sc.cells[b, :n].tolist()
        assert len(set(cells)) == n and len(set(sc.colors[b, :n].tolist())) == n
        assert (sc.cells[b, n:] == -1).all()
    img = M.render(sc, g, noise=0.0)
    b = 0
    c0, k0 = int(sc.cells[b, 0]), int(sc.colors[b, 0])
    r, c = divmod(c0, M.G)
    px = img[b, :, r * M.PATCH:(r + 1) * M.PATCH, c * M.PATCH:(c + 1) * M.PATCH]
    assert torch.allclose(px, M.RGB[k0][:, None, None].expand_as(px))
    q, a = M.make_qa(sc, "where", g)
    for b in range(20):
        row, col = a[b, 0] - M.ROW0, a[b, 1] - M.COL0
        color = int(q[b, 3]) - M.COLOR0
        j = sc.colors[b].tolist().index(color)
        assert int(sc.cells[b, j]) == int(row) * M.G + int(col)
    q, a = M.make_qa(sc, "count", g)
    assert torch.equal(a[:, 0] - M.NUM0, sc.nobj)


def test_caption_is_sorted_by_cell_and_decodable():
    g = torch.Generator().manual_seed(8)
    sc = M.sample_scenes(30, g)
    for objs, cap in zip(M.sorted_objects(sc), M.caption_tokens(sc)):
        assert [c for c, _ in objs] == sorted(c for c, _ in objs)
        assert len(cap) == 6 * len(objs) + (len(objs) - 1)


# 8. 결정성과 학습이 실제로 되는지
def test_same_seed_same_loss():
    def run():
        torch.manual_seed(7)
        m = small_model()
        g = torch.Generator().manual_seed(7)
        return M.train_stage(m, 2, g, steps=3, batch=8, lr=1e-3)
    assert run() == run()


def test_short_stage2_training_reduces_loss():
    torch.manual_seed(0)
    m = small_model()
    g = torch.Generator().manual_seed(0)
    _, img, ids, labels, *_ = batch(n=16, seed=2)
    before = float(m(img, ids, labels).detach())
    M.train_stage(m, 2, g, steps=60, batch=32, lr=2e-3)
    after = float(m(img, ids, labels).detach())
    assert after < before
