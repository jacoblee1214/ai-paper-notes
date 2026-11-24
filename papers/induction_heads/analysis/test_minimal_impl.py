"""minimal_impl.py 시험. 실행: python -m pytest analysis/test_minimal_impl.py -q (torch와 pytest가 설치된 python, CPU, 약 30-50초).

학습이 필요한 시험은 아주 작은 설정(어휘 64, 문맥 64, d=128, 700 스텝)의 attention-only 1층·2층 모델을 한 번만 학습해 같이 쓴다.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_impl import (Cfg, TinyLM, ablation_effects, head_table, induction_target_mask, ov_copy_eig_score,
                          prefix_matching_score, previous_token_score, repeat_batch, repeat_losses, train)

V, T, LMIN, LMAX = 64, 64, 6, 20          # run_experiments.py의 합성 설정과 같음


# ------------------------------------------------------------------ 손으로 만든 패턴
def uniform_causal(T_):
    p = torch.tril(torch.ones(T_, T_))
    return p / p.sum(-1, keepdim=True)


def test_prefix_matching_hand_patterns():
    L, n = 3, 3                                   # BOS + 3x3 = 10 토큰
    T_ = 1 + L * n
    M = induction_target_mask(L, n)
    # 목적지 4..9의 표적: 4->2, 5->3, 6->4, 7->{5,2}, 8->{6,3}, 9->{7,4}
    assert M[4].nonzero().flatten().tolist() == [2]
    assert M[7].nonzero().flatten().tolist() == [2, 5]
    assert M[9].nonzero().flatten().tolist() == [4, 7]
    ideal = M / M.sum(-1, keepdim=True).clamp_min(1)
    ideal[:4] = uniform_causal(T_)[:4]                 # 첫 구간 행은 점수에 안 들어간다
    assert prefix_matching_score(ideal, L, n).item() == pytest.approx(1.0)
    only_bos = torch.zeros(T_, T_)
    only_bos[:, 0] = 1
    assert prefix_matching_score(only_bos, L, n).item() == pytest.approx(0.0)
    prev = torch.zeros(T_, T_)
    prev[0, 0] = 1
    prev[torch.arange(1, T_), torch.arange(0, T_ - 1)] = 1
    assert prefix_matching_score(prev, L, n).item() == pytest.approx(0.0)   # 이전 토큰 head는 prefix matching이 아니다
    assert previous_token_score(prev).item() == pytest.approx(1.0)
    expected = sum(k / (i + 1) for i, k in zip(range(4, 10), [1, 1, 1, 2, 2, 2])) / 6   # 균등 causal attention
    assert prefix_matching_score(uniform_causal(T_), L, n).item() == pytest.approx(expected)


def test_ov_copy_eig_score():
    I = torch.eye(8)
    assert ov_copy_eig_score(I, I, I) == pytest.approx(1.0)
    assert ov_copy_eig_score(I, -I, I) == pytest.approx(-1.0)
    A = torch.randn(8, 8)
    assert ov_copy_eig_score(I, A @ A.T, I) == pytest.approx(1.0)       # 양의 준정부호 -> 고유값 모두 >= 0


def test_repeat_batch_is_periodic_and_L_varies():
    g = torch.Generator().manual_seed(0)
    toks, L = repeat_batch(g, 16, T, V, LMIN, LMAX)
    assert (toks[:, 0] == 0).all() and toks[:, 1:].min() >= 1
    for b in range(16):
        l = int(L[b])
        assert torch.equal(toks[b, 1:T - l], toks[b, 1 + l:T])
    assert len(set(L.tolist())) > 1


# ------------------------------------------------------------------ 구조 성질 (학습 없이)
def randomize(m, std=0.5):
    for p in m.parameters():
        torch.nn.init.normal_(p, std=std)
    return m


def log_ratio(model, toks, i, j, j2, layer):
    _, c = model(toks, cache=True)
    p = c["patterns"][layer][0, 0]
    return (p[i, j].log() - p[i, j2].log()).item()


def test_one_layer_scores_ignore_previous_token():
    """1층에서는 위치 j의 키가 토큰 j만 본다: 토큰 m을 바꿔도 m이 아닌 두 위치의 attention 비는 그대로다(= K-composition 없이는 prefix matching 불가)."""
    torch.manual_seed(0)
    m = randomize(TinyLM(Cfg(vocab=V, d=32, n_layer=1, n_head=2, ctx=T)))
    toks = torch.randint(1, V, (1, 20))
    alt = toks.clone()
    alt[0, 7] = (toks[0, 7] + 5) % V or 1
    a = log_ratio(m, toks, 15, 8, 3, 0)          # j = 8 = m+1, j2 = 3, 둘 다 m=7이 아님
    b = log_ratio(m, alt, 15, 8, 3, 0)
    assert a == pytest.approx(b, abs=1e-5)


def test_two_layer_second_layer_sees_previous_token():
    torch.manual_seed(0)
    m = randomize(TinyLM(Cfg(vocab=V, d=32, n_layer=2, n_head=2, ctx=T)), std=0.3)
    toks = torch.randint(1, V, (1, 20))
    alt = toks.clone()
    alt[0, 7] = (toks[0, 7] + 5) % V or 1
    assert abs(log_ratio(m, toks, 15, 8, 3, 1) - log_ratio(m, alt, 15, 8, 3, 1)) > 1e-3     # 층 1 키는 앞 토큰을 안다
    assert log_ratio(m, toks, 15, 8, 3, 0) == pytest.approx(log_ratio(m, alt, 15, 8, 3, 0), abs=1e-5)


def test_smeared_key_makes_one_layer_scores_depend_on_previous_token():
    """alpha -> -inf이면 k_j' = k_{j-1}: 토큰 m을 바꾸면 위치 m+1의 점수만 변한다."""
    torch.manual_seed(0)
    m = randomize(TinyLM(Cfg(vocab=V, d=32, n_layer=1, n_head=2, ctx=T, smear=True)))
    with torch.no_grad():
        m.blocks[0].attn.alpha.fill_(-30.0)
    toks = torch.randint(1, V, (1, 20))
    alt = toks.clone()
    alt[0, 7] = (toks[0, 7] + 5) % V or 1
    # m=7을 바꾸면 위치 8의 키만 달라진다. 8이 아닌 두 위치(3, 12)의 비는 그대로, 8과의 비는 변한다.
    assert log_ratio(m, toks, 15, 12, 3, 0) == pytest.approx(log_ratio(m, alt, 15, 12, 3, 0), abs=1e-4)
    assert abs(log_ratio(m, toks, 15, 8, 3, 0) - log_ratio(m, alt, 15, 8, 3, 0)) > 1e-3
    # alpha -> +inf이면 smear 없는 모델과 같다
    m2 = TinyLM(Cfg(vocab=V, d=32, n_layer=1, n_head=2, ctx=T, smear=False))
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if "alpha" not in k})
    with torch.no_grad():
        m.blocks[0].attn.alpha.fill_(30.0)
    assert torch.allclose(m(toks), m2(toks), atol=1e-4)


def test_pattern_preserving_ablation_matches_full_in_last_layer():
    torch.manual_seed(0)
    m = randomize(TinyLM(Cfg(vocab=V, d=32, n_layer=2, n_head=2, ctx=T)), std=0.3)
    toks = torch.randint(1, V, (4, 20))
    _, c = m(toks, cache=True)
    fr = [p.detach() for p in c["patterns"]]
    # 마지막 층 헤드: 뒤에 아무것도 없으므로 두 방식이 같다
    assert torch.allclose(m(toks, ablate=(1, 0)), m(toks, ablate=(1, 0), freeze=fr), atol=1e-5)
    # 첫 층 헤드: 아래 층의 attention 패턴 변화를 막으므로 다르다
    assert not torch.allclose(m(toks, ablate=(0, 0)), m(toks, ablate=(0, 0), freeze=fr), atol=1e-3)
    # 아무것도 안 끄고 패턴만 고정하면 깨끗한 실행과 같다
    assert torch.allclose(m(toks), m(toks, freeze=fr), atol=1e-5)


# ------------------------------------------------------------------ 학습이 필요한 시험
@pytest.fixture(scope="module")
def trained():
    out = {}
    gev = torch.Generator().manual_seed(999)
    ev, evL = repeat_batch(gev, 128, T, V, LMIN, LMAX)
    for n_layer in (1, 2):
        torch.manual_seed(0)
        m = TinyLM(Cfg(vocab=V, d=128, n_layer=n_layer, n_head=4, ctx=T))
        g = torch.Generator().manual_seed(1)
        train(m, lambda s: repeat_batch(g, 64, T, V, LMIN, LMAX)[0], 700, 5e-4, warmup=50)
        m.eval()
        with torch.no_grad():
            ht = head_table(m, torch.Generator().manual_seed(7), "cpu", L=12, n_rep=4, hi=V)
            rep, first = repeat_losses(m(ev), ev, evL)
        out[n_layer] = dict(model=m, heads=ht, rep=rep.item(), first=first.item())
    out["ev"], out["evL"] = ev, evL
    return out


def test_two_layer_has_induction_head_one_layer_does_not(trained):
    p1 = trained[1]["heads"]["prefix"].max().item()
    p2 = trained[2]["heads"]["prefix"].max().item()
    assert p2 > 0.5 and p2 > p1 + 0.3
    assert trained[2]["heads"]["prefix"][1].max() > trained[2]["heads"]["prefix"][0].max()      # induction head는 2번째 층


def test_in_context_learning_repeat_loss_below_first(trained):
    r = trained[2]
    assert r["rep"] < r["first"] - 1.5
    assert r["rep"] < trained[1]["rep"] - 0.5               # 2층이 1층보다 반복 구간을 훨씬 잘 맞힌다


def test_ablating_induction_heads_raises_repeat_loss(trained):
    """induction head(prefix 점수 > 0.3인 헤드 전부)를 함께 끄면 반복 구간 손실이 크게 오른다.
    2층 attention-only 토이에서는 마지막 층의 헤드가 모두 induction head라서 하나씩 끄면 효과가 작고(중복),
    묶음으로 꺼야 보인다(논문이 말한 '비슷한 head가 있으면 하나씩 끄는 ablation은 중요도를 가린다')."""
    m, ev, evL = trained[2]["model"], trained["ev"], trained["evL"]
    pm = trained[2]["heads"]["prefix"]
    group = [(l, h) for l in range(pm.shape[0]) for h in range(pm.shape[1]) if pm[l, h] > 0.3]
    assert group and all(l == 1 for l, _ in group)
    ab = ablation_effects(m, ev, lambda lg, t: repeat_losses(lg, t, evL)[0], "cpu", {"ind": group})
    assert ab["groups"]["ind"]["full"] > 1.0 and ab["groups"]["ind"]["preserve"] > 1.0
