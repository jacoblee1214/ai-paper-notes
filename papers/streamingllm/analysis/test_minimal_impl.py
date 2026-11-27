"""minimal_impl.py 시험. 무작위 초기화한 작은 HF 모델만 쓰므로 가중치 다운로드가 필요 없다(CPU, 수 초).
실행: python3 -m pytest test_minimal_impl.py -q
"""
import pytest
import torch

from minimal_impl import MiniLM, RollingKVCache, nll_recompute, rope

torch.manual_seed(0)
DT = torch.float64


def hf_tiny(kind):
    from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM
    if kind == "neox":
        try:
            cfg = GPTNeoXConfig(vocab_size=97, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
                                intermediate_size=64, max_position_embeddings=256, use_parallel_residual=True,
                                rope_parameters={"rope_theta": 10000, "partial_rotary_factor": 0.25,
                                                 "rope_type": "default"})
        except TypeError:
            cfg = GPTNeoXConfig(vocab_size=97, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
                                intermediate_size=64, max_position_embeddings=256, use_parallel_residual=True,
                                rotary_pct=0.25)
        hf = GPTNeoXForCausalLM(cfg)
    else:
        hf = GPT2LMHeadModel(GPT2Config(vocab_size=97, n_embd=32, n_layer=3, n_head=4, n_positions=128))
    return hf.to(DT).eval()


@pytest.fixture(scope="module", params=["neox", "gpt2"])
def pair(request):
    hf = hf_tiny(request.param)
    return hf, MiniLM.from_hf(hf).to(DT)


def ids(n=2, t=60, v=97):
    return torch.randint(0, v, (n, t))


def test_matches_hf_logits(pair):
    hf, m = pair
    x = ids()
    with torch.no_grad():
        ref = hf(x).logits
    # HF의 RoPE 각도는 float32라 NeoX는 1e-5 수준까지만 맞는다
    assert (m.logits(x) - ref).abs().max() < (1e-5 if m.kind == "neox" else 1e-9)


def test_streaming_equals_dense_when_context_fits(pair):
    """T <= C이면 아무것도 빠지지 않으므로 싱크 수와 상관없이 전체 attention과 같다."""
    _, m = pair
    x = ids(t=40)
    ref = m.logits(x)
    for s in (0, 1, 4):
        assert (m.logits(x, spec=(s, 40 - s)) - ref).abs().max() < 1e-9


@pytest.mark.parametrize("s,w", [(0, 12), (1, 11), (4, 8), (4, 20)])
def test_rolling_cache_matches_parallel(pair, s, w):
    """토큰을 하나씩 넣는 롤링 KV 캐시(회전 전 키를 저장, 매 단계 캐시 안 위치로 회전)가
    한 번에 계산하는 attend()와 같은 로짓을 낸다. 길이 60 >> C이므로 여러 번 뺀다."""
    _, m = pair
    x = ids(t=60)
    modes = ["orig", "slot"] if m.kind == "gpt2" else ["orig"]
    for mode in modes:
        ref = m.logits(x, spec=(s, w), pos_mode=mode)
        c = RollingKVCache(m, s, w, mode)
        out = torch.stack([c.step(x[:, t]) for t in range(x.shape[1])], 1)
        assert (out - ref).abs().max() < 1e-9, mode


def test_cache_keeps_sinks_plus_recent(pair):
    _, m = pair
    s, w = 4, 8
    c = RollingKVCache(m, s, w)
    x = ids(t=50)
    for t in range(50):
        c.step(x[:, t])
        assert len(c.kept) == min(t + 1, s + w) and c.k[0].shape[2] == len(c.kept)
    assert c.kept == [0, 1, 2, 3] + list(range(50 - w, 50))


def test_window_only_equals_banded_attention_with_true_positions():
    """s=0이면 캐시 안 위치로 다시 매기는 것이 원래 위치의 대역(band) 마스크 attention과 같다.
    RoPE는 상대 거리만 보기 때문이다. 대역 마스크는 HF 모델에 4D 마스크로 넣어 독립적으로 만든다."""
    hf = hf_tiny("neox")
    m = MiniLM.from_hf(hf).to(DT)
    x, w = ids(t=50), 12
    i = torch.arange(50)
    band = (i[None] <= i[:, None]) & (i[None] >= i[:, None] - w + 1)
    mask = torch.zeros(1, 1, 50, 50, dtype=DT).masked_fill(~band, float("-inf"))
    with torch.no_grad():
        try:
            ref = hf(x, attention_mask=mask).logits
        except Exception:
            pytest.skip("이 transformers 판은 4D 마스크를 받지 않음")
    assert (m.logits(x, spec=(0, w)) - ref).abs().max() < 1e-4


def test_rope_depends_only_on_relative_position():
    """(R_a q)·(R_b k)는 a-b만의 함수다 -> 윈도 안에서 위치를 평행 이동해도 점수가 같다."""
    q, k = torch.randn(1, 1, 5, 16, dtype=DT), torch.randn(1, 1, 5, 16, dtype=DT)
    p = torch.arange(5)
    s1 = rope(q, p, 8) @ rope(k, p, 8).transpose(-1, -2)
    s2 = rope(q, p + 700, 8) @ rope(k, p + 700, 8).transpose(-1, -2)
    assert (s1 - s2).abs().max() < 1e-9


def test_neox_logits_shift_invariant_in_position():
    """RoPE 모델은 입력 전체의 위치를 같은 만큼 밀어도 로짓이 같다(상대 위치만 쓰므로)."""
    m = MiniLM.from_hf(hf_tiny("neox")).to(DT)
    x = ids(t=30)
    assert (m.logits(x) - m.logits(x, pos0=100)).abs().max() < 1e-9


def test_sink_positions_matter_for_rope():
    """싱크를 원래 위치(0..s-1)로 두고 윈도만 이동시키면 질의-싱크 거리가 캐시 안 거리와 달라져 결과가 바뀐다.
    attend()가 캐시 안 위치를 쓴다는 것은 RollingKVCache와 같다는 위 시험이 보장하고, 여기서는 그 선택이 의미 있음을 확인한다."""
    m = MiniLM.from_hf(hf_tiny("neox")).to(DT)
    x = ids(t=60)
    a = m.logits(x, spec=(4, 8))
    # 윈도가 충분히 커서 싱크가 이미 윈도에 포함되는 경우(4+56=60)와 비교하면 같다
    assert (m.logits(x, spec=(4, 56)) - m.logits(x)).abs().max() < 1e-9
    assert (a - m.logits(x)).abs().max() > 1e-3


def test_recompute_baseline():
    _, m = None, MiniLM.from_hf(hf_tiny("neox")).to(DT)
    x = ids(n=3, t=60)
    pos = [10, 30, 58]
    full = m.nll(x)                                # t < C-1이면 전체 계산과 같아야 함
    r = nll_recompute(m, x, 64, pos)
    assert (r - full[:, pos]).abs().max() < 1e-9
    r = nll_recompute(m, x, 16, pos)               # 윈도 16: 마지막 16토큰을 위치 0부터 다시 계산
    for j, t in enumerate(pos):
        ref = m.nll(x[:, max(0, t - 15):t + 2])[:, -1]
        assert (r[:, j] - ref).abs().max() < 1e-9


def test_gpt2_cannot_use_original_positions_beyond_table():
    m = MiniLM.from_hf(hf_tiny("gpt2")).to(DT)         # n_positions = 128
    x = ids(t=150)
    with pytest.raises(AssertionError):
        m.logits(x, spec=(4, 20), pos_mode="orig")
    out = m.logits(x, spec=(4, 20), pos_mode="slot")   # 넣을 때의 순번은 C-1 = 23을 넘지 않는다
    assert out.shape == (2, 150, 97)


def test_gpt2_position_is_baked_into_cached_kv():
    """GPT-2는 같은 캐시 구성에서도 pos_mode에 따라 결과가 다르다(위치가 K/V에 굳어 있어 캐시 안에서 다시 매길 수 없다)."""
    m = MiniLM.from_hf(hf_tiny("gpt2")).to(DT)
    x = ids(t=60)
    a, b = m.logits(x, spec=(4, 8), pos_mode="orig"), m.logits(x, spec=(4, 8), pos_mode="slot")
    assert (a[:, :12] - b[:, :12]).abs().max() < 1e-9      # 캐시가 차기 전에는 같음
    assert (a[:, 30:] - b[:, 30:]).abs().max() > 1e-4


def test_sink_query_position_ablation():
    """sink_pos="orig"(질의를 원래 위치 t로 회전)는 캐시가 차기 전에는 같고, 찬 뒤에는 달라진다."""
    m = MiniLM.from_hf(hf_tiny("neox")).to(DT)
    x = ids(t=60)
    a = m.logits(x, spec=(4, 8))
    m.sink_pos = "orig"
    b = m.logits(x, spec=(4, 8))
    assert (a[:, :12] - b[:, :12]).abs().max() < 1e-9
    assert (a[:, 20:] - b[:, 20:]).abs().max() > 1e-4
