import math
import random

import pytest
import torch

from minimal_impl import (BlockManager, OutOfBlocks, PagedKVCache, PagedServer, Req,
                          dense_attention)

H, D = 2, 8


def _fill(mgr, cache, seq, toks, g):
    ks, vs = [], []
    for _ in range(toks):
        k, v = torch.randn(H, D, generator=g), torch.randn(H, D, generator=g)
        copy = mgr.append_slot(seq)
        if copy:
            cache.copy_block(*copy)
        cache.write(seq, seq.num_tokens - 1, k, v)
        ks.append(k), vs.append(v)
    return ks, vs


def test_block_table_mapping_and_waste_bound():
    for B in (1, 4, 16):
        for n in (1, 5, 16, 17, 100):
            m = BlockManager(128, B)
            s = m.new_sequence()
            m.allocate(s, n)
            assert len(s.block_table) == math.ceil(n / B)
            assert len(set(s.block_table)) == len(s.block_table)
            assert len(s.block_table) * B - n < B  # waste confined to the last block


def test_free_returns_all_blocks_and_refcounts_zero():
    m = BlockManager(16, 4)
    s = m.new_sequence()
    m.allocate(s, 10)
    c = m.fork(s)
    assert all(m.gpu.refcount(b) == 2 for b in s.block_table)
    m.free(s)
    assert all(m.gpu.refcount(b) == 1 for b in c.block_table)
    m.free(c)
    assert m.gpu.num_free == 16 and not m.gpu.ref


def test_out_of_blocks():
    m = BlockManager(2, 4)
    s = m.new_sequence()
    with pytest.raises(OutOfBlocks):
        m.allocate(s, 9)
    assert not m.can_allocate(9) and m.can_allocate(8)


@pytest.mark.parametrize("B", [1, 3, 16])
@pytest.mark.parametrize("mode", ["gather", "blockwise", "online"])
def test_paged_attention_equals_dense_with_shuffled_blocks(B, mode):
    g = torch.Generator().manual_seed(0)
    nb = 40
    m = BlockManager(nb, B)
    cache = PagedKVCache(nb, B, H, D)
    # scramble the free list so physical order != logical order
    perm = list(range(nb))
    random.Random(1).shuffle(perm)
    m.gpu.free_list.clear()
    m.gpu._fresh = nb  # all ids are 'recycled' ids now, in scrambled order
    m.gpu.free_list.extend(perm)
    s = m.new_sequence()
    ks, vs = _fill(m, cache, s, 37, g)
    assert s.block_table != sorted(s.block_table)
    q = torch.randn(H, D, generator=g)
    ref = dense_attention(q, torch.stack(ks), torch.stack(vs))
    assert torch.allclose(cache.paged_attention(q, s, mode), ref, atol=1e-5)


def test_parallel_sampling_cow_isolates_samples_and_saves_blocks():
    B, P = 4, 10  # prompt 10 tokens: blocks 0,1 full, block 2 holds 2 tokens
    g = torch.Generator().manual_seed(1)
    m = BlockManager(32, B)
    cache = PagedKVCache(32, B, H, D)
    a = m.new_sequence()
    pk, pv = _fill(m, cache, a, P, g)
    a2 = m.fork(a)
    assert m.physical_in_use() == 3  # prompt stored once
    # each sample appends different tokens; the shared partial block is copied on first write
    ka, va = _fill(m, cache, a, 5, g)
    kb, vb = _fill(m, cache, a2, 5, g)
    assert a.block_table[:2] == a2.block_table[:2]  # full prompt blocks stay shared
    assert a.block_table[2] != a2.block_table[2]
    assert m.gpu.refcount(a.block_table[0]) == 2 and m.gpu.refcount(a.block_table[2]) == 1
    q = torch.randn(H, D, generator=g)
    for seq, k, v in ((a, pk + ka, pv + va), (a2, pk + kb, pv + vb)):
        ref = dense_attention(q, torch.stack(k), torch.stack(v))
        assert torch.allclose(cache.paged_attention(q, seq), ref, atol=1e-5)
    # 15 tokens = 4 logical blocks per sample; physical = 2 shared + 2 copies of block 2 + 2 new = 6
    assert len(a.block_table) + len(a2.block_table) == 8 and m.physical_in_use() == 6


def test_cow_not_triggered_when_block_boundary():
    m = BlockManager(16, 4)
    a = m.new_sequence()
    m.allocate(a, 8)  # exactly 2 full blocks
    b = m.fork(a)
    assert m.append_slot(a) is None and m.append_slot(b) is None  # new private blocks, no copy
    assert m.physical_in_use() == 4


def test_beam_search_style_fork_free_keeps_refcounts_consistent():
    rng = random.Random(0)
    m = BlockManager(512, 4)
    root = m.new_sequence()
    m.allocate(root, 9)
    beams = [m.fork(root) for _ in range(4)]
    m.free(root)
    for _ in range(50):
        parents = [rng.randrange(4) for _ in range(4)]
        children = [m.fork(beams[p]) for p in parents]
        for b in beams:
            m.free(b)
        beams = children
        for b in beams:
            m.append_slot(b)
        used = {x for b in beams for x in b.block_table}
        assert m.physical_in_use() == len(used)
        for x in used:
            assert m.gpu.refcount(x) == sum(b.block_table.count(x) for b in beams)
    for b in beams:
        m.free(b)
    assert m.gpu.num_free == 512


def test_swap_round_trip_preserves_content_and_sharing():
    g = torch.Generator().manual_seed(2)
    B = 4
    m = BlockManager(16, B, 16)
    gpu = PagedKVCache(16, B, H, D)
    cpu = PagedKVCache(16, B, H, D)
    a = m.new_sequence()
    ks, vs = _fill(m, gpu, a, 10, g)
    c = m.fork(a)
    q = torch.randn(H, D, generator=g)
    before = gpu.paged_attention(q, a)
    out = m.swap_out([a, c])
    for src, dst in out:
        cpu.k[dst], cpu.v[dst] = gpu.k[src].clone(), gpu.v[src].clone()
    assert m.gpu.num_free == 16 and len(out) == 3  # shared blocks moved once
    gpu.k.zero_(), gpu.v.zero_()
    back = m.swap_in([a, c])
    for src, dst in back:
        gpu.k[dst], gpu.v[dst] = cpu.k[src].clone(), cpu.v[src].clone()
    assert a.block_table == c.block_table  # sharing preserved
    assert torch.allclose(gpu.paged_attention(q, a), before, atol=1e-6)
    assert m.cpu.num_free == 16


@pytest.mark.parametrize("policy", ["recompute", "swap"])
def test_server_conserves_blocks_and_finishes_all(policy):
    rng = random.Random(3)
    reqs = [Req(i, rng.randint(5, 60), rng.randint(5, 80)) for i in range(60)]
    srv = PagedServer(capacity_slots=16 * 12, block_size=4, policy=policy)
    st = srv.run(reqs)
    assert st.finished == 60
    assert all(r.generated == r.out_len for r in reqs)
    assert srv.mgr.gpu.num_free == srv.mgr.gpu.num_blocks
    assert st.preemptions > 0  # capacity is tight on purpose
    if policy == "recompute":
        assert st.recomputed_tokens > 0 and st.swapped_blocks == 0
    else:
        assert st.swapped_blocks > 0 and st.recomputed_tokens == 0


def test_server_internal_fragmentation_bounded_by_block_size():
    rng = random.Random(4)
    B = 16
    reqs = [Req(i, rng.randint(10, 100), rng.randint(10, 100)) for i in range(200)]
    st = PagedServer(capacity_slots=4096, block_size=B).run(reqs)
    slack = (st.sum_alloc_slots - st.sum_tokens) / st.sum_batch
    assert 0 <= slack < B  # per request, waste < one block
