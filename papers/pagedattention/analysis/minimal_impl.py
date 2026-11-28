"""PagedAttention (arXiv:2309.06180) minimal implementation: block-table KV-cache manager
plus a pure-torch paged attention (numerics only, no speed claims).

Pieces
- BlockAllocator: free list + reference counts for physical blocks (one pool per device).
- BlockManager: logical -> physical block tables, allocate / append_slot (with copy-on-write
  on a shared last block) / fork / free / swap_out / swap_in (paper 4.3-4.5).
- PagedKVCache: the actual K/V tensors [num_blocks, B, H, D] and two attention kernels over a
  block table: `paged_attention(mode="blockwise")` follows the paper's block-wise Eq. 4 (global
  softmax denominator), `mode="online"` is the one-pass running-max variant, `mode="gather"`
  materializes the sequence first. All equal dense attention up to float error.
- PagedServer: token-level toy server (FCFS, all-or-nothing preemption by recompute or swap)
  that records memory composition per iteration. It models memory only, not time.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    """Free list of physical block ids with reference counts.

    Memory is O(blocks ever used): ids never handed out are tracked by a counter, freed ids by a
    deque, and reference counts live in a dict (absent = 0).
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._fresh = 0                      # ids >= _fresh were never allocated
        self.free_list: deque[int] = deque() # recycled ids (FIFO)
        self.ref: dict[int, int] = {}

    @property
    def num_free(self) -> int:
        return len(self.free_list) + (self.num_blocks - self._fresh)

    def refcount(self, b: int) -> int:
        return self.ref.get(b, 0)

    def allocate(self) -> int:
        if self.free_list:
            b = self.free_list.popleft()
        elif self._fresh < self.num_blocks:
            b = self._fresh
            self._fresh += 1
        else:
            raise OutOfBlocks
        assert b not in self.ref
        self.ref[b] = 1
        return b

    def retain(self, b: int) -> None:
        assert self.ref[b] > 0
        self.ref[b] += 1

    def release(self, b: int) -> None:
        assert self.ref[b] > 0
        self.ref[b] -= 1
        if self.ref[b] == 0:
            del self.ref[b]
            self.free_list.append(b)


@dataclass
class Sequence:
    seq_id: int
    block_table: list[int] = field(default_factory=list)  # logical block j -> physical block id
    num_tokens: int = 0  # number of token slots filled (last block may be partly filled)
    on_cpu: bool = False


class BlockManager:
    """Logical -> physical block mapping with sharing (paper 4.2-4.5)."""

    def __init__(self, num_gpu_blocks: int, block_size: int, num_cpu_blocks: int = 0):
        self.B = block_size
        self.gpu = BlockAllocator(num_gpu_blocks)
        self.cpu = BlockAllocator(num_cpu_blocks)
        self._next_id = 0

    def new_sequence(self) -> Sequence:
        self._next_id += 1
        return Sequence(self._next_id)

    # ---- allocation -------------------------------------------------------------
    def can_allocate(self, num_tokens: int) -> bool:
        return self.gpu.num_free >= math.ceil(num_tokens / self.B)

    def allocate(self, seq: Sequence, num_tokens: int) -> None:
        """Prompt phase: map ceil(n/B) logical blocks to fresh physical blocks."""
        assert not seq.block_table
        need = math.ceil(num_tokens / self.B)
        if self.gpu.num_free < need:
            raise OutOfBlocks
        seq.block_table = [self.gpu.allocate() for _ in range(need)]
        seq.num_tokens = num_tokens

    def can_append_slot(self, seq: Sequence) -> bool:
        """A new block is needed when the last one is full; a CoW also needs one free block."""
        if seq.num_tokens % self.B == 0:
            return self.gpu.num_free >= 1
        return self.gpu.refcount(seq.block_table[-1]) == 1 or self.gpu.num_free >= 1

    def append_slot(self, seq: Sequence) -> Optional[tuple[int, int]]:
        """Reserve the slot for the next token. Returns (src, dst) if a block copy is needed."""
        copy = None
        if seq.num_tokens % self.B == 0:
            seq.block_table.append(self.gpu.allocate())
        else:
            last = seq.block_table[-1]
            if self.gpu.refcount(last) > 1:  # shared: copy-on-write at block granularity
                dst = self.gpu.allocate()
                self.gpu.release(last)
                seq.block_table[-1] = dst
                copy = (last, dst)
        seq.num_tokens += 1
        return copy

    def fork(self, parent: Sequence) -> Sequence:
        """Child shares every physical block of the parent (parallel sampling, beam search)."""
        child = self.new_sequence()
        child.block_table = list(parent.block_table)
        child.num_tokens = parent.num_tokens
        for b in child.block_table:
            self.gpu.retain(b)
        return child

    def free(self, seq: Sequence) -> None:
        alloc = self.cpu if seq.on_cpu else self.gpu
        for b in seq.block_table:
            alloc.release(b)
        seq.block_table = []
        seq.num_tokens = 0

    # ---- swapping (a sequence group is moved together, sharing preserved) ---------------
    def swap_out(self, seqs: list[Sequence]) -> list[tuple[int, int]]:
        mapping: dict[int, int] = {}
        for s in seqs:
            new_table = []
            for b in s.block_table:
                if b not in mapping:
                    mapping[b] = self.cpu.allocate()
                else:
                    self.cpu.retain(mapping[b])
                new_table.append(mapping[b])
                self.gpu.release(b)
            s.block_table, s.on_cpu = new_table, True
        return list(mapping.items())  # (gpu block, cpu block)

    def swap_in(self, seqs: list[Sequence]) -> list[tuple[int, int]]:
        mapping: dict[int, int] = {}
        for s in seqs:
            new_table = []
            for b in s.block_table:
                if b not in mapping:
                    mapping[b] = self.gpu.allocate()
                else:
                    self.gpu.retain(mapping[b])
                new_table.append(mapping[b])
                self.cpu.release(b)
            s.block_table, s.on_cpu = new_table, False
        return list(mapping.items())  # (cpu block, gpu block)

    # ---- accounting ---------------------------------------------------------------------
    def physical_in_use(self) -> int:
        return self.gpu.num_blocks - self.gpu.num_free


class PagedKVCache:
    """K/V storage for one layer: [num_blocks, B, H, D]."""

    def __init__(self, num_blocks, block_size, heads, dim, dtype=torch.float32, device="cpu"):
        self.B = block_size
        shape = (num_blocks, block_size, heads, dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    def write(self, seq: Sequence, pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        b = seq.block_table[pos // self.B]
        self.k[b, pos % self.B] = k
        self.v[b, pos % self.B] = v

    def write_many(self, seq: Sequence, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        for i in range(k.shape[0]):
            self.write(seq, start + i, k[i], v[i])

    def copy_block(self, src: int, dst: int) -> None:
        self.k[dst] = self.k[src]
        self.v[dst] = self.v[src]

    def gather(self, seq: Sequence):
        n = seq.num_tokens
        idx = torch.tensor(seq.block_table, device=self.k.device)
        k = self.k[idx].reshape(-1, *self.k.shape[2:])[:n]
        v = self.v[idx].reshape(-1, *self.v.shape[2:])[:n]
        return k, v

    def paged_attention(self, q: torch.Tensor, seq: Sequence, mode: str = "blockwise") -> torch.Tensor:
        """q: [H, D] (one query token). Returns [H, D]."""
        H, D = q.shape
        scale = 1.0 / math.sqrt(D)
        n = seq.num_tokens
        if mode == "gather":
            k, v = self.gather(seq)
            a = torch.softmax(torch.einsum("hd,nhd->hn", q, k) * scale, dim=-1)
            return torch.einsum("hn,nhd->hd", a, v)
        blocks = [(b, min(self.B, n - j * self.B)) for j, b in enumerate(seq.block_table)]
        if mode == "blockwise":  # paper Eq. 4: scores per block, one global denominator
            scores = [torch.einsum("hd,bhd->hb", q, self.k[b, :m]) * scale for b, m in blocks]
            a = torch.softmax(torch.cat(scores, dim=-1), dim=-1)
            out, off = torch.zeros_like(q), 0
            for (b, m) in blocks:
                out += torch.einsum("hb,bhd->hd", a[:, off:off + m], self.v[b, :m])
                off += m
            return out
        if mode == "online":  # one pass, running max / denominator
            m_run = torch.full((H,), float("-inf"), dtype=q.dtype, device=q.device)
            l_run = torch.zeros(H, dtype=q.dtype, device=q.device)
            acc = torch.zeros_like(q)
            for b, m in blocks:
                s = torch.einsum("hd,bhd->hb", q, self.k[b, :m]) * scale
                m_new = torch.maximum(m_run, s.max(dim=-1).values)
                corr = torch.exp(m_run - m_new)
                p = torch.exp(s - m_new[:, None])
                l_run = l_run * corr + p.sum(-1)
                acc = acc * corr[:, None] + torch.einsum("hb,bhd->hd", p, self.v[b, :m])
                m_run = m_new
            return acc / l_run[:, None]
        raise ValueError(mode)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q [H,D], k/v [N,H,D] -> [H,D] (reference)."""
    a = torch.softmax(torch.einsum("hd,nhd->hn", q, k) / math.sqrt(q.shape[-1]), dim=-1)
    return torch.einsum("hn,nhd->hd", a, v)


# ------------------------------------------------------------------------------------------
# Token-level toy server (memory only). All requests are queued at t=0 (saturated backlog).
# ------------------------------------------------------------------------------------------
@dataclass
class Req:
    rid: int
    in_len: int
    out_len: int
    generated: int = 0
    seq: Optional[Sequence] = None


@dataclass
class ServerStats:
    iters: int = 0
    sum_tokens: float = 0.0        # live token states
    sum_alloc_slots: float = 0.0   # allocated blocks * B (token states + in-block slack)
    sum_batch: float = 0.0
    preemptions: int = 0
    recomputed_tokens: int = 0
    swapped_blocks: int = 0
    finished: int = 0


class PagedServer:
    """FCFS, iteration-level batching, all-or-nothing preemption (paper 4.5).

    policy="recompute": victim's blocks are freed, the request is re-queued at the head and its
    prompt+generated tokens are prefilled again. policy="swap": victim's blocks move to a CPU
    pool and the request resumes (ahead of new requests) when enough GPU blocks are free.
    """

    def __init__(self, capacity_slots: int, block_size: int, policy: str = "recompute",
                 cpu_slots: int = 10 ** 9, watermark: float = 0.01):
        B = block_size
        self.mgr = BlockManager(capacity_slots // B, B, cpu_slots // B)
        # admit a request only if this fraction of blocks stays free (fewer immediate preemptions)
        self.watermark_blocks = int(watermark * (capacity_slots // B))
        self.policy = policy
        self.waiting: deque[Req] = deque()
        self.swapped: deque[Req] = deque()
        self.running: list[Req] = []
        self.stats = ServerStats()

    def _preempt(self, r: Req) -> None:
        self.running.remove(r)
        self.stats.preemptions += 1
        if self.policy == "swap":
            self.stats.swapped_blocks += len(r.seq.block_table)
            self.mgr.swap_out([r.seq])
            self.swapped.appendleft(r)
        else:
            self.mgr.free(r.seq)
            r.seq = None
            self.waiting.appendleft(r)

    def step(self) -> None:
        mgr, st = self.mgr, self.stats
        # decode: every running request gets one more token slot; preempt the latest arrival
        for r in list(self.running):
            if r not in self.running:
                continue
            while not mgr.can_append_slot(r.seq):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is r:
                    break
            if r not in self.running:
                continue
            mgr.append_slot(r.seq)
            r.generated += 1
        # admit: swapped requests first, then waiting (FCFS)
        while self.swapped and (mgr.gpu.num_free - len(self.swapped[0].seq.block_table) - 1
                                >= self.watermark_blocks):
            r = self.swapped.popleft()
            st.swapped_blocks += len(r.seq.block_table)
            mgr.swap_in([r.seq])
            self.running.append(r)
        while not self.swapped and self.waiting:
            r = self.waiting[0]
            cur = r.in_len + r.generated
            if mgr.gpu.num_free - math.ceil((cur + 1) / mgr.B) < self.watermark_blocks:
                break
            self.waiting.popleft()
            r.seq = mgr.new_sequence()
            mgr.allocate(r.seq, cur)
            if r.generated:
                st.recomputed_tokens += cur
            mgr.append_slot(r.seq)  # prefill also emits one token
            r.generated += 1
            self.running.append(r)
        for r in list(self.running):
            if r.generated >= r.out_len:
                mgr.free(r.seq)
                self.running.remove(r)
                st.finished += 1
        if self.waiting or self.swapped:  # saturated: queue still backed up
            st.iters += 1
            st.sum_tokens += sum(r.seq.num_tokens for r in self.running)
            st.sum_alloc_slots += mgr.physical_in_use() * mgr.B
            st.sum_batch += len(self.running)

    def run(self, reqs: list[Req], max_iters: int = 10 ** 7) -> ServerStats:
        self.waiting.extend(reqs)
        n = 0
        while (self.waiting or self.swapped or self.running) and n < max_iters:
            self.step()
            n += 1
        return self.stats
