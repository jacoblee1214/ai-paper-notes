"""Memory saving from block sharing (paper Fig. 15): parallel sampling and beam search.

saving = (logical blocks - physical blocks) / logical blocks, summed over decoding steps of every
request ("number of blocks saved by sharing / number of blocks without sharing"). Requests are
independent and each is stepped once per iteration, so summing block-steps over requests equals
the time-average over the batch. Lengths: truncated lognormals calibrated to the paper's Fig. 11
means (spread assumed, see fragmentation_sim.py). Block size 16.

Beam search needs a model of how beams pick parents. The paper gives none, so each step the k
survivors pick parents i.i.d. with probability proportional to r**rank (r=1 uniform, small r =
beams collapse onto the best parent; r=-1 is the other extreme, each beam keeps its own lineage so
only the prompt is shared); r is swept and is NOT identified by the paper.
"""
import json
import random

import numpy as np

from fragmentation_sim import calibrated_params, sample_requests, MAX_LEN
from minimal_impl import BlockManager

B = 16
NREQ = {"alpaca": 600, "sharegpt": 150}


def parallel(a, outs):
    m = BlockManager(10 ** 6, B)
    root = m.new_sequence()
    m.allocate(root, a)
    live = [[s, o] for s, o in zip([root] + [m.fork(root) for _ in range(len(outs) - 1)], outs)]
    logical = physical = 0
    while live:
        for s, _ in live:
            m.append_slot(s)
        logical += sum(len(s.block_table) for s, _ in live)
        physical += m.physical_in_use()
        for e in live:
            e[1] -= 1
        for s, o in live:
            if o <= 0:
                m.free(s)
        live = [e for e in live if e[1] > 0]
    return logical, physical


def beam(a, out, k, r, rng):
    m = BlockManager(10 ** 6, B)
    root = m.new_sequence()
    m.allocate(root, a)
    beams = [m.fork(root) for _ in range(k)]
    m.free(root)
    w = np.array([r ** i for i in range(k)])
    w = w / w.sum()
    logical = physical = 0
    for _ in range(out):
        parents = np.arange(k) if r < 0 else rng.choice(k, size=k, p=w)  # r<0: lineages never merge
        children = [m.fork(beams[p]) for p in parents]
        for b in beams:
            m.free(b)
        beams = children
        for b in beams:
            m.append_slot(b)
        logical += sum(len(b.block_table) for b in beams)
        physical += m.physical_in_use()
    return logical, physical


def boot(per_req, n=1000, seed=0):
    x = np.array(per_req, dtype=float)
    rng = np.random.default_rng(seed)
    est = lambda y: 1 - y[:, 1].sum() / y[:, 0].sum()
    bs = [est(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [100 * est(x), 100 * float(np.percentile(bs, 2.5)), 100 * float(np.percentile(bs, 97.5))]


if __name__ == "__main__":
    out = {"block_size": B, "n_requests": NREQ, "parallel": {}, "beam": {}}
    for ds in ("alpaca", "sharegpt"):
        reqs = sample_requests(ds, 1.0, NREQ[ds], 0)
        rng = np.random.default_rng(1)
        for n in (2, 4, 6):
            pr = []
            for i, (a, o) in enumerate(reqs):
                outs = [o] + [x[1] for x in sample_requests(ds, 1.0, n - 1, 1000 + i)]
                outs = [min(x, MAX_LEN - a) for x in outs]
                pr.append(parallel(a, outs))
            out["parallel"][f"{ds}|n={n}"] = boot(pr)
            print(ds, "parallel", n, out["parallel"][f"{ds}|n={n}"], flush=True)
        for k in (2, 4, 6):
            for r in (-1, 0.3, 0.7, 1.0):
                pr = [beam(a, o, k, r, rng) for a, o in reqs[:NREQ[ds] // 3]]
                out["beam"][f"{ds}|k={k}|r={r}"] = boot(pr)
                print(ds, "beam", k, r, out["beam"][f"{ds}|k={k}|r={r}"], flush=True)
    json.dump(out, open("sharing_sim_results.json", "w"), indent=1)
