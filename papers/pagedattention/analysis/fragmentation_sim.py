"""KV-cache memory composition: contiguous reservation (Orca-style, buddy allocator) vs paged.

Setting taken from the paper: OPT-13B on one A100, KV-cache pool = 15.7K token slots (Table 1;
12 GiB / 800 KiB per token), max length 2048, requests queued at t=0 (saturated backlog), FCFS,
iteration-level batching. Length distributions are NOT the ShareGPT/Alpaca files (not
downloaded): they are truncated lognormals whose means are calibrated to the paper's Fig. 11
(ShareGPT in/out 161.31/337.99, Alpaca 19.31/58.45); the spread (CV) is an assumption and is
varied (CV_SCALE). Memory model only: no time, no compute.

Categories (paper Fig. 3): token states | reservation (slots the request will fill later) |
internal fragmentation (reserved but never used) | external fragmentation & others (buddy
rounding, free-but-unusable, unallocated).
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import deque
from multiprocessing import Pool

import numpy as np
from scipy import stats

from minimal_impl import PagedServer, Req

CAPACITY = 15728          # 12 GiB / 819200 B per token (OPT-13B)
MAX_LEN = 2048
N_REQ = {"sharegpt": 600, "alpaca": 3000}  # per trace; Alpaca needs more so the backlog outlasts the batch
SEEDS = list(range(20))
PAPER_MEAN = {"sharegpt": (161.31, 337.99), "alpaca": (19.31, 58.45)}
BASE_CV = {"sharegpt": (2.0, 1.2), "alpaca": (0.8, 1.2)}  # assumed spread (in, out)


def _trunc_lognormal(rng, mean, cv, size):
    s2 = math.log(1 + cv * cv)
    mu = math.log(mean) - s2 / 2
    return np.maximum(1, np.round(rng.lognormal(mu, math.sqrt(s2), size))).astype(int)


_CAL: dict = {}


def calibrated_params(ds: str, cv_scale: float):
    """Find lognormal means so the *truncated, rounded* sample means match the paper's means."""
    key = (ds, cv_scale)
    if key in _CAL:
        return _CAL[key]
    target = PAPER_MEAN[ds]
    cvs = tuple(c * cv_scale for c in BASE_CV[ds])
    m = list(target)
    rng = np.random.default_rng(12345)
    for _ in range(25):
        a = _trunc_lognormal(rng, m[0], cvs[0], 200000)
        b = _trunc_lognormal(rng, m[1], cvs[1], 200000)
        ok = a + b <= MAX_LEN
        got = (a[ok].mean(), b[ok].mean())
        m = [m[i] * target[i] / got[i] for i in range(2)]
    _CAL[key] = (m, cvs, got)
    return _CAL[key]


def sample_requests(ds, cv_scale, n, seed):
    m, cvs, _ = calibrated_params(ds, cv_scale)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        a = _trunc_lognormal(rng, m[0], cvs[0], 1)[0]
        b = _trunc_lognormal(rng, m[1], cvs[1], 1)[0]
        if a + b <= MAX_LEN:
            out.append((int(a), int(b)))
    return out


# ------------------------------------------------------------------ buddy allocator
class Buddy:
    def __init__(self, capacity):
        self.free = {}  # order -> set(offset)
        off = 0
        for k in range(capacity.bit_length() - 1, -1, -1):
            if capacity >> k & 1:
                self.free.setdefault(k, set()).add(off)
                off += 1 << k

    def alloc(self, size):
        k = max(0, (size - 1).bit_length())
        j = k
        while j < 40 and not self.free.get(j):
            j += 1
        if j >= 40:
            return None
        off = min(self.free[j])
        self.free[j].remove(off)
        while j > k:
            j -= 1
            self.free.setdefault(j, set()).add(off + (1 << j))
        return off, k

    def release(self, off, k):
        while True:
            buddy = off ^ (1 << k)
            if buddy in self.free.get(k, ()):
                self.free[k].remove(buddy)
                off = min(off, buddy)
                k += 1
            else:
                self.free.setdefault(k, set()).add(off)
                return


def reserve_size(system, a, b):
    if system == "max":
        return MAX_LEN
    if system == "pow2":
        return min(MAX_LEN, a + (1 << (b - 1).bit_length()))
    if system == "oracle":
        return a + b
    raise ValueError(system)


def run_contiguous(system, reqs):
    buddy = Buddy(CAPACITY)
    waiting = deque(reqs)
    running = []  # [in, out, generated, off, k, reserve]
    tok = resv = intf = batch = iters = 0.0
    while waiting or running:
        for r in running:
            r[2] += 1
        for r in [r for r in running if r[2] >= r[1]]:
            buddy.release(r[3], r[4])
            running.remove(r)
        while waiting:  # FCFS with head-of-line blocking
            a, b = waiting[0]
            res = reserve_size(system, a, b)
            got = buddy.alloc(res)
            if got is None:
                break
            waiting.popleft()
            running.append([a, b, 1, got[0], got[1], res])  # prefill emits token 1
        if waiting:
            iters += 1
            cur = sum(r[0] + r[2] for r in running)
            tok += cur
            resv += sum(r[0] + r[1] - (r[0] + r[2]) for r in running)
            intf += sum(r[5] - (r[0] + r[1]) for r in running)
            batch += len(running)
    t = iters * CAPACITY
    return {"token": tok / t, "reservation": resv / t, "internal": intf / t,
            "external_others": 1 - (tok + resv + intf) / t, "batch": batch / iters}


def run_paged(B, reqs, policy="recompute"):
    srv = PagedServer(CAPACITY, B, policy)
    st = srv.run([Req(i, a, b) for i, (a, b) in enumerate(reqs)])
    t = st.iters * (CAPACITY // B) * B
    gen = sum(b for _, b in reqs)
    return {"token": st.sum_tokens / t, "reservation": 0.0,
            "internal": (st.sum_alloc_slots - st.sum_tokens) / t,
            "external_others": 1 - st.sum_alloc_slots / t, "batch": st.sum_batch / st.iters,
            "preempt_per_req": st.preemptions / len(reqs),
            "recomputed_frac": st.recomputed_tokens / gen,
            "swapped_blocks_per_req": st.swapped_blocks / len(reqs),
            "slack_per_seq": (st.sum_alloc_slots - st.sum_tokens) / st.sum_batch}


def job(args):
    ds, cv_scale, system, seed = args
    reqs = sample_requests(ds, cv_scale, N_REQ[ds], seed)
    if system in ("max", "pow2", "oracle"):
        r = run_contiguous(system, reqs)
    else:
        B, policy = system
        r = run_paged(B, reqs, policy)
    return (ds, cv_scale, system if isinstance(system, str) else f"paged_B{system[0]}_{system[1]}", seed, r)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


if __name__ == "__main__":
    BLOCKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    jobs = []
    for ds in ("sharegpt", "alpaca"):
        for cvs in (1.0,):
            for sysm in ("max", "pow2", "oracle"):
                jobs += [(ds, cvs, sysm, s) for s in SEEDS]
            jobs += [(ds, cvs, (B, "recompute"), s) for B in BLOCKS for s in SEEDS]
            jobs += [(ds, cvs, (B, "swap"), s) for B in (16,) for s in SEEDS]
        for cvs in (0.6, 1.4):  # sensitivity to the assumed spread
            for sysm in ("max", "pow2", "oracle"):
                jobs += [(ds, cvs, sysm, s) for s in SEEDS[:10]]
            jobs += [(ds, cvs, (16, "recompute"), s) for s in SEEDS[:10]]
    for ds in PAPER_MEAN:  # calibrate once, before forking
        for cvs in (0.6, 1.0, 1.4):
            calibrated_params(ds, cvs)
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    table: dict = {}
    for ds, cvs, name, seed, r in res:
        table.setdefault((ds, cvs, name), {})[seed] = r
    out = {"config": {"capacity_slots": CAPACITY, "max_len": MAX_LEN, "n_req_per_trace": N_REQ,
                      "seeds": SEEDS, "base_cv(in,out)": BASE_CV, "paper_means": PAPER_MEAN},
           "calibration": {}, "summary": {}, "tests": {}}
    for ds in PAPER_MEAN:
        for cvs in (0.6, 1.0, 1.4):
            m, cv, got = calibrated_params(ds, cvs)
            out["calibration"][f"{ds}_cvscale{cvs}"] = {"lognormal_means": m, "cv": cv,
                                                         "truncated_sample_means": list(map(float, got))}
    for (ds, cvs, name), d in sorted(table.items(), key=str):
        seeds = sorted(d)
        row = {}
        for key in d[seeds[0]]:
            row[key] = boot_ci([d[s][key] for s in seeds])
        row["n_traces"] = len(seeds)
        out["summary"][f"{ds}|cv{cvs}|{name}"] = row
    # paired Wilcoxon: paged B=16 vs each contiguous baseline, token fraction (CV scale 1.0)
    ps, labels = [], []
    for ds in PAPER_MEAN:
        v = table[(ds, 1.0, "paged_B16_recompute")]
        for base in ("max", "pow2", "oracle"):
            b = table[(ds, 1.0, base)]
            x = [v[s]["token"] - b[s]["token"] for s in SEEDS]
            ps.append(stats.wilcoxon(x).pvalue)
            labels.append(f"{ds}: paged_B16 - {base}")
    adj = holm(np.array(ps))
    out["tests"] = {l: {"p": float(p), "p_holm": float(a)} for l, p, a in zip(labels, ps, adj)}
    json.dump(out, open("fragmentation_sim_results.json", "w"), indent=1)
    for k, r in out["summary"].items():
        if "cv1.0" in k:
            print(k, {a: round(r[a][0], 3) for a in ("token", "reservation", "internal", "external_others", "batch")})
