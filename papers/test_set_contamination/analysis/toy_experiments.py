"""토이 n-gram LM에서 교환성 검정의 1종 오류율과 검정력을 잰다. 결과: analysis/toy_results.json

실험
  E1  오염 없음(k=0)에서 p값 분포와 1종 오류율 (순열 검정 M=25, 샤드 검정 r=10 m=25, 공식 코드식 ddof=0)
  E2  정준 순서로 k번 섞어 넣은 오염: k = 0,1,2,4,8,16에서 검정력 (같은 계산량 m=25, 순열 M=100은 4배 계산량)
  E3  복사본마다 순서를 새로 섞어 넣은 경우(정준 순서 없음): 검정은 검출하면 안 됨
  E4  샤드 수 r과 샤드당 순열 수 m에 대한 민감도 (k=4)

토이 설정(논문 값이 아님): V=256, 벤치마크 200문항(길이 6-12), 배경 10만 토큰, 4-gram, 고차 n-gram 출현 보존율 0.01.
실행: python3 analysis/toy_experiments.py  (약 2-3분, CPU 병렬)
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_impl import (NgramLM, build_stream, make_background, make_benchmark,  # noqa: E402
                          permutation_test, permutation_z_test, sharded_test)

V, N_ITEMS, BG_TOKENS, ORDER, RETAIN = 256, 200, 100_000, 4, 0.01
R, M = 10, 25


def one_trial(args):
    seed, k, canonical, r, m, extras = args
    rng = np.random.default_rng(seed)
    bench = make_benchmark(rng, N_ITEMS, V)
    stream = build_stream(rng, make_background(rng, BG_TOKENS, V), bench, k, canonical)
    lm = NgramLM(V, ORDER, RETAIN, seed).fit(stream)
    items = bench if canonical else [bench[i] for i in rng.permutation(N_ITEMS)]
    ll = lm.loglik_items
    sh = sharded_test(ll, items, r, m, rng)
    s = sh["s"]
    t0 = s.mean() / s.std(ddof=0) * np.sqrt(len(s))
    out = {"sharded": sh["p"], "sharded_ddof0": float(stats.t.sf(t0, df=len(s) - 1))}
    if extras:
        out["perm25"] = permutation_test(ll, items, m, rng)["p"]
        out["perm100"] = permutation_test(ll, items, 4 * m, rng)["p"]
        out["perm_z"] = permutation_z_test(ll, items, m, rng)["p"]
    return out


def run(pool, k, canonical, trials, seed0, r=R, m=M, extras=True):
    res = list(pool.map(one_trial, [(seed0 + i, k, canonical, r, m, extras) for i in range(trials)], chunksize=4))
    keys = res[0].keys()
    return {key: [x[key] for x in res] for key in keys}


def summarize(ps):
    ps = np.asarray(ps)
    return {"median_p": float(np.median(ps)), "median_log10p": float(np.median(np.log10(np.maximum(ps, 1e-300)))),
            "rej05": float((ps <= 0.05).mean()), "rej01": float((ps <= 0.01).mean())}


def main():
    out = {"setup": dict(V=V, n_items=N_ITEMS, bg_tokens=BG_TOKENS, order=ORDER, retain=RETAIN, r=R, m=M)}
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        out["E1_null"] = run(pool, 0, True, 1000, 10_000)
        print("E1", time.time() - t0, flush=True)
        out["E2_power"] = {}
        for k in [0, 1, 2, 4, 8, 16]:
            raw = run(pool, k, True, 200, 20_000 + 1000 * k)
            out["E2_power"][str(k)] = {name: summarize(v) for name, v in raw.items()}
            out["E2_power"][str(k)]["raw"] = raw
            print("E2", k, time.time() - t0, flush=True)
        out["E3_unordered"] = {}
        for k in [4, 16]:
            raw = run(pool, k, False, 1000, 100_000 + 2000 * k)
            out["E3_unordered"][str(k)] = {name: summarize(v) for name, v in raw.items()}
            print("E3", k, time.time() - t0, flush=True)
        out["E4_sensitivity"] = {"vs_shards": {}, "vs_perms": {}}
        for r in [2, 5, 10, 20, 50]:
            raw = run(pool, 4, True, 100, 60_000, r=r, m=M, extras=False)
            out["E4_sensitivity"]["vs_shards"][str(r)] = summarize(raw["sharded"])
        for m in [1, 2, 5, 10, 25, 50]:
            raw = run(pool, 4, True, 100, 60_000, r=R, m=m, extras=False)
            out["E4_sensitivity"]["vs_perms"][str(m)] = summarize(raw["sharded"])
        print("E4", time.time() - t0, flush=True)
    Path(__file__).with_name("toy_results.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
