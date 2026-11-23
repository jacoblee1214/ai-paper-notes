"""03 §3의 설명 확인: 샤드 t 검정이 전체 순열 정규 근사보다 약한 이유는 t 통계량이 H1 아래에서 s_i의 표본 표준편차
(= 귀무 순열 변동 + 샤드 간 이득 이질성)로 나누기 때문이다. 결과: analysis/toy_variance_results.json
실행: python3 analysis/toy_variance_check.py (약 20초)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_impl import NgramLM, build_stream, make_background, make_benchmark, split_shards  # noqa: E402

V, N, R, M = 256, 200, 10, 25
out = {}
for k in (0, 4, 16):
    rows = []
    for seed in range(40):
        rng = np.random.default_rng(777_000 + 100 * k + seed)
        bench = make_benchmark(rng, N, V)
        lm = NgramLM(V, 4, 0.01, seed).fit(build_stream(rng, make_background(rng, 100_000, V), bench, k, True))
        s, tau = [], []
        for shard in split_shards(bench, R):
            lc = lm.loglik_items(shard)
            lp = np.array([lm.loglik_items([shard[i] for i in rng.permutation(len(shard))]) for _ in range(M)])
            s.append(lc - lp.mean())
            tau.append(lp.std(ddof=1))
        s, tau = np.array(s), np.array(tau)
        null_sd = np.sqrt(np.mean(tau ** 2) * (1 + 1 / M))
        rows.append({"mean_s": s.mean(), "sd_s": s.std(ddof=1), "null_sd": null_sd,
                     "t_sharded": s.mean() / s.std(ddof=1) * np.sqrt(R), "z_null_normalized": s.mean() / null_sd * np.sqrt(R)})
    out[str(k)] = {key: float(np.median([r[key] for r in rows])) for key in rows[0]}
    out[str(k)]["median_sd_over_null_sd"] = float(np.median([r["sd_s"] / r["null_sd"] for r in rows]))
Path(__file__).with_name("toy_variance_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
