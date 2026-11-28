"""보정 문장 수(4, 8, 32개 x 512토큰)에 따른 AWQ(scale만) perplexity. 평가 문서는 run_experiments.py와 같다(보정 32개 기준으로 만든 64문서).
탐색 토큰 수는 보정 문장 수 x 64. 시드 0. 사용법: calib_size.py model out.json"""
import json
import sys

import numpy as np
import torch

import run_experiments as R
from minimal_impl import awq_search, scaled_quantize


def main():
    name, outp = sys.argv[1], sys.argv[2]
    tok = R.AutoTokenizer.from_pretrained(R.MODELS[name])
    model = R.AutoModelForCausalLM.from_pretrained(R.MODELS[name], dtype=torch.float32).eval()
    calib_all, docs, _ = R.build_data(tok)
    lins = R.linears(model, name)
    orig = {id(m): R.get_w(m, conv) for _, _, m, conv in lins}
    res = {}
    for n in (4, 8, 32):
        R.N_CALIB, R.N_SEARCH = n, 64 * n
        stat, rows = R.collect_stats(model, lins, calib_all[:n])
        for nb in (4, 3):
            alphas = []
            for _, _, m, conv in lins:
                r = awq_search(orig[id(m)], rows[id(m)], nb, R.GS, x_stat=stat[id(m)].view(1, -1))
                alphas.append(r["alpha"])
                R.set_w(m, conv, scaled_quantize(orig[id(m)], r["scales"], nb, R.GS))
            nll, cnt = R.eval_docs(model, docs)
            for _, _, m, conv in lins:
                R.set_w(m, conv, orig[id(m)])
            res[f"n{n}_w{nb}"] = dict(nll=nll, cnt=cnt, alpha_mean=float(np.mean(alphas)))
            print(name, n, nb, "ppl %.3f alpha %.3f" % (np.exp(sum(nll) / sum(cnt)), np.mean(alphas)), flush=True)
        del rows
    json.dump(res, open(outp, "w"))


if __name__ == "__main__":
    main()
