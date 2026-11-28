"""진단: AWQ 탐색 행에서 각 문장의 첫 토큰(위치 0)을 빼면 scale-only AWQ perplexity가 달라지는가. 보정 32문장, 탐색 2048행, 시드 0.
사용법: pos0_check.py model out.json"""
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
    calib, docs, _ = R.build_data(tok)
    lins = R.linears(model, name)
    orig = {id(m): R.get_w(m, conv) for _, _, m, conv in lins}
    res = {}
    for skip in (False, True):
        stat, rows = R.collect_stats(model, lins, calib, skip_first=skip)
        # 첫 토큰 행이 탐색 행에 몇 개 들어 있는지는 위치를 따로 기록하지 않아 계산하지 않고, 손실 기여만 본다.
        for nb in (4, 3):
            for _, _, m, conv in lins:
                r = awq_search(orig[id(m)], rows[id(m)], nb, R.GS, x_stat=stat[id(m)].view(1, -1))
                R.set_w(m, conv, scaled_quantize(orig[id(m)], r["scales"], nb, R.GS))
            nll, cnt = R.eval_docs(model, docs)
            for _, _, m, conv in lins:
                R.set_w(m, conv, orig[id(m)])
            res[f"skip{int(skip)}_w{nb}"] = dict(nll=nll, cnt=cnt)
            print(name, "skip_pos0" if skip else "all_rows", nb, "ppl %.3f" % np.exp(sum(nll) / sum(cnt)), flush=True)
        del rows
    json.dump(res, open(outp, "w"))


if __name__ == "__main__":
    main()
