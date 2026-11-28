"""CPU에서 Pythia-14M·70M·160M의 토큰당 디코딩 시간(KV 캐시, 배치 1, 문맥 96)과 k토큰 병렬 검증 시간을 잰다.
논문 3.3절의 비용 계수 c = (Mq 한 번)/(Mp 한 번)과, 병렬 검증이 "공짜"라는 가정(k토큰 호출이 1토큰 호출과 같은 시간)이 이 CPU에서 맞는지 본다.
  - 스레드 4개, 블록 5개 x 반복 40회의 중앙값, 블록 간 범위를 함께 기록. 예시용 측정이며 GPU 수치와 비교하지 않는다.
결과: real_timing_results.json
실행: HF_HUB_OFFLINE=1 python3 real_timing.py  (CPU, 약 1분)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import real_common as R

CTX, KS, BLOCKS, REPS = 96, [1, 2, 3, 4, 6, 9, 12], 5, 40
OUT = Path(__file__).with_name("real_timing_results.json")


@torch.inference_mode()
def time_model(model):
    ids = torch.randint(1000, 20000, (1, CTX)); cache = model(input_ids=ids, use_cache=True).past_key_values   # 반환 객체의 속성 이름
    res = {}
    for k in KS:
        new = torch.randint(1000, 20000, (1, k)); meds = []
        for _ in range(3):                                                      # 예열
            model(input_ids=new, cache=cache, use_cache=True); cache.crop(-k)
        for _ in range(BLOCKS):
            ts = []
            for _ in range(REPS):
                t = time.perf_counter(); model(input_ids=new, cache=cache, use_cache=True); ts.append(time.perf_counter() - t); cache.crop(-k)
            meds.append(float(np.median(ts)))
        res[k] = dict(median_ms=1000 * float(np.median(meds)), block_min_ms=1000 * min(meds), block_max_ms=1000 * max(meds))
    return res


def main():
    out = {"threads": torch.get_num_threads(), "ctx": CTX}
    for size in ("14m", "70m", "160m"):
        out[size] = time_model(R.load_model(size))
        print(size, {k: round(v["median_ms"], 2) for k, v in out[size].items()}, flush=True)
    for tgt in ("70m", "160m"):
        c = out["14m"][1]["median_ms"] / out[tgt][1]["median_ms"]
        lo = out["14m"][1]["block_min_ms"] / out[tgt][1]["block_max_ms"]; hi = out["14m"][1]["block_max_ms"] / out[tgt][1]["block_min_ms"]
        v = {k: out[tgt][k]["median_ms"] / out[tgt][1]["median_ms"] for k in KS}
        out[f"c_14m_over_{tgt}"] = dict(c=c, c_range=[lo, hi], verify_cost_ratio_by_k=v)
        print(tgt, "c", round(c, 3), [round(lo, 3), round(hi, 3)], "v(k)", {k: round(x, 2) for k, x in v.items()})
    # k=1 대 k>=2의 시간 도약이 스레드 수와 무관한지 (70M, 스레드 1·2·4·8)
    m70, chk = R.load_model("70m"), {}
    for th in (1, 2, 4, 8):
        torch.set_num_threads(th)
        with torch.inference_mode():
            cache = m70(input_ids=torch.randint(1000, 20000, (1, CTX)), use_cache=True).past_key_values
            row = {}
            for k in (1, 2, 4, 9):
                new = torch.randint(1000, 20000, (1, k)); ts = []
                for _ in range(REPS):
                    t = time.perf_counter(); m70(input_ids=new, cache=cache, use_cache=True); ts.append(time.perf_counter() - t); cache.crop(-k)
                row[k] = 1000 * float(np.median(ts))
        chk[th] = row
    out["thread_check_70m_median_ms"] = chk
    print("thread check", {th: {k: round(v, 2) for k, v in r.items()} for th, r in chk.items()})
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
