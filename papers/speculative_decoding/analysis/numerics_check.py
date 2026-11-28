"""'출력 분포가 같다'는 정확한 산술에서의 보장이다. 실제로는 검증 호출(접두사 + 추측 5개를 한 번에)과 일반 디코딩(접두사만)이
같은 위치에서 부동소수점으로 조금 다른 분포를 낸다. 그 크기를 Pythia-70M으로 잰다(인과 모델이라 수학적으로 같아야 하는 값).
  - 입력: pile-10k 문서 100개의 앞 40토큰. 위치 31(32토큰 접두사 뒤)의 다음 토큰 분포를
    (a) 32토큰만 넣은 호출, (b) 40토큰을 넣은 호출(위치 31의 출력)에서 구해 비교
  - 같은 비교를 KV 캐시 경로로도: (c) 캐시에 31토큰을 넣고 토큰 31 하나만 넣는 호출(k=1), (d) 토큰 31..39 아홉 개를 넣는 호출(k=9)의 첫 출력
  - float32와 bfloat16, 지표: TV, 최대 확률의 상대 차이, argmax가 달라진 비율
결과: numerics_results.json   실행: HF_HUB_OFFLINE=1 python3 numerics_check.py (CPU, 약 20초)
"""
import json
from pathlib import Path

import torch

import real_common as R

OUT = Path(__file__).with_name("numerics_results.json")


@torch.inference_mode()
def main():
    tok = R.load_tok()
    ids = R.load_prompts(tok, 100, 40, seed=0)
    res = {}
    for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        m = R.load_model("70m").to(dtype)
        pairs = {k: ([], [], 0) for k in ("nocache_32_vs_40", "cache_k1_vs_k9")}
        for i in range(ids.shape[0]):
            x = ids[i : i + 1]
            outs = {}
            outs["nocache_32_vs_40"] = (m(input_ids=x[:, :32]).logits[0, -1], m(input_ids=x).logits[0, 31])
            c1 = m(input_ids=x[:, :31], use_cache=True).past_key_values
            a = m(input_ids=x[:, 31:32], cache=c1, use_cache=True).logits[0, 0]
            c2 = m(input_ids=x[:, :31], use_cache=True).past_key_values
            b = m(input_ids=x[:, 31:40], cache=c2, use_cache=True).logits[0, 0]
            outs["cache_k1_vs_k9"] = (a, b)
            for k, (la, lb) in outs.items():
                pa, pb = torch.softmax(la.float(), -1), torch.softmax(lb.float(), -1)
                tv, rel, fl = pairs[k]
                tv.append(0.5 * (pa - pb).abs().sum().item()); rel.append((pa.max() - pb.max()).abs().item() / pa.max().item())
                pairs[k] = (tv, rel, fl + int(pa.argmax() != pb.argmax()))
        res[name] = {}
        for k, (tv, rel, fl) in pairs.items():
            t = torch.tensor(tv)
            res[name][k] = dict(n=ids.shape[0], tv_mean=t.mean().item(), tv_max=t.max().item(), pmax_rel_diff_mean=float(torch.tensor(rel).mean()),
                                pmax_rel_diff_max=float(torch.tensor(rel).max()), argmax_flips=fl)
        print(name, res[name])
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
