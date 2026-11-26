"""논문 표 재계산 (학습 없음). 표 5·8은 paper.txt에서 읽어 오고, 차이·부호 검정·교차점·평균을 다시 계산한다.
  python3 analysis/table_recalc.py   -> analysis/table_recalc_results.json
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
TXT = (HERE.parent / "paper" / "paper.txt").read_text().split("\n")


def num(s):
    return None if s in ("-", "") else float(s)


def parse_grid(start_marker, batch_re):
    """Table 5/8 행: '<배치> v v v v ...' 형태 줄을 읽는다. 값은 sigmoid, softmax 쌍의 열."""
    i = next(k for k, l in enumerate(TXT) if start_marker in l)
    return i


def rows_between(a, b, ncols):
    out = {}
    for l in TXT[a:b]:
        m = re.match(r"^(\d+ ?k?)\s+(.*)$", l.strip())
        if not m:
            continue
        vals = m.group(2).split()
        if len(vals) == ncols and all(re.fullmatch(r"[\d.]+|-", v) for v in vals):
            lab = m.group(1).replace(" ", "")
            out["0.5k" if lab == "512" else lab] = [num(v) for v in vals]
    return out


i5 = next(k for k, l in enumerate(TXT) if "Table 5: SigLIP" in l)
i8 = next(k for k, l in enumerate(TXT) if "Table 8: SigLiT" in l)
# 표 5는 오른쪽 열의 본문이 같은 줄에 섞여 파싱이 깨져 paper.txt L898-L911 값을 옮겨 적었다. [3B sigmoid, 3B softmax, 9B sigmoid, 9B softmax]
t5 = {"0.5k": [51.5, 47.7, None, None], "1k": [57.3, 53.2, None, None], "2k": [62.1, 59.3, None, None],
      "4k": [65.3, 63.8, 68.4, 66.6], "8k": [68.6, 66.6, 70.6, 69.4], "16k": [None, None, 72.3, 71.7],
      "32k": [69.9, 69.9, 73.4, 72.9], "98k": [69.5, 69.7, 73.0, 73.2], "307k": [None, None, 71.6, 72.6]}
t8 = rows_between(i8 - 22, i8, 8)  # [450M sig, soft, 900M, 3B, 18B ...]
res = {"table5": t5, "table8": t8}


def sign_stats(diffs):
    d = [x for x in diffs if x is not None]
    pos, neg, zero = sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)
    n = pos + neg
    k = max(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k, n + 1)) / 2**n) if n else 1.0  # 양측 부호 검정(동점 제외)
    return {"n": len(d), "sigmoid_better": pos, "softmax_better": neg, "tie": zero, "mean_diff": sum(d) / len(d), "sign_test_p": p}


def diffs_for(table, cols):
    """cols: [(sigmoid 열, softmax 열), ...] -> {열 쌍: [(배치, 차이)]}"""
    out = {}
    for name, (a, b) in cols.items():
        out[name] = [(bs, round(v[a] - v[b], 2)) for bs, v in table.items() if v[a] is not None and v[b] is not None]
    return out


d8 = diffs_for(t8, {"450M": (0, 1), "900M": (2, 3), "3B": (4, 5), "18B": (6, 7)})
d5 = diffs_for(t5, {"3B": (0, 1), "9B": (2, 3)})
res["table8_diffs"], res["table5_diffs"] = d8, d5


def kv(bs):
    return 0.5 if bs == "0.5k" else int(bs.rstrip("k"))


small = lambda ds, thr: [x for bs, x in ds if kv(bs) < thr]
large = lambda ds, thr: [x for bs, x in ds if kv(bs) >= thr]
summ = {}
for tab, dd in (("table8", d8), ("table5", d5)):
    for k, ds in dd.items():
        summ[f"{tab}|{k}"] = {"small(<16k)": sign_stats(small(ds, 16)) if small(ds, 16) else None,
                              "mid(16k-64k)": sign_stats([x for bs, x in ds if 16 <= kv(bs) <= 64]) if [1 for bs, x in ds if 16 <= kv(bs) <= 64] else None,
                              "large(>=128k)": sign_stats(large(ds, 128)) if large(ds, 128) else None}
res["summary"] = summ

# 열마다 최댓값 위치(어느 배치 크기에서 최고인가)
peaks = {}
for name, tab, col in [("T8 18B sigmoid", t8, 6), ("T8 18B softmax", t8, 7), ("T8 3B sigmoid", t8, 4), ("T8 3B softmax", t8, 5),
                       ("T8 900M sigmoid", t8, 2), ("T8 900M softmax", t8, 3), ("T5 9B sigmoid", t5, 2), ("T5 9B softmax", t5, 3)]:
    vals = {bs: v[col] for bs, v in tab.items() if v[col] is not None}
    mx = max(vals.values())
    peaks[name] = {"max": mx, "at": [bs for bs, v in vals.items() if v == mx]}
res["peaks"] = peaks

# 표 7 (시드 5개 표준편차): 32k, 3B에서 차이 0.2 대 sigma
res["table7"] = {"softmax": [69.9, 0.1], "sigmoid_adafactor": [70.1, 0.2], "sigmoid_adamw": [70.3, 0.1],
                 "diff_sigmoid_minus_softmax": 0.2, "note": "표 5의 같은 설정(32k, 3B)은 69.9 대 69.9로 차이 0.0. 표 7과 표 5가 다른 실행이거나 반올림이 다름"}
# 표 1 대 본문: SigLiT B/8 79.8(표, v4) 대 79.7(본문, v1 표도 79.7)
res["table1_text_mismatch"] = {"table1_v4": 79.8, "text_v4": 79.7, "table1_v1": 79.7}

# 표 9: T->I, I->T 36개 언어 평균 재계산
i9 = next(k for k, l in enumerate(TXT) if "Table 9: Image-to-text" in l)
rows = {}
for l in TXT[i9 - 60:i9]:
    m = re.match(r"^\s*([a-z]{2,3})\s+((?:[\d.]+\s+){11}[\d.]+)\s*$", l)
    if m and m.group(1) != "avg":
        rows[m.group(1)] = [float(x) for x in m.group(2).split()]
avg_row = next(l for l in TXT[i9 - 10:i9] if l.strip().startswith("avg")).split()[1:]
avg_row = [float(x) for x in avg_row]
cols = list(zip(*rows.values()))
res["table9"] = {"n_languages": len(rows), "reported_avg": avg_row, "recomputed_avg": [round(sum(c) / len(c), 2) for c in cols],
                 "max_abs_diff": round(max(abs(a - sum(c) / len(c)) for a, c in zip(avg_row, cols)), 3),
                 "tr_avg_16k_to_240k_I2T": avg_row[:5], "tr_avg_16k_to_240k_T2I": avg_row[6:11]}
# 배치 32k 이상에서 영어 외 언어: 16k 대 240k T->I 차이가 있는 언어 수
t2i = {k: v[6:11] for k, v in rows.items()}
res["table9"]["langs_T2I_240k_lt_32k"] = sum(v[4] < v[1] for v in t2i.values())
res["table9"]["langs_T2I_240k_ge_32k"] = sum(v[4] >= v[1] for v in t2i.values())
(HERE / "table_recalc_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("summary", "peaks", "table9")}, indent=1)[:5000])
