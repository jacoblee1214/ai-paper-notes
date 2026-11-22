"""PDF 원표의 3인 이진 주석에서 Gwet AC1(쌍별 평균)과 Cohen κ를 나란히 계산한다. 참 비율이 낮을 때 두 값의 차이를 본다.
실행: python3 -I analysis/ac1_check.py   (.scratch/ 에 pdftotext -layout 결과 필요)
"""
import re, os, json, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(os.path.dirname(HERE), ".scratch")


def parse(f, ntr):
    rows = []
    for l in open(os.path.join(SC, f)):
        tk = re.findall(r"\b(TRUE|FALSE)\b", l)
        if tk:
            v = [x == "TRUE" for x in tk]
            rows.append(v + [np.nan] * (3 * ntr - len(v)))
    return np.array(rows, float).reshape(-1, 3)


def pair(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    po = np.mean(x == y)
    pe_k = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    pi = (x.mean() + y.mean()) / 2
    pe_g = 2 * pi * (1 - pi)
    return (po - pe_k) / (1 - pe_k), (po - pe_g) / (1 - pe_g)


res = {}
for name, f, n in [("라운드 2", "interannotator_annotations_round_2_trim.pdf.txt", 5), ("라운드 3", "interannotator_annotations_round3_trim.pdf.txt", 5),
                   ("새 벤치마크 4 trace", "interannotator_annotations_new_benchmarks_trim.pdf.txt", 4)]:
    M = parse(f, n)
    ks, gs = zip(*[pair(M[:, i], M[:, j]) for i, j in itertools.combinations(range(3), 2)])
    res[name] = {"cohen_mean": float(np.mean(ks)), "ac1_mean": float(np.mean(gs)), "pos_rate": float(np.nanmean(M))}
    print(name, {k: round(v, 3) for k, v in res[name].items()})
json.dump(res, open(os.path.join(HERE, "ac1_results.json"), "w"), ensure_ascii=False, indent=1)
