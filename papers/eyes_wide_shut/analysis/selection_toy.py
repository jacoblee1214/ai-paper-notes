"""선별 효과 토이 (numpy, CPU 수 초): "인코더 A에서는 비슷하고 인코더 B에서는 다른 쌍"을 고르면 그 쌍으로 잰 점수가 어떻게 되나.

논문 §2.1은 CLIP ViT-L/14 코사인 > 0.95 이고 DINOv2 ViT-L/14 코사인 < 0.6 인 쌍을 고른다. 이 토이는 그 구조만 따온다.
- 이미지 = (의미 s: 군집 중심 + 작은 변동, 세부 d: 자유 변동). 속성 라벨 = d의 첫 좌표 부호(예: 방향).
- 인코더 A(CLIP 비슷) = 의미 위주 + d를 조금, B(DINO 비슷) = d 위주 + 의미 조금, C(선별에 쓰지 않은 제3 인코더) = 둘을 같은 비중으로 섞음. 모두 무작위 회전과 잡음 포함.
- 읽기: 무작위 쌍으로 학습한 선형 읽기(두 임베딩 차이 -> 속성 차이 부호)를 선별한 쌍에서 평가한다.
S1 (A 비슷, B 다름)으로 선별: A, B, C, A+B, A+C 정확도. 선별 없는 쌍과 비교.
S2 (A 비슷, C 다름)으로 선별: 같은 비교. 선별에 쓴 인코더가 얼마나 유리해지는지가 S1 대 S2의 'B 이득' 차이.
시드 40번 반복해 평균과 표준편차를 낸다. 토이의 목적은 크기를 맞추는 것이 아니라 방향(선별에 쓴 쪽의 이득이 구조적으로 커지나)을 보는 것이다.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DS, DD = 6, 6
K, PER = 1500, 20
SEEDS = 40


def cos(u, v):
    return (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)


def run(seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, (K, DS))
    s = np.repeat(centers, PER, axis=0) + rng.normal(0, 0.15, (K * PER, DS))
    d = rng.normal(0, 1.0, (K * PER, DD))
    x = np.concatenate([s, d], 1)
    Q = [np.linalg.qr(rng.normal(size=(DS + DD, DS + DD)))[0] for _ in range(3)]
    W = {"A": np.r_[np.ones(DS) * 1.0, np.ones(DD) * 0.3], "B": np.r_[np.ones(DS) * 0.4, np.ones(DD) * 1.0], "C": np.r_[np.ones(DS) * 0.7, np.ones(DD) * 0.7]}
    E = {k: (x * W[k]) @ Q[n] + rng.normal(0, 0.6, x.shape) for n, k in enumerate("ABC")}
    i = np.repeat(np.arange(K) * PER, PER // 2) + np.tile(np.arange(PER // 2), K)
    j = i + PER // 2
    label = np.sign(d[i, 0] - d[j, 0])
    sim = {k: cos(E[k][i], E[k][j]) for k in E}

    def feats(names, idx):
        return np.concatenate([E[n][i[idx]] - E[n][j[idx]] for n in names], 1)

    perm = rng.permutation(len(i))
    train, rest = perm[: len(perm) // 2], perm[len(perm) // 2:]

    def fit(names):
        X = feats(names, train)
        return np.linalg.lstsq(np.c_[X, np.ones(len(X))], label[train], rcond=None)[0]

    combos = [["A"], ["B"], ["C"], ["A", "B"], ["A", "C"]]
    Ws = {"+".join(c): fit(c) for c in combos}

    def ev(idx):
        out = {}
        for n, c in zip(Ws, combos):
            Xt = feats(c, idx)
            out[n] = float((np.sign(np.c_[Xt, np.ones(len(Xt))] @ Ws[n]) == label[idx]).mean() * 100)
        return out

    def select(other, qa=0.7, qo=0.3):
        ta, to = np.quantile(sim["A"][rest], qa), np.quantile(sim[other][rest], qo)
        return rest[(sim["A"][rest] > ta) & (sim[other][rest] < to)]

    sb, sc = select("B"), select("C")
    return {"random": ev(rest), "selB": ev(sb), "selC": ev(sc), "n_selB": len(sb), "n_selC": len(sc),
            "cosA_selB": float(sim["A"][sb].mean()), "cosB_selB": float(sim["B"][sb].mean())}


runs = [run(s) for s in range(SEEDS)]
names = ["A", "B", "C", "A+B", "A+C"]
out = {"seeds": SEEDS, "n_selected_mean": {"selB": float(np.mean([r["n_selB"] for r in runs])), "selC": float(np.mean([r["n_selC"] for r in runs]))}}
for cond in ("random", "selB", "selC"):
    out[cond] = {n: {"mean": round(float(np.mean([r[cond][n] for r in runs])), 1), "sd": round(float(np.std([r[cond][n] for r in runs])), 1)} for n in names}
gain = {}
for cond in ("random", "selB", "selC"):
    for add in ("A+B", "A+C"):
        v = np.array([r[cond][add] - r[cond]["A"] for r in runs])
        gain[f"{cond}: {add} - A"] = {"mean": round(float(v.mean()), 1), "sd": round(float(v.std()), 1)}
out["gain_over_A_pts"] = gain
# 핵심: B로 선별했을 때 'A+B 이득' 대 C로 선별했을 때 'A+B 이득' (같은 인코더, 선별 기준만 다름)
d1 = np.array([(r["selB"]["A+B"] - r["selB"]["A"]) - (r["selC"]["A+B"] - r["selC"]["A"]) for r in runs])
d2 = np.array([(r["selC"]["A+C"] - r["selC"]["A"]) - (r["selB"]["A+C"] - r["selB"]["A"]) for r in runs])
out["selection_inflation_pts"] = {"A+B gain: selB minus selC": {"mean": round(float(d1.mean()), 1), "sd": round(float(d1.std()), 1)},
                                  "A+C gain: selC minus selB": {"mean": round(float(d2.mean()), 1), "sd": round(float(d2.std()), 1)}}
out["cos_in_selB"] = {"A": round(float(np.mean([r["cosA_selB"] for r in runs])), 3), "B": round(float(np.mean([r["cosB_selB"] for r in runs])), 3)}
(HERE / "selection_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
