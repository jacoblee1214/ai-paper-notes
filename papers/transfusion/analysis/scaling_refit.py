"""그림 5의 체크포인트 점에서 Transfusion·Chameleon 거듭제곱 법칙을 다시 맞추고 동등 FLOP 비율(표 3)을 재계산한다.

입력: figure_values.json (figure_extract.py가 PDF 벡터에서 읽은 점: [log10 FLOPs, 지표])
방법: ln(지표) = a + b ln(FLOPs) 최소제곱(논문 §4.2: "log-metric over log-FLOPs 선형 추세선").
동등 FLOP 비율 = Transfusion이 Chameleon 7B와 같은 지표에 도달하는 데 필요한 FLOPs / Chameleon 7B의 FLOPs (6ND).
  변형 A: 목표값 = 표 3의 Chameleon 7B 실측값, 변형 B: 목표값 = Chameleon 맞춤선이 7B FLOPs에서 주는 값.
불확실성: (1) 점 부트스트랩(25점을 독립으로 봄, 과소추정), (2) 모델 크기 군집 부트스트랩(5개 학습 줄기를 단위로 재표집),
         (3) 잔차 부트스트랩(맞춤선 + 재표집 잔차).
실행: python3 scaling_refit.py   (numpy만, 수 초)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "figure_values.json").read_text())
rng = np.random.default_rng(0)
LN10 = np.log(10.0)

# 표 3 (7B, 0.5T 토큰): 지표별 Transfusion, Chameleon, 논문의 동등 FLOP 비율
T3 = {
    "c4_ppl": (7.72, 8.41, 0.489), "wiki_ppl": (4.28, 4.69, 0.526), "llama_acc": (61.5, 59.1, 0.600),
    "coco_cider": (27.2, 18.0, 0.218), "coco_fid": (16.8, 29.6, 0.029), "coco_clip": (25.5, 24.3, 0.319),
}
# 7B Llama형 구성의 파라미터: 32층, d=4096, ffn=11008, 어휘 32000 (입력·출력 임베딩 분리) -> 6.74B
N7 = 32 * (4 * 4096**2 + 3 * 4096 * 11008) + 2 * 32000 * 4096
X7 = 6 * N7 * 0.5e12  # 6ND, D=0.5T
SIZE_N = {  # 모델 크기 군집: 점 하나의 FLOPs를 6*D(0.1~0.5T)로 나눈 N으로 군집을 정한다
    "0.16B": 16 * (4 * 768**2 + 3 * 768 * 2048) + 2 * 32000 * 768,
    "0.37B": 24 * (4 * 1024**2 + 3 * 1024 * 2816) + 2 * 32000 * 1024,
    "0.76B": 24 * (4 * 1536**2 + 3 * 1536 * 4096) + 2 * 32000 * 1536,
    "1.4B": 24 * (4 * 2048**2 + 3 * 2048 * 5632) + 2 * 32000 * 2048,
    "7B": N7,
}


def fit(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # a, b  (모두 자연로그)


def x_at(coef, ylog):
    return (ylog - coef[0]) / coef[1]


def cluster_of(logflops):
    f = 10 ** logflops
    best, bd = None, 9
    for k, n in SIZE_N.items():
        d = f / (6 * n) / 1e11  # 0.1T 단위의 토큰 수
        dev = abs(d - np.round(d))
        if 0.9 <= np.round(d) <= 5.1 and dev < bd:
            best, bd = k, dev
    return best


def ratio(cT, cC, y_target_log):
    return float(np.exp(x_at(cT, y_target_log) - np.log(X7)) if False else np.exp(x_at(cT, y_target_log)) / X7)


res = {"X7_flops_6ND": X7, "N7": N7, "panels": {}}
for name, (tv, cv, paper_ratio) in T3.items():
    P = V[name]["points"]
    xT = np.array([p[0] for p in P["Transfusion"]]) * LN10
    yT = np.log([p[1] for p in P["Transfusion"]])
    xC = np.array([p[0] for p in P["Chameleon"]]) * LN10
    yC = np.log([p[1] for p in P["Chameleon"]])
    cT, cC = fit(xT, yT), fit(xC, yC)
    xs = np.log(X7)
    yA = np.log(cv)                      # 변형 A: 실측 Chameleon 7B
    yB = cC[0] + cC[1] * xs              # 변형 B: Chameleon 맞춤선의 7B 값
    rA = float(np.exp(x_at(cT, yA)) / X7)
    rB = float(np.exp(x_at(cT, yB)) / X7)
    # 논문이 그린 추세선 끝점과 비교(기울기)
    ends = V[name]["paper_trendline_ends"]
    slope_paper = {}
    for k in ("Transfusion", "Chameleon"):
        (xa, ya), (xb, yb) = ends[k]
        slope_paper[k] = (np.log(yb) - np.log(ya)) / ((xb - xa) * LN10)
    # 군집 부트스트랩(25점 패널: 크기별 5군집). CIDEr 패널은 점이 60개라 크기 군집을 확정하기 어려워 점 부트스트랩만 쓴다.
    def boot(kind, B=4000):
        out = []; outA = []
        cl_T = [cluster_of(p[0]) for p in P["Transfusion"]]
        cl_C = [cluster_of(p[0]) for p in P["Chameleon"]]
        for _ in range(B):
            if kind == "point":
                iT = rng.integers(0, len(xT), len(xT)); iC = rng.integers(0, len(xC), len(xC))
            elif kind == "cluster":
                keysT = sorted(set(cl_T)); keysC = sorted(set(cl_C))
                # 두 방법이 같은 크기 줄기를 공유한다고 보고 크기를 같은 확률로 재표집(방법별 독립 재표집)
                pickT = rng.choice(keysT, len(keysT)); pickC = rng.choice(keysC, len(keysC))
                iT = np.concatenate([[i for i, c in enumerate(cl_T) if c == k] for k in pickT]).astype(int)
                iC = np.concatenate([[i for i, c in enumerate(cl_C) if c == k] for k in pickC]).astype(int)
                if len(set(xT[iT])) < 3 or len(set(xC[iC])) < 3:
                    continue
            elif kind == "resid":
                rT = yT - (cT[0] + cT[1] * xT); rC = yC - (cC[0] + cC[1] * xC)
                yTb = cT[0] + cT[1] * xT + rng.choice(rT, len(rT)); yCb = cC[0] + cC[1] * xC + rng.choice(rC, len(rC))
                bT, bC = fit(xT, yTb), fit(xC, yCb)
                yBb = bC[0] + bC[1] * xs
                outA.append(float(np.exp(x_at(bT, yA)) / X7)); out.append(float(np.exp(x_at(bT, yBb)) / X7)); continue
            bT, bC = fit(xT[iT], yT[iT]), fit(xC[iC], yC[iC])
            yBb = bC[0] + bC[1] * xs
            outA.append(float(np.exp(x_at(bT, yA)) / X7)); out.append(float(np.exp(x_at(bT, yBb)) / X7))
        return {"A": np.percentile(outA, [2.5, 50, 97.5]).round(4).tolist(), "B": np.percentile(out, [2.5, 50, 97.5]).round(4).tolist()}
    ci = {"point": boot("point"), "resid": boot("resid")}
    if len(xT) == 25 and len(xC) in (25, 20, 16):
        ci["cluster"] = boot("cluster")
    # 외삽 점검: 목표 FLOPs가 Transfusion 점 범위 안인가
    xT_need = float(np.exp(x_at(cT, yB)))
    inside = bool(10 ** V[name]["points"]["Transfusion"][0][0] if False else (np.exp(xT.min()) <= xT_need <= np.exp(xT.max())))
    resT = yT - (cT[0] + cT[1] * xT); resC = yC - (cC[0] + cC[1] * xC)
    res["panels"][name] = {
        "n_points": {"Transfusion": len(xT), "Chameleon": len(xC)},
        "fit_slope_lnln": {"Transfusion": round(float(cT[1]), 4), "Chameleon": round(float(cC[1]), 4)},
        "paper_drawn_slope": {k: round(float(v), 4) for k, v in slope_paper.items()},
        "resid_sd_ln": {"Transfusion": round(float(resT.std(ddof=2)), 4), "Chameleon": round(float(resC.std(ddof=2)), 4)},
        "R2": {"Transfusion": round(1 - float(resT.var() / yT.var()), 3), "Chameleon": round(1 - float(resC.var() / yC.var()), 3)},
        "table3_values": {"Transfusion": tv, "Chameleon": cv}, "paper_parity_ratio": paper_ratio,
        "ratio_A_actual_C7B": round(rA, 4), "ratio_A_nominal_X_2.1e22": round(rA * X7 / 2.1e22, 4), "ratio_B_fitted_C7B": round(rB, 4),
        "ratio_95ci": ci,
        "T_flops_needed_inside_data_range": inside,
        "T_flops_needed": xT_need,
        "T_fit_at_7B_flops": float(np.exp(cT[0] + cT[1] * xs)), "C_fit_at_7B_flops": float(np.exp(yB)),
        "coef_lnln": {"Transfusion": [float(cT[0]), float(cT[1])], "Chameleon": [float(cC[0]), float(cC[1])]},
        "lines_cross_at_FLOPs": (float(np.exp((cC[0] - cT[0]) / (cT[1] - cC[1]))) if abs(cT[1] - cC[1]) > 1e-9 else None),
    }
    # 평행하지 않으면 비율이 목표에 따라 달라진다: 목표를 Chameleon 점 범위의 양 끝으로 바꿔 본다
    lo, hi = np.exp(xC.min()), np.exp(xC.max())
    res["panels"][name]["parity_ratio_by_C_flops"] = {
        f"{f:.2e}": round(float(np.exp(x_at(cT, cC[0] + cC[1] * np.log(f))) / f), 4) for f in (lo, np.sqrt(lo * hi), hi)
    }
    print(name, "A", round(rA, 3), "paper", paper_ratio, "CI_A cluster/point/resid", ci.get("cluster", {}).get("A"), ci["point"]["A"], ci["resid"]["A"], "| B", round(rB, 3), ci["point"]["B"], "slopes", round(float(cT[1]),3), round(float(cC[1]),3), {k: round(float(v),3) for k,v in slope_paper.items()})

# CIDEr: Chameleon 이상점 두 개(≈4.7, 1.6~1.9e21)를 뺀 민감도
P = V["coco_cider"]["points"]
keep = [p for p in P["Chameleon"] if p[1] > 5.5]
xC = np.array([p[0] for p in keep]) * LN10; yC = np.log([p[1] for p in keep])
cC = fit(xC, yC)
xT = np.array([p[0] for p in P["Transfusion"]]) * LN10; yT = np.log([p[1] for p in P["Transfusion"]])
cT = fit(xT, yT)
yB = cC[0] + cC[1] * np.log(X7)
res["cider_without_two_outliers"] = {"n_C": len(keep), "ratio_B": round(float(np.exp(x_at(cT, yB)) / X7), 4),
                                      "slope_C": round(float(cC[1]), 4)}
(HERE / "scaling_results.json").write_text(json.dumps(res, indent=1))
print("N7", N7, "X7", f"{X7:.3e}", res["cider_without_two_outliers"])
