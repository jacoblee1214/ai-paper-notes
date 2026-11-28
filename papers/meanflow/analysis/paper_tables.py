"""표 1·2·4·5 재계산: 상대 개선율, 파라미터·FLOPs 산술, 절제 표의 차이를 시드 잡음 가정과 비교, 확장 곡선.
논문 값은 paper.txt(arXiv v1)에서 옮겨 적은 것이다. 학습·추론 없음."""
import json
import math
from pathlib import Path
import numpy as np
from scipy import stats, linalg

OUT = Path(__file__).with_name("tables_results.json")
res = {}

# ---- 1. 표 2의 상대 개선율 -----------------------------------------------
rel = lambda base, new: (base - new) / base
res["relative_improvement"] = {
    "1NFE 3.43 vs Shortcut-XL/2 10.60 (본문: 약 70%)": rel(10.60, 3.43),
    "1NFE 3.43 vs iCT-XL/2 34.24": rel(34.24, 3.43),
    "1NFE 3.43 vs IMM 7.77 (IMM은 1x2=2NFE; 본문: 50% 넘게)": rel(7.77, 3.43),
    "2NFE 2.93 vs IMM 7.77 (같은 2NFE)": rel(7.77, 2.93),
    "2NFE 2.93 vs iCT 20.30": rel(20.30, 2.93),
    "2NFE 2.20(XL/2+, 1000ep) vs 2.93(XL/2, 240ep)": rel(2.93, 2.20),
}
res["text_vs_table_inconsistency"] = {
    "본문 §5.2 SiT-XL/2 FID": 2.15, "표 2 SiT-XL/2 FID": 2.06, "DiT-XL/2 본문=표": 2.27,
    "MF-XL/2+ 2.20 대 본문 SiT 2.15": 2.20 - 2.15, "MF-XL/2+ 2.20 대 표 SiT 2.06": 2.20 - 2.06,
}

# ---- 2. 에폭 <-> 반복 수, JVP 오버헤드, 추론 비용 -------------------------------
IN = 1_281_167
bs = 256
res["epochs_iters"] = {
    "80 epoch -> iters (논문: 400K)": 80 * IN / bs,
    "240 epoch -> iters": 240 * IN / bs, "1000 epoch -> iters": 1000 * IN / bs,
    "DiT/SiT 7M iters -> epochs": 7_000_000 * bs / IN,
}
res["jvp_overhead"] = {"0.052/0.045 - 1": 0.052 / 0.045 - 1, "논문 본문: <20%, 부록: 16%": None}
xl_fwd = 118.6  # DiT-XL/2 순전파 (GMACs), 아래 산술로 재확인
res["sampling_cost_ratio"] = {"SiT-XL/2 250x2 NFE 대 MF 1NFE (NFE 비)": 500.0, "MF 2NFE 대 SiT": 250.0}

# ---- 3. 파라미터·FLOPs 산술 (DiT 구조, 코드의 모듈 구성 기준) -----------------------
def dit_counts(depth, d, p, in_ch=4, latent=32, ncls=1000, freq=256):
    T = (latent // p) ** 2
    blk = (3 * d * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d) + (d * 6 * d + 6 * d)
    emb_t = 2 * ((freq * d + d) + (d * d + d))             # t, h 두 개
    emb_y = (ncls + 1) * d
    patch = p * p * in_ch * d + d
    final = (d * 2 * d + 2 * d) + (d * p * p * in_ch + p * p * in_ch)
    params = depth * blk + emb_t + emb_y + patch + final
    macs_tok = depth * (12 * d * d * T + 2 * T * T * d)     # qkv(3)+proj(1)+mlp(8) d^2 T, 어텐션 QK^T와 AV
    macs = macs_tok + T * p * p * in_ch * d + T * d * p * p * in_ch
    return params / 1e6, macs / 1e9


cfgs = {"B/4": (12, 768, 4), "B/2": (12, 768, 2), "M/2": (16, 1024, 2), "L/2": (24, 1024, 2), "XL/2": (28, 1152, 2)}
paper4 = {"B/4": (131, 5.6), "B/2": (131, 23.1), "M/2": (497.8, 54.0), "L/2": (459, 119.0), "XL/2": (676, 119.0)}
paper2 = {"B/2": 131, "M/2": 308, "L/2": 459, "XL/2": 676}
tab = {}
for k, (dep, d, p) in cfgs.items():
    P, G = dit_counts(dep, d, p)
    tab[k] = {"params_M_ours": P, "GMACs_ours": G, "table4_params": paper4[k][0], "table4_GFLOPs": paper4[k][1],
              "table2_params": paper2.get(k), "params_rel_diff_vs_table4": P / paper4[k][0] - 1,
              "flops_rel_diff_vs_table4": G / paper4[k][1] - 1}
res["params_flops"] = tab

# ---- 4. 절제 표 vs 시드 잡음 가정 -------------------------------------------------
tables = {
    "1a r!=t 비율": ({"0%": 328.91, "25%": 61.06, "50%": 63.14, "100%": 67.32}, "25%"),
    "1b JVP 접선": ({"(v,0,1)": 61.06, "(v,0,0)": 268.06, "(v,1,0)": 329.22, "(v,1,1)": 137.96}, "(v,0,1)"),
    "1c 시간 조건": ({"(t,r)": 61.75, "(t,t-r)": 61.06, "(t,r,t-r)": 63.98, "t-r only": 63.13}, "(t,t-r)"),
    "1d 시간 표집기": ({"uniform": 65.90, "ln(-0.2,1.0)": 63.83, "ln(-0.2,1.2)": 64.72, "ln(-0.4,1.0)": 61.06, "ln(-0.4,1.2)": 61.79}, "ln(-0.4,1.0)"),
    "1e 손실 p": ({"0.0": 79.75, "0.5": 63.98, "1.0": 61.06, "1.5": 66.57, "2.0": 69.19}, "1.0"),
    "1f CFG ω": ({"1.0": 61.06, "1.5": 33.33, "2.0": 20.15, "3.0": 15.53, "5.0": 20.75}, "3.0"),
    "5 κ (ω'=2)": ({"0.0": 20.15, "0.5": 19.15, "0.8": 19.10, "0.9": 18.63, "0.95": 19.17}, "0.9"),
}
res["ablation_vs_noise"] = {}
for cov in [0.01, 0.02]:
    out = {}
    for name, (d, ref) in tables.items():
        # 각 항목을 표의 '최저 FID' 항목과 비교 (선택 편향 때문에 최저 대비 비교가 가장 보수적이지 않으므로 Holm 보정)
        best = min(d, key=d.get)
        rows, ps = [], []
        for kname, val in d.items():
            if kname == best:
                continue
            sd = cov * math.sqrt(val ** 2 + d[best] ** 2)         # 두 실행 차의 표준편차 (CoV는 FID에 비례)
            z = (val - d[best]) / sd
            ps.append(2 * stats.norm.sf(abs(z)))
            rows.append({"item": kname, "fid": val, "diff_vs_best": val - d[best], "rel_diff": (val - d[best]) / d[best], "z": z,
                         "breakeven_cov_p05": abs(val - d[best]) / (1.959964 * math.sqrt(val ** 2 + d[best] ** 2))})
        order = np.argsort(ps)
        m = len(ps)
        holm = [None] * m
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * ps[i]))
            holm[i] = run
        for i, row in enumerate(rows):
            row["p_raw"] = ps[i]
            row["p_holm"] = holm[i]
            row["distinguishable"] = holm[i] < 0.05
        out[name] = {"best": best, "rows": rows}
    res["ablation_vs_noise"][f"CoV={cov}"] = out

# ---- 5. 확장: 크기별 1-NFE FID (표 2, 그림 4), 로그-로그 기울기 (LOO 포함) -------------
P = np.array([131, 308, 459, 676.0])
F = np.array([6.17, 5.01, 3.84, 3.43])
x, y = np.log(P), np.log(F)
slope = np.polyfit(x, y, 1)[0]
loo = [np.polyfit(np.delete(x, i), np.delete(y, i), 1)[0] for i in range(4)]
res["scaling_loglog"] = {"slope": slope, "leave_one_out": loo,
                         "FID_ratio_XL_over_B2": F[-1] / F[0], "note": "점 4개, 각 한 번의 실행. 크기는 에폭 240 고정이고 XL/2는 CFG 구간 [0,0.75]로 다른 설정"}

# ---- 6. 장난감: 50k 표본 FID 추정기의 표본 잡음 (평가 잡음만, 학습 시드 잡음 아님) -----
rng = np.random.default_rng(0)
d, n_s, reps = 256, 50000, 12
lam = 1.0 / np.arange(1, d + 1)
lam = lam / lam.sum() * d * 0.5
sd = np.sqrt(lam)


def fid_est(x, mu, cov):
    m = x.mean(0)
    c = np.cov(x, rowvar=False)
    covmean = linalg.sqrtm(cov @ c, disp=False)[0].real
    return float(((m - mu) ** 2).sum() + np.trace(cov + c - 2 * covmean))


mu0 = np.zeros(d)
cov0 = np.diag(lam)
toy = {}
for target in [3.4, 20.0, 61.0]:
    shift = np.zeros(d); shift[0] = math.sqrt(target)
    vals = []
    for _ in range(reps):
        x = shift + rng.standard_normal((n_s, d)) * sd
        vals.append(fid_est(x, mu0, cov0))
    vals = np.array(vals)
    toy[str(target)] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "cov": float(vals.std(ddof=1) / vals.mean()), "n_samples": n_s, "reps": reps, "d": d, "seed": 0}
res["toy_fid_estimator_noise_gaussian"] = toy

OUT.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res[k] for k in ["relative_improvement", "epochs_iters", "params_flops", "scaling_loglog", "toy_fid_estimator_noise_gaussian"]}, indent=1))
for cov, o in res["ablation_vs_noise"].items():
    for name, v in o.items():
        print(cov, name, "best", v["best"], [(r["item"], round(r["rel_diff"] * 100, 1), "%.3f" % r["p_holm"], "%.3f" % r["breakeven_cov_p05"]) for r in v["rows"]])
