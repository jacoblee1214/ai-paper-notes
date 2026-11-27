"""FlashAttention 분석용 실험. 결과는 results.json. 실행: python3 run_experiments.py  (CPU, 약 1-2분)

E1  Theorem 2 / Theorem 5: (N, d, M) 격자에서 Alg. 1, Alg. 4의 HBM 접근 수를 시뮬레이터로 세어 지수와 상수를 적합
E2  표준 attention 대비 교차점: FlashAttention이 HBM 접근을 줄이기 시작하는 M의 문턱 (닫힌 식 ~ d(3d+4))
E3  Proposition 4: 블록 희소 마스크의 비율 s에 대한 HBM 접근 수 (무작위 마스크와 국소 창)
E4  Alg. 1 블록 크기가 쓰는 SRAM / M 비
E5  논문 표 재계산: Table 1 Welch 검정, Table 2/4 속도 향상, Table 3 평균, Fig. 2(왼쪽) 비율, 산술 강도
E6  정확성 허용 오차: float64 / float32 / float16 에서 flash vs naive 오차 (시드 20개, 부호 검정은 Holm 보정)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import minimal_impl as mi

OUT = Path(__file__).parent / "results.json"
res = {}


# ---------------------------------------------------------------- E1
def e1():
    rows = []
    for N in (256, 512, 1024, 2048):
        for d in (16, 32, 64, 128):
            for M in (2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17):
                if not (d <= M <= N * d):
                    continue
                bc, br = mi.block_sizes(M, d)
                if math.ceil(N / br) * math.ceil(N / bc) > 70000:
                    continue
                cf, cb = mi.count_flash(N, d, M)
                sf, sb = mi.count_standard(N, d)
                rows.append(dict(N=N, d=d, M=M, Bc=bc, Br=br, Tc=math.ceil(N / bc), flash_fwd=cf.total, flash_bwd=cb.total,
                                 std_fwd=sf.total, std_bwd=sb.total, peak_sram=cf.peak_sram))
    out = {"n_configs": len(rows)}
    X = np.array([[1, math.log(r["N"]), math.log(r["d"]), math.log(r["M"])] for r in rows])
    for key in ("flash_fwd", "flash_bwd"):
        y = np.log([r[key] for r in rows])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = resid @ resid / (len(y) - 4)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        out[f"loglog_{key}"] = dict(exp_N=beta[1], exp_d=beta[2], exp_M=beta[3], se=list(se[1:]), r2=1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())))
        # 두 항 모형 A = a * N^2 d^2 / M + b * N d
        Z = np.array([[r["N"] ** 2 * r["d"] ** 2 / r["M"], r["N"] * r["d"]] for r in rows])
        A = np.array([r[key] for r in rows], float)
        w = 1 / A  # 상대 오차 가중
        coef, *_ = np.linalg.lstsq(Z * w[:, None], A * w, rcond=None)
        pred = Z @ coef
        out[f"two_term_{key}"] = dict(a=coef[0], b=coef[1], max_rel_err=float(np.max(np.abs(pred - A) / A)))
    # 닫힌 식과의 일치
    out["closed_form_exact"] = all(r["flash_fwd"] == mi.hbm_flash_forward_closed_form(r["N"], r["d"], r["M"]) and
                                   r["flash_bwd"] == mi.hbm_flash_backward_closed_form(r["N"], r["d"], r["M"]) for r in rows)
    res["E1_rows"] = rows
    res["E1"] = out


# ---------------------------------------------------------------- E2
def e2():
    out = {}
    for N in (1024, 4096):
        for d in (32, 64, 128):
            std = mi.hbm_standard_forward(N, d)
            M_grid = [int(d * 1.05 ** k) for k in range(0, 400)]
            M_star = next((M for M in M_grid if mi.hbm_flash_forward_closed_form(N, d, M) < std), None)
            out[f"N{N}_d{d}"] = dict(M_star_fwd=M_star, formula=d * (3 * d + 4), ratio_M_star_over_d2=None if M_star is None else M_star / d**2,
                                     bytes_fp16_KB=None if M_star is None else 2 * M_star / 1024)
    # 전형적 SRAM에서의 순전파 비율 (A100 SM당 192 KB = fp16 98,304 원소; 164 KB 사용 가능이라는 보고도 있어 몇 값을 훑음)
    ratios = {}
    for d in (64, 128):
        for M in (16384, 32768, 65536, 98304):
            for N in (512, 1024, 4096):
                cf, cb = mi.count_flash(N, d, M)
                sf, sb = mi.count_standard(N, d)
                ratios[f"d{d}_M{M}_N{N}"] = dict(fwd=sf.total / cf.total, bwd=sb.total / cb.total, both=(sf.total + sb.total) / (cf.total + cb.total))
    out["ratios_std_over_flash"] = ratios
    res["E2"] = out


# ---------------------------------------------------------------- E3
def e3():
    N, d, M = 1024, 32, 4096
    bc, br = mi.block_sizes(M, d)
    Tr, Tc = math.ceil(N / br), math.ceil(N / bc)
    rows = []
    dense = mi.count_flash(N, d, M, backward=False)[0].total
    for pattern in ("random", "local"):
        for s_target in (0.0625, 0.125, 0.25, 0.5, 1.0):
            g = torch.Generator().manual_seed(3)
            if pattern == "random":
                mask = torch.rand(Tr, Tc, generator=g) < s_target
                mask[torch.arange(Tr), torch.arange(Tr) * Tc // Tr] = True
            else:
                w = max(1, round(s_target * Tc))
                idx = torch.arange(Tr)[:, None] * Tc // Tr
                mask = (torch.arange(Tc)[None, :] - idx).abs() <= w // 2
            s = mask.float().mean().item()
            cf, cb = mi.count_flash(N, d, M, block_mask=mask)
            rows.append(dict(pattern=pattern, s=s, fwd=cf.total, bwd=cb.total, fwd_over_dense=cf.total / dense))
    s = np.array([r["s"] for r in rows]); y = np.array([r["fwd"] for r in rows], float)
    A = np.stack([np.ones_like(s), s], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    res["E3"] = dict(N=N, d=d, M=M, Tr=Tr, Tc=Tc, rows=rows, intercept=coef[0], slope=coef[1], intercept_over_2Nd=coef[0] / (2 * N * d),
                     max_rel_err=float(np.max(np.abs(pred - y) / y)), dense_fwd=dense)


# ---------------------------------------------------------------- E4
def e4():
    ratios = []
    for d in (16, 32, 64, 128):
        for M in (4 * d, 8 * d, 2 * d * d, 4 * d * d, 8 * d * d, 8 * d * d + 8 * d, 2**14, 2**16, 98304):
            bc, br = mi.block_sizes(M, d)
            ratios.append(dict(d=d, M=M, Bc=bc, Br=br, footprint=mi.sram_footprint(br, bc, d), ratio=mi.sram_footprint(br, bc, d) / M))
    res["E4"] = dict(rows=ratios, max_ratio=max(r["ratio"] for r in ratios),
                     ratio_at_98304=[r["ratio"] for r in ratios if r["M"] == 98304])


# ---------------------------------------------------------------- E5
def e5():
    out = {}
    # Table 1: 20.0 +- 1.5 vs 17.4 +- 1.4 (10회). 논문은 +-가 표준편차인지 표준오차인지 적지 않음
    for label, sd_fac in (("pm_is_sd", 1.0), ("pm_is_se", math.sqrt(10))):
        a_sd, b_sd = 1.5 * sd_fac, 1.4 * sd_fac
        t = stats.ttest_ind_from_stats(20.0, a_sd, 10, 17.4, b_sd, 10, equal_var=False)
        se = math.sqrt(a_sd**2 / 10 + b_sd**2 / 10)
        dfree = se**4 / ((a_sd**2 / 10) ** 2 / 9 + (b_sd**2 / 10) ** 2 / 9)
        crit = stats.t.ppf(0.975, dfree)
        out[f"table1_{label}"] = dict(t=float(t.statistic), p=float(t.pvalue), diff_min=2.6, ci95=[2.6 - crit * se, 2.6 + crit * se],
                                       speedup=20.0 / 17.4)
    # Table 2, 4
    t2 = {"small_hf": 9.5, "small_megatron": 4.7, "small_flash": 2.7, "med_hf": 21.0, "med_megatron": 11.5, "med_flash": 6.9}
    out["table2"] = dict(small_flash_vs_hf=t2["small_hf"] / t2["small_flash"], small_flash_vs_megatron=t2["small_megatron"] / t2["small_flash"],
                         small_megatron_vs_hf=t2["small_hf"] / t2["small_megatron"],
                         med_flash_vs_hf=t2["med_hf"] / t2["med_flash"], med_flash_vs_megatron=t2["med_megatron"] / t2["med_flash"],
                         med_megatron_vs_hf=t2["med_hf"] / t2["med_megatron"])
    out["table4"] = {k: 4.7 / v for k, v in {"flash_1k": 2.7, "flash_2k": 3.0, "flash_4k": 3.6}.items()}
    out["table4_time_reduction_4k_vs_megatron1k"] = 1 - 3.6 / 4.7
    # Table 3 평균
    t3 = {"Transformer": ([36.0, 63.6, 81.6, 42.3, 72.7], 59.3), "FlashAttention": ([37.6, 63.9, 81.4, 43.5, 72.7], 59.8),
          "Block-sparse": ([37.0, 63.0, 81.3, 43.6, 73.3], 59.6), "Linformer": ([35.6, 55.9, 77.7, 37.8, 67.6], 54.9),
          "Linear": ([38.8, 63.2, 80.7, 42.6, 72.5], 59.6), "Performer": ([36.8, 63.6, 82.2, 42.1, 69.9], 58.9),
          "Local": ([36.1, 60.2, 76.7, 40.6, 66.6], 56.0), "Reformer": ([36.5, 63.8, 78.5, 39.6, 69.4], 57.6),
          "Smyrf": ([36.1, 64.1, 79.0, 39.6, 70.5], 57.9)}
    out["table3_avg"] = {k: dict(recomputed=float(np.mean(v)), reported=r, diff=float(np.mean(v) - r)) for k, (v, r) in t3.items()}
    tf, ff = np.array(t3["Transformer"][0]), np.array(t3["FlashAttention"][0])
    out["table3_exact_pair_gap"] = dict(per_task=list(np.round(ff - tf, 2)), mean_abs=float(np.mean(np.abs(ff - tf))), max_abs=float(np.max(np.abs(ff - tf))))
    # Fig. 2 왼쪽: GFLOPs 66.6 vs 75.2, HBM 40.3 vs 4.4 GB, runtime 41.7 vs 7.3 ms
    N, d, B, H = 1024, 64, 64, 16
    bh = B * H
    fl_std = bh * 12 * N * N * d / 1e9
    fl_fl = bh * 14 * N * N * d / 1e9
    out["fig2_left"] = dict(reported=dict(gflops=[66.6, 75.2], hbm_gb=[40.3, 4.4], ms=[41.7, 7.3]),
                            ratios=dict(flops=75.2 / 66.6, hbm=40.3 / 4.4, runtime=41.7 / 7.3),
                            matmul_flops_ratio_14_over_12=14 / 12,
                            our_matmul_gflops_fwd_bwd=dict(standard=fl_std, flash=fl_fl),
                            our_over_reported_standard=fl_std / 66.6)
    hbm = {}
    for M in (32768, 65536, 98304):
        cf, cb = mi.count_flash(N, d, M)
        sf, sb = mi.count_standard(N, d)
        gb = lambda n: n * bh * 2 / 1e9  # fp16
        hbm[str(M)] = dict(flash_gb=gb(cf.total + cb.total), standard_gb=gb(sf.total + sb.total), ratio=(sf.total + sb.total) / (cf.total + cb.total))
    out["fig2_left_sim_hbm_fp16"] = hbm
    # 산술 강도(표준 attention 순전파): FLOPs / byte(fp16). 기기 사양은 논문 밖 값(NVIDIA A100 사양: fp16 텐서코어 312 TFLOP/s, HBM 약 1.5-2.0 TB/s는 논문 2.1절)
    flops = 4 * N * N * d
    byt = 2 * mi.hbm_standard_forward(N, d)
    out["arith_intensity"] = dict(standard_fwd_flop_per_byte=flops / byt, a100_ridge_flop_per_byte=dict(hbm_1_5TBs=312e12 / 1.5e12, hbm_2_0TBs=312e12 / 2.0e12))
    res["E5"] = out


# ---------------------------------------------------------------- E6
def e6():
    rows = []
    for scale in (1.0, 4.0, 8.0):
        for seed in range(20):
            g = torch.Generator().manual_seed(100 + seed)
            N, d, M = 256, 64, 16384
            Q, K, V = (torch.randn(N, d, generator=g, dtype=torch.float64) for _ in range(3))
            Q = Q * scale
            row = dict(scale=scale, seed=seed)
            S64 = (Q @ K.T) / math.sqrt(d)
            row["max_logit"] = float(S64.abs().max())
            ref = mi.naive_reference(Q, K, V)
            O64 = mi.flash_forward(Q, K, V, M)[0]
            row["f64_flash"] = float((O64 - ref).abs().max())
            q32, k32, v32 = Q.float(), K.float(), V.float()
            ref32 = mi.naive_reference(q32, k32, v32)  # 같은 float32 입력의 float64 정답
            row["f32_flash"] = float((mi.flash_forward(q32, k32, v32, M)[0].double() - ref32).abs().max())
            S = (q32 @ k32.T) / math.sqrt(d)
            row["f32_naive"] = float(((torch.softmax(S, -1) @ v32).double() - ref32).abs().max())
            q16, k16, v16 = Q.half(), K.half(), V.half()
            ref16 = mi.naive_reference(q16, k16, v16)
            row["f16_flash"] = float((mi.flash_forward(q16, k16, v16, M)[0].double() - ref16).abs().max())
            row["f16_flash_roundP"] = float((mi.flash_forward(q16, k16, v16, M, p_round_dtype=torch.float16)[0].double() - ref16).abs().max())
            S16 = (q16 @ k16.T) / math.sqrt(d)  # PyTorch eager 관례: 행렬곱 결과를 fp16으로 반올림해 저장
            P16 = torch.softmax(S16, -1)
            row["f16_naive"] = float(((P16 @ v16).double() - ref16).abs().max())
            rows.append(row)
    summ = {}
    for scale in (1.0, 4.0, 8.0):
        sub = [r for r in rows if r["scale"] == scale]
        summ[str(scale)] = {k: float(np.mean([r[k] for r in sub])) for k in ("max_logit", "f64_flash", "f32_flash", "f32_naive", "f16_flash", "f16_flash_roundP", "f16_naive")}
        k = sum(r["f16_flash"] < r["f16_naive"] for r in sub)
        summ[str(scale)]["flash_better_fp16_seeds"] = f"{k}/{len(sub)}"
        summ[str(scale)]["sign_test_p"] = float(stats.binomtest(k, len(sub), 0.5).pvalue)
        ratio = np.array([r["f16_naive"] / r["f16_flash"] for r in sub])
        rng = np.random.default_rng(0)
        boots = [np.exp(np.mean(np.log(rng.choice(ratio, len(ratio))))) for _ in range(2000)]
        summ[str(scale)]["naive_over_flash_err_geomean"] = float(np.exp(np.mean(np.log(ratio))))
        summ[str(scale)]["naive_over_flash_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    order = sorted(summ, key=lambda k: summ[k]["sign_test_p"])
    for rank, k in enumerate(order):  # Holm
        summ[k]["sign_test_p_holm"] = min(1.0, max(summ[o]["sign_test_p"] * (len(order) - r_) for r_, o in enumerate(order) if r_ <= rank))
    res["E6"] = dict(rows=rows, summary=summ)


if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(0)
    for f in (e1, e2, e3, e4, e5, e6):
        f()
        print(f.__name__, f"{time.time() - t0:.1f}s")
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "E1_rows"}, indent=1, default=float)[:6000])
