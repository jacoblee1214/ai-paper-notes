"""논문(arXiv:2006.11239v2)의 표와 본문 산술을 다시 계산한다. 학습·이미지·Inception 없음(FID, IS는 재계산 불가).

1. 표 1: 학습·시험 NLL 간격 ("at most 0.03 bits/dim")
2. §4.3, 표 4: 율-왜곡 산술, 이산 디코더 산술로 왜곡 1.97 bits/dim 이 어떤 sigma_1과 맞는지, 0율 왜곡 상한
3. 표 2: IS의 +-를 분할 표준편차로 가정한 짝 차이 (학습 시드 분산은 포함하지 않음)
4. 표 3, 표 1의 FID 비교 산술
5. 부록 B: 학습·표본 시간 산술, U-Net 모수 수 (공식 코드의 구조를 읽어 닫힌 식으로 센다)
결과는 paper_arithmetic_results.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
LN2 = math.log(2.0)
R = {}


def sched():
    b = np.linspace(1e-4, 0.02, 1000, dtype=np.float64)
    ab = np.cumprod(1 - b)
    return b, ab


def table1():
    rows = {"L (fixed isotropic Sigma)": (3.70, 3.69), "L_simple": (3.75, 3.72)}
    R["T1_nll_gap"] = {k: {"test": v[0], "train": v[1], "gap": round(v[0] - v[1], 4), "within_0.03": (v[0] - v[1]) <= 0.03 + 1e-9}
                       for k, v in rows.items()}


def disc_nll_bits(sig, err_rms, shape="gauss"):
    """이산 가우시안 디코더(빈 너비 1, [0,255] 스케일)의 기대 NLL (bits/dim). 오차 e = x0 - mu의 분포를 가정한다.
    P(x0 | mu) = Phi((e + .5)/sig) - Phi((e - .5)/sig). 가장자리 효과는 무시(논문 §3.4와 같음)."""
    e = np.linspace(-12 * err_rms, 12 * err_rms, 400001)
    if shape == "gauss":
        pdf = stats.norm.pdf(e, scale=err_rms)
    else:  # 같은 RMS의 라플라스
        pdf = stats.laplace.pdf(e, scale=err_rms / math.sqrt(2))
    P = stats.norm.cdf((e + 0.5) / sig) - stats.norm.cdf((e - 0.5) / sig)
    nll = -np.log2(np.maximum(P, 1e-300))
    return float(np.trapz(pdf * nll, e))


def section43():
    b, ab = sched()
    rate, dist_bits, rmse = 1.78, 1.97, 0.95
    t4 = {1000: (1.77581, 0.95136), 900: (0.11994, 12.02277), 800: (0.05415, 18.47482), 700: (0.02866, 24.43656),
          600: (0.01507, 30.80948), 500: (0.00716, 38.03236), 400: (0.00282, 46.12765), 300: (0.00081, 54.18826),
          200: (0.00013, 60.97170), 100: (0.0, 67.60125)}
    keys = sorted(t4, reverse=True)
    rates = [t4[k][0] for k in keys]
    dists = [t4[k][1] for k in keys]
    # 표 4의 행은 T-t+1 = 1000..100 이다. t = 1001 - key.
    abar_at = {k: float(ab[1001 - k - 1]) for k in keys}
    # 디코더 분산 후보 ([-1,1] -> [0,255] 스케일 곱 127.5)
    beta1 = b[0]
    bt2 = b[1] * (1 - ab[0]) / (1 - ab[1])
    cand = {"sigma1^2=beta_1 (논문 §3.3의 sigma_t^2=beta_t)": math.sqrt(beta1) * 127.5,
            "sigma1^2=beta~_2 (공식 코드 fixedlarge의 t=1 로그 분산)": math.sqrt(bt2) * 127.5}
    nll = {}
    for name, sig in cand.items():
        nll[name] = {"sigma_255scale": sig, "gauss_err": disc_nll_bits(sig, 0.95136, "gauss"), "laplace_err": disc_nll_bits(sig, 0.95136, "laplace")}
    # 1.97 bits를 주는 sigma (가우시안 오차 가정). NLL(sigma)은 U자형이라 최솟값 양쪽의 해를 모두 구한다.
    grid = np.linspace(0.3, 3.0, 541)
    vals = np.array([disc_nll_bits(g, 0.95136) for g in grid])
    imin = int(vals.argmin())
    from scipy.optimize import brentq
    f = lambda x: disc_nll_bits(x, 0.95136) - 1.97
    roots = []
    if vals[0] > 1.97 >= vals[imin]:
        roots.append(brentq(f, grid[0], grid[imin]))
    if vals[-1] > 1.97 >= vals[imin]:
        roots.append(brentq(f, grid[imin], grid[-1]))
    sig_fit = roots
    # 0율 왜곡의 상한: 데이터 평균만 예측하는 예측기의 RMSE ([0,255] 스케일, CIFAR10 널리 알려진 채널 표준편차, 재계산 안 함)
    std255 = np.array([0.2470, 0.2435, 0.2616]) * 255
    const_rmse = float(np.sqrt((std255 ** 2).mean()))
    k = 100
    amp = 1 / math.sqrt(abar_at[k])
    mse_x_excess = (67.60125 ** 2 - const_rmse ** 2) / 127.5 ** 2  # [-1,1] 스케일^2
    R["S43_T4"] = {
        "rate_plus_distortion": rate + dist_bits, "table1_L_simple_nll": 3.75,
        "frac_of_codelength_in_distortion": dist_bits / (rate + dist_bits),
        "table4_rate_1000_vs_text_1.78": t4[1000][0], "table4_dist_1000_vs_text_0.95": t4[1000][1],
        "rate_monotone_nonincreasing_with_fewer_steps": all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1)),
        "dist_monotone": all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1)),
        "rate_fraction_in_first_step_row_1000_to_900": (t4[1000][0] - t4[900][0]) / t4[1000][0],
        "discretized_decoder_nll_bits_given_rmse_0.95": nll,
        "nll_min_over_sigma_bits": float(vals[imin]), "sigma_at_min_255scale": float(grid[imin]),
        "sigma_255scale_roots_for_1.97_gauss_err": sig_fit,
        "sigma_roots_as_variance_in_[-1,1]_scale": [(r / 127.5) ** 2 for r in sig_fit],
        "beta_1": float(beta1), "beta_tilde_2": float(bt2),
        "const_predictor_rmse_255": const_rmse, "table4_rmse_100_steps": 67.60125,
        "abar_at_T-t+1=100": abar_at[k], "amplification_1/sqrt(abar)": amp,
        "excess_mse_x_[-1,1]^2": mse_x_excess,
        "implied_excess_eps_mse_per_dim": mse_x_excess * abar_at[k] / (1 - abar_at[k]),
        "eps_mse_floor_if_Ex2=0.256": 0.256 * abar_at[k] / (1 - abar_at[k]),
    }


def table2_3():
    # 표 2 (CIFAR10 비조건부): IS ± (분할 표준편차로 가정, 10분할), FID
    t2 = {"mu~ pred, L, learned diag Sigma": (7.28, 0.10, 23.69), "mu~ pred, L, fixed isotropic Sigma": (8.06, 0.09, 13.22),
          "eps pred, L, fixed isotropic Sigma": (7.67, 0.13, 13.51), "eps pred, L_simple": (9.46, 0.11, 3.17)}
    def z(a, b, splits=10):
        d = t2[a][0] - t2[b][0]
        se = math.sqrt(t2[a][1] ** 2 + t2[b][1] ** 2) / math.sqrt(splits)
        return {"diff_IS": d, "se_if_pm_is_split_std_10": se, "z": d / se}
    comp = {"eps L_simple vs eps L": z("eps pred, L_simple", "eps pred, L, fixed isotropic Sigma"),
            "mu~ L vs eps L (fixed Sigma)": z("mu~ pred, L, fixed isotropic Sigma", "eps pred, L, fixed isotropic Sigma")}
    fid = {"eps L vs mu~ L (FID, fixed Sigma)": 13.51 - 13.22, "eps L_simple vs eps L": 13.51 - 3.17,
           "learned Sigma vs fixed (mu~, L)": 23.69 - 13.22}
    R["T2_ablation"] = {"IS_pairs": comp, "FID_gaps": fid,
                        "note": "FID에는 +-가 없다. 최종 실험은 한 번 학습했고(부록 B) 학습 중 평가한 최소 FID를 보고했다."}
    # 표 3, 표 1 FID 비교
    R["T3_T1_fid"] = {
        "ours_Lsimple_vs_StyleGAN2+ADA_v1_uncond(3.26)": 3.26 - 3.17,
        "ours_vs_StyleGAN2+ADA_v1_cond(2.67)": 3.17 - 2.67,
        "lsun_bedroom_ours_6.36_vs_ProgressiveGAN_8.34": 8.34 - 6.36, "lsun_church_ours_7.89_vs_ProgressiveGAN_6.42": 7.89 - 6.42,
        "lsun_cat_ours_19.75_vs_ProgressiveGAN_37.52": 37.52 - 19.75, "lsun_bedroom_large_4.90_vs_StyleGAN_2.65": 4.90 - 2.65,
        "cifar_test_set_FID_5.24_minus_train_FID_3.17": 5.24 - 3.17}


def count_unet(ch, ch_mult, nres, attn_res, size, in_ch=3, out_ch=3):
    """공식 코드(diffusion_tf/models/unet.py)의 구조를 따라 모수 수를 센다. 그룹 정규화는 채널당 (scale, shift) 2개."""
    temb = 4 * ch

    def conv(cin, cout, k=3):
        return k * k * cin * cout + cout

    def gn(c):
        return 2 * c

    def resnet(cin, cout):
        p = gn(cin) + conv(cin, cout) + (temb * cout + cout) + gn(cout) + conv(cout, cout)
        if cin != cout:
            p += cin * cout + cout  # nin shortcut (1x1)
        return p

    def attn(c):
        return gn(c) + 4 * (c * c + c)

    p = (ch * temb + temb) + (temb * temb + temb)  # 시간 임베딩 dense0, dense1
    p += conv(in_ch, ch)
    hs = [ch]
    h, res = ch, size
    for i, m in enumerate(ch_mult):
        out = ch * m
        for _ in range(nres):
            p += resnet(h, out)
            h = out
            if res in attn_res:
                p += attn(h)
            hs.append(h)
        if i != len(ch_mult) - 1:
            p += conv(h, h)
            hs.append(h)
            res //= 2
    p += resnet(h, h) + attn(h) + resnet(h, h)
    for i in reversed(range(len(ch_mult))):
        out = ch * ch_mult[i]
        for _ in range(nres + 1):
            p += resnet(h + hs.pop(), out)
            h = out
            if res in attn_res:
                p += attn(h)
        if i != 0:
            p += conv(h, h)
            res *= 2
    assert not hs
    p += gn(h) + conv(h, out_ch)
    return p


def appendix_b():
    cifar = count_unet(128, (1, 2, 2, 2), 2, (16,), 32)
    big = count_unet(128, (1, 1, 2, 2, 4, 4), 2, (16,), 256)
    ch192 = count_unet(192, (1, 1, 2, 2, 4, 4), 2, (16,), 256)
    R["B_params"] = {"cifar10_unet2d16b2": cifar, "paper_cifar_35.7M_ratio": cifar / 35.7e6,
                     "lsun_celebahq_ch128_mult112244": big, "paper_114M_ratio": big / 114e6,
                     "hypothetical_ch192": ch192, "paper_256M_ratio": ch192 / 256e6,
                     "note": "ch=192는 추정이다(공식 코드에는 256M 설정이 없다). 논문은 '필터 수를 늘렸다'고만 쓴다."}
    steps_s = 21.0
    R["B_time"] = {
        "cifar_800k_steps_hours": 800000 / steps_s / 3600, "paper_10.6h": 10.6,
        "cifar_sample_256_17s_per_step_ms": 17.0 / 1000 * 1000,
        "cifar_FID_50k_samples_minutes": 50000 / 256 * 17 / 60,
        "cifar_img_per_s_sampling": 256 / 17,
        "lsun_bedroom_2.4M_steps_days": 2.4e6 / 2.2 / 86400, "lsun_cat_1.8M_days": 1.8e6 / 2.2 / 86400,
        "lsun_church_1.2M_days": 1.2e6 / 2.2 / 86400, "celebahq_0.5M_days": 0.5e6 / 2.2 / 86400,
        "lsun_sampling_128_per_300s_s_per_image": 300 / 128, "lsun_FID_50k_hours": 50000 / 128 * 300 / 3600,
        "cifar_train_images_seen_M": 800000 * 128 / 1e6, "cifar_epochs_of_50k": 800000 * 128 / 50000,
        "cifar_nfe_per_image_sampling": 1000,
    }


def main():
    table1()
    section43()
    table2_3()
    appendix_b()
    (HERE / "paper_arithmetic_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
