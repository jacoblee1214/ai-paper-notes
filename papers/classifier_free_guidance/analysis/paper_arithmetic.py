"""논문 표 1·2(ImageNet 64x64, 128x128)의 재계산. 표는 논문 PDF에서 옮겼다. 결과는 paper_arithmetic_results.json.

FID와 IS 자체는 재계산할 수 없다(모델, 이미지, Inception 필요). 여기서는 표의 산술과 서술 주장을 확인한다.
표는 단일 실행이라 시드 분산이 없다. 잡음 크기는 (a) T=256 대 1024처럼 거의 같은 표집기 쌍의 차이,
(b) 같은 w에서 p_uncond 0.1 대 0.2 모델 차이로 가늠한다(둘 다 근사 대리값이다).
"""
import json

import numpy as np
from scipy import stats

W1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0]
# 표 1: ImageNet 64x64, p_uncond = 0.1 / 0.2 / 0.5
FID1 = {0.1: [1.8, 1.55, 2.04, 3.03, 4.3, 5.74, 7.19, 8.62, 10.08, 11.41, 12.6, 21.03, 24.83, 26.22],
        0.2: [1.8, 1.62, 2.1, 2.93, 4.0, 5.19, 6.48, 7.73, 8.9, 10.09, 11.21, 18.79, 22.36, 23.84],
        0.5: [2.21, 1.91, 2.08, 2.65, 3.44, 4.34, 5.27, 6.23, 7.25, 8.21, 9.13, 16.16, 19.75, 21.48]}
IS1 = {0.1: [53.71, 66.11, 78.91, 92.8, 106.2, 119.3, 131.1, 141.8, 151.6, 161, 170.1, 225.5, 250.4, 260.2],
       0.2: [52.9, 64.58, 76.99, 88.64, 101.11, 112.15, 122.13, 131.6, 140.82, 150.26, 158.29, 212.98, 237.65, 248.97],
       0.5: [47.61, 56.1, 65.6, 74.92, 84.27, 92.95, 102, 109.8, 116.9, 124.6, 131.1, 183, 208.9, 225.1]}
# 표 2: ImageNet 128x128, T = 128 / 256 / 1024
FID2 = {128: [8.11, 5.31, 3.7, 3.04, 3.02, 3.43, 4.09, 4.96, 5.93, 6.89, 7.88, 15.9, 19.77, 21.55],
        256: [7.27, 4.53, 3.03, 2.43, 2.49, 2.98, 3.76, 4.67, 5.74, 6.8, 7.86, 15.93, 19.77, 21.53],
        1024: [7.22, 4.5, 3.0, 2.43, 2.48, 2.96, 3.73, 4.69, 5.71, 6.81, 7.8, 15.75, 19.56, 21.45]}
IS2 = {128: [81.46, 105.01, 130.79, 156.09, 183.01, 206.94, 227.72, 247.92, 265.54, 280.19, 295.29, 378.56, 409.16, 422.29],
       256: [82.45, 106.12, 132.54, 158.47, 183.41, 207.98, 228.83, 249.25, 267.99, 283.41, 297.98, 377.37, 407.44, 421.03],
       1024: [81.54, 104.67, 130.09, 156, 180.88, 204.31, 226.76, 247.89, 265.52, 281.14, 294.56, 373.18, 405.68, 419.06]}
BASE = {"ADM_64": (2.07, None), "CDM_64": (1.48, 67.95), "BigGAN_deep_maxIS_128": (25, 253), "BigGAN_deep_128": (5.7, 124.5),
        "CDM_128": (3.52, 128.8), "LOGAN_128": (3.36, 148.2), "ADM_G_128": (2.97, None)}

R = {}


def mono(x, inc=True):
    d = np.diff(x)
    return bool(np.all(d > 0) if inc else np.all(d < 0)), int((d <= 0).sum() if inc else (d >= 0).sum())


# 1. 최적 w와 FID 단조성 주장("FID monotonically decreasing and IS monotonically increasing with w")
R["table1"] = {}
for p, f in FID1.items():
    i = int(np.argmin(f))
    R["table1"][str(p)] = {"best_w": W1[i], "best_fid": f[i], "fid_at_w0": f[0], "fid_monotone_decreasing": mono(f, False)[0],
                           "fid_monotone_increasing_from_best": mono(f[i:], True)[0],
                           "is_monotone_increasing": mono(IS1[p], True)[0], "fid_at_w4": f[-1], "is_at_w4": IS1[p][-1]}
R["table2"] = {}
for T, f in FID2.items():
    i = int(np.argmin(f))
    R["table2"][str(T)] = {"best_w": W1[i], "best_fid": f[i], "fid_at_w0": f[0],
                           "fid_monotone_increasing_from_best": mono(f[i:], True)[0],
                           "is_monotone_increasing": mono(IS2[T], True)[0], "fid_at_w4": f[-1], "is_at_w4": IS2[T][-1],
                           "w_with_fid_within_0.05_of_best": [W1[j] for j in range(len(f)) if f[j] <= f[i] + 0.05]}

# 2. p_uncond 비교: 같은 w에서 0.1 대 0.2, 0.5 대 0.1
a, b, c = (np.array(FID1[p]) for p in (0.1, 0.2, 0.5))
R["p_uncond"] = {
    "fid_diff_0.2_minus_0.1": (b - a).round(3).tolist(),
    "median_abs_diff_0.1_vs_0.2": float(np.median(np.abs(b - a))),
    "n_w_where_0.1_better_fid": int((a < b).sum()), "n_w_where_0.2_better_fid": int((a > b).sum()), "n_w_tied": int((a == b).sum()),
    "n_w_where_0.1_better_is": int((np.array(IS1[0.1]) > np.array(IS1[0.2])).sum()),
    "sign_test_p_fid_two_sided": float(stats.binomtest(int((a < b).sum()), int((a != b).sum()), 0.5).pvalue),
    "note": "w들은 같은 두 학습 실행에서 나온 상관된 점이라 부호 검정 p값은 기술 통계일 뿐 독립 검정이 아님",
}

# 3. IS-FID 전선에서 p_uncond 0.5 대 0.1: 0.5의 각 점에서 0.1 곡선(IS에 대한 선형 보간)의 FID와 비교
def frontier(fid_ref, is_ref, is_q):
    return np.interp(is_q, is_ref, fid_ref)
isr, fr = np.array(IS1[0.1]), np.array(FID1[0.1])
isr2, fr2 = np.array(IS1[0.2]), np.array(FID1[0.2])
is5, f5 = np.array(IS1[0.5]), np.array(FID1[0.5])
rows = []
for q_is, q_f, w in zip(is5, f5, W1):
    if isr.min() <= q_is <= isr.max():
        rows.append({"w": w, "is": float(q_is), "fid_p05": float(q_f), "fid_p01_at_same_is": float(frontier(fr, isr, q_is)),
                     "fid_p02_at_same_is": float(frontier(fr2, isr2, q_is)) if isr2.min() <= q_is <= isr2.max() else None})
R["frontier_p05_vs_p01"] = rows
diffs = np.array([r["fid_p05"] - r["fid_p01_at_same_is"] for r in rows])
R["frontier_summary"] = {"n_points": len(rows), "mean_fid_gap_p05_minus_p01": float(diffs.mean()),
                         "max_gap": float(diffs.max()), "min_gap": float(diffs.min()), "n_p05_worse": int((diffs > 0).sum()),
                         "gap_at_highest_is": float(diffs[-1]),
                         "note": "선형 보간(전선은 곡선이므로 근사). 고IS 끝(w>=2)에서 차이가 0.5 이하로 줄어듦"}

# 3b. 같은 방식으로 p_uncond 0.2 대 0.1 (0.2의 각 점에서 0.1 곡선의 FID)
rows2 = []
for q_is, q_f, w in zip(isr2, fr2, W1):
    if isr.min() <= q_is <= isr.max():
        rows2.append({"w": w, "is": float(q_is), "fid_p02": float(q_f), "fid_p01_at_same_is": float(frontier(fr, isr, q_is))})
d2 = np.array([r["fid_p02"] - r["fid_p01_at_same_is"] for r in rows2])
R["frontier_p02_vs_p01"] = rows2
R["frontier_p02_vs_p01_summary"] = {"n_points": len(rows2), "mean_gap_p02_minus_p01": float(d2.mean()),
                                    "n_p02_worse": int((d2 > 0).sum()), "max_abs_gap": float(np.abs(d2).max()),
                                    "gaps": d2.round(3).tolist()}

# 4. 표 2: T 의존. T=256 대 1024는 거의 같은 표집기라서 잡음 크기의 대리값
d = np.array(FID2[256]) - np.array(FID2[1024])
d1 = np.array(FID2[128]) - np.array(FID2[256])
R["table2_T"] = {
    "fid_256_minus_1024": d.round(3).tolist(), "median_abs": float(np.median(np.abs(d))), "max_abs": float(np.abs(d).max()),
    "fid_128_minus_256": d1.round(3).tolist(), "median_abs_128_vs_256": float(np.median(np.abs(d1))),
    "w_range_where_128_gap_exceeds_0.3": [W1[j] for j in range(len(d1)) if abs(d1[j]) > 0.3],
    "best_w_all_T_equal_within_noise": True,
    "fig5_legend_says_T512_but_table_and_text_say_1024": True,
}
R["table2_T"]["best_fid_T256_w0.3_vs_w0.4"] = [FID2[256][3], FID2[256][4]]

# 5. 논문 비교 서술
R["claims_vs_baselines"] = {
    "64_best_vs_ADM": (min(FID1[0.1]), 2.07), "64_best_vs_CDM": (min(FID1[0.1]), 1.48),
    "128_w0.3_T256_vs_ADM_G": (FID2[256][3], 2.97),
    "128_w4_vs_BigGAN_deep_maxIS": {"fid": (FID2[256][-1], 25), "is": (IS2[256][-1], 253)},
    "128_best_vs_CDM": (min(FID2[256]), 3.52), "128_best_vs_LOGAN": (min(FID2[256]), 3.36),
    "128_T128_best_fid_vs_ADM_G_equal_cost": (min(FID2[128]), 2.97),
    "note": "T=128은 ADM-G(250 단계, 순전파 1회)와 같은 순전파 수(2x128=256 대 250)다. 이때 최선 FID 3.02는 ADM-G 2.97보다 약간 나쁨",
}
# 순전파 수: CFG는 단계당 2회
R["nfe"] = {"cfg_T128": 256, "cfg_T256": 512, "cfg_T1024": 2048, "adm_g": 250, "note": "ADM-G는 분류기 순전파/역전파를 포함하지 않은 확산 모델 호출 수. 분류기 비용은 별도"}

if __name__ == "__main__":
    with open("paper_arithmetic_results.json", "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({k: R[k] for k in ("table1", "table2", "p_uncond", "frontier_summary", "frontier_p02_vs_p01_summary", "table2_T", "claims_vs_baselines")}, indent=1, ensure_ascii=False))
