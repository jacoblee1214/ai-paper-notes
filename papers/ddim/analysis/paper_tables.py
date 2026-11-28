"""DDIM 논문(arXiv:2010.02502 v4)의 표 1·2·3을 다시 읽는다. numpy만 쓴다.

표의 수치는 PDF 본문(pdftotext)에서 옮겼다. 표 1·3은 FID 한 줄이고 표본 반복·신뢰구간이 없으므로,
표집 잡음 크기를 가정하는 민감도 분석으로 "구별 가능한 차이"를 따진다.
실행: python3 paper_tables.py
"""
import json
from pathlib import Path

import numpy as np

S_LIST = [10, 20, 50, 100, 1000]
ETAS = ["0.0", "0.2", "0.5", "1.0", "hat"]
T1 = {
    "CIFAR10": {"0.0": [13.36, 6.84, 4.67, 4.16, 4.04], "0.2": [14.04, 7.11, 4.77, 4.25, 4.09],
                "0.5": [16.66, 8.35, 5.25, 4.46, 4.29], "1.0": [41.07, 18.36, 8.01, 5.78, 4.73],
                "hat": [367.43, 133.37, 32.72, 9.99, 3.17]},
    "CelebA": {"0.0": [17.33, 13.73, 9.17, 6.53, 3.51], "0.2": [17.66, 14.11, 9.51, 6.79, 3.64],
               "0.5": [19.86, 16.06, 11.01, 8.09, 4.28], "1.0": [33.12, 26.03, 18.48, 13.93, 5.98],
               "hat": [299.71, 183.83, 71.71, 45.20, 3.26]},
}
T2_S = [10, 20, 50, 100, 200, 500, 1000]
T2_ERR = [0.014, 0.0065, 0.0023, 0.0009, 0.0004, 0.0001, 0.0001]   # 1e-4 로 반올림
T3 = {"Bedroom": {"ddim": [16.95, 8.89, 6.75, 6.62], "ddpm": [42.78, 22.77, 10.81, 6.81], "ddpm1000": 6.36},
      "Church": {"ddim": [19.45, 12.47, 10.84, 10.58], "ddpm": [51.56, 23.37, 11.16, 8.27], "ddpm1000": 7.89}}
S3 = [10, 20, 50, 100]


def speedup(row, ref, tol):
    """ref(기준 FID)의 (1+tol)배 이내가 되는 가장 작은 S와 그때의 가속(1000/S). 없으면 None."""
    for S, v in zip(S_LIST, row):
        if v <= ref * (1 + tol):
            return {"S": S, "speedup": 1000 // S, "fid": v, "ratio": v / ref}
    return None


def main():
    out = {}
    # ---- 표 1: 구조 검사
    chk = {}
    for ds, rows in T1.items():
        mono_S = all(all(np.diff(rows[e]) < 0) for e in ETAS)
        order = {}
        for j, S in enumerate(S_LIST):
            vals = [rows[e][j] for e in ETAS]
            order[str(S)] = {"increasing_in_eta_0_to_1": bool(all(np.diff(vals[:4]) > 0)),
                             "best": ETAS[int(np.argmin(vals))]}
        chk[ds] = {"fid_decreases_with_S_every_row": bool(mono_S), "by_S": order}
    out["table1_structure"] = chk

    # ---- 표 1: 가속 주장 (기준 = 같은 데이터셋 DDIM S=1000, 그리고 각 η 의 자기 S=1000)
    sp = {}
    for ds, rows in T1.items():
        ref = rows["0.0"][-1]
        sp[ds] = {"ref_ddim_S1000": ref,
                  "ratio_to_ref": {e: [round(v / ref, 3) for v in rows[e]] for e in ETAS},
                  "min_S_within": {f"{int(tol * 100)}%": speedup(rows["0.0"], ref, tol) for tol in (0.05, 0.1, 0.25, 0.5, 1.0)}}
        # 'DDPM 1000 단계(η=1)와 같은 품질'에 필요한 DDIM S: 로그 S 선형보간
        ref1 = rows["1.0"][-1]
        xs, ys = np.log(S_LIST[:-1]), np.array(rows["0.0"][:-1])
        S_eq = float(np.exp(np.interp(ref1, ys[::-1], xs[::-1]))) if ys.min() <= ref1 <= ys.max() else None
        sp[ds]["ddim_S_equal_to_eta1_S1000"] = {"eta1_S1000_fid": ref1, "ddim_S_interp": S_eq,
                                                "note": "S<=100 에서 DDIM FID가 η=1 S=1000 값에 못 미쳐 보간 불가(None)" if S_eq is None else "로그 S 선형보간"}
    out["table1_speedup"] = sp
    # 논문 5.1: "CelebA에서 100단계 DDPM의 FID는 20단계 DDIM과 비슷"
    out["claim_celeba_ddpm100_vs_ddim20"] = {"ddpm_eta1_S100": T1["CelebA"]["1.0"][3], "ddim_S20": T1["CelebA"]["0.0"][1],
                                              "rel_diff": T1["CelebA"]["1.0"][3] / T1["CelebA"]["0.0"][1] - 1}
    out["claim_cifar_ddpm100_vs_ddim20"] = {"ddpm_eta1_S100": T1["CIFAR10"]["1.0"][3], "ddim_S20": T1["CIFAR10"]["0.0"][1],
                                             "ddim_S50": T1["CIFAR10"]["0.0"][2]}

    # ---- 표 1: 단일 실행 잡음 민감도. 표집 FID의 상대 표준편차를 0.5%, 1%, 2%로 가정(과 가정 근거는 03 문서)
    sens = {}
    for rel_sd in (0.005, 0.01, 0.02):
        res = {}
        for ds, rows in T1.items():
            pairs = []
            for j, S in enumerate(S_LIST):
                for e in ("0.2", "0.5", "1.0"):
                    a, b = rows["0.0"][j], rows[e][j]
                    sd = rel_sd * np.sqrt(a * a + b * b)
                    z = (b - a) / sd
                    pairs.append({"S": S, "eta": e, "diff": round(b - a, 3), "z": round(float(z), 2), "resolved_z_gt_2": bool(z > 2)})
            res[ds] = pairs
        unresolved = [(ds, p["S"], p["eta"]) for ds, ps in res.items() for p in ps if not p["resolved_z_gt_2"]]
        sens[f"rel_sd_{rel_sd}"] = {"unresolved_vs_eta0": unresolved, "n_pairs": sum(len(v) for v in res.values()),
                                    "detail": res}
    out["table1_noise_sensitivity"] = sens

    # ---- 표 1: σ̂ 대 DDIM 의 S=1000 순위
    out["table1_sigma_hat_vs_ddim_S1000"] = {ds: {"hat": rows["hat"][-1], "ddim": rows["0.0"][-1], "eta1": rows["1.0"][-1],
                                                  "hat_minus_ddim": round(rows["hat"][-1] - rows["0.0"][-1], 3)} for ds, rows in T1.items()}

    # ---- 표 2: 재구성 오차의 S 의존성
    S2, E2 = np.array(T2_S, dtype=float), np.array(T2_ERR)
    fit = lambda idx: float(np.polyfit(np.log(S2[idx]), np.log(E2[idx]), 1)[0])
    loc = [float(np.log(E2[i + 1] / E2[i]) / np.log(S2[i + 1] / S2[i])) for i in range(len(S2) - 1)]
    out["table2"] = {"slope_S10_to_200": fit(slice(0, 5)), "slope_S10_to_100": fit(slice(0, 4)),
                     "local_slopes": [round(v, 2) for v in loc],
                     "rounding_note": "S>=500 의 0.0001 은 1e-4 반올림이라 참값이 5e-5–1.5e-4. 이 두 점은 기울기 적합에서 제외",
                     "rmse_ratio_S10_to_S100": float(np.sqrt(E2[0] / E2[3])),
                     "mse_ratio_S10_to_S100": float(E2[0] / E2[3])}

    # ---- 표 3: LSUN
    t3 = {}
    for ds, v in T3.items():
        t3[ds] = {"ddpm1000": v["ddpm1000"],
                  "ddim_ratio_to_ddpm1000": [round(x / v["ddpm1000"], 3) for x in v["ddim"]],
                  "ddpm_eta1_ratio_to_ddpm1000": [round(x / v["ddpm1000"], 3) for x in v["ddpm"]],
                  "ddim_minus_ddpm_same_S": [round(a - b, 2) for a, b in zip(v["ddim"], v["ddpm"])],
                  "ddim_wins_S": [S for S, a, b in zip(S3, v["ddim"], v["ddpm"]) if a < b]}
    out["table3"] = t3

    # ---- 14개 (η=0 대 η=1) 같은 S 비교에서 DDIM 이 이기는 수
    wins = sum(1 for rows in T1.values() for j in range(5) if rows["0.0"][j] < rows["1.0"][j])
    wins += sum(1 for v in T3.values() for a, b in zip(v["ddim"], v["ddpm"]) if a < b)
    out["ddim_beats_eta1_count"] = {"wins": wins, "of": 2 * 5 + 2 * 4}
    Path(__file__).with_name("paper_tables_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("table1_structure", "table1_speedup", "claim_celeba_ddpm100_vs_ddim20",
                                          "table2", "table3", "ddim_beats_eta1_count", "table1_sigma_hat_vs_ddim_S1000")},
                     indent=1, ensure_ascii=False)[:6000])
    for k, v in sens.items():
        print(k, v["unresolved_vs_eta0"])


if __name__ == "__main__":
    main()
