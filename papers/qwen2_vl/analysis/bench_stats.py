"""논문 표 2, 7, 8과 그림 5의 차이를 Wilson 구간과 MDE로 따져 본다 (표준 라이브러리, 수 초).

- 표의 수치는 `paper/paper.txt`(v2)에서 옮긴 것이다.
- 문항 수 n은 논문에 없다. Hugging Face datasets-server `size` API(2026-11-26 조회)와 공개 데이터셋 통계로 채웠고,
  어느 분할을 썼는지 논문이 적지 않은 곳은 `assumed`로 표시했다.
- MDE = 2.80 * sqrt(d / n) (쌍 비교, d = 두 모델이 다르게 답하는 문항 비율, 검정력 80%, 양측 5%).
  d=0.2를 기본으로 하고 0.1, 0.3을 민감도로 둔다. 문항별 출력이 공개되지 않아 d는 실제로 알 수 없다.
- 점수가 정확도가 아닌 벤치마크(ANLS, 상대 정확도, GPT 판정, 합산 점수)는 `approx`로 두고 참고용으로만 쓴다.

실행: python3 analysis/bench_stats.py   (결과: analysis/bench_results.json)
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).with_name("bench_results.json")


def wilson(p, n, z=1.96):
    if n is None:
        return None
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def mde(n, d):
    return None if n is None else 100 * 2.80 * math.sqrt(d / n)


# 이름: (n, 근거, 종류, [prev, claude, gpt4o, q72, q7, q2], 배율)
# prev/claude/gpt4o가 None이면 표의 '-'
T2 = {
    "MMMU val": (900, "MMMU 30과목x30 (HF size API)", "acc", [66.1, 68.3, 69.1, 64.5, 54.1, 41.1], 1),
    "DocVQA test": (5188, "lmms-lab DocVQA test (HF size API)", "approx(ANLS)", [94.1, 95.2, 92.8, 96.5, 94.5, 90.1], 1),
    "InfoVQA test": (3288, "InfographicVQA test (HF size API)", "approx(ANLS)", [82.0, None, None, 84.5, 76.5, 65.5], 1),
    "AI2D": (3088, "lmms-lab ai2d test (HF size API)", "acc", [87.6, 80.2, 84.6, 88.1, 83.0, 74.7], 1),
    "ChartQA test": (2500, "lmms-lab ChartQA test (HF size API)", "approx(relaxed acc)", [88.4, 90.8, 85.7, 88.3, 83.0, 73.5], 1),
    "TextVQA val": (5000, "표준 val 5,000 (assumed)", "approx(VQA acc)", [84.4, None, None, 85.5, 84.3, 79.7], 1),
    "OCRBench": (1000, "echo840/OCRBench 1,000 (HF size API), 점수=맞은 문항 수", "acc", [852, 788, 736, 877, 866, 809], 0.1),
    "RealWorldQA": (765, "lmms-lab RealWorldQA test (HF size API)", "acc", [72.2, 60.1, 75.4, 77.8, 70.1, 62.9], 1),
    "MMBench-EN test": (6666, "lmms-lab MMBench en test (HF size API)", "acc", [86.5, 79.7, 83.4, 86.5, 83.0, 74.9], 1),
    "MMBench-CN test": (6666, "lmms-lab MMBench cn test (HF size API)", "acc", [86.3, 80.7, 82.1, 86.6, 80.5, 73.5], 1),
    "MMStar": (1500, "Lin-Chen/MMStar val 1,500 (HF size API)", "acc", [67.1, 62.2, 63.9, 68.3, 60.7, 48.0], 1),
    "MMVet (GPT-4-Turbo 판정)": (218, "lmms-lab MMVet 218 (HF size API)", "approx(GPT 판정, 연속 점수)", [67.5, 66.0, 69.1, 74.0, 62.0, 49.5], 1),
    "MathVista testmini": (1000, "AI4Math/MathVista testmini (HF size API)", "acc", [69.0, 67.7, 63.8, 70.5, 58.2, 43.0], 1),
    "MathVision": (3040, "논문 3.2.4가 3,040문항이라 명시", "acc", [30.3, None, 30.4, 25.9, 16.3, 12.4], 1),
    "MMMU-Pro": (1730, "MMMU/MMMU_Pro standard 1,730 (HF size API; 논문 설정 미기재)", "acc", [46.9, 51.5, 51.9, 46.2, 43.5, 37.6], 1),
}
# 표에서 문항 수를 확인하지 못해 건너뛴 행: MTVQA, VCR(en/zh), MME 합산(2,374문항이지만 점수가 정확도가 아님), MMBench-V1.1,
# MMT-Bench, HallBench 평균.

res = {"table2": [], "table7": [], "table8": [], "fig5": []}
for name, (n, src, kind, vals, scale) in T2.items():
    prev, claude, gpt, q72, q7, q2 = vals
    r = {"benchmark": name, "n": n, "n_source": src, "kind": kind, "q72": q72}
    p72 = q72 * scale / 100
    r["q72_wilson"] = wilson(p72, n)
    if scale != 1:
        r["q72_wilson"] = [x for x in r["q72_wilson"]]  # 이미 % 단위 (OCRBench는 /1000 -> %)
    r["mde"] = {f"d={d}": mde(n, d) for d in (0.1, 0.2, 0.3)}
    for lab, other in (("prev_sota", prev), ("claude35", claude), ("gpt4o", gpt)):
        if other is None:
            r[lab] = None
            continue
        delta = (q72 - other) * scale
        m2 = mde(n, 0.2)
        r[lab] = {"other": other, "delta_pts": delta, "outside_mde_d0.2": abs(delta) >= m2, "outside_mde_d0.3": abs(delta) >= mde(n, 0.3),
                  "outside_mde_d0.1": abs(delta) >= mde(n, 0.1)}
    r["q72_minus_q7_pts"] = (q72 - q7) * scale
    res["table2"].append(r)

# ---- 표 7: 고정 토큰 수 대 동적 해상도 (Qwen2-VL-7B, 한 번의 실행)
T7 = {"tokens": [64, 576, 1600, 3136, "dynamic(1924)"],
      "InfoVQA val": (2801, [28.85, 65.72, 74.99, 77.27, 75.89], "approx(ANLS)"),
      "RealWorldQA": (765, [56.47, 65.88, 69.54, 70.59, 70.07], "acc"),
      "OCRBench": (1000, [572, 828, 824, 786, 866], "acc(/1000)"),
      "MMMU": (900, [53.33, 52.78, 52.89, 53.44, 53.44], "acc")}
for k, (n, vals, kind) in list(T7.items())[1:]:
    sc = 0.1 if k == "OCRBench" else 1
    dyn = vals[-1] * sc
    rows = []
    for tk, v in zip(T7["tokens"][:-1], vals[:-1]):
        delta = dyn - v * sc
        rows.append({"fixed_tokens": tk, "fixed": v, "dynamic_minus_fixed_pts": delta, "mde_d0.2": mde(n, 0.2), "outside": abs(delta) >= mde(n, 0.2)})
    res["table7"].append({"benchmark": k, "n": n, "kind": kind, "rows": rows,
                          "range_across_fixed_ge576": max(v * sc for v in vals[1:-1]) - min(v * sc for v in vals[1:-1])})

# ---- 표 8: 1D-RoPE 대 M-RoPE (Qwen2-1.5B + ViT-L, 사전학습 모델)
T8 = {"MathVista": (1000, "assumed testmini", 39.2, 43.4), "MMB": (4329, "assumed en dev", 58.6, 60.6), "MMStar": (1500, "val", 36.7, 36.7),
      "RWQ": (765, "test", 54.5, 53.7), "DocVQA": (5349, "assumed val", 82.5, 82.8), "ChartQA": (2500, "test", 68.0, 68.4),
      "InfoVQA": (2801, "assumed val", 50.8, 50.3), "TextVQA": (5000, "assumed val", 71.3, 71.8),
      "PerceptionTest": (None, "n 미확인", 46.6, 47.4), "NextQA": (None, "n 미확인", 43.9, 46.0), "STAR": (None, "n 미확인", 55.5, 57.9)}
for k, (n, note, a, b) in T8.items():
    delta = b - a
    res["table8"].append({"benchmark": k, "n": n, "n_note": note, "1d": a, "mrope": b, "delta_pts": delta,
                          "mde_d0.2": mde(n, 0.2), "outside": None if n is None else abs(delta) >= mde(n, 0.2)})
img = [r for r in res["table8"] if r["benchmark"] in ("MathVista", "MMB", "MMStar", "RWQ", "DocVQA", "ChartQA", "InfoVQA", "TextVQA")]
res["table8_summary"] = {"image_mean_delta": sum(r["delta_pts"] for r in img) / len(img),
                         "image_positive": sum(r["delta_pts"] > 0 for r in img), "image_zero": sum(r["delta_pts"] == 0 for r in img), "image_negative": sum(r["delta_pts"] < 0 for r in img),
                         "image_outside_mde": sum(bool(r["outside"]) for r in img), "n_image": len(img),
                         "video_deltas": [r["delta_pts"] for r in res["table8"] if r["benchmark"] in ("PerceptionTest", "NextQA", "STAR")]}

# ---- 그림 5: Video-MME medium (72B), 추정 n = 900 (300 영상 x 3 문항, 공개 통계 기억, 미확인)
fig5 = {"8K": 68.2, "16K": 70.1, "32K": 70.4, "48K": 71.3, "64K": 70.4, "80K": 71.1}
n5 = 900
res["fig5"] = {"n_assumed": n5, "values_read_from_figure": fig5, "mde_d0.2": mde(n5, 0.2),
               "delta_16K_to_80K": fig5["80K"] - fig5["16K"], "delta_8K_to_16K": fig5["16K"] - fig5["8K"],
               "max_minus_min_16K_to_80K": max(list(fig5.values())[1:]) - min(list(fig5.values())[1:])}

res["table8_video_conditional"] = {"note": "영상 벤치마크 n은 확인하지 못했다. n이 4,000~7,000이라고 가정할 때의 MDE(d=0.2)", "mde_n4000": mde(4000, 0.2), "mde_n7000": mde(7000, 0.2)}


# 표 2 요약: 비교 대상별로 72B가 MDE(d=0.2) 밖에서 앞선/뒤진 행, 안에 든 행의 수
summ = {}
for lab in ("gpt4o", "claude35", "prev_sota"):
    rows = [r for r in res["table2"] if r[lab] is not None]
    for dd in ("0.1", "0.2", "0.3"):
        up = [r["benchmark"] for r in rows if r[lab][f"outside_mde_d{dd}"] and r[lab]["delta_pts"] > 0]
        dn = [r["benchmark"] for r in rows if r[lab][f"outside_mde_d{dd}"] and r[lab]["delta_pts"] < 0]
        summ[f"{lab}_d{dd}"] = {"rows": len(rows), "ahead_outside": len(up), "behind_outside": len(dn), "inside": len(rows) - len(up) - len(dn),
                              "ahead_names": up, "behind_names": dn}
res["table2_summary"] = summ

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))

# 요약
print("표 2: 72B 대 GPT-4o / Claude / Previous SoTA (MDE d=0.2 기준 밖이면 O)")
for r in res["table2"]:
    def f(k):
        v = r[k]
        return "  -  " if v is None else f"{v['delta_pts']:+5.1f}{'O' if v['outside_mde_d0.2'] else 'x'}"
    w = r["q72_wilson"]
    print(f"{r['benchmark']:26s} n={r['n']:5d} q72={r['q72']:6.1f} wilson[{w[0]:.1f},{w[1]:.1f}] mde={r['mde']['d=0.2']:.2f}  gpt4o {f('gpt4o')} claude {f('claude35')} prev {f('prev_sota')}  [{r['kind']}]")
print("표 7")
for r in res["table7"]:
    print(r["benchmark"], r["n"], [(x["fixed_tokens"], round(x["dynamic_minus_fixed_pts"], 2), "O" if x["outside"] else "x") for x in r["rows"]], "range>=576", round(r["range_across_fixed_ge576"], 2))
print("표 8")
for r in res["table8"]:
    print(r["benchmark"], r["n"], r["delta_pts"], r["mde_d0.2"] and round(r["mde_d0.2"], 2), r["outside"])
print(res["table8_summary"])
print(res["fig5"])

for k, v in res["table2_summary"].items():
    print(k, v["rows"], v["ahead_outside"], v["behind_outside"], v["inside"], v["behind_names"] if k.endswith("d0.2") else "")
