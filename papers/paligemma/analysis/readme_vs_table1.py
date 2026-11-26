"""논문 표 1과 공식 저장소 README의 '전이 결과' 표(공개된 전이 체크포인트의 점수)를 대조한다.

README는 big_vision/configs/proj/paligemma/README.md 줄 175-247(커밋 0127fb6)에서 옮겨 적었다. 코드를 실행하지 않는다.
허용 오차: 표 1은 소수 첫째 자리, README는 소수 둘째 자리라서 |차이| <= 0.06이면 반올림 차이로 본다.
POPE는 README의 세 분할 평균, ChartQA는 README가 (human, aug)의 평균만 주므로 표 1 두 값의 평균과 비교한다.
출력: readme_vs_table1_results.json
"""
import json
from pathlib import Path

import paper_tables as T

HERE = Path(__file__).resolve().parent
rows = {r[1]: r for r in T.TABLE1}

# (표 1 과제명, 열 인덱스(0=224,1=448,2=896), README 값). README의 값이 여러 지표를 담을 때는 해당하는 것만 골랐다.
README = {
    "COCOcap": (141.92, 144.60, None), "NoCaps": (121.72, 123.58, None),
    "COCO-35L (en)": (139.2, 141.2, None), "COCO-35L (avg34)": (115.8, 118.0, None),
    "XM3600 (en)": (78.1, 80.0, None), "XM3600 (avg35)": (41.3, 41.9, None),
    "TextCaps": (127.48, 153.94, None), "SciCap": (162.25, 181.49, None), "Screen2Words": (117.57, 119.59, None),
    "WidgetCap": (136.07, 148.36, None), "VQAv2": (83.19, 85.64, None), "MMVP": (47.33, 45.33, None),
    "POPE": ((87.80 + 85.87 + 84.27) / 3, (88.23 + 86.77 + 85.90) / 3, None), "Objaverse Multiview": (62.7, 62.8, None),
    "OKVQA": (63.54, 63.15, None), "AOKVQA-MC": (76.37, 76.90, None), "AOKVQA-DA": (61.85, 63.22, None),
    "GQA": (65.61, 67.03, None), "NLVR2": (90.02, 88.93, None), "MARVL (avg5)": (80.57, 76.78, None),
    "AI2D": (72.12, 73.28, None), "ScienceQA": (95.39, 95.93, None), "RSVQA-lr": (92.65, 93.11, None),
    "RSVQA-hr (test)": (92.61, 92.79, None), "RSVQA-hr (test2)": (90.58, 90.54, None),
    "VizWizVQA": (73.7, 75.52, None), "TallyQA (simple)": (81.72, 84.86, None), "TallyQA (complex)": (69.56, 72.27, None),
    "OCR-VQA": (73.24, 75.60, 75.90), "TextVQA": (55.47, 73.15, 76.48), "DocVQA": (43.74, 78.02, 84.77),
    "InfoVQA": (28.46, 40.47, 47.75), "ST-VQA": (63.29, 81.82, 84.40),
    "ActivityNet-QA": (50.78, None, None), "ActivityNet-CAP": (34.62, None, None), "MSRVTT-QA": (50.09, None, None),
    "MSRVTT-CAP": (70.54, None, None), "MSVD-QA": (60.22, None, None), "VATEX": (79.73, None, None),
}
CHARTQA = {"mean(human,aug)": (57.08, 71.36)}

out, bad = [], []
n = 0
for name, vals in README.items():
    r = rows[name]
    for j, v in enumerate(vals):
        if v is None:
            continue
        pv = r[3 + j][0] if r[3 + j] else None
        if pv is None:
            continue
        n += 1
        d = round(pv - v, 2)
        ok = abs(d) <= 0.06
        out.append({"task": name, "res": (224, 448, 896)[j], "paper": pv, "readme": round(v, 2), "diff": d, "ok": ok})
        if not ok:
            bad.append((name, (224, 448, 896)[j], pv, round(v, 2), d))
for j, v in enumerate(CHARTQA["mean(human,aug)"]):
    pv = (rows["ChartQA (human)"][3 + j][0] + rows["ChartQA (aug)"][3 + j][0]) / 2
    n += 1
    d = round(pv - v, 2)
    out.append({"task": "ChartQA mean(human,aug)", "res": (224, 448)[j], "paper": round(pv, 2), "readme": v, "diff": d, "ok": abs(d) <= 0.06})
res = {"n_compared": n, "n_within_0.06": sum(1 for o in out if o["ok"]), "mismatches": bad, "rows": out,
       "notes": ["xGQA(avg7)는 README가 8개 언어(en 포함) 평균(58.37/59.07)이라 직접 비교하지 않았다(논문 avg7은 en 제외로 보이며 영어 점수를 약 66으로 두면 맞는다, 추정).",
                 "RefCOCO는 README가 검증 분할(73.40/68.32/67.65 등), 표 1이 시험 분할(testA/testB)이라 비교하지 않았다. 표 4의 RefCOCO(val) 73.4/68.3/67.7은 README와 맞는다.",
                 "README 요약: XM3600·COCO-35L 열의 정의(avg-35, avg-34, avg)는 README 표머리에 (en / avg-35 / avg)로만 적혀 있다."]}
# README 안의 내부 일관성: avg(36) = (en + 35·avg35)/36
res["internal_consistency"] = {
    "XM3600_224_readme_avg36_expected": round((78.1 + 35 * 41.3) / 36, 2), "XM3600_224_readme_avg36": 42.4,
    "XM3600_224_if_paper_avg35_41.9": round((78.0 + 35 * 41.9) / 36, 2),
    "XM3600_448_readme_avg36_expected": round((80.0 + 35 * 41.9) / 36, 2), "XM3600_448_readme_avg36": 42.9,
    "COCO35L_224_readme_avg35_expected": round((139.2 + 34 * 115.8) / 35, 2), "COCO35L_224_readme_all": 116.4,
}
(HERE / "readme_vs_table1_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"비교 {n}칸, 반올림 허용 일치 {res['n_within_0.06']}, 불일치 {len(bad)}")
for b in bad:
    print("  불일치:", b)
print(res["internal_consistency"])
