"""논문 표 값 산술 확인: 표 1(검출 한계), 표 2(공개 모델 감사: 균등성·다중 검정), 표 3(토큰 산술), 표 4(MMLU Fisher 합산).
입력은 paper/paper.txt (pdftotext -layout). 결과: analysis/recompute_results.json
실행: python3 -I analysis/recompute.py   (수 초)
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
txt = (ROOT / "paper" / "paper.txt").read_text(encoding="utf-8")
res = {}

# ---- 표 1: 순열 검정 하한과 검출 ------------------------------------------------
t1 = {"BoolQ": (1, .099, .156), "HellaSwag": (1, .485, .478), "OpenbookQA": (1, .544, .462),
      "MNLI": (10, .009, 1.96e-11), "TruthfulQA": (10, .009, 3.43e-13), "NaturalQuestions": (10, .009, 1e-38),
      "PIQA": (50, .009, 1e-38), "MMLU_ProPsych": (50, .009, 1e-38), "MMLU_ProLaw": (50, .009, 1e-38),
      "MMLU_HSPsych": (100, .009, 1e-38)}
res["table1"] = {
    "perm_floor_if_100_perms": 1 / 101,          # 0.0099: 논문 표의 0.009는 이 값의 소수점 아래 버림으로 보임
    "perm_floor_if_110_perms": 1 / 111,          # 0.0090: m=110이면 0.009. 본문의 '100'과 '50'이 서로 안 맞음
    "perms_needed_for_1.96e-11": 1 / 1.96e-11,   # 5.1e10: 논문은 '10^11'
    "perm_floor_for_25_perms": 1 / 26,           # 0.0385: §4.2 '0.038'
    "detected_sharded_005": [k for k, (d, _, ps) in t1.items() if ps < .05],
    "detected_perm_005": [k for k, (d, pp, _) in t1.items() if pp < .05],
    "dup1_all_missed_both": all(ps > .05 and pp > .05 for d, pp, ps in t1.values() if d == 1),
    "dup_ge10_all_detected_both": all(ps < .05 and pp < .05 for d, pp, ps in t1.values() if d >= 10),
}

# ---- 표 3: 토큰 산술 --------------------------------------------------------------
tok = {"BoolQ": (110, 1), "HellaSwag": (185, 1), "OpenbookQA": (20, 1), "NaturalQ": (32, 10), "MNLI": (235, 10),
       "TruthfulQA": (25, 10), "PIQA": (50, 50), "MMLU_Law": (200, 50), "MMLU_Psych": (30, 50), "MMLU_HSPsych": (20, 100)}
inj = sum(a * b for a, b in tok.values()) * 1e3
res["table3"] = {
    "injected_tokens_with_dup": inj,                        # 19.235M: 부록 C와 일치
    "frac_of_20.2B": inj / 20.2e9,                          # 0.095% (<0.1%)
    "steps_46000_tokens": 46000 * 256 * 2048,               # 24.1B
    "steps_needed_for_20.2B_once": 20.2e9 / (256 * 2048),   # 38.5k
    "ratio_trained_over_dataset": 46000 * 256 * 2048 / 20.2e9,
    "law_words_x_examples": 2000 * 1000,                    # 표 3의 '2000 단어/문항'이면 2M 단어, 토큰 200k와 모순
}

# ---- 표 2: 공개 모델 감사 -----------------------------------------------------------
t2 = {  # 데이터셋: [LLaMA2, Mistral, Pythia, GPT-2 XL, BioMedLM]
    "Arc-Easy": [.318, .001, .686, .929, .795], "BoolQ": [.421, .543, .861, .903, .946],
    "GSM8K": [.594, .507, .619, .770, .975], "LAMBADA": [.284, .944, .969, .084, .427],
    "NaturalQA": [.912, .700, .948, .463, .595], "OpenBookQA": [.513, .638, .364, .902, .236],
    "PIQA": [.877, .966, .956, .959, .619]}
a = np.array(list(t2.values()))
flat = a.ravel()
res["table2"] = {
    "n_pvalues": int(flat.size),
    "n_below_005": int((flat < .05).sum()), "expected_below_005_under_null": .05 * flat.size,
    "n_below_001": int((flat <= .001).sum()),
    "n_above_09": int((flat > .9).sum()), "expected_above_09_under_null": .1 * flat.size,
    "frac_above_05": float((flat > .5).mean()),
    "mean_p": float(flat.mean()),
    "ks_vs_uniform_stat_p": [float(x) for x in stats.kstest(flat, "uniform")],
    "ks_excluding_mistral_arc": [float(x) for x in stats.kstest(np.delete(flat, 1), "uniform")],
    "col_means_llama_mistral_pythia_gpt2_biomed": a.mean(0).tolist(),
    "bonferroni_threshold_28_tests_4models_x7": .05 / 28,
    "bonferroni_threshold_35_tests_5models_x7": .05 / 35,
    "bonferroni_adjusted_mistral_arc_28": .001 * 28, "bonferroni_adjusted_mistral_arc_35": .001 * 35,
    "bonferroni_adjusted_mistral_arc_65_files": .001 * 65,
    "min_p_of_28_null_uniform_leq_0.001_prob": 1 - (1 - .001) ** 28,
    "binom_tail_P(X>=12 | n=35, p=0.1)": float(stats.binom.sf(11, 35, .1)),
}

# ---- 표 4: MMLU 전체 파일별 p 와 Fisher 합산 ------------------------------------------
rows = []
start = txt.index("Table 4: Full MMLU")
for line in txt[start:].splitlines():
    m = re.match(r"\s*([A-Z][A-Za-z\-]+)\s+(\d+)\s+((?:\S+\s+){4}\S+)\s*$", line)
    if m and not line.strip().startswith("Dataset"):
        vals = [float(x.replace("e-38", "e-38")) for x in m.group(3).split()]
        rows.append((m.group(1), int(m.group(2)), vals))
names = [r[0] for r in rows]
P = np.array([r[2] for r in rows])  # 열: LLaMA2, Mistral, Pythia, GPT-2 XL, BioMedLM
res["table4"] = {"n_files_parsed": len(rows), "sum_sizes": int(sum(r[1] for r in rows))}


def fisher(p):
    p = np.maximum(p, 1e-38)
    return float(stats.chi2.sf(-2 * np.log(p).sum(), 2 * len(p)))


# '두 음성 대조(GPT-2 XL, BioMedLM) 중 하나라도 유의'한 파일 제외. 임계값은 논문에 없으니 14개가 되는 값을 찾는다.
thr_scan = {}
for thr in [.05, .01, .005, .003, .002, .001]:
    excl = (P[:, 3] <= thr) | (P[:, 4] <= thr)
    keep = ~excl
    thr_scan[str(thr)] = {"n_excluded": int(excl.sum()),
                          "fisher_llama2": fisher(P[keep, 0]), "fisher_mistral": fisher(P[keep, 1]),
                          "fisher_pythia": fisher(P[keep, 2]), "n_kept": int(keep.sum())}
res["table4"]["threshold_scan"] = thr_scan
res["table4"]["paper_reported_fisher"] = {"llama2": .014, "mistral": .011, "pythia": .362}
res["table4"]["fisher_all_57_files"] = {n: fisher(P[:, j]) for j, n in enumerate(["llama2", "mistral", "pythia", "gpt2xl", "biomedlm"])}
# 균등성: 필터 전 음성 대조 열의 p 중 0.05 미만 비율(기대 5%)
res["table4"]["negative_control_below_005"] = {"gpt2xl": int((P[:, 3] < .05).sum()), "biomedlm": int((P[:, 4] < .05).sum()),
                                               "expected": .05 * len(P)}
keep05 = ~((P[:, 3] <= .05) | (P[:, 4] <= .05))
res["table4"]["after_filter_005"] = {
    "n_kept": int(keep05.sum()),
    "n_below_005_per_model": {n: int((P[keep05, j] < .05).sum()) for j, n in enumerate(["llama2", "mistral", "pythia"])},
    "expected_below_005": .05 * int(keep05.sum()),
    "ks_p_vs_uniform": {n: float(stats.kstest(P[keep05, j], "uniform").pvalue) for j, n in enumerate(["llama2", "mistral", "pythia"])},
    "excluded_files": [names[i] for i in range(len(names)) if not keep05[i]],
    "note": "표 4는 56행(professional_law 없음). 13개 제외 + professional_law(공식 zip에 1534문항으로 존재)를 14번째로 보면 본문의 '14개 제외, 43개 남음'과 일치",
}
res["table4"]["negative_control_1e-38_files"] = [names[i] for i in range(len(names)) if (P[i, 3:] <= 1e-30).any()]
(ROOT / "analysis" / "recompute_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
