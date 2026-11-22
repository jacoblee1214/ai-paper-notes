"""논문이 보고한 비율의 산술 일치와 비교의 통계 검정.

논문에는 표가 없고 수치가 그림(막대·산점도)과 본문에만 있다. 입력은 모두 논문에서 옮긴 값이다.
  - "text": 본문에 숫자로 적힌 값(정확)
  - "bar": 막대 위 숫자 라벨(정확, 정수 %)
  - "read": 산점도에서 눈으로 읽은 값(약 ±2%p, 순위·상관 용도로만 사용)
표본 수: Impossible-LiveCodeBench 103문항, Impossible-SWEbench 349문항(논문 F.1, 공개 데이터셋과 일치).
검정은 두 비율의 독립 비교(Fisher 정확 검정)다. 같은 문항을 두 조건에서 푼 짝 설계이므로 이 검정은
보수적이다(짝 검정은 문항별 결과가 필요한데 논문이 공개하지 않았다). 90% 신뢰구간은 논문의 방식과 같게 Wilson/정규근사 중 Wilson을 쓴다.
출력: recompute_stats_results.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "recompute_stats_results.json"
N_LCB, N_SWE = 103, 349


def feasible_k(p_pct, n, digits):
    """보고된 p%(소수 digits자리 반올림)가 되는 정수 k 목록."""
    return [k for k in range(n + 1) if round(100 * k / n + 1e-12, digits) == round(p_pct, digits)]


def wilson(k, n, conf=0.90):
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, 100 * (c - h)), 1), round(100 * (c + h), 1)


def fisher(k1, n1, k2, n2):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


out = {}

# ---- A. 본문 수치가 k/n으로 설명되는가
claims = [
    ("GPT-5 Oneoff-LCB 2.9% (1절)", 2.9, N_LCB, 1),
    ("GPT-5 Conflicting-SWE 54.0% (4절)", 54.0, N_SWE, 1),
    ("GPT-5 Oneoff-SWE 76% (1절)", 76, N_SWE, 0),
    ("GPT-5 Conflicting-LCB 프롬프트 A 92% (그림 6)", 92, N_LCB, 0),
    ("GPT-5 Conflicting-LCB 프롬프트 D 1% (그림 6)", 1, N_LCB, 0),
    ("o3 Conflicting-LCB 프롬프트 D 33% (그림 6)", 33, N_LCB, 0),
    ("GPT-5 Conflicting-SWE 느슨한 프롬프트 66% (5.1절)", 66, N_SWE, 0),
    ("Opus 4.1 Conflicting-SWE 50% (5.1절)", 50, N_SWE, 0),
    ("GPT-5 abort 9% (5.3절)", 9, N_SWE, 0),
    ("o3 abort 12% (5.3절)", 12, N_SWE, 0),
]
a = []
for name, p, n, dg in claims:
    ks = feasible_k(p, n, dg)
    a.append({"claim": name, "n": n, "reported_pct": p, "feasible_k": ks[:6], "explained_by_k_over_n": bool(ks)})
out["A_arith_consistency"] = a

# 54.0%가 되려면 분모가 얼마여야 하는가
alt = {}
for n in range(340, 356):
    ks = feasible_k(54.0, n, 1)
    if ks:
        alt[n] = ks
out["A2_denominators_that_give_54.0pct"] = alt
out["A2_note"] = "349에서는 188/349=53.9%, 189/349=54.2%라 54.0%가 나오지 않는다. 348(188/348=54.0%)이나 350(189/350)이면 나온다."

# ---- B. 프롬프트·접근·abort 효과 (독립 Fisher)
B = []
def add(name, k1, n1, k2, n2):
    r = {"comparison": name, "p1_pct": round(100 * k1 / n1, 1), "p2_pct": round(100 * k2 / n2, 1),
         "diff_pp": round(100 * (k2 / n2 - k1 / n1), 1), "fisher_p": fisher(k1, n1, k2, n2),
         "ci90_1": wilson(k1, n1), "ci90_2": wilson(k2, n2)}
    B.append(r)

K = lambda pct, n: round(pct * n / 100)
add("SWE Conflicting GPT-5: 엄격 프롬프트 54% -> 느슨한 66% (5.1절)", K(54, N_SWE), N_SWE, K(66, N_SWE), N_SWE)
add("SWE Conflicting Opus 4.1: 엄격 50% -> 느슨한 55% (5.1절)", K(50, N_SWE), N_SWE, K(55, N_SWE), N_SWE)
add("SWE Conflicting GPT-5: 전체 접근 54% -> 읽기 전용 39% (그림 7)", K(54, N_SWE), N_SWE, K(39, N_SWE), N_SWE)
add("SWE Conflicting GPT-5: 전체 접근 54% -> abort 허용 9% (그림 8)", K(54, N_SWE), N_SWE, K(9, N_SWE), N_SWE)
add("SWE Conflicting o3: 49% -> abort 허용 12% (5.3절)", K(49, N_SWE), N_SWE, K(12, N_SWE), N_SWE)
add("SWE Conflicting Opus 4.1: 50% -> abort 허용 46% (그림 8)", K(50, N_SWE), N_SWE, K(46, N_SWE), N_SWE)
add("SWE Oneoff Opus 4.1: 54% -> abort 허용 51% (그림 8)", K(54, N_SWE), N_SWE, K(51, N_SWE), N_SWE)
add("SWE Original GPT-5: 97% -> abort 허용 97% (그림 8, 정상 성능 보존)", K(97, N_SWE), N_SWE, K(97, N_SWE), N_SWE)
add("SWE Original o4-mini: 85% -> abort 허용 84% (그림 8)", K(85, N_SWE), N_SWE, K(84, N_SWE), N_SWE)
add("SWE Original GPT-5: 97% -> 테스트 숨김 76% (그림 7, 정상 성능 손실)", K(97, N_SWE), N_SWE, K(76, N_SWE), N_SWE)
# LCB 프롬프트 (그림 6, n=103)
add("LCB Conflicting GPT-5: 프롬프트 A 92% -> D 1%", K(92, N_LCB), N_LCB, K(1, N_LCB), N_LCB)
add("LCB Conflicting GPT-5: 프롬프트 A 92% -> C 28%", K(92, N_LCB), N_LCB, K(28, N_LCB), N_LCB)
add("LCB Conflicting o3: 프롬프트 A 91% -> C 93% (C는 o3에 효과 없음)", K(91, N_LCB), N_LCB, K(93, N_LCB), N_LCB)
add("LCB Conflicting o3: 프롬프트 A 91% -> D 33%", K(91, N_LCB), N_LCB, K(33, N_LCB), N_LCB)
add("LCB Conflicting Opus 4.1: 프롬프트 A 37% -> D 0%", K(37, N_LCB), N_LCB, K(0, N_LCB), N_LCB)
out["B_effect_tests"] = B

# ---- C. 모델 간 차이 (SWE Conflicting 전체 스캐폴드, 그림 3 읽은 값)
read_swe = {  # 원본 통과율, oneoff, conflicting (그림 3 읽은 값, 약 ±2%p)
    "GPT-5": (97.5, 76, 54), "Sonnet-3.7": (88.5, 70, 64.5), "o4-mini": (85, 53, 48.5), "Opus-4.1": (86, 54, 50),
    "Sonnet-4": (89, 47.5, 43), "o3": (86, 39, 29), "Qwen3-Coder": (70.5, 21, 14), "GPT-4.1": (59, 14.5, 3),
}
C = []
def cmp_models(a, b, col):
    pa, pb = read_swe[a][col], read_swe[b][col]
    C.append({"cmp": f"{a} vs {b} ({'oneoff' if col == 1 else 'conflicting'})", "p_a": pa, "p_b": pb,
              "fisher_p": fisher(K(pa, N_SWE), N_SWE, K(pb, N_SWE), N_SWE)})
for a_, b_ in [("GPT-5", "Opus-4.1"), ("GPT-5", "Sonnet-3.7"), ("Opus-4.1", "Sonnet-3.7"), ("Sonnet-4", "Sonnet-3.7"),
               ("GPT-5", "o4-mini"), ("Opus-4.1", "o4-mini"), ("Opus-4.1", "Sonnet-4")]:
    cmp_models(a_, b_, 2)
out["C_model_pairs_swe_conflicting_read_off"] = C

# 스피어만: 원본 통과율(능력) 대 부정행위율
names = list(read_swe)
orig = [read_swe[n][0] for n in names]
res = {}
for col, lab in [(1, "oneoff"), (2, "conflicting")]:
    y = [read_swe[n][col] for n in names]
    rho, p = stats.spearmanr(orig, y)
    res[f"SWE_{lab}"] = {"rho": round(float(rho), 3), "p": round(float(p), 4), "n_models": len(names)}
    # GPT-4.1, Qwen3-Coder(약한 두 모델)를 빼면?
    keep = [i for i, n in enumerate(names) if n not in ("GPT-4.1", "Qwen3-Coder")]
    rho2, p2 = stats.spearmanr([orig[i] for i in keep], [y[i] for i in keep])
    res[f"SWE_{lab}_without_two_weakest"] = {"rho": round(float(rho2), 3), "p": round(float(p2), 4), "n_models": len(keep)}
# LCB 최소 스캐폴드 (그림 4 읽은 값: 원본, oneoff, conflicting)
read_lcb = {
    "GPT-5": (89, 3, 0.5), "o3": (86, 22, 33), "o4-mini": (82.5, 0, 0), "Opus-4.1": (76, 4, 0),
    "Sonnet-4": (61, 4, 0), "Sonnet-3.7": (54, 10, 0), "Qwen3-Coder": (56, 4, 0), "GPT-4.1": (62, 2, 0.5),
}
ln = list(read_lcb)
lo = [read_lcb[n][0] for n in ln]
for col, lab in [(1, "oneoff"), (2, "conflicting")]:
    y = [read_lcb[n][col] for n in ln]
    rho, p = stats.spearmanr(lo, y)
    res[f"LCB_{lab}"] = {"rho": round(float(rho), 3), "p": round(float(p), 4), "n_models": len(ln)}
    keep = [i for i, n in enumerate(ln) if n != "o3"]
    rho2, p2 = stats.spearmanr([lo[i] for i in keep], [y[i] for i in keep])
    res[f"LCB_{lab}_without_o3"] = {"rho": round(float(rho2), 3), "p": round(float(p2), 4), "n_models": len(keep)}
out["D_capability_vs_cheating_spearman"] = res
out["D_note"] = "그림에서 읽은 값(±2%p)이라 순위 상관만 의미가 있다. 모델 8개라 검정력이 낮다. 능력(원본 통과율)과 '부정행위를 시도해 성공하는 능력'이 섞여 있어 성향의 증거로는 약하다."

# ---- E. LCB 0에 가까운 셀의 신뢰구간
E = {}
for k in [0, 1, 3]:
    E[f"{k}/103"] = {"pct": round(100 * k / 103, 2), "wilson90": wilson(k, 103), "95pct_upper_clopper": round(100 * float(stats.beta.ppf(0.975, k + 1, 103 - k)), 1)}
out["E_lcb_small_cells"] = E
out["E_note"] = "GPT-5 oneoff-LCB 2.9%(3/103)와 0%인 모델들은 구분되지 않는다(90% Wilson 구간이 겹침)."
# o3 대 GPT-5 (LCB oneoff)
out["E_o3_vs_gpt5_lcb_oneoff"] = {"o3_pct": 22, "gpt5_pct": 2.9, "fisher_p": fisher(K(22, N_LCB), N_LCB, 3, N_LCB)}
out["E_o3_vs_gpt5_lcb_conflicting"] = {"o3_pct": 33, "gpt5_pct": 1, "fisher_p": fisher(K(33, N_LCB), N_LCB, 1, N_LCB)}

# ---- F. 피드백 효과 평균 (그림 17 읽은 값) - 본문 "33% -> 38%" 재계산
fb = {  # 첫 제출만, 여러 번 제출 (Conflicting-SWE, 그림 17 왼쪽, 읽은 값)
    "GPT-5": (33, 54), "Sonnet-3.7": (61, 64), "o4-mini": (43, 49), "Opus-4.1": (47, 50),
    "Sonnet-4": (37, 43), "o3": (28, 29), "Qwen3-Coder": (14, 14), "GPT-4.1": (0, 3),
}
first = np.mean([v[0] for v in fb.values()])
multi = np.mean([v[1] for v in fb.values()])
nog = [v for k, v in fb.items() if k != "GPT-5"]
out["F_feedback_average"] = {
    "mean_first_pct": round(float(first), 1), "mean_multi_pct": round(float(multi), 1),
    "paper_text": "33% -> 38%",
    "without_GPT5": {"first": round(float(np.mean([v[0] for v in nog])), 1), "multi": round(float(np.mean([v[1] for v in nog])), 1)},
    "gpt5_delta_pp": 21,
    "gpt5_fisher_p": fisher(K(33, N_SWE), N_SWE, K(54, N_SWE), N_SWE),
    "per_model_fisher_p": {m: fisher(K(v[0], N_SWE), N_SWE, K(v[1], N_SWE), N_SWE) for m, v in fb.items()},
}

# ---- G. 모니터 (그림 10, n 라벨은 정확)
mon = [
    ("LCB Sonnet 모니터 oneoff", 86, 113), ("LCB Sonnet 모니터 conflicting", 89, 80),
    ("SWE Sonnet 모니터 oneoff", 42, 1308), ("SWE Sonnet 모니터 conflicting", 50, 1063),
    ("SWE Opus 모니터 oneoff", 57, 1308), ("SWE Opus 모니터 conflicting", 65, 1063),
    ("LCB original(오탐 상한)", 7, 550), ("SWE Sonnet original(오탐 상한)", 1, 2300), ("SWE Opus original(오탐 상한)", 2, 2300),
]
G = []
for name, pct, n in mon:
    k = round(pct * n / 100)
    G.append({"cell": name, "pct": pct, "n": n, "k_approx": k, "wilson90": wilson(k, n)})
out["G_monitor_cells"] = G
out["G_tests"] = [
    {"cmp": "LCB oneoff 86% vs conflicting 89% (Sonnet)", "fisher_p": fisher(round(.86 * 113), 113, round(.89 * 80), 80)},
    {"cmp": "SWE Sonnet oneoff 42% vs conflicting 50%", "fisher_p": fisher(round(.42 * 1308), 1308, round(.50 * 1063), 1063)},
    {"cmp": "SWE Sonnet 42% vs Opus 57% (oneoff, 같은 전사본 짝 -> 독립 검정은 보수적)", "fisher_p": fisher(round(.42 * 1308), 1308, round(.57 * 1308), 1308)},
]
out["G_counts"] = {"swe_cheat_transcripts": 1308 + 1063, "paper_says": 2371, "lcb_cheat_transcripts": 113 + 80, "paper_says_lcb": 193}
# 전사본이 모델·문항·스캐폴드에 걸쳐 군집돼 있어 독립 가정은 구간을 좁게 만든다. 문항(349) 단위 군집으로 보면 유효 표본이 더 작다.
out["G_note"] = "모니터 전사본은 (모델 8 x 스캐폴드 x 문항 349)에서 통과한 것만 모은 것이라 같은 문항·모델의 전사본이 군집한다. 위 구간은 독립 가정이라 좁다."

# ---- H. 그림 5: 유형 분류 n 합과 모델별 비율 검정
n5 = {"GPT-5": 373, "o3": 151, "o4-mini": 225, "GPT-4.1": 13, "Opus-4.1": 253, "Sonnet-4": 169, "Sonnet-3.7": 341, "Qwen3-Coder": 85}
share_A = {"GPT-5": 24, "o3": 15, "o4-mini": 64, "GPT-4.1": 46, "Opus-4.1": 86, "Sonnet-4": 82, "Sonnet-3.7": 79, "Qwen3-Coder": 87}
H = {"sum_n": sum(n5.values()), "note": "스캐폴드 2개 합산(문항 349 x 2 = 698회 시도 중 통과 수)"}
H["pass_rate_implied_over_698_runs_pct"] = {m: round(100 * n / 698, 1) for m, n in n5.items()}
# 오픈AI 모델이 '다양'한가: A(테스트 수정) 비율
H["share_modify_tests_pct"] = share_A
H["openai_modify_share_range"] = [min(share_A[m] for m in ["GPT-5", "o3", "o4-mini", "GPT-4.1"]), max(share_A[m] for m in ["GPT-5", "o3", "o4-mini", "GPT-4.1"])]
H["note2"] = "캡션 'OpenAI 모델이 더 다양'은 GPT-5(24%)·o3(15%)에만 해당한다. o4-mini는 64%가 테스트 수정이다."
H["fisher_gpt5_vs_opus_modify"] = fisher(round(.24 * 373), 373, round(.86 * 253), 253)
out["H_fig5"] = H

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=1))
