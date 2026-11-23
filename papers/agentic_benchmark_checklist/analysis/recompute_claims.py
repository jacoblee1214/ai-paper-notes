"""ABC 논문의 정량 주장(과대·과소 추정, 부록 F의 BIRD 신뢰구간 표)을 문항 수와 함께 재계산.
- 상대 오차와 절대 오차(%p)를 나누고, 표본 수가 정해진 경우 Wilson 95% 구간을 붙인다.
- 표 15(BIRD 신뢰구간): 논문 식으로 구간을 다시 만들고, 점추정이 구간 안에 드는지, 오차 보정의 올바른 방향이 어느 쪽인지 검사.
입력: paper/paper.txt, repo/benchmarks/webarena/config_files. 출력: recompute_claims_results.json
실행: python3 -I analysis/recompute_claims.py
"""
import json, math, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parent.parent
out = {}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

# --- 1. OSWorld chrome (README 정량 분석 + 논문 5.2) ---
n_chrome, before, after, n_all = 46, 24, 37, 369
out['osworld'] = {
    'chrome_broken_fraction_pct': round(100 * 13 / n_chrome, 2),
    'chrome_before_pct': round(100 * before / n_chrome, 1), 'chrome_after_pct': round(100 * after / n_chrome, 1),
    'abs_gain_pp_chrome': round(100 * (after - before) / n_chrome, 2),
    'rel_gain_vs_before_pct': round(100 * (after - before) / before, 1),
    'overall_gain_pp_if_369': round(100 * (after - before) / n_all, 2),
    'overall_before_pct_claim': 42.5, 'overall_after_pct_claim': 46.0,
    'rel_error_claim_7_6_pct_check': {'gain/after': round(100 * 3.5 / 46.0, 2), 'gain/before': round(100 * 3.5 / 42.5, 2)},
    'wilson_24_of_46': wilson(24, 46), 'wilson_37_of_46': wilson(37, 46),
    'note': '13개 모두를 UI-TARS가 맞게 푼 것으로 수작업 판정해 24→37. 판정 대상은 한 에이전트의 실패 궤적이라 거짓 양성(낡은 선택자가 우연히 통과)은 찾지 않았다.'}

# --- 2. WebArena: 빈 응답이 통과하는 N/A 과제 수 ---
na = 0; n = 0
for i in range(812):
    d = json.load(open(ROOT / f'repo/benchmarks/webarena/config_files/{i}.json'))
    n += 1
    ra = d['eval'].get('reference_answers') or {}
    if 'N/A' in json.dumps(ra): na += 1
out['webarena'] = {'n_tasks': n, 'na_reference_tasks': na, 'pct': round(100 * na / n, 2), 'wilson95_if_sample': wilson(na, n),
                   'claim_pct': 4.4, 'note': '36/812 = 4.43%. 이 과제 집합이 전부라 표본 오차가 아니라 정확한 비율. 1.4–5.2% 과대 추정은 궤적 데이터(gdown)가 필요해 재계산 못함.'}

# --- 3. SWE-bench (논문 E.1과 관련 연구 절의 숫자) ---
out['swe_bench'] = {
    'verified_n500': {'5.2%': round(500 * 0.052, 2), '5.3%': round(500 * 0.053, 2), 'k=26 -> %': round(100 * 26 / 500, 2), 'wilson_26_of_500': wilson(26, 500)},
    'lite_n300': {'7.7%': round(300 * 0.077, 2), 'k=23 -> %': round(100 * 23 / 300, 2), 'wilson_23_of_300': wilson(23, 300)},
    'note': '관련 연구 절은 Verified 5.2%, E.1은 5.3%. 500문항에서 5.3%는 정수 개수에 맞지 않는다(26.5). UTBoost 초록: 리더보드 변화 Lite 40.9%, Verified 24.4%. E.1은 Verified 40.9, Lite 24.4로 뒤바뀌어 적었고 서론의 "상위 50위 중 24%"(Verified)와도 어긋난다.'}

# --- 4. τ-bench 트리비얼 에이전트 vs GPT-4o (서론: 38% > GPT-4o 35%) ---
a, b = 19, 50  # do-nothing airline (recompute_tau_bench.py에서 정확히 일치)
out['tau_vs_gpt4o'] = {'do_nothing': wilson(a, b), 'gpt4o_35pct_n50': wilson(round(0.35 * 50), 50),
                       'gap_pp': 38.0 - 35.0,
                       'note': 'GPT-4o 35%(논문 서론)는 같은 50문항의 여러 시행 평균. 두 구간이 크게 겹치고 3%p 차이는 문항 50개 표본 오차(±13%p) 안. "트리비얼 에이전트가 GPT-4o를 앞선다"는 점추정 순위이지 통계적 구분이 아니다. 단 do-nothing 38%는 고정 과제 집합의 결정적 계산이라 그 자체에는 표본 오차가 없다.'}

# --- 5. 부록 F Table 15: BIRD 신뢰구간 ---
rows = []
txt = (ROOT / 'paper/paper.txt').read_text().split('\n')
seg = txt[1984:2060]
pat = re.compile(r'^\s*(.+?)\s{2,}(\d+\.\d)\s+\[(\d+\.\d),\s*(\d+\.\d)\]\s+(\d+)\s+(\d+)-(\d+)\s*$')
for ln in seg:
    m = pat.match(ln)
    if m: rows.append({'method': m.group(1).strip(), 'acc': float(m.group(2)), 'lo': float(m.group(3)), 'hi': float(m.group(4)),
                       'rank': int(m.group(5)), 'r_lo': int(m.group(6)), 'r_hi': int(m.group(7))})
e = 0.1165
def paper_ci(p0, N):
    p0 /= 100
    mu = e + (1 - 2 * e) * p0
    se = math.sqrt(mu * (1 - mu) / N)
    return 100 * (mu - 1.96 * se), 100 * (mu + 1.96 * se), 100 * mu
best = None
for N in range(1000, 2200):
    err = max(max(abs(paper_ci(r['acc'], N)[0] - r['lo']), abs(paper_ci(r['acc'], N)[1] - r['hi'])) for r in rows)
    if best is None or err < best[1]: best = (N, err)
N = best[0]
inside = sum(1 for r in rows if r['lo'] <= r['acc'] <= r['hi'])
# 올바른 방향(관측 a = e + (1-2e) p 이면 p = (a-e)/(1-2e)): 점추정의 보정
corrected = [round(100 * (r['acc'] / 100 - e) / (1 - 2 * e), 1) for r in rows]
# 가능 순위 범위: 이 구간의 겹침으로 재현되나?
def rank_range(i):
    lo_i, hi_i = rows[i]['lo'], rows[i]['hi']
    better = sum(1 for j, r in enumerate(rows) if j != i and r['lo'] > hi_i)
    worse = sum(1 for j, r in enumerate(rows) if j != i and r['hi'] < lo_i)
    return better + 1, len(rows) - worse
rr_match = sum(1 for i, r in enumerate(rows) if rank_range(i) == (r['r_lo'], r['r_hi']))
out['bird_table15'] = {'n_rows': len(rows), 'fit_N': N, 'max_abs_err_pp_fit': round(best[1], 3),
    'acc_inside_ci': inside, 'acc_above_ci': sum(1 for r in rows if r['acc'] > r['hi']), 'acc_below_ci': sum(1 for r in rows if r['acc'] < r['lo']),
    'first_rows': rows[:3], 'last_rows': rows[-3:],
    'corrected_point_example': {'method': rows[0]['method'], 'reported': rows[0]['acc'], 'inverse_map_p': corrected[0]},
    'rank_range_rule_overlap_matches': rr_match,
    'note': '논문 식 mu=e+(1-2e)p0은 "참 정확도 p가 주어졌을 때 관측 기대값"의 방향이다. 이것을 관측값 p0에 적용했으므로 구간 중심이 점추정과 어긋난다(고득점은 아래로, 저득점은 위로 끌림). 구간이 점추정을 포함하지 않는 행이 대부분이다.'}
json.dump(out, open(ROOT / 'analysis/recompute_claims_results.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
