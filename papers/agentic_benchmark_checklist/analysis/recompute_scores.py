"""ABC 평가표(논문 부록 D Table 5-14 = 저장소 assessments/*.yaml)에서 절별 평균 점수와 위반 집계를 재계산.
입력: .scratch/assess_raw.json (parse_assessments.py가 만든다). 출력: recompute_scores_results.json
실행: python3 -I analysis/parse_assessments.py && python3 -I analysis/recompute_scores.py
"""
import json, pathlib, collections, re, yaml
ROOT = pathlib.Path(__file__).resolve().parent.parent
raw = json.loads((ROOT/'.scratch/assess_raw.json').read_text())
sc = raw['repo']   # 표와 yaml이 일치(parse_assessments.py에서 확인)
# 항목 정의(논문 Figure 2-4 기준 번호)
ITEMS = {'O': ['a.1','a.2','b.1','b.2','b.3','c.1','c.2','d.1','d.2','e.1','e.2','e.3','f.1','f.2','g.1','g.2','g.3','h.1','h.2','I.1'],
         'T': [f'{i}' for i in range(1, 11)], 'R': [f'{i}' for i in range(1, 14)]}
print('항목 수 O/T/R/합:', {k: len(v) for k, v in ITEMS.items()}, sum(len(v) for v in ITEMS.values()))
out = {'item_counts': {k: len(v) for k, v in ITEMS.items()}, 'per_benchmark': {}}
tot = collections.Counter(); viol = collections.defaultdict(list)
for b, d in sc.items():
    rec = {}
    for sec in 'OTR':
        vals = {k: v for k, v in d.items() if k.startswith(sec + '.')}
        n = len(vals); s = sum(vals.values())
        rec[sec] = {'applicable': n, 'score_sum': s, 'avg_pct': round(100 * s / n, 1) if n else None,
                    'zeros': sorted(k for k, v in vals.items() if v == 0)}
        if rec[sec]['zeros']: viol[sec].append(b)
    out['per_benchmark'][b] = rec
    for sec in 'OTR': tot[sec] += rec[sec]['applicable']
out['benchmarks_with_any_zero'] = {k: {'n': len(v), 'list': v} for k, v in viol.items()}
out['applicable_items_total'] = dict(tot)
# 해당 없음(N/A)인데 1점으로 센 T 항목: yaml 이유 문구로 식별("does not use/require ...")
NA = re.compile(r"does not (use|require)|not required|no external", re.I)
name_map = {'SWE-Bench-Lancer':'swe-lancer','Bird-Bench':'bird-bench','CyBench':'cy-bench','SWE-Bench-Verified':'swe-bench','tau-Bench':'tau-bench',
            'MLE-Bench':'mle-bench','WebArena':'webarena','GAIA':'gaia','OSWorld':'osworld','KernelBench':'kernel-bench'}
out['T_na_adjusted'] = {}
for b, fn in name_map.items():
    y = yaml.safe_load((ROOT/f'repo/assessments/{fn}.yaml').read_text().split('\n\n',1)[1])
    na = [k for k, v in y.items() if k.startswith('T.') and v['score'] == 1 and NA.search(v['reason'])]
    t = out['per_benchmark'][b]['T']
    n_app = t['applicable'] - len(na); s_ = t['score_sum'] - len(na)
    out['T_na_adjusted'][b] = {'na_items': na, 'T_paper_pct': t['avg_pct'], 'T_adjusted_pct': round(100 * s_ / n_app, 1)}
# 가장 자주 0인 항목
z = collections.Counter(); app = collections.Counter()
for d in sc.values():
    for k, v in d.items():
        app[k] += 1; z[k] += (v == 0)
out['item_zero_counts'] = {k: [z[k], app[k]] for k in sorted(app, key=lambda x: (-z[x]/app[x], x)) if z[k]}
# 'R.13 = trivial agent' 충족 수
out['R13'] = {b: d.get('R.13') for b, d in sc.items()}
out['R10'] = {b: d.get('R.10') for b, d in sc.items()}
out['O_applicable_sets'] = {b: sorted(k for k in d if k.startswith('O.')) for b, d in sc.items()}
json.dump(out, open(ROOT/'analysis/recompute_scores_results.json', 'w'), ensure_ascii=False, indent=1)
for b, r in out['per_benchmark'].items():
    print(f"{b:20s} T {r['T']['avg_pct']:5} ({r['T']['zeros']})  O {r['O']['avg_pct']} ({r['O']['zeros']})  R {r['R']['avg_pct']}")
print({k: (v['n']) for k, v in out['benchmarks_with_any_zero'].items()})
print('R13 충족:', sum(1 for v in out['R13'].values() if v == 1), 'R10 충족:', sum(1 for v in out['R10'].values() if v == 1))
print(out['item_zero_counts'])
for b, r in out['T_na_adjusted'].items(): print('T 보정', b, r)
