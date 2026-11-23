"""τ-bench 과제 파일을 정적으로(ast) 읽어 ABC 논문의 do-nothing(38%/6.0%)과 spamming(40%/9.6%) 수치를 재계산.
- 코드를 import·실행하지 않는다(litellm 등 불필요). 과제 정의(Task(actions=..., outputs=...))와 DB json만 읽는다.
- do-nothing이 통과하는 과제 = 정답 액션에 DB를 바꾸는(쓰기) 도구가 없고 outputs가 비어 있음(base.py calculate_reward 논리).
- spamming이 통과하는 과제 = 쓰기 도구가 없고, 모든 outputs 문자열이 DB 텍스트에 그대로 있음(응답에서 쉼표 제거·소문자 비교).
입력: data/tau-bench (커밋 14bf0ef, ABC README가 고정한 판). 출력: recompute_tau_bench_results.json
실행: python3 -I analysis/recompute_tau_bench.py
"""
import ast, json, pathlib, math
ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV = ROOT / 'data/tau-bench/tau_bench/envs'
WRITE = {'airline': {'book_reservation', 'cancel_reservation', 'send_certificate', 'update_reservation_baggages',
                     'update_reservation_flights', 'update_reservation_passengers'},
         'retail': {'cancel_pending_order', 'exchange_delivered_order_items', 'modify_pending_order_address',
                    'modify_pending_order_items', 'modify_pending_order_payment', 'modify_user_address',
                    'return_delivered_order_items'}}
TERMINATE = {'transfer_to_human_agents'}

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

def load_tasks(path):
    tree = ast.parse(path.read_text())
    tasks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'Task':
            kw = {k.arg: k.value for k in node.keywords}
            acts = [a for a in ast.walk(kw['actions']) if isinstance(a, ast.Call) and getattr(a.func, 'id', '') == 'Action']
            names = []
            for a in acts:
                for k in a.keywords:
                    if k.arg == 'name': names.append(ast.literal_eval(k.value))
            outs = ast.literal_eval(kw['outputs'])
            tasks.append({'names': names, 'outputs': outs})
    return tasks

def db_text(domain):
    d = ENV / domain / 'data'
    txt = ' '.join(p.read_text() for p in sorted(d.glob('*.json')))
    return txt.lower().replace(',', '')

res = {}
for domain, files in {'airline': ['tasks_test.py'], 'retail': ['tasks_test.py', 'tasks_train.py', 'tasks_dev.py']}.items():
    txt = db_text(domain)
    for f in files:
        ts = load_tasks(ENV / domain / f)
        n = len(ts)
        no_write = [t for t in ts if not (set(t['names']) & WRITE[domain])]
        nothing = [t for t in no_write if not t['outputs']]
        # 출력 문자열이 DB 텍스트에 그대로 있나(공백·쉼표 제거 후 소문자)
        spam = [t for t in no_write if t['outputs'] and all(o.lower() in txt for o in t['outputs'])]
        out_not_in_db = [t for t in no_write if t['outputs'] and not all(o.lower() in txt for o in t['outputs'])]
        k0, k1 = len(nothing), len(nothing) + len(spam)
        res[f'{domain}/{f}'] = {
            'n_tasks': n, 'no_write_actions': len(no_write),
            'do_nothing_pass': k0, 'do_nothing_pct': round(100 * k0 / n, 2), 'do_nothing_wilson95': wilson(k0, n),
            'no_write_with_outputs': len([t for t in no_write if t['outputs']]),
            'outputs_all_in_db': len(spam), 'outputs_not_in_db': len(out_not_in_db),
            'spam_pass_upper': k1, 'spam_pct_upper': round(100 * k1 / n, 2), 'spam_wilson95': wilson(k1, n),
            'spam_only_pct': round(100 * len(spam) / n, 2),
            'outputs_of_spam_tasks': sorted({o for t in spam for o in t['outputs']}),
        }
        print(domain, f, {k: v for k, v in res[f'{domain}/{f}'].items() if k != 'outputs_of_spam_tasks'})
        print('   spam outputs:', res[f'{domain}/{f}']['outputs_of_spam_tasks'][:20])
# 논문 값과 정수 일치 검사: 38/6.0/40/9.6/2/3.6 %가 어떤 n에서 정수인가
claims = {'airline_do_nothing': 38.0, 'retail_do_nothing': 6.0, 'airline_spam': 40.0, 'retail_spam': 9.6,
          'airline_db_text_gt': 2.0, 'retail_db_text_gt': 3.6}
fit = {}
for name, pct in claims.items():
    fit[name] = {}
    for n in (50, 115, 500, 20):
        k = round(pct * n / 100)
        fit[name][n] = {'k': k, 'k/n %': round(100 * k / n, 2), 'rounds_to_claim_1dp': round(100 * k / n, 1) == pct}
res['claims_integer_fit'] = fit
json.dump(res, open(ROOT / 'analysis/recompute_tau_bench_results.json', 'w'), ensure_ascii=False, indent=1)
for k, v in fit.items(): print(k, {n: (x['k'], x['k/n %'], x['rounds_to_claim_1dp']) for n, x in v.items()})
