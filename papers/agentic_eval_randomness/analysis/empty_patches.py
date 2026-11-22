"""실행별 빈 패치 수(에이전트가 패치를 내지 못한 문항)와 해결 수의 관계. 실행 간 분산에 얼마나 기여하는지 본다.
입력: data/*/*/*.json 의 empty_patch_instances, resolved_instances, error_instances.  실행: python3 -I analysis/empty_patches.py
출력: empty_patches_results.json
"""
import glob, json, os
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), "data")
out = {}
for d in sorted(os.listdir(DATA)):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{DATA}/{d}/*/*.json"))]
    res = np.array([r["resolved_instances"] for r in rows]); emp = np.array([r["empty_patch_instances"] for r in rows])
    err = np.array([r["error_instances"] for r in rows])
    cond = res / (500 - emp) * 100  # 패치를 낸 문항 중 해결 비율(%)
    rho = stats.spearmanr(res, emp)
    out[d] = {"empty_mean": emp.mean(), "empty_sd": emp.std(ddof=1), "empty_min": int(emp.min()), "empty_max": int(emp.max()),
              "resolved_sd": res.std(ddof=1), "resolved_rate_given_patch_pct": float(cond.mean()), "resolved_rate_pct": float(res.mean() / 5), "error_max": int(err.max()), "spearman_resolved_vs_empty": float(rho.statistic), "p": float(rho.pvalue)}
    print(f"{d:50s} 빈패치 평균 {emp.mean():6.1f} sd {emp.std(ddof=1):5.1f} [{emp.min()},{emp.max()}]  해결 sd {res.std(ddof=1):5.1f}  rho={rho.statistic:+.2f} p={rho.pvalue:.2f} err_max={err.max()}  해결률 {res.mean()/5:5.1f}% 패치있는문항중 {cond.mean():5.1f}%")
json.dump(out, open(os.path.join(HERE, "empty_patches_results.json"), "w"), indent=1, default=float)
