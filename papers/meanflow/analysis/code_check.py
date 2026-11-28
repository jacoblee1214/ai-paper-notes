"""공식 JAX 코드(meanflow.py)의 마스크 논리를 numpy로 옮겨 논문 서술과의 차이를 수치로 확인한다. 코드는 실행하지 않는다."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path(__file__).with_name("code_check_results.json")
rng = np.random.default_rng(0)
bz, p_data, p_drop = 256, 0.75, 0.1
data_size = int(bz * p_data)
n_batches = 20000
drop_and_neq, ncls_neq, ncls, ndrop = 0, 0, 0, []
for _ in range(n_batches):
    num_drop = int((rng.random(bz) < p_drop).sum())
    idx = np.arange(bz)
    r_eq_t = idx < data_size          # sample_tr: 앞 75%가 r=t
    dropped = idx < num_drop          # cond_drop: 앞 num_drop 개가 라벨 제거
    drop_and_neq += int((dropped & ~r_eq_t).sum())
    cond = ~dropped
    ncls += int(cond.sum())
    ncls_neq += int((cond & ~r_eq_t).sum())
    ndrop.append(num_drop)
res = {
    "batch": bz, "n_batches": n_batches, "seed": 0,
    "라벨 제거 표본 중 r!=t 인 표본 수 (전체 합)": drop_and_neq,
    "P[num_drop > data_size] (이항 꼬리)": float(stats.binom.sf(data_size, bz, p_drop)),
    "평균 num_drop": float(np.mean(ndrop)),
    "클래스 조건 표본 중 r!=t 비율 (논문 서술 25%)": ncls_neq / ncls,
    "이론값 64/(256-25.6)": 64 / (256 - 25.6),
}
# 독립 베르누이 드롭(논문 서술)이면 라벨 제거 표본 중 r!=t 비율은 25%여야 한다
res["독립 드롭 가정의 라벨 제거 표본 중 r!=t 비율"] = 1 - p_data
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
