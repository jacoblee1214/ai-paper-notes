"""논문 §5.2: 20회 실행 정확도 분포가 정규가 아니다(KS 검정 p<1e-9)는 주장을 v2 공개 20회 데이터로 다시 검정한다.

입력: data/v2/release_data/20_run/{gpt-4o,llama-3-70b}_mmlu_college_mathematics_{0..19}.csv (열 new_extracted_pred, gt)
검정: (a) 표본 평균·표준편차로 표준화한 KS(Lilliefors 보정 없음, p는 보수적이지 않음), (b) 표본 평균·표준편차를 쓴 몬테카를로 Lilliefors,
      (c) Shapiro-Wilk, (d) 표준화하지 않고 N(0,1)과 비교한 KS(p가 아주 작게 나오는 경우를 확인하려는 대조).
실행: python3 -I analysis/normality_check.py     출력: analysis/normality_results.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "v2" / "release_data" / "20_run"
out = {}
rng = np.random.default_rng(0)
for m in ["gpt-4o", "llama-3-70b"]:
    acc = []
    for i in range(20):
        df = pd.read_csv(D / f"{m}_mmlu_college_mathematics_{i}.csv")
        acc.append(100 * (df["new_extracted_pred"] == df["gt"]).mean())
    a = np.array(acc)
    z = (a - a.mean()) / a.std(ddof=1)
    ks_std = stats.kstest(z, "norm")
    # Lilliefors 몬테카를로
    D0 = ks_std.statistic
    sims = []
    for _ in range(5000):
        x = rng.normal(size=20)
        sims.append(stats.kstest((x - x.mean()) / x.std(ddof=1), "norm").statistic)
    p_lillie = float((1 + (np.array(sims) >= D0).sum()) / 5001)
    out[m] = dict(acc=a.round(1).tolist(), mean=float(a.mean()), median=float(np.median(a)), sd=float(a.std(ddof=1)),
                  min=float(a.min()), max=float(a.max()),
                  ks_standardized_p_naive=float(ks_std.pvalue), lilliefors_mc_p=p_lillie,
                  shapiro_p=float(stats.shapiro(a).pvalue),
                  ks_vs_N01_unstandardized_p=float(stats.kstest(a, "norm").pvalue))
Path(ROOT / "analysis" / "normality_results.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
