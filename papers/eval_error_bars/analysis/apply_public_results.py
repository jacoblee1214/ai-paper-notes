"""공개된 문항별 결과 파일에 minimal_impl의 식을 적용한다(읽기만 한 파일의 작은 사본 사용, 수 초).

  A. SWE-bench 공개 결과(SWE-bench/experiments, BM25 기준선 세 모델, 전체 2,294문항과 Lite 300문항):
     비짝 SE, 짝지은 SE, 저장소 군집 짝지은 SE (식 (2), (4), (7), (8)).
  B. On Randomness in Agentic Evals 공개 결과(SWE-bench Verified 500문항, 설정 12개, 설정당 10회):
     1회 대 10회 평균의 SE, 분산 분해(3.1절), 짝지은 비교, 식 (10)의 최소 검출 효과.

실행: python3 -I analysis/apply_public_results.py   (data/ 필요: README 재실행 절 참고)
결과: analysis/apply_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimal_impl as m  # noqa: E402

DATA = ROOT / "data"


def repo_of(iid: str) -> str:
    return iid.split("__")[0] + "/" + iid.split("__")[1].rsplit("-", 1)[0]


# ------------------------------------------------------------------ A. SWE-bench
SWE_MODELS = {"Claude 3 Opus": "20240402_rag_claude3opus", "Claude 2": "20231010_rag_claude2",
              "GPT-4-turbo": "20240402_rag_gpt4"}


def part_a():
    out = {}
    for split in ("test", "lite"):
        res = {}
        for name, d in SWE_MODELS.items():
            j = json.load(open(DATA / "swe_bench" / split / d / "results.json"))
            res[name] = j
        ids = sorted(set(res["Claude 2"]["generated"]) | set(res["Claude 2"]["no_generation"]))
        idx = {i: k for k, i in enumerate(ids)}
        score = {}
        for name, j in res.items():
            s = np.zeros(len(ids))
            for i in j["resolved"]:
                s[idx[i]] = 1
            score[name] = s
        cl = np.array([repo_of(i) for i in ids])
        rows = {"n": len(ids), "n_repos": int(len(set(cl))), "single": {}, "pairs": {}}
        for name, s in score.items():
            rows["single"][name] = {"mean": s.mean(), "se_bernoulli": m.se_bernoulli(s), "se_clt": m.se_clt(s),
                                    "se_cluster_repo": m.se_clustered(s, cl),
                                    "se_stratified_repo": m.se_stratified(s, cl),
                                    "deff_repo": m.design_effect(s, cl)}
        names = list(score)
        for a in range(3):
            for b in range(a + 1, 3):
                A, B = score[names[a]], score[names[b]]
                c = m.compare(A, B, cl)
                c["se_paired_cluster_cr1"] = float(np.sqrt(len(set(cl)) / (len(set(cl)) - 1)) * c["se_paired_clustered"])
                c["n_A_only"] = int(((A == 1) & (B == 0)).sum())
                c["n_B_only"] = int(((A == 0) & (B == 1)).sum())
                c["ratio_unpaired_over_paired"] = c["se_unpaired"] / c["se_paired"]
                c["ratio_cluster_over_paired"] = c["se_paired_clustered"] / c["se_paired"]
                rows["pairs"][f"{names[a]} - {names[b]}"] = c
        out[split] = rows
    return out


# ------------------------------------------------------------------ B. 반복 실행
def load_config(d: Path):
    runs = sorted(p for p in d.iterdir() if p.is_dir())   # run_N/ 또는 traj_*_run_N/ 안에 JSON 하나
    sets, ids = [], set()
    for r in runs:
        j = json.load(open(next(r.glob("*.json"))))
        sets.append(set(j["resolved_ids"]))
        ids |= set(j["submitted_ids"])
    ids = sorted(ids)
    sc = np.zeros((len(ids), len(runs)))
    for k, s in enumerate(sets):
        for i, q in enumerate(ids):
            sc[i, k] = q in s
    return ids, sc


def part_b():
    configs = {p.name: load_config(p) for p in sorted((DATA / "agentic").iterdir())}
    ids = next(iter(configs.values()))[0]
    cl = np.array([repo_of(i) for i in ids])
    out = {"n": len(ids), "n_repos": int(len(set(cl))), "single": {}, "pairs": {}}
    for name, (i2, sc) in configs.items():
        assert i2 == ids
        k = sc.shape[1]
        vx, es = m.variance_components(sc)
        xbar = sc.mean(1)
        se1 = [m.se_clt(sc[:, r]) for r in range(k)]
        out["single"][name] = {
            "mean": float(sc.mean()), "runs": k,
            "se_one_run_mean": float(np.mean(se1)), "se_k10": m.se_clt(xbar),
            "se_k10_cluster_repo": m.se_clustered(xbar, cl),
            "var_x": vx, "e_sigma2": es, "var_ratio_k10_vs_k1": m.variance_ratio_vs_k1(vx, es, k),
            "sd_of_run_means": float(sc.mean(0).std(ddof=1)),
            "sd_pred_from_decomposition": float(np.sqrt(es / len(ids)))}
    pairs = [("nano-agent-Qwen_Qwen3-32B", "nano-agent-Qwen_Qwen3-32B-temp0"),
             ("nano-agent-mistral_devstral-2512", "nano-agent-mistral_devstral-2512__temp0"),
             ("r2e-gym-agentica-org__DeepSWE-preview", "r2e-gym-agentica-org__DeepSWE-preview__temp0"),
             ("r2e-gym-mistral_devstral-2512", "nano-agent-mistral_devstral-2512"),
             ("r2e-gym-Qwen_Qwen3-32B", "nano-agent-Qwen_Qwen3-32B"),
             ("r2e-gym-mistral_devstral-2512", "r2e-gym-Qwen_Qwen3-32B")]
    for a, b in pairs:
        A, B = configs[a][1], configs[b][1]
        k = A.shape[1]
        n = len(ids)
        xa, xb = A.mean(1), B.mean(1)
        # 1회씩(run r vs run r)의 평균 SE와, 10회 평균 위의 SE
        se_p1 = float(np.mean([m.se_paired(A[:, r], B[:, r]) for r in range(k)]))
        se_u1 = float(np.mean([m.se_unpaired(A[:, r], B[:, r]) for r in range(k)]))
        o2 = m.omega2_paired(A, B)
        sa, sb = m.variance_components(A)[1], m.variance_components(B)[1]
        row = {"diff": float(xa.mean() - xb.mean()),
               "se_unpaired_k1": se_u1, "se_paired_k1": se_p1,
               "se_unpaired_k10": m.se_unpaired(xa, xb), "se_paired_k10": m.se_paired(xa, xb),
               "se_paired_k10_cluster_repo": m.se_paired_clustered(xa, xb, cl),
               "corr_k10": float(np.corrcoef(xa, xb)[0, 1]),
               "omega2": o2, "sigma2_A": sa, "sigma2_B": sb,
               "mde_k1": m.mde(n, o2, sa, sb, 1, 1), "mde_k10": m.mde(n, o2, sa, sb, 10, 10),
               "mde_k_inf": m.mde(n, o2),
               "n_for_2pp_k1": m.n_required(0.02, o2, sa, sb, 1, 1),
               "n_for_2pp_k10": m.n_required(0.02, o2, sa, sb, 10, 10)}
        out["pairs"][f"{a} - {b}"] = row
    return out


if __name__ == "__main__":
    res = {"A_swe_bench_rag": part_a(), "B_agentic_runs": part_b()}

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    (ROOT / "analysis" / "apply_results.json").write_text(json.dumps(res, indent=1, default=conv, ensure_ascii=False))
    a = res["A_swe_bench_rag"]
    for split, r in a.items():
        print(f"[A {split}] n={r['n']} repos={r['n_repos']}")
        for k, v in r["single"].items():
            print(f"  {k:14s} {100*v['mean']:.2f}%  SE bern {100*v['se_bernoulli']:.3f} clt {100*v['se_clt']:.3f} repo-cluster {100*v['se_cluster_repo']:.3f} strat {100*v['se_stratified_repo']:.3f} deff {v['deff_repo']:.2f}")
        for k, v in r["pairs"].items():
            print(f"  {k:28s} diff {100*v['diff']:+.2f} unp {100*v['se_unpaired']:.3f} pair {100*v['se_paired']:.3f} clpair {100*v['se_paired_clustered']:.3f} cr1 {100*v['se_paired_cluster_cr1']:.3f} corr {v['corr']:.3f} z {v['z']:.2f} only {v['n_A_only']}/{v['n_B_only']}")
    b = res["B_agentic_runs"]
    print(f"[B] n={b['n']}")
    for k, v in b["single"].items():
        print(f"  {k:48s} {100*v['mean']:.1f}% se1 {100*v['se_one_run_mean']:.2f} se10 {100*v['se_k10']:.2f} cl10 {100*v['se_k10_cluster_repo']:.2f} varx {v['var_x']:.4f} es {v['e_sigma2']:.4f} ratio {v['var_ratio_k10_vs_k1']:.3f} sd_runs {100*v['sd_of_run_means']:.2f} pred {100*v['sd_pred_from_decomposition']:.2f}")
    for k, v in b["pairs"].items():
        print(f"  {k}\n    diff {100*v['diff']:+.2f} unp1 {100*v['se_unpaired_k1']:.2f} pair1 {100*v['se_paired_k1']:.2f} unp10 {100*v['se_unpaired_k10']:.2f} pair10 {100*v['se_paired_k10']:.2f} cl10 {100*v['se_paired_k10_cluster_repo']:.2f} corr {v['corr_k10']:.2f} omega2 {v['omega2']:.3f} MDE k1 {100*v['mde_k1']:.1f} k10 {100*v['mde_k10']:.1f} kinf {100*v['mde_k_inf']:.1f}")
