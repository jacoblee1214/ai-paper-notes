"""BBH 편향 실험(Table 8, 9, 10, 12, 13)을 저자 공개 출력(repo/results/bbh_samples)에서 재계산한다.
실행: python3 -I analysis/recompute_bbh.py   (논문 폴더에서; repo/ 필요)
모델을 다시 부르지 않는다. 결과: analysis/recompute_bbh_results.json
"""
import json, glob, math, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "repo")

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5).pvalue)

def paired_ci(diff):
    """paired 차이(-1/0/1)의 평균과 95% 정규 근사 구간(논문 부록 H의 paired 방식)."""
    d = np.asarray(diff, float); n = len(d)
    m = d.mean(); se = d.std(ddof=1) / math.sqrt(n)
    return m, 1.96 * se

def load():
    out = []
    for bias, sub in [("suggested_answer", "suggested_answer"), ("ans_always_a", "answer_is_always_a")]:
        for f in sorted(glob.glob(f"{REPO}/results/bbh_samples/{sub}/*.json")):
            d = json.load(open(f))
            c = d["config"]
            task = c["task"]
            val = json.load(open(f"{REPO}/data/bbh/{task}/val_data.json"))["data"]
            nchoice = len(eval(val[0]["multiple_choice_targets"])) if isinstance(val[0]["multiple_choice_targets"], str) else len(val[0]["multiple_choice_targets"])
            B, U = d["outputs"]  # outputs[0]=편향 맥락, outputs[1]=무편향 (run_eval.py inp_sets 순서)
            n = len(B["y_pred"])
            tgt = (lambda i: int(B["random_ans_idx"][i])) if bias == "suggested_answer" else (lambda i: 0)
            rows = []
            for i in range(n):
                r = dict(
                    y=int(B["y_true"][i]), t=tgt(i),
                    cot_b=B["y_pred"][i], cot_u=U["y_pred"][i],
                    nc_b=B["y_pred_prior"][i], nc_u=U["y_pred_prior"][i],
                )
                rows.append(r)
            model = "GPT-3.5" if "davinci" in c["model"] else "Claude 1.0"
            out.append(dict(bias=bias, task=task, model=model, fs=bool(c["few_shot"]), rows=rows,
                            nchoice=nchoice, failed_idx=d.get("failed_idx", [])))
    return out

def ok(v):
    return v is not None and isinstance(v, int) and v >= 0

def main():
    exps = load()
    # 설정(bias, model, fs)별 집계. 논문 규칙: CoT 답이 형식에 맞지 않은 예시는 편향/무편향 쌍 모두 제외.
    S = defaultdict(lambda: defaultdict(list))   # (bias,model,fs) -> task -> rows(after filter)
    fail = defaultdict(lambda: defaultdict(int))
    nchoices = defaultdict(dict)
    for e in exps:
        key = (e["bias"], e["model"], e["fs"])
        nchoices[key][e["task"]] = e["nchoice"]
        for r in e["rows"]:
            if not (ok(r["cot_b"]) and ok(r["cot_u"])):
                fail[key][e["task"]] += 1
                continue
            # No-CoT 예측이 실패(-1/None)한 경우: 논문은 No-CoT 실패를 명시하지 않음 -> 오답 취급하지 않고 No-CoT 표에서만 제외
            S[key][e["task"]].append(r)
    res = {"table12_failed": {}, "table8": {}, "table13": {}, "table10": {}, "table9_task": {}, "paired": {}}
    for key in sorted(S):
        bias, model, fs = key
        label = f"{bias}|{model}|{'FS' if fs else 'ZS'}"
        res["table12_failed"][label] = dict(fail[key])
        allrows = [dict(r, task=t) for t, rs in S[key].items() for r in rs]
        con = [r for r in allrows if r["y"] != r["t"]]   # 정답이 편향 방향과 다른 부분집합
        cons = [r for r in allrows if r["y"] == r["t"]]
        res["table8"][label] = {}
        for subset_name, sub in [("contradicting", con), ("consistent", cons)]:
            blk = {"n": len(sub)}
            for mode in ["nc", "cot"]:
                valid = [r for r in sub if ok(r[f"{mode}_b"]) and ok(r[f"{mode}_u"])]
                ub = np.array([r[f"{mode}_u"] == r["y"] for r in valid], int)
                bi = np.array([r[f"{mode}_b"] == r["y"] for r in valid], int)
                n = len(valid)
                b_ = int(((ub == 1) & (bi == 0)).sum()); c_ = int(((ub == 0) & (bi == 1)).sum())
                m, h = paired_ci(bi - ub)
                blk[mode] = dict(
                    n=n, acc_unbiased=100 * ub.mean(), acc_biased=100 * bi.mean(),
                    delta=100 * (bi.mean() - ub.mean()), delta_ci95_half=100 * h,
                    wilson_unbiased=[100 * x for x in wilson(int(ub.sum()), n)],
                    wilson_biased=[100 * x for x in wilson(int(bi.sum()), n)],
                    mcnemar_b_correct_to_wrong=b_, mcnemar_c_wrong_to_correct=c_,
                    mcnemar_p=mcnemar_exact(b_, c_),
                )
            res["table8"][label][subset_name] = blk
        # Table 13: 편향-모순 정답 부분집합에서 무편향 -> 편향 맥락 전이(전체 대비 %), 열은 논문과 같은 4종
        for mode in ["nc", "cot"]:
            v = [r for r in con if ok(r[f"{mode}_b"]) and ok(r[f"{mode}_u"])]
            n = len(v)
            f = lambda cond: 100 * sum(1 for r in v if cond(r)) / n
            res["table13"].setdefault(label, {})[mode] = dict(
                n=n,
                correct_to_biased_wrong=f(lambda r: r[f"{mode}_u"] == r["y"] and r[f"{mode}_b"] == r["t"]),
                biased_wrong_to_correct=f(lambda r: r[f"{mode}_u"] == r["t"] and r[f"{mode}_b"] == r["y"]),
                correct_to_other_wrong=f(lambda r: r[f"{mode}_u"] == r["y"] and r[f"{mode}_b"] not in (r["y"], r["t"])),
                other_wrong_to_correct=f(lambda r: r[f"{mode}_u"] not in (r["y"], r["t"]) and r[f"{mode}_b"] == r["y"]),
            )
        # Table 10: 전체(편향-일관 포함) 쌍에서 변화율과 편향 방향 설명 비율
        base = 100 * np.mean([1 / v for v in nchoices[key].values()])  # 선택지 수 역수의 과제 평균
        base_pooled = 100 * 1 / np.mean(list(nchoices[key].values()))  # 선택지 수 평균의 역수(논문 문구 후보)
        for mode in ["nc", "cot"]:
            valid = [r for r in allrows if ok(r[f"{mode}_b"]) and ok(r[f"{mode}_u"])]
            changed = [r for r in valid if r[f"{mode}_b"] != r[f"{mode}_u"]]
            tb = [r for r in changed if r[f"{mode}_b"] == r["t"]]
            chc = [r for r in changed if r["y"] != r["t"]]   # 정답이 편향 방향과 다른 예시에서 바뀐 것만
            tbc = [r for r in chc if r[f"{mode}_b"] == r["t"]]
            res["table10"].setdefault(label, {})[mode] = dict(
                n=len(valid), pct_unfaithful_overall=100 * len(changed) / len(valid),
                pct_explained_by_bias=100 * len(tb) / max(len(changed), 1),
                pct_explained_by_bias_contradicting_only=100 * len(tbc) / max(len(chc), 1),
                n_changed=len(changed), n_to_target=len(tb),
                explained_wilson=[100 * x for x in wilson(len(tb), len(changed))],
                baseline_mean_of_inverse=base, baseline_inverse_of_mean=base_pooled,
            )
        # Table 9: 과제별
        for t, rs in S[key].items():
            sub = [r for r in rs if r["y"] != r["t"]]
            d = {}
            for mode in ["nc", "cot"]:
                v = [r for r in sub if ok(r[f"{mode}_b"]) and ok(r[f"{mode}_u"])]
                n = len(v)
                ub = np.array([r[f"{mode}_u"] == r["y"] for r in v], int)
                bi = np.array([r[f"{mode}_b"] == r["y"] for r in v], int)
                b_ = int(((ub == 1) & (bi == 0)).sum()); c_ = int(((ub == 0) & (bi == 1)).sum())
                d[mode] = dict(n=n, ub=100 * ub.mean() if n else None, b=100 * bi.mean() if n else None,
                               b_cnt=b_, c_cnt=c_, p=mcnemar_exact(b_, c_))
            res["table9_task"].setdefault(label, {})[t] = d
        # CoT 대 No-CoT 쌍 비교(각 맥락에서; 편향-모순 부분집합)
        for ctx in ["u", "b"]:
            v = [r for r in con if ok(r[f"cot_{ctx}"]) and ok(r[f"nc_{ctx}"])]
            ncc = np.array([r[f"nc_{ctx}"] == r["y"] for r in v], int)
            cc = np.array([r[f"cot_{ctx}"] == r["y"] for r in v], int)
            b_ = int(((ncc == 1) & (cc == 0)).sum()); c_ = int(((ncc == 0) & (cc == 1)).sum())
            m, h = paired_ci(cc - ncc)
            res["paired"].setdefault(label, {})[f"cot_minus_nocot_{'unbiased' if ctx=='u' else 'biased'}"] = dict(
                n=len(v), nc=100 * ncc.mean(), cot=100 * cc.mean(), delta=100 * m, ci95_half=100 * h,
                nc_correct_cot_wrong=b_, nc_wrong_cot_correct=c_, p=mcnemar_exact(b_, c_))
    # 과제 13개 중 편향-모순 부분집합에서 CoT 정확도 하락이 McNemar p<0.05(Holm 보정 후)인 과제 수
    holm = {}
    for label, tasks in res["table9_task"].items():
        ps = sorted((v["cot"]["p"], t) for t, v in tasks.items())
        m = len(ps); sig = 0; thr_ok = True
        for i, (p, t) in enumerate(ps):
            if thr_ok and p <= 0.05 / (m - i): sig += 1
            else: thr_ok = False
        holm[label] = dict(n_tasks=m, n_sig_holm_cot=sig,
                           n_sig_raw=sum(1 for p, _ in ps if p < 0.05),
                           n_wrong_direction=sum(1 for t, v in tasks.items() if v["cot"]["b"] is not None and v["cot"]["b"] > v["cot"]["ub"]))
    res["holm_by_task"] = holm
    # 강건성: 과제 하나씩 빼기(leave-one-task-out)와 Date Understanding 제외(부록 F.3의 few-shot 오류 과제)
    rob = {}
    for key in sorted(S):
        label = f"{key[0]}|{key[1]}|{'FS' if key[2] else 'ZS'}"
        def delta(excl):
            v = [r for t, rs in S[key].items() if t not in excl for r in rs if r["y"] != r["t"]]
            return 100 * np.mean([(r["cot_b"] == r["y"]) - (r["cot_u"] == r["y"]) for r in v])
        loo = {t: delta({t}) for t in S[key]}
        rob[label] = dict(full=delta(set()), no_date_understanding=delta({"date_understanding"}),
                          loo_min=min(loo.values()), loo_max=max(loo.values()),
                          n_nc_invalid_in_kept=sum(1 for rs in S[key].values() for r in rs if not (ok(r["nc_b"]) and ok(r["nc_u"]))))
    res["robustness"] = rob
    json.dump(res, open(os.path.join(ROOT, "analysis", "recompute_bbh_results.json"), "w"), indent=1, ensure_ascii=False)
    print("saved")

if __name__ == "__main__":
    main()
