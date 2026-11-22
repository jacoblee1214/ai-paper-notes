"""Who&When 공개 데이터(repo/Who&When) 통계와 단순 기준선 재계산.

- 손수 만든(Hand-Crafted) 58건, 알고리즘 생성(Algorithm-Generated) 126건
- 궤적 길이, 실패 단계 위치, 정답 에이전트 분포
- 무작위 / 항상 첫 단계 / 항상 마지막 단계 / 최다 발화 에이전트 기준선
- 논문 Table 1의 'Random' 행(29.10, 19.06, 12.00, 4.16)이 어떤 정의로 나오는지 대조
- 공식 evaluate.py의 부분 문자열 일치(`actual in predicted`)가 단계 정확도를 얼마나 부풀리는지
사용: python3 -I analysis/data_stats.py [데이터 폴더]  (기본: repo/Who&When)
"""
import json, os, sys, collections, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "repo", "Who&When")
OUT = os.path.join(ROOT, "analysis", "data_stats_results.json")


def load(sub):
    d = os.path.join(DATA, sub)
    files = sorted((f for f in os.listdir(d) if f.endswith(".json")), key=lambda x: int(x.split(".")[0]))
    rows = []
    for f in files:
        j = json.load(open(os.path.join(d, f), encoding="utf-8"))
        key = "role" if sub == "Hand-Crafted" else "name"
        hist = j["history"]
        names = [h.get(key, "?") for h in hist]
        base = [n.split(" (")[0] for n in names]  # 'Orchestrator (-> WebSurfer)' -> 'Orchestrator'
        rows.append(dict(
            file=f, n=len(hist), names=names, base=base, agent=j["mistake_agent"], step=int(j["mistake_step"]),
            qid=j["question_ID"], gt=j["ground_truth"], level=j.get("level"),
        ))
    return rows


def pct(x, n):
    return round(100 * x / n, 2)


def quantiles(v):
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]
    return dict(min=v[0], q25=q(.25), median=q(.5), q75=q(.75), max=v[-1], mean=round(st.mean(v), 2))


def sub_match(actual, pred):  # evaluate.py: `actual_step in pred['predicted_step']` (문자열 부분 일치)
    return str(actual) in str(pred)


res = {}
for sub in ["Hand-Crafted", "Algorithm-Generated"]:
    R = load(sub)
    N = len(R)
    out = dict(N=N)
    out["unique_question_ids"] = len({r["qid"] for r in R})
    out["length"] = quantiles([r["n"] for r in R])
    out["n_agents_distinct"] = quantiles([len(set(r["names"])) for r in R])
    out["mistake_step"] = quantiles([r["step"] for r in R])
    rel = [r["step"] / (r["n"] - 1) if r["n"] > 1 else 0 for r in R]
    out["mistake_step_relative"] = quantiles([round(x, 3) for x in rel])
    out["step_out_of_range"] = sum(1 for r in R if not (0 <= r["step"] < r["n"]))
    out["step_eq_0"] = sum(r["step"] == 0 for r in R)
    out["step_eq_last"] = sum(r["step"] == r["n"] - 1 for r in R)
    out["step_in_first_third"] = sum(rel_i < 1 / 3 for rel_i in rel)
    out["step_le_5"] = sum(r["step"] <= 5 for r in R)
    # 정답 에이전트가 그 단계의 발화자인가 (라벨 일관성)
    out["agent_equals_speaker_at_step"] = sum(0 <= r["step"] < r["n"] and r["names"][r["step"]] == r["agent"] for r in R)
    out["agent_in_history"] = sum(r["agent"] in r["names"] for r in R)
    out["agent_label_counts"] = collections.Counter(r["agent"] for r in R).most_common(8)
    # --- 기준선 (분수 = 맞힌 수 / N)
    b = {}
    # 에이전트 수준
    b["agent_random_uniform_over_distinct_agents"] = sum(1 / len(set(r["names"])) for r in R)  # 기대 맞힌 수
    b["agent_random_via_random_step"] = sum(sum(1 for x in r["names"] if x == r["agent"]) / r["n"] for r in R)
    b["agent_most_talkative"] = sum(collections.Counter(r["names"]).most_common(1)[0][0] == r["agent"] for r in R)
    b["agent_first_speaker"] = sum(r["names"][0] == r["agent"] for r in R)
    b["agent_last_speaker"] = sum(r["names"][-1] == r["agent"] for r in R)
    # 가장 흔한 라벨을 모두에게 (이 데이터셋에서는 오라클 상수)
    top = collections.Counter(r["agent"] for r in R).most_common(1)[0]
    b["agent_constant_most_common_label"] = (top[0], top[1])
    # 단계 수준
    b["step_random_uniform"] = sum(1 / r["n"] for r in R)
    b["step_always_first(0)"] = sum(r["step"] == 0 for r in R)
    b["step_always_last"] = sum(r["step"] == r["n"] - 1 for r in R)
    b["step_always_middle"] = sum(r["step"] == (r["n"] - 1) // 2 for r in R)
    c = collections.Counter(r["step"] for r in R).most_common(1)[0]
    b["step_constant_most_common_label"] = (c[0], c[1])
    # 허용 오차
    for tol in [1, 2, 3, 4, 5]:
        b[f"step_random_tol{tol}"] = sum(len([p for p in range(r["n"]) if abs(p - r["step"]) <= tol]) / r["n"] for r in R)
        b[f"step_first_tol{tol}"] = sum(abs(0 - r["step"]) <= tol for r in R)
        b[f"step_last_tol{tol}"] = sum(abs(r["n"] - 1 - r["step"]) <= tol for r in R)
    # 첫 단계를 에이전트 예측에도 쓰는 결합 기준선: (agent=first speaker, step=0)
    b["joint_first_step_first_agent"] = sum(r["step"] == 0 and r["names"][0] == r["agent"] for r in R)
    out["baselines_counts"] = {k: (v if isinstance(v, tuple) else round(v, 3)) for k, v in b.items()}
    out["baselines_pct"] = {k: pct(v, N) for k, v in b.items() if not isinstance(v, tuple)}
    # --- evaluate.py 부분 문자열 일치 효과: 무작위 단계 예측의 기대 정확도 (엄격 vs 부분 문자열)
    strict = sum(1 / r["n"] for r in R)
    sub_exp = sum(sum(sub_match(r["step"], p) for p in range(r["n"])) / r["n"] for r in R)
    out["substring_effect"] = dict(
        random_step_strict_pct=pct(strict, N), random_step_substring_pct=pct(sub_exp, N),
        always_last_strict_pct=pct(sum(r["step"] == r["n"] - 1 for r in R), N),
        always_last_substring_pct=pct(sum(sub_match(r["step"], r["n"] - 1) for r in R), N),
        # 한 단계를 예측했을 때 라벨이 부분 문자열로 맞는 '다른' 예측 수 평균
        mean_spurious_matching_predictions=round(st.mean(sum(sub_match(r["step"], p) and p != r["step"] for p in range(r["n"])) for r in R), 3),
        trajectories_with_any_spurious=sum(any(sub_match(r["step"], p) and p != r["step"] for p in range(r["n"])) for r in R),
    )

    # 에이전트 부분 문자열 효과: 같은 궤적의 다른 에이전트 이름 안에 정답 이름이 들어 있나 (evaluate.py `actual_agent in predicted_agent`)
    def regex_name(n):  # evaluate.py 정규식 [\\w_]+ 로 뽑히는 부분 (공백·괄호 앞까지)
        import re
        m = re.match(r"[\w_]+", n)
        return m.group(0) if m else n
    sp = 0
    exp_strict = exp_sub = 0.0
    for r in R:
        cands = sorted({regex_name(x) for x in r["names"]} - ({"human"} if sub == "Hand-Crafted" else set()))
        if r["agent"] not in cands:
            cands.append(r["agent"])
        wrong_but_match = [c for c in cands if c != r["agent"] and r["agent"] in c]
        sp += bool(wrong_but_match)
        exp_strict += 1 / len(cands)
        exp_sub += sum(r["agent"] in c for c in cands) / len(cands)
    out["agent_substring_effect"] = dict(trajectories_with_spurious_agent_match=sp,
        random_agent_strict_pct=pct(exp_strict, N), random_agent_substring_pct=pct(exp_sub, N))
    # --- 정규화한 에이전트 기준선: 괄호 꼬리표를 떼고 'human'(질문 입력)을 뺀다
    def real(r):
        return [x for x in r["base"] if x != "human"]
    out["n_base_agents_excl_human"] = quantiles([len(set(real(r))) for r in R])
    nb = {}
    nb["agent_random_uniform_over_base_agents"] = sum(1 / len(set(real(r))) for r in R)
    nb["agent_most_talkative_base"] = sum(collections.Counter(real(r)).most_common(1)[0][0] == r["agent"] for r in R)
    nb["agent_first_nonhuman_speaker_base"] = sum(real(r)[0] == r["agent"] for r in R)
    nb["agent_last_speaker_base"] = sum(real(r)[-1] == r["agent"] for r in R)
    nb["agent_label_equals_speaker_at_step_base"] = sum(r["base"][r["step"]] == r["agent"] for r in R)
    for a, cnt in collections.Counter(r["agent"] for r in R).most_common(3):
        nb[f"agent_constant_{a}"] = cnt  # 라벨 분포를 미리 알고 한 에이전트로 고정(표본 내 상한 성격)
    out["baselines_base_counts"] = {k: round(v, 3) for k, v in nb.items()}
    out["baselines_base_pct"] = {k: pct(v, N) for k, v in nb.items()}
    if sub == "Algorithm-Generated":
        grp = lambda r: "AssistantBench" if r["level"] in ("Hard", "Medium") else "GAIA"
        for ss in ("GAIA", "AssistantBench"):
            Rs = [r for r in R if grp(r) == ss]
            out[f"subset_{ss}"] = dict(N=len(Rs), step_random_pct=pct(sum(1 / r["n"] for r in Rs), len(Rs)),
                agent_random_pct=pct(sum(1 / len(set(r["names"])) for r in Rs), len(Rs)),
                first_speaker_pct=pct(sum(r["names"][0] == r["agent"] for r in Rs), len(Rs)),
                step0_pct=pct(sum(r["step"] == 0 for r in Rs), len(Rs)), mean_len=round(st.mean(r["n"] for r in Rs), 2))
    res[sub] = out

# 하위 집합(GAIA / AssistantBench) 구성: 질문 ID 형식으로 구분이 안 되므로 level 필드 유무로만 확인
res["Algorithm-Generated"]["has_level_field"] = sum(1 for r in load("Algorithm-Generated") if r["level"] is not None)

# --- 논문 Table 5(최대·최소 에이전트 수, 로그 길이) 대조: 알고리즘 생성은 level 로 GAIA/AssistantBench 분리 가능
t5 = {}
Ralg = load("Algorithm-Generated")
for ss in ("GAIA", "AssistantBench"):
    Rs = [r for r in Ralg if ("AssistantBench" if r["level"] in ("Hard", "Medium") else "GAIA") == ss]
    t5[f"alg_{ss}"] = dict(N=len(Rs), agents_max=max(len(set(r["names"])) for r in Rs), agents_min=min(len(set(r["names"])) for r in Rs),
                          len_max=max(r["n"] for r in Rs), len_min=min(r["n"] for r in Rs))
Rh = load("Hand-Crafted")
t5["hand_all"] = dict(N=len(Rh), base_agents_excl_human_max=max(len({x for x in r["base"] if x != "human"}) for r in Rh),
                      base_agents_excl_human_min=min(len({x for x in r["base"] if x != "human"}) for r in Rh),
                      distinct_role_strings_max=max(len(set(r["names"])) for r in Rh), len_max=max(r["n"] for r in Rh), len_min=min(r["n"] for r in Rh))
res["table5_check"] = t5
print("table5_check", t5)

# --- 논문 4.4 길이 수준(손수 만든): 수준별 표본 수와 무작위·마지막 단계 기준선 (accuracy 하락이 기준선 하락과 같은 방향인지)
lv_edges = [(1, 5, 17), (2, 19, 29), (3, 31, 49), (4, 51, 91), (5, 93, 130)]
lvl = []
for lv, lo, hi in lv_edges:
    Rs = [r for r in Rh if lo <= r["n"] <= hi]
    lvl.append(dict(level=lv, range=f"{lo}-{hi}", N=len(Rs),
        step_random_pct=round(100 * sum(1 / r["n"] for r in Rs) / max(1, len(Rs)), 2),
        step_always_last_pct=round(100 * sum(r["step"] == r["n"] - 1 for r in Rs) / max(1, len(Rs)), 2),
        agent_last_speaker_pct=round(100 * sum([x for x in r["base"] if x != "human"][-1] == r["agent"] for r in Rs) / max(1, len(Rs)), 2),
        agent_random_base_pct=round(100 * sum(1 / len({x for x in r["base"] if x != "human"}) for r in Rs) / max(1, len(Rs)), 2)))
res["hand_length_levels"] = lvl
res["hand_length_levels_unassigned"] = sum(1 for r in Rh if not any(lo <= r["n"] <= hi for _, lo, hi in lv_edges))
print("levels", lvl)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in res.items():
    if k in ("table5_check", "hand_length_levels", "hand_length_levels_unassigned"):
        continue
    print("==", k, "N", v["N"])
    for kk, vv in v.items():
        if kk not in ("baselines_counts",):
            print(" ", kk, vv)
