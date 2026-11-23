"""GPT-4 판정자와 사람의 일치도(논문 표 5·13, 그림 2·3b) 재계산, 불확실성, 우연 보정.

입력: data/hf_mtbench/human.parquet, gpt4_pair.parquet (HF lmsys/mt_bench_human_judgments, CC BY 4.0)
출력: analysis/agreement_results.json

정의(논문 부록 D.3, FastChat compute_agreement.py와 같은 규칙)
  - 한 항목 = (turn, question_id, 모델 쌍). 모델 이름 사전순으로 A/B를 정해 순서를 맞춘다.
  - "사람" = judge 이름이 expert로 시작하는 58명, "저자" = author로 시작하는 7명.
  - GPT-4 쌍 비교의 'tie (inconsistent)'는 tie로 센다. S1 = 모두 포함, S2 = 양쪽 모두 tie 아닌 표만.
  - G4-사람 일치: 항목의 GPT-4 표 1개와 그 항목의 사람 표 각각을 비교(사람 표 1개 = 1표).
  - 사람-사람 일치: 같은 항목에 표를 던진 서로 다른 두 사람의 모든 쌍.
불확실성: 질문(80개) 단위 군집 부트스트랩. 같은 질문의 여러 모델 쌍·두 턴·여러 사람 표가 한 질문 효과를 공유한다.
"""
from __future__ import annotations

import collections
import itertools
import json
import os

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]
RNG = np.random.default_rng(0)
B = 4000  # 부트스트랩 반복
FLIP = {"model_a": "model_b", "model_b": "model_a"}


def load():
    H = pq.read_table(f"{ROOT}/data/hf_mtbench/human.parquet", columns=COLS).to_pylist()
    G = pq.read_table(f"{ROOT}/data/hf_mtbench/gpt4_pair.parquet", columns=COLS).to_pylist()

    def canon(r):
        a, b, w = r["model_a"], r["model_b"], r["winner"]
        if not a < b:
            a, b, w = b, a, FLIP.get(w, w)
        lab = {"model_a": "A", "model_b": "B"}.get(w, "tie")
        return (r["turn"], r["question_id"], a, b), lab, w

    g, hv, av = {}, collections.defaultdict(list), collections.defaultdict(list)
    for r in G:
        k, lab, w = canon(r)
        g[k] = lab
    for r in H:
        k, lab, _ = canon(r)
        (hv if r["judge"].startswith("expert") else av)[k].append((r["judge"], lab))
    return g, hv, av


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def pct(a, lo_hi=None):
    return None if a is None else round(float(a), 4)


# ------------------------------------------------------------ 항목 단위 일치 통계(질문별 합계로 모은다)
def per_question_stats(g, hv, av, turn, s2, kind):
    """질문별(81..160 -> 0..79) (일치 수, 비교 수) 배열. kind: g4_human, g4_author, human_human, g4_humanM"""
    agree = np.zeros(80)
    n = np.zeros(80)
    for (t, q, a, b), votes in hv.items():
        if t != turn:
            continue
        i = q - 81
        gv = g[(t, q, a, b)]
        if kind == "g4_human":
            if s2 and gv == "tie":
                continue
            for _, v in votes:
                if s2 and v == "tie":
                    continue
                n[i] += 1
                agree[i] += (gv == v)
        elif kind == "human_human":
            for (_, v1), (_, v2) in itertools.combinations(votes, 2):
                if s2 and (v1 == "tie" or v2 == "tie"):
                    continue
                n[i] += 1
                agree[i] += (v1 == v2)
        elif kind == "g4_humanM":
            # 사람 다수결: 항목당 1표. 동수면 균등 분할(부록 D.3). S2는 tie 표를 먼저 버리고 다수결.
            if s2 and gv == "tie":
                continue
            vs = [v for _, v in votes if not (s2 and v == "tie")]
            if not vs:
                continue
            c = collections.Counter(vs)
            top = max(c.values())
            winners = [k for k, v in c.items() if v == top]
            n[i] += 1
            agree[i] += sum(w == gv for w in winners) / len(winners)
    if kind == "g4_author":
        for (t, q, a, b), votes in av.items():
            if t != turn:
                continue
            gv = g[(t, q, a, b)]
            if s2 and gv == "tie":
                continue
            for _, v in votes:
                if s2 and v == "tie":
                    continue
                n[q - 81] += 1
                agree[q - 81] += (gv == v)
    return agree, n


def boot_idx(nq=80):
    return RNG.integers(0, nq, size=(B, nq))


def ratio_ci(agree, n, idx):
    """질문 군집 부트스트랩 퍼센타일 95% 구간"""
    A = agree[idx].sum(1)
    N = n[idx].sum(1)
    r = A / N
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), r


def main():
    g, hv, av = load()
    qids = sorted({k[1] for k in g})
    assert len(qids) == 80 and qids[0] == 81 and qids[-1] == 160
    # 질문 id 81..160을 0..79로
    def stats(turn, s2, kind):
        ag, nn = per_question_stats(g, hv, av, turn, s2, kind)
        return ag, nn

    idx = boot_idx()
    out = {"meta": {"B": B, "cluster": "question_id (80)", "seed": 0}}

    # ---------------- 표 5, 13 재계산
    paper = {  # (turn, s2, kind): (논문 비율, 논문 표 수)
        (1, False, "g4_human"): (0.66, 1343), (1, True, "g4_human"): (0.85, 859),
        (1, False, "human_human"): (0.63, 721), (1, True, "human_human"): (0.81, 479),
        (2, False, "g4_human"): (0.66, 1325), (2, True, "g4_human"): (0.85, 864),
        (2, False, "human_human"): (0.67, 707), (2, True, "human_human"): (0.82, 474),
        (1, False, "g4_author"): (0.69, 345), (1, True, "g4_author"): (0.92, 201),
        (2, False, "g4_author"): (0.66, 341), (2, True, "g4_author"): (0.88, 205),
        (1, False, "g4_humanM"): (0.67, 821), (1, True, "g4_humanM"): (0.85, 546),
        (2, False, "g4_humanM"): (0.68, 812), (2, True, "g4_humanM"): (0.85, 557),
    }
    rows = {}
    cache = {}
    for (turn, s2, kind), (pp, pn) in paper.items():
        ag, nn = stats(turn, s2, kind)
        cache[(turn, s2, kind)] = (ag, nn)
        k, n = int(round(ag.sum())), nn.sum()
        lo, hi, _ = ratio_ci(ag, nn, idx)
        wl, wh = wilson(ag.sum(), n)
        rows[f"turn{turn}_{'S2' if s2 else 'S1'}_{kind}"] = {
            "agree": round(float(ag.sum()), 1), "n": float(n), "rate": round(float(ag.sum() / n), 4),
            "wilson95": [round(wl, 4), round(wh, 4)], "cluster_boot95": [round(lo, 4), round(hi, 4)],
            "design_effect_cluster_vs_wilson": round(((hi - lo) / (wh - wl)) ** 2, 2),
            "paper_rate": pp, "paper_n": pn, "n_match": bool(abs(n - pn) < 0.5),
            "rate_match_rounded": bool(round(ag.sum() / n * 100) == round(pp * 100)),
        }
    out["table5_13_recompute"] = rows

    # ---------------- 주장: G4-사람 일치 ≈ 사람-사람 일치 (차이와 군집 구간)
    cmp_ = {}
    for turn in (1, 2):
        for s2 in (False, True):
            a1, n1 = cache[(turn, s2, "g4_human")]
            a2, n2 = cache[(turn, s2, "human_human")]
            r1 = a1[idx].sum(1) / n1[idx].sum(1)
            r2 = a2[idx].sum(1) / n2[idx].sum(1)
            d = r1 - r2
            point = a1.sum() / n1.sum() - a2.sum() / n2.sum()
            cmp_[f"turn{turn}_{'S2' if s2 else 'S1'}"] = {
                "diff_g4h_minus_hh": round(float(point), 4),
                "cluster_boot95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                "frac_boot_diff_le_0": round(float((d <= 0).mean()), 4),
            }
    out["g4h_vs_hh_diff"] = cmp_

    # ---------------- 같은 항목 비교: 사람 표가 2개 이상인 항목에서만 G4-사람 일치를 다시 계산
    matched = {}
    for turn in (1, 2):
        for s2 in (False, True):
            ag = np.zeros(80); nn = np.zeros(80)
            for (t, q, a, b), votes in hv.items():
                if t != turn or len(votes) < 2:
                    continue
                gv = g[(t, q, a, b)]
                if s2 and gv == "tie":
                    continue
                for _, v in votes:
                    if s2 and v == "tie":
                        continue
                    nn[q - 81] += 1
                    ag[q - 81] += (gv == v)
            a2, n2 = cache[(turn, s2, "human_human")]
            r1 = ag[idx].sum(1) / nn[idx].sum(1)
            r2 = a2[idx].sum(1) / n2[idx].sum(1)
            d = r1 - r2
            matched[f"turn{turn}_{'S2' if s2 else 'S1'}"] = {
                "g4_human_on_multivote_items": {"rate": round(float(ag.sum() / nn.sum()), 4), "n": float(nn.sum())},
                "human_human": {"rate": round(float(a2.sum() / n2.sum()), 4), "n": float(n2.sum())},
                "diff": round(float(ag.sum() / nn.sum() - a2.sum() / n2.sum()), 4),
                "cluster_boot95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            }
    out["matched_items_diff"] = matched

    # ---------------- 문맥 없는 기준선과 우연 보정(κ)
    out["chance_corrected"] = chance_corrected(g, hv, idx)

    # ---------------- 쌍별 일치와 승률 차이(그림 2 주장)
    out["by_model_pair"] = by_pair(g, hv)
    from scipy import stats as _st
    gap = [r["human_winrate_gap_in_pair"] for r in out["by_model_pair"]]
    ag_ = [r["agree"] for r in out["by_model_pair"]]
    rho, p = _st.spearmanr(gap, ag_)
    out["by_model_pair_spearman_gap_vs_agree"] = {"rho": round(float(rho), 3), "p": round(float(p), 4), "n_pairs": len(gap)}

    # ---------------- GPT-4 쌍 비교 판정만으로 보는 위치 편향
    out["gpt4_pair_position"] = position_from_gpt4_pair(g, hv)
    gapd = {r["pair"]: r["human_winrate_gap_in_pair"] for r in out["by_model_pair"]}
    pp = out["gpt4_pair_position"]["per_pair"]
    rho, p = _st.spearmanr([gapd[k] for k in pp], [pp[k]["rate"] for k in pp])
    out["gpt4_pair_position"]["spearman_gap_vs_inconsistent"] = {"rho": round(float(rho), 3), "p": round(float(p), 4), "n_pairs": len(pp)}

    # ---------------- 자기 선호(GPT-4 승률 대 사람 승률, 같은 항목 위에서)
    out["win_rate_self_preference"] = win_rates(g, hv, idx)

    # ---------------- S1 혼동표: 행 = GPT-4 쌍 비교(동점에 비일관 포함), 열 = 전문가 표 (두 턴 합침)
    conf = collections.Counter()
    for (t, q, a, b), votes in hv.items():
        for _, v in votes:
            conf[(g[(t, q, a, b)], v)] += 1
    out["confusion_S1_g4_rows_human_cols"] = {f"{r}|{c}": conf[(r, c)] for r in ("A", "B", "tie") for c in ("A", "B", "tie")}
    out["g4_tie_incl_inconsistent_rate_on_voted_items"] = round(sum(v for (r, c), v in conf.items() if r == "tie") / sum(conf.values()), 4)
    out["human_tie_rate"] = round(sum(v for (r, c), v in conf.items() if c == "tie") / sum(conf.values()), 4)

    # ---------------- 민감도: 사람이 GPT-4 판정을 본 뒤 바꾼 표가 기록에 들어갔다면(논문 §4.2: 불일치 시 GPT-4 판정을 보여 주고 34%가 바꿀 의향)
    sens = {}
    for turn in (1, 2):
        r = out["table5_13_recompute"][f"turn{turn}_S2_g4_human"]["rate"]
        d_obs = 1 - r
        d_pre = d_obs / (1 - 0.34)
        sens[f"turn{turn}_S2"] = {"observed_agreement": r, "hypothetical_pre_exposure_agreement_if_34pct_of_prior_disagreements_flipped": round(1 - d_pre, 4)}
    sens["note"] = "가정 시나리오다. 기록된 표가 노출 전인지 후인지는 논문과 데이터 카드에 없다. 34%는 '바꿀 의향'이지 실제 변경 비율이 아니다."
    out["sensitivity_exposure_to_gpt4"] = sens
    # S2가 남기는 표의 비율
    ret = {}
    for turn in (1, 2):
        ret[f"turn{turn}_g4_human"] = round(out["table5_13_recompute"][f"turn{turn}_S2_g4_human"]["n"] / out["table5_13_recompute"][f"turn{turn}_S1_g4_human"]["n"], 3)
        ret[f"turn{turn}_human_human"] = round(out["table5_13_recompute"][f"turn{turn}_S2_human_human"]["n"] / out["table5_13_recompute"][f"turn{turn}_S1_human_human"]["n"], 3)
    out["s2_retention"] = ret

    with open(f"{ROOT}/analysis/agreement_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print_summary(out)


# ------------------------------------------------------------------------------------
def chance_corrected(g, hv, idx):
    """(1) 논문의 R=33%/50% 기준 정규화, (2) 모델 쌍×턴 층별 주변분포로 기대한 일치 p_e, (3) 문맥 없는 기준선.
    층 = (turn, 모델 쌍). 층 안에서 GPT-4 표와 사람 표의 주변분포 곱의 합이 우연 일치.
    기준선 = '그 모델 쌍에서 다른 질문의 사람 표 다수결'을 그대로 찍는 규칙(질문 내용을 보지 않음)."""
    res = {}
    for s2 in (False, True):
        for turn in (1, 2):
            labs = ["A", "B"] if s2 else ["A", "B", "tie"]
            # 층별 개수
            strat_g = collections.defaultdict(collections.Counter)
            strat_h = collections.defaultdict(collections.Counter)
            recs = []  # (q, stratum, gv, hv)
            for (t, q, a, b), votes in hv.items():
                if t != turn:
                    continue
                gv = g[(t, q, a, b)]
                if s2 and gv == "tie":
                    continue
                for _, v in votes:
                    if s2 and v == "tie":
                        continue
                    recs.append((q - 81, (a, b), gv, v))
                    strat_g[(a, b)][gv] += 1
                    strat_h[(a, b)][v] += 1
            n = len(recs)
            po = sum(r[2] == r[3] for r in recs) / n
            pe = 0.0
            for s, cg in strat_g.items():
                ng = sum(cg.values()); nh = sum(strat_h[s].values())
                pe += ng / n * sum(cg[c] / ng * strat_h[s][c] / nh for c in labs)
            # 쌍을 무시한 풀링 κ
            cg = collections.Counter(r[2] for r in recs); ch = collections.Counter(r[3] for r in recs)
            pe_pool = sum(cg[c] / n * ch[c] / n for c in labs)
            R = 0.5 if s2 else 1 / 3
            # 문맥 없는 기준선: 층별 사람 표 다수결(해당 질문 제외)
            by_q_strat = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
            for q, s, gv, v in recs:
                pass
            base_ag = base_ag_g = 0
            base_n = 0
            per_q_ag = np.zeros(80); per_q_n = np.zeros(80); per_q_agk = np.zeros(80)
            human_votes_by = collections.defaultdict(list)  # 층 -> [(q, v)]
            for q, s, gv, v in recs:
                human_votes_by[s].append((q, v))
            for q, s, gv, v in recs:
                c = collections.Counter(vv for qq, vv in human_votes_by[s] if qq != q)
                if not c:
                    continue
                pred = max(labs, key=lambda x: (c[x], x))  # 동수는 사전순
                per_q_n[q] += 1
                per_q_ag[q] += (pred == v)
            kappa_strat = (po - pe) / (1 - pe)
            kappa_R = (po - R) / (1 - R)
            # κ의 질문 군집 부트스트랩
            qa = np.zeros(80); qn = np.zeros(80)
            for q, s, gv, v in recs:
                qn[q] += 1; qa[q] += (gv == v)
            ks = []
            for bi in idx[:1000]:
                po_b = qa[bi].sum() / qn[bi].sum()
                ks.append((po_b - pe) / (1 - pe))  # p_e는 점추정치로 고정(근사)
            res[f"turn{turn}_{'S2' if s2 else 'S1'}"] = {
                "n": n, "p_o": round(po, 4), "R_paper": round(R, 4),
                "kappa_vs_paper_R": round(kappa_R, 4),
                "p_e_pooled": round(pe_pool, 4), "kappa_pooled": round((po - pe_pool) / (1 - pe_pool), 4),
                "p_e_pair_stratified": round(pe, 4), "kappa_pair_stratified": round(kappa_strat, 4),
                "kappa_pair_stratified_cluster_boot95_pe_fixed": [round(float(np.percentile(ks, 2.5)), 4), round(float(np.percentile(ks, 97.5)), 4)],
                "baseline_pair_majority_agree_with_human": round(float(per_q_ag.sum() / per_q_n.sum()), 4),
                "baseline_n": float(per_q_n.sum()),
            }
    # 사람-사람 쌍(S1·S2)의 층별 기대 일치
    for s2 in (False, True):
        for turn in (1, 2):
            labs = ["A", "B"] if s2 else ["A", "B", "tie"]
            strat = collections.defaultdict(collections.Counter)
            for (t, q, a, b), votes in hv.items():
                if t != turn or len(votes) < 2:
                    continue
                for _, v in votes:
                    if s2 and v == "tie":
                        continue
                    strat[(a, b)][v] += 1
            po_n = po_d = 0.0; pe_acc = 0.0; tot = 0
            for (t, q, a, b), votes in hv.items():
                if t != turn:
                    continue
                for (_, v1), (_, v2) in itertools.combinations(votes, 2):
                    if s2 and (v1 == "tie" or v2 == "tie"):
                        continue
                    po_d += 1; po_n += (v1 == v2)
                    c = strat[(a, b)]; m = sum(c[x] for x in labs)
                    pe_acc += sum((c[x] / m) ** 2 for x in labs)
            po = po_n / po_d; pe = pe_acc / po_d
            res[f"turn{turn}_{'S2' if s2 else 'S1'}_human_human"] = {
                "n_pairs": int(po_d), "p_o": round(po, 4), "p_e_pair_stratified": round(pe, 4),
                "kappa_pair_stratified": round((po - pe) / (1 - pe), 4),
                "kappa_vs_paper_R": round((po - (0.5 if s2 else 1 / 3)) / (1 - (0.5 if s2 else 1 / 3)), 4),
            }
    return res


def by_pair(g, hv):
    """모델 쌍별 G4-사람 S2 일치(그림 2의 점). x = 사람 승률 차이(비동점 사람 표 기준 A의 승률 - 0.5의 2배 등이 아닌, 두 모델 평균 승률 차이)"""
    pair_rec = collections.defaultdict(lambda: [0, 0])
    wins = collections.defaultdict(lambda: collections.Counter())
    for (t, q, a, b), votes in hv.items():
        gv = g[(t, q, a, b)]
        for _, v in votes:
            if v != "tie":
                wins[(a, b)][v] += 1
            if v == "tie" or gv == "tie":
                continue
            pair_rec[(a, b)][1] += 1
            pair_rec[(a, b)][0] += (gv == v)
    out = []
    for (a, b), (k, n) in sorted(pair_rec.items()):
        wa = wins[(a, b)]["A"]; wb = wins[(a, b)]["B"]
        diff = abs(wa - wb) / max(wa + wb, 1)
        out.append({"pair": f"{a} vs {b}", "n": n, "agree": round(k / n, 4), "wilson95": [round(x, 4) for x in wilson(k, n)],
                    "human_nontie_votes": wa + wb, "human_winrate_gap_in_pair": round(diff, 4)})
    return out


def position_from_gpt4_pair(g, hv):
    """gpt4_pair의 tie (inconsistent) 비율: 순서를 바꿔 두 번 물어 한 번이라도 결과가 달랐던 항목의 비율(직접 계산 가능한 하한)."""
    raw = pq.read_table(f"{ROOT}/data/hf_mtbench/gpt4_pair.parquet", columns=COLS).to_pylist()
    c = collections.Counter(r["winner"] for r in raw)
    tot = sum(c.values())
    res = {"counts": dict(c), "n": tot, "inconsistent_rate": round(c["tie (inconsistent)"] / tot, 4),
           "inconsistent_wilson95": [round(x, 4) for x in wilson(c["tie (inconsistent)"], tot)]}
    # 승률 차이가 큰 쌍일수록 비일관이 줄어드는가(표 11의 주장): 쌍별 비일관 비율과 사람 승률 차이
    per_pair = collections.defaultdict(lambda: [0, 0])
    for r in raw:
        a, b = sorted([r["model_a"], r["model_b"]])
        per_pair[(a, b)][1] += 1
        per_pair[(a, b)][0] += (r["winner"] == "tie (inconsistent)")
    pp = {f"{a} vs {b}": {"inconsistent": k, "n": n, "rate": round(k / n, 3)} for (a, b), (k, n) in sorted(per_pair.items())}
    res["per_pair"] = pp
    return res


def win_rates(g, hv, idx):
    """논문 그림 3(b): 비동점 승률(상대별 승/(승+패)를 상대 평균). 같은 항목·같은 표 수 위에서 GPT-4 쌍 비교와 사람을 비교.
    GPT-4 표는 사람 표 수만큼 복제(사람 표가 있는 항목만 비교하므로 표본 구성이 같다)."""
    models = sorted({k[2] for k in g} | {k[3] for k in g})
    mi = {m: i for i, m in enumerate(models)}
    M = len(models)

    def tab(turn, judge):
        # [q, i, j] = i가 j를 이긴 표 수
        W = np.zeros((80, M, M))
        for (t, q, a, b), votes in hv.items():
            if turn is not None and t != turn:
                continue
            gv = g[(t, q, a, b)]
            for _, v in votes:
                x = gv if judge == "g4" else v
                if x == "A":
                    W[q - 81, mi[a], mi[b]] += 1
                elif x == "B":
                    W[q - 81, mi[b], mi[a]] += 1
        return W

    def avg_wr(W):
        T = W.sum(0)
        wr = np.full(M, np.nan)
        for i in range(M):
            vals = []
            for j in range(M):
                if i != j and T[i, j] + T[j, i] > 0:
                    vals.append(T[i, j] / (T[i, j] + T[j, i]))
            wr[i] = np.mean(vals)
        return wr

    out = {}
    for turn in (1, 2, None):
        Wg, Wh = tab(turn, "g4"), tab(turn, "h")
        # 사람과 GPT-4가 모두 비동점인 표만 같은 표 위에서 비교하도록 따로 만든다(비동점 표 정의 차이 통제)
        wg, wh = avg_wr(Wg), avg_wr(Wh)
        diffs = []
        for bi in idx[:2000]:
            d = avg_wr(Wg[bi]) - avg_wr(Wh[bi])
            diffs.append(d)
        diffs = np.array(diffs)
        name = "all" if turn is None else f"turn{turn}"
        out[name] = {m: {"g4_winrate": round(float(wg[mi[m]]), 3), "human_winrate": round(float(wh[mi[m]]), 3),
                         "diff": round(float(wg[mi[m]] - wh[mi[m]]), 3),
                         "diff_cluster_boot95": [round(float(np.nanpercentile(diffs[:, mi[m]], 2.5)), 3),
                                                 round(float(np.nanpercentile(diffs[:, mi[m]], 97.5)), 3)]} for m in models}
    return out


def print_summary(out):
    print("== 표 5/13 재계산 ==")
    for k, v in out["table5_13_recompute"].items():
        print(f"{k:32s} {v['rate']:.3f} n={v['n']:.0f} 논문 {v['paper_rate']:.2f}/{v['paper_n']} n일치={v['n_match']} 비율일치={v['rate_match_rounded']} "
              f"Wilson={v['wilson95']} 군집={v['cluster_boot95']}")
    print("== G4-사람 대 사람-사람 차이 ==")
    for k, v in out["g4h_vs_hh_diff"].items():
        print(k, v)
    print("== 같은 항목 ==")
    for k, v in out["matched_items_diff"].items():
        print(k, v)
    print("== κ ==")
    for k, v in out["chance_corrected"].items():
        print(k, v)


if __name__ == "__main__":
    main()
