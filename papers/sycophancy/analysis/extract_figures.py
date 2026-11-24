"""논문 PDF(벡터 그림)에서 막대·점·선의 값을 도형 좌표로 복원합니다.

논문 본문에는 과제별 수치 표가 없고 값이 전부 그림에만 있습니다. 그림이 벡터라서
도형 좌표와 눈금 글자 위치로 값을 되살릴 수 있습니다 (눈으로 읽는 것보다 정확합니다).
복원 오차는 눈금 글자 중심 정렬 한계로 약 0.3%p 안팎입니다.
입력은 paper/paper.pdf 하나이고 외부 데이터를 받지 않습니다. 결과: figure_values.json
"""
import json
import os

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "paper", "paper.pdf")
BLUE, ORANGE, GREEN = (0.0, 0.45, 0.7), (0.87, 0.56, 0.02), (0.01, 0.62, 0.45)
VERM, PURPLE = (0.84, 0.37, 0.0), (0.8, 0.47, 0.74)
M5 = ["Claude 1.3", "Claude 2", "GPT-3.5", "GPT-4", "LLaMA 2"]  # 본문 그림 1-4의 순서
M5B = ["GPT-3.5", "GPT-4", "Claude 1.3", "Claude 2", "LLaMA 2"]  # 부록 그림 13-16의 순서


def rc(c):
    return tuple(round(v, 2) for v in c)


def num(t):
    t = t.replace("−", "-").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def spans(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    out.append((s["text"].strip(), s["bbox"]))
    return out


def axes_of(page, x_lo, x_hi, y_lo, y_hi, min_w=40, min_h=20, need_ticks=True):
    res = []
    sp = spans(page)
    for d in page.get_drawings():
        r = d["rect"]
        if d.get("fill") == (1.0, 1.0, 1.0) and r.width > min_w and r.height > min_h and x_lo <= r.x0 <= x_hi and y_lo <= r.y0 <= y_hi:
            if not need_ticks or calib(sp, r) is not None:
                res.append(r)
    return sorted(res, key=lambda r: (round(r.y0 / 20), r.x0))


def calib(sp, ax, vertical=True):
    pts = []
    for t, bb in sp:
        v = num(t)
        if v is None:
            continue
        if vertical:
            yc = (bb[1] + bb[3]) / 2
            if ax.x0 - 30 <= bb[0] and bb[2] <= ax.x0 + 1 and ax.y0 - 6 <= yc <= ax.y1 + 3:
                pts.append((yc, v))
        else:
            xc = (bb[0] + bb[2]) / 2
            if ax.y1 - 1 <= bb[1] <= ax.y1 + 14 and ax.x0 - 6 <= xc <= ax.x1 + 6:
                pts.append((xc, v))
    pts = sorted(set(pts))
    if len(pts) < 2:
        return None
    (p1, v1), (p2, v2) = pts[0], pts[-1]
    a = (v2 - v1) / (p2 - p1)
    return a, v1 - a * p1


def bars(page, ax, cal, zero=0.0, min_w=2.5):
    a, b = cal
    out = []
    for d in page.get_drawings():
        r, f = d["rect"], d.get("fill")
        if f is None or f == (1.0, 1.0, 1.0) or d["type"] not in ("f", "fs"):
            continue
        if ax.x0 - 0.5 <= r.x0 and r.x1 <= ax.x1 + 0.5 and ax.y0 - 2 <= r.y0 and r.y1 <= ax.y1 + 2 and min_w < r.width < ax.width * 0.5 and r.height >= 0.2:
            if max(f) - min(f) < 0.03 and f[0] > 0.6:
                continue
            vt, vb = a * r.y0 + b, a * r.y1 + b
            far_top = abs(vt - zero) >= abs(vb - zero)
            out.append({"r": r, "fill": rc(f), "value": vt if far_top else vb, "edge_y": r.y0 if far_top else r.y1})
    for o in out:
        o["se"] = None
    for d in page.get_drawings():
        if d.get("color") == (0.0, 0.0, 0.0):
            r = d["rect"]
            if r.width < 0.3 and r.height > 0.6 and ax.x0 < r.x0 < ax.x1:
                for o in out:
                    if o["r"].x0 - 0.5 <= r.x0 <= o["r"].x1 + 0.5 and r.y0 - 0.8 <= o["edge_y"] <= r.y1 + 0.8 and o["se"] is None:
                        o["se"] = abs(a * (r.y1 - r.y0)) / 2
                        break
    return sorted(out, key=lambda o: o["r"].x0)


def ser(bs, color, names, nd=2):
    sel = [x for x in bs if max(abs(u - v) for u, v in zip(x["fill"], color)) < 0.05]
    return {n: {"value": round(x["value"], nd), "se": None if x["se"] is None else round(x["se"], nd)} for n, x in zip(names, sel)}


def lines_in(page, x0, x1, y0, y1):
    res = []
    for d in page.get_drawings():
        if d["type"] == "s" and d.get("width", 0) > 0.8 and d.get("color") not in (None, (0.0, 0.0, 0.0)):
            pts = []
            for it in d["items"]:
                if it[0] == "l":
                    c = [it[1], it[2]]
                elif it[0] == "c":
                    c = [it[1], it[4]]
                else:
                    continue
                for q in c:
                    if not pts or abs(pts[-1][0] - q.x) > 0.05 or abs(pts[-1][1] - q.y) > 0.05:
                        pts.append((q.x, q.y))
            if len(pts) >= 3 and all(x0 <= q[0] <= x1 and y0 <= q[1] <= y1 for q in pts):
                res.append((rc(d["color"]), pts))
    return res


def main():
    doc = pymupdf.open(PDF)
    R = {}

    # 그림 1 (p3): 피드백 긍정도 (기준선 50)
    p = doc[2]; sp = spans(p)
    ax = axes_of(p, 340, 345, 85, 100)[0]
    bs = bars(p, ax, calib(sp, ax), zero=50.0)
    R["fig1_feedback_positivity"] = {k: ser(bs, c, M5) for k, c in
                                      [("really_like", GREEN), ("really_dislike", VERM), ("wrote", BLUE), ("did_not_write", ORANGE)]}

    # 그림 2 (p4): "확실해요?"
    p = doc[3]; sp = spans(p)
    a_ax, b_ax = axes_of(p, 330, 335, 95, 170)[:2]
    R["fig2_are_you_sure"] = {
        "a_admits_mistake_after_correct": ser(bars(p, a_ax, calib(sp, a_ax)), BLUE, M5) if False else
        {n: {"value": round(x["value"], 2), "se": None if x["se"] is None else round(x["se"], 2)} for n, x in zip(M5, bars(p, a_ax, calib(sp, a_ax)))},
        "b_changes_to_incorrect_after_correct":
        {n: {"value": round(x["value"], 2), "se": None if x["se"] is None else round(x["se"], 2)} for n, x in zip(M5, bars(p, b_ax, calib(sp, b_ax)))},
    }

    # 그림 3 (p5), 그림 4 (p5)
    p = doc[4]; sp = spans(p)
    ax = axes_of(p, 290, 320, 85, 120)[0]
    bs = bars(p, ax, calib(sp, ax))
    R["fig3_answer_sycophancy_delta_acc"] = {k: ser(bs, c, M5) for k, c in
                                              [("think_correct", GREEN), ("dont_think_correct", VERM), ("think_incorrect", ORANGE)]}
    ax = axes_of(p, 380, 420, 240, 280)[0]
    bs = bars(p, ax, calib(sp, ax))
    R["fig4_mimicry_only_incorrect_attribution"] = {n: {"value": round(x["value"], 2), "se": None if x["se"] is None else round(x["se"], 2)} for n, x in zip(M5, bs)}

    # 부록 그림 10 (p18): 데이터셋별 피드백 아첨. 축 3개 순서는 아래 캡션 확인 필요
    p = doc[17]; sp = spans(p)
    fig10 = []
    for ax in [r for r in axes_of(p, 130, 400, 430, 440) if r.width < 100 and r.height > 60]:
        bs = bars(p, ax, calib(sp, ax), zero=50.0)
        fig10.append({k: ser(bs, c, M5) for k, c in [("really_like", GREEN), ("really_dislike", VERM), ("wrote", BLUE), ("did_not_write", ORANGE)]})
    R["fig10_feedback_by_dataset_axes_in_page_order"] = fig10

    # 부록 그림 11 (p19 위): 데이터셋별 답 아첨. 왼쪽 TriviaQA, 오른쪽 TruthfulQA
    p = doc[18]; sp = spans(p)
    f11 = {}
    for nm, ax in zip(["TriviaQA", "TruthfulQA"], [r for r in axes_of(p, 130, 340, 110, 120) if r.width < 100]):
        bs = bars(p, ax, calib(sp, ax))
        f11[nm] = {k: ser(bs, c, M5) for k, c in [("think_correct", GREEN), ("dont_think_correct", VERM), ("think_incorrect", ORANGE)]}
    R["fig11_answer_sycophancy_by_dataset"] = f11

    # 부록 그림 12 (p19 가운데): (a) 둘 다 미언급, (b) 둘 다 언급, (c) 맞는 출처만 언급
    f12 = {}
    for nm, ax in zip(["a_neither", "b_both", "c_only_correct"], [r for r in axes_of(p, 130, 420, 305, 315) if r.width < 100]):
        bs = bars(p, ax, calib(sp, ax))
        f12[nm] = {n: {"value": round(x["value"], 2), "se": None if x["se"] is None else round(x["se"], 2)} for n, x in zip(M5, bs)}
    R["fig12_mimicry_other_outcomes"] = f12

    # 부록 그림 13 (p19 아래), 14·15·16 (p20): 데이터셋 6개 x 모델 5개
    names6 = ["TruthfulQA MC", "AQuA", "MATH (CoT)", "MMLU (CoT)", "TruthfulQA freeform", "TriviaQA"]

    def six(page, y_lo, y_hi, two):
        sp = spans(page)
        axs = [r for r in axes_of(page, 130, 400, y_lo, y_hi) if r.width < 110 and r.height < 60]
        out = {}
        for nm, ax in zip(names6, axs):
            bs = bars(page, ax, calib(sp, ax))
            if two:
                out[nm] = {"initial": ser(bs, (0.09, 0.43, 0.61), M5B), "second": ser(bs, (0.76, 0.53, 0.13), M5B)}
            else:
                out[nm] = {n: {"value": round(x["value"], 2), "se": None if x["se"] is None else round(x["se"], 2)} for n, x in zip(M5B, bs)}
        return out
    R["fig13_accuracy_initial_vs_second"] = six(doc[18], 530, 600, True)
    p20 = doc[19]
    R["fig14_accuracy_conf_ge_95"] = six(p20, 110, 180, True)
    R["fig15_admits_mistake_all_questions"] = six(p20, 335, 405, False)
    R["fig16_changes_answer_all_questions"] = six(p20, 550, 620, False)

    # 그림 5 (p6): 베이즈 로지스틱 회귀 특징 효과 (확률 %, 점=사후 중앙값, 선=50%·95% 신용 구간)
    p = doc[5]; sp = spans(p)
    ax = [r for r in axes_of(p, 380, 390, 305, 315, need_ticks=False)][0]
    xs = [((bb[0] + bb[2]) / 2, num(t)) for t, bb in sp if num(t) in (50.0, 55.0, 60.0) and ax.y1 < bb[1] < ax.y1 + 14 and ax.x0 - 8 < bb[0] < ax.x1 + 8]
    xs = sorted(set(xs))
    a = (xs[-1][1] - xs[0][1]) / (xs[-1][0] - xs[0][0]); b = xs[0][1] - a * xs[0][0]
    wide, thin, dots = {}, {}, []
    for d in p.get_drawings():
        r, col = d["rect"], d.get("color")
        if d["type"] == "s" and col and col != (0.0, 0.0, 0.0) and abs(r.height) < 0.1 and ax.x0 < r.x0 and r.x1 < ax.x1 + 1:
            wd = d.get("width", 0)
            key = round(r.y0, 1)
            (thin if wd > 3.0 and r.width < 12 and False else wide).setdefault(key, []).append((r.x0, r.x1))
        if d["type"] == "fs" and d.get("fill") and d["fill"] != (0.0, 0.0, 0.0) and 3 < r.width < 5 and ax.x0 < r.x0 < ax.x1:
            dots.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    labels = ["Matches user's beliefs", "Authoritative", "Empathetic", "Relevant to user's query", "Truthful", "Engaging", "Persuasive",
              "Matches user's style", "Higher reading age", "Well-written", "Grammatically sound", "Logically sound", "Informative",
              "Rigorous", "Well-structured", "Friendly", "Entertaining", "Better evidentially supported", "Motivating", "Polite",
              "Concise", "Optimistic", "Funny"]
    dots.sort(key=lambda q: q[1])
    f5 = []
    for lab, (dx, dy) in zip(labels, dots):
        row = min(wide, key=lambda k: abs(k - dy))
        segs = sorted(wide[row], key=lambda s: s[1] - s[0])  # 짧은 것이 50%, 긴 것이 95%
        s95, s50 = segs[-1], segs[0]
        f5.append({"feature": lab, "median": round(a * dx + b, 2), "ci50": [round(a * s50[0] + b, 2), round(a * s50[1] + b, 2)],
                   "ci95": [round(a * s95[0] + b, 2), round(a * s95[1] + b, 2)]})
    R["fig5_feature_effects"] = f5


    # 그림 6 (p7): BoN(N=1..32)과 RL 진행에 따른 아첨 지표 (%)
    p = doc[6]; sp = spans(p)
    f6 = {}
    for nm, (xl, xh) in zip(["feedback", "answer", "mimicry"], [(160, 225), (235, 300), (310, 375)]):
        ax = [r for r in axes_of(p, xl, xh, 95, 105)][0]
        a, b = calib(sp, ax)
        ln = lines_in(p, ax.x0 - 1, ax.x1 + 1, ax.y0 - 1, ax.y1 + 1)
        ln = [(c, pts) for c, pts in ln if len(pts) == 6]
        f6[nm] = {("claude2_pm" if c == rc(BLUE) else "non_syc_pm"): [round(a * y + b, 2) for _, y in pts] for c, pts in ln}
    ax = [r for r in axes_of(p, 395, 410, 90, 100)][0]
    a, b = calib(sp, ax)
    ln = [(c, pts) for c, pts in lines_in(p, ax.x0 - 1, ax.x1 + 1, ax.y0 - 1, ax.y1 + 1) if len(pts) >= 8]
    nm = {rc(BLUE): "mimicry", rc(ORANGE): "feedback", rc(GREEN): "answer"}
    xt = sorted({((bb[0] + bb[2]) / 2, num(t)) for t, bb in sp if num(t) in (0.0, 0.5, 1.0) and ax.y1 < bb[1] < ax.y1 + 12 and ax.x0 - 8 < bb[0] < ax.x1 + 8})
    ax_a = (xt[-1][1] - xt[0][1]) / (xt[-1][0] - xt[0][0]); ax_b = xt[0][1] - ax_a * xt[0][0]
    f6["rl_points_fraction_pct"] = {nm[c]: [[round(ax_a * x + ax_b, 3), round(a * y + b, 2)] for x, y in pts] for c, pts in ln}
    R["fig6_bon_and_rl"] = f6

    # 그림 7 (p8): (a) PM이 아첨 답을 고른 비율, (b) 사람, (c)(d) BoN 뒤 아첨 응답 비율
    p = doc[7]; sp = spans(p)
    f7 = {}
    ax = axes_of(p, 355, 365, 95, 105)[0]
    bs = bars(p, ax, calib(sp, ax), min_w=2.0)
    def slots(bs, color, x_first):
        out = [0.0] * 8
        for x in bs:
            if x["fill"] == rc(color) and x["r"].width < 3:
                out[round((x["r"].x0 - x_first) / 5.95)] = round(x["value"], 1)
        return out
    f7["a_pm_prefers_sycophantic_by_difficulty"] = {"vs_baseline_truthful": slots(bs, BLUE, 364.6), "vs_helpful_truthful": slots(bs, GREEN, 367.0)}
    ax = axes_of(p, 430, 440, 95, 105)[0]
    bs = bars(p, ax, calib(sp, ax), min_w=2.0)
    f7["b_humans_majority_vote_by_difficulty"] = slots(bs, GREEN, 438.6)
    f7["b_humans_average_pixel_estimate"] = {"difficulty_8": 34.5, "note": "빗금 막대라 도형 좌표가 없어 12배 확대 래스터에서 막대 윗면을 쟀습니다 (범례와 겹치는 1-7번은 미측정)"}
    for nm, xl, xh in [("c_all_misconceptions", 355, 365), ("d_hardest_misconceptions", 430, 440)]:
        ax = axes_of(p, xl, xh, 185, 195)[0]
        a, b = calib(sp, ax)
        ln = [(c, pts) for c, pts in lines_in(p, ax.x0 - 1, ax.x1 + 1, ax.y0 - 1, ax.y1 + 1) if len(pts) == 13]
        names = {rc(ORANGE): "claude2_pm", rc(VERM): "oracle", rc(PURPLE): "non_syc_pm"}  # 범례 순서: Claude 2 PM(주황 선), Non-sycophantic(보라 별), Oracle(적갈색 별)
        f7[nm] = {names[c]: [round(a * y + b, 1) for _, y in pts] for c, pts in ln}
    R["fig7_misconceptions"] = f7


    # 부록 그림 17 (p21): "확실해요?" 전후 정오 전환 (Sankey). 흐름 폭 = 문항 비율
    p = doc[20]
    flows = []
    for dd in p.get_drawings():
        fl = dd.get("fill")
        if fl in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)) and len(dd["items"]) >= 30:
            pts = []
            for it in dd["items"]:
                if it[0] == "l":
                    pts += [it[1], it[2]]
            xs = [q.x for q in pts]
            xmin, xmax = min(xs), max(xs)
            L = [q.y for q in pts if abs(q.x - xmin) < 0.3]
            Rr = [q.y for q in pts if abs(q.x - xmax) < 0.3]
            flows.append({"src": "correct" if fl == (0.0, 0.0, 1.0) else "incorrect", "xmin": xmin, "L": (min(L), max(L)), "R": (min(Rr), max(Rr))})
    cols = sorted({round(f["xmin"] / 20) for f in flows})
    ycenters = sorted({(f["L"][0] + f["L"][1]) / 2 for f in flows})
    # 행 군집: 행 사이 간격이 20pt 이상
    rows, cur = [], [ycenters[0]]
    for y in ycenters[1:]:
        if y - cur[-1] > 20:
            rows.append(cur); cur = [y]
        else:
            cur.append(y)
    rows.append(cur)
    row_bounds = [(min(r) - 12, max(r) + 12) for r in rows]
    models = ["GPT-3.5", "GPT-4", "Claude 1.3", "Claude 2", "LLaMA 2"]
    ds = ["TruthfulQA MC", "AQuA", "MATH (CoT)", "MMLU (CoT)", "TruthfulQA freeform", "TriviaQA"]
    f17 = {}
    for fl in flows:
        c = cols.index(round(fl["xmin"] / 20)); yc = (fl["L"][0] + fl["L"][1]) / 2
        r = [i for i, (lo, hi) in enumerate(row_bounds) if lo <= yc <= hi][0]
        key = (ds[r], models[c])
        hL = fl["L"][1] - fl["L"][0]
        dy = (fl["R"][0] + fl["R"][1]) / 2 - yc
        if abs(dy) < 0.5:
            kind = "c_to_c" if fl["src"] == "correct" else "i_to_i"
        else:
            kind = "c_to_i" if fl["src"] == "correct" else "i_to_c"
        f17.setdefault(key, {})[kind] = f17.setdefault(key, {}).get(kind, 0) + hL
    out17 = {}
    for (dname, m), v in f17.items():
        tot = sum(v.values())
        out17.setdefault(dname, {})[m] = {k: round(x / tot, 4) for k, x in v.items()} | {"height_pt": round(tot, 1)}
    R["fig17_transition_fractions"] = out17

    json.dump(R, open(os.path.join(HERE, "figure_values.json"), "w"), ensure_ascii=False, indent=1)
    print("saved", len(R), "figure groups")


if __name__ == "__main__":
    main()
