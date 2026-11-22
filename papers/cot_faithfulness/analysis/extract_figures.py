"""논문 PDF의 벡터 그림(그림 2–4, 7, 8)에서 곡선 좌표를 추출해 figure_values.json로 저장.

PDF 그림이 벡터이므로 눈으로 읽지 않고 선분 좌표를 축 눈금으로 보정해 값을 얻는다.
실행: python3 -I analysis/extract_figures.py   (cot_faithfulness/ 에서)
"""
import json, math, pymupdf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "paper" / "paper.pdf"

# 범례 순서(그림 3 범례)와 matplotlib 색(Tol muted)
TASKS = {
    (0.2, 0.13, 0.53): "AQuA", (0.53, 0.8, 0.93): "TruthfulQA", (0.27, 0.67, 0.6): "MMLU",
    (0.07, 0.47, 0.2): "OpenBookQA", (0.6, 0.6, 0.2): "ARC (Challenge)",
    (0.87, 0.8, 0.47): "LogiQA", (0.8, 0.4, 0.47): "ARC (Easy)", (0.53, 0.13, 0.33): "HellaSwag",
}

def key(c):
    return tuple(round(v, 2) for v in c) if c else ()

def pts(d):
    out = []
    for it in d["items"]:
        if it[0] == "l":
            if not out:
                out.append((it[1].x, it[1].y))
            out.append((it[2].x, it[2].y))
    return out

def axis_ticks(dr, rect_lo, rect_hi):
    """한 패널(축 상자)의 눈금: 'fs' 검정 한 점 길이 선분. x 눈금(세로 짧은 선), y 눈금(가로 짧은 선)."""
    xt, yt = [], []
    for d in dr:
        if d["type"] == "fs" and key(d["color"]) == (0.0, 0.0, 0.0):
            r = d["rect"]
            if not (rect_lo[0] - 8 <= r.x0 <= rect_hi[0] + 8 and rect_lo[1] - 8 <= r.y0 <= rect_hi[1] + 8):
                continue
            if r.width < 1e-6 and r.height < 2.5:
                xt.append(r.x0)
            elif r.height < 1e-6 and r.width < 2.5:
                yt.append(r.y0)
    return sorted(set(round(v, 3) for v in xt)), sorted(set(round(v, 3) for v in yt))

def panels(dr):
    """검정 's' 4개짜리 축 상자(넓이 > 50)를 찾는다."""
    boxes = []
    for d in dr:
        r = d["rect"]
        if d["type"] == "s" and key(d["color"]) == (0.0, 0.0, 0.0) and r.height < 1e-6 and r.width > 50:
            boxes.append(r)
    # 같은 x0,x1를 가진 가로선 두 개(위, 아래)를 한 패널로
    byx = {}
    for r in boxes:
        byx.setdefault((round(r.x0, 1), round(r.x1, 1)), []).append(r.y0)
    out = []
    for (x0, x1), ys in byx.items():
        ys = sorted(ys)
        for i in range(0, len(ys) - 1, 2):
            out.append((x0, x1, ys[i], ys[i + 1]))
    return sorted(out, key=lambda b: (round(b[2]), b[0]))

def lin(a0, a1, v0, v1):
    return lambda a: v0 + (a - a0) * (v1 - v0) / (a1 - a0)

def curves_in_panel(dr, box):
    x0, x1, y0, y1 = box
    res = {}
    for d in dr:
        c = key(d["color"])
        if c in TASKS and d["type"] == "s":
            r = d["rect"]
            if r.x0 >= x0 - 1 and r.x1 <= x1 + 1 and r.y0 >= y0 - 1 and r.y1 <= y1 + 1 and len(d["items"]) >= 2:
                res.setdefault(TASKS[c], []).append(d)
    return res

def errbars_in_panel(dr, box):
    x0, x1, y0, y1 = box
    res = {}
    for d in dr:
        c = key(d["color"])
        if c in TASKS and d["type"] == "s" and len(d["items"]) == 1:
            r = d["rect"]
            if r.width < 1e-6 and x0 - 1 <= r.x0 <= x1 + 1 and y0 - 1 <= r.y0 <= y1 + 1:
                res.setdefault(TASKS[c], []).append((r.x0, r.y0, r.y1))
    return res

def extract_page(doc, pno, xcal, ycal_for, logx=False):
    p = doc[pno]
    dr = p.get_drawings()
    out = []
    for box in panels(dr):
        xt, yt = axis_ticks(dr, (box[0], box[2]), (box[1], box[3]))
        out.append((box, xt, yt, dr))
    return out

if __name__ == "__main__":
    doc = pymupdf.open(str(PDF))
    result = {}
    # ---- 그림 3 (쪽 4), 그림 4 (쪽 5): 패널 4개 = 3,4,5,6단계 CoT
    for fig, pno, xvals in (("fig3_early_answering", 3, [0, 25, 50, 75, 100]), ("fig4_adding_mistakes", 4, [0, 20, 40, 60, 80])):
        p = doc[pno]
        dr = p.get_drawings()
        pans = [b for b in panels(dr) if 40 < b[3] - b[2] < 80 and b[1] - b[0] < 120]
        # 읽는 순서: 위쪽 행 왼쪽→오른쪽, 아래쪽 행 왼쪽→오른쪽 = 3,4,5,6단계
        pans = sorted(pans, key=lambda b: (round(b[2]), b[0]))[:4]
        fig_out = {}
        for idx, box in enumerate(pans):
            xt, yt = axis_ticks(dr, (box[0], box[2]), (box[1], box[3]))
            # y 눈금: 위쪽 패널 40,60,80,100(아래에서 위) / 아래 패널도 같음
            ytv = [100, 80, 60, 40][:len(yt)] if len(yt) == 4 else None
            ymap = lin(yt[0], yt[-1], 100, 40) if len(yt) == 4 else None
            xmap = lin(xt[0], xt[-1], xvals[0], xvals[-1])
            steps = 3 + idx
            cur = curves_in_panel(dr, box)
            eb = errbars_in_panel(dr, box)
            for task, ds in cur.items():
                # 가장 점이 많은 폴리라인(= 평균 곡선)
                d = max(ds, key=lambda d: len(d["items"]))
                P = pts(d)
                fig_out.setdefault(task, {})[f"{steps}-step"] = {
                    "x": [round(xmap(a), 2) for a, _ in P], "y": [round(ymap(b), 2) for _, b in P],
                    "err_halfwidth": [round(abs(ymap(e[1]) - ymap(e[2])) / 2, 2) for e in sorted(eb.get(task, []))],
                }
        result[fig] = fig_out
    # ---- 그림 6 (쪽 7): 패러프레이즈 정확도. 패널 4개, x = 패러프레이즈한 비율(20..100), y = 정확도(40..100)
    p = doc[6]
    dr = p.get_drawings()
    pans = sorted([b for b in panels(dr) if 40 < b[3] - b[2] < 80 and b[1] - b[0] < 120], key=lambda b: (round(b[2]), b[0]))[:4]
    f6 = {}
    for idx, box in enumerate(pans):
        xt, yt = axis_ticks(dr, (box[0], box[2]), (box[1], box[3]))
        xmap = lin(xt[0], xt[-1], 20, 100)
        ymap = lin(yt[0], yt[-1], 100, 40)
        steps = 3 + idx
        for d in dr:
            c = key(d["color"])
            r = d["rect"]
            if c in TASKS and d["type"] == "s" and len(d["items"]) >= 2 and box[0] - 1 <= r.x0 and r.x1 <= box[1] + 1 and box[2] - 1 <= r.y0 and r.y1 <= box[3] + 1:
                P = pts(d)
                kind = "dashed_nonparaphrased" if d.get("dashes", "[] 0") != "[] 0" else "paraphrased"
                f6.setdefault(TASKS[c], {}).setdefault(f"{steps}-step", {})[kind] = {
                    "x": [round(xmap(a), 1) for a, _ in P], "y": [round(ymap(b), 2) for _, b in P]}
    result["fig6_paraphrase_accuracy"] = f6
    # ---- 그림 5 (쪽 6): 필러 토큰 길이(백분위, x 0..100)별 정확도(y 20..90), 점선 = CoT 없는 정확도
    p = doc[5]
    dr = p.get_drawings()
    box = [b for b in panels(dr)][0]
    xt, yt = axis_ticks(dr, (box[0], box[2]), (box[1], box[3]))
    xmap = lin(xt[0], xt[-1], 0, 100)
    ymap = lin(yt[-1], yt[0], 20, 90)
    f5 = {}
    for d in dr:
        c = key(d["color"])
        r = d["rect"]
        if c in TASKS and d["type"] == "s" and len(d["items"]) >= 3 and box[0] - 1 <= r.x0 and r.x1 <= box[1] + 1 and box[2] - 1 <= r.y0 and r.y1 <= box[3] + 1:
            P = pts(d)
            kind = "dashed_no_cot" if d.get("dashes", "[] 0") != "[] 0" else "filler_curve"
            f5.setdefault(TASKS[c], {})[kind] = {"x": [round(xmap(a), 1) for a, _ in P], "y": [round(ymap(b), 2) for _, b in P]}
    result["fig5_filler_accuracy"] = f5
    # ---- 그림 7 (쪽 7): 모델 크기별 CoT 유무 같은 답 비율 / 그림 10, 11 (쪽 13): 정확도, CoT 이득
    SIZES = [0.81, 1.6, 3.5, 6.4, 13, 22, 52, 175]
    def sized(pno, box, majors, y_lo_px, y_lo_val, y_hi_px, y_hi_val, label):
        dr = doc[pno].get_drawings()
        ymap = lin(y_lo_px, y_hi_px, y_lo_val, y_hi_val)
        out = {}
        for task, ds in curves_in_panel(dr, box).items():
            for d in ds:
                if len(d["items"]) != 7:
                    continue
                P = pts(d)
                kind = "dashed" if d.get("dashes", "[] 0") != "[] 0" else "solid"
                out.setdefault(task, {})[kind] = {"model_B": SIZES[:len(P)], "y": [round(ymap(b), 2) for _, b in P]}
        for task, eb in errbars_in_panel(dr, box).items():
            xs = sorted(set(round(e[0], 1) for e in eb))
            halves = []
            for x in xs:
                es = [e for e in eb if round(e[0], 1) == x]
                halves.append(round(max(abs(ymap(e[1]) - ymap(e[2])) / 2 for e in es), 2))
            out.setdefault(task, {})["err_halfwidth_by_x"] = halves[:8] if len(halves) >= 8 else halves
        result[label] = out
    sized(6, (331.2, 538.5, 86.3, 223.6), None, 223.601, 0, 86.314, 100, "fig7_same_answer_with_without_cot")
    sized(12, (329.9, 525.8, 200.2, 329.9), None, 314.179, 20, 201.228, 100, "fig10_accuracy_by_model_size")
    sized(12, (329.2, 538.4, 457.9, 596.5), None, 558.19, 0, 477.926, 15, "fig11_cot_gain_by_model_size")
    Path(ROOT / "analysis" / "figure_values.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print({k: len(v) for k, v in result.items()})
    for t, v in result["fig3_early_answering"].items():
        print(t, {k: vv["y"] for k, vv in v.items()})
