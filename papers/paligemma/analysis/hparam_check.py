"""논문 부록 J의 전이 하이퍼파라미터 표와 공식 코드의 `sweep_best` 설정을 정적으로 대조한다(코드 실행 없음).

big_vision/configs/proj/paligemma/transfers/<과제>.py 를 ast로만 읽는다.
  유효값 = get_config()의 기본값(c.total_epochs, c.lr, c.wd, c.label_smoothing, c.input.batch_size,
           c.model.llm=dict(dropout=...), parse_arg의 freeze_vit)  위에  sweep_best()의 add(...) 인자를 덮어쓴 것.
해석할 수 없는 항(배치 크기를 해상도별 사전으로 고르는 경우 등)은 'unparsed'로 두고 비교하지 않는다.
실행: python3 hparam_check.py [repo 경로]   (기본 ../repo/big_vision/configs/proj/paligemma/transfers)
"""
import ast
import json
import sys
from pathlib import Path

from paper_tables import HPARAMS  # 부록 J 표 (task, res, epochs, bs, lr, wd, dropout, label_smoothing, freeze_vit, decode)

HERE = Path(__file__).resolve().parent
CFG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "repo/big_vision/configs/proj/paligemma/transfers"

FILES = {
    "COCOcap": "cococap", "COCO-35L": "coco35l", "Screen2Words": "screen2words", "TextCaps": "textcaps",
    "SciCap": "scicap", "WidgetCap": "widgetcap", "VQAv2": "vqav2", "OKVQA": "okvqa", "AOKVQA-MC": "aokvqa_mc",
    "AOKVQA-DA": "aokvqa_da", "GQA": "gqa", "NLVR2": "nlvr2", "AI2D": "ai2d", "ScienceQA": "science_qa",
    "RSVQA-lr": "rsvqa_lr", "RSVQA-hr": "rsvqa_hr", "ChartQA": "chartqa", "VizWizVQA": "vizwizvqa",
    "TallyQA": "tallyqa", "OCR-VQA": "ocrvqa", "TextVQA": "textvqa", "DocVQA": "docvqa", "InfoVQA": "infovqa",
    "ST-VQA": "stvqa", "RefCOCO": "refcoco_seg", "ActivityNet-QA": "activitynet_qa",
    "ActivityNet-CAP": "activitynet_cap", "MSRVTT-QA": "msrvtt_qa", "MSRVTT-CAP": "msrvtt_cap",
    "MSVD-QA": "msvd_qa", "VATEX": "vatex_cap",
}


def lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return "unparsed"


def attr_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def kwargs_of(call):
    """add(...) 호출의 키워드를 모은다. **bvcc.arg(res=..., freeze_vit=...) 와 **{'a.b': v} 도 푼다."""
    out = {}
    for kw in call.keywords:
        if kw.arg is not None:
            out[kw.arg] = lit(kw.value)
        else:
            v = kw.value
            if isinstance(v, ast.Dict):
                for k, val in zip(v.keys, v.values):
                    out[lit(k)] = lit(val)
            elif isinstance(v, ast.Call):
                for kw2 in v.keywords:
                    if kw2.arg is not None:
                        out[kw2.arg] = lit(kw2.value)
    return out


def parse_config(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defaults = {"total_epochs": None, "lr": None, "wd": None, "label_smoothing": 0.0, "batch_size": None,
                "dropout": None, "freeze_vit": False, "beam": None}
    sweeps = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        if fn.name == "get_config":
            # get_config의 최상위 문장만 본다(if c.mode == 'runlocal' 안의 batch_size = 2 같은 시험용 값은 제외)
            for node in fn.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    p = attr_path(node.targets[0])
                    if p in ("c.total_epochs", "c.lr", "c.wd", "c.label_smoothing"):
                        defaults[p[2:]] = lit(node.value)
                    elif p == "c.input.batch_size":
                        v = node.value
                        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Dict):   # {224: 256, 448: 128}[c.res]
                            defaults["batch_size"] = {lit(k): lit(x) for k, x in zip(v.value.keys, v.value.values)}
                        else:
                            defaults["batch_size"] = lit(v)
                    elif p == "c.model.llm" and isinstance(node.value, ast.Call):
                        for kw in node.value.keywords:
                            if kw.arg == "dropout":
                                defaults["dropout"] = lit(kw.value)
                    elif (isinstance(node.value, ast.Call) and attr_path(node.value.func) == "bvcc.parse_arg"
                          and attr_path(node.targets[0]) == "c"):
                        for kw in node.value.keywords:
                            if kw.arg == "freeze_vit":
                                defaults["freeze_vit"] = lit(kw.value)
                            if kw.arg == "beam_size":
                                defaults["beam"] = lit(kw.value)
        if fn.name == "sweep_best":
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "add":
                    sweeps.append(kwargs_of(node))
    return defaults, sweeps


def effective(defaults, kw):
    eff = dict(defaults)
    for src, dst in (("lr", "lr"), ("wd", "wd"), ("total_epochs", "total_epochs"),
                     ("label_smoothing", "label_smoothing"), ("input.batch_size", "batch_size"),
                     ("model.llm.dropout", "dropout"), ("freeze_vit", "freeze_vit")):
        if src in kw:
            eff[dst] = kw[src]
    eff["res"] = kw.get("res")
    if isinstance(eff["batch_size"], dict):
        eff["batch_size"] = eff["batch_size"].get(eff["res"], "unparsed")
    return eff


def close(a, b):
    if a is None or b is None or a == "unparsed" or b == "unparsed":
        return None
    return abs(float(a) - float(b)) <= 1e-12 + 1e-9 * abs(float(b))


def main():
    results, mism = [], []
    n_cmp = n_ok = n_unparsed = n_missing = 0
    for task, fname in FILES.items():
        path = CFG / f"{fname}.py"
        if not path.exists():
            results.append({"task": task, "file": fname, "status": "config 없음"})
            continue
        defaults, sweeps = parse_config(path)
        by_res = {}
        for kw in sweeps:
            by_res[kw.get("res")] = effective(defaults, kw)
        for row in [r for r in HPARAMS if r[0] == task]:
            _, res, ep, bs, lr, wd, dr, ls, fz, dec = row
            eff = by_res.get(res)
            rec = {"task": task, "res": res, "paper": {"epochs": ep, "bs": bs, "lr": lr, "wd": wd, "dropout": dr,
                                                      "label_smoothing": ls, "freeze_vit": fz}}
            if eff is None:
                rec["status"] = "sweep_best에 이 해상도 항 없음"
                n_missing += 1
                results.append(rec)
                mism.append(f"{task} {res}px: 논문 표에는 있으나 공개 sweep_best에 항이 없음")
                continue
            code = {"epochs": eff["total_epochs"], "bs": eff["batch_size"], "lr": eff["lr"], "wd": eff["wd"],
                    "dropout": eff["dropout"], "label_smoothing": eff["label_smoothing"],
                    "freeze_vit": eff["freeze_vit"]}
            rec["code"] = code
            diffs = []
            for k, pv in rec["paper"].items():
                cv = code[k]
                if k == "freeze_vit":
                    ok = (cv == pv) if isinstance(cv, bool) else None
                else:
                    ok = close(cv, pv)
                if ok is None:
                    n_unparsed += 1
                else:
                    n_cmp += 1
                    n_ok += ok
                    if not ok:
                        diffs.append(f"{k}: 논문 {pv} 대 코드 {cv}")
            rec["diffs"] = diffs
            results.append(rec)
            if diffs:
                mism.append(f"{task} {res}px: " + "; ".join(diffs))
    out = {"config_dir": "repo/big_vision/configs/proj/paligemma/transfers", "n_fields_compared": n_cmp, "n_fields_equal": n_ok,
           "n_fields_unparsed": n_unparsed, "n_rows_missing_in_sweep_best": n_missing,
           "mismatches": mism, "rows": results}
    (HERE / "hparam_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"비교한 칸 {n_cmp}, 일치 {n_ok}, 해석 불가 {n_unparsed}, 불일치 {n_cmp - n_ok}")
    for m in mism:
        print(" -", m)


if __name__ == "__main__":
    main()
