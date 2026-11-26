"""논문 부록 Table 2-4(전체 결과)를 paper/paper.txt에서 파싱해 analysis/tables.json으로 저장한다.

실행: python3 analysis/parse_tables.py   (paper/paper.txt가 있어야 한다. 없으면 이미 만든 tables.json을 쓴다)
표의 모델 이름은 섹션 이름 아래에 있고, 같은 이름이 섹션마다 반복된다(예: Vicuña v1.5 7B).
저장 형식: {"cols": {표: [열 이름]}, "rows": [{"section", "model", "scores": {열: 값 또는 null}}]}
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TXT = HERE.parent / "paper" / "paper.txt"

TABLES = {
    "T2": ("Table 2. All Results on VQA Benchmarks", ["VQAv2", "GQA", "VizWiz", "TextVQA+OCR", "TextVQA"]),
    "T3": ("Table 3. All Results on Localization Benchmarks", ["RefCOCO", "RefCOCO+", "RefCOCOg", "OCIDRef"]),
    "T4": ("Table 4. All Results on Challenge Benchmarks", ["VSR", "POPE", "TallyQA", "AI2D"]),
}
SECTIONS = {
    "Official Models", "Reproduction & Optimization Procedure", "Visual Representation", "Visual Representations",
    "Image Preprocessing", "Ensembling Visual Features", "Base vs. Instruct Tuned LMs", "Better LMs",
    "Co-training on Language Safety Data", "Scaling Train Time", "Scaling Training Time", "Scaling Data",
    "Prism 7B", "Prism 13B",
}


def parse():
    lines = TXT.read_text(encoding="utf-8").splitlines()
    out = {}
    for tid, (title, cols) in TABLES.items():
        start = next(i for i, l in enumerate(lines) if title in l)
        sec, rows = None, []
        for l in lines[start + 2:]:
            s = l.strip()
            if not s or s.startswith("Investigating the Design") or re.fullmatch(r"\d+", s):
                if re.fullmatch(r"\d+", s) and rows:
                    break
                continue
            if s.startswith("Model"):
                continue
            if s in SECTIONS:
                sec = "Visual Representation" if s == "Visual Representations" else (
                    "Scaling Train Time" if s == "Scaling Training Time" else s)
                continue
            m = re.match(r"^(.*?)\s{2,}((?:N/A|[\d.]+)(?:\s+(?:N/A|[\d.]+))*)$", s)
            if not m:
                continue
            vals = m.group(2).split()
            if len(vals) != len(cols):
                continue
            rows.append({"section": sec, "model": m.group(1).strip(),
                         "scores": {c: (None if v == "N/A" else float(v)) for c, v in zip(cols, vals)}})
        out[tid] = {"cols": cols, "rows": rows}
    return out


def main():
    t = parse()
    for k, v in t.items():
        print(k, len(v["rows"]), "rows")
    (HERE / "tables.json").write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
